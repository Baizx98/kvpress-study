# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
import math
from typing import Any

import torch
from torch import nn
from transformers.models.llama.modeling_llama import repeat_kv

from kvpress.presses.base_press import BasePress
from kvpress.utils import get_prerope_query_states


@dataclass
class BlockWisePress(BasePress):
    """
    Lightweight block-granularity query-aware KV compression.

    The block importance score is built in three stages:
    1. Aggregate the last few queries with an adaptive window.
    2. Score each token using a hybrid of average and peak query responses.
    3. Score each block using a hybrid of average and peak token responses, then
       aggregate across KV heads with optional head filtering / weighting.

    The resulting block scores are cheap to compute and can also be reused as a
    block heat indicator for future KV offload / prefetch policies.
    """

    compression_ratio: float = 0.0
    block_size: int = 16
    min_q_window: int = 4
    max_q_window: int = 64
    q_window_scale: float = 4.0
    q_mean_weight: float = 0.5
    token_peak_ratio: float = 0.25
    token_peak_weight: float = 0.5
    head_select_ratio: float = 1.0
    head_weight_exponent: float = 1.0
    eps: float = 1e-6
    last_block_heat: dict[int, torch.Tensor] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        assert 0 <= self.compression_ratio < 1, "compression_ratio must be in [0, 1)"
        assert self.block_size > 0, "block_size must be > 0"
        assert self.min_q_window > 0, "min_q_window must be > 0"
        assert self.max_q_window >= self.min_q_window, "max_q_window must be >= min_q_window"
        assert self.q_window_scale > 0, "q_window_scale must be > 0"
        assert 0 <= self.q_mean_weight <= 1, "q_mean_weight must be in [0, 1]"
        assert 0 <= self.token_peak_ratio <= 1, "token_peak_ratio must be in [0, 1]"
        assert 0 <= self.token_peak_weight <= 1, "token_peak_weight must be in [0, 1]"
        assert 0 < self.head_select_ratio <= 1, "head_select_ratio must be in (0, 1]"
        assert self.head_weight_exponent >= 0, "head_weight_exponent must be >= 0"

    def _resolve_layer_idx(self, module: nn.Module) -> int:
        raw = getattr(module, "layer_idx", 0)
        if isinstance(raw, torch.Tensor):
            return int(raw.item())
        return int(raw)

    def _resolve_q_window(self, q_len: int, key_len: int) -> int:
        adaptive_window = int(math.ceil(self.q_window_scale * math.sqrt(max(key_len, 1))))
        adaptive_window = max(self.min_q_window, adaptive_window)
        adaptive_window = min(self.max_q_window, adaptive_window)
        return min(q_len, adaptive_window)

    def _aggregate_over_queries(self, scores: torch.Tensor) -> torch.Tensor:
        mean_scores = scores.mean(dim=-2)
        peak_scores = scores.max(dim=-2).values
        return self.q_mean_weight * mean_scores + (1.0 - self.q_mean_weight) * peak_scores

    def _aggregate_tokens_into_blocks(self, token_scores: torch.Tensor, key_len: int) -> torch.Tensor:
        bsz, num_key_value_heads, _ = token_scores.shape
        num_blocks = math.ceil(key_len / self.block_size)
        block_scores = []

        for block_idx in range(num_blocks):
            start = block_idx * self.block_size
            end = min(start + self.block_size, key_len)
            block_token_scores = token_scores[:, :, start:end]
            mean_score = block_token_scores.mean(dim=-1)

            peak_count = max(1, int(math.ceil(block_token_scores.shape[-1] * self.token_peak_ratio)))
            peak_score = block_token_scores.topk(peak_count, dim=-1).values.mean(dim=-1)
            block_score = (1.0 - self.token_peak_weight) * mean_score + self.token_peak_weight * peak_score
            block_scores.append(block_score)

        return torch.stack(block_scores, dim=-1) if block_scores else token_scores.new_zeros((bsz, num_key_value_heads, 0))

    def _compute_head_weights(self, block_scores_per_head: torch.Tensor) -> torch.Tensor:
        head_strength = block_scores_per_head.max(dim=-1).values.clamp_min(0)
        if self.head_weight_exponent != 1.0:
            head_strength = head_strength.pow(self.head_weight_exponent)

        if self.head_select_ratio < 1.0 and head_strength.shape[1] > 0:
            n_selected_heads = max(1, int(math.ceil(head_strength.shape[1] * self.head_select_ratio)))
            top_indices = head_strength.topk(n_selected_heads, dim=-1).indices
            mask = torch.zeros_like(head_strength)
            mask.scatter_(1, top_indices, 1.0)
            head_strength = head_strength * mask

        normalizer = head_strength.sum(dim=1, keepdim=True)
        uniform_weights = torch.full_like(head_strength, 1.0 / max(head_strength.shape[1], 1))
        return torch.where(normalizer > 0, head_strength / normalizer.clamp_min(self.eps), uniform_weights)

    def analyze_blocks(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> dict[str, Any]:
        del values, attentions, kwargs

        bsz, num_key_value_heads, key_len, head_dim = keys.shape
        if key_len == 0:
            return {
                "q_window": 0,
                "token_scores": keys.new_zeros((bsz, num_key_value_heads, 0)),
                "block_scores_per_head": keys.new_zeros((bsz, num_key_value_heads, 0)),
                "head_weights": keys.new_zeros((bsz, num_key_value_heads)),
                "block_scores": keys.new_zeros((bsz, 0)),
            }

        q_window = self._resolve_q_window(hidden_states.shape[1], key_len)
        query_states = get_prerope_query_states(module, hidden_states[:, -q_window:])
        num_key_value_groups = module.config.num_attention_heads // num_key_value_heads
        key_states = repeat_kv(keys, num_key_value_groups)

        attn_scores = torch.matmul(query_states, key_states.transpose(-1, -2)) / math.sqrt(head_dim)
        attn_scores = attn_scores.view(
            bsz,
            num_key_value_heads,
            num_key_value_groups,
            q_window,
            key_len,
        ).mean(dim=2)

        token_scores = self._aggregate_over_queries(attn_scores)
        block_scores_per_head = self._aggregate_tokens_into_blocks(token_scores, key_len)
        head_weights = self._compute_head_weights(block_scores_per_head)
        block_scores = (block_scores_per_head * head_weights.unsqueeze(-1)).sum(dim=1)

        layer_idx = self._resolve_layer_idx(module)
        self.last_block_heat[layer_idx] = block_scores.detach()

        return {
            "q_window": q_window,
            "token_scores": token_scores,
            "block_scores_per_head": block_scores_per_head,
            "head_weights": head_weights,
            "block_scores": block_scores,
        }

    def build_block_plan(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
        compression_ratio: float | None = None,
    ) -> dict[str, Any]:
        analysis = self.analyze_blocks(module, hidden_states, keys, values, attentions, kwargs)
        key_len = keys.shape[2]
        num_blocks = analysis["block_scores"].shape[-1]
        ratio = self.compression_ratio if compression_ratio is None else compression_ratio

        n_kept_blocks = min(num_blocks, max(0, int(math.ceil(num_blocks * (1.0 - ratio)))))
        has_partial_tail_block = key_len % self.block_size != 0
        tail_block_idx = num_blocks - 1

        if n_kept_blocks == 0:
            kept_block_indices = torch.empty(
                analysis["block_scores"].shape[0],
                0,
                dtype=torch.long,
                device=keys.device,
            )
        elif n_kept_blocks == num_blocks:
            kept_block_indices = torch.arange(num_blocks, device=keys.device).expand(analysis["block_scores"].shape[0], -1)
        else:
            candidate_scores = analysis["block_scores"]
            blocks_to_sample = n_kept_blocks

            if has_partial_tail_block:
                blocks_to_sample -= 1
                candidate_scores = candidate_scores[..., :-1]

            if blocks_to_sample > 0:
                kept_block_indices = candidate_scores.topk(blocks_to_sample, dim=-1).indices
            else:
                kept_block_indices = torch.empty(
                    analysis["block_scores"].shape[0],
                    0,
                    dtype=torch.long,
                    device=keys.device,
                )

            if has_partial_tail_block:
                tail_indices = torch.full(
                    (analysis["block_scores"].shape[0], 1),
                    tail_block_idx,
                    dtype=torch.long,
                    device=keys.device,
                )
                kept_block_indices = torch.cat([kept_block_indices, tail_indices], dim=-1)

            kept_block_indices = kept_block_indices.sort(dim=-1).values

        token_indices_list = []
        expected_kept_len = None
        for batch_idx in range(keys.shape[0]):
            token_indices = []
            for block_idx in kept_block_indices[batch_idx].tolist():
                start = block_idx * self.block_size
                end = min(start + self.block_size, key_len)
                token_indices.extend(range(start, end))
            token_tensor = torch.tensor(token_indices, dtype=torch.long, device=keys.device)
            if expected_kept_len is None:
                expected_kept_len = token_tensor.numel()
            elif token_tensor.numel() != expected_kept_len:
                raise ValueError(
                    "BlockWisePress got different kept token counts across batch items. "
                    "Please use batch_size=1 or configure block selection so each sample keeps the same tail layout."
                )
            token_indices_list.append(token_tensor)

        if token_indices_list:
            token_indices = torch.stack(token_indices_list, dim=0)
        else:
            token_indices = torch.empty(keys.shape[0], 0, dtype=torch.long, device=keys.device)

        analysis.update(
            {
                "num_blocks": num_blocks,
                "n_kept_blocks": n_kept_blocks,
                "kept_block_indices": kept_block_indices,
                "token_indices": token_indices,
            }
        )
        return analysis

    def gather_by_token_indices(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        token_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if token_indices.numel() == 0:
            return keys[:, :, :0], values[:, :, :0]

        head_dim = keys.shape[-1]
        gather_indices = token_indices[:, None, :, None].expand(-1, keys.shape[1], -1, head_dim)
        gathered_keys = keys.gather(dim=2, index=gather_indices).contiguous()
        gathered_values = values.gather(dim=2, index=gather_indices).contiguous()
        return gathered_keys, gathered_values

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.compression_ratio == 0:
            return keys, values

        assert attentions is None, "BlockWisePress does not support attentions."

        plan = self.build_block_plan(module, hidden_states, keys, values, attentions, kwargs)
        return self.gather_by_token_indices(keys, values, plan["token_indices"])
