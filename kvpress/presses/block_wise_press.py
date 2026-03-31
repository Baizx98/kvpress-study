# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
import math
from typing import Any, Literal

import torch
from torch import nn
from transformers.models.llama.modeling_llama import repeat_kv

from kvpress.presses.base_press import BasePress
from kvpress.utils import get_prerope_query_states


@dataclass
class BlockWisePress(BasePress):
    """
    Low-overhead block-granularity KV compression.

    Instead of first scoring every token and then aggregating to block scores,
    this press builds a compact summary for each block and scores the summaries
    directly with the last question-aware queries.
    """

    compression_ratio: float = 0.0
    block_size: int = 16
    min_q_window: int = 4
    max_q_window: int = 64
    q_window_scale: float = 4.0
    q_mean_weight: float = 0.5
    block_peak_weight: float = 0.5
    head_scoring_method: Literal["max", "topk_mean", "percentile"] = "max"
    head_topk_ratio: float = 0.25
    head_percentile: float = 0.9
    head_select_ratio: float = 1.0
    head_weight_exponent: float = 1.0
    head_redundancy_alpha: float = 0.0
    eps: float = 1e-6
    require_question_aware: bool = True

    last_block_heat: dict[int, torch.Tensor] = field(default_factory=dict, init=False, repr=False)
    last_block_heat_ema: dict[int, torch.Tensor] = field(default_factory=dict, init=False, repr=False)
    last_block_summary: dict[int, dict[str, torch.Tensor]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        assert 0 <= self.compression_ratio < 1, "compression_ratio must be in [0, 1)"
        assert self.block_size > 0, "block_size must be > 0"
        assert self.min_q_window > 0, "min_q_window must be > 0"
        assert self.max_q_window >= self.min_q_window, "max_q_window must be >= min_q_window"
        assert self.q_window_scale > 0, "q_window_scale must be > 0"
        assert 0 <= self.q_mean_weight <= 1, "q_mean_weight must be in [0, 1]"
        assert 0 <= self.block_peak_weight <= 1, "block_peak_weight must be in [0, 1]"
        assert self.head_scoring_method in {"max", "topk_mean", "percentile"}, "invalid head_scoring_method"
        assert 0 < self.head_topk_ratio <= 1, "head_topk_ratio must be in (0, 1]"
        assert 0 <= self.head_percentile <= 1, "head_percentile must be in [0, 1]"
        assert 0 < self.head_select_ratio <= 1, "head_select_ratio must be in (0, 1]"
        assert self.head_weight_exponent >= 0, "head_weight_exponent must be >= 0"
        assert 0 <= self.head_redundancy_alpha <= 1, "head_redundancy_alpha must be in [0, 1]"

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

    def _score_heads(self, block_scores_per_head: torch.Tensor) -> torch.Tensor:
        if block_scores_per_head.shape[-1] == 0:
            return block_scores_per_head.new_zeros(block_scores_per_head.shape[:2])

        if self.head_scoring_method == "max":
            return block_scores_per_head.max(dim=-1).values

        if self.head_scoring_method == "topk_mean":
            k = max(1, int(math.ceil(block_scores_per_head.shape[-1] * self.head_topk_ratio)))
            return block_scores_per_head.topk(k, dim=-1).values.mean(dim=-1)

        sorted_scores = block_scores_per_head.sort(dim=-1).values
        index = int(round((sorted_scores.shape[-1] - 1) * self.head_percentile))
        index = max(0, min(index, sorted_scores.shape[-1] - 1))
        return sorted_scores[..., index]

    def _apply_head_redundancy_penalty(
        self,
        head_strength: torch.Tensor,
        block_summary: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if self.head_redundancy_alpha == 0 or head_strength.shape[1] <= 1:
            return head_strength

        mean_keys = block_summary["mean_keys"]
        flattened = mean_keys.reshape(mean_keys.shape[0], mean_keys.shape[1], -1)
        flattened = flattened / flattened.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        similarity = torch.matmul(flattened, flattened.transpose(1, 2)).abs()

        eye = torch.eye(similarity.shape[-1], device=similarity.device, dtype=similarity.dtype).unsqueeze(0)
        redundancy = ((similarity * (1 - eye)).sum(dim=-1) / max(similarity.shape[-1] - 1, 1)).clamp(0, 1)
        return head_strength * (1.0 - self.head_redundancy_alpha * redundancy)

    def _compute_head_weights(
        self,
        block_scores_per_head: torch.Tensor,
        block_summary: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        head_strength = self._score_heads(block_scores_per_head).clamp_min(0)
        head_strength = self._apply_head_redundancy_penalty(head_strength, block_summary)
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

    def _summarize_blocks(self, keys: torch.Tensor, values: torch.Tensor) -> dict[str, torch.Tensor]:
        bsz, num_key_value_heads, key_len, head_dim = keys.shape
        num_blocks = math.ceil(key_len / self.block_size)

        mean_keys = []
        peak_keys = []
        mean_values = []
        token_counts = []

        for block_idx in range(num_blocks):
            start = block_idx * self.block_size
            end = min(start + self.block_size, key_len)
            block_keys = keys[:, :, start:end]
            block_values = values[:, :, start:end]

            mean_keys.append(block_keys.mean(dim=2))
            mean_values.append(block_values.mean(dim=2))

            token_norms = block_keys.norm(dim=-1)
            peak_token_indices = token_norms.max(dim=-1).indices
            gather_index = peak_token_indices[:, :, None, None].expand(-1, -1, 1, head_dim)
            peak_keys.append(block_keys.gather(2, gather_index).squeeze(2))

            token_counts.append(torch.full((bsz,), end - start, dtype=torch.long, device=keys.device))

        if not mean_keys:
            return {
                "mean_keys": keys.new_zeros((bsz, num_key_value_heads, 0, head_dim)),
                "peak_keys": keys.new_zeros((bsz, num_key_value_heads, 0, head_dim)),
                "mean_values": values.new_zeros((bsz, num_key_value_heads, 0, head_dim)),
                "token_counts": torch.zeros((bsz, 0), dtype=torch.long, device=keys.device),
            }

        return {
            "mean_keys": torch.stack(mean_keys, dim=2),
            "peak_keys": torch.stack(peak_keys, dim=2),
            "mean_values": torch.stack(mean_values, dim=2),
            "token_counts": torch.stack(token_counts, dim=1),
        }

    def build_or_refresh_block_summary(
        self,
        module: nn.Module,
        keys: torch.Tensor,
        values: torch.Tensor,
        force_refresh: bool = False,
    ) -> dict[str, torch.Tensor]:
        layer_idx = self._resolve_layer_idx(module)
        cached = self.last_block_summary.get(layer_idx)
        if (
            not force_refresh
            and cached is not None
            and int(cached["key_len"].item()) == keys.shape[2]
            and cached["mean_keys"].shape[2] == math.ceil(keys.shape[2] / self.block_size)
        ):
            return cached

        summary = self._summarize_blocks(keys, values)
        summary["key_len"] = torch.tensor(keys.shape[2], dtype=torch.long, device=keys.device)
        self.last_block_summary[layer_idx] = summary
        return summary

    def analyze_blocks(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor | None,
        kwargs: dict,
        force_refresh_summary: bool = False,
    ) -> dict[str, Any]:
        del attentions, kwargs

        bsz, num_key_value_heads, key_len, head_dim = keys.shape
        summary = self.build_or_refresh_block_summary(module, keys, values, force_refresh=force_refresh_summary)

        if key_len == 0:
            return {
                "q_window": 0,
                "block_summary": summary,
                "block_scores_per_head": keys.new_zeros((bsz, num_key_value_heads, 0)),
                "head_weights": keys.new_zeros((bsz, num_key_value_heads)),
                "block_scores": keys.new_zeros((bsz, 0)),
            }

        q_window = self._resolve_q_window(hidden_states.shape[1], key_len)
        query_states = get_prerope_query_states(module, hidden_states[:, -q_window:])
        num_key_value_groups = module.config.num_attention_heads // num_key_value_heads

        mean_key_states = repeat_kv(summary["mean_keys"], num_key_value_groups)
        peak_key_states = repeat_kv(summary["peak_keys"], num_key_value_groups)

        mean_scores = torch.matmul(query_states, mean_key_states.transpose(-1, -2)) / math.sqrt(head_dim)
        peak_scores = torch.matmul(query_states, peak_key_states.transpose(-1, -2)) / math.sqrt(head_dim)

        num_blocks = summary["mean_keys"].shape[2]
        mean_scores = mean_scores.view(bsz, num_key_value_heads, num_key_value_groups, q_window, num_blocks).mean(dim=2)
        peak_scores = peak_scores.view(bsz, num_key_value_heads, num_key_value_groups, q_window, num_blocks).mean(dim=2)

        block_mean_scores = self._aggregate_over_queries(mean_scores)
        block_peak_scores = self._aggregate_over_queries(peak_scores)
        block_scores_per_head = (1.0 - self.block_peak_weight) * block_mean_scores + self.block_peak_weight * block_peak_scores

        head_weights = self._compute_head_weights(block_scores_per_head, summary)
        block_scores = (block_scores_per_head * head_weights.unsqueeze(-1)).sum(dim=1)

        layer_idx = self._resolve_layer_idx(module)
        self.last_block_heat[layer_idx] = block_scores.detach()
        previous_ema = self.last_block_heat_ema.get(layer_idx)
        if previous_ema is None or previous_ema.shape != block_scores.shape:
            self.last_block_heat_ema[layer_idx] = block_scores.detach()
        else:
            self.last_block_heat_ema[layer_idx] = 0.8 * previous_ema + 0.2 * block_scores.detach()

        return {
            "q_window": q_window,
            "block_summary": summary,
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
        attentions: torch.Tensor | None,
        kwargs: dict,
        compression_ratio: float | None = None,
        force_refresh_summary: bool = False,
    ) -> dict[str, Any]:
        analysis = self.analyze_blocks(
            module,
            hidden_states,
            keys,
            values,
            attentions,
            kwargs,
            force_refresh_summary=force_refresh_summary,
        )
        key_len = keys.shape[2]
        num_blocks = analysis["block_scores"].shape[-1]
        ratio = self.compression_ratio if compression_ratio is None else compression_ratio

        n_kept_blocks = min(num_blocks, max(0, int(math.ceil(num_blocks * (1.0 - ratio)))))
        has_partial_tail_block = key_len % self.block_size != 0
        tail_block_idx = num_blocks - 1

        if n_kept_blocks == 0:
            kept_block_indices = torch.empty(keys.shape[0], 0, dtype=torch.long, device=keys.device)
        elif n_kept_blocks == num_blocks:
            kept_block_indices = torch.arange(num_blocks, device=keys.device).expand(keys.shape[0], -1)
        else:
            candidate_scores = analysis["block_scores"]
            sampled_blocks = n_kept_blocks

            if has_partial_tail_block:
                sampled_blocks -= 1
                candidate_scores = candidate_scores[..., :-1]

            if sampled_blocks > 0:
                kept_block_indices = candidate_scores.topk(sampled_blocks, dim=-1).indices
            else:
                kept_block_indices = torch.empty(keys.shape[0], 0, dtype=torch.long, device=keys.device)

            if has_partial_tail_block:
                tail_indices = torch.full((keys.shape[0], 1), tail_block_idx, dtype=torch.long, device=keys.device)
                kept_block_indices = torch.cat([kept_block_indices, tail_indices], dim=-1)

            kept_block_indices = kept_block_indices.sort(dim=-1).values

        token_indices = self.expand_blocks_to_token_indices(keys.shape[0], key_len, kept_block_indices, keys.device)
        analysis.update(
            {
                "num_blocks": num_blocks,
                "n_kept_blocks": n_kept_blocks,
                "kept_block_indices": kept_block_indices,
                "token_indices": token_indices,
            }
        )
        return analysis

    def expand_blocks_to_token_indices(
        self,
        batch_size: int,
        key_len: int,
        block_indices: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        token_indices_list = []
        expected_kept_len = None

        for batch_idx in range(batch_size):
            token_indices = []
            for block_idx in block_indices[batch_idx].tolist():
                start = block_idx * self.block_size
                end = min(start + self.block_size, key_len)
                token_indices.extend(range(start, end))

            token_tensor = torch.tensor(token_indices, dtype=torch.long, device=device)
            if expected_kept_len is None:
                expected_kept_len = token_tensor.numel()
            elif token_tensor.numel() != expected_kept_len:
                raise ValueError(
                    "BlockWisePress got different kept token counts across batch items. "
                    "Please use batch_size=1 or configure block selection so each sample keeps the same tail layout."
                )
            token_indices_list.append(token_tensor)

        if token_indices_list:
            return torch.stack(token_indices_list, dim=0)
        return torch.empty(batch_size, 0, dtype=torch.long, device=device)

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
        attentions: torch.Tensor | None,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.compression_ratio == 0:
            return keys, values

        assert attentions is None, "BlockWisePress does not support attentions."

        plan = self.build_block_plan(module, hidden_states, keys, values, attentions, kwargs, force_refresh_summary=True)
        compressed_keys, compressed_values = self.gather_by_token_indices(keys, values, plan["token_indices"])
        self.build_or_refresh_block_summary(module, compressed_keys, compressed_values, force_refresh=True)
        return compressed_keys, compressed_values
