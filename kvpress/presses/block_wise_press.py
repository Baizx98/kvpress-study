# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
import logging
import math
from typing import Any

import torch
from torch import nn
from transformers.models.llama.modeling_llama import repeat_kv

from kvpress.presses.base_press import BasePress
from kvpress.utils import get_prerope_query_states


logger = logging.getLogger(__name__)


@dataclass
class BlockWisePress(BasePress):
    """
    Low-overhead block-granularity KV compression with compact block summaries.

    Each block is represented by:
    - a mean key for the whole block
    - a mean key over the top-k highest-norm tokens in the block

    The last question-aware queries interact only with these summaries, which
    keeps both compute and metadata cost low enough for future block offload.
    """

    compression_ratio: float = 0.0
    block_size: int = 16
    q_window_size: int = 64
    summary_topk_keys: int = 2
    mean_key_weight: float = 0.5
    protected_recent_blocks: int = 4
    eps: float = 1e-6
    require_question_aware: bool = True

    last_block_heat: dict[int, torch.Tensor] = field(default_factory=dict, init=False, repr=False)
    last_block_heat_ema: dict[int, torch.Tensor] = field(default_factory=dict, init=False, repr=False)
    last_block_summary: dict[int, dict[str, torch.Tensor]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        assert 0 <= self.compression_ratio < 1, "compression_ratio must be in [0, 1)"
        assert self.block_size > 0, "block_size must be > 0"
        assert self.q_window_size > 0, "q_window_size must be > 0"
        assert self.summary_topk_keys > 0, "summary_topk_keys must be > 0"
        assert 0 <= self.mean_key_weight <= 1, "mean_key_weight must be in [0, 1]"
        assert self.protected_recent_blocks >= 0, "protected_recent_blocks must be >= 0"

    def _resolve_layer_idx(self, module: nn.Module) -> int:
        raw = getattr(module, "layer_idx", 0)
        if isinstance(raw, torch.Tensor):
            return int(raw.item())
        return int(raw)

    def _resolve_q_window(self, q_len: int) -> int:
        return min(q_len, self.q_window_size)

    def _resolve_summary_topk(self) -> int:
        return min(self.summary_topk_keys, self.block_size)

    def _summarize_blocks(self, keys: torch.Tensor, values: torch.Tensor) -> dict[str, torch.Tensor]:
        bsz, num_key_value_heads, key_len, head_dim = keys.shape
        num_blocks = math.ceil(key_len / self.block_size)
        topk = self._resolve_summary_topk()

        mean_keys = []
        topk_key_means = []
        mean_values = []
        token_counts = []

        for block_idx in range(num_blocks):
            start = block_idx * self.block_size
            end = min(start + self.block_size, key_len)
            block_keys = keys[:, :, start:end]
            block_values = values[:, :, start:end]
            block_len = end - start
            actual_topk = min(topk, block_len)

            mean_keys.append(block_keys.mean(dim=2))
            mean_values.append(block_values.mean(dim=2))

            token_norms = block_keys.norm(dim=-1)
            topk_token_indices = token_norms.topk(actual_topk, dim=-1).indices
            gather_index = topk_token_indices[..., None].expand(-1, -1, -1, head_dim)
            topk_keys = block_keys.gather(2, gather_index)
            topk_key_means.append(topk_keys.mean(dim=2))

            token_counts.append(
                torch.full((bsz,), block_len, dtype=torch.long, device=keys.device)
            )

        if not mean_keys:
            return {
                "mean_keys": keys.new_zeros((bsz, num_key_value_heads, 0, head_dim)),
                "topk_key_means": keys.new_zeros((bsz, num_key_value_heads, 0, head_dim)),
                "mean_values": values.new_zeros((bsz, num_key_value_heads, 0, head_dim)),
                "token_counts": torch.zeros((bsz, 0), dtype=torch.long, device=keys.device),
            }

        return {
            "mean_keys": torch.stack(mean_keys, dim=2),
            "topk_key_means": torch.stack(topk_key_means, dim=2),
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
        summary = self.build_or_refresh_block_summary(
            module, keys, values, force_refresh=force_refresh_summary
        )

        if key_len == 0:
            return {
                "q_window": 0,
                "block_summary": summary,
                "block_scores_per_head": keys.new_zeros((bsz, num_key_value_heads, 0)),
                "block_scores": keys.new_zeros((bsz, 0)),
            }

        q_window = self._resolve_q_window(hidden_states.shape[1])
        query_states = get_prerope_query_states(module, hidden_states[:, -q_window:])
        num_key_value_groups = module.config.num_attention_heads // num_key_value_heads
        num_blocks = summary["mean_keys"].shape[2]

        mean_key_states = repeat_kv(summary["mean_keys"], num_key_value_groups)
        topk_key_states = repeat_kv(summary["topk_key_means"], num_key_value_groups)

        mean_scores = torch.matmul(query_states, mean_key_states.transpose(-1, -2)) / math.sqrt(head_dim)
        topk_scores = torch.matmul(query_states, topk_key_states.transpose(-1, -2)) / math.sqrt(head_dim)

        mean_scores = mean_scores.view(
            bsz, num_key_value_heads, num_key_value_groups, q_window, num_blocks
        ).mean(dim=2)
        topk_scores = topk_scores.view(
            bsz, num_key_value_heads, num_key_value_groups, q_window, num_blocks
        ).mean(dim=2)

        block_mean_scores = mean_scores.mean(dim=-2)
        block_topk_scores = topk_scores.mean(dim=-2)
        block_scores_per_head = (
            self.mean_key_weight * block_mean_scores
            + (1.0 - self.mean_key_weight) * block_topk_scores
        )
        block_scores = block_scores_per_head.mean(dim=1)

        layer_idx = self._resolve_layer_idx(module)
        detached_scores = block_scores.detach()
        self.last_block_heat[layer_idx] = detached_scores
        previous_ema = self.last_block_heat_ema.get(layer_idx)
        if previous_ema is None or previous_ema.shape != block_scores.shape:
            self.last_block_heat_ema[layer_idx] = detached_scores
        else:
            self.last_block_heat_ema[layer_idx] = 0.8 * previous_ema + 0.2 * detached_scores

        return {
            "q_window": q_window,
            "block_summary": summary,
            "block_scores_per_head": block_scores_per_head,
            "block_scores": block_scores,
        }

    def _select_top_block_indices(
        self,
        scores: torch.Tensor,
        candidates: list[int],
        count: int,
        device: torch.device,
    ) -> torch.Tensor:
        if count <= 0 or not candidates:
            return torch.empty(scores.shape[0], 0, dtype=torch.long, device=device)

        candidate_tensor = torch.tensor(candidates, dtype=torch.long, device=device)
        candidate_scores = scores.index_select(dim=-1, index=candidate_tensor)
        top_indices = candidate_scores.topk(min(count, candidate_tensor.numel()), dim=-1).indices
        return candidate_tensor[top_indices]

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

        keep_budget = min(num_blocks, max(0, int(math.ceil(num_blocks * (1.0 - ratio)))))
        has_partial_tail_block = key_len % self.block_size != 0
        tail_block_idx = num_blocks - 1

        if keep_budget == 0:
            kept_block_indices = torch.empty(keys.shape[0], 0, dtype=torch.long, device=keys.device)
        elif keep_budget >= num_blocks:
            kept_block_indices = torch.arange(num_blocks, device=keys.device).expand(keys.shape[0], -1)
        else:
            recent_count = min(self.protected_recent_blocks, num_blocks)
            protected_recent_indices = set(range(max(0, num_blocks - recent_count), num_blocks))
            protected_tail_indices = {tail_block_idx} if has_partial_tail_block and num_blocks > 0 else set()
            protected_indices = protected_recent_indices | protected_tail_indices

            if len(protected_indices) <= keep_budget:
                remaining_candidates = [idx for idx in range(num_blocks) if idx not in protected_indices]
                additional_keeps = keep_budget - len(protected_indices)
                selected_remaining = self._select_top_block_indices(
                    analysis["block_scores"], remaining_candidates, additional_keeps, keys.device
                )
                protected_tensor = torch.tensor(
                    sorted(protected_indices), dtype=torch.long, device=keys.device
                ).expand(keys.shape[0], -1)
                kept_block_indices = (
                    torch.cat([protected_tensor, selected_remaining], dim=-1).sort(dim=-1).values
                )
            else:
                logger.info(
                    "Requested compression is too aggressive: protected recent blocks exceed keep budget. "
                    "Falling back to score-based selection over all blocks."
                )
                kept_block_indices = self._select_top_block_indices(
                    analysis["block_scores"], list(range(num_blocks)), keep_budget, keys.device
                ).sort(dim=-1).values

        token_indices = self.expand_blocks_to_token_indices(
            keys.shape[0], key_len, kept_block_indices, keys.device
        )
        analysis.update(
            {
                "num_blocks": num_blocks,
                "n_kept_blocks": kept_block_indices.shape[-1],
                "keep_budget": keep_budget,
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

        plan = self.build_block_plan(
            module,
            hidden_states,
            keys,
            values,
            attentions,
            kwargs,
            force_refresh_summary=True,
        )
        compressed_keys, compressed_values = self.gather_by_token_indices(
            keys, values, plan["token_indices"]
        )
        self.build_or_refresh_block_summary(
            module, compressed_keys, compressed_values, force_refresh=True
        )
        return compressed_keys, compressed_values
