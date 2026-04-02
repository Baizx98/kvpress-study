# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
import logging
import math
from typing import Any, Literal

import torch
from torch import nn
from transformers.models.llama.modeling_llama import repeat_kv

from kvpress.presses.base_press import BasePress
from kvpress.utils import get_prerope_query_states


logger = logging.getLogger(__name__)


@dataclass
class BlockWisePress(BasePress):
    """
    Low-overhead block-granularity KV compression.

    This press keeps the block-level design needed by future block offload and
    memory management, but strengthens block summaries so that a few critical
    tokens inside a block are less likely to be washed out by block means.
    """

    compression_ratio: float = 0.0
    block_size: int = 16
    min_q_window: int = 4
    max_q_window: int = 64
    q_window_scale: float = 4.0
    q_mean_weight: float = 0.5
    block_peak_weight: float = 0.3
    multi_peak_weight: float = 0.3
    summary_topk_keys: int | None = None
    summary_topk_max: int = 4
    protected_recent_blocks: int | None = None
    auto_recent_min_blocks: int = 2
    auto_recent_max_blocks: int = 8
    protected_hot_blocks: int = 0
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
        assert 0 <= self.multi_peak_weight <= 1, "multi_peak_weight must be in [0, 1]"
        assert self.block_peak_weight + self.multi_peak_weight <= 1, (
            "block_peak_weight + multi_peak_weight must be <= 1"
        )
        assert self.summary_topk_keys is None or self.summary_topk_keys > 0, (
            "summary_topk_keys must be > 0 when specified"
        )
        assert self.summary_topk_max > 0, "summary_topk_max must be > 0"
        assert self.protected_recent_blocks is None or self.protected_recent_blocks >= 0, (
            "protected_recent_blocks must be >= 0 when specified"
        )
        assert self.auto_recent_min_blocks >= 0, "auto_recent_min_blocks must be >= 0"
        assert self.auto_recent_max_blocks >= 0, "auto_recent_max_blocks must be >= 0"
        assert self.auto_recent_max_blocks >= self.auto_recent_min_blocks, (
            "auto_recent_max_blocks must be >= auto_recent_min_blocks"
        )
        assert self.protected_hot_blocks >= 0, "protected_hot_blocks must be >= 0"
        assert self.head_scoring_method in {"max", "topk_mean", "percentile"}, (
            "invalid head_scoring_method"
        )
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

    def _resolve_summary_topk(self) -> int:
        if self.summary_topk_keys is not None:
            return max(1, min(self.summary_topk_keys, self.block_size))

        auto_k = int(round(math.sqrt(self.block_size) / 2.0))
        auto_k = max(1, auto_k)
        auto_k = min(auto_k, self.summary_topk_max, self.block_size)
        return auto_k

    def _resolve_recent_block_count(self, num_blocks: int) -> int:
        if num_blocks <= 0:
            return 0
        if self.protected_recent_blocks is not None:
            return min(max(self.protected_recent_blocks, 0), num_blocks)
        if self.auto_recent_max_blocks <= 0:
            return 0

        recent_blocks = int(round(math.log2(max(num_blocks, 1))))
        recent_blocks = max(self.auto_recent_min_blocks, recent_blocks)
        recent_blocks = min(self.auto_recent_max_blocks, recent_blocks, num_blocks)
        return recent_blocks

    def _repeat_block_summary_kv(
        self,
        tensor: torch.Tensor,
        num_key_value_groups: int,
    ) -> torch.Tensor:
        if tensor.ndim == 4:
            return repeat_kv(tensor, num_key_value_groups)
        if tensor.ndim != 5:
            raise ValueError(f"Unsupported block summary rank: {tensor.ndim}")

        batch, num_key_value_heads, num_blocks, topk_size, head_dim = tensor.shape
        expanded = tensor[:, :, None, :, :, :].expand(
            batch,
            num_key_value_heads,
            num_key_value_groups,
            num_blocks,
            topk_size,
            head_dim,
        )
        return expanded.reshape(
            batch,
            num_key_value_heads * num_key_value_groups,
            num_blocks,
            topk_size,
            head_dim,
        )

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

        eye = torch.eye(
            similarity.shape[-1], device=similarity.device, dtype=similarity.dtype
        ).unsqueeze(0)
        redundancy = (
            (similarity * (1 - eye)).sum(dim=-1) / max(similarity.shape[-1] - 1, 1)
        ).clamp(0, 1)
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
            n_selected_heads = max(
                1, int(math.ceil(head_strength.shape[1] * self.head_select_ratio))
            )
            top_indices = head_strength.topk(n_selected_heads, dim=-1).indices
            mask = torch.zeros_like(head_strength)
            mask.scatter_(1, top_indices, 1.0)
            head_strength = head_strength * mask

        normalizer = head_strength.sum(dim=1, keepdim=True)
        uniform_weights = torch.full_like(
            head_strength, 1.0 / max(head_strength.shape[1], 1)
        )
        return torch.where(
            normalizer > 0,
            head_strength / normalizer.clamp_min(self.eps),
            uniform_weights,
        )

    def _summarize_blocks(self, keys: torch.Tensor, values: torch.Tensor) -> dict[str, torch.Tensor]:
        bsz, num_key_value_heads, key_len, head_dim = keys.shape
        num_blocks = math.ceil(key_len / self.block_size)
        summary_topk = self._resolve_summary_topk()

        mean_keys = []
        peak_keys = []
        topk_peak_keys = []
        topk_peak_counts = []
        mean_values = []
        token_counts = []

        for block_idx in range(num_blocks):
            start = block_idx * self.block_size
            end = min(start + self.block_size, key_len)
            block_keys = keys[:, :, start:end]
            block_values = values[:, :, start:end]
            block_len = end - start

            mean_keys.append(block_keys.mean(dim=2))
            mean_values.append(block_values.mean(dim=2))

            token_norms = block_keys.norm(dim=-1)
            actual_topk = min(summary_topk, block_len)
            peak_token_indices = token_norms.topk(actual_topk, dim=-1).indices
            gather_index = peak_token_indices[..., None].expand(-1, -1, -1, head_dim)
            representative_keys = block_keys.gather(2, gather_index)

            if actual_topk < summary_topk:
                pad = representative_keys[:, :, -1:, :].expand(
                    -1, -1, summary_topk - actual_topk, -1
                )
                representative_keys = torch.cat([representative_keys, pad], dim=2)

            topk_peak_keys.append(representative_keys)
            peak_keys.append(representative_keys[:, :, 0])
            topk_peak_counts.append(
                torch.full((bsz,), actual_topk, dtype=torch.long, device=keys.device)
            )
            token_counts.append(
                torch.full((bsz,), block_len, dtype=torch.long, device=keys.device)
            )

        if not mean_keys:
            return {
                "mean_keys": keys.new_zeros((bsz, num_key_value_heads, 0, head_dim)),
                "peak_keys": keys.new_zeros((bsz, num_key_value_heads, 0, head_dim)),
                "topk_peak_keys": keys.new_zeros((bsz, num_key_value_heads, 0, summary_topk, head_dim)),
                "topk_peak_counts": torch.zeros((bsz, 0), dtype=torch.long, device=keys.device),
                "mean_values": values.new_zeros((bsz, num_key_value_heads, 0, head_dim)),
                "token_counts": torch.zeros((bsz, 0), dtype=torch.long, device=keys.device),
            }

        return {
            "mean_keys": torch.stack(mean_keys, dim=2),
            "peak_keys": torch.stack(peak_keys, dim=2),
            "topk_peak_keys": torch.stack(topk_peak_keys, dim=2),
            "topk_peak_counts": torch.stack(topk_peak_counts, dim=1),
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

    def _reduce_multi_peak_scores(
        self,
        scores: torch.Tensor,
        topk_peak_counts: torch.Tensor,
    ) -> torch.Tensor:
        # scores: [batch, kv_heads, blocks, topk]
        topk_size = scores.shape[-1]
        valid_mask = (
            torch.arange(topk_size, device=scores.device)[None, None, None, :]
            < topk_peak_counts[:, None, :, None]
        ).to(scores.dtype)
        weighted_scores = (scores * valid_mask).sum(dim=-1)
        normalizer = valid_mask.sum(dim=-1).clamp_min(1.0)
        return weighted_scores / normalizer

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
                "head_weights": keys.new_zeros((bsz, num_key_value_heads)),
                "block_scores": keys.new_zeros((bsz, 0)),
            }

        q_window = self._resolve_q_window(hidden_states.shape[1], key_len)
        query_states = get_prerope_query_states(module, hidden_states[:, -q_window:])
        num_key_value_groups = module.config.num_attention_heads // num_key_value_heads
        num_blocks = summary["mean_keys"].shape[2]

        mean_key_states = self._repeat_block_summary_kv(summary["mean_keys"], num_key_value_groups)
        peak_key_states = self._repeat_block_summary_kv(summary["peak_keys"], num_key_value_groups)

        mean_scores = torch.matmul(query_states, mean_key_states.transpose(-1, -2)) / math.sqrt(head_dim)
        peak_scores = torch.matmul(query_states, peak_key_states.transpose(-1, -2)) / math.sqrt(head_dim)

        mean_scores = mean_scores.view(
            bsz, num_key_value_heads, num_key_value_groups, q_window, num_blocks
        ).mean(dim=2)
        peak_scores = peak_scores.view(
            bsz, num_key_value_heads, num_key_value_groups, q_window, num_blocks
        ).mean(dim=2)

        topk_peak_states = self._repeat_block_summary_kv(
            summary["topk_peak_keys"], num_key_value_groups
        )
        topk_scores = torch.einsum(
            "bhqd,bhnkd->bhqnk", query_states, topk_peak_states
        ) / math.sqrt(head_dim)
        topk_scores = topk_scores.view(
            bsz,
            num_key_value_heads,
            num_key_value_groups,
            q_window,
            num_blocks,
            summary["topk_peak_keys"].shape[-2],
        ).mean(dim=2)

        block_mean_scores = self._aggregate_over_queries(mean_scores)
        block_peak_scores = self._aggregate_over_queries(peak_scores)

        topk_mean_scores = topk_scores.mean(dim=2)
        topk_peak_query_scores = topk_scores.max(dim=2).values
        topk_query_scores = self.q_mean_weight * topk_mean_scores + (
            1.0 - self.q_mean_weight
        ) * topk_peak_query_scores
        block_multi_peak_scores = self._reduce_multi_peak_scores(
            topk_query_scores, summary["topk_peak_counts"]
        )

        mean_weight = 1.0 - self.block_peak_weight - self.multi_peak_weight
        block_scores_per_head = (
            mean_weight * block_mean_scores
            + self.block_peak_weight * block_peak_scores
            + self.multi_peak_weight * block_multi_peak_scores
        )

        head_weights = self._compute_head_weights(block_scores_per_head, summary)
        block_scores = (block_scores_per_head * head_weights.unsqueeze(-1)).sum(dim=1)

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
            "head_weights": head_weights,
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

        all_block_indices = list(range(num_blocks))
        recent_count = self._resolve_recent_block_count(num_blocks)
        recent_block_indices = list(range(max(0, num_blocks - recent_count), num_blocks))

        protected_recent_indices = set(recent_block_indices)
        protected_hot_indices: set[int] = set()
        if self.protected_hot_blocks > 0 and num_blocks > 0:
            hot_candidates = [idx for idx in all_block_indices if idx not in protected_recent_indices]
            hot_tensor = self._select_top_block_indices(
                analysis["block_scores"],
                hot_candidates,
                self.protected_hot_blocks,
                keys.device,
            )
            protected_hot_indices = set(hot_tensor[0].tolist()) if hot_tensor.numel() > 0 else set()

        protected_tail_indices = {tail_block_idx} if has_partial_tail_block and num_blocks > 0 else set()
        protected_indices = protected_recent_indices | protected_hot_indices | protected_tail_indices

        if keep_budget == 0:
            kept_block_indices = torch.empty(keys.shape[0], 0, dtype=torch.long, device=keys.device)
        elif keep_budget >= num_blocks:
            kept_block_indices = torch.arange(num_blocks, device=keys.device).expand(keys.shape[0], -1)
        elif len(protected_indices) <= keep_budget:
            remaining_candidates = [idx for idx in all_block_indices if idx not in protected_indices]
            additional_keeps = keep_budget - len(protected_indices)
            selected_remaining = self._select_top_block_indices(
                analysis["block_scores"], remaining_candidates, additional_keeps, keys.device
            )
            protected_tensor = torch.tensor(
                sorted(protected_indices), dtype=torch.long, device=keys.device
            ).expand(keys.shape[0], -1)
            kept_block_indices = torch.cat([protected_tensor, selected_remaining], dim=-1).sort(dim=-1).values
        else:
            logger.info(
                "Requested compression is too aggressive for protected recent/hot blocks. "
                "Falling back to scoring within protected blocks to satisfy the keep budget."
            )
            forced_candidates = all_block_indices if not has_partial_tail_block else all_block_indices[:-1]
            selected_blocks = self._select_top_block_indices(
                analysis["block_scores"], forced_candidates, keep_budget - len(protected_tail_indices), keys.device
            )
            if has_partial_tail_block:
                tail_tensor = torch.full(
                    (keys.shape[0], 1), tail_block_idx, dtype=torch.long, device=keys.device
                )
                kept_block_indices = torch.cat([selected_blocks, tail_tensor], dim=-1).sort(dim=-1).values
            else:
                kept_block_indices = selected_blocks.sort(dim=-1).values

        token_indices = self.expand_blocks_to_token_indices(
            keys.shape[0], key_len, kept_block_indices, keys.device
        )
        analysis.update(
            {
                "num_blocks": num_blocks,
                "n_kept_blocks": kept_block_indices.shape[-1],
                "keep_budget": keep_budget,
                "kept_block_indices": kept_block_indices,
                "protected_recent_block_indices": torch.tensor(
                    sorted(protected_recent_indices), dtype=torch.long, device=keys.device
                ),
                "protected_hot_block_indices": torch.tensor(
                    sorted(protected_hot_indices), dtype=torch.long, device=keys.device
                ),
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
