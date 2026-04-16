# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
from typing import Any

import torch
from torch import nn

from kvpress.presses.base_press import BasePress
from kvpress.presses.blockwise_components import (
    HEAD_AGG_MODES,
    QUERY_AGG_MODES,
    REPRESENTATIVE_MODES,
    SUMMARY_MODES,
    aggregate_head_scores,
    aggregate_query_scores,
    resolve_query_topr,
    select_representative_indices,
)
from kvpress.utils import get_prerope_query_states


logger = logging.getLogger(__name__)


@dataclass
class BlockWisePress(BasePress):
    """
    Configurable block-granularity KV compression for prefill.

    The implementation is split into four independently configurable steps:
    1. build block summaries
    2. select block-internal representatives
    3. aggregate the tail query window
    4. aggregate head-level block scores
    """

    compression_ratio: float = 0.0
    block_size: int = 16
    q_window_size: int = 32
    summary_topk_keys: int = 4
    mean_key_weight: float = 0.75
    prefix_sink_blocks: int = 1
    protected_recent_blocks: int = 2
    eps: float = 1e-6
    require_question_aware: bool = True

    summary_mode: str = "mean_plus_norm_topk_mean"
    representative_mode: str = "key_norm"
    query_agg_mode: str = "mean"
    head_agg_mode: str = "uniform_mean"
    representative_k: int = 4
    multi_rep_k: int = 4
    query_topr: int | None = None
    head_topk: int = 1
    random_seed: int = 42

    last_block_heat: dict[int, torch.Tensor] = field(default_factory=dict, init=False, repr=False)
    last_block_heat_ema: dict[int, torch.Tensor] = field(default_factory=dict, init=False, repr=False)
    last_block_summary: dict[int, dict[str, torch.Tensor]] = field(default_factory=dict, init=False, repr=False)
    last_kept_block_indices: dict[int, torch.Tensor] = field(default_factory=dict, init=False, repr=False)
    last_kept_token_indices: dict[int, torch.Tensor] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        assert 0 <= self.compression_ratio < 1, "compression_ratio must be in [0, 1)"
        assert self.block_size > 0, "block_size must be > 0"
        assert self.q_window_size > 0, "q_window_size must be > 0"
        assert self.summary_topk_keys > 0, "summary_topk_keys must be > 0"
        assert 0 <= self.mean_key_weight <= 1, "mean_key_weight must be in [0, 1]"
        assert self.prefix_sink_blocks >= 0, "prefix_sink_blocks must be >= 0"
        assert self.protected_recent_blocks >= 0, "protected_recent_blocks must be >= 0"
        assert self.summary_mode in SUMMARY_MODES, f"Unsupported summary_mode: {self.summary_mode}"
        assert self.representative_mode in REPRESENTATIVE_MODES, (
            f"Unsupported representative_mode: {self.representative_mode}"
        )
        assert self.query_agg_mode in QUERY_AGG_MODES, (
            f"Unsupported query_agg_mode: {self.query_agg_mode}"
        )
        assert self.head_agg_mode in HEAD_AGG_MODES, (
            f"Unsupported head_agg_mode: {self.head_agg_mode}"
        )
        assert self.representative_k > 0, "representative_k must be > 0"
        assert self.multi_rep_k > 0, "multi_rep_k must be > 0"
        assert self.head_topk > 0, "head_topk must be > 0"

    def _resolve_layer_idx(self, module: nn.Module) -> int:
        raw = getattr(module, "layer_idx", 0)
        if isinstance(raw, torch.Tensor):
            return int(raw.item())
        return int(raw)

    def _resolve_q_window(self, q_len: int) -> int:
        return min(q_len, self.q_window_size)

    def _resolve_summary_topk(self) -> int:
        return min(self.summary_topk_keys, self.block_size)

    def _resolve_representative_k(self) -> int:
        return min(self.representative_k, self.block_size)

    def _resolve_multi_rep_k(self) -> int:
        return min(self.multi_rep_k, self.block_size)

    def blend_score_reuse_hint(
        self,
        block_scores: torch.Tensor,
        score_reuse_hint: torch.Tensor | None = None,
        score_reuse_weight: float = 0.0,
    ) -> tuple[torch.Tensor, bool]:
        if score_reuse_hint is None or score_reuse_weight <= 0:
            return block_scores, False
        if score_reuse_hint.shape != block_scores.shape:
            return block_scores, False
        blended = (1.0 - score_reuse_weight) * block_scores + score_reuse_weight * score_reuse_hint
        return blended, True

    def _empty_summary(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        bsz, num_key_value_heads, _, head_dim = keys.shape
        return {
            "num_blocks": torch.tensor(0, dtype=torch.long, device=keys.device),
            "mean_keys": keys.new_zeros((bsz, num_key_value_heads, 0, head_dim)),
            "topk_key_means": keys.new_zeros((bsz, num_key_value_heads, 0, head_dim)),
            "multi_rep_keys": keys.new_zeros((bsz, num_key_value_heads, 0, 0, head_dim)),
            "mean_values": values.new_zeros((bsz, num_key_value_heads, 0, head_dim)),
            "token_counts": torch.zeros((bsz, 0), dtype=torch.long, device=keys.device),
        }

    def _summarize_blocks(
        self,
        module: nn.Module,
        keys: torch.Tensor,
        values: torch.Tensor,
        tail_query_states: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        bsz, num_key_value_heads, key_len, head_dim = keys.shape
        num_blocks = math.ceil(key_len / self.block_size)
        if num_blocks == 0:
            return self._empty_summary(keys, values)

        topk = self._resolve_summary_topk()
        representative_k = self._resolve_representative_k()
        multi_rep_k = self._resolve_multi_rep_k()
        layer_idx = self._resolve_layer_idx(module)

        mean_keys = []
        topk_key_means = []
        mean_values = []
        multi_rep_keys = []
        token_counts = []

        for block_idx in range(num_blocks):
            start = block_idx * self.block_size
            end = min(start + self.block_size, key_len)
            block_keys = keys[:, :, start:end]
            block_values = values[:, :, start:end]
            block_len = end - start

            mean_keys.append(block_keys.mean(dim=2))
            mean_values.append(block_values.mean(dim=2))

            topk_token_indices = select_representative_indices(
                mode=self.representative_mode,
                block_keys=block_keys,
                representative_k=min(topk, representative_k, block_len),
                block_start=start,
                layer_idx=layer_idx,
                seed=self.random_seed,
                tail_query_states=tail_query_states,
            )
            topk_gather = topk_token_indices[..., None].expand(-1, -1, -1, head_dim)
            topk_keys = block_keys.gather(2, topk_gather)
            topk_key_means.append(topk_keys.mean(dim=2))

            multi_rep_indices = select_representative_indices(
                mode=self.representative_mode,
                block_keys=block_keys,
                representative_k=min(multi_rep_k, block_len),
                block_start=start,
                layer_idx=layer_idx,
                seed=self.random_seed + 1009,
                tail_query_states=tail_query_states,
            )
            multi_rep_gather = multi_rep_indices[..., None].expand(-1, -1, -1, head_dim)
            selected_keys = block_keys.gather(2, multi_rep_gather)
            if selected_keys.shape[2] < multi_rep_k:
                pad_count = multi_rep_k - selected_keys.shape[2]
                pad_source = selected_keys[:, :, -1:, :].expand(-1, -1, pad_count, -1)
                selected_keys = torch.cat([selected_keys, pad_source], dim=2)
            multi_rep_keys.append(selected_keys)

            token_counts.append(torch.full((bsz,), block_len, dtype=torch.long, device=keys.device))

        return {
            "num_blocks": torch.tensor(num_blocks, dtype=torch.long, device=keys.device),
            "mean_keys": torch.stack(mean_keys, dim=2),
            "topk_key_means": torch.stack(topk_key_means, dim=2),
            "multi_rep_keys": torch.stack(multi_rep_keys, dim=2),
            "mean_values": torch.stack(mean_values, dim=2),
            "token_counts": torch.stack(token_counts, dim=1),
        }

    def build_or_refresh_block_summary(
        self,
        module: nn.Module,
        keys: torch.Tensor,
        values: torch.Tensor,
        force_refresh: bool = False,
        tail_query_states: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        layer_idx = self._resolve_layer_idx(module)
        cached = self.last_block_summary.get(layer_idx)
        expected_blocks = math.ceil(keys.shape[2] / self.block_size)
        if (
            not force_refresh
            and cached is not None
            and int(cached["key_len"].item()) == keys.shape[2]
            and int(cached["num_blocks"].item()) == expected_blocks
        ):
            return cached

        summary = self._summarize_blocks(module, keys, values, tail_query_states=tail_query_states)
        summary["key_len"] = torch.tensor(keys.shape[2], dtype=torch.long, device=keys.device)
        self.last_block_summary[layer_idx] = summary
        return summary

    def _repeat_kv_queries(self, module: nn.Module, query_states: torch.Tensor, num_key_value_heads: int):
        num_key_value_groups = module.config.num_attention_heads // num_key_value_heads
        repeated_queries = query_states.view(
            query_states.shape[0],
            num_key_value_heads,
            num_key_value_groups,
            query_states.shape[2],
            query_states.shape[3],
        ).mean(dim=2)
        return repeated_queries, num_key_value_groups

    def _score_summary_anchors(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
    ) -> torch.Tensor:
        return torch.einsum("bhqd,bhkd->bhqk", query_states, key_states) / math.sqrt(query_states.shape[-1])

    def _compute_summary_scores_per_head(
        self,
        query_states: torch.Tensor,
        summary: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        mean_scores = aggregate_query_scores(
            self._score_summary_anchors(query_states, summary["mean_keys"]),
            mode=self.query_agg_mode,
            topr=resolve_query_topr(query_states.shape[-2], self.query_topr),
        )
        if self.summary_mode == "mean_only":
            return mean_scores

        topk_scores = aggregate_query_scores(
            self._score_summary_anchors(query_states, summary["topk_key_means"]),
            mode=self.query_agg_mode,
            topr=resolve_query_topr(query_states.shape[-2], self.query_topr),
        )
        if self.summary_mode == "norm_topk_mean_only":
            return topk_scores
        if self.summary_mode == "mean_plus_norm_topk_mean":
            return self.mean_key_weight * mean_scores + (1.0 - self.mean_key_weight) * topk_scores
        if self.summary_mode == "multi_rep_max":
            rep_scores = torch.einsum(
                "bhqd,bhkrd->bhqkr",
                query_states,
                summary["multi_rep_keys"],
            ) / math.sqrt(query_states.shape[-1])
            rep_scores = rep_scores.max(dim=-1).values
            return aggregate_query_scores(
                rep_scores,
                mode=self.query_agg_mode,
                topr=resolve_query_topr(query_states.shape[-2], self.query_topr),
            )
        if self.summary_mode == "adaptive_fusion_v1":
            rep_scores = torch.einsum(
                "bhqd,bhkrd->bhqkr",
                query_states,
                summary["multi_rep_keys"],
            ) / math.sqrt(query_states.shape[-1])
            rep_scores = rep_scores.max(dim=-1).values
            rep_scores = aggregate_query_scores(
                rep_scores,
                mode=self.query_agg_mode,
                topr=resolve_query_topr(query_states.shape[-2], self.query_topr),
            )

            token_counts = summary["token_counts"].to(query_states.dtype)[:, None, :]
            mean_norm = summary["mean_keys"].norm(dim=-1)
            topk_norm = summary["topk_key_means"].norm(dim=-1)
            rep_norms = summary["multi_rep_keys"].norm(dim=-1)

            concentration = (topk_norm / mean_norm.clamp_min(self.eps)).clamp(0.0, 4.0)
            norm_var = rep_norms.var(dim=-1, unbiased=False)
            rep_center = rep_norms.mean(dim=-1, keepdim=True)
            rep_dispersion = ((rep_norms - rep_center) ** 2).mean(dim=-1)

            topk_weight = (concentration - 1.0).clamp_min(0.0)
            rep_weight = (rep_dispersion + norm_var).sqrt()
            mean_weight = token_counts.reciprocal().sqrt().expand_as(topk_weight)

            weights = torch.stack([mean_weight, topk_weight, rep_weight], dim=-1)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(self.eps)

            stacked_scores = torch.stack([mean_scores, topk_scores, rep_scores], dim=-1)
            return (stacked_scores * weights).sum(dim=-1)
        raise ValueError(f"Unsupported summary_mode: {self.summary_mode}")

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

        bsz, num_key_value_heads, key_len, _ = keys.shape
        if key_len == 0:
            summary = self._empty_summary(keys, values)
            return {
                "q_window": 0,
                "block_summary": summary,
                "block_scores_per_head": keys.new_zeros((bsz, num_key_value_heads, 0)),
                "block_scores": keys.new_zeros((bsz, 0)),
            }

        q_window = self._resolve_q_window(hidden_states.shape[1])
        query_states = get_prerope_query_states(module, hidden_states[:, -q_window:])
        kv_query_states, _ = self._repeat_kv_queries(module, query_states, num_key_value_heads)
        summary = self.build_or_refresh_block_summary(
            module,
            keys,
            values,
            force_refresh=force_refresh_summary,
            tail_query_states=kv_query_states if self.representative_mode == "tail_query_relevance" else None,
        )

        summary_scores_per_head = self._compute_summary_scores_per_head(kv_query_states, summary)
        block_scores = aggregate_head_scores(
            summary_scores_per_head,
            self.head_agg_mode,
            self.eps,
            topk=self.head_topk,
        )

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
            "summary_scores_per_head": summary_scores_per_head,
            "block_scores_per_head": summary_scores_per_head,
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
        keep_budget: int | None = None,
        force_refresh_summary: bool = False,
        score_reuse_hint: torch.Tensor | None = None,
        score_reuse_weight: float = 0.0,
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
        blended_block_scores, reuse_applied = self.blend_score_reuse_hint(
            analysis["block_scores"],
            score_reuse_hint=score_reuse_hint,
            score_reuse_weight=score_reuse_weight,
        )
        analysis["raw_block_scores"] = analysis["block_scores"]
        analysis["reused_block_scores"] = score_reuse_hint
        analysis["score_reuse_applied"] = reuse_applied
        analysis["block_scores"] = blended_block_scores

        key_len = keys.shape[2]
        num_blocks = analysis["block_scores"].shape[-1]
        ratio = self.compression_ratio if compression_ratio is None else compression_ratio

        if keep_budget is None:
            effective_keep_budget = min(num_blocks, max(0, int(math.ceil(num_blocks * (1.0 - ratio)))))
        else:
            effective_keep_budget = min(num_blocks, max(0, int(keep_budget)))
        has_partial_tail_block = key_len % self.block_size != 0
        tail_block_idx = num_blocks - 1

        if effective_keep_budget == 0:
            kept_block_indices = torch.empty(keys.shape[0], 0, dtype=torch.long, device=keys.device)
        elif effective_keep_budget >= num_blocks:
            kept_block_indices = torch.arange(num_blocks, device=keys.device).expand(keys.shape[0], -1)
        else:
            sink_count = min(self.prefix_sink_blocks, num_blocks)
            recent_count = min(self.protected_recent_blocks, num_blocks)
            protected_sink_indices = set(range(sink_count))
            protected_recent_indices = set(range(max(0, num_blocks - recent_count), num_blocks))
            protected_tail_indices = {tail_block_idx} if has_partial_tail_block and num_blocks > 0 else set()
            protected_indices = protected_sink_indices | protected_recent_indices | protected_tail_indices

            if len(protected_indices) <= effective_keep_budget:
                remaining_candidates = [idx for idx in range(num_blocks) if idx not in protected_indices]
                additional_keeps = effective_keep_budget - len(protected_indices)
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
                    "Requested compression is too aggressive: sink/recent protected blocks exceed keep budget. "
                    "Falling back to score-based selection over all blocks."
                )
                kept_block_indices = self._select_top_block_indices(
                    analysis["block_scores"], list(range(num_blocks)), effective_keep_budget, keys.device
                ).sort(dim=-1).values

        token_indices = self.expand_blocks_to_token_indices(
            keys.shape[0], key_len, kept_block_indices, keys.device
        )
        analysis.update(
            {
                "num_blocks": num_blocks,
                "n_kept_blocks": kept_block_indices.shape[-1],
                "keep_budget": effective_keep_budget,
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
        layer_idx = self._resolve_layer_idx(module)
        self.last_kept_block_indices[layer_idx] = plan["kept_block_indices"].detach().clone()
        self.last_kept_token_indices[layer_idx] = plan["token_indices"].detach().clone()
        compressed_keys, compressed_values = self.gather_by_token_indices(
            keys, values, plan["token_indices"]
        )
        self.build_or_refresh_block_summary(
            module, compressed_keys, compressed_values, force_refresh=True
        )
        return compressed_keys, compressed_values
