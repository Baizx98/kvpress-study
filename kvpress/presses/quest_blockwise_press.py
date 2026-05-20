# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from kvpress.presses.block_wise_press import BlockWisePress
from kvpress.presses.blockwise_components import (
    aggregate_head_scores,
    aggregate_query_scores,
    resolve_query_topr,
)
from kvpress.utils import get_prerope_query_states


@dataclass
class QuestBlockwisePress(BlockWisePress):
    """
    Quest-style prefill block scorer using per-block key min/max envelopes.

    The physical compression unit remains a full block, which makes it directly
    comparable to BlockWisePress while changing only the coarse scoring rule.
    """

    quest_score_mode: str = "minmax"

    def __post_init__(self):
        super().__post_init__()
        assert self.quest_score_mode in {"minmax"}, (
            f"Unsupported quest_score_mode: {self.quest_score_mode}"
        )

    def _empty_summary(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        summary = super()._empty_summary(keys, values)
        bsz, num_key_value_heads, _, head_dim = keys.shape
        summary["min_keys"] = keys.new_zeros((bsz, num_key_value_heads, 0, head_dim))
        summary["max_keys"] = keys.new_zeros((bsz, num_key_value_heads, 0, head_dim))
        return summary

    def _summarize_blocks(
        self,
        module,
        keys: torch.Tensor,
        values: torch.Tensor,
        tail_query_states: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del module, values, tail_query_states
        bsz, num_key_value_heads, key_len, head_dim = keys.shape
        num_blocks = (key_len + self.block_size - 1) // self.block_size
        if num_blocks == 0:
            return self._empty_summary(keys, values)

        min_keys = []
        max_keys = []
        token_counts = []
        for block_idx in range(num_blocks):
            start = block_idx * self.block_size
            end = min(start + self.block_size, key_len)
            block_keys = keys[:, :, start:end]
            min_keys.append(block_keys.amin(dim=2))
            max_keys.append(block_keys.amax(dim=2))
            token_counts.append(torch.full((bsz,), end - start, dtype=torch.long, device=keys.device))

        return {
            "num_blocks": torch.tensor(num_blocks, dtype=torch.long, device=keys.device),
            "min_keys": torch.stack(min_keys, dim=2),
            "max_keys": torch.stack(max_keys, dim=2),
            "token_counts": torch.stack(token_counts, dim=1),
        }

    def analyze_blocks(
        self,
        module,
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
        )

        positive_query = kv_query_states.clamp_min(0)
        negative_query = kv_query_states.clamp_max(0)
        coarse_scores = torch.einsum(
            "bhqd,bhkd->bhqk",
            positive_query,
            summary["max_keys"],
        ) + torch.einsum(
            "bhqd,bhkd->bhqk",
            negative_query,
            summary["min_keys"],
        )
        coarse_scores = coarse_scores / (kv_query_states.shape[-1] ** 0.5)
        block_scores_per_head = aggregate_query_scores(
            coarse_scores,
            mode=self.query_agg_mode,
            topr=resolve_query_topr(kv_query_states.shape[-2], self.query_topr),
        )
        block_scores = aggregate_head_scores(
            block_scores_per_head,
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
            "summary_scores_per_head": block_scores_per_head,
            "block_scores_per_head": block_scores_per_head,
            "block_scores": block_scores,
        }
