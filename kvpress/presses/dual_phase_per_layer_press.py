# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import math
from typing import Literal, Optional

import torch
import torch.nn as nn
from transformers import QuantizedCache

from kvpress.presses.base_press import BasePress
from kvpress.presses.block_wise_press import BlockWisePress
from kvpress.utils import extract_keys_and_values


DecodeMode = Literal["permanent_fixed_budget", "compute_cold_fixed_budget", "hybrid_fixed_budget"]
PhaseName = Literal["prefill", "decode"]


@dataclass
class DualPhasePerLayerPress(BasePress):
    """
    Two-phase block policy for the current long-output decode experiments.

    The press intentionally keeps a small surface area:
    - prefill is always physical BlockWise compression
    - decode is either fixed-budget permanent eviction or fixed-budget compute-cold masking
    - decode budget is anchored to the number of blocks kept after prefill

    Older experimental paths such as per-layer ratio tables, score reuse, and
    offload/prefetch state simulation are intentionally not implemented here.
    """

    prefill_press: BlockWisePress
    decode_press: BlockWisePress
    decode_mode: DecodeMode = "compute_cold_fixed_budget"
    block_size: int = 16
    compression_interval: int = 16
    decode_hidden_states_buffer_size: int = 16
    decode_block_budget: int | None = None
    decode_cold_block_budget: int | None = None
    decode_top_p_threshold: float | None = None
    decode_skip_first_layers: int = 0
    decode_budget_scale: float = 1.0
    decode_cold_budget_scale: float = 1.0
    require_question_aware: bool = True

    layer_decode_steps: dict[int, int] = field(default_factory=lambda: defaultdict(int), init=False, repr=False)
    decode_hidden_states_buffer: dict[int, list[torch.Tensor]] = field(
        default_factory=lambda: defaultdict(list),
        init=False,
        repr=False,
    )
    layer_prefill_kept_blocks: dict[int, int] = field(default_factory=dict, init=False, repr=False)
    layer_cached_masks: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None] = field(
        default_factory=lambda: defaultdict(lambda: None),
        init=False,
        repr=False,
    )
    layer_block_states: dict[int, dict[str, torch.Tensor | int | str]] = field(
        default_factory=lambda: defaultdict(dict),
        init=False,
        repr=False,
    )

    @property
    def compression_ratio(self) -> float:
        return float(self.prefill_press.compression_ratio)

    @compression_ratio.setter
    def compression_ratio(self, value: float):
        ratio = float(value)
        assert 0 <= ratio < 1, "compression_ratio must be in [0, 1)"
        self.prefill_press.compression_ratio = ratio

    def __post_init__(self):
        assert self.decode_mode in {"permanent_fixed_budget", "compute_cold_fixed_budget", "hybrid_fixed_budget"}
        assert self.block_size > 0, "block_size must be > 0"
        assert self.compression_interval > 0, "compression_interval must be > 0"
        assert self.decode_hidden_states_buffer_size > 0, "decode_hidden_states_buffer_size must be > 0"
        assert self.decode_block_budget is None or self.decode_block_budget >= 0
        assert self.decode_cold_block_budget is None or self.decode_cold_block_budget >= 0
        assert self.decode_top_p_threshold is None or 0 < self.decode_top_p_threshold <= 1
        assert self.decode_skip_first_layers >= 0, "decode_skip_first_layers must be >= 0"
        assert self.decode_budget_scale >= 0, "decode_budget_scale must be >= 0"
        assert self.decode_cold_budget_scale >= 0, "decode_cold_budget_scale must be >= 0"
        assert isinstance(self.prefill_press, BlockWisePress), "prefill_press must be BlockWisePress"
        assert isinstance(self.decode_press, BlockWisePress), "decode_press must be BlockWisePress"
        self._sync_blockwise_presses()

    @classmethod
    def init_class_vars(
        cls,
        layer_phase_ratios: Optional[dict[int, list[float]]] = None,
        block_size: int = 16,
        default_phase_ratios: Optional[list[float]] = None,
        compression_interval: int = 16,
        decode_hidden_states_buffer_size: int = 16,
        **_: object,
    ) -> "DualPhasePerLayerPress":
        del layer_phase_ratios
        prefill_ratio = default_phase_ratios[0] if default_phase_ratios else 0.0
        prefill_press = BlockWisePress(compression_ratio=prefill_ratio, block_size=block_size, q_window_size=block_size)
        decode_press = BlockWisePress(compression_ratio=0.0, block_size=block_size, q_window_size=block_size)
        return cls(
            prefill_press=prefill_press,
            decode_press=decode_press,
            block_size=block_size,
            compression_interval=compression_interval,
            decode_hidden_states_buffer_size=decode_hidden_states_buffer_size,
        )

    def post_init_from_model(self, model):
        self.prefill_press.post_init_from_model(model)
        if id(self.decode_press) != id(self.prefill_press):
            self.decode_press.post_init_from_model(model)

    def reset(self):
        self.layer_decode_steps = defaultdict(int)
        self.decode_hidden_states_buffer = defaultdict(list)
        self.layer_prefill_kept_blocks = {}
        self.layer_cached_masks = defaultdict(lambda: None)
        self.layer_block_states = defaultdict(dict)

    def forward_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: list):
        del input
        hidden_states = kwargs["hidden_states"]
        cache = kwargs["past_key_values"]
        layer_idx = self._resolve_layer_idx(module)
        phase = self._resolve_phase(hidden_states, kwargs)

        if phase == "prefill":
            module.masked_key_indices = None
            keys, values = extract_keys_and_values(cache, layer_idx)
            attentions = output[1] if len(output) > 1 and output[1] is not None else None
            keys, values = self._compress_prefill(module, hidden_states, keys, values, attentions, kwargs)
            self._write_back_cache(cache, layer_idx, keys, values)
            return output

        if layer_idx < self.decode_skip_first_layers:
            module.masked_key_indices = None
            self.layer_cached_masks[layer_idx] = None
            self.layer_block_states[layer_idx] = {
                "mode": "decode_skipped_layer",
            }
            return output

        self._append_decode_hidden_states(layer_idx, hidden_states)
        self.layer_decode_steps[layer_idx] += hidden_states.shape[1]

        keys, values = extract_keys_and_values(cache, layer_idx)
        has_decode_state = self.layer_block_states[layer_idx].get("mode") in {
            "permanent_fixed_budget",
            "compute_cold_fixed_budget",
            "hybrid_fixed_budget",
        }
        should_refresh = (
            self.layer_decode_steps[layer_idx] >= self.compression_interval
            or not has_decode_state
        )
        if not should_refresh:
            module.masked_key_indices = self._validate_mask(self.layer_cached_masks[layer_idx], keys)
            self.layer_cached_masks[layer_idx] = module.masked_key_indices
            return output

        self.layer_decode_steps[layer_idx] = 0
        buffered_hidden_states = torch.cat(self.decode_hidden_states_buffer[layer_idx], dim=1)
        attentions = output[1] if len(output) > 1 and output[1] is not None else None

        if self.decode_mode == "permanent_fixed_budget":
            keys, values = self._permanent_decode_step(
                module,
                buffered_hidden_states,
                keys,
                values,
                attentions,
                kwargs,
            )
            self._write_back_cache(cache, layer_idx, keys, values)
        elif self.decode_mode == "hybrid_fixed_budget":
            keys, values = self._hybrid_decode_step(
                module,
                buffered_hidden_states,
                keys,
                values,
                attentions,
                kwargs,
            )
            self._write_back_cache(cache, layer_idx, keys, values)
        else:
            self._compute_cold_decode_step(
                module,
                buffered_hidden_states,
                keys,
                values,
                attentions,
                kwargs,
            )

        return output

    def _compress_prefill(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor | None,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        original_ratio = self.prefill_press.compression_ratio
        try:
            keys, values = self.prefill_press.compress(module, hidden_states, keys, values, attentions, kwargs)
        finally:
            self.prefill_press.compression_ratio = original_ratio

        layer_idx = self._resolve_layer_idx(module)
        kept_blocks = math.ceil(keys.shape[2] / self.block_size)
        self.layer_prefill_kept_blocks[layer_idx] = kept_blocks
        self.layer_cached_masks[layer_idx] = None
        self.layer_block_states[layer_idx] = {
            "mode": "prefill",
            "active_blocks": kept_blocks,
            "live_blocks": kept_blocks,
            "deleted_blocks": 0,
        }
        return keys, values

    def _permanent_decode_step(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor | None,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        layer_idx = self._resolve_layer_idx(module)
        keep_budget = self._resolve_decode_budget(layer_idx, keys, cold=False)
        plan = self.decode_press.build_block_plan(
            module,
            hidden_states,
            keys,
            values,
            attentions,
            kwargs,
            compression_ratio=0.0,
            keep_budget=keep_budget,
            top_p_threshold=self.decode_top_p_threshold if keep_budget is None else None,
            force_refresh_summary=True,
        )
        original_num_blocks = plan["num_blocks"]
        deleted_block_indices = self._complement_block_indices(
            original_num_blocks,
            plan["kept_block_indices"],
            keys.device,
        )
        keys, values = self.decode_press.gather_by_token_indices(keys, values, plan["token_indices"])
        self.decode_press.build_or_refresh_block_summary(module, keys, values, force_refresh=True)
        module.masked_key_indices = None
        self.layer_cached_masks[layer_idx] = None
        self.layer_block_states[layer_idx] = {
            "mode": "permanent_fixed_budget",
            "active": plan["kept_block_indices"].detach().clone(),
            "permanently_deleted": deleted_block_indices.detach().clone(),
            "active_blocks": int(plan["n_kept_blocks"]),
            "live_blocks": math.ceil(keys.shape[2] / self.block_size),
            "deleted_blocks": int(deleted_block_indices.shape[-1]),
            "keep_budget": int(plan["keep_budget"]),
        }
        return keys, values

    def _compute_cold_decode_step(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor | None,
        kwargs: dict,
    ) -> None:
        layer_idx = self._resolve_layer_idx(module)
        keep_budget = self._resolve_decode_budget(layer_idx, keys, cold=True)
        plan = self.decode_press.build_block_plan(
            module,
            hidden_states,
            keys,
            values,
            attentions,
            kwargs,
            compression_ratio=0.0,
            keep_budget=keep_budget,
            top_p_threshold=self.decode_top_p_threshold if keep_budget is None else None,
            force_refresh_summary=True,
        )
        mask = self._build_mask_from_active_blocks(keys, self.block_size, plan["kept_block_indices"])
        mask = self._validate_mask(mask, keys)
        self.layer_cached_masks[layer_idx] = mask
        module.masked_key_indices = mask
        self.layer_block_states[layer_idx] = {
            "mode": "compute_cold_fixed_budget",
            "active": plan["kept_block_indices"].detach().clone(),
            "active_blocks": int(plan["n_kept_blocks"]),
            "live_blocks": math.ceil(keys.shape[2] / self.block_size),
            "masked_tokens": 0 if mask is None else int(mask[0].numel()),
            "keep_budget": int(plan["keep_budget"]),
        }

    def _hybrid_decode_step(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor | None,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        layer_idx = self._resolve_layer_idx(module)
        total_keep_budget = self._resolve_decode_budget(layer_idx, keys, cold=False)
        active_keep_budget = self._resolve_decode_budget(layer_idx, keys, cold=True)
        if total_keep_budget is not None:
            total_keep_budget = max(0, total_keep_budget)
        if active_keep_budget is not None:
            active_keep_budget = max(0, active_keep_budget)
        if total_keep_budget is not None and active_keep_budget is not None:
            active_keep_budget = min(active_keep_budget, total_keep_budget)

        total_plan = self.decode_press.build_block_plan(
            module,
            hidden_states,
            keys,
            values,
            attentions,
            kwargs,
            compression_ratio=0.0,
            keep_budget=total_keep_budget,
            top_p_threshold=self.decode_top_p_threshold if total_keep_budget is None else None,
            force_refresh_summary=True,
        )

        original_num_blocks = total_plan["num_blocks"]
        deleted_block_indices = self._complement_block_indices(
            original_num_blocks,
            total_plan["kept_block_indices"],
            keys.device,
        )
        retained_keys, retained_values = self.decode_press.gather_by_token_indices(
            keys,
            values,
            total_plan["token_indices"],
        )

        active_plan = self.decode_press.build_block_plan(
            module,
            hidden_states,
            retained_keys,
            retained_values,
            attentions,
            kwargs,
            compression_ratio=0.0,
            keep_budget=active_keep_budget,
            top_p_threshold=self.decode_top_p_threshold if active_keep_budget is None else None,
            force_refresh_summary=True,
        )
        self.decode_press.build_or_refresh_block_summary(
            module,
            retained_keys,
            retained_values,
            force_refresh=True,
        )
        mask = self._build_mask_from_active_blocks(
            retained_keys,
            self.block_size,
            active_plan["kept_block_indices"],
        )
        mask = self._validate_mask(mask, retained_keys)
        self.layer_cached_masks[layer_idx] = mask
        module.masked_key_indices = mask
        self.layer_block_states[layer_idx] = {
            "mode": "hybrid_fixed_budget",
            "active": active_plan["kept_block_indices"].detach().clone(),
            "retained": total_plan["kept_block_indices"].detach().clone(),
            "permanently_deleted": deleted_block_indices.detach().clone(),
            "active_blocks": int(active_plan["n_kept_blocks"]),
            "live_blocks": math.ceil(retained_keys.shape[2] / self.block_size),
            "deleted_blocks": int(deleted_block_indices.shape[-1]),
            "masked_tokens": 0 if mask is None else int(mask[0].numel()),
            "keep_budget": int(total_plan["keep_budget"]),
            "active_budget": int(active_plan["keep_budget"]),
        }
        return retained_keys, retained_values

    def _resolve_decode_budget(self, layer_idx: int, keys: torch.Tensor, cold: bool) -> int | None:
        num_blocks = math.ceil(keys.shape[2] / self.block_size)
        explicit_budget = self.decode_cold_block_budget if cold else self.decode_block_budget
        if explicit_budget is not None:
            base_budget = explicit_budget
        elif self.decode_top_p_threshold is not None:
            return None
        else:
            prefill_budget = self.layer_prefill_kept_blocks.get(layer_idx, num_blocks)
            scale = self.decode_cold_budget_scale if cold else self.decode_budget_scale
            base_budget = int(math.ceil(prefill_budget * scale))
        # The recent decode tail is protected on top of the historical budget.
        total_budget = base_budget + self.decode_press.protected_recent_blocks
        return min(num_blocks, max(0, total_budget))

    def _append_decode_hidden_states(self, layer_idx: int, hidden_states: torch.Tensor) -> None:
        self.decode_hidden_states_buffer[layer_idx].append(hidden_states.detach().clone())
        self.decode_hidden_states_buffer[layer_idx] = self.decode_hidden_states_buffer[layer_idx][
            -self.decode_hidden_states_buffer_size :
        ]

    def _sync_blockwise_presses(self) -> None:
        self.prefill_press.block_size = self.block_size
        self.decode_press.block_size = self.block_size
        self.decode_press.q_window_size = self.block_size

    def _resolve_phase(self, hidden_states: torch.Tensor, kwargs: dict) -> PhaseName:
        q_len = hidden_states.shape[1]
        return "prefill" if kwargs["cache_position"][-1] <= q_len else "decode"

    def _resolve_layer_idx(self, module: nn.Module) -> int:
        raw = getattr(module, "layer_idx")
        if isinstance(raw, torch.Tensor):
            return int(raw.item())
        return int(raw)

    def _build_mask_from_active_blocks(
        self,
        keys: torch.Tensor,
        block_size: int,
        active_block_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        num_blocks = math.ceil(keys.shape[2] / block_size)
        if num_blocks == 0:
            return None

        active_mask = torch.zeros(keys.shape[0], num_blocks, dtype=torch.bool, device=keys.device)
        if active_block_indices.numel() > 0:
            valid_blocks = active_block_indices.clamp(min=0, max=max(num_blocks - 1, 0))
            active_mask.scatter_(1, valid_blocks, True)

        token_blocks = torch.arange(keys.shape[2], device=keys.device) // block_size
        inactive_tokens = ~active_mask[:, token_blocks]
        inactive_tokens = inactive_tokens[:, None, :].expand(-1, keys.shape[1], -1)
        batch_indices, head_indices, seq_indices = inactive_tokens.nonzero(as_tuple=True)
        if batch_indices.numel() == 0:
            return None
        return batch_indices, head_indices, seq_indices

    def _validate_mask(self, mask, keys: torch.Tensor):
        if mask is None:
            return None
        batch_indices, head_indices, seq_indices = mask
        valid_indices = (
            (batch_indices >= 0)
            & (batch_indices < keys.shape[0])
            & (head_indices >= 0)
            & (head_indices < keys.shape[1])
            & (seq_indices >= 0)
            & (seq_indices < keys.shape[2])
        )
        if bool(valid_indices.all()):
            return mask
        batch_indices = batch_indices[valid_indices]
        head_indices = head_indices[valid_indices]
        seq_indices = seq_indices[valid_indices]
        if batch_indices.numel() == 0:
            return None
        return batch_indices, head_indices, seq_indices

    def _complement_block_indices(self, num_blocks: int, kept: torch.Tensor, device: torch.device) -> torch.Tensor:
        if num_blocks == 0:
            return torch.empty(kept.shape[0], 0, dtype=torch.long, device=device)
        universe = torch.arange(num_blocks, device=device).expand(kept.shape[0], -1)
        diff_lists = []
        for batch_idx in range(universe.shape[0]):
            mask = torch.ones(num_blocks, dtype=torch.bool, device=device)
            if kept.shape[1] > 0:
                mask[kept[batch_idx]] = False
            diff_lists.append(universe[batch_idx][mask])
        return self._stack_or_empty(diff_lists, device)

    def _stack_or_empty(self, tensors: list[torch.Tensor], device: torch.device) -> torch.Tensor:
        if not tensors:
            return torch.empty(0, 0, dtype=torch.long, device=device)
        max_len = max(t.numel() for t in tensors)
        if max_len == 0:
            return torch.empty(len(tensors), 0, dtype=torch.long, device=device)
        padded = []
        for tensor in tensors:
            if tensor.numel() == max_len:
                padded.append(tensor)
                continue
            pad = torch.full((max_len - tensor.numel(),), -1, dtype=torch.long, device=device)
            padded.append(torch.cat([tensor, pad], dim=0))
        return torch.stack(padded, dim=0)

    def _write_back_cache(self, cache, layer_idx: int, keys: torch.Tensor, values: torch.Tensor) -> None:
        cache_layer = cache.layers[layer_idx]
        if isinstance(cache, QuantizedCache):
            cache_layer._quantized_keys = cache_layer._quantize(keys, axis=cache_layer.axis_key)
            cache_layer._quantized_values = cache_layer._quantize(values, axis=cache_layer.axis_value)
            cache_layer.keys = torch.zeros(0, dtype=keys.dtype, device=keys.device)  # type: ignore[index]
            cache_layer.values = torch.zeros(0, dtype=keys.dtype, device=keys.device)  # type: ignore[index]
            cache_layer.cumulative_length = keys.shape[2]
        else:
            cache_layer.keys = keys
            cache_layer.values = values
