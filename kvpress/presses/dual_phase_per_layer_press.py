# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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

PhaseName = Literal["prefill", "decode"]


@dataclass
class DualPhasePerLayerPress(BasePress):
    """
    Layer-aware / phase-aware block policy on top of BlockWisePress.

    Decode uses two levels of approximation:
    - full refresh only every `score_refresh_interval` steps
    - block heat EMA reused between refreshes

    The press also emits logical block states for a future offload system.
    """

    prefill_press: BlockWisePress
    decode_press: BlockWisePress

    layer_phase_ratios: dict[int, list[float]] = field(default_factory=dict)
    default_phase_ratios: list[float] = field(default_factory=lambda: [0.0, 0.0])
    layer_phase_cold_ratios: dict[int, list[float]] = field(default_factory=dict)
    default_phase_cold_ratios: list[float] = field(default_factory=lambda: [0.0, 0.0])

    prefill_layer_presses: dict[int, BlockWisePress] = field(default_factory=dict)
    decode_layer_presses: dict[int, BlockWisePress] = field(default_factory=dict)

    block_size: int = 16
    compression_interval: int = 1
    score_refresh_interval: int = 4
    decode_hidden_states_buffer_size: int = 32
    history_momentum: float = 0.8
    resident_gpu_ratio: float = 0.5
    prefetch_ratio: float = 0.5
    require_question_aware: bool = True

    @property
    def compression_ratio(self) -> float:
        return float(self.default_phase_ratios[0])

    @compression_ratio.setter
    def compression_ratio(self, value: float):
        ratio = float(value)
        assert 0 <= ratio < 1, "compression_ratio must be in [0, 1)"
        self.default_phase_ratios = [ratio, ratio]

    def __post_init__(self):
        assert self.compression_interval > 0, "compression_interval must be > 0"
        assert self.score_refresh_interval > 0, "score_refresh_interval must be > 0"
        assert len(self.default_phase_ratios) == 2, "default_phase_ratios must be [prefill_ratio, decode_ratio]"
        assert len(self.default_phase_cold_ratios) == 2, "default_phase_cold_ratios must be [prefill_cold, decode_cold]"
        assert self.decode_hidden_states_buffer_size > 0, "decode_hidden_states_buffer_size must be > 0"
        assert 0 <= self.history_momentum < 1, "history_momentum must be in [0, 1)"
        assert 0 <= self.resident_gpu_ratio <= 1, "resident_gpu_ratio must be in [0, 1]"
        assert 0 <= self.prefetch_ratio <= 1, "prefetch_ratio must be in [0, 1]"
        assert isinstance(self.prefill_press, BlockWisePress), "prefill_press must be BlockWisePress"
        assert isinstance(self.decode_press, BlockWisePress), "decode_press must be BlockWisePress"

        self.layer_decode_steps = defaultdict(int)
        self.layer_score_steps = defaultdict(int)
        self.layer_decode_generated_tokens = defaultdict(int)
        self.decode_hidden_states_buffer = defaultdict(list)
        self.layer_block_states = defaultdict(dict)
        self.layer_cached_masks = defaultdict(lambda: None)
        self.layer_heat_ema = defaultdict(lambda: None)

    @classmethod
    def init_class_vars(
        cls,
        layer_phase_ratios: dict[int, list[float]],
        block_size: int = 16,
        default_phase_ratios: Optional[list[float]] = None,
        compression_interval: int = 1,
        prefill_layer_presses: Optional[dict[int, BlockWisePress]] = None,
        decode_layer_presses: Optional[dict[int, BlockWisePress]] = None,
        layer_phase_cold_ratios: Optional[dict[int, list[float]]] = None,
        default_phase_cold_ratios: Optional[list[float]] = None,
        decode_hidden_states_buffer_size: int = 32,
        score_refresh_interval: int = 4,
    ) -> "DualPhasePerLayerPress":
        if default_phase_ratios is None:
            default_phase_ratios = [0.0, 0.0]
        if default_phase_cold_ratios is None:
            default_phase_cold_ratios = [0.0, 0.0]

        prefill_press = BlockWisePress(compression_ratio=default_phase_ratios[0], block_size=block_size)
        decode_press = BlockWisePress(compression_ratio=default_phase_ratios[1], block_size=block_size)

        return cls(
            prefill_press=prefill_press,
            decode_press=decode_press,
            layer_phase_ratios=layer_phase_ratios,
            default_phase_ratios=default_phase_ratios,
            layer_phase_cold_ratios=dict(layer_phase_cold_ratios or {}),
            default_phase_cold_ratios=default_phase_cold_ratios,
            prefill_layer_presses=dict(prefill_layer_presses or {}),
            decode_layer_presses=dict(decode_layer_presses or {}),
            block_size=block_size,
            compression_interval=compression_interval,
            score_refresh_interval=score_refresh_interval,
            decode_hidden_states_buffer_size=decode_hidden_states_buffer_size,
        )

    def post_init_from_model(self, model):
        seen = set()
        all_presses = [self.prefill_press, self.decode_press]
        all_presses.extend(self.prefill_layer_presses.values())
        all_presses.extend(self.decode_layer_presses.values())

        for press in all_presses:
            if id(press) in seen:
                continue
            press.post_init_from_model(model)
            seen.add(id(press))

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor | None,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        phase = self._resolve_phase(hidden_states, kwargs)
        layer_idx = self._resolve_layer_idx(module)
        active_press = self._resolve_active_press(layer_idx, phase)
        ratio = self._resolve_ratio(layer_idx, phase)
        cold_ratio = self._resolve_cold_ratio(layer_idx, phase)

        force_refresh_summary = bool(kwargs.get("_force_refresh_summary", False))

        if phase == "prefill":
            original_ratio = active_press.compression_ratio
            active_press.compression_ratio = ratio
            try:
                keys, values = active_press.compress(module, hidden_states, keys, values, attentions, kwargs)
                self._record_block_states(layer_idx, 0, active_press.last_block_heat.get(layer_idx), torch.empty(1, 0, dtype=torch.long, device=keys.device), torch.empty(1, 0, dtype=torch.long, device=keys.device))
                return keys, values
            finally:
                active_press.compression_ratio = original_ratio

        plan = active_press.build_block_plan(
            module,
            hidden_states,
            keys,
            values,
            attentions,
            kwargs,
            compression_ratio=ratio,
            force_refresh_summary=force_refresh_summary,
        )
        original_num_blocks = plan["num_blocks"]
        deleted_block_indices = self._complement_block_indices(original_num_blocks, plan["kept_block_indices"], keys.device)
        keys, values = active_press.gather_by_token_indices(keys, values, plan["token_indices"])

        retained_plan = active_press.build_block_plan(
            module,
            hidden_states,
            keys,
            values,
            attentions,
            kwargs,
            compression_ratio=cold_ratio,
            force_refresh_summary=force_refresh_summary,
        )

        active_block_indices = retained_plan["kept_block_indices"]
        cached_mask = self._build_mask_from_active_blocks(keys, active_press.block_size, active_block_indices)
        self.layer_cached_masks[layer_idx] = cached_mask
        module.masked_key_indices = cached_mask

        heat = retained_plan["block_scores"]
        self._record_block_states(layer_idx, retained_plan["num_blocks"], heat, active_block_indices, deleted_block_indices)
        active_press.build_or_refresh_block_summary(module, keys, values, force_refresh=True)
        return keys, values

    def forward_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: list):
        hidden_states = kwargs["hidden_states"]
        cache = kwargs["past_key_values"]
        layer_idx = self._resolve_layer_idx(module)
        phase = self._resolve_phase(hidden_states, kwargs)

        if phase == "prefill":
            module.masked_key_indices = None
            keys, values = extract_keys_and_values(cache, layer_idx)
            attentions = output[1] if len(output) > 1 and output[1] is not None else None
            keys, values = self.compress(module, hidden_states, keys, values, attentions, kwargs)
            self._write_back_cache(cache, layer_idx, keys, values)
            return output

        self.decode_hidden_states_buffer[layer_idx].append(hidden_states.detach().clone())
        self.decode_hidden_states_buffer[layer_idx] = self.decode_hidden_states_buffer[layer_idx][
            -self.decode_hidden_states_buffer_size :
        ]

        self.layer_decode_steps[layer_idx] += 1
        self.layer_score_steps[layer_idx] += 1
        self.layer_decode_generated_tokens[layer_idx] += hidden_states.shape[1]

        new_block_formed = self.layer_decode_generated_tokens[layer_idx] % self.block_size == 0

        refresh_scores = (
            new_block_formed
            or self.layer_score_steps[layer_idx] >= self.score_refresh_interval
            or not self.layer_block_states[layer_idx]
        )
        refresh_compression = (
            new_block_formed
            or self.layer_decode_steps[layer_idx] >= self.compression_interval
            or not self.layer_block_states[layer_idx]
        )

        if not refresh_scores:
            module.masked_key_indices = self.layer_cached_masks[layer_idx]
            return output

        self.layer_score_steps[layer_idx] = 0
        if refresh_compression:
            self.layer_decode_steps[layer_idx] = 0

        buffered_hidden_states = torch.cat(self.decode_hidden_states_buffer[layer_idx], dim=1)
        keys, values = extract_keys_and_values(cache, layer_idx)
        attentions = output[1] if len(output) > 1 and output[1] is not None else None

        if refresh_compression:
            compression_kwargs = dict(kwargs)
            compression_kwargs["_force_refresh_summary"] = new_block_formed
            keys, values = self.compress(module, buffered_hidden_states, keys, values, attentions, compression_kwargs)
            self._write_back_cache(cache, layer_idx, keys, values)
        else:
            active_press = self._resolve_active_press(layer_idx, "decode")
            retained_plan = active_press.build_block_plan(
                module,
                buffered_hidden_states,
                keys,
                values,
                attentions,
                kwargs,
                compression_ratio=self._resolve_cold_ratio(layer_idx, "decode"),
                force_refresh_summary=new_block_formed,
            )
            active_block_indices = retained_plan["kept_block_indices"]
            self.layer_cached_masks[layer_idx] = self._build_mask_from_active_blocks(
                keys,
                active_press.block_size,
                active_block_indices,
            )
            module.masked_key_indices = self.layer_cached_masks[layer_idx]
            self._record_block_states(
                layer_idx,
                retained_plan["num_blocks"],
                retained_plan["block_scores"],
                active_block_indices,
                torch.empty(keys.shape[0], 0, dtype=torch.long, device=keys.device),
            )

        return output

    def reset(self):
        self.layer_decode_steps = defaultdict(int)
        self.layer_score_steps = defaultdict(int)
        self.layer_decode_generated_tokens = defaultdict(int)
        self.decode_hidden_states_buffer = defaultdict(list)
        self.layer_block_states = defaultdict(dict)
        self.layer_cached_masks = defaultdict(lambda: None)
        self.layer_heat_ema = defaultdict(lambda: None)

    def _record_block_states(
        self,
        layer_idx: int,
        num_blocks: int,
        heat: torch.Tensor | None,
        active_block_indices: torch.Tensor,
        deleted_block_indices: torch.Tensor,
    ):
        if num_blocks == 0:
            self.layer_block_states[layer_idx] = {
                "active": active_block_indices,
                "resident_gpu": active_block_indices,
                "permanently_deleted": deleted_block_indices,
                "offloaded_cpu": active_block_indices,
                "prefetch_to_gpu": active_block_indices,
            }
            return

        device = active_block_indices.device if active_block_indices.numel() > 0 else deleted_block_indices.device
        all_block_indices = torch.arange(num_blocks, device=device).expand(active_block_indices.shape[0], -1)
        inactive_block_indices = self._difference_block_indices(all_block_indices, active_block_indices)

        if heat is None or heat.shape[-1] != num_blocks:
            if layer_idx in self.layer_heat_ema and self.layer_heat_ema[layer_idx] is not None:
                heat = self.layer_heat_ema[layer_idx]
            else:
                heat = torch.zeros(active_block_indices.shape[0], num_blocks, device=device)

        previous = self.layer_heat_ema[layer_idx]
        if previous is None or previous.shape != heat.shape:
            heat_ema = heat.detach()
        else:
            heat_ema = self.history_momentum * previous + (1.0 - self.history_momentum) * heat.detach()
        self.layer_heat_ema[layer_idx] = heat_ema

        resident_gpu_indices, offloaded_cpu_indices, prefetch_to_gpu_indices = self._split_inactive_blocks(
            inactive_block_indices,
            heat_ema,
        )

        self.layer_block_states[layer_idx] = {
            "active": active_block_indices.detach().clone(),
            "resident_gpu": resident_gpu_indices,
            "permanently_deleted": deleted_block_indices.detach().clone(),
            "offloaded_cpu": offloaded_cpu_indices,
            "prefetch_to_gpu": prefetch_to_gpu_indices,
            "heat": heat.detach().clone(),
            "heat_ema": heat_ema.detach().clone(),
        }

    def _split_inactive_blocks(
        self,
        inactive_block_indices: torch.Tensor,
        heat_ema: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        resident_lists = []
        offload_lists = []
        prefetch_lists = []

        for batch_idx in range(inactive_block_indices.shape[0]):
            batch_inactive = inactive_block_indices[batch_idx]
            if batch_inactive.numel() == 0:
                resident_lists.append(batch_inactive)
                offload_lists.append(batch_inactive)
                prefetch_lists.append(batch_inactive)
                continue

            batch_heat = heat_ema[batch_idx, batch_inactive]
            sorted_order = batch_heat.argsort(descending=True)
            sorted_blocks = batch_inactive[sorted_order]

            resident_count = int(math.ceil(sorted_blocks.numel() * self.resident_gpu_ratio))
            resident_blocks = sorted_blocks[:resident_count]
            remaining_blocks = sorted_blocks[resident_count:]

            prefetch_count = int(math.ceil(remaining_blocks.numel() * self.prefetch_ratio)) if remaining_blocks.numel() > 0 else 0
            prefetch_blocks = remaining_blocks[:prefetch_count]
            offloaded_blocks = remaining_blocks[prefetch_count:]

            resident_lists.append(resident_blocks.sort().values)
            offload_lists.append(offloaded_blocks.sort().values)
            prefetch_lists.append(prefetch_blocks.sort().values)

        return (
            self._stack_or_empty(resident_lists, inactive_block_indices.device),
            self._stack_or_empty(offload_lists, inactive_block_indices.device),
            self._stack_or_empty(prefetch_lists, inactive_block_indices.device),
        )

    def _build_mask_from_active_blocks(
        self,
        keys: torch.Tensor,
        block_size: int,
        active_block_indices: torch.Tensor,
    ):
        num_blocks = math.ceil(keys.shape[2] / block_size)
        active_mask = torch.zeros(keys.shape[0], num_blocks, dtype=torch.bool, device=keys.device)
        if active_block_indices.numel() > 0:
            active_mask.scatter_(1, active_block_indices, True)

        batch_indices = []
        head_indices = []
        seq_indices = []

        for batch_idx in range(keys.shape[0]):
            for block_idx in range(num_blocks):
                if active_mask[batch_idx, block_idx]:
                    continue
                start = block_idx * block_size
                end = min(start + block_size, keys.shape[2])
                for token_idx in range(start, end):
                    for head_idx in range(keys.shape[1]):
                        batch_indices.append(batch_idx)
                        head_indices.append(head_idx)
                        seq_indices.append(token_idx)

        if not batch_indices:
            return None

        return (
            torch.tensor(batch_indices, dtype=torch.long, device=keys.device),
            torch.tensor(head_indices, dtype=torch.long, device=keys.device),
            torch.tensor(seq_indices, dtype=torch.long, device=keys.device),
        )

    def _difference_block_indices(self, universe: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
        diff_lists = []
        for batch_idx in range(universe.shape[0]):
            if selected.shape[1] == 0:
                diff_lists.append(universe[batch_idx])
                continue
            mask = torch.ones(universe.shape[1], dtype=torch.bool, device=universe.device)
            mask[selected[batch_idx]] = False
            diff_lists.append(universe[batch_idx][mask])
        return self._stack_or_empty(diff_lists, universe.device)

    def _complement_block_indices(self, num_blocks: int, kept: torch.Tensor, device: torch.device) -> torch.Tensor:
        if num_blocks == 0:
            return torch.empty(kept.shape[0], 0, dtype=torch.long, device=device)
        universe = torch.arange(num_blocks, device=device).expand(kept.shape[0], -1)
        return self._difference_block_indices(universe, kept)

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
            else:
                pad = torch.full((max_len - tensor.numel(),), -1, dtype=torch.long, device=device)
                padded.append(torch.cat([tensor, pad], dim=0))
        return torch.stack(padded, dim=0)

    def _write_back_cache(self, cache, layer_idx: int, keys: torch.Tensor, values: torch.Tensor):
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

    def _resolve_phase(self, hidden_states: torch.Tensor, kwargs: dict) -> PhaseName:
        q_len = hidden_states.shape[1]
        return "prefill" if kwargs["cache_position"][-1] <= q_len else "decode"

    def _resolve_active_press(self, layer_idx: int, phase: PhaseName) -> BlockWisePress:
        if phase == "prefill":
            return self.prefill_layer_presses.get(layer_idx, self.prefill_press)
        return self.decode_layer_presses.get(layer_idx, self.decode_press)

    def _resolve_layer_idx(self, module: nn.Module) -> int:
        raw = getattr(module, "layer_idx")
        if isinstance(raw, torch.Tensor):
            return int(raw.item())
        return int(raw)

    def _resolve_ratio(self, layer_idx: int, phase: PhaseName) -> float:
        ratios = self.layer_phase_ratios.get(layer_idx, self.default_phase_ratios)
        assert len(ratios) == 2, f"layer_phase_ratios[{layer_idx}] must be [prefill_ratio, decode_ratio]"
        ratio = ratios[0] if phase == "prefill" else ratios[1]
        assert 0 <= ratio < 1, f"compression ratio for layer {layer_idx} in phase {phase} must be in [0, 1)"
        return ratio

    def _resolve_cold_ratio(self, layer_idx: int, phase: PhaseName) -> float:
        cold_ratios = self.layer_phase_cold_ratios.get(layer_idx, self.default_phase_cold_ratios)
        assert len(cold_ratios) == 2, f"layer_phase_cold_ratios[{layer_idx}] must be [prefill_cold, decode_cold]"
        ratio = cold_ratios[0] if phase == "prefill" else cold_ratios[1]
        assert 0 <= ratio < 1, f"cold ratio for layer {layer_idx} in phase {phase} must be in [0, 1)"
        return ratio
