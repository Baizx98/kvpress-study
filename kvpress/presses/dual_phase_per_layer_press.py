# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from dataclasses import dataclass, field
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
    Layer-aware / phase-aware block-wise compression with optional temporary cold blocks.

    Permanent pruning physically removes low-importance blocks from the KV cache.
    Temporary cold blocks stay in the cache but are masked out for the current decode
    iteration, so they can be reactivated later if their block heat increases again.
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
    decode_hidden_states_buffer_size: int = 32

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
        assert len(self.default_phase_ratios) == 2, "default_phase_ratios must be [prefill_ratio, decode_ratio]"
        assert len(self.default_phase_cold_ratios) == 2, "default_phase_cold_ratios must be [prefill_cold, decode_cold]"
        assert self.decode_hidden_states_buffer_size > 0, "decode_hidden_states_buffer_size must be > 0"
        assert isinstance(self.prefill_press, BlockWisePress), "prefill_press must be BlockWisePress"
        assert isinstance(self.decode_press, BlockWisePress), "decode_press must be BlockWisePress"

        self.layer_decode_steps = defaultdict(int)
        self.decode_hidden_states_buffer = defaultdict(list)

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
    ) -> "DualPhasePerLayerPress":
        if default_phase_ratios is None:
            default_phase_ratios = [0.0, 0.0]
        if default_phase_cold_ratios is None:
            default_phase_cold_ratios = [0.0, 0.0]

        assert len(default_phase_ratios) == 2, "default_phase_ratios must be [prefill_ratio, decode_ratio]"
        assert len(default_phase_cold_ratios) == 2, "default_phase_cold_ratios must be [prefill_cold, decode_cold]"
        assert block_size > 0, "block_size must be > 0"

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

        module.masked_key_indices = None

        original_ratio = active_press.compression_ratio
        active_press.compression_ratio = ratio

        try:
            if phase == "prefill":
                return active_press.compress(module, hidden_states, keys, values, attentions, kwargs)  # type: ignore[arg-type]

            plan = active_press.build_block_plan(
                module,
                hidden_states,
                keys,
                values,
                attentions,
                kwargs,
                compression_ratio=ratio,
            )
            keys, values = active_press.gather_by_token_indices(keys, values, plan["token_indices"])

            if cold_ratio > 0 and keys.shape[2] > 0:
                self._apply_temporary_cold_mask(module, active_press, hidden_states, keys, values, kwargs, cold_ratio)

            return keys, values
        finally:
            active_press.compression_ratio = original_ratio

    def forward_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: list):
        hidden_states = kwargs["hidden_states"]
        cache = kwargs["past_key_values"]
        layer_idx = self._resolve_layer_idx(module)
        phase = self._resolve_phase(hidden_states, kwargs)

        module.masked_key_indices = None

        if phase == "decode":
            self.decode_hidden_states_buffer[layer_idx].append(hidden_states.detach().clone())
            self.decode_hidden_states_buffer[layer_idx] = self.decode_hidden_states_buffer[layer_idx][
                -self.decode_hidden_states_buffer_size :
            ]

            self.layer_decode_steps[layer_idx] += 1
            if self.layer_decode_steps[layer_idx] < self.compression_interval:
                return output
            self.layer_decode_steps[layer_idx] = 0

            hidden_states = torch.cat(self.decode_hidden_states_buffer[layer_idx], dim=1)

        keys, values = extract_keys_and_values(cache, layer_idx)
        attentions = output[1] if len(output) > 1 and output[1] is not None else None
        keys, values = self.compress(module, hidden_states, keys, values, attentions, kwargs)

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

        return output

    def reset(self):
        self.layer_decode_steps = defaultdict(int)
        self.decode_hidden_states_buffer = defaultdict(list)

    def _apply_temporary_cold_mask(
        self,
        module: nn.Module,
        active_press: BlockWisePress,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        kwargs: dict,
        cold_ratio: float,
    ):
        cold_plan = active_press.build_block_plan(
            module,
            hidden_states,
            keys,
            values,
            attentions=None,
            kwargs=kwargs,
            compression_ratio=cold_ratio,
        )

        num_blocks = cold_plan["num_blocks"]
        n_active_blocks = cold_plan["n_kept_blocks"]
        if n_active_blocks >= num_blocks:
            return

        block_mask = torch.zeros(keys.shape[0], num_blocks, dtype=torch.bool, device=keys.device)
        if n_active_blocks > 0:
            block_mask.scatter_(1, cold_plan["kept_block_indices"], True)

        cold_positions = []
        for batch_idx in range(keys.shape[0]):
            for block_idx in range(num_blocks):
                if block_mask[batch_idx, block_idx]:
                    continue
                start = block_idx * active_press.block_size
                end = min(start + active_press.block_size, keys.shape[2])
                cold_positions.extend((batch_idx, token_idx) for token_idx in range(start, end))

        if not cold_positions:
            return

        batch_indices = []
        head_indices = []
        seq_indices = []
        num_heads = keys.shape[1]
        for batch_idx, token_idx in cold_positions:
            for head_idx in range(num_heads):
                batch_indices.append(batch_idx)
                head_indices.append(head_idx)
                seq_indices.append(token_idx)

        module.masked_key_indices = (
            torch.tensor(batch_indices, dtype=torch.long, device=keys.device),
            torch.tensor(head_indices, dtype=torch.long, device=keys.device),
            torch.tensor(seq_indices, dtype=torch.long, device=keys.device),
        )

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
