# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
import inspect
from typing import Optional

import torch
from torch import nn
from transformers import PreTrainedModel

from kvpress.presses.base_press import BasePress


@dataclass
class PrefillPerLayerRatioPress(BasePress):
    """
    Test-only wrapper that applies layer-specific compression ratios in prefill.

    The wrapped press is used unchanged, except that its `compression_ratio` is
    temporarily overridden on a per-layer basis during prefill. Decode is kept
    compression-free by default to isolate the effect of prefill-only layering.
    """

    press: BasePress
    base_compression_ratio: float = 0.0
    skip_first_layers: int = 0
    decode_compression_ratio: float = 0.0
    compression_ratios: Optional[list[float]] = None
    effective_compression_ratios: list[float] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self):
        assert hasattr(type(self.press), "forward_hook") or hasattr(self.press, "forward_hook"), (
            "Wrapped press must implement forward_hook"
        )
        assert hasattr(type(self.press), "compression_ratio") or hasattr(self.press, "compression_ratio"), (
            "Wrapped press must expose compression_ratio"
        )
        assert 0 <= self.base_compression_ratio < 1, "base_compression_ratio must be in [0, 1)"
        assert self.skip_first_layers >= 0, "skip_first_layers must be >= 0"
        assert 0 <= self.decode_compression_ratio < 1, "decode_compression_ratio must be in [0, 1)"
        if self.compression_ratios is not None:
            assert all(0 <= ratio < 1 for ratio in self.compression_ratios), (
                "All compression_ratios must be in [0, 1)"
            )

    @property
    def compression_ratio(self) -> float:
        return self.base_compression_ratio

    @compression_ratio.setter
    def compression_ratio(self, value: float):
        ratio = float(value)
        assert 0 <= ratio < 1, "compression_ratio must be in [0, 1)"
        self.base_compression_ratio = ratio

    def _resolve_num_layers(self, model: PreTrainedModel) -> int:
        language_model = model.model.language_model if hasattr(model.model, "language_model") else model.model
        return len(language_model.layers)

    def _build_effective_ratios(self, num_layers: int) -> list[float]:
        if self.compression_ratios is not None:
            assert len(self.compression_ratios) == num_layers, (
                f"compression_ratios length ({len(self.compression_ratios)}) must match num_layers ({num_layers})"
            )
            return [float(ratio) for ratio in self.compression_ratios]
        return [
            0.0 if layer_idx < self.skip_first_layers else float(self.base_compression_ratio)
            for layer_idx in range(num_layers)
        ]

    def post_init_from_model(self, model: PreTrainedModel):
        self.press.post_init_from_model(model)
        self.effective_compression_ratios = self._build_effective_ratios(self._resolve_num_layers(model))

    def _is_decode(self, kwargs: dict) -> bool:
        hidden_states = kwargs["hidden_states"]
        q_len = hidden_states.shape[1]
        return bool(kwargs["cache_position"][-1] > q_len)

    def _resolve_layer_ratio(self, module: nn.Module) -> float:
        layer_idx = int(module.layer_idx)
        if not self.effective_compression_ratios:
            return self.base_compression_ratio
        return float(self.effective_compression_ratios[layer_idx])

    def forward_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: list):
        original_compression_ratio = self.press.compression_ratio  # type: ignore[attr-defined]
        try:
            self.press.compression_ratio = (
                self.decode_compression_ratio if self._is_decode(kwargs) else self._resolve_layer_ratio(module)
            )
            return self.press.forward_hook(module, input, kwargs, output)
        finally:
            self.press.compression_ratio = original_compression_ratio  # type: ignore[attr-defined]

