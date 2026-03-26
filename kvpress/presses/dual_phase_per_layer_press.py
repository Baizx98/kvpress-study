# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Literal, Optional

import torch
import torch.nn as nn
from transformers import QuantizedCache

from kvpress.presses.base_press import BasePress
from kvpress.presses.block_score_press import BlockScorePress
from kvpress.presses.block_wise_press import BlockWisePress
from kvpress.utils import extract_keys_and_values

PhaseName = Literal["prefill", "decode"]


@dataclass
class DualPhasePerLayerPress(BasePress):
    """
    最小化实现：
    1) 区分 prefill / decode 两个阶段。
    2) 区分每个 attention layer 的压缩率。
    3) 两个阶段都使用 BlockWisePress。

    配置说明
    -------
    layer_phase_ratios: dict[int, list[float]]
        形如 {layer_id: [prefill_ratio, decode_ratio]}。
        - prefill_ratio: prefill 阶段该层压缩率
        - decode_ratio: decode 阶段该层压缩率

    prefill_press / decode_press:
        阶段级默认 press；如果你希望某层使用独立 press，可通过
        prefill_layer_presses / decode_layer_presses 覆盖。
    """

    prefill_press: BlockWisePress
    decode_press: BlockWisePress

    layer_phase_ratios: dict[int, list[float]] = field(default_factory=dict)
    default_phase_ratios: list[float] = field(default_factory=lambda: [0.0, 0.0])

    # 可选：每层覆盖阶段默认 press；不提供时沿用阶段默认 press
    prefill_layer_presses: dict[int, BlockWisePress] = field(default_factory=dict)
    decode_layer_presses: dict[int, BlockWisePress] = field(default_factory=dict)

    # prefill和decode阶段的press的压缩粒度，一旦确定就不应该改变，否则会给内存管理带来麻烦
    block_size: int = 16

    # decode 每层可配置压缩间隔（最小实现默认每步压缩）
    compression_interval: int = 1

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
        self.layer_decode_steps = defaultdict(int)

        # 基础类型约束，避免把 score press 直接传进来
        assert isinstance(self.prefill_press, BlockWisePress), "prefill_press must be BlockWisePress"
        assert isinstance(self.decode_press, BlockWisePress), "decode_press must be BlockWisePress"

    @classmethod
    def init_class_vars(
        cls,
        layer_phase_ratios: dict[int, list[float]],
        block_size: int = 16,
        default_phase_ratios: Optional[list[float]] = None,
        compression_interval: int = 1,
        prefill_layer_presses: Optional[dict[int, BlockWisePress]] = None,
        decode_layer_presses: Optional[dict[int, BlockWisePress]] = None,
    ) -> "DualPhasePerLayerPress":
        """
        类函数：集中初始化构造所需变量。

        适用场景：
        - 你只想提供每层 ratio 配置，让类自动创建默认的 block-wise press。
        - 你想统一设置 block_size，避免外部重复构造 press 对象。

        注意：
        - 不会根据 layer_phase_ratios 自动创建每层独立 press。
        - 每层 ratio 会在 compress 中动态注入当前激活的 press。
        """
        if default_phase_ratios is None:
            default_phase_ratios = [0.0, 0.0]

        assert len(default_phase_ratios) == 2, "default_phase_ratios must be [prefill_ratio, decode_ratio]"
        assert block_size > 0, "block_size must be > 0"

        prefill_press = BlockWisePress(
            press=BlockScorePress(compression_ratio=default_phase_ratios[0], block_size=block_size)
        )
        decode_press = BlockWisePress(
            press=BlockScorePress(compression_ratio=default_phase_ratios[1], block_size=block_size)
        )

        prefill_layer_presses = dict(prefill_layer_presses or {})
        decode_layer_presses = dict(decode_layer_presses or {})

        return cls(
            prefill_press=prefill_press,
            decode_press=decode_press,
            layer_phase_ratios=layer_phase_ratios,
            default_phase_ratios=default_phase_ratios,
            prefill_layer_presses=prefill_layer_presses,
            decode_layer_presses=decode_layer_presses,
            block_size=block_size,
            compression_interval=compression_interval,
        )

    def post_init_from_model(self, model):
        """初始化所有会被用到的 press，支持共享实例去重。"""
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
        """
        参考 PrefillDecodingPress 的分流思路：
        - 先判定当前是 prefill 还是 decode
        - 再按 layer 选择压缩率
        - 最后调用对应阶段的 BlockWisePress.compress
        """
        phase = self._resolve_phase(hidden_states, kwargs)
        layer_idx = self._resolve_layer_idx(module)

        active_press = self._resolve_active_press(layer_idx, phase)
        ratio = self._resolve_ratio(layer_idx, phase)

        # 动态更新当前层当前阶段的压缩率
        original_ratio = active_press.compression_ratio
        active_press.compression_ratio = ratio

        try:
            return active_press.compress(module, hidden_states, keys, values, attentions, kwargs)  # type: ignore[arg-type]
        finally:
            active_press.compression_ratio = original_ratio

    def forward_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: list):
        """
        在 hook 中动态更新参数并执行压缩。

        说明：
        - 不走 BasePress 的 prefill-only 默认逻辑，直接在此处理 prefill+decode。
        - 返回值仍保持 BasePress 约定：只更新 cache，output 原样返回。
        """
        hidden_states = kwargs["hidden_states"]
        cache = kwargs["past_key_values"]
        layer_idx = self._resolve_layer_idx(module)
        phase = self._resolve_phase(hidden_states, kwargs)

        if phase == "decode":
            self.layer_decode_steps[layer_idx] += 1
            if self.layer_decode_steps[layer_idx] < self.compression_interval:
                return output
            self.layer_decode_steps[layer_idx] = 0

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

    def _resolve_phase(self, hidden_states: torch.Tensor, kwargs: dict) -> PhaseName:
        q_len = hidden_states.shape[1]
        return "prefill" if kwargs["cache_position"][-1] <= q_len else "decode"

    def _resolve_active_press(self, layer_idx: int, phase: PhaseName) -> BlockWisePress:
        if phase == "prefill":
            return self.prefill_layer_presses.get(layer_idx, self.prefill_press)
        return self.decode_layer_presses.get(layer_idx, self.decode_press)

    def _resolve_layer_idx(self, module: nn.Module) -> int:
        """兼容不同模块实现中的 layer_idx 表达。"""
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
