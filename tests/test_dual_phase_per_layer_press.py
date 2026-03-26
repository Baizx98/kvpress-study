# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import torch
from torch import nn

from kvpress.presses.base_press import BasePress
from kvpress.presses.block_score_press import BlockScorePress
from kvpress.presses.block_wise_press import BlockWisePress
from kvpress.presses.dual_phase_per_layer_press import DualPhasePerLayerPress


class DummyModule(nn.Module):
    def __init__(self, layer_idx: int, head_dim: int = 4):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = head_dim


def make_kv(batch: int = 1, heads: int = 2, seq_len: int = 8, head_dim: int = 4):
    keys = torch.arange(batch * heads * seq_len * head_dim, dtype=torch.float32).view(batch, heads, seq_len, head_dim)
    values = keys + 1000
    return keys, values


def build_press(layer_phase_ratios=None):
    prefill_press = BlockWisePress(press=BlockScorePress(compression_ratio=0.0, block_size=2))
    decode_press = BlockWisePress(press=BlockScorePress(compression_ratio=0.0, block_size=2))
    return DualPhasePerLayerPress(
        prefill_press=prefill_press,
        decode_press=decode_press,
        layer_phase_ratios=layer_phase_ratios or {},
        default_phase_ratios=[0.0, 0.0],
    )


def test_compress_prefill_decode_split_with_per_layer_ratio():
    press = build_press(
        layer_phase_ratios={
            0: [0.5, 0.25],
            1: [0.0, 0.5],
        }
    )

    keys, values = make_kv(seq_len=8)
    hidden_states_prefill = torch.zeros((1, 8, 16), dtype=torch.float32)
    hidden_states_decode = torch.zeros((1, 1, 16), dtype=torch.float32)

    kwargs_prefill = {"cache_position": torch.tensor([8])}
    kwargs_decode = {"cache_position": torch.tensor([10])}

    layer0 = DummyModule(layer_idx=0)
    layer1 = DummyModule(layer_idx=1)

    # layer 0 prefill: 0.5 -> keep 4
    k0p, _ = press.compress(layer0, hidden_states_prefill, keys, values, None, kwargs_prefill)
    assert k0p.shape[2] == 4

    # layer 0 decode: 0.25 -> keep 6
    k0d, _ = press.compress(layer0, hidden_states_decode, keys, values, None, kwargs_decode)
    assert k0d.shape[2] == 6

    # layer 1 prefill: 0.0 -> keep 8
    k1p, _ = press.compress(layer1, hidden_states_prefill, keys, values, None, kwargs_prefill)
    assert k1p.shape[2] == 8

    # layer 1 decode: 0.5 -> keep 4
    k1d, _ = press.compress(layer1, hidden_states_decode, keys, values, None, kwargs_decode)
    assert k1d.shape[2] == 4


def test_phase_can_share_same_press_instance():
    shared = BlockWisePress(press=BlockScorePress(compression_ratio=0.0, block_size=2))
    press = DualPhasePerLayerPress(
        prefill_press=shared,
        decode_press=shared,
        layer_phase_ratios={0: [0.5, 0.25]},
    )

    keys, values = make_kv(seq_len=8)
    layer0 = DummyModule(layer_idx=0)

    k_prefill, _ = press.compress(
        layer0,
        torch.zeros((1, 8, 16), dtype=torch.float32),
        keys,
        values,
        None,
        {"cache_position": torch.tensor([8])},
    )
    k_decode, _ = press.compress(
        layer0,
        torch.zeros((1, 1, 16), dtype=torch.float32),
        keys,
        values,
        None,
        {"cache_position": torch.tensor([10])},
    )

    assert k_prefill.shape[2] == 4
    assert k_decode.shape[2] == 6


def test_forward_hook_updates_ratio_dynamically():
    press = build_press(layer_phase_ratios={0: [0.5, 0.25]})
    layer0 = DummyModule(layer_idx=0)

    # 用最小 fake cache 模拟 BasePress 约定的 cache 结构
    class FakeCacheLayer:
        def __init__(self, keys, values):
            self.keys = keys
            self.values = values

    class FakeCache:
        def __init__(self, keys, values):
            self.layers = [FakeCacheLayer(keys, values)]

    keys, values = make_kv(seq_len=8)
    cache = FakeCache(keys, values)

    kwargs = {
        "hidden_states": torch.zeros((1, 8, 16), dtype=torch.float32),
        "past_key_values": cache,
        "cache_position": torch.tensor([8]),
    }
    output = [None, None]

    press.forward_hook(layer0, [], kwargs, output)

    # prefill ratio=0.5 -> 8 变 4
    assert cache.layers[0].keys.shape[2] == 4
    assert cache.layers[0].values.shape[2] == 4
