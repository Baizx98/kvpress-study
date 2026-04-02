# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch import nn

from kvpress.presses.block_wise_press import BlockWisePress
from kvpress.presses.dual_phase_per_layer_press import DualPhasePerLayerPress


class DummyConfig:
    def __init__(self, num_attention_heads: int, num_key_value_heads: int):
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads


class DummyModule(nn.Module):
    def __init__(self, layer_idx: int, hidden_dim: int = 16, num_heads: int = 4, num_kv_heads: int = 2):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = hidden_dim // num_heads
        self.config = DummyConfig(num_attention_heads=num_heads, num_key_value_heads=num_kv_heads)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        with torch.no_grad():
            self.q_proj.weight.copy_(torch.eye(hidden_dim))


def make_kv(batch: int = 1, heads: int = 2, seq_len: int = 8, head_dim: int = 4):
    keys = torch.arange(batch * heads * seq_len * head_dim, dtype=torch.float32).view(batch, heads, seq_len, head_dim)
    values = keys + 1000
    return keys, values


def make_hidden_states(seq_len: int, hidden_dim: int = 16):
    return torch.arange(seq_len * hidden_dim, dtype=torch.float32).view(1, seq_len, hidden_dim)


def build_press(layer_phase_ratios=None, layer_phase_cold_ratios=None):
    prefill_press = BlockWisePress(compression_ratio=0.0, block_size=2, min_q_window=1, max_q_window=4)
    decode_press = BlockWisePress(compression_ratio=0.0, block_size=2, min_q_window=1, max_q_window=4)
    return DualPhasePerLayerPress(
        prefill_press=prefill_press,
        decode_press=decode_press,
        layer_phase_ratios=layer_phase_ratios or {},
        layer_phase_cold_ratios=layer_phase_cold_ratios or {},
        default_phase_ratios=[0.0, 0.0],
        default_phase_cold_ratios=[0.0, 0.0],
        decode_hidden_states_buffer_size=8,
    )


def test_compress_prefill_decode_split_with_per_layer_ratio():
    press = build_press(
        layer_phase_ratios={
            0: [0.5, 0.25],
            1: [0.0, 0.5],
        }
    )

    keys, values = make_kv(seq_len=8)
    hidden_states_prefill = make_hidden_states(8)
    hidden_states_decode = make_hidden_states(1)

    kwargs_prefill = {"cache_position": torch.tensor([8])}
    kwargs_decode = {"cache_position": torch.tensor([10])}

    layer0 = DummyModule(layer_idx=0)
    layer1 = DummyModule(layer_idx=1)

    k0p, _ = press.compress(layer0, hidden_states_prefill, keys, values, None, kwargs_prefill)
    assert k0p.shape[2] == 4

    k0d, _ = press.compress(layer0, hidden_states_decode, keys, values, None, kwargs_decode)
    assert k0d.shape[2] == 6

    k1p, _ = press.compress(layer1, hidden_states_prefill, keys, values, None, kwargs_prefill)
    assert k1p.shape[2] == 8

    k1d, _ = press.compress(layer1, hidden_states_decode, keys, values, None, kwargs_decode)
    assert k1d.shape[2] == 4


def test_phase_can_share_same_press_instance():
    shared = BlockWisePress(compression_ratio=0.0, block_size=2, min_q_window=1, max_q_window=4)
    press = DualPhasePerLayerPress(
        prefill_press=shared,
        decode_press=shared,
        layer_phase_ratios={0: [0.5, 0.25]},
        decode_hidden_states_buffer_size=8,
    )

    keys, values = make_kv(seq_len=8)
    layer0 = DummyModule(layer_idx=0)

    k_prefill, _ = press.compress(
        layer0,
        make_hidden_states(8),
        keys,
        values,
        None,
        {"cache_position": torch.tensor([8])},
    )
    k_decode, _ = press.compress(
        layer0,
        make_hidden_states(1),
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
        "hidden_states": make_hidden_states(8),
        "past_key_values": cache,
        "cache_position": torch.tensor([8]),
    }
    output = [None, None]

    press.forward_hook(layer0, [], kwargs, output)

    assert cache.layers[0].keys.shape[2] == 4
    assert cache.layers[0].values.shape[2] == 4


def test_decode_cold_blocks_are_masked_without_physical_deletion():
    press = build_press(
        layer_phase_ratios={0: [0.0, 0.0]},
        layer_phase_cold_ratios={0: [0.0, 0.5]},
    )
    layer0 = DummyModule(layer_idx=0)

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
        "hidden_states": make_hidden_states(1),
        "past_key_values": cache,
        "cache_position": torch.tensor([10]),
    }
    output = [None, None]

    press.forward_hook(layer0, [], kwargs, output)

    assert cache.layers[0].keys.shape[2] == 8
    assert layer0.masked_key_indices is not None
    batch_indices, head_indices, seq_indices = layer0.masked_key_indices
    assert len(batch_indices) == len(head_indices) == len(seq_indices)
    assert len(seq_indices) > 0


def test_decode_can_mix_permanent_delete_and_temporary_cold_blocks():
    press = build_press(
        layer_phase_ratios={0: [0.0, 0.5]},
        layer_phase_cold_ratios={0: [0.0, 0.5]},
    )
    layer0 = DummyModule(layer_idx=0)
    keys, values = make_kv(seq_len=8)

    compressed_keys, compressed_values = press.compress(
        layer0,
        make_hidden_states(4),
        keys,
        values,
        None,
        {"cache_position": torch.tensor([10])},
    )

    assert compressed_keys.shape[2] == 4
    assert compressed_values.shape[2] == 4
    assert layer0.masked_key_indices is not None


def test_partial_tail_block_is_kept_to_preserve_consistent_cache_length():
    press = BlockWisePress(compression_ratio=0.5, block_size=4, min_q_window=1, max_q_window=4)
    layer0 = DummyModule(layer_idx=0)
    keys, values = make_kv(seq_len=10)

    compressed_keys, compressed_values = press.compress(
        layer0,
        make_hidden_states(10),
        keys,
        values,
        None,
        {"cache_position": torch.tensor([10])},
    )

    assert compressed_keys.shape[2] == 6
    assert compressed_values.shape[2] == 6


def test_block_wise_press_builds_block_summary_cache():
    press = BlockWisePress(compression_ratio=0.5, block_size=2, min_q_window=1, max_q_window=4)
    layer0 = DummyModule(layer_idx=0)
    keys, values = make_kv(seq_len=8)

    press.compress(
        layer0,
        make_hidden_states(8),
        keys,
        values,
        None,
        {"cache_position": torch.tensor([8])},
    )

    assert layer0.layer_idx in press.last_block_summary
    summary = press.last_block_summary[layer0.layer_idx]
    assert summary["mean_keys"].shape[2] > 0
    assert summary["peak_keys"].shape == summary["mean_keys"].shape
    assert summary["topk_peak_keys"].shape[-2] >= 1


def test_block_wise_press_supports_multiple_head_scoring_methods():
    keys, values = make_kv(seq_len=8)
    layer0 = DummyModule(layer_idx=0)
    hidden_states = make_hidden_states(8)

    for method in ["max", "topk_mean", "percentile"]:
        press = BlockWisePress(
            compression_ratio=0.5,
            block_size=2,
            min_q_window=1,
            max_q_window=4,
            head_scoring_method=method,
            head_topk_ratio=0.5,
            head_percentile=0.5,
        )
        result_keys, result_values = press.compress(
            layer0,
            hidden_states,
            keys,
            values,
            None,
            {"cache_position": torch.tensor([8])},
        )
        assert result_keys.shape[2] == 4
        assert result_values.shape[2] == 4


def test_block_wise_press_head_redundancy_penalty_is_supported():
    keys, values = make_kv(seq_len=8)
    layer0 = DummyModule(layer_idx=0)
    press = BlockWisePress(
        compression_ratio=0.5,
        block_size=2,
        min_q_window=1,
        max_q_window=4,
        head_redundancy_alpha=0.5,
    )

    result_keys, result_values = press.compress(
        layer0,
        make_hidden_states(8),
        keys,
        values,
        None,
        {"cache_position": torch.tensor([8])},
    )

    assert result_keys.shape[2] == 4
    assert result_values.shape[2] == 4


def test_block_wise_press_summary_topk_scales_with_block_size():
    keys, values = make_kv(seq_len=32)
    hidden_states = make_hidden_states(32)
    layer0 = DummyModule(layer_idx=0)

    press16 = BlockWisePress(compression_ratio=0.5, block_size=16, min_q_window=1, max_q_window=8)
    press16.compress(layer0, hidden_states, keys, values, None, {"cache_position": torch.tensor([32])})
    assert press16.last_block_summary[layer0.layer_idx]["topk_peak_keys"].shape[-2] == 2

    layer1 = DummyModule(layer_idx=1)
    press32 = BlockWisePress(compression_ratio=0.5, block_size=32, min_q_window=1, max_q_window=8)
    press32.compress(layer1, hidden_states, keys, values, None, {"cache_position": torch.tensor([32])})
    assert press32.last_block_summary[layer1.layer_idx]["topk_peak_keys"].shape[-2] == 3


def test_block_wise_press_recent_blocks_expand_keep_budget():
    press = BlockWisePress(
        compression_ratio=0.5,
        block_size=2,
        min_q_window=1,
        max_q_window=4,
        protected_recent_blocks=1,
        protected_hot_blocks=0,
    )
    layer0 = DummyModule(layer_idx=0)
    keys, values = make_kv(seq_len=8)

    compressed_keys, compressed_values = press.compress(
        layer0,
        make_hidden_states(8),
        keys,
        values,
        None,
        {"cache_position": torch.tensor([8])},
    )

    assert compressed_keys.shape[2] == 4
    assert compressed_values.shape[2] == 4
    kept_tokens = compressed_keys[0, 0, :, 0].tolist()
    assert 24.0 in kept_tokens and 28.0 in kept_tokens


def test_block_wise_press_extreme_compression_can_override_recent_blocks(caplog):
    press = BlockWisePress(
        compression_ratio=0.99,
        block_size=2,
        min_q_window=1,
        max_q_window=4,
        protected_recent_blocks=3,
        protected_hot_blocks=0,
    )
    layer0 = DummyModule(layer_idx=0)
    keys, values = make_kv(seq_len=8)

    with caplog.at_level("INFO"):
        compressed_keys, compressed_values = press.compress(
            layer0,
            make_hidden_states(8),
            keys,
            values,
            None,
            {"cache_position": torch.tensor([8])},
        )

    assert compressed_keys.shape[2] == 2
    assert compressed_values.shape[2] == 2
    assert any("too aggressive" in record.message for record in caplog.records)


def test_dual_phase_press_records_block_states():
    press = build_press(
        layer_phase_ratios={0: [0.0, 0.5]},
        layer_phase_cold_ratios={0: [0.0, 0.5]},
    )
    layer0 = DummyModule(layer_idx=0)

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
        "hidden_states": make_hidden_states(1),
        "past_key_values": cache,
        "cache_position": torch.tensor([10]),
    }
    press.forward_hook(layer0, [], kwargs, [None, None])

    states = press.layer_block_states[layer0.layer_idx]
    assert "active" in states
    assert "resident_gpu" in states
    assert "permanently_deleted" in states
    assert "offloaded_cpu" in states
    assert "prefetch_to_gpu" in states


def test_dual_phase_press_triggers_decode_compression_on_new_block_boundary():
    press = build_press(layer_phase_ratios={0: [0.0, 0.5]}, layer_phase_cold_ratios={0: [0.0, 0.5]})
    press.compression_interval = 100
    press.score_refresh_interval = 100
    layer0 = DummyModule(layer_idx=0)

    class FakeCacheLayer:
        def __init__(self, keys, values):
            self.keys = keys
            self.values = values

    class FakeCache:
        def __init__(self, keys, values):
            self.layers = [FakeCacheLayer(keys, values)]

    keys, values = make_kv(seq_len=3)
    cache = FakeCache(keys, values)
    output = [None, None]

    for _ in range(2):
        cache.layers[0].keys = torch.cat([cache.layers[0].keys, keys[:, :, :1]], dim=2)
        cache.layers[0].values = torch.cat([cache.layers[0].values, values[:, :, :1]], dim=2)
        kwargs = {
            "hidden_states": make_hidden_states(1),
            "past_key_values": cache,
            "cache_position": torch.tensor([10]),
        }
        press.forward_hook(layer0, [], kwargs, output)

    assert layer0.layer_idx in press.layer_block_states
    assert press.layer_block_states[layer0.layer_idx]["active"].numel() >= 0
