# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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


class FakeCacheLayer:
    def __init__(self, keys, values):
        self.keys = keys
        self.values = values


class FakeCache:
    def __init__(self, keys, values):
        self.layers = [FakeCacheLayer(keys, values)]


def make_kv(batch: int = 1, heads: int = 2, seq_len: int = 8, head_dim: int = 4):
    keys = torch.arange(batch * heads * seq_len * head_dim, dtype=torch.float32).view(batch, heads, seq_len, head_dim)
    values = keys + 1000
    return keys, values


def make_hidden_states(seq_len: int, hidden_dim: int = 16):
    return torch.arange(seq_len * hidden_dim, dtype=torch.float32).view(1, seq_len, hidden_dim)


def build_dual_phase(decode_mode: str = "compute_cold_fixed_budget") -> DualPhasePerLayerPress:
    prefill_press = BlockWisePress(
        compression_ratio=0.5,
        block_size=2,
        q_window_size=2,
        prefix_sink_blocks=0,
        protected_recent_blocks=0,
        query_agg_mode="max",
    )
    decode_press = BlockWisePress(
        compression_ratio=0.0,
        block_size=2,
        q_window_size=2,
        prefix_sink_blocks=0,
        protected_recent_blocks=0,
        query_agg_mode="max",
    )
    return DualPhasePerLayerPress(
        prefill_press=prefill_press,
        decode_press=decode_press,
        decode_mode=decode_mode,  # type: ignore[arg-type]
        block_size=2,
        compression_interval=2,
        decode_hidden_states_buffer_size=2,
    )


def test_block_wise_press_build_block_plan_supports_fixed_keep_budget():
    press = BlockWisePress(
        compression_ratio=0.99,
        block_size=2,
        q_window_size=2,
        prefix_sink_blocks=0,
        protected_recent_blocks=0,
    )
    layer0 = DummyModule(layer_idx=0)
    keys, values = make_kv(seq_len=8)

    plan = press.build_block_plan(
        layer0,
        make_hidden_states(8),
        keys,
        values,
        None,
        {"cache_position": torch.tensor([8])},
        keep_budget=2,
        force_refresh_summary=True,
    )

    assert plan["keep_budget"] == 2
    assert plan["n_kept_blocks"] == 2
    assert plan["token_indices"].shape[-1] == 4


def test_prefill_uses_blockwise_physical_compression_and_records_budget():
    press = build_dual_phase()
    layer0 = DummyModule(layer_idx=0)
    keys, values = make_kv(seq_len=8)

    compressed_keys, compressed_values = press._compress_prefill(
        layer0,
        make_hidden_states(8),
        keys,
        values,
        None,
        {"cache_position": torch.tensor([8])},
    )

    assert compressed_keys.shape[2] == 4
    assert compressed_values.shape[2] == 4
    assert press.layer_prefill_kept_blocks[layer0.layer_idx] == 2


def test_permanent_decode_uses_prefill_budget_and_deletes_physically():
    press = build_dual_phase(decode_mode="permanent_fixed_budget")
    layer0 = DummyModule(layer_idx=0)
    keys, values = make_kv(seq_len=8)
    prefill_keys, prefill_values = press._compress_prefill(
        layer0,
        make_hidden_states(8),
        keys,
        values,
        None,
        {"cache_position": torch.tensor([8])},
    )
    extra_keys, extra_values = make_kv(seq_len=4)
    decode_keys = torch.cat([prefill_keys, extra_keys], dim=2)
    decode_values = torch.cat([prefill_values, extra_values], dim=2)

    compressed_keys, compressed_values = press._permanent_decode_step(
        layer0,
        make_hidden_states(2),
        decode_keys,
        decode_values,
        None,
        {"cache_position": torch.tensor([10])},
    )

    assert compressed_keys.shape[2] == 4
    assert compressed_values.shape[2] == 4
    assert layer0.masked_key_indices is None
    assert press.layer_block_states[layer0.layer_idx]["mode"] == "permanent_fixed_budget"


def test_compute_cold_decode_uses_prefill_budget_without_physical_deletion():
    press = build_dual_phase(decode_mode="compute_cold_fixed_budget")
    layer0 = DummyModule(layer_idx=0)
    keys, values = make_kv(seq_len=8)
    prefill_keys, prefill_values = press._compress_prefill(
        layer0,
        make_hidden_states(8),
        keys,
        values,
        None,
        {"cache_position": torch.tensor([8])},
    )
    extra_keys, extra_values = make_kv(seq_len=4)
    decode_keys = torch.cat([prefill_keys, extra_keys], dim=2)
    decode_values = torch.cat([prefill_values, extra_values], dim=2)

    press._compute_cold_decode_step(
        layer0,
        make_hidden_states(2),
        decode_keys,
        decode_values,
        None,
        {"cache_position": torch.tensor([10])},
    )

    assert decode_keys.shape[2] == 8
    assert layer0.masked_key_indices is not None
    assert press.layer_block_states[layer0.layer_idx]["mode"] == "compute_cold_fixed_budget"


def test_hybrid_decode_physically_retains_total_budget_and_masks_active_subset():
    press = build_dual_phase(decode_mode="hybrid_fixed_budget")
    press.decode_block_budget = 3
    press.decode_cold_block_budget = 2
    layer0 = DummyModule(layer_idx=0)
    keys, values = make_kv(seq_len=8)
    prefill_keys, prefill_values = press._compress_prefill(
        layer0,
        make_hidden_states(8),
        keys,
        values,
        None,
        {"cache_position": torch.tensor([8])},
    )
    extra_keys, extra_values = make_kv(seq_len=4)
    decode_keys = torch.cat([prefill_keys, extra_keys], dim=2)
    decode_values = torch.cat([prefill_values, extra_values], dim=2)

    retained_keys, retained_values = press._hybrid_decode_step(
        layer0,
        make_hidden_states(2),
        decode_keys,
        decode_values,
        None,
        {"cache_position": torch.tensor([10])},
    )

    assert retained_keys.shape[2] == 6
    assert retained_values.shape[2] == 6
    assert layer0.masked_key_indices is not None
    assert press.layer_block_states[layer0.layer_idx]["mode"] == "hybrid_fixed_budget"
    assert press.layer_block_states[layer0.layer_idx]["live_blocks"] == 3
    assert press.layer_block_states[layer0.layer_idx]["active_blocks"] == 2


def test_forward_hook_prefill_writes_compressed_cache():
    press = build_dual_phase()
    layer0 = DummyModule(layer_idx=0)
    keys, values = make_kv(seq_len=8)
    cache = FakeCache(keys, values)

    press.forward_hook(
        layer0,
        [],
        {
            "hidden_states": make_hidden_states(8),
            "past_key_values": cache,
            "cache_position": torch.tensor([8]),
        },
        [None, None],
    )

    assert cache.layers[0].keys.shape[2] == 4
    assert cache.layers[0].values.shape[2] == 4


def test_forward_hook_compute_cold_masks_decode_cache_without_deletion():
    press = build_dual_phase(decode_mode="compute_cold_fixed_budget")
    layer0 = DummyModule(layer_idx=0)
    keys, values = make_kv(seq_len=8)
    cache = FakeCache(keys, values)

    press.forward_hook(
        layer0,
        [],
        {
            "hidden_states": make_hidden_states(8),
            "past_key_values": cache,
            "cache_position": torch.tensor([8]),
        },
        [None, None],
    )
    cache.layers[0].keys = torch.cat([cache.layers[0].keys, keys[:, :, :4]], dim=2)
    cache.layers[0].values = torch.cat([cache.layers[0].values, values[:, :, :4]], dim=2)

    press.forward_hook(
        layer0,
        [],
        {
            "hidden_states": make_hidden_states(1),
            "past_key_values": cache,
            "cache_position": torch.tensor([10]),
        },
        [None, None],
    )

    assert cache.layers[0].keys.shape[2] == 8
    assert layer0.masked_key_indices is not None


def test_forward_hook_hybrid_retains_cache_and_applies_mask():
    press = build_dual_phase(decode_mode="hybrid_fixed_budget")
    press.decode_block_budget = 3
    press.decode_cold_block_budget = 2
    layer0 = DummyModule(layer_idx=0)
    keys, values = make_kv(seq_len=8)
    cache = FakeCache(keys, values)

    press.forward_hook(
        layer0,
        [],
        {
            "hidden_states": make_hidden_states(8),
            "past_key_values": cache,
            "cache_position": torch.tensor([8]),
        },
        [None, None],
    )
    cache.layers[0].keys = torch.cat([cache.layers[0].keys, keys[:, :, :4]], dim=2)
    cache.layers[0].values = torch.cat([cache.layers[0].values, values[:, :, :4]], dim=2)

    press.forward_hook(
        layer0,
        [],
        {
            "hidden_states": make_hidden_states(1),
            "past_key_values": cache,
            "cache_position": torch.tensor([10]),
        },
        [None, None],
    )

    assert cache.layers[0].keys.shape[2] == 6
    assert cache.layers[0].values.shape[2] == 6
    assert layer0.masked_key_indices is not None


def test_cached_mask_validation_drops_out_of_range_indices():
    press = build_dual_phase()
    keys, _ = make_kv(seq_len=4)
    mask = (
        torch.tensor([0, 0]),
        torch.tensor([0, 0]),
        torch.tensor([1, 999]),
    )

    valid_mask = press._validate_mask(mask, keys)

    assert valid_mask is not None
    assert valid_mask[2].tolist() == [1]
