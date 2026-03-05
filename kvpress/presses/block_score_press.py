# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.llama.modeling_llama import repeat_kv

from kvpress.presses.scorer_press import ScorerPress
from kvpress.utils import get_prerope_query_states


@dataclass
class BlockScorePress(ScorerPress):
    block_size: int = 16

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:
        """
        Compute block-level query-aware attention scores and expand them
        back to token-level importance scores.

        Returns
        -------
        torch.Tensor
            Importance scores with shape (batch_size, num_kv_heads, key_len).
            Higher scores indicate more important tokens. The tokens with the
            lowest scores will be pruned during compression.
        """

        bsz, q_len, hidden_dim = hidden_states.shape
        _, num_key_value_heads, key_len, head_dim = keys.shape
        block_size = self.block_size
        num_key_value_groups = module.config.num_attention_heads // num_key_value_heads
        num_blocks = math.ceil(key_len / block_size)
        pad_len = num_blocks * block_size - key_len
        if pad_len > 0:
            keys = F.pad(keys, (0, 0, 0, pad_len))
            # V need not be used in scoring, so no need to pad values

        # Step 1: K分块后，计算块内token各个头的和
        # [B, H_kv, num_blocks, block_size, D]
        keys_blocked = keys.view(
            bsz,
            num_key_value_heads,
            num_blocks,
            block_size,
            head_dim,
        )
        # [B, H_kv, num_blocks, D]
        block_keys = keys_blocked.mean(dim=3)

        # Step 2: Compute query statistics 并在seq_len维度上聚合成一个Q
        # [B, H_attn, L, D] L可能为1
        query_states = get_prerope_query_states(module, hidden_states)

        # 聚合所有seq_len位置的query，但这样会混合q的语义，并不好的
        # [B, H_attn, L, D] -> [B, H_attn, 1, D]
        if q_len > 1:
            query_states = query_states.mean(dim=2, keepdim=True)

        # align query heads with KV heads (GQA / MQA)
        block_keys = repeat_kv(block_keys, num_key_value_groups)
        scale = 1.0 / math.sqrt(head_dim)
        # [B, H_attn, 1, D] x [B, H_attn, D, num_blocks] -> [B, H_attn, 1, num_blocks]
        block_scores = torch.matmul(query_states, block_keys.transpose(-1, -2)) * scale

        # [B, H_attn, num_blocks]
        block_scores = block_scores.squeeze(2)

        # Average scores across groups
        # [B, H_kv, groups, num_blocks]
        block_scores = block_scores.view(bsz, num_key_value_heads, num_key_value_groups, num_blocks)
        # [B, H_kv, num_blocks]
        block_scores = block_scores.mean(dim=2)

        # [B, H_kv, num_blocks, block_size]
        token_scores = block_scores.unsqueeze(-1).expand(-1, -1, -1, block_size)

        # [B, H_kv, num_blocks * block_size]
        token_scores = token_scores.reshape(
            bsz,
            num_key_value_heads,
            num_blocks * block_size,
        )

        # 移除padding，返回原始key_len长度
        token_scores = token_scores[:, :, :key_len]

        # (batch_size, num_kv_heads, seq_len)
        return token_scores
