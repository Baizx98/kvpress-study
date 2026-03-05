# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F

from kvpress.presses.base_press import BasePress
from kvpress.presses.block_score_press import BlockScorePress


@dataclass
class BlockWisePress(BasePress):
    """
    BlockWisePress: Uniform compression through independent block processing.

    This wrapper enhances any BlockScorerPress by applying compression independently
    to fixed-size blocks of the sequence. Unlike global compression methods that
    may concentrate selection in high-importance regions, BlockWisePress ensures
    uniform compression across the entire context by processing each block separately.

    Parameters
    ----------
    press : BlockScorerPress
        The underlying scoring method to apply to each block independently.
    block_size : int, default=1024
        Length of each block for independent compression.
    """

    press: BlockScorePress

    def __post_init__(self):
        assert isinstance(self.press, BlockScorePress), "BlockWisePress requires a BlockScorePress as input"

    def post_init_from_model(self, model):
        self.press.post_init_from_model(model)

    @property
    def compression_ratio(self):
        return self.press.compression_ratio

    @compression_ratio.setter
    def compression_ratio(self, value):
        self.press.compression_ratio = value

    @property
    def block_size(self):
        return self.press.block_size

    @block_size.setter
    def block_size(self, value):
        self.press.block_size = value

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Block-wise KV cache compression using block-level importance scores.

        Assumptions
        -----------
        - score() returns token-level scores with shape (B, H_kv, kv_len)
        - Tokens within the same block have identical scores
        - KV compression is head-agnostic (same tokens kept for all KV heads)
        """

        # No compression
        if self.press.compression_ratio == 0:
            return keys, values

        assert attentions is None, "BlockWisePress does not support attentions."

        # print("original keys shape:", keys.shape)

        device = keys.device
        B, H_kv, kv_len, head_dim = keys.shape
        block_size = self.press.block_size

        # ------------------------------------------------------------------
        # Step 1: Get token-level importance scores
        # (B, H_kv, kv_len)
        # ------------------------------------------------------------------
        token_scores = self.press.score(
            module,
            hidden_states,
            keys,
            values,
            attentions,
            kwargs,
        )

        # ------------------------------------------------------------------
        # Step 2: Reshape token scores into blocks
        # ------------------------------------------------------------------
        num_blocks = (kv_len + block_size - 1) // block_size
        pad_len = num_blocks * block_size - kv_len

        if pad_len > 0:
            token_scores = F.pad(token_scores, (0, pad_len))

        # (B, H_kv, num_blocks, block_size)
        block_token_scores = token_scores.view(B, H_kv, num_blocks, block_size)

        # ------------------------------------------------------------------
        # Step 3: One score per block (take first token in each block)
        # ------------------------------------------------------------------
        # (B, H_kv, num_blocks)
        block_scores = block_token_scores[..., 0]

        # ------------------------------------------------------------------
        # Step 4: Aggregate over KV heads (head-agnostic compression)
        # ------------------------------------------------------------------
        # (B, num_blocks)
        block_scores = block_scores.mean(dim=1)

        # ------------------------------------------------------------------
        # Step 5: Decide how many blocks to keep
        # ------------------------------------------------------------------
        n_kept_tokens = int(kv_len * (1 - self.press.compression_ratio))
        n_kept_blocks = math.ceil(n_kept_tokens / block_size)
        n_kept_blocks = min(n_kept_blocks, num_blocks)

        # ------------------------------------------------------------------
        # Step 6: Top-k block selection
        # ------------------------------------------------------------------
        # (B, n_kept_blocks)
        top_block_indices = torch.topk(
            block_scores,
            k=n_kept_blocks,
            dim=-1,
        ).indices

        # ------------------------------------------------------------------
        # Step 7: Map block indices -> token indices
        # ------------------------------------------------------------------
        # (block_size,)
        offsets = torch.arange(block_size, device=device)

        # (B, n_kept_blocks, block_size)
        token_indices = top_block_indices.unsqueeze(-1) * block_size + offsets

        # (B, n_kept_blocks * block_size)
        token_indices = token_indices.view(B, -1)

        # Remove out-of-range tokens from last block
        # token_indices = token_indices[token_indices < kv_len]
        token_indices = token_indices.clamp(max=kv_len - 1)

        # ------------------------------------------------------------------
        # Step 8: Gather KV cache (same tokens for all KV heads)
        # ------------------------------------------------------------------
        # (B, 1, kept_len, 1) -> broadcast
        gather_indices = token_indices[:, None, :, None].expand(-1, H_kv, -1, head_dim)

        keys = keys.gather(dim=2, index=gather_indices).contiguous()
        values = values.gather(dim=2, index=gather_indices).contiguous()

        # print("compressed keys shape:", keys.shape)

        return keys, values
