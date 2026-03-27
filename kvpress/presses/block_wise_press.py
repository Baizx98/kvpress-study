# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.llama.modeling_llama import repeat_kv

from kvpress.presses.base_press import BasePress
from kvpress.utils import get_prerope_query_states


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
    compression_ratio: float = 0.0
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
        Compute block-level query-aware attention scores and expand them back
        to token-level importance scores.
        """

        bsz, q_len, _ = hidden_states.shape
        _, num_key_value_heads, key_len, head_dim = keys.shape
        num_key_value_groups = module.config.num_attention_heads // num_key_value_heads
        num_blocks = math.ceil(key_len / self.block_size)
        pad_len = num_blocks * self.block_size - key_len
        if pad_len > 0:
            keys = F.pad(keys, (0, 0, 0, pad_len))

        keys_blocked = keys.view(
            bsz,
            num_key_value_heads,
            num_blocks,
            self.block_size,
            head_dim,
        )
        block_keys = keys_blocked.mean(dim=3)

        query_states = get_prerope_query_states(module, hidden_states)
        if q_len > 1:
            query_states = query_states.mean(dim=2, keepdim=True)

        block_keys = repeat_kv(block_keys, num_key_value_groups)
        scale = 1.0 / math.sqrt(head_dim)
        block_scores = torch.matmul(query_states, block_keys.transpose(-1, -2)) * scale
        block_scores = block_scores.squeeze(2)
        block_scores = block_scores.view(bsz, num_key_value_heads, num_key_value_groups, num_blocks)
        block_scores = block_scores.mean(dim=2)

        token_scores = block_scores.unsqueeze(-1).expand(-1, -1, -1, self.block_size)
        token_scores = token_scores.reshape(
            bsz,
            num_key_value_heads,
            num_blocks * self.block_size,
        )

        return token_scores[:, :, :key_len]

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
        if self.compression_ratio == 0:
            return keys, values

        assert attentions is None, "BlockWisePress does not support attentions."

        # print("original keys shape:", keys.shape)

        device = keys.device
        B, H_kv, kv_len, head_dim = keys.shape
        block_size = self.block_size

        # ------------------------------------------------------------------
        # Step 1: Get token-level importance scores
        # (B, H_kv, kv_len)
        # ------------------------------------------------------------------
        token_scores = self.score(
            module,
            hidden_states,
            keys,
            values,
            attentions,
            kwargs,
        )

        # ------------------------------------------------------------------
        # Step 2: Build block-level scores without padding tail tokens.
        # ------------------------------------------------------------------
        num_blocks = (kv_len + block_size - 1) // block_size
        block_scores = torch.stack(
            [token_scores[:, :, block_idx * block_size] for block_idx in range(num_blocks)],
            dim=-1,
        )

        # ------------------------------------------------------------------
        # Step 4: Aggregate over KV heads (head-agnostic compression)
        # ------------------------------------------------------------------
        # (B, num_blocks)
        block_scores = block_scores.mean(dim=1)

        # ------------------------------------------------------------------
        # Step 5: Decide how many blocks to keep.
        # ------------------------------------------------------------------
        n_kept_blocks = math.ceil(num_blocks * (1 - self.compression_ratio))
        
        if n_kept_blocks <= 0:
            return keys[:, :, :0], values[:, :, :0]

        n_kept_blocks = min(n_kept_blocks, num_blocks)

        if n_kept_blocks == num_blocks:
            return keys, values

        # ------------------------------------------------------------------
        # Step 6: Top-k block selection
        # ------------------------------------------------------------------
        # (B, n_kept_blocks)
        top_block_indices = torch.topk(
            block_scores,
            k=n_kept_blocks,
            dim=-1,
        ).indices

        # Keep temporal order to preserve cache ordering after compression.
        top_block_indices = top_block_indices.sort(dim=-1).values

        # ------------------------------------------------------------------
        # Step 7: Expand selected block indices to token indices.
        # For each selected block, collect all real tokens (handle tail block correctly).
        # ------------------------------------------------------------------
        # Create mask to indicate which tokens belong to selected blocks.
        # Shape: (B, kv_len)
        mask = torch.zeros(B, kv_len, dtype=torch.bool, device=device)
        
        for batch_idx in range(B):
            for block_offset in range(n_kept_blocks):
                selected_block_idx = top_block_indices[batch_idx, block_offset].item()
                token_start = selected_block_idx * block_size
                token_end = min((selected_block_idx + 1) * block_size, kv_len)
                mask[batch_idx, token_start:token_end] = True

        # Extract token indices from mask.
        token_indices_list = []
        expected_kept_len = None
        for batch_idx in range(B):
            token_idx = torch.where(mask[batch_idx])[0]
            if expected_kept_len is None:
                expected_kept_len = token_idx.numel()
            elif token_idx.numel() != expected_kept_len:
                raise ValueError(
                    f"BlockWisePress got different kept token counts across batch items. "
                    f"First batch: {expected_kept_len}, batch {batch_idx}: {token_idx.numel()}. "
                    "Please use batch_size=1 or adjust settings."
                )
            token_indices_list.append(token_idx)

        token_indices = torch.stack(token_indices_list, dim=0)

        # ------------------------------------------------------------------
        # Step 8: Gather KV cache (same tokens for all KV heads)
        # ------------------------------------------------------------------
        # (B, 1, kept_len, 1) -> broadcast
        gather_indices = token_indices[:, None, :, None].expand(-1, H_kv, -1, head_dim)

        keys = keys.gather(dim=2, index=gather_indices).contiguous()
        values = values.gather(dim=2, index=gather_indices).contiguous()

        return keys, values
