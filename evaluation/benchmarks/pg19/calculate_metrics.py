# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math


def calculate_metrics(df):
    total_nll = float(df["target_nll"].sum())
    total_target_tokens = int(df["target_token_count"].sum())
    total_target_words = int(df["target_word_count"].sum())

    avg_nll_per_token = total_nll / max(total_target_tokens, 1)
    avg_nll_per_word = total_nll / max(total_target_words, 1)

    return {
        "subword_perplexity": round(math.exp(avg_nll_per_token), 4),
        "word_perplexity": round(math.exp(avg_nll_per_word), 4),
        "avg_nll_per_token": round(avg_nll_per_token, 6),
        "avg_nll_per_word": round(avg_nll_per_word, 6),
        "evaluated_books": int(len(df)),
        "total_target_tokens": total_target_tokens,
        "total_target_words": total_target_words,
        "mean_context_tokens": round(float(df["context_token_count"].mean()), 2),
        "mean_target_tokens": round(float(df["target_token_count"].mean()), 2),
    }
