# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Iterable

import pandas as pd
from datasets import Dataset, load_dataset


DEFAULT_PG19_DATASET_ID = "pg19"
SMOKE_TEST_DATASET_ID = "emozilla/pg19-test"


@dataclass(frozen=True)
class PG19PreparationConfig:
    max_context_tokens: int = 4096
    target_tokens: int = 256


def load_pg19_source_dataframe(dataset_id: str = DEFAULT_PG19_DATASET_ID, split: str = "test") -> pd.DataFrame:
    trust_remote_code = dataset_id == DEFAULT_PG19_DATASET_ID
    last_error = None
    for _ in range(3):
        try:
            dataset = load_dataset(dataset_id, split=split, trust_remote_code=trust_remote_code)
            return dataset.to_pandas()
        except Exception as exc:  # pragma: no cover - network dependent
            last_error = exc
    fallback_dataset_id = os.environ.get("PG19_FALLBACK_SOURCE_DATASET")
    if fallback_dataset_id and fallback_dataset_id != dataset_id:
        dataset = load_dataset(fallback_dataset_id, split=split, trust_remote_code=False)
        return dataset.to_pandas()
    raise RuntimeError(
        f"Failed to load PG19 source dataset '{dataset_id}' after 3 attempts. "
        "The official builder downloads external book assets and may fail under unstable network conditions."
    ) from last_error


def _decode_tokens(tokenizer, token_ids: list[int]) -> str:
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def build_pg19_evaluation_dataframe(
    source_df: pd.DataFrame,
    tokenizer,
    max_context_tokens: int,
    target_tokens: int,
) -> pd.DataFrame:
    assert max_context_tokens > 0, "max_context_tokens must be positive"
    assert target_tokens > 0, "target_tokens must be positive"

    rows: list[dict] = []
    for row_idx, row in source_df.reset_index(drop=True).iterrows():
        token_ids = tokenizer(
            row["text"],
            add_special_tokens=False,
            return_attention_mask=False,
            verbose=False,
        )["input_ids"]
        if len(token_ids) < 2:
            continue

        context_len = min(max_context_tokens, len(token_ids) - 1)
        eval_target_len = min(target_tokens, len(token_ids) - context_len)
        if eval_target_len <= 0:
            continue

        context_ids = token_ids[:context_len]
        target_ids = token_ids[context_len : context_len + eval_target_len]

        rows.append(
            {
                "book_id": str(row_idx),
                "short_book_title": row["short_book_title"],
                "publication_date": row["publication_date"],
                "url": row["url"],
                "task": "pg19",
                "question": "",
                "answer_prefix": "",
                "context": _decode_tokens(tokenizer, context_ids),
                "target_text": _decode_tokens(tokenizer, target_ids),
                "context_ids": context_ids,
                "target_ids": target_ids,
                "context_token_count": len(context_ids),
                "target_token_count": len(target_ids),
                "target_word_count": max(1, len(_decode_tokens(tokenizer, target_ids).split())),
                "source_token_count": len(token_ids),
                "max_new_tokens": len(target_ids),
            }
        )

    return pd.DataFrame(rows)


def create_processed_pg19_dataset(
    tokenizer,
    dataset_id: str = DEFAULT_PG19_DATASET_ID,
    split: str = "test",
    max_context_tokens: int = 4096,
    target_tokens: int = 256,
) -> Dataset:
    source_df = load_pg19_source_dataframe(dataset_id=dataset_id, split=split)
    eval_df = build_pg19_evaluation_dataframe(
        source_df=source_df,
        tokenizer=tokenizer,
        max_context_tokens=max_context_tokens,
        target_tokens=target_tokens,
    )
    return Dataset.from_pandas(eval_df, preserve_index=False)


def iter_pg19_titles(source_df: pd.DataFrame) -> Iterable[str]:
    for title in source_df["short_book_title"].tolist():
        yield str(title)
