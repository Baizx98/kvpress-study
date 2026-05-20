from __future__ import annotations

from pathlib import Path

import pandas as pd
from rouge import Rouge


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_NAME = "ATC26_needle_8k_token_length_depth_heatmap"
ARTIFACTS_DIR = REPO_ROOT / "evaluation" / "results" / "experiments" / EXPERIMENT_NAME / "artifacts"
PREDICTIONS_PATH = ARTIFACTS_DIR / "ATC26_needle_8k_predictions.csv"
METRICS_LONG_PATH = ARTIFACTS_DIR / "ATC26_needle_8k_metrics_long.csv"
METRICS_CELL_PATH = ARTIFACTS_DIR / "ATC26_needle_8k_metrics_cell.csv"
METRICS_LONG_VALID_PATH = ARTIFACTS_DIR / "ATC26_needle_8k_metrics_long_valid_depth0_90.csv"
METRICS_CELL_VALID_PATH = ARTIFACTS_DIR / "ATC26_needle_8k_metrics_cell_valid_depth0_90.csv"


def _normalize_text(value: object) -> str:
    return " ".join(str(value).lower().split())


def _is_correct(needle: object, answer_prefix: object, predicted_answer: object) -> bool:
    needle_text = _normalize_text(needle)
    answer_text = _normalize_text(f"{answer_prefix} {predicted_answer}")
    if not needle_text:
        return False
    if needle_text in answer_text:
        return True
    core = needle_text.removeprefix("remember,").strip()
    return core in answer_text


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    if not PREDICTIONS_PATH.exists():
        raise SystemExit(f"Missing predictions: {PREDICTIONS_PATH}")

    df = pd.read_csv(PREDICTIONS_PATH)
    if "mode" in df.columns and df["mode"].eq("full").any():
        df = df[df["mode"].eq("full")].copy()
    scorer = Rouge()
    rouge_l = []
    correct = []
    for _, row in df.iterrows():
        pred = str(row["predicted_answer"]).strip()
        needle = str(row["needle"]).strip()
        rouge_l.append(scorer.get_scores(needle, pred)[0]["rouge-l"]["f"])
        correct.append(int(_is_correct(row["needle"], row["answer_prefix"], row["predicted_answer"])))

    df["correct"] = correct
    df["rouge_l_f1"] = rouge_l
    df.to_csv(METRICS_LONG_PATH, index=False)
    valid_df = df[df["needle_depth"].lt(100)].copy()
    valid_df.to_csv(METRICS_LONG_VALID_PATH, index=False)

    cell = (
        df.groupby(["method", "token_length", "needle_depth"], as_index=False)
        .agg(
            accuracy=("correct", "mean"),
            samples=("correct", "size"),
            rouge_l_f1=("rouge_l_f1", "mean"),
        )
        .sort_values(["method", "token_length", "needle_depth"])
    )
    cell.to_csv(METRICS_CELL_PATH, index=False)

    valid_cell = (
        valid_df.groupby(["method", "token_length", "needle_depth"], as_index=False)
        .agg(
            accuracy=("correct", "mean"),
            samples=("correct", "size"),
            rouge_l_f1=("rouge_l_f1", "mean"),
        )
        .sort_values(["method", "token_length", "needle_depth"])
    )
    valid_cell.to_csv(METRICS_CELL_VALID_PATH, index=False)


if __name__ == "__main__":
    main()
