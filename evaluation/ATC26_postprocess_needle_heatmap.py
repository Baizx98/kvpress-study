from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_NAME = "ATC26_needle_heatmap_llama31_8b_ratio50"
ARTIFACTS_DIR = REPO_ROOT / "evaluation" / "results" / "experiments" / EXPERIMENT_NAME / "artifacts"
RAW_DIR = ARTIFACTS_DIR / "raw"
PREDICTIONS_OUT = ARTIFACTS_DIR / "ATC26_needle_heatmap_predictions.csv"
METRICS_LONG_OUT = ARTIFACTS_DIR / "ATC26_needle_heatmap_metrics_long.csv"
METRICS_CELL_OUT = ARTIFACTS_DIR / "ATC26_needle_heatmap_metrics_cell.csv"
FULL_DEPTHS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


def _load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def _method_from_config(config: dict) -> str:
    return str(config.get("press_name", "unknown"))


def _context_length_from_config(config: dict) -> int:
    return int(config["max_context_length"])


def _compression_ratio_from_config(config: dict) -> float:
    return float(config.get("compression_ratio", 0.0))


def _normalize_text(value: object) -> str:
    return " ".join(str(value).lower().split())


def _is_correct(needle: object, answer_prefix: object, predicted_answer: object) -> bool:
    needle_text = _normalize_text(needle)
    answer_text = _normalize_text(f"{answer_prefix} {predicted_answer}")
    if not needle_text:
        return False
    if needle_text in answer_text:
        return True

    # The default Paul Graham needle uses an answer prefix that already contains
    # the beginning of the expected answer, so the generated suffix may not
    # include the full needle sentence verbatim.
    core = needle_text.removeprefix("remember,").strip()
    return core in answer_text


def main() -> None:
    rows = []
    prediction_frames = []

    for config_path in sorted(RAW_DIR.rglob("ATC26_config.yaml")):
        result_dir = config_path.parent
        predictions_path = result_dir / "ATC26_predictions.csv"
        metrics_path = result_dir / "ATC26_metrics.json"
        if not predictions_path.exists() or not metrics_path.exists():
            continue

        import yaml

        with config_path.open() as f:
            config = yaml.safe_load(f) or {}
        if config.get("needle_depth") != FULL_DEPTHS:
            continue

        predictions = pd.read_csv(predictions_path)
        metrics = _load_json(metrics_path)
        method = _method_from_config(config)
        context_length = _context_length_from_config(config)
        compression_ratio = _compression_ratio_from_config(config)

        predictions["method"] = method
        predictions["context_length"] = context_length
        predictions["configured_compression_ratio"] = compression_ratio
        predictions["result_dir"] = str(result_dir)
        prediction_frames.append(predictions)

        for idx, metric in enumerate(metrics):
            pred = predictions.iloc[idx]
            answer = str(pred.get("needle", "")).strip()
            predicted = str(pred.get("predicted_answer", "")).strip()
            answer_prefix = str(pred.get("answer_prefix", "")).strip()
            correct = _is_correct(answer, answer_prefix, predicted)
            rouge_l_f1 = None
            try:
                rouge_l_f1 = float(metric["rouge-l"]["f"])
            except Exception:
                pass
            rows.append(
                {
                    "method": method,
                    "context_length": context_length,
                    "needle_depth": int(pred["needle_depth"]),
                    "compression_ratio": compression_ratio,
                    "correct": int(correct),
                    "rouge_l_f1": rouge_l_f1,
                    "predicted_answer": predicted,
                    "needle": answer,
                    "result_dir": str(result_dir),
                }
            )

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    if prediction_frames:
        pd.concat(prediction_frames, ignore_index=True).to_csv(PREDICTIONS_OUT, index=False)

    long_df = pd.DataFrame(rows)
    long_df.to_csv(METRICS_LONG_OUT, index=False)
    if long_df.empty:
        pd.DataFrame().to_csv(METRICS_CELL_OUT, index=False)
        return

    cell_df = (
        long_df.groupby(["method", "context_length", "needle_depth", "compression_ratio"], as_index=False)
        .agg(
            accuracy=("correct", "mean"),
            samples=("correct", "size"),
            rouge_l_f1=("rouge_l_f1", "mean"),
        )
        .sort_values(["method", "context_length", "needle_depth"])
    )
    cell_df.to_csv(METRICS_CELL_OUT, index=False)


if __name__ == "__main__":
    main()
