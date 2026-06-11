from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_SCRIPT = REPO_ROOT / "figure" / "plot_end2end_paper_draft_model_comparison.py"
SOURCE_EXPERIMENT_NAME = "end2end_serving_paper_draft_modelaware_predicted_20260610"
SOURCE_CSV = (
    REPO_ROOT
    / "figure"
    / "experiments"
    / SOURCE_EXPERIMENT_NAME
    / "paper_draft_end2end_modelaware_metrics_table.csv"
)
EXPERIMENT_NAME = "end2end_serving_paper_draft_modelaware_comparison_20260610"
FIGURE_ROOT = REPO_ROOT / "figure" / "experiments" / EXPERIMENT_NAME
FIGURE_README = FIGURE_ROOT / "README.md"
FIGURE_INDEX = REPO_ROOT / "figure" / "EXPERIMENT_INDEX.md"


def load_base_module():
    spec = importlib.util.spec_from_file_location("model_compare_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load comparison script: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.SOURCE_EXPERIMENT_NAME = SOURCE_EXPERIMENT_NAME
    module.SOURCE_CSV = SOURCE_CSV
    module.EXPERIMENT_NAME = EXPERIMENT_NAME
    module.FIGURE_ROOT = FIGURE_ROOT
    module.FIGURE_README = FIGURE_README
    return module


base = load_base_module()


def write_readme(outputs: list[Path]) -> None:
    output_lines = "\n".join(f"- `{path.name}`" for path in outputs)
    FIGURE_README.write_text(
        f"""# {EXPERIMENT_NAME}

## Purpose

Diagnostic three-model comparison figures for the model-aware paper-draft prediction table.

## Source

- Metrics table: `figure/experiments/{SOURCE_EXPERIMENT_NAME}/paper_draft_end2end_modelaware_metrics_table.csv`
- Plotting script: `figure/plot_end2end_paper_draft_modelaware_comparison.py`

## Figures

{output_lines}

## Layout

Each metric is one figure with three panels: vLLM, InfiniGen, and KVCore. Within each panel, bars compare Llama-3.1, Mistral, and Qwen3 at the same batch size and output length.
"""
    )


def main() -> int:
    base.setup_style()
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    metrics = base.pd.read_csv(SOURCE_CSV)
    outputs: list[Path] = []
    for metric, ylabel, stem in base.METRICS_TO_PLOT:
        outputs.extend(base.plot_metric(metrics, metric, ylabel, f"modelaware_{stem}"))
    write_readme(outputs)
    base.ensure_index_entry(
        FIGURE_INDEX,
        f"- `{EXPERIMENT_NAME}`: diagnostic model-comparison figures for model-aware paper-draft serving predictions.",
    )
    print(f"Read metrics: {SOURCE_CSV}")
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
