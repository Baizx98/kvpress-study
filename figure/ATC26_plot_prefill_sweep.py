from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_NAME = "ATC26_prefill_sweep_blockwise_snapkv_chunkkv_longbench_needle_pg19"
ARTIFACTS_DIR = REPO_ROOT / "evaluation" / "results" / "experiments" / EXPERIMENT_NAME / "artifacts"
METRICS_LONG = ARTIFACTS_DIR / "ATC26_metrics_long.csv"
METRICS_FULL_LONG = ARTIFACTS_DIR / "ATC26_metrics_full_long.csv"
FIGURE_DIR = REPO_ROOT / "figure" / "experiments" / EXPERIMENT_NAME
FIGURE_README = FIGURE_DIR / "README.md"
FIGURE_INDEX = REPO_ROOT / "figure" / "EXPERIMENT_INDEX.md"

METHOD_LABELS = {
    "blockwise": "BlockWise",
    "snapkv": "SnapKV",
    "chunkkv": "ChunkKV",
}
METHOD_COLORS = {
    "blockwise": "#1f77b4",
    "snapkv": "#2ca02c",
    "chunkkv": "#ff7f0e",
}


def ensure_index_entry(index_path: Path, entry: str) -> None:
    text = index_path.read_text() if index_path.exists() else ""
    if entry in text:
        return
    if text and not text.endswith("\n"):
        text += "\n"
    index_path.write_text(text + entry + "\n")


def plot_group(df: pd.DataFrame, dataset_selector, title: str, ylabel: str, filename: str) -> None:
    subset = df[dataset_selector(df)].copy()
    if subset.empty:
        return
    grouped = (
        subset.groupby(["model", "method", "compression_ratio"], as_index=False)["score"]
        .mean()
        .sort_values(["model", "method", "compression_ratio"])
    )
    models = sorted(grouped["model"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 4.0), squeeze=False)
    for ax, model in zip(axes[0], models):
        model_df = grouped[grouped["model"] == model]
        for method in ["blockwise", "snapkv", "chunkkv"]:
            method_df = model_df[model_df["method"] == method]
            if method_df.empty:
                continue
            ax.plot(
                method_df["compression_ratio"],
                method_df["score"],
                marker="o",
                linewidth=2,
                label=METHOD_LABELS.get(method, method),
                color=METHOD_COLORS.get(method),
            )
        ax.set_title(model)
        ax.set_xlabel("Compression ratio")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / filename, dpi=200)
    plt.close(fig)


def write_readme(generated: list[str]) -> None:
    text = f"""# {EXPERIMENT_NAME}

## 说明

本目录保存 ATC26 prefill-only 压缩实验的聚合图。图像由 `figure/ATC26_plot_prefill_sweep.py` 优先从 `ATC26_metrics_full_long.csv` 生成。

## 图像

{chr(10).join(f"- `{name}`" for name in generated) if generated else "- 尚无可生成图像"}
"""
    FIGURE_README.write_text(text)
    ensure_index_entry(
        FIGURE_INDEX,
        f"- `{EXPERIMENT_NAME}`: ATC26 prefill-only sweep figures for BlockWise, SnapKV, and ChunkKV.",
    )


def main() -> int:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = METRICS_FULL_LONG if METRICS_FULL_LONG.exists() else METRICS_LONG
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    df = pd.read_csv(metrics_path)
    generated: list[str] = []
    if df.empty:
        write_readme(generated)
        return 0

    specs = [
        (
            lambda frame: frame["dataset"].eq("longbench"),
            "ATC26 LongBench Quality vs Compression",
            "LongBench score",
            "ATC26_longbench_quality_vs_compression_by_model.png",
        ),
        (
            lambda frame: frame["dataset"].eq("needle_in_haystack"),
            "ATC26 Needle Quality vs Compression",
            "Avg ROUGE-L F1",
            "ATC26_needle_accuracy_vs_compression_by_model.png",
        ),
        (
            lambda frame: frame["dataset"].eq("pg19"),
            "ATC26 PG19 Perplexity vs Compression",
            "Subword perplexity",
            "ATC26_pg19_ppl_vs_compression_by_model.png",
        ),
    ]
    for selector, title, ylabel, filename in specs:
        before = set(FIGURE_DIR.glob(filename))
        plot_group(df, selector, title, ylabel, filename)
        after = set(FIGURE_DIR.glob(filename))
        if after - before or (FIGURE_DIR / filename).exists():
            generated.append(filename)

    write_readme(generated)
    print(f"Wrote {len(generated)} figures to {FIGURE_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
