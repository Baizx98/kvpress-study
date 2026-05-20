from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_NAME = "scoring_overhead_snapkv_chunkkv"
ARTIFACTS_DIR = REPO_ROOT / "evaluation" / "results" / "experiments" / EXPERIMENT_NAME / "artifacts"
SUMMARY_CSV = ARTIFACTS_DIR / "scoring_overhead_summary.csv"
RESULT_README = REPO_ROOT / "evaluation" / "results" / "experiments" / EXPERIMENT_NAME / "README.md"
FIGURE_DIR = REPO_ROOT / "figure" / "experiments" / EXPERIMENT_NAME
FIGURE_README = FIGURE_DIR / "README.md"
FIGURE_INDEX = REPO_ROOT / "figure" / "EXPERIMENT_INDEX.md"


COLORS = {
    "prefill": "#4C78A8",
    "decode": "#72B7B2",
    "window": "#54A24B",
    "snap": "#E45756",
    "chunk": "#F58518",
    "extra": "#B279A2",
}


def ensure_index_entry(index_path: Path, entry: str) -> None:
    text = index_path.read_text() if index_path.exists() else ""
    if entry in text:
        return
    if text and not text.endswith("\n"):
        text += "\n"
    index_path.write_text(text + entry + "\n")


def format_length_ticks(ax, lengths: pd.Series) -> None:
    ticks = list(lengths)
    labels = [f"{int(value // 1024)}k" for value in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)


def save_figure(fig: plt.Figure, stem: str) -> list[str]:
    png_path = FIGURE_DIR / f"{stem}.png"
    pdf_path = FIGURE_DIR / f"{stem}.pdf"
    fig.tight_layout()
    fig.savefig(png_path, dpi=240)
    fig.savefig(pdf_path)
    plt.close(fig)
    return [png_path.name, pdf_path.name]


def plot_absolute_time(df: pd.DataFrame) -> list[str]:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    series = [
        ("fair_prefill_fa_ms_median", "Fair prefill attention", COLORS["prefill"], "o"),
        ("decode_fair_batched_fa_ms_median", "Fair decode attention", COLORS["decode"], "s"),
        ("score_shape_fa_ms_median", "Score-shape attention", COLORS["window"], "^"),
        ("snap_total_no_gather_ms_median", "SnapKV score+topk", COLORS["snap"], "D"),
        ("chunkkv_total_no_gather_ms_median", "ChunkKV score+index", COLORS["chunk"], "P"),
    ]
    for column, label, color, marker in series:
        if column not in df:
            continue
        ax.plot(df["length"], df[column], label=label, color=color, marker=marker, linewidth=2.0, markersize=5)
    ax.set_xlabel("Request length")
    ax.set_ylabel("Median time (ms)")
    ax.set_yscale("log")
    format_length_ticks(ax, df["length"])
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.legend(frameon=False, ncols=2)
    ax.set_title("Scoring overhead vs attention kernels")
    return save_figure(fig, "scoring_overhead_absolute_time")


def plot_ratios(df: pd.DataFrame) -> list[str]:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    series = [
        ("snap_vs_fair_prefill_fa", "SnapKV / fair prefill", COLORS["prefill"], "o"),
        ("snap_vs_decode_fair_batched_fa", "SnapKV / fair decode", COLORS["snap"], "D"),
        ("chunk_vs_fair_prefill_fa", "ChunkKV / fair prefill", COLORS["window"], "^"),
        ("chunk_vs_decode_fair_batched_fa", "ChunkKV / fair decode", COLORS["chunk"], "P"),
    ]
    for column, label, color, marker in series:
        if column not in df:
            continue
        ax.plot(df["length"], df[column], label=label, color=color, marker=marker, linewidth=2.0, markersize=5)
    ax.axhline(1.0, color="#333333", linewidth=1.0, linestyle="--", alpha=0.65)
    ax.set_xlabel("Request length")
    ax.set_ylabel("Overhead ratio")
    ax.set_yscale("log")
    format_length_ticks(ax, df["length"])
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.legend(frameon=False, ncols=2)
    ax.set_title("Scoring overhead ratio")
    return save_figure(fig, "scoring_overhead_ratios")


def plot_chunk_breakdown(df: pd.DataFrame) -> list[str]:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    labels = [f"{int(value // 1024)}k" for value in df["length"]]
    x = range(len(df))
    snap = df["snap_score_ms_median"]
    snap_topk = df.get("snap_topk_index_ms_median", 0)
    chunk_index = df.get("chunk_index_ms_median", df.get("chunk_extra_ms_median", 0))
    ax.bar(x, snap, label="SnapKV attention score", color=COLORS["snap"], width=0.58)
    ax.bar(x, snap_topk, bottom=snap, label="SnapKV token topk", color=COLORS["window"], width=0.58)
    ax.bar(x, chunk_index, bottom=snap + snap_topk, label="Chunk index construction", color=COLORS["extra"], width=0.58)
    total_column = "chunkkv_total_no_gather_ms_median" if "chunkkv_total_no_gather_ms_median" in df else "chunkkv_total_ms_median"
    for idx, total in enumerate(df[total_column]):
        ax.text(idx, total * 1.03, f"{total:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Request length")
    ax.set_ylabel("Median time (ms)")
    ax.set_title("ChunkKV score/index overhead breakdown")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.legend(frameon=False)
    return save_figure(fig, "chunkkv_overhead_breakdown")


def plot_presentation_summary(df: pd.DataFrame) -> list[str]:
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))

    ax = axes[0]
    ax.plot(
        df["length"],
        df["snap_total_no_gather_ms_median"] if "snap_total_no_gather_ms_median" in df else df["snap_score_ms_median"],
        label="SnapKV score+topk",
        color=COLORS["snap"],
        marker="D",
        linewidth=2.0,
    )
    ax.plot(
        df["length"],
        df["decode_fair_batched_fa_ms_median"] if "decode_fair_batched_fa_ms_median" in df else df["decode_fa_ms_median"],
        label="Fair decode attention",
        color=COLORS["decode"],
        marker="s",
        linewidth=2.0,
    )
    ax.set_yscale("log")
    ax.set_xlabel("Request length")
    ax.set_ylabel("Median time (ms)")
    format_length_ticks(ax, df["length"])
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.legend(frameon=False)
    ax.set_title("SnapKV index cost vs fair decode attention")

    ax = axes[1]
    ax.plot(
        df["length"],
        df["snap_vs_decode_fair_batched_fa"] if "snap_vs_decode_fair_batched_fa" in df else df["snap_vs_decode_fa"],
        color=COLORS["snap"],
        marker="D",
        linewidth=2.0,
    )
    ax.axhline(1.0, color="#333333", linewidth=1.0, linestyle="--", alpha=0.65)
    ax.set_xlabel("Request length")
    ax.set_ylabel("SnapKV / fair decode attention")
    format_length_ticks(ax, df["length"])
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_title("Decode-like overhead grows with length")

    return save_figure(fig, "scoring_overhead_presentation_summary")


def write_readmes(generated: list[str]) -> None:
    figure_lines = "\n".join(f"- `{name}`" for name in generated)
    FIGURE_README.write_text(
        f"""# {EXPERIMENT_NAME}

## Purpose

Figures for the SnapKV / ChunkKV scoring-overhead microbenchmark on the RTX 3090.
The current figures use the revised no-gather timing definition: score computation plus top-k/index construction only.

## Source Data

- `evaluation/results/experiments/{EXPERIMENT_NAME}/artifacts/scoring_overhead_summary.csv`
- `evaluation/results/experiments/{EXPERIMENT_NAME}/artifacts/metadata.json`

## Figures

{figure_lines}

## Notes

The attention baseline uses `flash_attn.flash_attn_func` when available. On the current host it falls back to PyTorch SDPA forced `FLASH_ATTENTION` backend because the local `flash_attn` package cannot be imported due to a GLIBC version mismatch.
"""
    )
    RESULT_README.write_text(
        f"""# {EXPERIMENT_NAME}

## Purpose

Benchmark whether SnapKV / ChunkKV scoring overhead is large compared with fused attention kernels under different request lengths.

## Run Script

- `evaluation/bench_scoring_overhead_snapkv_chunkkv.py`

## Results

- Raw artifacts: `evaluation/results/experiments/{EXPERIMENT_NAME}/artifacts/`
- Figures: `figure/experiments/{EXPERIMENT_NAME}/`
- Revised plan: `note/snapkv_chunkkv_scoring_overhead_revised_plan_zh.md`
- Revised analysis note: `note/snapkv_chunkkv_scoring_overhead_revised_results_zh.md`
"""
    )
    ensure_index_entry(
        FIGURE_INDEX,
        f"- `{EXPERIMENT_NAME}`: SnapKV and ChunkKV scoring overhead figures against fused attention kernels.",
    )


def main() -> int:
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(SUMMARY_CSV)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SUMMARY_CSV).sort_values("length")
    generated: list[str] = []
    generated.extend(plot_absolute_time(df))
    generated.extend(plot_ratios(df))
    generated.extend(plot_chunk_breakdown(df))
    generated.extend(plot_presentation_summary(df))
    write_readmes(generated)
    print(f"Wrote {len(generated)} files to {FIGURE_DIR}")
    for name in generated:
        print(FIGURE_DIR / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
