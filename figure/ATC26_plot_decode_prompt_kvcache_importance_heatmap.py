from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "DejaVu Sans"

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_NAME = "ATC26_decode_prompt_kvcache_importance_heatmap_longbench"
ARTIFACT_DIR = ROOT / "evaluation" / "results" / "experiments" / EXPERIMENT_NAME / "artifacts"
RAW_DIR = ARTIFACT_DIR / "raw"
SUMMARY_CSV = ARTIFACT_DIR / "summary_metrics.csv"
OUTDIR = ROOT / "figure" / "experiments" / EXPERIMENT_NAME

DISCARD_BLUE = "#A0C4FF"
KEEP_PINK = "#FFC6FF"
BINARY_CMAP = ListedColormap([DISCARD_BLUE, KEEP_PINK])
LINE_COLORS = ["#0072B2", "#FF2D95", "#00AEEF", "#E69F00", "#009E73", "#D55E00"]


def load_summary_rows() -> list[dict[str, str]]:
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(f"Missing {SUMMARY_CSV}; run the collection script first.")
    with SUMMARY_CSV.open() as f:
        return list(csv.DictReader(f))


def ratio_tag(value: float | str) -> str:
    return str(value).replace(".", "p")


def plot_binary_heatmap(mask: np.ndarray, row: dict[str, str], output: Path) -> None:
    steps, prompt_tokens = mask.shape
    width = min(18.0, max(7.5, prompt_tokens / 650))
    height = min(9.0, max(4.8, steps / 80))
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
    ax.imshow(mask.astype(np.uint8), aspect="auto", interpolation="nearest", cmap=BINARY_CMAP, vmin=0, vmax=1)
    ax.set_xlabel("Prompt token position")
    ax.set_ylabel("Decode step")
    ax.set_title(
        f"{row['dataset']} row {row['dataset_row_index']}, "
        f"steps={row['decode_steps']}, compression={float(row['compression_ratio']):g}"
    )
    ax.set_xlim(0, prompt_tokens - 1)
    ax.set_ylim(steps - 1, 0)
    ax.set_yticks(np.linspace(0, steps - 1, min(6, steps), dtype=int))
    ax.set_xticks(np.linspace(0, prompt_tokens - 1, min(8, prompt_tokens), dtype=int))

    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", color=DISCARD_BLUE, label="discard"),
        plt.Line2D([0], [0], marker="s", linestyle="", color=KEEP_PINK, label="keep"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=True, fontsize=8)
    fig.savefig(output, dpi=320, bbox_inches="tight")
    plt.close(fig)


def plot_keep_frequency(mask: np.ndarray, row: dict[str, str], output: Path) -> None:
    freq = mask.mean(axis=0)
    fig, ax = plt.subplots(figsize=(9.0, 2.6), constrained_layout=True)
    ax.plot(np.arange(freq.shape[0]), freq, color=KEEP_PINK, linewidth=0.8)
    ax.set_xlabel("Prompt token position")
    ax.set_ylabel("Keep frequency")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(
        f"{row['dataset']} row {row['dataset_row_index']}, "
        f"steps={row['decode_steps']}, compression={float(row['compression_ratio']):g}"
    )
    fig.savefig(output, dpi=320, bbox_inches="tight")
    plt.close(fig)


def plot_jaccard_summary(rows: list[dict[str, str]], output: Path) -> None:
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(7.4, 4.2), constrained_layout=True)
    grouped: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for row in rows:
        key = (row["dataset"], row["decode_steps"])
        grouped.setdefault(key, []).append((float(row["compression_ratio"]), float(row["adjacent_jaccard"])))
    for line_idx, ((dataset, steps), points) in enumerate(sorted(grouped.items())):
        points = sorted(points)
        ax.plot(
            [p[0] for p in points],
            [p[1] for p in points],
            marker="o",
            linewidth=1.6,
            color=LINE_COLORS[line_idx % len(LINE_COLORS)],
            label=f"{dataset}, {steps} steps",
        )
    ax.set_xlabel("Compression ratio")
    ax.set_ylabel("Adjacent-step Jaccard")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8)
    fig.savefig(output, dpi=320, bbox_inches="tight")
    plt.close(fig)


def plot_paper_single_column(mask: np.ndarray, output: Path) -> None:
    steps, prompt_tokens = mask.shape
    fig, ax = plt.subplots(figsize=(3.45, 1.55))
    ax.imshow(
        mask.astype(np.uint8),
        aspect="auto",
        interpolation="nearest",
        cmap=BINARY_CMAP,
        vmin=0,
        vmax=1,
        extent=(0, prompt_tokens, 0, steps),
    )
    ax.set_xlabel("Prompt token position", fontsize=7, labelpad=2)
    ax.set_ylabel("Decoding step", fontsize=7, labelpad=2)
    ax.set_xlim(0, prompt_tokens)
    ax.set_ylim(0, steps)
    ax.margins(x=0, y=0)
    ax.set_xticks(np.linspace(0, prompt_tokens, 4, dtype=int)[1:])
    ax.set_yticks(np.linspace(0, steps, 3, dtype=int))
    ax.tick_params(axis="both", labelsize=6, length=2.2, width=0.6, pad=1)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
    fig.subplots_adjust(left=0.145, right=0.995, bottom=0.255, top=0.985)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows = load_summary_rows()
    figures: list[str] = []

    for row in rows:
        npz_path = ROOT / row["npz"]
        data = np.load(npz_path)
        steps = int(row["decode_steps"])
        ratio = float(row["compression_ratio"])
        key = f"keep_mask_s{steps}_r{ratio_tag(ratio)}"
        if key not in data:
            raise KeyError(f"{key} missing from {npz_path}")
        mask = data[key]
        prefix = (
            f"{row['dataset']}__row{row['dataset_row_index']}__"
            f"s{steps}__r{ratio_tag(ratio)}"
        )
        heatmap_path = OUTDIR / f"heatmap_{prefix}.png"
        freq_path = OUTDIR / f"keep_frequency_{prefix}.png"
        plot_binary_heatmap(mask, row, heatmap_path)
        plot_keep_frequency(mask, row, freq_path)
        figures.extend([heatmap_path.name, freq_path.name])
        if row["dataset"] == "gov_report" and row["dataset_row_index"] == "66" and steps == 256 and ratio == 0.5:
            paper_pdf = OUTDIR / "paper_single_column_heatmap_gov_report__row66__s256__r0p5.pdf"
            plot_paper_single_column(mask, paper_pdf)
            figures.append(paper_pdf.name)

    jaccard_path = OUTDIR / "jaccard_adjacent_summary.png"
    plot_jaccard_summary(rows, jaccard_path)
    figures.append(jaccard_path.name)

    summary = {
        "experiment_name": EXPERIMENT_NAME,
        "source": str(SUMMARY_CSV.relative_to(ROOT)),
        "figures": sorted(figures),
    }
    (OUTDIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(OUTDIR)


if __name__ == "__main__":
    main()
