from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE_EXPERIMENT = "ATC26_blockwise_attention_similarity_hotpotqa_3samples"
EXPERIMENT_NAME = "ATC26_design_block_sparse_head_layer_similarity"
SOURCE_JSON = (
    ROOT
    / "evaluation"
    / "results"
    / "experiments"
    / SOURCE_EXPERIMENT
    / "artifacts"
    / "ATC26_attention_similarity_aggregate.json"
)
OUTDIR = ROOT / "figure" / "experiments" / EXPERIMENT_NAME

MODEL_LABELS = {
    "llama31_8b_instruct": "Llama-3.1-8B",
    "mistral_7b_instruct_v03": "Mistral-7B-v0.3",
    "qwen3_8b": "Qwen3-8B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot paper-style ATC26 head/layer sparse block similarity heatmaps."
    )
    parser.add_argument("--model", default="llama31_8b_instruct")
    parser.add_argument("--ratio", type=float, default=0.5)
    parser.add_argument("--fig-width", type=float, default=3.35)
    parser.add_argument("--fig-height", type=float, default=2.85)
    parser.add_argument("--cmap", default="viridis")
    return parser.parse_args()


def load_config(model_key: str, ratio: float) -> dict:
    if not SOURCE_JSON.exists():
        raise FileNotFoundError(f"Missing source aggregate JSON: {SOURCE_JSON}")
    payload = json.loads(SOURCE_JSON.read_text())
    for cfg in payload["configs"]:
        if cfg["model_key"] == model_key and abs(float(cfg["compression_ratio"]) - ratio) < 1e-9:
            return cfg
    available = [(cfg["model_key"], cfg["compression_ratio"]) for cfg in payload["configs"]]
    raise KeyError(f"Missing config model={model_key}, ratio={ratio}. Available: {available}")


def axis_ticks(size: int) -> np.ndarray:
    if size <= 10:
        return np.arange(size)
    ticks = list(range(0, size, 8))
    if ticks[-1] != size - 1:
        ticks.append(size - 1)
    return np.asarray(ticks)


def plot_heatmap(
    matrix: np.ndarray,
    output_stem: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    cbar_label: str,
    figsize: tuple[float, float],
    cmap: str,
) -> None:
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.18, 0.23, 0.53, 0.623])
    cax = fig.add_axes([0.73, 0.23, 0.04, 0.623])
    image = ax.imshow(
        matrix,
        vmin=0.0,
        vmax=1.0,
        cmap=cmap,
        origin="lower",
        interpolation="nearest",
        aspect="equal",
    )

    ticks = axis_ticks(matrix.shape[0])
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([str(int(t)) for t in ticks])
    ax.set_yticklabels([str(int(t)) for t in ticks])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=5)
    ax.grid(False)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_edgecolor("#333333")

    cbar = fig.colorbar(image, cax=cax, ticks=[0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_label(cbar_label)
    cbar.ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1"])

    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches=None)
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches=None)
    plt.close(fig)


def write_readme(cfg: dict, outputs: list[Path]) -> None:
    model_key = cfg["model_key"]
    ratio = float(cfg["compression_ratio"])
    model_label = MODEL_LABELS.get(model_key, model_key)
    lines = [
        f"# {EXPERIMENT_NAME}",
        "",
        "## 图像说明",
        "",
        "本目录保存 ATC26 Design 中块摘要稀疏方法的独立 head/layer 相似度热力图。",
        "",
        "## 数据来源",
        "",
        f"- Source experiment: `{SOURCE_EXPERIMENT}`",
        f"- Aggregate JSON: `{SOURCE_JSON.relative_to(ROOT)}`",
        "- 原始模型输出未重跑，本脚本只复用旧相似度数据重新绘图。",
        "",
        "## 绘图配置",
        "",
        f"- Model: `{model_label}`",
        f"- Compression ratio: `{ratio:g}`",
        "- Dataset: LongBench HotpotQA, 3 samples",
        "- Colormap/range: `viridis`, fixed `[0, 1]`",
        "- Export: PDF and PNG",
        "",
        "## 输出文件",
        "",
    ]
    for path in outputs:
        lines.append(f"- `{path.name}`")
    lines += [
        "",
        "## 运行脚本",
        "",
        "- `figure/ATC26_plot_design_block_sparse_head_layer_similarity.py`",
        "",
        "## 指标",
        "",
        "- `KV-head score cosine`: 不同 KV head group 的 block score vector 余弦相似度。",
        "- `Layer block-index Jaccard`: 不同 layer 最终 kept block index 集合的 Jaccard 相似度。",
        "",
    ]
    (OUTDIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def apply_research_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 14,
            "axes.labelsize": 14,
            "axes.titlesize": 15,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "figure.titlesize": 16,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "0.2",
            "axes.labelcolor": "0.13",
            "xtick.color": "0.13",
            "ytick.color": "0.13",
            "text.color": "0.13",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
            "savefig.pad_inches": 0.04,
        }
    )


def main() -> None:
    args = parse_args()
    apply_research_style()

    cfg = load_config(args.model, args.ratio)
    model_label = MODEL_LABELS.get(args.model, args.model)
    ratio_tag = f"r{args.ratio:g}".replace(".", "p")
    OUTDIR.mkdir(parents=True, exist_ok=True)

    head_matrix = np.asarray(cfg["head_score_cosine_matrix_mean"], dtype=np.float64)
    layer_matrix = np.asarray(cfg["layer_similarity_matrix_mean"], dtype=np.float64)
    figsize = (args.fig_width, args.fig_height)

    head_stem = OUTDIR / f"ATC26_design_{args.model}__{ratio_tag}__kv_head_score_cosine"
    layer_stem = OUTDIR / f"ATC26_design_{args.model}__{ratio_tag}__layer_block_index_jaccard"

    plot_heatmap(
        head_matrix,
        head_stem,
        title="",
        xlabel="KV head group",
        ylabel="KV head group",
        cbar_label="Similarity",
        figsize=figsize,
        cmap=args.cmap,
    )
    plot_heatmap(
        layer_matrix,
        layer_stem,
        title="",
        xlabel="Layer",
        ylabel="Layer",
        cbar_label="Similarity",
        figsize=figsize,
        cmap=args.cmap,
    )

    outputs = [
        head_stem.with_suffix(".pdf"),
        head_stem.with_suffix(".png"),
        layer_stem.with_suffix(".pdf"),
        layer_stem.with_suffix(".png"),
    ]
    summary = {
        "experiment_name": EXPERIMENT_NAME,
        "source_experiment": SOURCE_EXPERIMENT,
        "source_json": str(SOURCE_JSON.relative_to(ROOT)),
        "model_key": args.model,
        "model_label": model_label,
        "compression_ratio": args.ratio,
        "figure_size_inches": list(figsize),
        "colormap": args.cmap,
        "value_range": [0.0, 1.0],
        "head_score_cosine_upper_mean": cfg["head_score_cosine_upper_mean"],
        "layer_similarity_upper_mean": cfg["layer_similarity_upper_mean"],
        "figures": [str(path.relative_to(ROOT)) for path in outputs],
    }
    (OUTDIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    write_readme(cfg, outputs)
    print(OUTDIR)


if __name__ == "__main__":
    main()
