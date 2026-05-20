from __future__ import annotations

import argparse
from matplotlib.ticker import FixedLocator, FormatStrFormatter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_NAME = "pg19_dense_position_ppl_llama31_8b_snapkv_chunkkv_blockwise_ratio50"
DEFAULT_RUN_TAG = "full_3090_f20_stride512_window256"
RESULT_ROOT = REPO_ROOT / "evaluation" / "results" / "experiments" / EXPERIMENT_NAME
FIGURE_ROOT = REPO_ROOT / "figure" / "experiments" / EXPERIMENT_NAME

METHOD_LABELS = {
    "snapkv": "SnapKV",
    "chunkkv": "ChunkKV",
    "blockwise": "KVCore",
    "no_press": "No compression",
}
METHOD_COLORS = {
    "snapkv": "#0072B2",
    "chunkkv": "#E69F00",
    "blockwise": "#009E73",
    "no_press": "#777777",
}
METHOD_MARKERS = {
    "snapkv": "o",
    "chunkkv": "s",
    "blockwise": "^",
    "no_press": "D",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot dense PG19 position PPL curves.")
    parser.add_argument("--run_tag", default=DEFAULT_RUN_TAG)
    parser.add_argument("--input_csv", default=None)
    parser.add_argument("--output_dir", default=str(FIGURE_ROOT))
    parser.add_argument("--rolling", type=int, default=1)
    parser.add_argument("--log_y", action="store_true")
    parser.add_argument("--max_token_length", type=int, default=None)
    parser.add_argument("--paper_acm", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_csv = (
        Path(args.input_csv)
        if args.input_csv
        else RESULT_ROOT / "artifacts" / args.run_tag / "pg19_dense_position_ppl_metrics.csv"
    )
    if not input_csv.exists():
        raise FileNotFoundError(input_csv)

    df = pd.read_csv(input_csv)
    df = df.dropna(subset=["token_length", "subword_ppl"])
    df["token_length"] = df["token_length"].astype(int)
    if args.max_token_length is not None:
        df = df[df["token_length"] <= args.max_token_length]
        if df.empty:
            raise ValueError(f"No rows remain after max_token_length={args.max_token_length}")
    df = df.sort_values(["method", "token_length"])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.paper_acm:
        plt.rcParams.update(
            {
                "font.size": 9,
                "axes.labelsize": 9,
                "xtick.labelsize": 8,
                "ytick.labelsize": 8,
                "legend.fontsize": 8,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
            }
        )
        fig, ax = plt.subplots(figsize=(3.33, 2.25))
    else:
        fig, ax = plt.subplots(figsize=(7.2, 4.2))

    for method, method_df in df.groupby("method", sort=False):
        method_df = method_df.sort_values("token_length")
        y = method_df["subword_ppl"]
        if args.rolling > 1:
            y = y.rolling(args.rolling, center=True, min_periods=1).mean()
        ax.plot(
            method_df["token_length"],
            y,
            label=METHOD_LABELS.get(method, method),
            color=METHOD_COLORS.get(method),
            marker=METHOD_MARKERS.get(method, "o"),
            markersize=2.4 if args.paper_acm else 3.2,
            linewidth=1.35 if args.paper_acm else 1.8,
            markevery=4 if args.paper_acm else 1,
        )

    ax.set_xlabel("Token length")
    ax.set_ylabel("PPL")
    if args.paper_acm:
        ax.set_xlim(-600, 20600)
        ax.set_xticks([0, 4000, 8000, 12000, 16000, 20000])
        ax.set_xticklabels(["0", "4k", "8k", "12k", "16k", "20k"])
    else:
        default_ticks = [2048, 4096, 8192, 16384, 24576, 32768]
        default_labels = ["2k", "4k", "8k", "16k", "24k", "32k"]
        if args.max_token_length is not None:
            tick_pairs = [
                (tick, label)
                for tick, label in zip(default_ticks, default_labels)
                if tick <= args.max_token_length
            ]
            if args.max_token_length not in {tick for tick, _ in tick_pairs}:
                tick_pairs.append((args.max_token_length, f"{args.max_token_length // 1000}k"))
            ax.set_xlim(int(df["token_length"].min()), args.max_token_length)
        else:
            tick_pairs = list(zip(default_ticks, default_labels))
        ax.set_xticks([tick for tick, _ in tick_pairs])
        ax.set_xticklabels([label for _, label in tick_pairs])
    if args.log_y:
        ax.set_yscale("log")
    if args.paper_acm:
        y_min = float(df["subword_ppl"].min())
        y_max = float(df["subword_ppl"].max())
        tick_start = max(0.5, int(y_min * 2) / 2)
        tick_end = int(y_max * 2 + 1) / 2
        ticks = []
        current = tick_start
        while current <= tick_end + 1e-9:
            ticks.append(round(current, 1))
            current += 0.5
        ax.set_ylim(tick_start, tick_end)
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.yaxis.set_minor_locator(FixedLocator([]))
        ax.tick_params(axis="y", direction="in", length=3.0, width=0.8)
        ax.tick_params(axis="x", direction="in", length=3.0, width=0.8)
    else:
        ax.tick_params(axis="both", direction="out")
    if args.paper_acm:
        ax.grid(True, which="major", axis="both", linestyle="--", linewidth=0.45, alpha=0.28)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(0.8)
        ax.legend(
            frameon=False,
            ncols=3,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.18),
            handlelength=1.5,
            columnspacing=0.9,
            handletextpad=0.35,
        )
        fig.tight_layout(pad=0.35)
    else:
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
        ax.legend(frameon=False, ncols=2)
        fig.tight_layout()

    suffix = "_rolling" if args.rolling > 1 else ""
    suffix += "_logy" if args.log_y else ""
    if args.max_token_length is not None:
        suffix += f"_max{args.max_token_length // 1000}k"
    if args.paper_acm:
        suffix += "_paper_acm"
    pdf_path = output_dir / f"pg19_dense_position_ppl{suffix}.pdf"
    png_path = output_dir / f"pg19_dense_position_ppl{suffix}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=220)
    print(pdf_path)
    print(png_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
