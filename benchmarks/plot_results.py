"""Turn the benchmark sweep into figures.

Reads benchmarks/results/sweep.csv and writes PNGs alongside it. Needs no GPU --
run it anywhere the data is.

Both axes are logarithmic on the latency and memory plots. That is not decoration:
on log-log axes a power law is a straight line and its exponent is the slope, so
quadratic and linear scaling are distinguishable by eye rather than by trusting
a caption. The whole claim of the project is a difference in exponent.

    python benchmarks/plot_results.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS = Path(__file__).parent / "results"

STYLE = {
    "standard": dict(color="#c1440e", marker="o", label="standard PyTorch"),
    "sdpa": dict(color="#1b6ca8", marker="s", label="PyTorch SDPA (FlashAttention CUDA)"),
    "triton": dict(color="#0b6e4f", marker="^", label="Triton"),
}


def load() -> pd.DataFrame:
    df = pd.read_csv(RESULTS / "sweep.csv")
    df["causal"] = df["causal"].astype(bool)
    return df


def _oom_marker(ax, df: pd.DataFrame, impl: str, column: str) -> None:
    """Mark where an implementation stopped fitting in memory.

    This is the result, not an error, so it gets drawn rather than dropped.
    """
    oom = df[(df.impl == impl) & (df.status == "oom")]
    if oom.empty:
        return
    n = int(oom.seq_len.min())
    survived = df[(df.impl == impl) & (df.status == "ok")]
    last = survived.loc[survived.seq_len.idxmax()]
    ax.plot([n], [last[column]], marker="x", markersize=14, markeredgewidth=3,
            color=STYLE[impl]["color"], linestyle="none", zorder=5)
    ax.annotate(
        f"out of memory\nat N={n:,}",
        xy=(n, last[column]),
        xytext=(10, -32),
        textcoords="offset points",
        fontsize=9,
        color=STYLE[impl]["color"],
        weight="bold",
    )


def line_plot(df: pd.DataFrame, column: str, ylabel: str, title: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for impl, style in STYLE.items():
        sub = df[(df.impl == impl) & (df.status == "ok")].sort_values("seq_len")
        ax.plot(sub.seq_len, sub[column], linewidth=2, markersize=6, **style)
        _oom_marker(ax, df, impl, column)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)
    ax.set_xlabel("sequence length (tokens)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25, linewidth=0.5)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS / filename, dpi=150)
    plt.close(fig)
    print("wrote", RESULTS / filename)


def throughput_plot(df: pd.DataFrame) -> None:
    """Sustained TFLOP/s. Flat lines mean the kernel is saturating, and the
    vertical gap between them is the tuning headroom."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    fwd = df[(df["mode"] == "forward") & (df.status == "ok") & df.tflops.notna()]

    for impl, style in STYLE.items():
        for causal, dash in ((False, "-"), (True, "--")):
            sub = fwd[(fwd.impl == impl) & (fwd.causal == causal)].sort_values("seq_len")
            if sub.empty:
                continue
            ax.plot(
                sub.seq_len,
                sub.tflops,
                dash,
                color=style["color"],
                marker=style["marker"],
                markersize=5,
                linewidth=2,
                label=f"{style['label']}{' , causal' if causal else ''}",
            )

    ax.set_xscale("log", base=2)
    ax.set_xlabel("sequence length (tokens)")
    ax.set_ylabel("sustained throughput (TFLOP/s)")
    ax.set_title("Forward pass throughput\nsolid = full attention, dashed = causal")
    ax.grid(True, which="both", alpha=0.25, linewidth=0.5)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(RESULTS / "throughput.png", dpi=150)
    plt.close(fig)
    print("wrote", RESULTS / "throughput.png")


def causal_effect_plot(df: pd.DataFrame) -> None:
    """What turning on causal masking does to each implementation.

    Above 1.0 the mask saves work; below 1.0 it costs work. The two
    implementations land on opposite sides, which is the clearest single
    illustration that tiling changes the algorithm rather than the constant.
    """
    fwd = df[(df["mode"] == "forward") & (df.status == "ok")]
    fig, ax = plt.subplots(figsize=(7.5, 5))

    for impl, style in STYLE.items():
        full = fwd[(fwd.impl == impl) & (~fwd.causal)].set_index("seq_len").latency_ms
        causal = fwd[(fwd.impl == impl) & (fwd.causal)].set_index("seq_len").latency_ms
        ratio = (full / causal).dropna().sort_index()
        ax.plot(ratio.index, ratio.values, linewidth=2, markersize=6, **style)

    ax.axhline(1.0, color="black", linewidth=1, linestyle=":")
    ax.annotate("above: masking saves work", xy=(0.02, 0.93), xycoords="axes fraction", fontsize=9)
    ax.annotate("below: masking costs work", xy=(0.02, 0.06), xycoords="axes fraction", fontsize=9)
    ax.axhline(2.0, color="grey", linewidth=1, linestyle="--")
    ax.annotate("theoretical maximum (2x)", xy=(0.55, 0.88), xycoords="axes fraction",
                fontsize=9, color="grey")

    ax.set_xscale("log", base=2)
    ax.set_xlabel("sequence length (tokens)")
    ax.set_ylabel("speedup from causal masking  (full ÷ causal)")
    ax.set_title("Effect of causal masking on the forward pass")
    ax.grid(True, which="both", alpha=0.25, linewidth=0.5)
    ax.legend(frameon=False, fontsize=9, loc="center right")
    fig.tight_layout()
    fig.savefig(RESULTS / "causal_effect.png", dpi=150)
    plt.close(fig)
    print("wrote", RESULTS / "causal_effect.png")


def main() -> None:
    df = load()
    train = df[(df["mode"] == "forward_backward") & df.causal]

    line_plot(
        train,
        "peak_memory_mb",
        "peak GPU memory (MiB)",
        "Peak memory, forward + backward (causal)\nB=4, H=8, D=64, fp16, RTX A5000 24 GB",
        "memory_vs_seqlen.png",
    )
    line_plot(
        train,
        "latency_ms",
        "latency (ms)",
        "Latency, forward + backward (causal)\nB=4, H=8, D=64, fp16, RTX A5000",
        "latency_vs_seqlen.png",
    )
    throughput_plot(df)
    causal_effect_plot(df)


if __name__ == "__main__":
    main()
