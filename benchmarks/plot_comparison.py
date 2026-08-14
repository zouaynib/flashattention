"""Figures for the four-way implementation comparison.

Reads implementation_comparison.json (and the measured hardware ceiling from
roofline_ceilings.json) and writes PNGs alongside them. No GPU needed.

    python benchmarks/plot_comparison.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS = Path(__file__).parent / "results"

# Ordered worst to best, so the bar chart reads bottom-up as a ranking.
STYLE = {
    "ours": ("#0b6e4f", "^", "ours (Triton, from scratch)"),
    "flash_attn_pkg": ("#d1495b", "D", "flash-attn 2.8.3 (CUDA, Dao)"),
    "sdpa": ("#1b6ca8", "s", "PyTorch SDPA (vendored FlashAttention CUDA)"),
    "triton_tutorial": ("#7a4fa3", "o", "Triton FlashAttention tutorial"),
}


def load():
    data = json.loads((RESULTS / "implementation_comparison.json").read_text())
    ceiling = json.loads((RESULTS / "roofline_ceilings.json").read_text())["peak_tflops"]
    return data, ceiling


def throughput_panels(data: dict, ceiling: float) -> None:
    """Throughput against sequence length, full attention and causal side by side.

    The horizontal line is the machine's measured ceiling -- a large cuBLAS fp16
    matmul, the friendliest shape tensor cores ever see. Nothing that also does a
    softmax should exceed it, so it frames how much of the hardware each
    implementation is actually using.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, causal, title in zip(axes, (False, True), ("full attention", "causal")):
        ax.axhline(ceiling, color="black", linestyle="--", linewidth=1.2)
        ax.annotate(
            f"measured machine ceiling — {ceiling:.1f} TFLOP/s",
            xy=(0.03, ceiling), xycoords=("axes fraction", "data"),
            xytext=(0, 5), textcoords="offset points", fontsize=8.5,
        )

        for impl, (colour, marker, label) in STYLE.items():
            pts = sorted(
                (r["seq_len"], r["tflops"])
                for r in data["results"]
                if r["impl"] == impl and r["causal"] == causal and r["status"] == "ok"
            )
            ax.plot([n for n, _ in pts], [t for _, t in pts], color=colour, marker=marker,
                    linewidth=2, markersize=6, label=label)

        ax.set_xscale("log", base=2)
        ax.set_xlabel("sequence length (tokens)")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.25, linewidth=0.5)

    axes[0].set_ylabel("throughput (TFLOP/s)")
    axes[0].set_ylim(0, ceiling * 1.15)
    axes[0].legend(frameon=False, fontsize=8.5, loc="lower right")

    cfg = data["config"]
    fig.suptitle(
        f"Forward-pass throughput, four implementations on one GPU\n"
        f"{data['gpu']} — B={cfg['batch']}, H={cfg['heads']}, D={cfg['head_dim']}, fp16",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(RESULTS / "comparison_throughput.png", dpi=150)
    plt.close(fig)
    print("wrote", RESULTS / "comparison_throughput.png")


def peak_utilisation(data: dict, ceiling: float) -> None:
    """How much of the machine each implementation uses. The headline figure.

    Averaged over N >= 2048, where every implementation has saturated and the
    ranking is stable.
    """
    means = {}
    for impl in STYLE:
        vals = [
            r["tflops"] for r in data["results"]
            if r["impl"] == impl and not r["causal"] and r["seq_len"] >= 2048
        ]
        means[impl] = sum(vals) / len(vals)

    order = sorted(means, key=means.get)  # worst first, so best sits on top
    fig, ax = plt.subplots(figsize=(8.5, 3.6))

    for i, impl in enumerate(order):
        colour, _, label = STYLE[impl]
        pct = 100 * means[impl] / ceiling
        ax.barh(i, pct, color=colour, height=0.62)
        ax.text(pct - 1.5, i, f"{pct:.0f}%", va="center", ha="right",
                color="white", fontsize=10, weight="bold")
        ax.text(pct + 1.2, i, f"{means[impl]:.1f} TFLOP/s", va="center", fontsize=9)

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([STYLE[i][2] for i in order], fontsize=9)
    ax.set_xlim(0, 118)
    ax.axvline(100, color="black", linestyle="--", linewidth=1.2)
    ax.annotate("machine\nceiling", xy=(101.5, 0.35), xycoords=("data", "axes fraction"),
                fontsize=8.5, color="#444444")
    ax.set_xlabel("share of measured peak throughput (%)")
    ax.set_title(
        "Fraction of the GPU each implementation actually uses\n"
        "full attention, averaged over N >= 2048",
        fontsize=11,
    )
    ax.grid(True, axis="x", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(RESULTS / "comparison_peak_utilisation.png", dpi=150)
    plt.close(fig)
    print("wrote", RESULTS / "comparison_peak_utilisation.png")


if __name__ == "__main__":
    data, ceiling = load()
    throughput_panels(data, ceiling)
    peak_utilisation(data, ceiling)
