"""Figures for the HuggingFace integration experiments.

Reads the HuggingFace integration results and writes PNGs alongside them.
No GPU needed.

    python benchmarks/plot_integration.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS = Path(__file__).parent / "results"

SDPA = "#1b6ca8"
OURS = "#0b6e4f"
EAGER = "#c1440e"
CARD_GB = 24  # RTX A5000


def loss_curves() -> None:
    """Two runs, identical seed and batches, differing only in attention.

    The top panel is the claim; the bottom panel is the evidence that the claim
    is not being carried by the eye. Plotting the residual separately matters
    because two noisy curves drawn on top of each other always look identical.
    """
    data = json.loads((RESULTS / "long_context_curves.json").read_text())
    a = data["curves"]["sdpa"]
    b = data["curves"]["flash_triton"]
    steps = range(len(a))

    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(7.5, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    top.plot(steps, a, color=SDPA, linewidth=2.4, label="PyTorch SDPA", alpha=0.9)
    top.plot(steps, b, color=OURS, linewidth=1.2, linestyle="--", label="ours (Triton)")
    top.set_ylabel("training loss")
    top.set_title(
        f"Fine-tuning {data['model'].split('/')[-1]} — identical seed, identical batches\n"
        f"context {data['seq_len']}, lr {data['lr']}, only the attention kernel differs"
    )
    top.grid(True, alpha=0.25, linewidth=0.5)
    top.legend(frameon=False, fontsize=9)

    drop_a, drop_b = a[0] - a[-1], b[0] - b[-1]
    top.annotate(
        f"loss reduction over {len(a)} steps\n  SDPA  {drop_a:.4f}\n  ours  {drop_b:.4f}",
        xy=(0.02, 0.04),
        xycoords="axes fraction",
        fontsize=9,
        family="monospace",
    )

    residual = [abs(x - y) for x, y in zip(a, b)]
    bottom.plot(steps, residual, color="#555555", linewidth=1.2)
    bottom.axhline(max(residual), color="grey", linestyle=":", linewidth=1)
    bottom.annotate(
        f"worst {max(residual):.2e}",
        xy=(0.98, 0.82),
        xycoords="axes fraction",
        ha="right",
        fontsize=8,
        color="grey",
    )
    bottom.set_xlabel("optimizer step")
    bottom.set_ylabel("|difference|")
    bottom.grid(True, alpha=0.25, linewidth=0.5)

    fig.tight_layout()
    fig.savefig(RESULTS / "integration_loss_curves.png", dpi=150)
    plt.close(fig)
    print("wrote", RESULTS / "integration_loss_curves.png")


def training_memory() -> None:
    """Peak memory for one training step, and where each implementation dies."""
    data = json.loads((RESULTS / "long_context_sweep.json").read_text())
    rows = data["results"]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    # Label offsets are per-implementation: SDPA and ours run out of memory at
    # the same length and within 0.3 GB of each other, so a shared offset would
    # print the two annotations on top of one another.
    style = {
        "eager": (EAGER, "o", "eager attention (materializes N x N)", (10, -6)),
        "sdpa": (SDPA, "s", "PyTorch SDPA", (12, 12)),
        "flash_triton": (OURS, "^", "ours (Triton)", (12, -18)),
    }

    for impl, (colour, marker, label, offset) in style.items():
        ok = [r for r in rows if r["impl"] == impl and r["status"] == "ok"]
        ax.plot(
            [r["seq_len"] for r in ok],
            [r["peak_memory_mb"] / 1024 for r in ok],
            color=colour, marker=marker, linewidth=2, markersize=7, label=label,
        )
        oom = [r for r in rows if r["impl"] == impl and r["status"] == "oom"]
        if oom and ok:
            n = min(r["seq_len"] for r in oom)
            last = max(ok, key=lambda r: r["seq_len"])
            ax.plot([n], [last["peak_memory_mb"] / 1024], marker="x", markersize=14,
                    markeredgewidth=3, color=colour, linestyle="none", zorder=5)
            ax.annotate(f"OOM at {n:,}", xy=(n, last["peak_memory_mb"] / 1024),
                        xytext=offset, textcoords="offset points",
                        fontsize=9, color=colour, weight="bold")

    ax.axhline(CARD_GB, color="black", linestyle="--", linewidth=1)
    ax.annotate(f"{CARD_GB} GB card", xy=(0.02, CARD_GB), xycoords=("axes fraction", "data"),
                xytext=(0, 4), textcoords="offset points", fontsize=9)

    ax.set_xscale("log", base=2)
    ax.set_xlabel("context length (tokens)")
    ax.set_ylabel("peak GPU memory, one training step (GB)")
    ax.set_title(
        "Fine-tuning Qwen2.5-0.5B: how far each attention implementation gets\n"
        "batch 1, AdamW, bf16",
        fontsize=11,
    )
    # The honest caveat belongs on the figure, not only in the caption.
    ax.annotate(
        "past 4k the 152k-token vocabulary\nprojection binds, not attention",
        xy=(0.03, 0.80), xycoords="axes fraction", ha="left",
        fontsize=8.5, color="#444444", style="italic",
    )
    ax.grid(True, which="both", alpha=0.25, linewidth=0.5)
    ax.set_xlim(2**9.6, 2**13.5)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(RESULTS / "integration_memory.png", dpi=150)
    plt.close(fig)
    print("wrote", RESULTS / "integration_memory.png")


if __name__ == "__main__":
    loss_curves()
    training_memory()
