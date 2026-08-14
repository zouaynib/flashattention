"""Explanatory diagrams for the write-up.

Two figures that carry the argument: the memory hierarchy that creates the
bottleneck, and the tiling that avoids it. Every number shown is measured on the
GPU used for this project rather than taken from a datasheet.

    python docs/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

FIGURES = Path(__file__).parent / "figures"
CEILINGS = Path(__file__).parent.parent / "benchmarks" / "results" / "roofline_ceilings.json"

INK = "#1c1b19"
MUTED = "#6b6864"
HOT = "#c1440e"
COOL = "#1b6ca8"
GREEN = "#0b6e4f"


def memory_hierarchy() -> None:
    """Why moving data, not multiplying it, is the constraint.

    The bandwidth and compute figures are measured (a large cuBLAS fp16 matmul
    and a large device-to-device copy); the capacities are the card's spec. The
    ridge point that falls out of them is the number that decides whether a
    kernel is worth optimizing for arithmetic or for traffic.
    """
    c = json.loads(CEILINGS.read_text())
    bw, peak = c["peak_bandwidth_gbs"], c["peak_tflops"]
    ridge = c["ridge_point_flops_per_byte"]

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Two boxes, sized to hint at the capacity gap rather than to scale it
    # (24 GB against 6 MB cannot be drawn to scale and stay visible).
    ax.add_patch(Rectangle((0.5, 3.4), 3.6, 2.0, facecolor="#fdf0e6",
                           edgecolor=HOT, linewidth=1.6))
    ax.text(2.3, 4.9, "SRAM  (on-chip)", ha="center", fontsize=11, weight="bold", color=HOT)
    ax.text(2.3, 4.35, "~100 KB per SM\n≈ 6 MB across the card",
            ha="center", fontsize=9.5, color=INK)
    ax.text(2.3, 3.65, "fast, and far too small\nfor an N × N matrix",
            ha="center", fontsize=8.5, color=MUTED, style="italic")

    ax.add_patch(Rectangle((0.5, 0.5), 3.6, 2.3, facecolor="#e9f0f6",
                           edgecolor=COOL, linewidth=1.6))
    ax.text(2.3, 2.35, "HBM  (device memory)", ha="center", fontsize=11,
            weight="bold", color=COOL)
    ax.text(2.3, 1.75, f"24 GB\n{bw:.0f} GB/s measured", ha="center", fontsize=9.5, color=INK)
    ax.text(2.3, 0.95, "holds everything,\nand every trip costs",
            ha="center", fontsize=8.5, color=MUTED, style="italic")

    for y0, y1, dx in ((3.3, 2.9, -0.35), (2.9, 3.3, 0.35)):
        ax.add_patch(FancyArrowPatch((2.3 + dx, y0), (2.3 + dx, y1),
                                     arrowstyle="-|>", mutation_scale=13,
                                     color=MUTED, linewidth=1.3))
    ax.text(3.1, 3.1, "every load\nand store", fontsize=8, color=MUTED, va="center")

    # The consequence.
    ax.text(5.1, 5.0, "The arithmetic is not the problem", fontsize=11,
            weight="bold", color=INK)
    ax.text(5.1, 3.55,
            f"compute      {peak:.1f} TFLOP/s\n"
            f"bandwidth      {bw:.0f} GB/s\n\n"
            f"ratio          {ridge:.0f} FLOP per byte",
            fontsize=10, family="monospace", color=INK, va="center")
    ax.text(5.1, 2.15,
            "A kernel must do more than\n"
            f"{ridge:.0f} floating-point operations per byte\n"
            "it fetches, or the multipliers sit idle\n"
            "waiting for operands.",
            fontsize=9.5, color=MUTED, va="center")
    ax.text(5.1, 0.75,
            "Standard attention writes the whole\nN × N matrix to HBM and reads it back.\n"
            "That traffic is what FlashAttention removes.",
            fontsize=9.5, color=HOT, va="center", weight="bold")

    fig.suptitle("The memory hierarchy an attention kernel lives in\nNVIDIA RTX A5000, measured",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES / "memory_hierarchy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIGURES / "memory_hierarchy.png")


def tiling() -> None:
    """The same computation, laid out two ways.

    Left: every cell exists at once, in HBM. Right: one tile exists at a time,
    in SRAM, and the running statistics carry the rest.
    """
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.5, 4.8))
    n, block = 8, 2

    for ax in (left, right):
        ax.set_xlim(-0.6, n + 1.9)
        ax.set_ylim(-1.5, n + 0.6)
        ax.invert_yaxis()
        ax.axis("off")

    # ---- left: materialized ------------------------------------------------
    for i in range(n):
        for j in range(n):
            left.add_patch(Rectangle((j, i), 1, 1, facecolor="#f6e3d8",
                                     edgecolor="#e3c4b2", linewidth=0.6))
    left.text(n / 2, -0.85, "S = QKᵀ,  all N² of it", ha="center",
              fontsize=11, weight="bold", color=HOT)
    left.text(n / 2, n + 0.35, "written to HBM, read back for softmax,\n"
                               "read again to multiply by V",
              ha="center", fontsize=9.5, color=MUTED, va="top")

    # ---- right: tiled ------------------------------------------------------
    active_row, active_col = 1, 2
    for i in range(n):
        for j in range(n):
            in_row = active_row * block <= i < (active_row + 1) * block
            done = in_row and j < active_col * block
            current = in_row and active_col * block <= j < (active_col + 1) * block
            face = "#0b6e4f" if current else "#dfe9e4" if done else "#f4f3f1"
            right.add_patch(Rectangle((j, i), 1, 1, facecolor=face,
                                      edgecolor="#cfd6d2", linewidth=0.6,
                                      alpha=1.0 if (current or done) else 0.55))

    right.add_patch(Rectangle((0, active_row * block), n, block, fill=False,
                              edgecolor=GREEN, linewidth=2.2))
    right.text(-0.35, active_row * block + block / 2, "Q block\n(stays put)",
               ha="right", va="center", fontsize=9, color=GREEN, weight="bold")
    right.add_patch(FancyArrowPatch((active_col * block + block + 0.15, active_row * block - 0.45),
                                    (n + 0.4, active_row * block - 0.45),
                                    arrowstyle="-|>", mutation_scale=14,
                                    color=GREEN, linewidth=1.6))
    right.text(n + 0.5, active_row * block + block / 2,
               "m\nℓ\nO", ha="center", va="center", fontsize=11,
               color=GREEN, weight="bold")
    right.text(n + 0.5, active_row * block + block + 0.9,
               "three numbers\nper row, carried\nbetween blocks",
               ha="center", va="top", fontsize=8, color=MUTED)
    right.text(n / 2, -0.85, "one tile in SRAM at a time", ha="center",
               fontsize=11, weight="bold", color=GREEN)
    right.text(n / 2, n + 0.35, "K/V blocks stream past; nothing quadratic\n"
                                "is ever allocated",
               ha="center", fontsize=9.5, color=MUTED, va="top")

    fig.suptitle("Same arithmetic, different memory traffic", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES / "tiling.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIGURES / "tiling.png")


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    memory_hierarchy()
    tiling()
