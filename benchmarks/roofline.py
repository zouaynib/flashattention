"""Roofline analysis: is the kernel limited by arithmetic or by memory traffic?

"We reach 78% of SDPA's throughput" is an observation. This turns it into a
diagnosis, by asking what the hardware could possibly deliver and which of its
two ceilings we are pressed against.

Two numbers define the machine:

* peak compute (FLOP/s) -- how fast it can multiply
* peak bandwidth (byte/s) -- how fast it can feed itself

Their ratio is the *ridge point*, an arithmetic intensity in FLOPs per byte.
A kernel below it is memory-bound and gets faster by moving less data; above
it, compute-bound, and only better arithmetic helps.

Attention's compulsory traffic is Q, K, V in and O out -- 4*B*H*N*D elements --
against 4*B*H*N^2*D FLOPs. So its arithmetic intensity is N/2 FLOPs per byte:
it grows with the sequence, and a long enough sequence is always compute-bound.
Where that crossover lands is the interesting part.

Both ceilings are MEASURED here rather than taken from a datasheet. Vendor
"tensor performance" figures assume sparsity and fp16 accumulation; this kernel
uses neither, so quoting them would flatter the result by roughly 2x. Without a
GPU the script falls back to documented specs and says so.

    python benchmarks/roofline.py --measure-only   # on the GPU box
    python benchmarks/roofline.py                 # anywhere, using those ceilings
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

# pandas and matplotlib are imported lazily inside the analysis functions.
# Measuring the hardware ceilings needs a GPU but no analysis stack; the
# analysis needs the stack but no GPU. Keeping the imports local lets each half
# run where it belongs, which matters when the GPU box is a minimal container.

RESULTS = Path(__file__).parent / "results"

# Fallbacks for when there is no GPU to measure. The bandwidth figure is
# arithmetic (384-bit GDDR6 at 16 Gbps); the compute figure is a datasheet
# number and is the one worth distrusting.
A5000_SPEC = {
    "peak_tflops": 111.1,
    "peak_bandwidth_gbs": 768.0,
    "source": "datasheet (unmeasured -- assumes fp16 accumulate, so likely optimistic)",
}


@dataclass
class Ceilings:
    peak_tflops: float
    peak_bandwidth_gbs: float
    source: str

    @property
    def ridge_point(self) -> float:
        """Arithmetic intensity, in FLOPs per byte, where the two roofs meet."""
        return self.peak_tflops * 1e12 / (self.peak_bandwidth_gbs * 1e9)


def measure_ceilings() -> Ceilings:
    """Measure what this machine actually sustains, if there is one."""
    try:
        import torch
        import triton

        if not torch.cuda.is_available():
            raise RuntimeError("no CUDA device")
    except Exception:
        return Ceilings(**A5000_SPEC)

    # Compute: a large square fp16 matmul is the friendliest possible shape for
    # the tensor cores, so cuBLAS here is a fair practical ceiling.
    n = 8192
    a = torch.randn(n, n, device="cuda", dtype=torch.float16)
    b = torch.randn(n, n, device="cuda", dtype=torch.float16)
    ms = triton.testing.do_bench(lambda: a @ b)
    peak_tflops = (2 * n**3) / (ms * 1e-3) / 1e12

    # Bandwidth: a straight copy, which reads and writes one array each.
    big = torch.empty(2**28, device="cuda", dtype=torch.float16)  # 512 MiB
    dst = torch.empty_like(big)
    ms = triton.testing.do_bench(lambda: dst.copy_(big))
    peak_bandwidth_gbs = (2 * big.numel() * big.element_size()) / (ms * 1e-3) / 1e9

    return Ceilings(peak_tflops, peak_bandwidth_gbs, f"measured on {torch.cuda.get_device_name(0)}")


def analyse(df, ceilings: Ceilings):
    """Attach arithmetic intensity and achieved rates to each forward result."""
    fwd = df[(df["mode"] == "forward") & (df.status == "ok")].copy()

    elements = 4 * fwd.batch * fwd.heads * fwd.seq_len * fwd.head_dim
    fwd["hbm_bytes"] = elements * 2  # fp16
    fwd["flops"] = fwd.tflops * 1e12 * (fwd.latency_ms * 1e-3)
    fwd["intensity"] = fwd.flops / fwd.hbm_bytes
    fwd["achieved_bandwidth_gbs"] = fwd.hbm_bytes / (fwd.latency_ms * 1e-3) / 1e9
    fwd["pct_of_peak_compute"] = 100 * fwd.tflops / ceilings.peak_tflops
    fwd["pct_of_peak_bandwidth"] = 100 * fwd.achieved_bandwidth_gbs / ceilings.peak_bandwidth_gbs
    fwd["bound_by"] = [
        "memory" if i < ceilings.ridge_point else "compute" for i in fwd.intensity
    ]
    return fwd


def report(fwd, ceilings: Ceilings) -> None:
    print(f"peak compute   : {ceilings.peak_tflops:7.1f} TFLOP/s")
    print(f"peak bandwidth : {ceilings.peak_bandwidth_gbs:7.1f} GB/s")
    print(f"ridge point    : {ceilings.ridge_point:7.1f} FLOP/byte")
    print(f"source         : {ceilings.source}\n")

    sub = fwd[(fwd.impl == "triton") & (~fwd.causal)].sort_values("seq_len")
    print("ours, forward, full attention:")
    print(f"{'N':>7}{'intensity':>11}{'bound by':>10}{'TFLOP/s':>10}{'% peak':>8}{'GB/s':>9}{'% BW':>7}")
    for _, r in sub.iterrows():
        print(
            f"{int(r.seq_len):>7}{r.intensity:>11.0f}{r.bound_by:>10}"
            f"{r.tflops:>10.1f}{r.pct_of_peak_compute:>8.0f}"
            f"{r.achieved_bandwidth_gbs:>9.0f}{r.pct_of_peak_bandwidth:>7.0f}"
        )

    crossover = sub[sub.bound_by == "compute"]
    if not crossover.empty:
        print(
            f"\ncrossover to compute-bound between N="
            f"{int(sub[sub.bound_by == 'memory'].seq_len.max()) if (sub.bound_by == 'memory').any() else '<128'}"
            f" and N={int(crossover.seq_len.min())}"
        )

    sat = sub[sub.seq_len >= 4096]
    print(f"saturated at {sat.pct_of_peak_compute.mean():.0f}% of assumed peak compute")

    # The measurements bound their own ceiling: no implementation can exceed
    # the hardware, so the fastest observed rate is a floor on the true peak.
    observed = fwd.tflops.max()
    print(f"\nfastest rate observed anywhere: {observed:.1f} TFLOP/s")
    print(f"  -> true peak is at least this, so ours sits between "
          f"{100 * sat.tflops.mean() / ceilings.peak_tflops:.0f}% and "
          f"{100 * sat.tflops.mean() / observed:.0f}% of it")
    if observed > ceilings.peak_tflops:
        print("  -> WARNING: observed exceeds the assumed peak; the ceiling is wrong")


def roofline_plot(fwd, ceilings: Ceilings) -> None:
    """The classic roofline: attainable performance against arithmetic intensity.

    The sloped roof is the bandwidth limit, the flat roof the compute limit, and
    they meet at the ridge point. A kernel sitting under the flat roof cannot be
    helped by moving less data.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 5))
    xs = [2**i for i in range(2, 18)]
    roof = [
        min(ceilings.peak_bandwidth_gbs * 1e9 * x, ceilings.peak_tflops * 1e12) / 1e12 for x in xs
    ]
    ax.plot(xs, roof, color="black", linewidth=2, label="roofline")
    ax.axvline(ceilings.ridge_point, color="grey", linestyle=":", linewidth=1)
    ax.annotate(
        f"ridge point\n{ceilings.ridge_point:.0f} FLOP/byte",
        xy=(ceilings.ridge_point, 1.5),
        xytext=(6, 0),
        textcoords="offset points",
        fontsize=8,
        color="grey",
    )

    for impl, colour, marker in (
        ("sdpa", "#1b6ca8", "s"),
        ("triton", "#0b6e4f", "^"),
        ("standard", "#c1440e", "o"),
    ):
        sub = fwd[(fwd.impl == impl) & (~fwd.causal)].sort_values("seq_len")
        ax.plot(sub.intensity, sub.tflops, marker, color=colour, markersize=7, linestyle="none",
                label=impl)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)
    ax.set_xlabel("arithmetic intensity (FLOP per byte of HBM traffic)")
    ax.set_ylabel("attained performance (TFLOP/s)")
    ax.set_title("Roofline, forward pass (full attention)\n"
                 "points right of the ridge cannot be helped by moving less data")
    ax.grid(True, which="both", alpha=0.25, linewidth=0.5)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(RESULTS / "roofline.png", dpi=150)
    plt.close(fig)
    print("wrote", RESULTS / "roofline.png")


CEILINGS_PATH = RESULTS / "roofline_ceilings.json"


def save_ceilings(ceilings: Ceilings) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    CEILINGS_PATH.write_text(
        json.dumps(
            {
                "peak_tflops": ceilings.peak_tflops,
                "peak_bandwidth_gbs": ceilings.peak_bandwidth_gbs,
                "ridge_point_flops_per_byte": ceilings.ridge_point,
                "source": ceilings.source,
            },
            indent=2,
        )
    )
    print(f"peak compute   : {ceilings.peak_tflops:7.1f} TFLOP/s")
    print(f"peak bandwidth : {ceilings.peak_bandwidth_gbs:7.1f} GB/s")
    print(f"ridge point    : {ceilings.ridge_point:7.1f} FLOP/byte")
    print(f"source         : {ceilings.source}")
    print(f"wrote {CEILINGS_PATH}")


def load_ceilings() -> Ceilings:
    """Prefer ceilings measured on real hardware over the datasheet."""
    if CEILINGS_PATH.exists():
        saved = json.loads(CEILINGS_PATH.read_text())
        if "measured" in saved.get("source", ""):
            return Ceilings(saved["peak_tflops"], saved["peak_bandwidth_gbs"], saved["source"])
    return measure_ceilings()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--measure-only",
        action="store_true",
        help="measure this machine's ceilings and exit; needs a GPU but no analysis stack",
    )
    args = parser.parse_args()

    if args.measure_only:
        save_ceilings(measure_ceilings())
        return

    import pandas as pd

    df = pd.read_csv(RESULTS / "sweep.csv")
    df["causal"] = df["causal"].astype(bool)

    ceilings = load_ceilings()
    fwd = analyse(df, ceilings)
    report(fwd, ceilings)
    roofline_plot(fwd, ceilings)

    fwd.to_csv(RESULTS / "roofline.csv", index=False)
    save_ceilings(ceilings)
    print(f"wrote {RESULTS / 'roofline.csv'}")


if __name__ == "__main__":
    main()
