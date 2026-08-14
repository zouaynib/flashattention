"""Latency and peak-memory benchmarks: naive PyTorch vs SDPA vs our Triton kernel.

The headline measurement is a sequence-length sweep at a fixed transformer
shape. Standard attention materializes the N x N score matrix and eventually
runs out of memory; the tiled kernel does not. Where that failure lands, and
how the two curves diverge before it, is the result this project exists to show.

Methodology, and why each choice matters:

* Timing uses `triton.testing.do_bench`, which warms up, takes a median over
  many repetitions, and flushes the L2 cache between them. Warmup matters
  because a Triton kernel's first call includes JIT compilation. The L2 flush
  matters more: without it, short sequences read a warm cache and report a
  bandwidth the hardware cannot sustain.
* Memory is `max_memory_allocated`, not `memory_reserved`. Reserved includes
  the caching allocator's spare capacity, which would inflate every number by
  an amount unrelated to the algorithm.
* Out-of-memory is recorded as a result, not an error. It is the point.
* Backward timings pass `grad_to_none` so gradient accumulation is not counted
  as kernel time.

Run:
    python benchmarks/bench_attention.py                # full sweep
    python benchmarks/bench_attention.py --quick        # short sweep, for smoke tests
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
import triton

from flash_attn_scratch.autograd import flash_attention
from flash_attn_scratch.naive import standard_attention

RESULTS_DIR = Path(__file__).parent / "results"

# A realistic small-transformer shape, held fixed while the sequence grows.
BATCH, HEADS, HEAD_DIM = 4, 8, 64
DTYPE = torch.float16

SEQ_LENGTHS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
QUICK_SEQ_LENGTHS = [128, 512, 2048]


@dataclass
class Result:
    impl: str
    mode: str  # "forward" or "forward_backward"
    causal: bool
    batch: int
    heads: int
    seq_len: int
    head_dim: int
    dtype: str
    latency_ms: float | None
    peak_memory_mb: float | None
    tflops: float | None
    status: str  # "ok" or "oom"


_OOM_ERRORS = (torch.cuda.OutOfMemoryError, RuntimeError)


def _is_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def _median_ms(fn, grad_to_none) -> float:
    """Median latency in milliseconds.

    `return_mode` is absent from some Triton releases, so fall back to the
    quantile form, which is not.
    """
    try:
        return triton.testing.do_bench(
            fn, warmup=25, rep=100, grad_to_none=grad_to_none, return_mode="median"
        )
    except TypeError:
        median, _, _ = triton.testing.do_bench(
            fn, warmup=25, rep=100, grad_to_none=grad_to_none, quantiles=[0.5, 0.2, 0.8]
        )
        return median


def attention_flops(batch: int, heads: int, seq_len: int, head_dim: int, causal: bool) -> float:
    """FLOPs for one forward pass.

    Two matmuls (QK^T and PV), each 2 * M * N * K. Causal masking halves the
    work asymptotically, since roughly half the score matrix is skipped.
    """
    flops = 4.0 * batch * heads * seq_len * seq_len * head_dim
    return flops * 0.5 if causal else flops


IMPLEMENTATIONS = {
    "standard": lambda q, k, v, causal: standard_attention(q, k, v, causal=causal),
    "sdpa": lambda q, k, v, causal: F.scaled_dot_product_attention(q, k, v, is_causal=causal),
    "triton": lambda q, k, v, causal: flash_attention(q, k, v, causal=causal),
}


def _make_inputs(seq_len: int, requires_grad: bool):
    shape = (BATCH, HEADS, seq_len, HEAD_DIM)
    return [
        torch.randn(shape, device="cuda", dtype=DTYPE, requires_grad=requires_grad)
        for _ in range(3)
    ]


def _measure_peak_memory(fn) -> float:
    """Peak allocation for a single call, in MiB."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 2**20


def benchmark_one(impl: str, seq_len: int, causal: bool, mode: str) -> Result:
    """One configuration. Returns a Result with status "oom" rather than raising."""
    common = dict(
        impl=impl,
        mode=mode,
        causal=causal,
        batch=BATCH,
        heads=HEADS,
        seq_len=seq_len,
        head_dim=HEAD_DIM,
        dtype=str(DTYPE).split(".")[-1],
    )
    backward = mode == "forward_backward"
    fn = IMPLEMENTATIONS[impl]

    try:
        q, k, v = _make_inputs(seq_len, requires_grad=backward)

        if backward:
            do = torch.randn_like(q)

            def step():
                fn(q, k, v, causal).backward(do)

            grad_to_none = [q, k, v]
        else:

            def step():
                with torch.no_grad():
                    fn(q, k, v, causal)

            grad_to_none = None

        peak_mb = _measure_peak_memory(step)
        latency_ms = _median_ms(step, grad_to_none)

        # Backward costs roughly 2x the forward's matmul work on top of a
        # recomputed forward, so ~2.5x total. Only reported for the forward.
        tflops = (
            attention_flops(BATCH, HEADS, seq_len, HEAD_DIM, causal) / (latency_ms * 1e-3) / 1e12
            if not backward
            else None
        )

        return Result(
            **common,
            latency_ms=latency_ms,
            peak_memory_mb=peak_mb,
            tflops=tflops,
            status="ok",
        )

    except _OOM_ERRORS as exc:
        # Not every out-of-memory surfaces as OutOfMemoryError; some allocation
        # paths raise a plain RuntimeError whose message says so.
        if not _is_oom(exc):
            raise
        return Result(**common, latency_ms=None, peak_memory_mb=None, tflops=None, status="oom")
    finally:
        # Never let one configuration's allocations bias the next one's peak.
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def run_sweep(seq_lengths: list[int]) -> list[Result]:
    results: list[Result] = []
    # Once an implementation runs out of memory it will not recover at a longer
    # sequence, so stop asking.
    exhausted: set[tuple[str, str, bool]] = set()

    for mode in ("forward", "forward_backward"):
        for causal in (False, True):
            for impl in IMPLEMENTATIONS:
                for seq_len in seq_lengths:
                    key = (impl, mode, causal)
                    if key in exhausted:
                        continue

                    result = benchmark_one(impl, seq_len, causal, mode)
                    results.append(result)

                    label = f"{impl:>8} {mode:<16} causal={str(causal):<5} N={seq_len:<6}"
                    if result.status == "oom":
                        print(f"{label} OOM")
                        exhausted.add(key)
                    else:
                        print(
                            f"{label} {result.latency_ms:8.3f} ms  "
                            f"{result.peak_memory_mb:9.1f} MiB"
                            + (f"  {result.tflops:6.2f} TFLOP/s" if result.tflops else "")
                        )
    return results


def save(results: list[Result], tag: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = [asdict(r) for r in results]

    metadata = {
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "python": platform.python_version(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    json_path = RESULTS_DIR / f"{tag}.json"
    json_path.write_text(json.dumps({"metadata": metadata, "results": rows}, indent=2))

    csv_path = RESULTS_DIR / f"{tag}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nwrote {json_path} and {csv_path}")
    print(f"GPU: {metadata['gpu']}  torch {metadata['torch']}  triton {metadata['triton']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="short sweep, for smoke tests")
    parser.add_argument("--tag", default=None, help="output filename stem")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("benchmarks require a CUDA device")

    seq_lengths = QUICK_SEQ_LENGTHS if args.quick else SEQ_LENGTHS
    tag = args.tag or ("sweep_quick" if args.quick else "sweep")

    print(f"{BATCH=} {HEADS=} {HEAD_DIM=} dtype={DTYPE}\n")
    save(run_sweep(seq_lengths), tag)


if __name__ == "__main__":
    main()
