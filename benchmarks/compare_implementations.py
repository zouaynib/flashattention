"""How does this kernel compare to the implementations it is meant to be measured against?

Three questions, one script:

1. Which backend does `F.scaled_dot_product_attention` actually dispatch to?
   Everything in this project's plots is labelled "SDPA (FlashAttention CUDA)",
   which is an assumption until the kernel names are read off a profile. PyTorch
   can pick FLASH_ATTENTION, EFFICIENT_ATTENTION (xFormers-derived), CUDNN, or a
   MATH fallback, and they are different algorithms with different performance.

2. How does it compare to Triton's own FlashAttention tutorial kernel? This is
   the most informative comparison available, because it holds the language and
   compiler fixed and varies only the implementation. Being behind hand-tuned
   CUTLASS says little; being behind another Triton kernel is actionable.

3. How does it compare to the standalone `flash-attn` package -- Tri Dao's
   reference implementation, and what people mean by "the official one".

Every comparison is optional. The script reports what it could and could not
run rather than failing, since each has its own install story.

Run:
    python benchmarks/compare_implementations.py
    python benchmarks/compare_implementations.py --triton-tutorial /path/to/06-fused-attention.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import triton

from flash_attn_scratch.autograd import flash_attention

RESULTS = Path(__file__).parent / "results"

BATCH, HEADS, HEAD_DIM = 4, 8, 64
DTYPE = torch.float16
SEQ_LENGTHS = [1024, 2048, 4096, 8192]


# --------------------------------------------------------------------------
# 1. Which SDPA backend actually runs?
# --------------------------------------------------------------------------


def detect_sdpa_backend(seq_len: int = 2048) -> dict:
    """Profile one SDPA call and read the CUDA kernel names.

    Kernel names are the ground truth: PyTorch's FlashAttention port emits
    kernels with "flash" in the name, the memory-efficient backend emits
    cutlass/fmha ones, and the MATH fallback emits ordinary gemm plus softmax.
    """
    q, k, v = (
        torch.randn(BATCH, HEADS, seq_len, HEAD_DIM, device="cuda", dtype=DTYPE)
        for _ in range(3)
    )

    with torch.no_grad():
        F.scaled_dot_product_attention(q, k, v, is_causal=True)  # warm up
        torch.cuda.synchronize()

        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]
        ) as prof:
            F.scaled_dot_product_attention(q, k, v, is_causal=True)
            torch.cuda.synchronize()

    kernels = [
        e.key for e in prof.key_averages() if e.device_time_total > 0 and "aten" not in e.key
    ]
    joined = " ".join(kernels).lower()
    if "flash" in joined:
        backend = "FLASH_ATTENTION"
    elif "cutlass" in joined or "fmha" in joined or "efficient" in joined:
        backend = "EFFICIENT_ATTENTION"
    elif "cudnn" in joined:
        backend = "CUDNN_ATTENTION"
    elif "gemm" in joined or "softmax" in joined:
        backend = "MATH (unfused fallback)"
    else:
        backend = "unrecognised"

    return {"inferred_backend": backend, "cuda_kernels": kernels}


def time_each_backend(seq_len: int = 2048) -> dict[str, float | None]:
    """Force each backend in turn, so the default's timing can be matched to one.

    Corroborates the kernel-name evidence: whichever forced backend matches the
    default's latency is the one being chosen.
    """
    from torch.nn.attention import SDPBackend, sdpa_kernel

    q, k, v = (
        torch.randn(BATCH, HEADS, seq_len, HEAD_DIM, device="cuda", dtype=DTYPE)
        for _ in range(3)
    )
    call = lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True)  # noqa: E731

    timings: dict[str, float | None] = {}
    with torch.no_grad():
        timings["default"] = triton.testing.do_bench(call)
        for name in ("FLASH_ATTENTION", "EFFICIENT_ATTENTION", "CUDNN_ATTENTION", "MATH"):
            try:
                with sdpa_kernel([getattr(SDPBackend, name)]):
                    timings[name] = triton.testing.do_bench(call)
            except Exception:
                timings[name] = None  # unavailable for this shape/dtype/hardware
    return timings


# --------------------------------------------------------------------------
# 2 & 3. The other implementations, each optional
# --------------------------------------------------------------------------


def load_triton_tutorial(path: str | None):
    """Triton's `06-fused-attention.py` lives in the source repo, not the wheel."""
    if path is None:
        return None
    spec = importlib.util.spec_from_file_location("triton_fa_tutorial", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, "attention", None)


def load_flash_attn():
    """Tri Dao's package, whose import name is `flash_attn`.

    That collision is why this project's package is `flash_attn_scratch`:
    installing the official library alongside a package named `flash_attn`
    shadows it, and every import in the repo breaks.
    """
    try:
        from flash_attn import flash_attn_func  # the official package

        return flash_attn_func
    except Exception:
        return None


def build_implementations(tutorial_path: str | None) -> dict:
    """Each entry takes (q, k, v, causal) in (B, H, N, D) and returns that layout.

    `flash_attn_func` is the exception: it wants (B, N, H, D), so it is wrapped
    with transposes. Passing it the wrong layout produces plausible-looking
    numbers rather than an error, which is why the wrapper lives here rather
    than at the call site.
    """
    impls = {
        "ours": lambda q, k, v, causal: flash_attention(q, k, v, causal=causal),
        "sdpa": lambda q, k, v, causal: F.scaled_dot_product_attention(q, k, v, is_causal=causal),
    }

    tutorial = load_triton_tutorial(tutorial_path)
    if tutorial is not None:
        scale = HEAD_DIM**-0.5
        impls["triton_tutorial"] = lambda q, k, v, causal: tutorial(q, k, v, causal, scale)

    fa = load_flash_attn()
    if fa is not None:
        impls["flash_attn_pkg"] = lambda q, k, v, causal: fa(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), causal=causal
        ).transpose(1, 2)

    return impls


def attention_flops(seq_len: int, causal: bool) -> float:
    flops = 4.0 * BATCH * HEADS * seq_len * seq_len * HEAD_DIM
    return flops * 0.5 if causal else flops


def check_agreement(impls: dict, seq_len: int = 512) -> dict[str, float]:
    """Every implementation must produce the same answer before its speed means
    anything. A layout mistake shows up here rather than as a suspiciously good
    benchmark number."""
    torch.manual_seed(0)
    q, k, v = (
        torch.randn(BATCH, HEADS, seq_len, HEAD_DIM, device="cuda", dtype=DTYPE)
        for _ in range(3)
    )
    with torch.no_grad():
        reference = impls["sdpa"](q, k, v, True).float()
        return {
            name: (fn(q, k, v, True).float() - reference).abs().max().item()
            for name, fn in impls.items()
        }


def benchmark(impls: dict) -> list[dict]:
    rows = []
    for causal in (False, True):
        for seq_len in SEQ_LENGTHS:
            q, k, v = (
                torch.randn(BATCH, HEADS, seq_len, HEAD_DIM, device="cuda", dtype=DTYPE)
                for _ in range(3)
            )
            for name, fn in impls.items():
                try:
                    with torch.no_grad():
                        ms = triton.testing.do_bench(lambda: fn(q, k, v, causal))
                    tflops = attention_flops(seq_len, causal) / (ms * 1e-3) / 1e12
                    rows.append(
                        dict(impl=name, causal=causal, seq_len=seq_len,
                             latency_ms=ms, tflops=tflops, status="ok")
                    )
                    print(f"  {name:>16} causal={str(causal):<5} N={seq_len:<6} "
                          f"{ms:8.3f} ms  {tflops:6.2f} TFLOP/s")
                except Exception as exc:
                    rows.append(dict(impl=name, causal=causal, seq_len=seq_len,
                                     latency_ms=None, tflops=None, status=f"error: {exc}"))
                    print(f"  {name:>16} causal={str(causal):<5} N={seq_len:<6} FAILED: {exc}")
            torch.cuda.empty_cache()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triton-tutorial", default=None,
                        help="path to Triton's 06-fused-attention.py")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("this needs a CUDA device")

    print("=== 1. which SDPA backend actually runs? ===")
    detection = detect_sdpa_backend()
    print(f"inferred: {detection['inferred_backend']}")
    for name in detection["cuda_kernels"]:
        print(f"  kernel: {name}")

    print("\nforced-backend timings (the match identifies the default):")
    timings = time_each_backend()
    for name, ms in timings.items():
        print(f"  {name:>20} {'unavailable' if ms is None else f'{ms:8.3f} ms'}")

    print("\n=== 2/3. implementations available ===")
    impls = build_implementations(args.triton_tutorial)
    for name in ("ours", "sdpa", "triton_tutorial", "flash_attn_pkg"):
        print(f"  {name:>16} {'yes' if name in impls else 'NOT AVAILABLE'}")

    print("\nagreement against SDPA (max abs difference, N=512, causal):")
    agreement = check_agreement(impls)
    for name, diff in agreement.items():
        print(f"  {name:>16} {diff:.5f}")

    print("\n=== benchmark ===")
    rows = benchmark(impls)

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "implementation_comparison.json"
    out.write_text(json.dumps({
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "config": dict(batch=BATCH, heads=HEADS, head_dim=HEAD_DIM, dtype=str(DTYPE)),
        "sdpa_backend_detection": detection,
        "forced_backend_timings_ms": timings,
        "agreement_vs_sdpa": agreement,
        "results": rows,
    }, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
