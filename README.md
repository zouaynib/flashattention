# FlashAttention-2 from scratch

A from-scratch Triton implementation of FlashAttention-2, built to study the
memory-bandwidth bottleneck that limits long-context transformers.

> Work in progress. Written up properly in Phase 8.

## Layout

| Path | Contents |
| --- | --- |
| `src/flash_attn/` | Naive baseline and the Triton kernels |
| `tests/` | Correctness suite vs. PyTorch reference implementations |
| `benchmarks/` | Latency and peak-memory sweeps, plots, raw data |
| `viz/` | Interactive online-softmax explainer |
| `docs/` | Write-up and diagrams |

## Running the tests

```bash
pip install -r requirements.txt
pip install -e .
pytest
```

Tests marked `gpu` require CUDA and Triton, and are skipped automatically on a
CPU-only machine. Everything else runs anywhere.

`pytest` runs the whole suite. Measured on an RTX A5000:

| | time |
| --- | --- |
| first run on a machine (cold Triton cache) | ~187 s |
| any subsequent run (warm cache) | ~12 s |
| `pytest -m "not slow"`, warm | ~12 s |

The gap is **kernel compilation, not kernel execution**. Triton JIT-compiles a
separate binary per unique `constexpr` combination — block sizes, head
dimension, causal flag, GQA group size — and additionally specializes integer
arguments on `x % 16 == 0` and `x == 1`. A cold run compiles a few hundred
variants and caches them on disk; every later run reuses them.

That is why the slowest tests are not the biggest ones. The worst offender in a
cold run is a `B=1, H=1, N=64` case, which is merely the first test to reach the
backward kernels and so pays for compiling them.

To keep the cache across runs on an ephemeral machine, point it at persistent
storage:

```bash
export TRITON_CACHE_DIR=/workspace/.triton_cache
```

`-m "not slow"` skips the long-context and 1024-token cases. On a warm cache
that saves a few seconds; it is a convenience, not the main lever.

### Why there is no `gradcheck`

`torch.autograd.gradcheck` is the usual way to validate a custom
`autograd.Function`, but it requires float64 and `tl.dot` only reaches tensor
cores in fp16/bf16. The gradients are instead checked against autograd through
the fp32 naive baseline, which is itself verified against PyTorch's SDPA.
