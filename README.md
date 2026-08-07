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
