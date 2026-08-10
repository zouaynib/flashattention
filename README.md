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

`pytest` runs the whole suite. For a faster pass while iterating, skip the
long-context and large fp32-reference cases:

```bash
pytest -m "not slow"
```

Every run prints its twelve slowest tests, so it stays obvious what the suite
is spending its time on.

### Why there is no `gradcheck`

`torch.autograd.gradcheck` is the usual way to validate a custom
`autograd.Function`, but it requires float64 and `tl.dot` only reaches tensor
cores in fp16/bf16. The gradients are instead checked against autograd through
the fp32 naive baseline, which is itself verified against PyTorch's SDPA.
