# FlashAttention-2 from scratch

A Triton implementation of FlashAttention-2, written from first principles to
study the memory-bandwidth bottleneck that limits long-context transformers —
then validated inside a real pretrained model.

**409 tests** · fp16 & bf16 · causal · grouped-query attention · variable
lengths · forward and backward · drop-in for `transformers`

---

## The problem

Attention compares every token with every other token, so the score matrix is
$N \times N$. Double the context and it quadruples. At a 4k context with a
modest 8-head model that matrix is already gigabytes — and it has to travel to
GPU memory and back, repeatedly.

That travel, not the arithmetic, is the bottleneck. A modern GPU can multiply
far faster than it can fetch operands: on the card used here, **86.6 TFLOP/s of
compute against 681 GB/s of bandwidth**. FlashAttention never builds the matrix.
It streams key/value blocks past a resident query block, keeping a running
softmax that is corrected in $O(1)$ per row whenever a larger score appears.

The consequence is a change in scaling exponent, not a constant factor:

![Peak memory against sequence length](benchmarks/results/memory_vs_seqlen.png)

Fitting the measurements, peak memory grows as $N^{1.94}$ for standard attention
and $N^{0.91}$ for this kernel. Standard attention runs out of memory at 16k
tokens; this kernel reaches **65,536 tokens in 2 GB**.

---

## Results

All measured on a single **RTX A5000 (24 GB, sm_86)**, PyTorch 2.4.1 + CUDA 12.4,
Triton 3.0.0. Raw data in [`benchmarks/results/`](benchmarks/results/).

### Memory and throughput

At N=8192 — the last length standard attention survives — forward + backward,
causal:

| | standard PyTorch | **this kernel** | ratio |
| --- | --- | --- | --- |
| peak memory | 16,656 MiB | **274 MiB** | **61× less** |
| latency | 189.3 ms | **25.4 ms** | **7.5× faster** |
| longest context reached | 8,192 | **65,536** | 8× further |

### Speed against the reference implementations

Being fast matters less than being honest about how fast. Four implementations,
same GPU, same shapes, same measurement harness:

![Share of peak throughput](benchmarks/results/comparison_peak_utilisation.png)

This kernel reaches **74% of what the hardware can actually deliver**. The best
implementation — Triton's own FlashAttention tutorial — reaches 97%, *above*
both CUDA implementations.

That last point reframes the gap. It is not Triton versus CUDA: a well-written
Triton kernel saturates this machine. The remaining 1.3× is **this
implementation**, and the [roofline analysis](benchmarks/roofline.py) says
precisely where it goes — the kernel is compute-bound above N=256, using 1–2% of
available bandwidth, so its non-matmul work (the exponential, the causal mask)
sits on the critical path instead of hiding behind the tensor cores.

The peak used as the denominator is **measured**, not quoted: a large cuBLAS
fp16 matmul on the same card. The datasheet figure (111 TFLOP/s) assumes fp16
accumulation and would have flattered every number here by 28%.

### Causal masking does opposite things to the two implementations

![Effect of causal masking](benchmarks/results/causal_effect.png)

Restricting each token to look only backwards makes standard attention **1.75×
slower** — it builds and applies a mask over work it still performs. It makes
this kernel **1.79× faster**, because whole blocks above the diagonal are never
loaded. Same feature, opposite sign. That contrast is the clearest evidence that
tiling changes the algorithm rather than its constant factor.

### Correct inside a real model

The kernel registers as a `transformers` attention implementation, so
**Qwen2.5-0.5B** runs on it unmodified — including its grouped-query attention
(14 query heads over 2 KV heads).

| | PyTorch SDPA | **this kernel** |
| --- | --- | --- |
| perplexity, WikiText-103 test (299k tokens) | 13.0761 | **13.0763** |
| loss reduction over 60 fine-tuning steps | 0.7836 | **0.7845** |
| attention calls recorded | 0 | **3,504** |

The last row matters: a silent dispatch failure would produce *perfect* parity,
because the reference would have run twice. The kernel counts its own
invocations, and 3,504 is exactly 24 layers × 146 batches.

![Loss curves](benchmarks/results/integration_loss_curves.png)

Two fine-tuning runs from an identical seed on identical batches, differing only
in the attention kernel. The curves are indistinguishable; the panel underneath
plots their difference, which stays below 4×10⁻³ with no drift. Perplexity only
exercises the forward pass — this is what tests the backward pass inside a real
optimizer.

---

## What it supports

| | |
| --- | --- |
| dtypes | fp16, bf16 |
| masking | causal, full |
| attention | multi-head, grouped-query, multi-query |
| lengths | square, and $q_{len} \neq kv_{len}$ (KV-cache decoding, chunked prefill) |
| head dims | 16, 32, 64, 128 (powers of two) |
| passes | forward, backward, `torch.autograd.Function` |
| integration | HuggingFace `transformers` attention registry |

### Limitations, stated rather than omitted

- **No arbitrary attention masks.** A dense additive mask is itself an
  $N \times N$ tensor — 32 GB at 64k context — which would negate the memory
  result. Padding-free batches only (batch size 1, or packed sequences).
  Causal, sliding-window and ALiBi-style patterns are all index-computable and
  do not need one; the reference implementations don't support dense masks
  either, for the same reason.
- **No attention dropout.**
- **No autotuning.** Block sizes come from a heuristic. The roofline quantifies
  what tuning could recover (~1.3×).
- **dK and dV use separate kernels** where a fused pass would share their
  recomputed probabilities.
- Causal attention requires at least as many keys as queries.

---

## How it is built

`triton_fwd.py` contains seven kernels. Six of them are dead code, and that is
deliberate — they are the derivation:

| kernel | what it isolates |
| --- | --- |
| `_copy_q_kernel` | grid launch, strided tile loads, boundary masking — no arithmetic |
| `_qk_tile_kernel` | one tile of $QK^\top$; tensor cores and the fp32 accumulator |
| `_softmax_tile_kernel` | safe softmax over a single tile; $-\infty$ masking |
| `_running_max_kernel` | the running maximum $m_i$ across blocks |
| `_running_sum_kernel` | the rescaling factor $\alpha = e^{m_{old}-m_{new}}$ |
| `_running_output_kernel` | the output accumulator — online softmax complete |
| `_fwd_kernel` | the assembled forward pass |

Each has its own tests pinning down one idea, so a failure names the concept
that broke rather than "attention is wrong". The backward pass follows the same
shape in `triton_bwd.py`.

### Correctness

Every result is checked against two independent references — a transparent fp32
baseline and PyTorch's own SDPA. Beyond example-based comparison, the suite
asserts *properties*:

- **Block-size invariance.** Any tiling must give the same answer; changing
  `BLOCK_M`/`BLOCK_N` changes iteration counts and accumulation order.
- **Causality.** Perturbing a future token must leave earlier outputs
  *bit-identical*, not merely close.
- **Incremental decoding.** One token at a time against a growing cache must
  equal a single full pass.
- **Numerical extremes.** Scores in the hundreds, where an unsafe softmax
  overflows to `inf` or underflows the denominator to zero.
- **Group accumulation.** Under GQA, a shared KV head's gradient must be the
  *sum* over its query group — an error there is a silent factor-of-`GROUP_SIZE`.

Tolerances are calibrated by measuring the dtype's own error floor rather than
picked by feel: fp16 lands at ~0.002, bf16 at ~0.013, and the bounds sit 6–10×
above.

**Why there is no `gradcheck`:** it requires float64, and `tl.dot` reaches
tensor cores only in fp16/bf16. Gradients are checked against autograd through
the fp32 baseline instead, which is itself verified against SDPA.

---

## Running it

```bash
pip install -e . && pip install pytest
pytest
```

Tests marked `gpu` need CUDA and Triton and skip automatically elsewhere; the
rest run anywhere. Measured timings:

| | |
| --- | --- |
| first run on a machine (cold Triton cache) | ~190 s |
| any later run | ~12 s |

The gap is **kernel compilation, not execution** — Triton builds a separate
binary per `constexpr` combination and additionally specializes integer
arguments on `x % 16 == 0` and `x == 1`. That is why the slowest test in a cold
run is a `B=1, H=1, N=64` case: it is merely the first to reach the backward
kernels. On an ephemeral machine, point the cache at persistent storage:

```bash
export TRITON_CACHE_DIR=/workspace/.triton_cache
```

Reproducing the measurements:

```bash
python benchmarks/bench_attention.py              # sequence-length sweep
python benchmarks/roofline.py --measure-only      # hardware ceilings (GPU)
python benchmarks/roofline.py                     # roofline analysis
python benchmarks/compare_implementations.py      # four-way comparison
python examples/perplexity_parity.py              # Qwen2.5-0.5B parity
python examples/long_context_training.py curves   # A/B loss curves
```

Using it:

```python
from flash_attn_scratch.autograd import flash_attention

out = flash_attention(q, k, v, causal=True)   # (B, H, N, D), fp16 or bf16
```

```python
from flash_attn_scratch.hf import register
register()
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B", dtype=torch.bfloat16, attn_implementation="flash_triton"
)
```

> The package is `flash_attn_scratch`, not `flash_attn`, so it can coexist with
> Tri Dao's library — which owns that import name.

---

## Future work

- **Fuse dK and dV**, sharing the recomputed probabilities. The comparison
  suggests this is where most of the backward gap lives.
- **`exp2` instead of `exp`**, folding $\log_2 e$ into the scale. Ampere has a
  hardware `ex2.approx.f32`; there is no hardware `exp`.
- **Split the causal loop** so blocks strictly below the diagonal skip the
  elementwise mask entirely.
- **Autotune block sizes** against the measured roofline.
- **Sequence-length vectors** for padded batches — $O(B)$ memory, unlike a dense
  mask.
- **A long-horizon time-series testbed**, using forecasting sequences as a
  second long-context workload.

## Prior art

The algorithm is from [FlashAttention](https://arxiv.org/abs/2205.14135) and
[FlashAttention-2](https://arxiv.org/abs/2307.08691) (Dao et al.). The reference
implementation is [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention);
Triton ships a [tutorial kernel](https://github.com/triton-lang/triton/blob/main/python/tutorials/06-fused-attention.py)
that is benchmarked here. Nothing was copied from either — this is a
from-scratch derivation, which is the point.

## License

MIT.
