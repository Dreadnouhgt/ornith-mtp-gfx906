# v3.1–v3.3 on DGX Spark (CUDA) — continuation of the gfx906 work

Continuation of this repo's v3 head on 2× DGX Spark (NVIDIA GB10, sm_121),
2026-07-18→20. Weights are on the shared drive (`mtp-v32-spark/`,
`mtp-v33-spark/`), not in the repo. All numbers below are proxies on captured
hidden states; the gfx906 A/B remains the certifier.

## The three runs

| run | change | held-out result |
|---|---|---|
| **v3.1** | loss target: corpus token → **target model's own greedy token** (`argmax(result_norm[i+1] @ lm_head)` — already in the captures, no re-capture) | +1.7 pts greedy step-1 acceptance |
| **v3.2** | +100 self-generated 30k-token docs captured at 32k, `MAX_SEQ=32768` (flash SDPA works at head_dim 256 on Blackwell) | +0.5 pts — mostly a dud; falsified the "weak 32k = position extrapolation" hypothesis (weakest bucket is 2k–8k) |
| **v3.3** | **multi-step**: train the serving recurrence — 4 chained steps, step k+1 eats step k's own hidden state + drafted-token embedding, grads through the chain, step weights 1.0/0.7/0.5/0.35 | deep steps revived, see below |

Clean-eval discipline note: v3.1/v3.2 use different DATA_DIRS → different
shuffles, so their nominal val sets overlap the other's train set. Every
number here is from an explicit both-held-out chunk list
(`eval_acceptance.py` / `eval_prefix.py` + `clean_val.txt`).

## Why multi-step (the finding that matters)

At serve time, `draft()` feeds step k+1 the head's **own output hidden state**
in place of `result_norm`. Trained single-step, the head never sees that
distribution — and single-step *corpus* training actively degrades it:

```
P(prefix >= k), greedy, 172k held-out rows
                 k=1    k=2    k=3    k=4     tokens/cycle @ n-max 2 / 4
v3 (this repo)  78.4%  47.4%  23.3%   8.8%      2.258  /  2.579
v3.2            80.6%  50.1%  25.7%  10.5%      2.306  /  2.668
v3.3 e0         80.8%  53.4%  33.3%  19.9%      2.342  /  2.874
```

v3's s3/s4 are below the *untrained* v2 init — the measured reason
`--spec-draft-n-max 2` benched as optimal here. v3.3 holds step 1 flat and
lifts the tail; the marginal value of the 3rd/4th draft more than doubles.
**v3.3 needs an n-max 2 vs 3 vs 4 A/B to cash out** (+3.7% at n-max 2,
up to +27% tokens/cycle at n-max 4 before draft-forward cost).

`token_embd.weight` is NOT tied to `output.weight` (cosine ~0.09) — the
multi-step trainer needs the real embedding table dequantized from the GGUF.

Also measured: greedy vs temp-0.7 vs temp-1.0 acceptance rank identically,
spread ~0.8 pts → argmax distillation is fine for both bench and production;
KL headroom ≤1 pt.

## Fused MoE (`code/mtp_module_fast.py`)

The reference `MTPBlock.moe()` loops 256 experts in Python; autograd through
that loop is the real cost (backward replays every gather/scatter):

```
T=1007, bf16 autocast, GB10:   forward      fwd+bwd
reference                      53 ms        11.5 s
fused (torch._grouped_mm)      36 ms        93 ms      (124x fwd+bwd)
```

Bit-identical (max diff ~6e-6 fp32, grad norms equal). Same stacked weights,
checkpoints interchangeable; subclasses the reference, which stays untouched.
If ROCm torch ships `_grouped_mm`, the same win applies here.

## Capture-tool fix (`capture-tool/hidden-capture.cpp`)

The documented "`-ub` >= chunk tokens" workaround stops working at 32k chunks
(fused MoE kernel asserts `nbytes_shared <= smpbo` on CUDA at `-ub 33024`).
Root cause fixed instead: `fill_f32_buf()` now **appends** per-ubatch rows
rather than replacing them, decoupling chunk size from `-ub`, plus a guard
refusing to write chunks whose row count != token count. Verified: chunk_0
recaptured at `-ub 512` reproduces the original (embd bitwise, result_norm
cosine 0.9971). `-b` must still cover the chunk; 32k config:
`-c 33024 -b 33024 -ub 2048`.

`capture-tool/capture-cuda.Dockerfile`: CUDA 13 / sm_121 build. sbsa quirk:
the driver stub needs explicit `-lcuda` from `targets/sbsa-linux/lib/stubs`.

## Cross-platform sanity

Q8_0 on GB10 vs gfx906, same corpus line: token ids + raw embeddings
**bitwise identical**; result_norm median cosine 0.9987 with ~0.3% outlier
rows that vary run-to-run on the same hardware (top-8-of-256 routing flips on
near-tie router scores — nondeterminism, not platform skew).

## Next lever (measured, not started)

Prefix survival at k=4 went 10.5%→19.9% in one epoch and hasn't plateaued:
train deeper than served (N_STEPS 6–8) on the fused module (~1 h/epoch now).
Measured dead ends: more corpus data (+0.34), KL (≤0.8 ceiling), extra polish
epochs (+0.1).
