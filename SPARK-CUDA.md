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

## Live A/B verdict (2026-07-20, 2x MI50, ECHO bench) — hypothesis falsified

The gfx906 A/B of v3.3-e0 contradicted the n-max recommendation above, in two
independent ways:

1. **Deeper drafts don't pay on this hardware regardless of head quality.**
   n-max 4 halves throughput (~90 → ~45 t/s at every context length); n-max 3
   is flat-to-negative. Each draft step is a separate batch-1 `llama_decode`
   of the head, and its fixed per-decode cost dominates whatever the extra
   survival buys. The tokens/cycle table above modelled the benefit and
   hand-waved this cost; the cost is the whole game. **n-max 2 stays.**
   Corollary: training still-deeper chains (a 6–8-step v3.4 was the planned
   next run) has no cash-out path here and is cancelled.
2. **The proxy missed a live step-1 regression.** The held-out proxy said
   v3.3's step-1 acceptance was flat vs v3.2; live at n-max 2 it is
   −7.6/−9.2 pp at short/mid context vs deployed v3.1-distill — while gaining
   +3.2/+6.3 pp at 16k/30k. The clean-split evals fix train/val
   *contamination* but not *distribution* mismatch: the held-out chunks are
   self-generated chat prose, the live bench is tool-call/payload-heavy.
   This repo's own "validation accuracy is not draft acceptance" lesson,
   re-confirmed from the other side. Open question (isolable by benching the
   v3.2 GGUF): whether the short-context dip is v3.3's or inherited from
   v3.2, where the 3x self-generated data entered.

v3.3-e0 is not deployed. What survives: the long-context gains at n-max 2,
the per-step measurement methodology, the fused MoE, and the capture fix.
Structural fix before any further training: build the offline eval from
ECHO-shaped captures so proxy numbers predict live ones.

Still-standing measured dead ends: more corpus data (+0.34), KL (≤0.8
ceiling), extra polish epochs (+0.1). The earlier recommendation here to
train N_STEPS 6–8 is withdrawn per the live verdict.

## Cross-hardware live bench (GB10)

The offline proxies never reproduced the live ordering (even on captures of the
exact ECHO token ids — generation-time acceptance is a different measurement),
so the live bench itself was replicated on a DGX Spark: stock HF GGUF + the
published donor heads grafted in (`code/graft_mtp_head.py` — the donor is not
loadable standalone as `-md`; grafting also needs `<arch>.nextn_predict_layers=1`
and `block_count` 40->41, since the loader reads nextn from `blk.(block_count-1)`),
then the ECHO protocol (`code/bench_live.py`: erase slot, n_predict 192, greedy,
`timings.draft_n_accepted/draft_n`, n-max 2, hardened flags).

GB10 acceptance / t/s (n-max 2):

| head | short | medium | long | xlong |
|---|---|---|---|---|
| v3 | 69.9% / 72.1 | 72.8% / 68.8 | 68.7% / 65.2 | 74.1% / 65.0 |
| v3.1-distill | 74.3% / 74.0 | 76.0% / 71.7 | 71.2% / 66.5 | 74.3% / 64.8 |
| v3.2 | 73.6% / 72.9 | 75.5% / 71.2 | 72.5% / 67.5 | 75.6% / 65.8 |
| v3.3-e0 | 73.9% / 74.6 | **77.0%** / 71.0 | **75.2%** / 67.5 | **77.6%** / 66.8 |

Versus the MI50: xlong ordering matches exactly (v33 > v32 > v31), long broadly,
v3.1's short-context strength on both — **but the medium bucket contradicts
outright**: your largest v3.3 regression (-9.2 pp) is a +1 pp win here. Same
tokens, same build, same protocol; the difference is backend numerics
(top-8-of-256 routing near-ties resolve differently per stack). So:

- **Acceptance ordering is partially hardware-dependent.** A CUDA bench screens
  but cannot arbitrate the MI50 short/medium result — your box stays the only
  authority for your deployment.
- **On CUDA, v3.3-e0 is the best head across the board.** The head-quality
  verdict itself is backend-dependent.
- GB10 serves this Q8 model at 65-75 t/s vs the MI50s' 80-98 (273 GB/s LPDDR5X
  vs ~1 TB/s HBM2) — measured in vivo.

## Night 2: three data-composition hypotheses, all falsified

With the fused module (epochs <1 h) three more heads chased v3.3's short/medium
dip. GB10 live bench, n-max 2:

| head | short | medium | long | xlong | what it was |
|---|---|---|---|---|---|
| v3.3-e0 | 73.9 | 77.0 | 75.2 | 77.6 | still best CUDA head |
| v3.3-e1 | 72.4 | 77.0 | 74.5 | 77.5 | 2nd epoch, diminishing |
| v3.3b | 73.2 | 76.8 | 71.1 | 75.4 | recurrence from clean v3.1 parent, no v4 data |
| v3.5 | 70.0 | 75.4 | 72.2 | 78.4 | v3.3-e0 + ECHO-shaped training data (3x) |

- **v3.3b** dropped the v4 self-generated data to fix short -> also lost the
  long/xlong gains (the v4 data is load-bearing) without recovering short.
- **v3.5** added a synthetic ECHO-shaped corpus (`code/gen_echo_corpus.py`:
  production system+tools prefix, tool-call structure, RNG payloads,
  model-generated `<think>`; captured via the new `HIDDEN_CAPTURE_CORPUS_IS_IDS`
  mode, seed-disjoint from the frozen bench) to train the serving distribution
  -> short got **worse** (73.9->70.0), only xlong nudged up.

**Bracketed conclusion:** the short/medium regression is not fixable by data
composition — removing v4 loses long context, adding distribution-matched data
regresses short. It looks intrinsic to the recurrence objective or the v3.2
lineage. The next genuinely different lever is architectural (an EAGLE3-format
drafter — none exists for Ornith yet), not more MTP-head training. Caveat: our
ECHO corpus is *synthetic* (matches your bench's structure, not its real
content), so "distribution-matching fails" vs "our synthetic distribution was
wrong" can't be separated without real production traffic. Checkpoints for MI50
validation: shared drive `mtp-night2-spark/`.
