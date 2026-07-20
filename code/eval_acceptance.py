"""Score checkpoints on the two metrics that correspond to the two ways the
drafter is actually used.

For single-token speculative decoding the acceptance probability is exact:

  greedy (temp 0):  argmax p_draft == argmax p_target
  temp T:           sum_x min(p_target(x), p_draft(x))     [= 1 - TV distance]

josh's benches are greedy; the n8n production node runs temperature 1. So a
checkpoint can win one and lose the other. Everything trained so far optimizes
argmax agreement, which is the greedy metric — this measures both, to see
whether that objective is costing anything in production.

Usage: eval_acceptance.py <ckpt.pt>[,<ckpt2.pt>...] [n_chunks]
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/work")
from mtp_module import MTPBlock
from dataset import iter_chunks
from dataset_distill import chunk_samples_v31

WORK = "/work"
DATA_DIRS = [d for d in ("/data/captured-mtp-v2", "/data/captured-mtp-v3") if os.path.isdir(d)]
MAX_SEQ = 8192
ROW_CHUNK = 256          # bound the 248k-vocab softmax working set
TEMPS = [1.0, 0.7]
SEED = 17
POS_BUCKETS = [(0, 512), (512, 2048), (2048, 8192)]

ckpts = sys.argv[1].split(",")
n_chunks = int(sys.argv[2]) if len(sys.argv) > 2 else 12

torch.set_grad_enabled(False)
dev = "cuda" if torch.cuda.is_available() else "cpu"

head = torch.from_numpy(np.load(f"{WORK}/lm_head_f16.npy")).to(dev).to(torch.bfloat16)

# An explicit chunk list is the only honest option here: v3.1 and v3.2 were
# split differently, so most of v3.1's val set is in v3.2's train set. The
# caller passes chunks held out from BOTH.
import random
if os.path.exists("/work/clean_val.txt"):
    val_chunks = [l.strip() for l in open("/work/clean_val.txt") if l.strip()]
    val_chunks = [c for c in val_chunks if os.path.exists(c + ".tokens.i32")]
    print(f"scoring {len(val_chunks)} chunks held out from BOTH models\n", flush=True)
else:
    random.seed(SEED)
    all_chunks = [c for d in DATA_DIRS for c in iter_chunks([d])]
    random.shuffle(all_chunks)
    n_val = max(2, int(len(all_chunks) * 0.05))
    val_chunks = all_chunks[:n_val][:n_chunks]
    print(f"scoring {len(val_chunks)} val chunks from {len(DATA_DIRS)} dirs\n", flush=True)


def overlap_and_top1(hs, h_next, temps):
    """Returns (top1_matches, {temp: summed overlap}, n_rows) accumulated in row chunks."""
    n = hs.shape[0]
    top1 = 0
    ov = {t: 0.0 for t in temps}
    for i in range(0, n, ROW_CHUNK):
        sl = slice(i, min(i + ROW_CHUNK, n))
        ld = (hs[sl].to(torch.bfloat16) @ head.t()).float()
        lt = (h_next[sl].to(torch.bfloat16) @ head.t()).float()
        top1 += (ld.argmax(-1) == lt.argmax(-1)).sum().item()
        for t in temps:
            pd = torch.softmax(ld / t, dim=-1)
            pt = torch.softmax(lt / t, dim=-1)
            ov[t] += torch.minimum(pd, pt).sum(-1).sum().item()
        del ld, lt
    return top1, ov, n


for ckpt in ckpts:
    model = MTPBlock()
    model.load_npz(f"{WORK}/mtp_init.npz")
    if ckpt != "init":
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.to(dev).eval()

    tot_top1 = tot_n = 0
    tot_ov = {t: 0.0 for t in TEMPS}
    b_top1 = np.zeros(len(POS_BUCKETS))
    b_n = np.zeros(len(POS_BUCKETS))

    for base in val_chunks:
        for s in chunk_samples_v31(base, MAX_SEQ):
            hs = model(s["e"].to(dev), s["h"].to(dev), s["pos"].to(dev))
            h_next = s["h_next"].to(dev)
            t1, ov, n = overlap_and_top1(hs, h_next, TEMPS)
            tot_top1 += t1
            tot_n += n
            for t in TEMPS:
                tot_ov[t] += ov[t]
            pos = s["pos"]
            # bucketed greedy agreement, recomputed cheaply per row chunk
            for bi, (lo, hi) in enumerate(POS_BUCKETS):
                m = ((pos >= lo) & (pos < hi))
                if m.sum() == 0:
                    continue
                idx = m.nonzero(as_tuple=True)[0].to(dev)
                ld = (hs[idx].to(torch.bfloat16) @ head.t()).float().argmax(-1)
                lt = (h_next[idx].to(torch.bfloat16) @ head.t()).float().argmax(-1)
                b_top1[bi] += (ld == lt).sum().item()
                b_n[bi] += len(idx)

    name = os.path.basename(ckpt)
    print(f"=== {name}  ({tot_n} rows)")
    print(f"  greedy accept (top-1)   : {100.0 * tot_top1 / tot_n:.2f}%")
    for t in TEMPS:
        print(f"  temp {t} accept (overlap): {100.0 * tot_ov[t] / tot_n:.2f}%")
    print("  greedy by position      : " + " ".join(
        f"[{lo}-{hi}) {100.0 * b_top1[bi] / max(b_n[bi], 1):.1f}%"
        for bi, (lo, hi) in enumerate(POS_BUCKETS)), flush=True)
    print()
    del model
    torch.cuda.empty_cache()
