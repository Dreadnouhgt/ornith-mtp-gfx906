"""Per-step draft acceptance for multiple checkpoints, free-running.

Replays the llama.cpp draft recurrence (token = prev drafted token, h = head's
own hidden state, pos+1) for N_STEPS and reports, per checkpoint:
  - s_k : greedy agreement with the target's own token at horizon k
  - E[accepted] : expected accepted draft length under sequential acceptance,
    modelling position-k correctness as independent with prob s_k
    (E = sum_k prod_{j<=k} s_j)

Scores only chunks in /work/clean_val.txt (held out from every trained model).

Usage: eval_multistep.py <ckptA.pt,ckptB.pt,...> [n_steps]
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/work")
from mtp_module import MTPBlock
from dataset_multistep import chunk_samples_multistep

WORK = "/work"
MAX_SEQ = 8192
N_STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 4
ckpts = sys.argv[1].split(",")

torch.set_grad_enabled(False)
dev = "cuda" if torch.cuda.is_available() else "cpu"

head = torch.from_numpy(np.load(f"{WORK}/lm_head_f16.npy")).to(dev).to(torch.bfloat16)
tok_embd = torch.from_numpy(np.load(f"{WORK}/token_embd_f16.npy")).to(dev).to(torch.float32)

val = [l.strip() for l in open(f"{WORK}/clean_val.txt") if l.strip()]
val = [c for c in val if os.path.exists(c + ".tokens.i32")]
print(f"per-step acceptance on {len(val)} held-out chunks, {N_STEPS} draft steps\n", flush=True)


def argmax_logits(h_rows):
    return (h_rows.to(torch.bfloat16) @ head.t()).argmax(-1)


def run_ckpt(path):
    model = MTPBlock()
    model.load_npz(f"{WORK}/mtp_init.npz")
    if path != "init":
        model.load_state_dict(torch.load(path, map_location="cpu"))
    model.to(dev).eval()

    acc = np.zeros(N_STEPS)
    n = np.zeros(N_STEPS)
    for base in val:
        for s in chunk_samples_multistep(base, MAX_SEQ, N_STEPS):
            e = s["e"].to(dev)
            h = s["h"].to(dev)
            pos = s["pos"].to(dev)
            T = e.shape[0]
            for k in range(1, N_STEPS + 1):
                tgt = argmax_logits(s[f"h_t{k}"].to(dev))
                hs = model(e, h, pos)
                pred = argmax_logits(hs)
                acc[k - 1] += (pred == tgt).sum().item()
                n[k - 1] += T
                if k < N_STEPS:
                    e = tok_embd[pred]
                    h = hs
                    pos = pos + 1
    del model
    torch.cuda.empty_cache()
    return acc / np.maximum(n, 1)


rows = {}
for c in ckpts:
    s = run_ckpt(c)
    rows[os.path.basename(c)] = s

# table
names = list(rows)
w = max(len(x) for x in names) + 2
hdr = "checkpoint".ljust(w) + "".join(f"  s{k+1}   " for k in range(N_STEPS)) + "  E[acc]"
print(hdr)
print("-" * len(hdr))
for name, s in rows.items():
    cum = np.cumprod(s)
    line = name.ljust(w) + "".join(f" {100*s[k]:5.1f}%" for k in range(N_STEPS))
    line += f"  {cum.sum():.3f}"
    print(line, flush=True)
