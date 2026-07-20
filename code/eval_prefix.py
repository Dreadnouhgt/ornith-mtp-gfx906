"""Simulated speculative acceptance: free-run the draft chain and measure the
ACTUAL accepted-prefix distribution per row (accept until first mismatch, like
greedy spec decode), instead of assuming per-step independence.

Reports, per checkpoint:
  - P(prefix >= k) for k=1..N
  - E[accepted | n-max=n] for each n — the number that picks the serving n-max
  - tokens/cycle = E[accepted]+1 (the +1 is the target's own token per cycle)

Usage: eval_prefix.py <ckptA,ckptB,...> [n_steps]
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/work")
from mtp_module_fast import MTPBlockFast
from dataset_multistep import chunk_samples_multistep

WORK = "/work"
MAX_SEQ = 8192
N = int(sys.argv[2]) if len(sys.argv) > 2 else 4
ckpts = sys.argv[1].split(",")

torch.set_grad_enabled(False)
dev = "cuda"

head = torch.from_numpy(np.load(f"{WORK}/lm_head_f16.npy")).to(dev).to(torch.bfloat16)
tok_embd = torch.from_numpy(np.load(f"{WORK}/token_embd_f16.npy")).to(dev).to(torch.float32)

val = [l.strip() for l in open(f"{WORK}/clean_val.txt") if l.strip()]
val = [c for c in val if os.path.exists(c + ".tokens.i32")]
print(f"prefix acceptance on {len(val)} held-out chunks, {N} draft steps\n", flush=True)


def am(h_rows):
    return (h_rows.to(torch.bfloat16) @ head.t()).argmax(-1)


for path in ckpts:
    model = MTPBlockFast()
    model.load_npz(f"{WORK}/mtp_init.npz")
    if path != "init":
        model.load_state_dict(torch.load(path, map_location="cpu"))
    model.to(dev).eval()

    # per-row correctness matrix -> prefix lengths
    tot = np.zeros(N + 1)   # tot[k] = #rows with accepted prefix exactly k
    n_rows = 0
    for base in val:
        for s in chunk_samples_multistep(base, MAX_SEQ, N):
            e = s["e"].to(dev)
            h = s["h"].to(dev)
            pos = s["pos"].to(dev)
            T = e.shape[0]
            ok = torch.ones(T, dtype=torch.bool, device=dev)
            plen = torch.zeros(T, dtype=torch.int64, device=dev)
            for k in range(1, N + 1):
                tgt = am(s[f"h_t{k}"].to(dev))
                hs = model(e, h, pos)
                pred = am(hs)
                ok = ok & (pred == tgt)
                plen += ok.long()
                if k < N:
                    e = tok_embd[pred]
                    h = hs
                    pos = pos + 1
            cnt = torch.bincount(plen, minlength=N + 1).cpu().numpy()
            tot += cnt
            n_rows += T

    p_exact = tot / n_rows
    p_ge = 1.0 - np.cumsum(np.concatenate([[0.0], p_exact]))[:-1]  # P(prefix>=k), k=0..N
    name = os.path.basename(path)
    print(f"=== {name} ({n_rows} rows)")
    print("  P(prefix>=k): " + " ".join(f"k{k}={100*p_ge[k]:.1f}%" for k in range(1, N + 1)))
    # E[accepted | n-max=n] = sum_{k=1..n} P(prefix>=k)
    for n in range(1, N + 1):
        e_acc = p_ge[1:n + 1].sum()
        print(f"  n-max={n}: E[accepted]={e_acc:.3f}  tokens/cycle={1 + e_acc:.3f}")
    print(flush=True)
    del model
    torch.cuda.empty_cache()
