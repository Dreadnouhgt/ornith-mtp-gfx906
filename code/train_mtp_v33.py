"""MTP v3.3: train the head to draft 4 tokens, not 1.

Why: at inference the head drafts autoregressively — llama.cpp
(common/speculative.cpp, draft()) feeds step k+1 the *previous step's drafted
token* plus **the head's own output hidden state** (`embeddings_nextn`), with
position+1. Every checkpoint so far was trained only on step 1, where h is the
target model's real `result_norm`. Steps 2+ feed the head a hidden state it
produced itself and has never been trained to consume — plausibly why
`--spec-draft-n-max 2` measured as the optimum and n-max 3 only won mid-context.

Training replays that exact recurrence, free-running:

    step 1: e = embd[i+1]          h = rn[i]     pos = i+1   -> y1
    step k: e = token_embd[y_{k-1}] h = hs_{k-1}  pos = i+k   -> yk

Targets stay v3.1-style distillation: step k is scored against the target
model's own greedy token at that horizon, argmax(rn[i+k] @ lm_head).

Gradients flow through the whole chain (hs_1 -> h_2 -> hs_2 -> ...), which is
the point — it teaches the head to emit a hidden state its next step can use.
Step weights decay so step 1 (the one that is accepted most often) still
dominates and cannot regress.

token_embd.weight is NOT tied to output.weight in this model (measured: cosine
~0.09), so the real embedding table is required and loaded separately.
"""
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/work")
from mtp_module import MTPBlock
from dataset import iter_chunks
from dataset_multistep import chunk_samples_multistep

WORK = "/work"
DATA_DIRS = [
    "/data/captured-mtp-v2",
    "/data/captured-mtp-v3",
    "/data/captured-mtp-v4a",
    "/data/captured-mtp-v4b",
]
CKPT_DIR = f"{WORK}/checkpoints"
N_STEPS = 4
STEP_W = [1.0, 0.7, 0.5, 0.35]      # step 1 must not regress; deeper steps are bonus
EPOCHS = 2
LR = 6e-6
LR_FLOOR = 0.1
WARMUP_STEPS = 50
GRAD_ACCUM = 4
MAX_SEQ = 8192
LOSS_ROWS = 1024                     # per step, so ~4096 rows/optimizer step total
VAL_FRACTION = 0.05
SEED = 17

os.makedirs(CKPT_DIR, exist_ok=True)
random.seed(SEED)
torch.manual_seed(SEED)

dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {dev} N_STEPS={N_STEPS} MAX_SEQ={MAX_SEQ} LOSS_ROWS={LOSS_ROWS}", flush=True)

model = MTPBlock()
model.load_npz(f"{WORK}/mtp_init.npz")
if os.path.exists(f"{WORK}/resume.pt"):
    model.load_state_dict(torch.load(f"{WORK}/resume.pt", map_location="cpu"))
    print("resumed from resume.pt", flush=True)
model.to(dev)

head = torch.from_numpy(np.load(f"{WORK}/lm_head_f16.npy")).to(dev).to(torch.bfloat16)
head.requires_grad_(False)
tok_embd = torch.from_numpy(np.load(f"{WORK}/token_embd_f16.npy")).to(dev).to(torch.float32)
tok_embd.requires_grad_(False)
print(f"lm_head {tuple(head.shape)}  token_embd {tuple(tok_embd.shape)}", flush=True)

chunks = [c for d in DATA_DIRS if os.path.isdir(d) for c in iter_chunks([d])]
random.shuffle(chunks)
n_val = max(2, int(len(chunks) * VAL_FRACTION))
val_chunks, train_chunks = chunks[:n_val], chunks[n_val:]
print(f"chunks: {len(train_chunks)} train, {len(val_chunks)} val", flush=True)

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.95))
TOTAL_OPT_STEPS = max(1, EPOCHS * len(train_chunks) // GRAD_ACCUM)


def lr_at(s):
    if s < WARMUP_STEPS:
        return LR * (s + 1) / WARMUP_STEPS
    t = min(1.0, (s - WARMUP_STEPS) / max(1, TOTAL_OPT_STEPS - WARMUP_STEPS))
    return LR * (LR_FLOOR + (1 - LR_FLOOR) * 0.5 * (1 + np.cos(np.pi * t)))


@torch.no_grad()
def argmax_logits(h_rows):
    """Full-vocab argmax without building a graph — used for teacher targets
    and for the drafted token fed to the next step."""
    return (h_rows.to(torch.bfloat16) @ head.t()).argmax(-1)


def run_chain(sample, train=True):
    """Replays the inference recurrence for N_STEPS. Returns per-step (loss, n_correct, n)."""
    e = sample["e"].to(dev)
    h = sample["h"].to(dev)
    pos = sample["pos"].to(dev)
    T = e.shape[0]
    out = []

    for k in range(1, N_STEPS + 1):
        tgt = argmax_logits(sample[f"h_t{k}"].to(dev))       # teacher at this horizon
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
            hs = model(e, h, pos)
            if train:
                # subsample only when there is something to subsample; a short
                # chunk must still take THIS branch or the loss has no graph
                if T > LOSS_ROWS:
                    idx = torch.randperm(T, device=dev)[:LOSS_ROWS]
                    hs_l, tgt_l = hs[idx], tgt[idx]
                else:
                    hs_l, tgt_l = hs, tgt
                logits = (hs_l.to(torch.bfloat16) @ head.t()).float()
                loss = F.cross_entropy(logits, tgt_l)
                correct = (logits.argmax(-1) == tgt_l).sum().item()
                n_rows = len(tgt_l)
            else:
                # eval: score every row, chunked to bound the 248k-vocab working set
                loss_sum = 0.0
                correct = 0
                for i in range(0, T, 512):
                    sl = slice(i, min(i + 512, T))
                    lg = (hs[sl].to(torch.bfloat16) @ head.t()).float()
                    loss_sum += F.cross_entropy(lg, tgt[sl], reduction="sum").item()
                    correct += (lg.argmax(-1) == tgt[sl]).sum().item()
                loss = torch.tensor(loss_sum / T, device=dev)
                n_rows = T
        out.append((loss, correct, n_rows))

        if k < N_STEPS:
            # exactly what llama.cpp feeds the next draft step
            y = argmax_logits(hs.detach())
            e = tok_embd[y]                 # drafted token's embedding
            h = hs                          # head's own hidden state (keeps grad)
            pos = pos + 1
    return out


@torch.no_grad()
def evaluate():
    model.eval()
    acc = np.zeros(N_STEPS)
    n = np.zeros(N_STEPS)
    losses = np.zeros(N_STEPS)
    nch = 0
    for base in val_chunks:
        for s in chunk_samples_multistep(base, MAX_SEQ, N_STEPS):
            for k, (loss, correct, cnt) in enumerate(run_chain(s, train=False)):
                acc[k] += correct
                n[k] += cnt
                losses[k] += loss.item()
            nch += 1
    model.train()
    per = " ".join(f"s{k+1}={100.0 * acc[k] / max(n[k], 1):.1f}%" for k in range(N_STEPS))
    return losses / max(nch, 1), per, acc / np.maximum(n, 1)


losses, per, accs = evaluate()
print(f"init: per-step teacher acc {per}  (loss s1={losses[0]:.4f})", flush=True)
best = accs[0] + 0.5 * accs[1:].mean()   # weight step 1, but reward the chain
print(f"init score = {best:.4f}", flush=True)

step = 0
t0 = time.time()
for epoch in range(EPOCHS):
    random.shuffle(train_chunks)
    opt.zero_grad()
    for ci, base in enumerate(train_chunks):
        for s in chunk_samples_multistep(base, MAX_SEQ, N_STEPS):
            per_step = run_chain(s, train=True)
            total = sum(STEP_W[k] * per_step[k][0] for k in range(N_STEPS)) / sum(STEP_W)
            (total / GRAD_ACCUM).backward()
            step += 1
            if step % GRAD_ACCUM == 0:
                for g in opt.param_groups:
                    g["lr"] = lr_at(step // GRAD_ACCUM)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
            if step % 25 == 0:
                accs_now = " ".join(f"s{k+1}={100.0 * per_step[k][1] / per_step[k][2]:.0f}%"
                                    for k in range(N_STEPS))
                print(f"e{epoch} step {step} ({ci}/{len(train_chunks)}) "
                      f"loss={total.item():.4f} {accs_now} "
                      f"lr={opt.param_groups[0]['lr']:.2e} ({time.time()-t0:.0f}s)", flush=True)

    losses, per, accs = evaluate()
    score = accs[0] + 0.5 * accs[1:].mean()
    print(f"epoch {epoch}: per-step teacher acc {per}  score={score:.4f}", flush=True)
    torch.save(model.state_dict(), f"{CKPT_DIR}/mtp_v33_epoch{epoch}.pt")
    if score > best:
        best = score
        torch.save(model.state_dict(), f"{CKPT_DIR}/mtp_v33_best.pt")
        print(f"  new best (score={score:.4f})", flush=True)

print(f"done. best score={best:.4f}", flush=True)
