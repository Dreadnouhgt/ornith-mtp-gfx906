"""MTP v3.2: resume from v3.1 and train on the combined corpus, including the
new 32k-token captures.

vs v3.1:
  - DATA_DIRS adds captured-mtp-v4a/v4b (32,769-token chunks). The old corpus
    topped out near 11k tokens, so the head had never seen rope positions past
    ~8k despite serving at 32k; these cover the full range.
  - MAX_SEQ=32768: one window per long chunk, so attention spans the whole
    document instead of being cut at 4096/8192. Only affordable because flash
    SDPA works at head_dim 256 on Blackwell (0.56 GiB at 32k vs ~68 GB for
    math SDPA) — the exact thing gfx906 could not do.
  - Flash is forced, not merely hoped for: a silent fallback to math SDPA at
    32k would OOM the box rather than run slowly.
  - LOSS_ROWS back on (4096). Attention sees every row; only the 248k-vocab
    head is subsampled, which is what actually costs memory.
  - Validation buckets extended to 32k to match the live acceptance table.

Same distillation target as v3.1: CE against the target model's own greedy
next-next token.
"""
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

sys.path.insert(0, "/work")
from mtp_module import MTPBlock
from dataset import iter_chunks
from dataset_v31 import chunk_samples_v31

WORK = "/work"
DATA_DIRS = [
    "/data/captured-mtp-v2",
    "/data/captured-mtp-v3",
    "/data/captured-mtp-v4a",
    "/data/captured-mtp-v4b",
]
CKPT_DIR = f"{WORK}/checkpoints"
EPOCHS = 2
LR = 8e-6
LR_FLOOR = 0.1
WARMUP_STEPS = 50
GRAD_ACCUM = 4
MAX_SEQ = 32768
LOSS_ROWS = 4096
VAL_FRACTION = 0.05
SEED = 17
POS_BUCKETS = [(0, 512), (512, 2048), (2048, 8192), (8192, 32768)]

os.makedirs(CKPT_DIR, exist_ok=True)
random.seed(SEED)
torch.manual_seed(SEED)

dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {dev} MAX_SEQ={MAX_SEQ} LOSS_ROWS={LOSS_ROWS}", flush=True)

model = MTPBlock()
model.load_npz(f"{WORK}/mtp_init.npz")
if os.path.exists(f"{WORK}/resume.pt"):
    model.load_state_dict(torch.load(f"{WORK}/resume.pt", map_location="cpu"))
    print("resumed from resume.pt", flush=True)
else:
    print("WARNING: no resume.pt — training from the v2 init", flush=True)
model.to(dev)

head = torch.from_numpy(np.load(f"{WORK}/lm_head_f16.npy")).to(dev).to(torch.bfloat16)
head.requires_grad_(False)

chunks = [c for d in DATA_DIRS if os.path.isdir(d) for c in iter_chunks([d])]
random.shuffle(chunks)
n_val = max(2, int(len(chunks) * VAL_FRACTION))
val_chunks, train_chunks = chunks[:n_val], chunks[n_val:]
print(f"chunks: {len(train_chunks)} train, {len(val_chunks)} val "
      f"(from {len([d for d in DATA_DIRS if os.path.isdir(d)])} dirs)", flush=True)

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.95))
TOTAL_OPT_STEPS = max(1, EPOCHS * len(train_chunks) // GRAD_ACCUM)


def lr_at(opt_step):
    if opt_step < WARMUP_STEPS:
        return LR * (opt_step + 1) / WARMUP_STEPS
    t = min(1.0, (opt_step - WARMUP_STEPS) / max(1, TOTAL_OPT_STEPS - WARMUP_STEPS))
    return LR * (LR_FLOOR + (1 - LR_FLOOR) * 0.5 * (1 + np.cos(np.pi * t)))


@torch.no_grad()
def teacher_targets(h_next):
    return (h_next.to(torch.bfloat16) @ head.t()).argmax(-1)


def run_batch(sample, train=True):
    e = sample["e"].to(dev)
    h = sample["h"].to(dev)
    pos = sample["pos"].to(dev)
    corpus_tgt = sample["target"].to(dev)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
        tgt = teacher_targets(sample["h_next"].to(dev))
        # force flash: math SDPA would allocate ~68 GB per copy at 32k
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            hs = model(e, h, pos)
        keep = None
        if train and len(tgt) > LOSS_ROWS:
            keep = torch.randperm(len(tgt), device=dev)[:LOSS_ROWS]
            hs, tgt_l, ctgt_l, pos_l = hs[keep], tgt[keep], corpus_tgt[keep], pos[keep]
        else:
            tgt_l, ctgt_l, pos_l = tgt, corpus_tgt, pos
        logits = (hs.to(torch.bfloat16) @ head.t()).float()
        loss = F.cross_entropy(logits, tgt_l)
    pred = logits.argmax(-1)
    return loss, pred, tgt_l, ctgt_l, pos_l


@torch.no_grad()
def evaluate():
    model.eval()
    tot_loss = tot_n = 0
    m_t = np.zeros(len(POS_BUCKETS))
    m_c = np.zeros(len(POS_BUCKETS))
    n_b = np.zeros(len(POS_BUCKETS))
    for base in val_chunks:
        for s in chunk_samples_v31(base, MAX_SEQ):
            loss, pred, tgt, ctgt, pos = run_batch(s, train=False)
            tot_loss += loss.item() * len(pred)
            tot_n += len(pred)
            for bi, (lo, hi) in enumerate(POS_BUCKETS):
                mask = (pos >= lo) & (pos < hi)
                n = mask.sum().item()
                if n == 0:
                    continue
                n_b[bi] += n
                m_t[bi] += (pred[mask] == tgt[mask]).sum().item()
                m_c[bi] += (pred[mask] == ctgt[mask]).sum().item()
    model.train()
    tt = 100.0 * m_t.sum() / max(n_b.sum(), 1)
    tc = 100.0 * m_c.sum() / max(n_b.sum(), 1)
    per = " ".join(
        f"[{lo}-{hi}) t={100.0 * m_t[bi] / max(n_b[bi], 1):.1f}%/c={100.0 * m_c[bi] / max(n_b[bi], 1):.1f}%(n={int(n_b[bi])})"
        for bi, (lo, hi) in enumerate(POS_BUCKETS))
    return tot_loss / max(tot_n, 1), tt, tc, per


val_loss, val_t, val_c, buckets = evaluate()
print(f"init: val_loss={val_loss:.4f} top1_teacher={val_t:.1f}% "
      f"top1_corpus={val_c:.1f}% {buckets}", flush=True)
best_acc = val_t

step = 0
t0 = time.time()
for epoch in range(EPOCHS):
    random.shuffle(train_chunks)
    opt.zero_grad()
    for ci, base in enumerate(train_chunks):
        for s in chunk_samples_v31(base, MAX_SEQ):
            loss, pred, tgt, ctgt, _ = run_batch(s)
            (loss / GRAD_ACCUM).backward()
            step += 1
            if step % GRAD_ACCUM == 0:
                for g in opt.param_groups:
                    g["lr"] = lr_at(step // GRAD_ACCUM)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
            if step % 25 == 0:
                t1t = (pred == tgt).float().mean().item()
                print(f"e{epoch} step {step} ({ci}/{len(train_chunks)}) "
                      f"loss={loss.item():.4f} t1_teacher={100*t1t:.1f}% "
                      f"rows={len(pred)} lr={opt.param_groups[0]['lr']:.2e} "
                      f"({time.time()-t0:.0f}s)", flush=True)

    val_loss, val_t, val_c, buckets = evaluate()
    print(f"epoch {epoch}: val_loss={val_loss:.4f} top1_teacher={val_t:.1f}% "
          f"top1_corpus={val_c:.1f}% {buckets}", flush=True)
    torch.save(model.state_dict(), f"{CKPT_DIR}/mtp_v32_epoch{epoch}.pt")
    if val_t > best_acc:
        best_acc = val_t
        torch.save(model.state_dict(), f"{CKPT_DIR}/mtp_v32_best.pt")
        print(f"  new best ({val_t:.1f}%)", flush=True)

print(f"done. best val top1_teacher={best_acc:.1f}%", flush=True)
