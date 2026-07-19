"""MTP v3.1: resume from mtp_v3_best.pt with three changes vs train_mtp_v3.py:

1. Distillation targets: CE against the TARGET MODEL's own greedy next-next
   token (argmax of result_norm[i+1] @ lm_head), not the corpus token. Live
   acceptance compares the draft to the target's token, so this optimizes the
   deployed metric directly. Corpus-token top-1 is still logged for A/B.
2. Full-fidelity windows: MAX_SEQ=8192 (v3 capture max — no window slicing at
   all on this corpus), no LOSS_ROWS subsampling. Both were 32GB-card
   compromises; GB10 unified memory doesn't need them.
3. Positional-bucket validation (<512 / <2048 / <8192) so the numbers map onto
   the live acceptance-by-context-depth table, plus cosine LR decay.

Mounts as train_mtp_v3.py: /work = code + mtp_init.npz + lm_head_f16.npy
(+ resume.pt), /data = captures parent. Checkpoints to /work/checkpoints/.
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
from dataset_v31 import chunk_samples_v31

WORK = "/work"
DATA_DIRS = ["/data/captured-mtp-v2", "/data/captured-mtp-v3"]
CKPT_DIR = f"{WORK}/checkpoints"
EPOCHS = 2
LR = 1e-5           # resuming a converged head: half of v3's 2e-5
LR_FLOOR = 0.1      # cosine decays to LR * LR_FLOOR
WARMUP_STEPS = 50
GRAD_ACCUM = 4
MAX_SEQ = 8192
VAL_FRACTION = 0.05
SEED = 17           # same seed as v3 -> same train/val chunk split
POS_BUCKETS = [(0, 512), (512, 2048), (2048, 8192)]

os.makedirs(CKPT_DIR, exist_ok=True)
random.seed(SEED)
torch.manual_seed(SEED)

dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {dev}", flush=True)

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

chunks = list(iter_chunks(DATA_DIRS))
random.shuffle(chunks)
n_val = max(2, int(len(chunks) * VAL_FRACTION))
val_chunks, train_chunks = chunks[:n_val], chunks[n_val:]
print(f"chunks: {len(train_chunks)} train, {len(val_chunks)} val", flush=True)

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.95))

# optimizer steps per epoch for the cosine schedule (windows / GRAD_ACCUM,
# approximated as one window per chunk at MAX_SEQ=8192 — capture max is 8192)
TOTAL_OPT_STEPS = max(1, EPOCHS * len(train_chunks) // GRAD_ACCUM)


def lr_at(opt_step):
    if opt_step < WARMUP_STEPS:
        return LR * (opt_step + 1) / WARMUP_STEPS
    t = min(1.0, (opt_step - WARMUP_STEPS) / max(1, TOTAL_OPT_STEPS - WARMUP_STEPS))
    return LR * (LR_FLOOR + (1 - LR_FLOOR) * 0.5 * (1 + np.cos(np.pi * t)))


@torch.no_grad()
def teacher_targets(h_next):
    """Target model's greedy next-next token from its own hidden state."""
    return (h_next.to(torch.bfloat16) @ head.t()).argmax(-1)


def run_batch(sample, train=True):
    e = sample["e"].to(dev)
    h = sample["h"].to(dev)
    pos = sample["pos"].to(dev)
    corpus_tgt = sample["target"].to(dev)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
        tgt = teacher_targets(sample["h_next"].to(dev))
        hs = model(e, h, pos)
        logits = (hs.to(torch.bfloat16) @ head.t()).float()
        loss = F.cross_entropy(logits, tgt)
    pred = logits.argmax(-1)
    top1_teacher = (pred == tgt).float().mean().item()
    top1_corpus = (pred == corpus_tgt).float().mean().item()
    return loss, top1_teacher, top1_corpus, pred, tgt, corpus_tgt, pos


@torch.no_grad()
def evaluate():
    model.eval()
    tot_loss = tot_n = 0
    m_teacher = np.zeros(len(POS_BUCKETS))
    m_corpus = np.zeros(len(POS_BUCKETS))
    n_bucket = np.zeros(len(POS_BUCKETS))
    for base in val_chunks:
        for s in chunk_samples_v31(base, MAX_SEQ):
            loss, _, _, pred, tgt, ctgt, pos = run_batch(s, train=False)
            tot_loss += loss.item() * len(pred)
            tot_n += len(pred)
            for bi, (lo, hi) in enumerate(POS_BUCKETS):
                mask = (pos >= lo) & (pos < hi)
                n = mask.sum().item()
                if n == 0:
                    continue
                n_bucket[bi] += n
                m_teacher[bi] += (pred[mask] == tgt[mask]).sum().item()
                m_corpus[bi] += (pred[mask] == ctgt[mask]).sum().item()
    model.train()
    tot_teacher = 100.0 * m_teacher.sum() / max(n_bucket.sum(), 1)
    tot_corpus = 100.0 * m_corpus.sum() / max(n_bucket.sum(), 1)
    per_bucket = " ".join(
        f"[{lo}-{hi}) t={100.0 * m_teacher[bi] / max(n_bucket[bi], 1):.1f}%"
        f"/c={100.0 * m_corpus[bi] / max(n_bucket[bi], 1):.1f}%"
        for bi, (lo, hi) in enumerate(POS_BUCKETS))
    return tot_loss / max(tot_n, 1), tot_teacher, tot_corpus, per_bucket


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
            loss, t1t, t1c, *_ = run_batch(s)
            (loss / GRAD_ACCUM).backward()
            step += 1
            if step % GRAD_ACCUM == 0:
                for g in opt.param_groups:
                    g["lr"] = lr_at(step // GRAD_ACCUM)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
            if step % 50 == 0:
                print(f"e{epoch} step {step} ({ci}/{len(train_chunks)} chunks) "
                      f"loss={loss.item():.4f} t1_teacher={100*t1t:.1f}% "
                      f"t1_corpus={100*t1c:.1f}% lr={opt.param_groups[0]['lr']:.2e} "
                      f"({time.time()-t0:.0f}s)", flush=True)

    val_loss, val_t, val_c, buckets = evaluate()
    print(f"epoch {epoch}: val_loss={val_loss:.4f} top1_teacher={val_t:.1f}% "
          f"top1_corpus={val_c:.1f}% {buckets}", flush=True)
    torch.save(model.state_dict(), f"{CKPT_DIR}/mtp_v31_epoch{epoch}.pt")
    if val_t > best_acc:
        best_acc = val_t
        torch.save(model.state_dict(), f"{CKPT_DIR}/mtp_v31_best.pt")
        print(f"  new best ({val_t:.1f}%)", flush=True)

print(f"done. best val top1_teacher={best_acc:.1f}%", flush=True)
