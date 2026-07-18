"""MTP v3 finetune: continue training the blk.40 nextn block on Q8-captured
hidden states (v2 short chunks + v3 long chunks), initialized from the
weights extracted out of the serving Q8 gguf.

Run inside pytorch-gfx906 with:
  -v mtp-train-v3:/work -v hidden-capture-build:/data
  python3 /work/train_mtp_v3.py

Loss: cross-entropy against the next-next token through the frozen lm_head.
Only blk.40 tensors train. Checkpoints to /work/checkpoints/.
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
from dataset import iter_chunks, chunk_samples

WORK = "/work"
DATA_DIRS = ["/data/captured-mtp-v2", "/data/captured-mtp-v3"]
CKPT_DIR = f"{WORK}/checkpoints"
EPOCHS = 2
LR = 2e-5
WARMUP_STEPS = 100
GRAD_ACCUM = 4
MAX_SEQ = 4096
VAL_FRACTION = 0.05
SEED = 17

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
model.to(dev)

head = torch.from_numpy(np.load(f"{WORK}/lm_head_f16.npy")).to(dev).to(torch.bfloat16)
head.requires_grad_(False)

chunks = list(iter_chunks(DATA_DIRS))
random.shuffle(chunks)
n_val = max(2, int(len(chunks) * VAL_FRACTION))
val_chunks, train_chunks = chunks[:n_val], chunks[n_val:]
print(f"chunks: {len(train_chunks)} train, {len(val_chunks)} val", flush=True)

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.95))
sched_step = 0


def lr_at(step):
    if step < WARMUP_STEPS:
        return LR * (step + 1) / WARMUP_STEPS
    return LR


LOSS_ROWS = 1536  # cap rows through the 248k-vocab head; full T is ~12GB of
                  # logits+logsoftmax+grad and OOMs — attention still sees all T


def run_batch(sample, train=True):
    e = sample["e"].to(dev)
    h = sample["h"].to(dev)
    pos = sample["pos"].to(dev)
    tgt = sample["target"].to(dev)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
        hs = model(e, h, pos)
        if train and len(tgt) > LOSS_ROWS:
            idx = torch.randperm(len(tgt), device=dev)[:LOSS_ROWS]
            hs, tgt = hs[idx], tgt[idx]
        logits = (hs.to(torch.bfloat16) @ head.t()).float()
        loss = F.cross_entropy(logits, tgt)
    top1 = (logits.argmax(-1) == tgt).float().mean().item()
    return loss, top1, len(tgt)


@torch.no_grad()
def evaluate():
    model.eval()
    tot_loss = tot_match = tot_n = 0
    for base in val_chunks:
        for s in chunk_samples(base, MAX_SEQ):
            loss, top1, n = run_batch(s, train=False)
            tot_loss += loss.item() * n
            tot_match += top1 * n
            tot_n += n
    model.train()
    return tot_loss / max(tot_n, 1), 100.0 * tot_match / max(tot_n, 1)


val_loss, val_acc = evaluate()
print(f"init: val_loss={val_loss:.4f} val_top1={val_acc:.1f}%", flush=True)
best_acc = val_acc

step = 0
t0 = time.time()
for epoch in range(EPOCHS):
    random.shuffle(train_chunks)
    opt.zero_grad()
    for ci, base in enumerate(train_chunks):
        samples = list(chunk_samples(base, MAX_SEQ))
        for s in samples:
            loss, top1, n = run_batch(s)
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
                      f"loss={loss.item():.4f} top1={100*top1:.1f}% "
                      f"({time.time()-t0:.0f}s)", flush=True)

    val_loss, val_acc = evaluate()
    print(f"epoch {epoch}: val_loss={val_loss:.4f} val_top1={val_acc:.1f}%", flush=True)
    torch.save(model.state_dict(), f"{CKPT_DIR}/mtp_v3_epoch{epoch}.pt")
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), f"{CKPT_DIR}/mtp_v3_best.pt")
        print(f"  new best ({val_acc:.1f}%)", flush=True)

print(f"done. best val_top1={best_acc:.1f}%", flush=True)
