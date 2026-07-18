"""Validate the torch reimplementation against the production v2 weights.

Loads blk.40 weights extracted from the serving Q8 gguf, runs captured v2
chunks through the torch module, and reports top-1 agreement between the MTP
prediction and the actual next-next token. If the graph replication is
correct this should land near the measured draft acceptance (~60-85% on
short text); a broken graph lands near zero.

Also tries pos and pos-1 to pin down the rope position convention.

Usage: python3 validate_reimpl.py [n_chunks]
"""
import sys

import numpy as np
import torch

sys.path.insert(0, "/work")
from mtp_module import MTPBlock
from dataset import iter_chunks, chunk_samples

WORK = "/work"
DATA = ["/data/captured-mtp-v2"]
N_CHUNKS = int(sys.argv[1]) if len(sys.argv) > 1 else 3

torch.set_grad_enabled(False)

model = MTPBlock()
model.load_npz(f"{WORK}/mtp_init.npz")
model.eval()

head = torch.from_numpy(np.load(f"{WORK}/lm_head_f16.npy")).float()

for pos_off in (0, -1):
    n_match = n_tot = 0
    for ci, base in enumerate(iter_chunks(DATA)):
        if ci >= N_CHUNKS:
            break
        for s in chunk_samples(base, max_seq=4096):
            hs = model(s["e"], s["h"], s["pos"] + pos_off)
            logits = hs @ head.t()
            pred = logits.argmax(-1)
            n_match += (pred == s["target"]).sum().item()
            n_tot += len(pred)
    print(f"pos_offset={pos_off}: top-1 agreement {n_match}/{n_tot} "
          f"= {100.0 * n_match / max(n_tot, 1):.1f}%")
