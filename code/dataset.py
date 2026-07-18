"""Chunk loader for hidden-capture output dirs.

Per capture chunk (see hidden-capture.cpp): tokens.i32 [T], embd.f32 [T,2048],
result_norm.f32 [T,2048]. The last row is the appended mask token and is
dropped. Training rows for positions i in [0, T-3]:
  e[i] = embd[i+1], h[i] = result_norm[i], pos[i] = i+1, target = tokens[i+2]
Long chunks are sliced into windows of at most max_seq rows, keeping true
absolute positions (matters for rope at long context).
"""
import os
import glob

import numpy as np
import torch

N_EMBD = 2048


def iter_chunks(data_dirs):
    for d in data_dirs:
        for tok_path in sorted(glob.glob(os.path.join(d, "chunk_*.tokens.i32"))):
            base = tok_path[: -len(".tokens.i32")]
            if not (os.path.exists(base + ".embd.f32")
                    and os.path.exists(base + ".result_norm.f32")):
                continue
            yield base


def load_chunk(base, drop_mask_row=True):
    tokens = np.fromfile(base + ".tokens.i32", dtype=np.int32)
    embd = np.fromfile(base + ".embd.f32", dtype=np.float32).reshape(-1, N_EMBD)
    rn = np.fromfile(base + ".result_norm.f32", dtype=np.float32).reshape(-1, N_EMBD)
    assert len(tokens) == len(embd) == len(rn), base
    if drop_mask_row:
        tokens, embd, rn = tokens[:-1], embd[:-1], rn[:-1]
    return tokens, embd, rn


def chunk_samples(base, max_seq=4096):
    tokens, embd, rn = load_chunk(base)
    T = len(tokens)
    if T < 8:
        return
    n_rows = T - 2
    for s in range(0, n_rows, max_seq):
        e = min(s + max_seq, n_rows)
        yield {
            "e": torch.from_numpy(embd[s + 1:e + 1].copy()),
            "h": torch.from_numpy(rn[s:e].copy()),
            "pos": torch.arange(s + 1, e + 1, dtype=torch.int32),
            "target": torch.from_numpy(tokens[s + 2:e + 2].astype(np.int64)),
        }
