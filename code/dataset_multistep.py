"""Chunk loader for multi-step draft training.

Step 1 is the normal MTP row. Steps 2..K replay what llama.cpp actually does
when drafting more than one token (common/speculative.cpp, draft()):

    token = the previous step's *drafted* token
    h     = the head's own output hidden state (embeddings_nextn)
    pos   = pos + 1

so the only per-step data needed from disk is the teacher state for each
horizon: rn[i+k] gives the target model's own prediction of token i+k+1.
"""
import sys

sys.path.insert(0, "/work")
import numpy as np
import torch

from dataset import load_chunk


def chunk_samples_multistep(base, max_seq=8192, n_steps=4):
    tokens, embd, rn = load_chunk(base)
    T = len(tokens)
    # need rn[i+n_steps] and tokens[i+1+n_steps] in range
    n_rows = T - (n_steps + 1)
    if n_rows < 8:
        return
    for s in range(0, n_rows, max_seq):
        e = min(s + max_seq, n_rows)
        out = {
            "e": torch.from_numpy(embd[s + 1:e + 1].copy()),   # step-1 token embedding
            "h": torch.from_numpy(rn[s:e].copy()),             # step-1 hidden state
            "pos": torch.arange(s + 1, e + 1, dtype=torch.int32),
        }
        # teacher hidden state per horizon: rn[i+k] predicts token i+k+1
        for k in range(1, n_steps + 1):
            out[f"h_t{k}"] = torch.from_numpy(rn[s + k:e + k].copy())
        return_val = out
        yield return_val
