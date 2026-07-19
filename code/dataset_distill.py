"""v3.1 chunk loader: same layout as dataset.py, plus h_next (result_norm[i+1])
so the trainer can distill against the target model's own next-next prediction
instead of the corpus token. See dataset.py for the base row convention.
"""
import sys

sys.path.insert(0, "/work")
import torch

from dataset import load_chunk


def chunk_samples_v31(base, max_seq=8192):
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
            # teacher state: what the target itself predicts for token i+2
            "h_next": torch.from_numpy(rn[s + 1:e + 1].copy()),
            "pos": torch.arange(s + 1, e + 1, dtype=torch.int32),
            "target": torch.from_numpy(tokens[s + 2:e + 2].astype("int64")),
        }
