"""Verify MTPBlockFast == MTPBlock on real weights + real data, then benchmark
forward and forward+backward for both, mirroring the v3.1/v3.3 training setup
(bf16 autocast, real chunk shapes).
"""
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/work")
from mtp_module import MTPBlock
from mtp_module_fast import MTPBlockFast
from dataset import load_chunk

WORK = "/work"
dev = "cuda"

ref = MTPBlock(); ref.load_npz(f"{WORK}/mtp_init.npz"); ref.to(dev)
fast = MTPBlockFast(); fast.load_state_dict(ref.state_dict()); fast.to(dev)

tokens, embd, rn = load_chunk("/data/captured-mtp-v3/chunk_0")
T = min(4096, len(tokens) - 2)
e = torch.from_numpy(embd[1:T + 1].copy()).to(dev)
h = torch.from_numpy(rn[:T].copy()).to(dev)
pos = torch.arange(1, T + 1, dtype=torch.int32, device=dev)
print(f"rows: {T}")

# ---- equivalence, eager fp32 (strictest) ----
with torch.no_grad():
    o_ref = ref(e, h, pos)
    o_fast = fast(e, h, pos)
d = (o_ref - o_fast).abs()
cos = torch.nn.functional.cosine_similarity(o_ref, o_fast, dim=-1)
print(f"fp32 eager : max|d|={d.max().item():.3e} mean|d|={d.mean().item():.3e} "
      f"min_cos={cos.min().item():.8f}")

# ---- equivalence under bf16 autocast (training condition) ----
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    o_ref = ref(e, h, pos)
    o_fast = fast(e, h, pos)
cos = torch.nn.functional.cosine_similarity(o_ref.float(), o_fast.float(), dim=-1)
print(f"bf16 amp   : min_cos={cos.min().item():.8f} mean_cos={cos.mean().item():.8f}")

# ---- gradient equivalence (small T for the slow ref backward) ----
Ts = 512
es, hs_, ps = e[:Ts], h[:Ts], pos[:Ts]
for m in (ref, fast):
    m.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = m(es, hs_, ps)
        loss = out.float().pow(2).mean()
    loss.backward()
g_ref = ref.gate_exps.grad
g_fast = fast.gate_exps.grad
gd = (g_ref - g_fast).abs()
print(f"grad gate_exps: max|d|={gd.max().item():.3e} "
      f"ref_norm={g_ref.norm().item():.4f} fast_norm={g_fast.norm().item():.4f}")

# ---- benchmark ----
def bench(model, train, iters=8):
    torch.cuda.synchronize()
    # warmup
    for _ in range(2):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(e, h, pos)
            if train:
                out.float().pow(2).mean().backward()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(e, h, pos)
            if train:
                out.float().pow(2).mean().backward()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1e3

for name, m in (("ref ", ref), ("fast", fast)):
    fwd = bench(m, train=False)
    fb = bench(m, train=True)
    print(f"{name}: forward {fwd:8.1f} ms   fwd+bwd {fb:8.1f} ms   (T={T})")
