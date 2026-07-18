"""Write a trained v3 checkpoint back into a COPY of the serving gguf,
in place: every blk.40 tensor is re-quantized to its existing type (Q8_0
mats, F32 norms), so byte sizes and offsets are unchanged.

Run inside the llama.cpp image (has gguf-py):
docker run --rm -v <models-dir>:/models \
  -v <work-dir>:/work \
  --entrypoint python3 mixa3607/llama.cpp-gfx906:b10043-rocm-7.2.4 \
  /work/export_patch_gguf.py /work/checkpoints/mtp_v3_best.pt \
  /models/q8-mtp-finetuned-v3/ornith-1.0-35b-Q8_0-MTP-finetuned-v3.gguf

The target file must already exist (cp the v2 gguf first). Requires torch
absent: checkpoint is loaded via numpy (torch.save with _use_new_zipfile
works through numpy only if saved as npz), so train_mtp_v3.py checkpoints
are converted with ckpt_to_npz.py first if torch is unavailable here.
"""
import sys

sys.path.insert(0, "/app/gguf-py")
import numpy as np
from gguf import GGUFReader
from gguf.quants import quantize

CKPT = sys.argv[1]
TARGET = sys.argv[2]

# attr -> gguf name (mirrors mtp_module.gguf_key_map)
KEY_MAP = {
    "enorm": "blk.40.nextn.enorm.weight",
    "hnorm": "blk.40.nextn.hnorm.weight",
    "eh_proj": "blk.40.nextn.eh_proj.weight",
    "attn_norm": "blk.40.attn_norm.weight",
    "wq": "blk.40.attn_q.weight",
    "wk": "blk.40.attn_k.weight",
    "wv": "blk.40.attn_v.weight",
    "wo": "blk.40.attn_output.weight",
    "q_norm": "blk.40.attn_q_norm.weight",
    "k_norm": "blk.40.attn_k_norm.weight",
    "attn_post_norm": "blk.40.post_attention_norm.weight",
    "gate_inp": "blk.40.ffn_gate_inp.weight",
    "gate_exps": "blk.40.ffn_gate_exps.weight",
    "up_exps": "blk.40.ffn_up_exps.weight",
    "down_exps": "blk.40.ffn_down_exps.weight",
    "gate_inp_shexp": "blk.40.ffn_gate_inp_shexp.weight",
    "gate_shexp": "blk.40.ffn_gate_shexp.weight",
    "up_shexp": "blk.40.ffn_up_shexp.weight",
    "down_shexp": "blk.40.ffn_down_shexp.weight",
    "shared_head_norm": "blk.40.nextn.shared_head_norm.weight",
}

if CKPT.endswith(".npz"):
    state = dict(np.load(CKPT).items())
else:
    import torch
    state = {k: v.float().numpy() for k, v in torch.load(CKPT, map_location="cpu").items()}

r = GGUFReader(TARGET, "r+")
by_name = {t.name: t for t in r.tensors}

for attr, name in KEY_MAP.items():
    t = by_name[name]
    arr = state[attr].astype(np.float32)
    # gguf-py expects the array in its stored (numpy) orientation
    arr = arr.reshape([int(d) for d in reversed(t.shape)][::-1] if arr.ndim > 1 else arr.shape)
    raw = quantize(arr, t.tensor_type)
    flat = raw.view(t.data.dtype).reshape(t.data.shape)
    t.data[:] = flat
    print(f"patched {name} ({t.tensor_type.name}, {t.data.nbytes} bytes)")

print(f"done: {TARGET}")
