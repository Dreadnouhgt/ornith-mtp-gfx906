"""Extract the MTP block (blk.40.* + nextn) plus the frozen LM head from the
production Q8 gguf into an npz for the v3 trainer. Run inside the llama.cpp
image (has gguf-py):

docker run --rm -v <models-dir>:/models:ro \
  -v <work-dir>:/out \
  --entrypoint python3 mixa3607/llama.cpp-gfx906:b10043-rocm-7.2.4 \
  /out/extract_mtp_weights.py
"""
import sys

sys.path.insert(0, "/app/gguf-py")
import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize

GGUF = "/models/q8-mtp-finetuned-v2/ornith-1.0-35b-Q8_0-MTP-finetuned-v2.gguf"
OUT = "/out/mtp_init.npz"
OUT_HEAD = "/out/lm_head_f16.npy"

r = GGUFReader(GGUF)

mtp = {}
head = None
for t in r.tensors:
    if t.name.startswith("blk.40."):
        arr = dequantize(t.data, t.tensor_type).astype(np.float32)
        # gguf stores ne = [in, out]; numpy arrives as [out, in] row-major
        mtp[t.name] = arr
        print(f"{t.name:55s} {arr.shape} {t.tensor_type.name}")
    elif t.name == "output.weight":
        head = dequantize(t.data, t.tensor_type).astype(np.float16)
        print(f"{t.name:55s} {head.shape} {t.tensor_type.name} (frozen head)")

assert head is not None, "no output.weight found"
assert len(mtp) == 20, f"expected 20 blk.40 tensors, got {len(mtp)}"

np.savez(OUT, **mtp)
np.save(OUT_HEAD, head)
print(f"wrote {OUT} and {OUT_HEAD}")
