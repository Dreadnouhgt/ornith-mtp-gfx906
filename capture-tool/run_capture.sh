#!/bin/bash
# v3 hidden-state capture from the production Q8 MTP gguf.
# CRITICAL: -ub must be >= chunk tokens — the capture callback keeps only the
# last ubatch's rows for embd/result_norm (this is why v2 used -ub = chunk).
# Server must already be stopped; does NOT restart it (training follows).
set -e

BUILD=<capture-dir>
OUT=$BUILD/captured-mtp-v3
mkdir -p "$OUT"

cat "$BUILD/corpus_v2.txt" "$BUILD/corpus_v3_long.txt" > "$BUILD/corpus_v3_all.txt"
wc -l "$BUILD/corpus_v3_all.txt"

docker run --rm \
  --device /dev/kfd --device /dev/dri \
  --group-add video --group-add render \
  -v <serving-dir>/models:/models:ro \
  -v "$BUILD":/data \
  -e HIDDEN_CAPTURE_CORPUS=/data/corpus_v3_all.txt \
  -e HIDDEN_CAPTURE_OUT_DIR=/data/captured-mtp-v3 \
  -e HIDDEN_CAPTURE_LAYERS=0 \
  -e HIDDEN_CAPTURE_CHUNK_TOKENS=8192 \
  -e HIDDEN_CAPTURE_MASK_TOKEN_ID=248044 \
  ornith-hidden-capture:local \
  -m /models/q8-mtp-finetuned-v2/ornith-1.0-35b-Q8_0-MTP-finetuned-v2.gguf \
  --device ROCm0,ROCm1 --gpu-layers 999 -c 8320 -b 8320 -ub 8320 --no-mmap

echo "[capture] done: $(ls "$OUT" | grep -c tokens) chunks"

echo "[capture] verifying consistency..."
python3 - <<'EOF'
import glob, os
n_ok = n_bad = 0
for tp in glob.glob("<capture-dir>/captured-mtp-v3/chunk_*.tokens.i32"):
    base = tp[:-len(".tokens.i32")]
    T = os.path.getsize(tp) // 4
    ok = all(os.path.exists(base + s) and os.path.getsize(base + s) == T * 2048 * 4
             for s in (".embd.f32", ".result_norm.f32"))
    n_ok += ok; n_bad += not ok
print(f"CONSISTENCY: {n_ok} ok, {n_bad} bad")
EOF
