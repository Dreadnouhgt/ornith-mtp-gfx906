"""Graft an MTP donor head GGUF onto a base Ornith GGUF.

The donor files carry only the blk.40 tensors (no model hyperparams), so they
cannot load as a standalone -md draft model — llama.cpp draft-mtp expects the
head embedded in the serving GGUF. This writes base + donor tensors into a new
file, copying all KV metadata and raw (still-quantized) tensor data.

Usage: graft_mtp_head.py <base.gguf> <head.gguf> <out.gguf>
"""
import sys

from gguf import GGUFReader, GGUFWriter

base_p, head_p, out_p = sys.argv[1], sys.argv[2], sys.argv[3]

base = GGUFReader(base_p)
head = GGUFReader(head_p)

arch = None
for f in base.fields.values():
    if f.name == "general.architecture":
        arch = str(bytes(f.parts[f.data[0]]), "utf-8")
print(f"base arch: {arch}, {len(base.tensors)} tensors; head: {len(head.tensors)} tensors")

w = GGUFWriter(out_p, arch)

# copy every KV except the ones GGUFWriter writes itself
skip = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count", "general.architecture"}
have_nextn = False
for f in base.fields.values():
    if f.name in skip:
        continue
    if f.name == f"{arch}.nextn_predict_layers":
        have_nextn = True
    if f.name == f"{arch}.block_count" and not any(
            t.name.startswith(f"blk.{f.contents()-1}.nextn") for t in base.tensors):
        # nextn lives at blk.<block_count-1>; base counts only the 40 main
        # layers, so the grafted head at blk.40 needs block_count = 41
        v = f.contents() + 1
        print(f"bumping {f.name}: {f.contents()} -> {v}")
        w.add_key_value(f.name, v, f.types[0])
        continue
    w.add_key_value(f.name, f.contents(), f.types[0])
if not have_nextn:
    # the loader only consumes blk.<last> nextn tensors if this key says so
    from gguf import GGUFValueType
    w.add_key_value(f"{arch}.nextn_predict_layers", 1, GGUFValueType.UINT32)
    print(f"added {arch}.nextn_predict_layers = 1")

def add_raw(t):
    # raw_shape = the data's byte-shape; gguf-py derives the logical shape
    # from it + raw_dtype (same pattern as gguf_new_metadata.py)
    w.add_tensor(t.name, t.data, raw_shape=t.data.shape, raw_dtype=t.tensor_type)

names_head = {t.name for t in head.tensors}
n_replaced = 0
for t in base.tensors:
    if t.name in names_head:
        n_replaced += 1
        continue
    add_raw(t)
for t in head.tensors:
    add_raw(t)
print(f"grafted {len(head.tensors)} head tensors ({n_replaced} replaced existing)")

w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print(f"done: {out_p}")
