"""
Converts AEON-7/AEON-DFlash-Qwen3.6-35B-A3B (safetensors) into the native
"dflash" GGUF architecture that llama.cpp's --spec-type draft-dflash loader
expects. Unlike z-lab's drafter, this one is trained specifically against
Ornith-AEON's own weights (base_model_relation: adapter on
AEON-7/Ornith-1.0-35B-AEON-Ultimate-Uncensored-NVFP4), not the generic
un-fine-tuned Qwen3.5/3.6-35B-A3B base -- and reportedly has better
acceptance (3.71 vs 3.35 tokens/step per public benchmarks) than z-lab's.

Differences from z-lab's drafter this script accounts for:
  - 8 layers, all full_attention (no sliding-window layers at all)
  - num_key_value_heads=4 (not 8)
  - target_layer_ids=[1,10,19,28,37] (5 layers, not 8)
  - mask_token_id=248070 (not 248077)
  - YaRN rope scaling (factor=64, beta_fast=32, beta_slow=1,
    original_max_position_embeddings=4096) -- z-lab's drafter used plain
    rope. dflash.cpp's graph code calls ggml_rope_ext with the full YaRN
    parameter set generically, so this should Just Work via the standard
    rope-scaling GGUF keys shared across all architectures.

Reuses Ornith's own tokenizer data (the draft borrows the target's token
embeddings/lm_head at inference time, so its "vocabulary" is the target's).
"""

import gguf
import numpy as np
import torch
from safetensors import safe_open

ORNITH_GGUF = "/models/ornith.gguf"
DFLASH_SAFETENSORS = "/model/model.safetensors"
OUT_PATH = "/model/ornith-dflash-aeon-draft-fixed.gguf"

HIDDEN_SIZE = 2048
N_LAYERS = 8
N_HEADS = 32
N_KV_HEADS = 4
HEAD_DIM = 128
FFN_INTERMEDIATE = 6144
ROPE_THETA = 10000000.0
RMS_EPS = 1e-6
# +1: GGUF stores llama.cpp layer_inp indices = HF target_layer_ids + 1
# (HF reads hidden_states[id+1] = input to layer id+1; llama.cpp taps layer_inp verbatim)
TARGET_LAYER_IDS = [2, 11, 20, 29, 38]
BLOCK_SIZE = 16
MASK_TOKEN_ID = 248070

# YaRN rope scaling, from config.json's rope_scaling block
YARN_FACTOR = 64.0
YARN_BETA_FAST = 32.0
YARN_BETA_SLOW = 1.0
YARN_ORIG_CTX_LEN = 4096

print("=== reading Ornith's tokenizer data ===")
ornith = gguf.GGUFReader(ORNITH_GGUF)
tok_model = ornith.fields["tokenizer.ggml.model"].contents()
tok_pre = ornith.fields["tokenizer.ggml.pre"].contents()
tokens = ornith.fields["tokenizer.ggml.tokens"].contents()
token_types = ornith.fields["tokenizer.ggml.token_type"].contents()
merges = ornith.fields["tokenizer.ggml.merges"].contents()
bos_id = ornith.fields["tokenizer.ggml.bos_token_id"].contents()
eos_id = ornith.fields["tokenizer.ggml.eos_token_id"].contents()
print(f"vocab={len(tokens)} merges={len(merges)} model={tok_model} pre={tok_pre}")

print("=== loading AEON DFlash safetensors ===")
sf = safe_open(DFLASH_SAFETENSORS, framework="pt", device="cpu")


def t(name):
    return sf.get_tensor(name)


writer = gguf.GGUFWriter(OUT_PATH, "dflash")
writer.add_name("Ornith-DFlash-Draft-AEON")

writer.add_uint32("dflash.block_count", N_LAYERS)
writer.add_uint32("dflash.context_length", 262144)
writer.add_uint32("dflash.embedding_length", HIDDEN_SIZE)
writer.add_uint32("dflash.feed_forward_length", FFN_INTERMEDIATE)
writer.add_uint32("dflash.attention.head_count", N_HEADS)
writer.add_uint32("dflash.attention.head_count_kv", N_KV_HEADS)
writer.add_float32("dflash.rope.freq_base", ROPE_THETA)
writer.add_float32("dflash.attention.layer_norm_rms_epsilon", RMS_EPS)
writer.add_uint32("dflash.attention.key_length", HEAD_DIM)
writer.add_uint32("dflash.attention.value_length", HEAD_DIM)
writer.add_array("dflash.target_layers", TARGET_LAYER_IDS)
writer.add_uint32("dflash.block_size", BLOCK_SIZE)

# No sliding window at all -- every layer is full_attention.

# YaRN rope scaling
writer.add_rope_scaling_type(gguf.RopeScalingType.YARN)
writer.add_rope_scaling_factor(YARN_FACTOR)
writer.add_rope_scaling_orig_ctx_len(YARN_ORIG_CTX_LEN)
writer.add_rope_scaling_yarn_beta_fast(YARN_BETA_FAST)
writer.add_rope_scaling_yarn_beta_slow(YARN_BETA_SLOW)

# tokenizer: reuse Ornith's exactly, plus the mask token id the drafter
# was actually trained with (only takes effect because tokenizer_model
# is a real BPE type, not "none" -- llama.cpp's vocab loader skips the
# special-token override block entirely for the "none" tokenizer type).
writer.add_tokenizer_model(tok_model)
writer.add_tokenizer_pre(tok_pre)
writer.add_token_list(tokens)
writer.add_token_types(token_types)
writer.add_token_merges(merges)
writer.add_bos_token_id(bos_id)
writer.add_eos_token_id(eos_id)
writer.add_mask_token_id(MASK_TOKEN_ID)

writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
writer.add_file_type(gguf.LlamaFileType.ALL_F32)


def add(name, tensor):
    arr = tensor.detach().to(torch.float32).numpy().astype(np.float32)
    writer.add_tensor(name, arr)


add("fc.weight", t("fc.weight"))
add("enc.output_norm.weight", t("hidden_norm.weight"))
add("output_norm.weight", t("norm.weight"))

for i in range(N_LAYERS):
    p = f"layers.{i}."
    add(f"blk.{i}.attn_norm.weight", t(p + "input_layernorm.weight"))
    add(f"blk.{i}.attn_q.weight", t(p + "self_attn.q_proj.weight"))
    add(f"blk.{i}.attn_k.weight", t(p + "self_attn.k_proj.weight"))
    add(f"blk.{i}.attn_v.weight", t(p + "self_attn.v_proj.weight"))
    add(f"blk.{i}.attn_output.weight", t(p + "self_attn.o_proj.weight"))
    add(f"blk.{i}.attn_q_norm.weight", t(p + "self_attn.q_norm.weight"))
    add(f"blk.{i}.attn_k_norm.weight", t(p + "self_attn.k_norm.weight"))
    add(f"blk.{i}.ffn_norm.weight", t(p + "post_attention_layernorm.weight"))
    add(f"blk.{i}.ffn_gate.weight", t(p + "mlp.gate_proj.weight"))
    add(f"blk.{i}.ffn_up.weight", t(p + "mlp.up_proj.weight"))
    add(f"blk.{i}.ffn_down.weight", t(p + "mlp.down_proj.weight"))

writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()
writer.close()

print(f"wrote {OUT_PATH}")
