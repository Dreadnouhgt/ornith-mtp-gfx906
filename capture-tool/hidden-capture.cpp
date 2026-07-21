// Dumps intermediate per-layer hidden states ("l_out-{layer}") plus the
// corresponding token ids to disk, for building DSpark/DFlash drafter
// training data.
//
// Config via environment variables (kept out of the CLI arg parser, mirroring
// the DSPARK_DUMP_AUX pattern used elsewhere in this project):
//   HIDDEN_CAPTURE_CORPUS      path to a UTF-8 text file, one document per line
//   HIDDEN_CAPTURE_OUT_DIR     output directory (must exist)
//   HIDDEN_CAPTURE_LAYERS      comma-separated layer indices (default "3,11,19,27,39")
//   HIDDEN_CAPTURE_CHUNK_TOKENS max tokens per document chunk (default 2048)
//   HIDDEN_CAPTURE_MASK_TOKEN_ID if set, appended as an extra token at the end
//                              of every chunk (causal attention means this
//                              cannot affect earlier positions), so its raw
//                              pre-layer0 embedding falls out "for free" at
//                              the last row of chunk_N.embd.f32.
//
// All other args (-m, --gpu-layers, --device, -c, ...) are the normal
// llama.cpp common_params flags.
//
// Output format, per chunk N:
//   <out>/chunk_N.tokens.i32      int32[n_tokens]   token ids (includes the
//                                 appended mask token as the last entry, if enabled)
//   <out>/chunk_N.layer_<il>.f32  float32[n_tokens * n_embd]  hidden states, row-major [token][dim]
//   <out>/chunk_N.embd.f32        float32[n_tokens * n_embd]  raw pre-layer0
//                                 token embeddings (real id_last embeddings
//                                 for every position, plus the mask token's
//                                 embedding at the last row if enabled)

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

struct capture_state {
    std::vector<int>            target_layers;
    std::map<int, std::vector<float>> layer_data;    // layer -> flat [n_tokens * n_embd]
    // il=-1 named tensors, keyed by exact ggml tensor name (e.g.
    // "model.input_embed" for raw pre-layer0 embeddings, "result_norm" for
    // the target's final post-norm hidden state before its lm_head).
    std::vector<std::string>          extra_names;
    std::map<std::string, std::vector<float>> extra_data;
    int64_t n_embd  = 0;
    int64_t n_tok   = 0;
};

static int parse_layer_from_name(const char * name) {
    // expects "l_out-<N>"
    static const char prefix[] = "l_out-";
    if (strncmp(name, prefix, sizeof(prefix) - 1) != 0) {
        return -1;
    }
    return atoi(name + sizeof(prefix) - 1);
}

// Appends this ubatch's rows to buf rather than replacing them. The callback
// fires once per ubatch; the original replaced the buffer each time, so any
// chunk longer than one ubatch silently kept only its final ubatch. Callers
// clear the buffers between chunks, and llama.cpp emits ubatches in position
// order for a single-sequence causal prefill, so appending reconstructs the
// full chunk and decouples chunk size from -ub.
static void fill_f32_buf(struct ggml_tensor * t, std::vector<float> & buf) {
    const int64_t n_elts = t->ne[0] * t->ne[1];
    const size_t  off    = buf.size();
    buf.resize(off + (size_t) n_elts);
    float * dst = buf.data() + off;

    const bool is_host = ggml_backend_buffer_is_host(t->buffer);
    if (is_host && t->type == GGML_TYPE_F32) {
        memcpy(dst, t->data, (size_t) n_elts * sizeof(float));
    } else if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, dst, 0, (size_t) n_elts * sizeof(float));
    } else {
        std::vector<uint8_t> raw(ggml_nbytes(t));
        if (is_host) {
            memcpy(raw.data(), t->data, raw.size());
        } else {
            ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
        }
        for (int64_t i = 0; i < n_elts; ++i) {
            if (t->type == GGML_TYPE_F16) {
                dst[i] = ggml_fp16_to_fp32(((const ggml_fp16_t *) raw.data())[i]);
            } else if (t->type == GGML_TYPE_BF16) {
                dst[i] = ggml_bf16_to_fp32(((const ggml_bf16_t *) raw.data())[i]);
            } else {
                GGML_ABORT("unsupported hidden-state tensor type for capture");
            }
        }
    }
}

static bool capture_cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * st = (capture_state *) user_data;

    bool is_extra = false;
    for (const auto & name : st->extra_names) {
        if (name == t->name) { is_extra = true; break; }
    }

    int il = is_extra ? -1 : parse_layer_from_name(t->name);
    bool wanted = is_extra;
    if (!wanted && il >= 0) {
        for (int l : st->target_layers) {
            if (l == il) { wanted = true; break; }
        }
    }
    if (!wanted) {
        return false;
    }

    if (ask) {
        return true;
    }

    st->n_embd = t->ne[0];
    st->n_tok += t->ne[1];   // accumulates across ubatches; reset per chunk

    if (is_extra) {
        fill_f32_buf(t, st->extra_data[t->name]);
        return true;
    }

    fill_f32_buf(t, st->layer_data[il]);
    return true;
}

static void write_bin(const std::string & path, const void * data, size_t n_bytes) {
    std::ofstream f(path, std::ios::binary);
    f.write((const char *) data, (std::streamsize) n_bytes);
}

int main(int argc, char ** argv) {
    common_params params;
    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    const char * corpus_path = getenv("HIDDEN_CAPTURE_CORPUS");
    const char * out_dir     = getenv("HIDDEN_CAPTURE_OUT_DIR");
    if (!corpus_path || !out_dir) {
        LOG_ERR("HIDDEN_CAPTURE_CORPUS and HIDDEN_CAPTURE_OUT_DIR must be set\n");
        return 1;
    }
    const char * layers_env = getenv("HIDDEN_CAPTURE_LAYERS");
    std::string layers_str = layers_env ? layers_env : "3,11,19,27,39";
    int chunk_tokens = 2048;
    if (const char * ct = getenv("HIDDEN_CAPTURE_CHUNK_TOKENS")) {
        chunk_tokens = atoi(ct);
    }

    capture_state st;
    {
        std::stringstream ss(layers_str);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            if (!tok.empty()) st.target_layers.push_back(atoi(tok.c_str()));
        }
    }
    LOG_INF("hidden-capture: target layers =");
    for (int l : st.target_layers) LOG_INF(" %d", l);
    LOG_INF("\n");

    const char * ids_env = getenv("HIDDEN_CAPTURE_CORPUS_IS_IDS");
    const bool corpus_is_ids = ids_env && atoi(ids_env) != 0;
    if (corpus_is_ids) {
        LOG_INF("hidden-capture: corpus lines are token ids, skipping tokenization\n");
    }

    int mask_token_id = -1;
    if (const char * mt = getenv("HIDDEN_CAPTURE_MASK_TOKEN_ID")) {
        mask_token_id = atoi(mt);
        st.extra_names.push_back("model.input_embed");
        st.extra_names.push_back("result_norm");
        LOG_INF("hidden-capture: appending mask_token_id=%d to each chunk, capturing raw embeddings + result_norm\n", mask_token_id);
    }

    params.cb_eval           = capture_cb_eval;
    params.cb_eval_user_data = &st;
    params.warmup            = false;

    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init = common_init_from_params(params);
    llama_model   * model = llama_init->model();
    llama_context * ctx   = llama_init->context();
    if (!model || !ctx) {
        LOG_ERR("failed to init model/context\n");
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);

    std::ifstream corpus(corpus_path);
    if (!corpus) {
        LOG_ERR("failed to open corpus file %s\n", corpus_path);
        return 1;
    }

    std::string line;
    int chunk_idx = 0;
    while (std::getline(corpus, line)) {
        if (line.empty()) continue;

        // HIDDEN_CAPTURE_CORPUS_IS_IDS=1: each corpus line is space-separated
        // token ids instead of text — for benching the exact token sequences a
        // live server was measured on (detokenize->retokenize does not
        // round-trip reliably through chat-template special tokens).
        std::vector<llama_token> tokens;
        if (corpus_is_ids) {
            std::stringstream ls(line);
            long v;
            while (ls >> v) tokens.push_back((llama_token) v);
        } else {
            tokens = common_tokenize(ctx, line, add_bos, true);
        }
        if ((int) tokens.size() > chunk_tokens) {
            tokens.resize(chunk_tokens);
        }
        if (tokens.size() < 8) continue; // too short to be useful training data

        // Append the mask token as an extra final position. Causal
        // self-attention means it cannot influence any earlier position's
        // hidden states, so this is free: its raw embedding just falls out
        // at the last row of chunk_N.embd.f32.
        if (mask_token_id >= 0) {
            tokens.push_back((llama_token) mask_token_id);
        }

        llama_memory_clear(llama_get_memory(ctx), true);

        llama_batch batch = llama_batch_init((int) tokens.size(), 0, 1);
        common_batch_clear(batch);
        for (size_t i = 0; i < tokens.size(); ++i) {
            common_batch_add(batch, tokens[i], (llama_pos) i, {0}, true);
        }

        st.layer_data.clear();
        st.extra_data.clear();
        st.n_tok = 0;
        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("decode failed on chunk %d, skipping\n", chunk_idx);
            llama_batch_free(batch);
            continue;
        }
        llama_batch_free(batch);

        if ((int) st.layer_data.size() != (int) st.target_layers.size()) {
            LOG_ERR("chunk %d: expected %zu captured layers, got %zu, skipping\n",
                     chunk_idx, st.target_layers.size(), st.layer_data.size());
            continue;
        }

        // Every captured buffer must hold exactly one row per token. A short
        // buffer means ubatches were dropped rather than accumulated, which
        // would poison training data silently — refuse to write it.
        {
            const size_t want = tokens.size() * (size_t) st.n_embd;
            bool bad = false;
            for (int il : st.target_layers) {
                if (st.layer_data[il].size() != want) {
                    LOG_ERR("chunk %d: layer %d has %zu floats, expected %zu — skipping chunk\n",
                            chunk_idx, il, st.layer_data[il].size(), want);
                    bad = true;
                }
            }
            for (const auto & name : st.extra_names) {
                auto it = st.extra_data.find(name);
                if (it != st.extra_data.end() && it->second.size() != want) {
                    LOG_ERR("chunk %d: extra '%s' has %zu floats, expected %zu — skipping chunk\n",
                            chunk_idx, name.c_str(), it->second.size(), want);
                    bad = true;
                }
            }
            if (bad) continue;
        }

        std::vector<int32_t> tok_i32(tokens.begin(), tokens.end());
        write_bin(std::string(out_dir) + "/chunk_" + std::to_string(chunk_idx) + ".tokens.i32",
                   tok_i32.data(), tok_i32.size() * sizeof(int32_t));

        for (int il : st.target_layers) {
            auto & buf = st.layer_data[il];
            write_bin(std::string(out_dir) + "/chunk_" + std::to_string(chunk_idx) +
                          ".layer_" + std::to_string(il) + ".f32",
                       buf.data(), buf.size() * sizeof(float));
        }

        static const std::map<std::string, std::string> extra_suffix = {
            {"model.input_embed", ".embd.f32"},
            {"result_norm",       ".result_norm.f32"},
        };
        for (const auto & name : st.extra_names) {
            auto it = st.extra_data.find(name);
            if (it == st.extra_data.end() || it->second.empty()) {
                LOG_ERR("chunk %d: expected extra tensor '%s' but none was captured, skipping its write\n",
                         chunk_idx, name.c_str());
                continue;
            }
            write_bin(std::string(out_dir) + "/chunk_" + std::to_string(chunk_idx) + extra_suffix.at(name),
                       it->second.data(), it->second.size() * sizeof(float));
        }

        LOG_INF("chunk %d: %zu tokens, %zu layers captured, n_embd=%lld\n",
                 chunk_idx, tokens.size(), st.layer_data.size(), (long long) st.n_embd);
        chunk_idx++;
    }

    LOG_INF("done: %d chunks written to %s\n", chunk_idx, out_dir);

    llama_backend_free();
    return 0;
}
