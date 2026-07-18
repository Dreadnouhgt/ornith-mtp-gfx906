"""Torch reimplementation of the qwen35moe MTP (nextn) block, replicating
llama.cpp-src/src/models/qwen35moe.cpp::graph_mtp exactly:

  concat(enorm(e), hnorm(h)) -> eh_proj
  -> attn_norm -> gated GQA attention (q/gate interleaved per head, partial
     rope n_rot=64 neox-style, q/k RMS head norms, sigmoid output gate)
  -> +residual -> attn_post_norm
  -> MoE FFN (softmax router over 256 experts, top-8, renormalized) +
     sigmoid-gated shared expert
  -> +residual -> shared_head_norm -> (frozen) lm_head

Trainable: every blk.40 tensor. Frozen: lm_head (output.weight).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_EMBD = 2048
N_HEAD = 16
N_KV_HEAD = 2
HEAD_DIM = 256
N_ROT = 64
FREQ_BASE = 1e7
N_EXPERT = 256
N_EXPERT_USED = 8
FF_EXP = 512
FF_SHEXP = 512
RMS_EPS = 1e-6
VOCAB = 248320


def rms_norm(x, w):
    return w * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + RMS_EPS))


def rope_neox_partial(x, pos):
    # x: [T, H, 256]; rotate first N_ROT dims as pairs (i, i+N_ROT/2)
    half = N_ROT // 2
    inv_freq = FREQ_BASE ** (-torch.arange(0, half, dtype=torch.float32, device=x.device) / half)
    ang = pos.float()[:, None] * inv_freq[None, :]          # [T, half]
    cos, sin = ang.cos()[:, None, :], ang.sin()[:, None, :]  # [T, 1, half]
    x1, x2, rest = x[..., :half], x[..., half:N_ROT], x[..., N_ROT:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos, rest], dim=-1)


class MTPBlock(nn.Module):
    def __init__(self):
        super().__init__()
        p = lambda *shape: nn.Parameter(torch.empty(*shape))
        self.enorm = p(N_EMBD)
        self.hnorm = p(N_EMBD)
        self.eh_proj = p(N_EMBD, 2 * N_EMBD)
        self.attn_norm = p(N_EMBD)
        self.wq = p(N_HEAD * HEAD_DIM * 2, N_EMBD)
        self.wk = p(N_KV_HEAD * HEAD_DIM, N_EMBD)
        self.wv = p(N_KV_HEAD * HEAD_DIM, N_EMBD)
        self.wo = p(N_EMBD, N_HEAD * HEAD_DIM)
        self.q_norm = p(HEAD_DIM)
        self.k_norm = p(HEAD_DIM)
        self.attn_post_norm = p(N_EMBD)
        self.gate_inp = p(N_EXPERT, N_EMBD)
        self.gate_exps = p(N_EXPERT, FF_EXP, N_EMBD)
        self.up_exps = p(N_EXPERT, FF_EXP, N_EMBD)
        self.down_exps = p(N_EXPERT, N_EMBD, FF_EXP)
        self.gate_inp_shexp = p(N_EMBD)
        self.gate_shexp = p(FF_SHEXP, N_EMBD)
        self.up_shexp = p(FF_SHEXP, N_EMBD)
        self.down_shexp = p(N_EMBD, FF_SHEXP)
        self.shared_head_norm = p(N_EMBD)

    @staticmethod
    def gguf_key_map():
        # torch attr -> gguf tensor name
        return {
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

    def load_npz(self, path):
        z = np.load(path)
        sd = {}
        for attr, key in self.gguf_key_map().items():
            arr = torch.from_numpy(np.ascontiguousarray(z[key])).float()
            want = getattr(self, attr).shape
            sd[attr] = arr.reshape(want)
        self.load_state_dict(sd)

    def moe(self, x):
        # router: softmax over all experts, then top-k, renormalize
        probs = F.softmax(x @ self.gate_inp.t(), dim=-1, dtype=torch.float32)
        topw, topi = torch.topk(probs, N_EXPERT_USED, dim=-1)
        topw = topw / topw.sum(-1, keepdim=True)
        out = torch.zeros_like(x)
        flat_i, flat_w = topi.reshape(-1), topw.reshape(-1).to(x.dtype)
        flat_rows = torch.arange(x.shape[0], device=x.device).repeat_interleave(N_EXPERT_USED)
        for e in flat_i.unique():
            m = flat_i == e
            rows = flat_rows[m]
            xe = x[rows]
            h = F.silu(xe @ self.gate_exps[e].t()) * (xe @ self.up_exps[e].t())
            out.index_add_(0, rows, (h @ self.down_exps[e].t()) * flat_w[m, None])
        # shared expert with sigmoid gate
        sh = F.silu(x @ self.gate_shexp.t()) * (x @ self.up_shexp.t())
        sh = (sh @ self.down_shexp.t()) * torch.sigmoid(x @ self.gate_inp_shexp)[:, None]
        return out + sh

    def forward(self, e, h, pos):
        # e: [T, 2048] embedding rows of token t+1; h: [T, 2048] result_norm
        # rows of token t; pos: [T] absolute positions of the e-token.
        x = torch.cat([rms_norm(e, self.enorm), rms_norm(h, self.hnorm)], dim=-1)
        x = x @ self.eh_proj.t()
        res = x

        c = rms_norm(x, self.attn_norm)
        qf = (c @ self.wq.t()).view(-1, N_HEAD, 2 * HEAD_DIM)
        q, gate = qf[..., :HEAD_DIM], qf[..., HEAD_DIM:]
        q = rms_norm(q, self.q_norm)
        k = rms_norm((c @ self.wk.t()).view(-1, N_KV_HEAD, HEAD_DIM), self.k_norm)
        v = (c @ self.wv.t()).view(-1, N_KV_HEAD, HEAD_DIM)

        q = rope_neox_partial(q, pos)
        k = rope_neox_partial(k, pos)

        # GQA: expand kv heads 2 -> 16
        k = k.repeat_interleave(N_HEAD // N_KV_HEAD, dim=1)
        v = v.repeat_interleave(N_HEAD // N_KV_HEAD, dim=1)
        # SDPA wants [1, H, T, D]
        a = F.scaled_dot_product_attention(
            q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None],
            is_causal=True, scale=1.0 / (HEAD_DIM ** 0.5),
        )[0].transpose(0, 1)                                   # [T, H, D]
        a = a * torch.sigmoid(gate)
        x = a.reshape(-1, N_HEAD * HEAD_DIM) @ self.wo.t() + res

        res = x
        c = rms_norm(x, self.attn_post_norm)
        x = self.moe(c) + res

        return rms_norm(x, self.shared_head_norm)              # [T, 2048]
