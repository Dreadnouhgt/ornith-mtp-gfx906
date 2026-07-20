"""MTPBlock with a fused MoE: identical math, ~3 grouped GEMMs instead of a
256-iteration Python loop (~1000 kernel launches per call).

Route: sort the T*8 (token, expert) pairs by expert, run one grouped GEMM per
projection over the expert-contiguous rows (torch._grouped_mm, group boundaries
from bincount cumsum), scatter-add back with routing weights. Same ops, same
fp32 router softmax, same renormalized top-8 — only the launch pattern changes.

Weights are the same stacked 3D parameters as the reference module, so
checkpoints load interchangeably.
"""
import torch
import torch.nn.functional as F

from mtp_module import MTPBlock, N_EXPERT, N_EXPERT_USED


class MTPBlockFast(MTPBlock):
    def moe(self, x):
        T = x.shape[0]
        probs = F.softmax(x @ self.gate_inp.t(), dim=-1, dtype=torch.float32)
        topw, topi = torch.topk(probs, N_EXPERT_USED, dim=-1)
        topw = topw / topw.sum(-1, keepdim=True)

        flat_i = topi.reshape(-1)                                # [T*8]
        flat_w = topw.reshape(-1).to(x.dtype)
        flat_rows = torch.arange(T, device=x.device).repeat_interleave(N_EXPERT_USED)

        # sort (token,expert) pairs so each expert's rows are contiguous
        order = torch.argsort(flat_i)
        s_rows = flat_rows[order]
        s_w = flat_w[order]
        xe = x[s_rows].contiguous()                              # [T*8, 2048]

        counts = torch.bincount(flat_i, minlength=N_EXPERT)
        offs = counts.cumsum(0).to(torch.int32)                  # group ends

        wdt = self.gate_exps.dtype if not torch.is_autocast_enabled() else torch.get_autocast_dtype("cuda")
        xe = xe.to(wdt)
        gate = torch._grouped_mm(xe, self.gate_exps.transpose(1, 2).to(wdt), offs=offs)
        up = torch._grouped_mm(xe, self.up_exps.transpose(1, 2).to(wdt), offs=offs)
        h = F.silu(gate) * up                                    # [T*8, 512]
        down = torch._grouped_mm(h, self.down_exps.transpose(1, 2).to(wdt), offs=offs)

        out = torch.zeros_like(x)
        out.index_add_(0, s_rows, (down * s_w[:, None]).to(x.dtype))

        # shared expert unchanged
        sh = F.silu(x @ self.gate_shexp.t()) * (x @ self.up_shexp.t())
        sh = (sh @ self.down_shexp.t()) * torch.sigmoid(x @ self.gate_inp_shexp)[:, None]
        return out + sh
