"""
=============================================================================
HSRU DeltaGrid Memory Architecture (Tier 2)
=============================================================================

Spatial Factorized Associative Fast-Weight Memory Layer.
Unifies Dense Bounded State, Multi-Axis Factorized Kronecker Routing,
and Exponential Sparse Product-Key Memory (PKM) with Closed-Loop Householder
Delta Error Erasure.
=============================================================================
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import DeltaGridConfig
from ..ops import (
    has_cuda_ext,
    get_cuda_backend,
    _DeltaGridAutogradFunction,
    _SparseDeltaPKMAutogradFunction,
)


class DeltaGridLayer(nn.Module):
    """
    DeltaGrid Associative Memory Layer (Tier 2).

    Unifies Dense Fast-Weight Associative Memory, Multi-Axis Factorized Kronecker Routing,
    and Exponential Sparse Product-Key Memory (PKM) with Closed-Loop Householder Delta Error Erasure.

    Modes of Operation:
      1. Standard Dense (default):
         State S in R^{Batch x Heads x Dh x Dh} (or Slots x Dh). Fits 100% on-chip in SRAM
         (zero DRAM roundtrips) accelerated via fused CUDA shared memory kernel.
      2. Factorized Kronecker Dense (grid_dim=(M_1, ..., M_N), top_k=None):
         Multi-axis Kronecker factorized key/query projections with linear parameters O(Dh * sum(M_i))
         addressing prod(M_i) slots with exact Householder unit norm ||w_k||_2 = 1.0.
      3. Sparse Delta-PKM (grid_dim=(1024, 1024), top_k=8):
         Exponential product coordinate space (e.g. 1,048,576 slots) with Top-K sparse routing
         restricting active compute to strictly 8 KB SRAM registers per step. Supports fused CUDA
         microsecond pkm_step execution with coalesced gather/scatter.
    """
    def __init__(
        self,
        d_model: Optional[Union[int, DeltaGridConfig]] = None,
        num_heads: int = 4,
        num_slots: Optional[int] = None,
        grid_dim: Optional[Tuple[int, ...]] = None,
        top_k: Optional[int] = None,
        decay_init: float = 1.0,
        dynamic_decay: bool = True,
        strict_slots: bool = False,
        use_chunkwise: bool = False,
        chunk_size: int = 64,
        config: Optional[DeltaGridConfig] = None,
        **kwargs: Any,
    ):
        super().__init__()
        if isinstance(d_model, DeltaGridConfig):
            cfg = d_model
        elif config is not None:
            cfg = config
        else:
            cfg = None

        if cfg is not None:
            d_model = cfg.d_model
            num_heads = cfg.num_heads
            num_slots = cfg.num_slots
            grid_dim = cfg.grid_dim
            top_k = cfg.top_k
            decay_init = cfg.decay_init
            dynamic_decay = cfg.dynamic_decay
            strict_slots = cfg.strict_slots
            use_chunkwise = cfg.use_chunkwise
            chunk_size = cfg.chunk_size
        else:
            if d_model is None:
                d_model = kwargs.get("d_model", 128)

        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Comprehensive Architectural Parameter Validation
        if top_k is not None and grid_dim is None:
            raise ValueError(
                f"top_k ({top_k}) was specified without grid_dim. "
                f"Please provide grid_dim (e.g. grid_dim=(64, 64), top_k={top_k})"
            )

        if grid_dim is not None:
            if not isinstance(grid_dim, (tuple, list)) or len(grid_dim) < 2:
                raise ValueError(f"grid_dim must be a tuple/list with at least 2 axes (e.g. (64, 64)), got {grid_dim}")
            for dim_idx, axis_size in enumerate(grid_dim):
                if not isinstance(axis_size, int) or axis_size < 2:
                    raise ValueError(f"grid_dim axis {dim_idx} must be an integer >= 2, got {axis_size}")
            if top_k is not None:
                if not isinstance(top_k, int) or top_k < 1:
                    raise ValueError(f"top_k must be an integer >= 1, got {top_k}")
                min_axis = min(grid_dim)
                if top_k > min_axis:
                    raise ValueError(f"top_k ({top_k}) cannot exceed the smallest grid dimension ({min_axis}) in {grid_dim}")
        elif top_k is not None:
            raise ValueError("top_k was specified without grid_dim. Please provide grid_dim (e.g. grid_dim=(64, 64), top_k=8)")

        self.grid_dim = tuple(grid_dim) if grid_dim is not None else None
        self.top_k = top_k
        self.dynamic_decay = dynamic_decay
        self.use_chunkwise = use_chunkwise
        self.chunk_size = chunk_size

        # Fused Linear Projection: Computes Q, K, V and scalar beta write gate in a single GEMM call.
        self.proj_qkv_beta = nn.Linear(d_model, 3 * d_model + num_heads, bias=False)
        with torch.no_grad():
            w_qkvb = self.proj_qkv_beta.weight
            nn.init.orthogonal_(w_qkvb[:d_model])
            nn.init.orthogonal_(w_qkvb[d_model:2 * d_model])
            nn.init.orthogonal_(w_qkvb[2 * d_model:3 * d_model])
            nn.init.zeros_(w_qkvb[3 * d_model:])

        # Configure memory coordinate routing based on mode:
        if self.top_k is not None:
            # Mode 3: Sparse Delta-PKM (Exponential Product-Key Memory)
            self.total_slots = math.prod(self.grid_dim)
            self.active_slots = self.top_k ** len(self.grid_dim)
            self.num_slots = self.total_slots
            self.actual_slots = self.total_slots
            self.axes_proj_k = nn.ModuleList([
                nn.Linear(self.head_dim, dim_size, bias=False) for dim_size in self.grid_dim
            ])
            self.axes_proj_q = nn.ModuleList([
                nn.Linear(self.head_dim, dim_size, bias=False) for dim_size in self.grid_dim
            ])
            for pk, pq in zip(self.axes_proj_k, self.axes_proj_q):
                nn.init.orthogonal_(pk.weight)
                with torch.no_grad():
                    pq.weight.copy_(pk.weight)
            self.slot_proj_k = None
            self.slot_proj_q = None

        elif self.grid_dim is not None:
            # Mode 2: Factorized Kronecker Dense
            self.total_slots = math.prod(self.grid_dim)
            self.active_slots = self.total_slots
            self.num_slots = self.total_slots
            self.actual_slots = self.total_slots
            self.axes_proj_k = nn.ModuleList([
                nn.Linear(self.head_dim, dim_size, bias=False) for dim_size in self.grid_dim
            ])
            self.axes_proj_q = nn.ModuleList([
                nn.Linear(self.head_dim, dim_size, bias=False) for dim_size in self.grid_dim
            ])
            for pk, pq in zip(self.axes_proj_k, self.axes_proj_q):
                nn.init.orthogonal_(pk.weight)
                with torch.no_grad():
                    pq.weight.copy_(pk.weight)
            self.slot_proj_k = None
            self.slot_proj_q = None

        else:
            # Mode 1: Standard Dense
            grid_slots = 0
            self.num_slots = max(self.head_dim, grid_slots) if num_slots is None else max(num_slots, grid_slots)
            self.actual_slots = num_slots if (strict_slots and num_slots is not None) else max(self.num_slots, self.head_dim)
            self.total_slots = self.actual_slots
            self.active_slots = self.actual_slots
            self.axes_proj_k = None
            self.axes_proj_q = None
            if self.actual_slots != self.head_dim:
                self.slot_proj_k = nn.Linear(self.head_dim, self.actual_slots, bias=False)
                self.slot_proj_q = nn.Linear(self.head_dim, self.actual_slots, bias=False)
                nn.init.orthogonal_(self.slot_proj_k.weight)
                with torch.no_grad():
                    self.slot_proj_q.weight.copy_(self.slot_proj_k.weight)
            else:
                self.slot_proj_k = None
                self.slot_proj_q = None

        # Learnable per-head Query Routing scale
        self.q_scale = nn.Parameter(torch.ones(num_heads))

        # Linear readout projection back to model hidden dimension
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

        # Optional Data-Dependent Dynamic Decay Gate
        if dynamic_decay:
            self.proj_alpha: Optional[nn.Linear] = nn.Linear(d_model, num_heads, bias=True)
            rel_init = min(max((decay_init - 0.99) / 0.01, 1e-3), 0.999)
            alpha_logit = math.log(rel_init / (1.0 - rel_init))
            nn.init.zeros_(self.proj_alpha.weight)
            nn.init.constant_(self.proj_alpha.bias, alpha_logit)
        else:
            self.proj_alpha = None

        self.decay = nn.Parameter(torch.full((1, num_heads, 1, 1), decay_init), requires_grad=False)
        self.beta_bias = nn.Parameter(torch.full((1, num_heads, 1, 1), 0.0))

    def _route_dense(self, k: torch.Tensor, q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Routes keys and queries in dense factorized or linear projection mode."""
        if self.axes_proj_k is not None and self.axes_proj_q is not None:
            axis_k = [F.normalize(p(k), p=2, dim=-1) for p in self.axes_proj_k]
            axis_q = [p(q) for p in self.axes_proj_q]
            wk = axis_k[0]
            for u in axis_k[1:]:
                wk = (wk.unsqueeze(-1) * u.unsqueeze(-2)).flatten(-2, -1)
            wq = axis_q[0]
            for u in axis_q[1:]:
                wq = (wq.unsqueeze(-1) * u.unsqueeze(-2)).flatten(-2, -1)
            return wk, wq
        elif self.slot_proj_k is not None and self.slot_proj_q is not None:
            wk = F.normalize(self.slot_proj_k(k), p=2, dim=-1)
            wq = self.slot_proj_q(q)
            return wk, wq
        else:
            return k, q

    def _route_sparse_pkm(
        self, k: torch.Tensor, q: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Projects k and q onto multi-axis coordinates and selects Top-K per axis.
        Supports both single-step [B, H, Dh] and batched sequence [B, H, T, Dh].
        """
        is_seq = (k.dim() == 4)
        if is_seq:
            b, h, t, dh = k.shape
            k_flat = k.reshape(b * h * t, dh)
            q_flat = q.reshape(b * h * t, dh)
        else:
            b, h, dh = k.shape
            k_flat = k.reshape(b * h, dh)
            q_flat = q.reshape(b * h, dh)

        u_k_list, u_q_list, idx_list = [], [], []
        for pk, pq in zip(self.axes_proj_k, self.axes_proj_q):
            sk = pk(k_flat)
            val_k, idx_k = torch.topk(sk, self.top_k, dim=-1)
            u_k = F.normalize(val_k, p=2, dim=-1)
            u_k_list.append(u_k)
            idx_list.append(idx_k)

            sq = pq(q_flat)
            uq = torch.gather(sq, -1, idx_k)
            u_q = F.normalize(uq, p=2, dim=-1)
            u_q_list.append(u_q)

        wk = u_k_list[0]
        wq = u_q_list[0]
        active_idx = idx_list[0]
        for i in range(1, len(self.grid_dim)):
            wk = (wk.unsqueeze(-1) * u_k_list[i].unsqueeze(-2)).flatten(-2, -1)
            wq = (wq.unsqueeze(-1) * u_q_list[i].unsqueeze(-2)).flatten(-2, -1)
            active_idx = (active_idx.unsqueeze(-1) * self.grid_dim[i] + idx_list[i].unsqueeze(-2)).flatten(-2, -1)

        if is_seq:
            wk = wk.view(b, h, t, self.active_slots)
            wq = wq.view(b, h, t, self.active_slots)
            active_idx = active_idx.view(b, h, t, self.active_slots)
        else:
            wk = wk.view(b, h, self.active_slots)
            wq = wq.view(b, h, self.active_slots)
            active_idx = active_idx.view(b, h, self.active_slots)
        return wk, wq, active_idx

    def compute_router_entropy(self, k: torch.Tensor) -> torch.Tensor:
        """Computes the average Shannon entropy across coordinate router projections."""
        if self.axes_proj_k is None:
            return torch.tensor(0.0, device=k.device, dtype=k.dtype)

        k_flat = k.reshape(-1, self.head_dim)
        entropies = []
        for pk in self.axes_proj_k:
            logits = pk(k_flat)
            probs = F.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1).mean()
            entropies.append(entropy)
        return torch.stack(entropies).mean()

    def compute_load_balance_loss(self, k: torch.Tensor) -> torch.Tensor:
        """Auxiliary load balancing loss for sparse coordinate codebooks."""
        if self.axes_proj_k is None or self.top_k is None:
            return torch.tensor(0.0, device=k.device, dtype=k.dtype)

        k_flat = k.reshape(-1, self.head_dim)
        losses = []
        for pk in self.axes_proj_k:
            logits = pk(k_flat)
            probs = F.softmax(logits, dim=-1)
            _, topk_idx = torch.topk(logits, self.top_k, dim=-1)
            one_hot = F.one_hot(topk_idx, num_classes=logits.size(-1)).float()
            freq = one_hot.sum(dim=1).mean(dim=0)
            mean_prob = probs.mean(dim=0)
            loss = logits.size(-1) * torch.sum(freq * mean_prob)
            losses.append(loss)
        return torch.stack(losses).mean()

    def chunk_forward(
        self,
        v: torch.Tensor,
        beta: torch.Tensor,
        wk: torch.Tensor,
        wq: torch.Tensor,
        decay: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fully Vectorized Chunkwise Parallel WY Tensor-Core Prefill."""
        b, h, t, dh = v.shape
        slots = wk.size(-1)
        c_size = min(chunk_size if chunk_size is not None else self.chunk_size, max(t, 1))
        orig_dtype = v.dtype

        v_f = v.float()
        wk_f = wk.float()
        wq_f = wq.float()
        beta_f = beta.squeeze(-1).float() if beta.dim() == 4 else beta.float()

        if state is None:
            state_f = torch.zeros(b, h, slots, dh, device=v.device, dtype=torch.float32)
        else:
            state_f = state.float()

        if decay.dim() == 3:
            alpha_all = decay.float()
        else:
            alpha_all = decay.view(1, h, 1).expand(b, h, t).float()

        rem = t % c_size
        pad_len = (c_size - rem) if rem != 0 else 0
        if pad_len > 0:
            v_f = F.pad(v_f, (0, 0, 0, pad_len), value=0.0)
            wk_f = F.pad(wk_f, (0, 0, 0, pad_len), value=0.0)
            wq_f = F.pad(wq_f, (0, 0, 0, pad_len), value=0.0)
            beta_f = F.pad(beta_f, (0, pad_len), value=0.0)
            alpha_all = F.pad(alpha_all, (0, pad_len), value=1.0)

        t_pad = t + pad_len
        num_chunks = t_pad // c_size

        v_c = v_f.view(b, h, num_chunks, c_size, dh)
        wk_c = wk_f.view(b, h, num_chunks, c_size, slots)
        wq_c = wq_f.view(b, h, num_chunks, c_size, slots)
        beta_c = beta_f.view(b, h, num_chunks, c_size)
        alpha_c = alpha_all.view(b, h, num_chunks, c_size)

        log_alpha = torch.log(alpha_c.clamp(min=1e-12))
        cum_log = F.pad(torch.cumsum(log_alpha, dim=-1), (1, 0), value=0.0)
        diff_log = cum_log[..., :c_size].unsqueeze(-1) - cum_log[..., 1:c_size + 1].unsqueeze(-2)
        strict_mask = torch.ones(c_size, c_size, device=v.device, dtype=torch.bool).tril(-1)
        gamma_mat = torch.where(strict_mask, torch.exp(diff_log.clamp(max=0.0)), torch.zeros_like(diff_log))

        a_init = torch.exp(cum_log[..., :c_size]).unsqueeze(-1)
        a_end_vec = torch.exp((cum_log[..., c_size:c_size + 1] - cum_log[..., 1:c_size + 1]).clamp(max=0.0)).unsqueeze(-1)
        a_total = torch.exp(cum_log[..., c_size:c_size + 1]).unsqueeze(-1)

        wk_T = wk_c.transpose(-1, -2)
        k_k = torch.matmul(wk_c, wk_T)
        l_mat = gamma_mat * (beta_c.unsqueeze(-2) * k_k)
        eye = torch.eye(c_size, device=v.device, dtype=torch.float32)
        wk_scaled = a_init * wk_c
        rhs_uw = torch.cat([v_c, wk_scaled], dim=-1)
        uw_solved = torch.linalg.solve_triangular(l_mat + eye, rhs_uw, upper=False, unitriangular=True)
        u_all = uw_solved[..., :dh]
        w_all = uw_solved[..., dh:]

        wk_end_T = ((a_end_vec * beta_c.unsqueeze(-1)) * wk_c).transpose(-1, -2)
        s_prev_list = []
        r_list = []
        for c_idx in range(num_chunks):
            s_prev_list.append(state_f)
            r_c = u_all[:, :, c_idx] - torch.matmul(w_all[:, :, c_idx], state_f)
            r_list.append(r_c)
            state_f = a_total[:, :, c_idx] * state_f + torch.matmul(wk_end_T[:, :, c_idx], r_c)

        s_prev_all = torch.stack(s_prev_list, dim=2)
        r_all = torch.stack(r_list, dim=2)
        q_k = torch.matmul(wq_c, wk_T)
        m_qk = gamma_mat * (beta_c.unsqueeze(-2) * q_k)
        y_all = torch.matmul(a_init * wq_c, s_prev_all) + torch.matmul(m_qk, r_all)

        out = y_all.reshape(b, h, t_pad, dh)[:, :, :t, :].to(orig_dtype)
        return out, state_f.to(orig_dtype)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        seq_offset: int = 0,
        use_chunkwise: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass across sequence T with Recurrent State S."""
        b, t, d = x.shape
        num_heads, head_dim = self.num_heads, self.head_dim

        packed = self.proj_qkv_beta(x)
        qkv, beta_raw = packed[:, :, :3 * d], packed[:, :, 3 * d:]
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(b, t, num_heads, head_dim).transpose(1, 2)
        k = F.normalize(k.view(b, t, num_heads, head_dim).transpose(1, 2), p=2, dim=-1)
        v = v.view(b, t, num_heads, head_dim).transpose(1, 2)
        beta = torch.sigmoid(beta_raw.view(b, t, num_heads, 1).transpose(1, 2) + self.beta_bias)

        if self.proj_alpha is not None:
            decay_tensor = (0.99 + 0.01 * torch.sigmoid(self.proj_alpha(x))).transpose(1, 2)
        else:
            decay_tensor = self.decay

        # Mode 3: Sparse Delta-PKM
        if self.top_k is not None:
            if state is None:
                state = torch.zeros(b, num_heads, self.total_slots, head_dim, device=x.device, dtype=v.dtype)

            q_scale_step = self.q_scale.to(dtype=q.dtype).view(1, num_heads, 1)
            wk_all, wq_all, active_idx_all = self._route_sparse_pkm(k, q)
            needs_grad = torch.is_grad_enabled() and (x.requires_grad or (state is not None and state.requires_grad))

            cuda_backend = get_cuda_backend()
            can_cuda = cuda_backend is not None and x.is_cuda and hasattr(cuda_backend, "pkm_step")

            if can_cuda:
                wq_scaled = wq_all * q_scale_step.unsqueeze(-1)
                y_all, state_run = _SparseDeltaPKMAutogradFunction.apply(
                    v, beta, wk_all, wq_scaled, active_idx_all.to(torch.int64), decay_tensor, state
                )
                out = y_all.transpose(1, 2).reshape(b, t, d)
                return self.norm(self.out_proj(out)), state_run

            # Pure PyTorch / CPU Fallback Reference Loop
            outputs = []
            state_run = state.clone()
            for step_idx in range(t):
                wkt = wk_all[:, :, step_idx]
                wqt = wq_all[:, :, step_idx] * q_scale_step
                idx_t = active_idx_all[:, :, step_idx]
                expand_idx = idx_t.unsqueeze(-1).expand(b, num_heads, self.active_slots, head_dim)

                s_act = torch.gather(state_run, 2, expand_idx)
                yt = torch.matmul(wqt.unsqueeze(-2), s_act).squeeze(-2)
                outputs.append(yt)

                pred_v = torch.matmul(wkt.unsqueeze(-2), s_act).squeeze(-2)
                rt = v[:, :, step_idx] - pred_v
                delta = (beta[:, :, step_idx] * wkt).unsqueeze(-1) * rt.unsqueeze(-2)
                at = decay_tensor[:, :, step_idx].unsqueeze(-1).unsqueeze(-1) if decay_tensor.dim() == 3 else decay_tensor
                s_act_new = (at * s_act + delta).to(state_run.dtype)
                state_run = state_run.scatter(2, expand_idx, s_act_new)

            out = torch.stack(outputs, dim=2).transpose(1, 2).reshape(b, t, d)
            return self.norm(self.out_proj(out)), state_run

        # Modes 1 & 2: Dense Factorized / Standard Routing
        q_scale = self.q_scale.to(dtype=q.dtype).view(1, num_heads, 1, 1)
        wk, wq = self._route_dense(k, q)
        wq = wq * q_scale

        run_chunkwise = self.use_chunkwise if use_chunkwise is None else use_chunkwise
        if run_chunkwise:
            out, new_state = self.chunk_forward(v, beta, wk, wq, decay_tensor, state=state)
            out = out.transpose(1, 2).reshape(b, t, d)
            return self.norm(self.out_proj(out)), new_state

        cuda_backend = get_cuda_backend()
        if cuda_backend is not None and x.is_cuda and not torch.jit.is_scripting():
            decay_in = decay_tensor.to(dtype=q.dtype) if decay_tensor.dim() == 3 else decay_tensor.to(dtype=q.dtype).view(num_heads)
            beta_squeeze = beta.squeeze(-1).to(dtype=q.dtype)
            k_in = k.to(dtype=q.dtype)
            v_in = v.to(dtype=q.dtype)
            wk_in = wk.to(dtype=q.dtype)
            wq_in = wq.to(dtype=q.dtype)
            s_in = state.to(dtype=q.dtype) if state is not None else None
            if torch.is_grad_enabled() and (x.requires_grad or decay_in.requires_grad):
                out, new_state = _DeltaGridAutogradFunction.apply(q, k_in, v_in, beta_squeeze, wk_in, wq_in, decay_in, s_in)
            else:
                out, new_state, _, _ = cuda_backend.forward(q, k_in, v_in, beta_squeeze, wk_in, wq_in, decay_in, s_in, False)
            out = out.transpose(1, 2).reshape(b, t, d)
            return self.norm(self.out_proj(out)), new_state

        if t >= 16 and not torch.jit.is_scripting():
            out, new_state = self.chunk_forward(v, beta, wk, wq, decay_tensor, state=state)
            out = out.transpose(1, 2).reshape(b, t, d)
            return self.norm(self.out_proj(out)), new_state

        if state is None:
            state = torch.zeros(b, num_heads, self.actual_slots, head_dim, device=x.device, dtype=x.dtype)

        outputs = []
        for step_idx in range(t):
            wqt = wq[:, :, step_idx]
            wkt = wk[:, :, step_idx]
            vt = v[:, :, step_idx]
            bt = beta[:, :, step_idx]
            at = decay_tensor[:, :, step_idx].unsqueeze(-1).unsqueeze(-1) if decay_tensor.dim() == 3 else decay_tensor

            yt = torch.matmul(wqt.unsqueeze(-2), state).squeeze(-2)
            outputs.append(yt)

            pred_v = torch.matmul(wkt.unsqueeze(-2), state).squeeze(-2)
            rt = vt - pred_v

            delta = (bt * wkt).unsqueeze(-1) * rt.unsqueeze(-2)
            state = at * state + delta

        out = torch.stack(outputs, dim=2).transpose(1, 2).reshape(b, t, d)
        return self.norm(self.out_proj(out)), state

    def step(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        seq_offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single-Step Inference for Autoregressive Generation."""
        b, t, d = x.shape
        assert t == 1, f"step() expects sequence length 1, got {t}"
        num_heads, head_dim = self.num_heads, self.head_dim

        x_sq = x.squeeze(1)
        packed = self.proj_qkv_beta(x_sq)
        qkv, beta_raw = packed[:, :3 * d], packed[:, 3 * d:]
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(b, num_heads, head_dim)
        k = F.normalize(k.view(b, num_heads, head_dim), p=2, dim=-1)
        v = v.view(b, num_heads, head_dim)
        beta = torch.sigmoid(beta_raw.view(b, num_heads, 1) + self.beta_bias.view(1, num_heads, 1))

        if self.proj_alpha is not None:
            decay_step_val = 0.99 + 0.01 * torch.sigmoid(self.proj_alpha(x_sq))
        else:
            decay_step_val = self.decay

        cuda_backend = get_cuda_backend()

        # Mode 3: Sparse Delta-PKM Single Step
        if self.top_k is not None:
            if state is None:
                state = torch.zeros(b, num_heads, self.total_slots, head_dim, device=x.device, dtype=v.dtype)

            q_scale = self.q_scale.to(dtype=q.dtype).view(1, num_heads, 1)
            wk_act, wq_act, active_idx = self._route_sparse_pkm(k, q)
            wq_act = wq_act * q_scale

            if cuda_backend is not None and x.is_cuda and hasattr(cuda_backend, "pkm_step"):
                decay_step = decay_step_val.to(dtype=q.dtype) if decay_step_val.dim() == 2 else decay_step_val.to(dtype=q.dtype).view(num_heads)
                beta_step = beta.squeeze(2).to(dtype=q.dtype)
                v_step = v.to(dtype=q.dtype)
                wk_step = wk_act.to(dtype=q.dtype)
                wq_step = wq_act.to(dtype=q.dtype)
                s_step = state.to(dtype=q.dtype)
                active_idx_c = active_idx.to(dtype=torch.int64)
                yt, new_state = cuda_backend.pkm_step(
                    v_step, beta_step, wk_step, wq_step, active_idx_c, decay_step, s_step
                )
                out = self.norm(self.out_proj(yt.view(b, d))).unsqueeze(1)
                return out, new_state

            expand_idx = active_idx.unsqueeze(-1).expand(b, num_heads, self.active_slots, head_dim)
            s_act = torch.gather(state, 2, expand_idx)
            yt = torch.matmul(wq_act.unsqueeze(-2), s_act).squeeze(-2)
            pred_v = torch.matmul(wk_act.unsqueeze(-2), s_act).squeeze(-2)
            rt = v - pred_v
            delta = (beta * wk_act).unsqueeze(-1) * rt.unsqueeze(-2)
            at = decay_step_val.unsqueeze(-1).unsqueeze(-1) if decay_step_val.dim() == 2 else decay_step_val
            s_act_new = at * s_act + delta
            new_state = state.scatter(2, expand_idx, s_act_new)

            out = self.norm(self.out_proj(yt.view(b, d))).unsqueeze(1)
            return out, new_state

        # Modes 1 & 2: Dense Factorized / Standard Single Step
        if state is None:
            state = torch.zeros(b, num_heads, self.actual_slots, head_dim, device=x.device, dtype=x.dtype)

        q_scale = self.q_scale.to(dtype=q.dtype).view(1, num_heads, 1)
        wk, wq = self._route_dense(k, q)
        wq = wq * q_scale

        if cuda_backend is not None and x.is_cuda and hasattr(cuda_backend, "step"):
            decay_step = decay_step_val.to(dtype=q.dtype) if decay_step_val.dim() == 2 else decay_step_val.to(dtype=q.dtype).view(num_heads)
            beta_step = beta.squeeze(2).to(dtype=q.dtype)
            k_step = k.to(dtype=q.dtype)
            v_step = v.to(dtype=q.dtype)
            wk_step = wk.to(dtype=q.dtype)
            wq_step = wq.to(dtype=q.dtype)
            s_step = state.to(dtype=q.dtype)
            yt, new_state = cuda_backend.step(
                q, k_step, v_step, beta_step, wk_step, wq_step, decay_step, s_step
            )
            out = self.norm(self.out_proj(yt.view(b, d))).unsqueeze(1)
            return out, new_state

        yt = torch.matmul(wq.unsqueeze(-2), state).squeeze(-2)
        pred_v = torch.matmul(wk.unsqueeze(-2), state).squeeze(-2)
        rt = v - pred_v
        delta = (beta * wk).unsqueeze(-1) * rt.unsqueeze(-2)
        at = decay_step_val.unsqueeze(-1).unsqueeze(-1) if decay_step_val.dim() == 2 else decay_step_val
        state = at * state + delta

        out = self.norm(self.out_proj(yt.view(b, d))).unsqueeze(1)
        return out, state


# Unified backward-compatibility alias: DeltaGridLayer handles all dense & sparse PKM modes
SparseDeltaPKMLayer = DeltaGridLayer
