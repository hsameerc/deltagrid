"""
=============================================================================
HSRU DeltaGrid 4-Tier Hierarchical Neural Block
=============================================================================

Complete 4-tier processing block:
  - Tier 0: Biological Harmonic Clock (sin/cos phase anchoring)
  - Tier 1: Causal Depthwise Conv1D (k=4) for local n-gram token binding
  - Tier 2: DeltaGridLayer for associative zero-KV memory
  - Tier 3: Non-Linear SwiGLU Logic Reasoning Engine
=============================================================================
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import DeltaGridConfig
from .memory import DeltaGridLayer

class SwiGLU(nn.Module):
    """Swish Gated Linear Unit (SwiGLU): SiLU(a) * b."""
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return super().__call__(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=self.dim)
        return F.silu(a) * b 

class DeltaGridBlock(nn.Module):
    """
    DeltaGrid Complete 4-Tier Hierarchical Block.

    Combines harmonic positional coordinates, high-frequency local convolution,
    associative fast-weight memory, and non-linear relational logic into a unified
    recurrent block with strictly constant O(1) RAM.
    """
    def __init__(
        self,
        d_model: Optional[Union[int, DeltaGridConfig]] = None,
        num_heads: int = 4,
        num_slots: Optional[int] = None,
        grid_dim: Optional[Tuple[int, ...]] = None,
        top_k: Optional[int] = None,
        has_memory: bool = True,
        kernel_size: int = 4,
        max_norm: Optional[float] = None,
        decay_init: float = 0.999,
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
        elif "cfg" in kwargs and isinstance(kwargs["cfg"], DeltaGridConfig):
            cfg = kwargs["cfg"]
        else:
            cfg = None

        dim: int
        if cfg is not None:
            dim = int(cfg.d_model)
            num_heads = cfg.num_heads
            num_slots = cfg.num_slots
            grid_dim = cfg.grid_dim
            top_k = cfg.top_k
            decay_init = cfg.decay_init
            dynamic_decay = cfg.dynamic_decay
            strict_slots = cfg.strict_slots
            use_chunkwise = cfg.use_chunkwise
            chunk_size = cfg.chunk_size
            kernel_size = cfg.kernel_size
            max_norm = cfg.max_norm
        else:
            if isinstance(d_model, int):
                dim = d_model
            else:
                raw_d = kwargs.get("d_model", 128)
                dim = int(raw_d) if isinstance(raw_d, int) else 128

        self.d_model: int = dim
        self.has_memory: bool = has_memory
        self.kernel_size: int = kernel_size
        self.max_norm: Optional[float] = max_norm

        # Tier 0: Biological Multi-Frequency Harmonic Clock
        self.div_term = nn.Parameter(
            torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim)),
            requires_grad=False,
        )
        self.clock_scale = nn.Parameter(torch.tensor(0.1))

        # Tier 1: Causal Depthwise Conv1D
        self.conv1d = nn.Conv1d(dim, dim, kernel_size=kernel_size, groups=dim, bias=True)
        self.norm_conv = nn.LayerNorm(dim)

        # Tier 2: Associative DeltaGrid Memory Layer
        if has_memory:
            subspace_slicing = getattr(cfg, "subspace_slicing", False) if cfg is not None else kwargs.get("subspace_slicing", False)
            self.memory: Optional[DeltaGridLayer] = DeltaGridLayer(
                d_model=dim,
                num_heads=num_heads,
                num_slots=num_slots,
                grid_dim=grid_dim,
                top_k=top_k,
                decay_init=decay_init,
                dynamic_decay=dynamic_decay,
                strict_slots=strict_slots,
                use_chunkwise=use_chunkwise,
                chunk_size=chunk_size,
                subspace_slicing=subspace_slicing,
            )
            self.scale_mem = nn.Parameter(torch.tensor(1.0))
        else:
            self.memory = None

        # Tier 3: Non-Linear SwiGLU Logic Reasoning Engine
        raw_exp = 1.0
        if cfg is not None:
            raw_exp = getattr(cfg, "logic_expansion", getattr(cfg, "logic_mult", 1.0))
        elif "logic_expansion" in kwargs:
            raw_exp = kwargs["logic_expansion"]
        elif "logic_mult" in kwargs:
            raw_exp = kwargs["logic_mult"]

        logic_exp: float = float(raw_exp) if isinstance(raw_exp, (int, float)) else 1.0
        ffn_inner: int = int(round(dim * 2 * logic_exp))
        ffn_mid: int = int(round(dim * logic_exp))

        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_inner),
            SwiGLU(dim=-1),
            nn.Linear(ffn_mid, dim),
        )
        self.norm_ffn = nn.LayerNorm(dim)
        self.scale_logic = nn.Parameter(torch.tensor(1.0))

    def get_clock(self, seq_len: int, device: torch.device, offset: int = 0) -> torch.Tensor:
        """Generates continuous multiscale sinusoidal coordinates [1, seq_len, d_model]."""
        pos = torch.arange(offset, offset + seq_len, dtype=torch.float32, device=device).unsqueeze(1)
        pe = torch.zeros(seq_len, self.d_model, device=device)
        pe[:, 0::2] = torch.sin(pos * self.div_term)
        pe[:, 1::2] = torch.cos(pos * self.div_term)
        return pe.unsqueeze(0)

    def _apply_conv(
        self,
        x_in: torch.Tensor,
        ind_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Unified causal Conv1D execution for both full sequence and single step."""
        if ind_state is not None:
            conv_in = torch.cat([ind_state, x_in.transpose(1, 2)], dim=-1)
        else:
            conv_in = F.pad(x_in.transpose(1, 2), (self.kernel_size - 1, 0))
        new_ind = conv_in[:, :, -(self.kernel_size - 1):]
        c_out = F.silu(self.conv1d(conv_in)).transpose(1, 2)
        h = self.norm_conv(x_in + c_out)
        return h, new_ind

    def __call__(
        self,
        x: torch.Tensor,
        mem_state: Optional[torch.Tensor] = None,
        ind_state: Optional[torch.Tensor] = None,
        thinking_cycles: int = 0,
        seq_offset: int = 0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        return super().__call__(
            x,
            mem_state=mem_state,
            ind_state=ind_state,
            thinking_cycles=thinking_cycles,
            seq_offset=seq_offset,
        )

    def forward(
        self,
        x: torch.Tensor,
        mem_state: Optional[torch.Tensor] = None,
        ind_state: Optional[torch.Tensor] = None,
        thinking_cycles: int = 0,
        seq_offset: int = 0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Full sequence forward pass through the 4-tier DeltaGrid hierarchy."""
        b, t, d = x.shape

        # Tier 0: Biological Harmonic Clock
        clock = self.get_clock(t, x.device, offset=seq_offset).to(dtype=x.dtype)
        x_in = x + clock * self.clock_scale

        # Tier 1: Causal Conv1D
        h, new_ind = self._apply_conv(x_in, ind_state)

        # Tier 2: DeltaGrid Memory
        if self.has_memory and self.memory is not None:
            memory_mod: DeltaGridLayer = self.memory
            mem_out, new_mem = memory_mod(h, state=mem_state, seq_offset=seq_offset)
            x_in = x_in + mem_out * self.scale_mem
        else:
            new_mem = None

        # Tier 3: SWIGLU Logic Reasoning
        logic_out = self.ffn(x_in)
        x_in = x_in + self.norm_ffn(logic_out) * self.scale_logic

        # Cognitive Recurrence (Working-Memory Scratchpad Thinking Cycles)
        if thinking_cycles > 0 and self.has_memory and self.memory is not None:
            memory_mod = self.memory
            q_h = x_in[:, -1:, :]
            think_mem = new_mem
            for cycle_idx in range(thinking_cycles):
                m_out, think_mem = memory_mod(
                    q_h, state=think_mem, seq_offset=seq_offset + t + cycle_idx
                )
                q_h = q_h + m_out * self.scale_mem
                l_out = self.norm_ffn(self.ffn(q_h))
                q_h = q_h + l_out * self.scale_logic
            x_in = torch.cat([x_in[:, :-1, :], q_h], dim=1)

        return x_in, new_mem, new_ind

    def step(
        self,
        x: torch.Tensor,
        mem_state: Optional[torch.Tensor] = None,
        ind_state: Optional[torch.Tensor] = None,
        thinking_cycles: int = 0,
        seq_offset: int = 0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Single-step inference for real-time streaming and autoregressive decoding."""
        b, t, d = x.shape
        assert t == 1, f"step() expects sequence length 1, got {t}"

        # Tier 0: Biological Harmonic Clock
        clock = self.get_clock(1, x.device, offset=seq_offset).to(dtype=x.dtype)
        x_in = x + clock * self.clock_scale

        # Tier 1: Causal Conv1D
        if ind_state is None:
            ind_state = torch.zeros(b, d, self.kernel_size - 1, device=x.device, dtype=x.dtype)
        h, new_ind = self._apply_conv(x_in, ind_state)

        # Tier 2: DeltaGrid Memory
        if self.has_memory and self.memory is not None:
            memory_mod: DeltaGridLayer = self.memory
            mem_out, new_mem = memory_mod.step(h, state=mem_state, seq_offset=seq_offset)
            x_in = x_in + mem_out * self.scale_mem
        else:
            new_mem = None

        # Tier 3: SWIGLU Logic Reasoning
        logic_out = self.ffn(x_in)
        x_in = x_in + self.norm_ffn(logic_out) * self.scale_logic

        # Cognitive Thinking Loops
        if thinking_cycles > 0 and self.has_memory and self.memory is not None:
            memory_mod = self.memory
            think_mem = new_mem
            for cycle_idx in range(thinking_cycles):
                m_out, think_mem = memory_mod.step(
                    x_in, state=think_mem, seq_offset=seq_offset + 1 + cycle_idx
                )
                x_in = x_in + m_out * self.scale_mem
                l_out = self.norm_ffn(self.ffn(x_in))
                x_in = x_in + l_out * self.scale_logic

        return x_in, new_mem, new_ind
