"""
=============================================================================
HSRU DeltaGrid 4-Tier Hierarchical Neural Block
=============================================================================

Complete 4-tier processing block:
  - Tier 0: Biological Harmonic Clock (sin/cos phase anchoring)
  - Tier 1: Causal Depthwise Conv1D (k=4) for local n-gram token binding
  - Tier 2: DeltaGridLayer for associative zero-KV memory
  - Tier 3: Non-Linear GLU Logic Reasoning Engine
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
            kernel_size = cfg.kernel_size
            max_norm = cfg.max_norm
        else:
            if d_model is None:
                d_model = kwargs.get("d_model", 128)

        self.d_model = d_model
        self.has_memory = has_memory
        self.kernel_size = kernel_size
        self.max_norm = max_norm

        # Tier 0: Biological Multi-Frequency Harmonic Clock
        self.div_term = nn.Parameter(
            torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)),
            requires_grad=False,
        )
        self.clock_scale = nn.Parameter(torch.tensor(0.1))

        # Tier 1: Causal Depthwise Conv1D
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, groups=d_model, bias=True)
        self.norm_conv = nn.LayerNorm(d_model)

        # Tier 2: Associative DeltaGrid Memory Layer
        if has_memory:
            self.memory: Optional[DeltaGridLayer] = DeltaGridLayer(
                d_model=d_model,
                num_heads=num_heads,
                num_slots=num_slots,
                grid_dim=grid_dim,
                top_k=top_k,
                decay_init=decay_init,
                dynamic_decay=dynamic_decay,
                strict_slots=strict_slots,
                use_chunkwise=use_chunkwise,
                chunk_size=chunk_size,
            )
            self.scale_mem = nn.Parameter(torch.tensor(1.0))
        else:
            self.memory = None

        # Tier 3: Non-Linear GLU Logic Reasoning Engine
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GLU(dim=-1),
            nn.Linear(d_model, d_model),
        )
        self.norm_ffn = nn.LayerNorm(d_model)
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
            mem_out, new_mem = self.memory(h, state=mem_state, seq_offset=seq_offset)
            x_in = x_in + mem_out * self.scale_mem
        else:
            new_mem = None

        # Tier 3: GLU Logic Reasoning
        logic_out = self.ffn(x_in)
        x_in = x_in + self.norm_ffn(logic_out) * self.scale_logic

        # Cognitive Recurrence (Working-Memory Scratchpad Thinking Cycles)
        if thinking_cycles > 0 and self.has_memory and self.memory is not None:
            q_h = x_in[:, -1:, :]
            think_mem = new_mem
            for cycle_idx in range(thinking_cycles):
                m_out, think_mem = self.memory(
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
            mem_out, new_mem = self.memory.step(h, state=mem_state, seq_offset=seq_offset)
            x_in = x_in + mem_out * self.scale_mem
        else:
            new_mem = None

        # Tier 3: GLU Logic Reasoning
        logic_out = self.ffn(x_in)
        x_in = x_in + self.norm_ffn(logic_out) * self.scale_logic

        # Cognitive Thinking Loops
        if thinking_cycles > 0 and self.has_memory and self.memory is not None:
            think_mem = new_mem
            for cycle_idx in range(thinking_cycles):
                m_out, think_mem = self.memory.step(
                    x_in, state=think_mem, seq_offset=seq_offset + 1 + cycle_idx
                )
                x_in = x_in + m_out * self.scale_mem
                l_out = self.norm_ffn(self.ffn(x_in))
                x_in = x_in + l_out * self.scale_logic

        return x_in, new_mem, new_ind
