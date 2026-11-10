"""
=============================================================================
HSRU DeltaGrid Base Sequence Model Backbone
=============================================================================

Shared multi-tier pipeline execution engine powering DeltaGridLM and
DeltaGridContinuous with full TBPTT state preservation and gradient checkpointing.
=============================================================================
"""

from __future__ import annotations

from typing import Optional, Tuple, List, Union, Any
import torch
import torch.nn as nn

from ..state import DeltaGridState, DeltaGridLayerState
from ..modules.block import DeltaGridBlock


class _DeltaGridSequenceModel(nn.Module):
    """
    Shared execution engine for DeltaGrid sequence models.

    Provides common block iteration for both full-sequence parallel context
    encoding (_forward_blocks) and single-step recurrent stepping (_step_blocks).
    """
    blocks: nn.ModuleList
    thinking_cycles: int

    def _forward_blocks(
        self,
        h: torch.Tensor,
        states: Optional[Union[DeltaGridState, List[Any]]] = None,
        seq_offset: int = 0,
        thinking_cycles: Optional[int] = None,
    ) -> Tuple[torch.Tensor, DeltaGridState]:
        """
        Sequentially executes all 4-tier blocks across the full sequence context.

        Args:
            h (torch.Tensor): Projected latent representations [batch_size, seq_len, d_model].
            states (Optional[Union[DeltaGridState, List[Any]]]): Optional recurrent states.
            seq_offset (int): Absolute sequence position offset.
            thinking_cycles (Optional[int]): Override for cognitive recurrence loops.

        Returns:
            Tuple[torch.Tensor, DeltaGridState]: Transformed representations and updated state container.
        """
        cycles = thinking_cycles if thinking_cycles is not None else self.thinking_cycles
        new_state_container = DeltaGridState()
        use_ckpt = (
            self.training
            and getattr(self, "gradient_checkpointing", False)
            and states is None
        )

        for i, block in enumerate(self.blocks):
            st = states[i] if (states is not None and i < len(states)) else None
            if isinstance(st, DeltaGridLayerState):
                mem_s, ind_s = st.memory, st.indicator
            elif isinstance(st, (tuple, list)) and len(st) >= 2:
                mem_s, ind_s = st[0], st[1]
            else:
                mem_s, ind_s = None, None

            if use_ckpt:
                h, new_m, new_i = torch.utils.checkpoint.checkpoint(
                    block, h, None, None, cycles, seq_offset, use_reentrant=False
                )
            else:
                h, new_m, new_i = block(
                    h,
                    mem_state=mem_s,
                    ind_state=ind_s,
                    thinking_cycles=cycles,
                    seq_offset=seq_offset,
                )
            new_state_container.append(memory=new_m, indicator=new_i)

        return h, new_state_container

    def gradient_checkpointing_enable(self) -> None:
        """Enables activation checkpointing across recurrent blocks to conserve VRAM."""
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        """Disables activation checkpointing across recurrent blocks."""
        self.gradient_checkpointing = False

    def _step_blocks(
        self,
        h: torch.Tensor,
        states: Optional[Union[DeltaGridState, List[Any]]] = None,
        seq_offset: int = 0,
    ) -> Tuple[torch.Tensor, DeltaGridState]:
        """
        Sequentially executes all 4-tier blocks for a single token / telemetry step.
        """
        new_state_container = DeltaGridState()
        for i, block in enumerate(self.blocks):
            st = states[i] if (states is not None and i < len(states)) else None
            if isinstance(st, DeltaGridLayerState):
                mem_s, ind_s = st.memory, st.indicator
            elif isinstance(st, (tuple, list)) and len(st) >= 2:
                mem_s, ind_s = st[0], st[1]
            else:
                mem_s, ind_s = None, None

            h, new_m, new_i = block.step(
                h,
                mem_state=mem_s,
                ind_state=ind_s,
                thinking_cycles=self.thinking_cycles,
                seq_offset=seq_offset,
            )
            new_state_container.append(memory=new_m, indicator=new_i)

        return h, new_state_container

