"""
=============================================================================
HSRU DeltaGrid Recurrent State Encapsulation Container
=============================================================================

Safely encapsulates multi-layer recurrent states for DeltaGrid models.
Provides explicit detach, clone, device migration, and zero-leak TBPTT boundaries
while maintaining 100% backward compatibility with legacy tuple unpacking.
=============================================================================
"""

from __future__ import annotations

from typing import Optional, List, Tuple, Union, Iterator, Any
import torch


class DeltaGridLayerState:
    """
    Encapsulated recurrent state for a single 4-tier DeltaGridBlock.

    Attributes:
        memory (Optional[torch.Tensor]): Tier 2 Associative fast-weight memory state
            S in R^{Batch x Heads x Slots x Dh} or R^{Batch x Heads x Dh x Dh}.
        indicator (Optional[torch.Tensor]): Tier 1 Causal Conv1D history buffer
            in R^{Batch x d_model x (kernel_size - 1)}.
    """
    __slots__ = ("memory", "indicator")

    def __init__(
        self,
        memory: Optional[torch.Tensor] = None,
        indicator: Optional[torch.Tensor] = None,
    ):
        self.memory = memory
        self.indicator = indicator

    def detach(self) -> DeltaGridLayerState:
        """Detaches state tensors from the autograd graph, severing backprop history."""
        return DeltaGridLayerState(
            memory=self.memory.detach() if self.memory is not None else None,
            indicator=self.indicator.detach() if self.indicator is not None else None,
        )

    def clone(self) -> DeltaGridLayerState:
        """Creates a deep copy of the state tensors."""
        return DeltaGridLayerState(
            memory=self.memory.clone() if self.memory is not None else None,
            indicator=self.indicator.clone() if self.indicator is not None else None,
        )

    def to(
        self,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> DeltaGridLayerState:
        """Migrates state tensors to specified device and/or dtype."""
        mem = self.memory
        ind = self.indicator
        if mem is not None:
            mem = mem.to(device=device, dtype=dtype) if dtype is not None else mem.to(device=device)
        if ind is not None:
            ind = ind.to(device=device, dtype=dtype) if dtype is not None else ind.to(device=device)
        return DeltaGridLayerState(memory=mem, indicator=ind)

    def __getitem__(self, idx: int) -> Optional[torch.Tensor]:
        if idx == 0:
            return self.memory
        elif idx == 1:
            return self.indicator
        raise IndexError(f"DeltaGridLayerState index out of range: {idx} (must be 0 or 1)")

    def __iter__(self) -> Iterator[Optional[torch.Tensor]]:
        yield self.memory
        yield self.indicator

    def __len__(self) -> int:
        return 2

    def __repr__(self) -> str:
        mem_shape = list(self.memory.shape) if self.memory is not None else None
        ind_shape = list(self.indicator.shape) if self.indicator is not None else None
        return f"DeltaGridLayerState(memory={mem_shape}, indicator={ind_shape})"


class DeltaGridState:
    """
    High-level recurrent state manager for DeltaGrid sequence models.

    Prevents memory leaks across sequence chunks by managing state detachment,
    device migration, and layer indexing. Supports 100% backward compatibility
    with code expecting a List[Tuple[Optional[Tensor], Optional[Tensor]]].
    """
    __slots__ = ("_layers",)

    def __init__(self, layers: Optional[List[DeltaGridLayerState]] = None):
        self._layers: List[DeltaGridLayerState] = list(layers) if layers is not None else []

    @classmethod
    def from_legacy(
        cls,
        states: Optional[Union[DeltaGridState, List[Any]]],
    ) -> Optional[DeltaGridState]:
        """Converts legacy lists of tuples into an encapsulated DeltaGridState."""
        if states is None:
            return None
        if isinstance(states, DeltaGridState):
            return states
        layer_states = []
        for s in states:
            if isinstance(s, DeltaGridLayerState):
                layer_states.append(s)
            elif isinstance(s, (tuple, list)) and len(s) >= 2:
                layer_states.append(DeltaGridLayerState(memory=s[0], indicator=s[1]))
            elif isinstance(s, (tuple, list)) and len(s) == 1:
                layer_states.append(DeltaGridLayerState(memory=s[0], indicator=None))
            else:
                layer_states.append(DeltaGridLayerState(memory=None, indicator=None))
        return cls(layer_states)

    def to_legacy(self) -> List[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]:
        """Converts to legacy List[Tuple[Tensor, Tensor]] format."""
        return [(l.memory, l.indicator) for l in self._layers]

    def detach(self) -> DeltaGridState:
        """
        Detaches all layers from the computational graph.
        Crucial for TBPTT across sequence chunks to prevent graph retention.
        """
        return DeltaGridState([l.detach() for l in self._layers])

    def clone(self) -> DeltaGridState:
        """Creates an independent clone of all layer states."""
        return DeltaGridState([l.clone() for l in self._layers])

    def to(
        self,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> DeltaGridState:
        """Migrates all layer states to specified device/dtype."""
        return DeltaGridState([l.to(device=device, dtype=dtype) for l in self._layers])

    def reset(self) -> None:
        """Clears all stored layer states."""
        self._layers.clear()

    def append(self, memory: Optional[torch.Tensor], indicator: Optional[torch.Tensor]) -> None:
        """Appends a new layer state."""
        self._layers.append(DeltaGridLayerState(memory=memory, indicator=indicator))

    def __getitem__(self, idx: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Index access returning legacy tuple (memory, indicator) for transparent backward compatibility."""
        l = self._layers[idx]
        return (l.memory, l.indicator)

    def get_layer(self, idx: int) -> DeltaGridLayerState:
        """Direct access to the strongly-typed DeltaGridLayerState object."""
        return self._layers[idx]

    def __iter__(self) -> Iterator[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]:
        for l in self._layers:
            yield (l.memory, l.indicator)

    def __len__(self) -> int:
        return len(self._layers)

    def __repr__(self) -> str:
        return f"DeltaGridState(num_layers={len(self._layers)})"

