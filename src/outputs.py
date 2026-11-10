"""
=============================================================================
HSRU DeltaGrid Structured Model Outputs
=============================================================================

Strongly-typed Hugging Face style model output containers with tuple
unpacking fallbacks for transparent backward compatibility with legacy trainers.
=============================================================================
"""

from __future__ import annotations

from typing import Optional, Union, Iterator, Any, Tuple
import torch
from .state import DeltaGridState


class DeltaGridLMOutput:
    """
    Structured output container for DeltaGridLM.

    Attributes:
        logits (torch.Tensor): Unnormalized next-token predictions [batch_size, seq_len, vocab_size].
        state (Optional[Union[DeltaGridState, List[Any]]]): Updated recurrent state container.
        loss (Optional[torch.Tensor]): Language modeling cross-entropy loss (if labels provided).
        aux_loss (Optional[torch.Tensor]): Auxiliary density or regularization loss.
    """
    __slots__ = ("logits", "state", "loss", "aux_loss", "_legacy_format")

    def __init__(
        self,
        logits: torch.Tensor,
        state: Optional[Union[DeltaGridState, Any]] = None,
        loss: Optional[torch.Tensor] = None,
        aux_loss: Optional[torch.Tensor] = None,
        legacy_format: str = "logits_state",
    ):
        self.logits = logits
        self.state = state
        self.loss = loss
        self.aux_loss = aux_loss
        self._legacy_format = legacy_format

    def _to_tuple(self) -> Tuple[Any, ...]:
        if self._legacy_format == "logits_aux_state":
            aux = self.aux_loss if self.aux_loss is not None else torch.tensor(0.0, device=self.logits.device)
            if self.state is not None:
                return (self.logits, aux, self.state)
            return (self.logits, aux)
        elif self._legacy_format == "logits_aux":
            aux = self.aux_loss if self.aux_loss is not None else torch.tensor(0.0, device=self.logits.device)
            return (self.logits, aux)
        elif self._legacy_format == "logits_only":
            return (self.logits,)
        else:
            if self.state is not None:
                return (self.logits, self.state)
            return (self.logits,)

    def __getitem__(self, idx: Any) -> Any:
        try:
            if isinstance(idx, int):
                return self._to_tuple()[idx]
        except IndexError:
            pass
        return self.logits[idx]

    def __iter__(self) -> Iterator[Any]:
        return iter(self._to_tuple())

    def __len__(self) -> int:
        return len(self._to_tuple())

    def __getattr__(self, name: str) -> Any:
        return getattr(self.logits, name)

    def __repr__(self) -> str:
        s_repr = repr(self.state) if self.state is not None else "None"
        return f"DeltaGridLMOutput(logits={list(self.logits.shape)}, state={s_repr}, loss={self.loss})"


class DeltaGridContinuousOutput:
    """
    Structured output container for DeltaGridContinuous.

    Attributes:
        out (torch.Tensor): Next-step sensory/control action predictions [batch_size, seq_len, output_dim].
        state (Optional[Union[DeltaGridState, List[Any]]]): Updated recurrent states.
        forecast (Optional[torch.Tensor]): Direct trajectory forecast tensor.
    """
    __slots__ = ("out", "state", "forecast", "_legacy_format")

    def __init__(
        self,
        out: torch.Tensor,
        state: Optional[Union[DeltaGridState, Any]] = None,
        forecast: Optional[torch.Tensor] = None,
        legacy_format: str = "out_state",
    ):
        self.out = out
        self.state = state
        self.forecast = forecast
        self._legacy_format = legacy_format

    def _to_tuple(self) -> Tuple[Any, ...]:
        if self._legacy_format == "out_state_forecast" and self.forecast is not None:
            if self.state is not None:
                return (self.out, self.state, self.forecast)
            return (self.out, self.forecast)
        else:
            if self.state is not None:
                return (self.out, self.state)
            return (self.out,)

    def __getitem__(self, idx: Any) -> Any:
        try:
            if isinstance(idx, int):
                return self._to_tuple()[idx]
        except IndexError:
            pass
        return self.out[idx]

    def __iter__(self) -> Iterator[Any]:
        return iter(self._to_tuple())

    def __len__(self) -> int:
        return len(self._to_tuple())

    def __getattr__(self, name: str) -> Any:
        return getattr(self.out, name)

    def __repr__(self) -> str:
        f_shape = list(self.forecast.shape) if self.forecast is not None else "None"
        return f"DeltaGridContinuousOutput(out={list(self.out.shape)}, forecast={f_shape})"


class DeltaGridClassificationOutput:
    """
    Structured output container for DeltaGridForSequenceClassification.

    Provides Hugging Face compatibility with loss/logits attribute access
    alongside transparent tuple unpacking (loss, logits) or (logits, state).

    Attributes:
        logits (torch.Tensor): Unnormalized class logits [batch_size, num_classes].
        loss (Optional[torch.Tensor]): Sequence classification loss (if labels provided).
        state (Optional[Union[DeltaGridState, Any]]): Final recurrent state container.
        pooled (Optional[torch.Tensor]): Pooled sequence latent representation [batch_size, hidden_dim].
        hidden_states (Optional[Tuple[torch.Tensor, ...]]): Sequence representations prior to pooling.
    """
    __slots__ = ("logits", "loss", "state", "pooled", "hidden_states", "_legacy_format")

    def __init__(
        self,
        logits: torch.Tensor,
        loss: Optional[torch.Tensor] = None,
        state: Optional[Union[DeltaGridState, Any]] = None,
        pooled: Optional[torch.Tensor] = None,
        hidden_states: Optional[Tuple[torch.Tensor, ...]] = None,
        legacy_format: str = "logits_state",
    ):
        self.logits = logits
        self.loss = loss
        self.state = state
        self.pooled = pooled
        self.hidden_states = hidden_states
        self._legacy_format = legacy_format

    def _to_tuple(self) -> Tuple[Any, ...]:
        if self._legacy_format == "loss_logits":
            if self.loss is not None:
                return (self.loss, self.logits)
            return (self.logits,)
        elif self._legacy_format == "logits_only":
            return (self.logits,)
        else:  # "logits_state" (default)
            if self.loss is not None:
                if self.state is not None:
                    return (self.loss, self.logits, self.state)
                return (self.loss, self.logits)
            if self.state is not None:
                return (self.logits, self.state)
            return (self.logits,)

    def __getitem__(self, idx: Any) -> Any:
        try:
            if isinstance(idx, int):
                return self._to_tuple()[idx]
        except IndexError:
            pass
        return self.logits[idx]

    def __iter__(self) -> Iterator[Any]:
        return iter(self._to_tuple())

    def __len__(self) -> int:
        return len(self._to_tuple())

    def __getattr__(self, name: str) -> Any:
        return getattr(self.logits, name)

    def __repr__(self) -> str:
        s_repr = repr(self.state) if self.state is not None else "None"
        l_repr = f"{self.loss.item():.4f}" if (self.loss is not None and self.loss.numel() == 1) else str(self.loss)
        return f"DeltaGridClassificationOutput(logits={list(self.logits.shape)}, loss={l_repr}, state={s_repr})"

