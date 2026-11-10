"""
=============================================================================
HSRU DeltaGrid Core Neural Architecture Components (Backward Compatibility)
=============================================================================

This module provides transparent backward compatibility for legacy imports
pointing to `src.core.delta_grid` or `hsru_delta.core.delta_grid`.
=============================================================================
"""

from __future__ import annotations

from typing import Optional, Tuple, List, Union
import torch

from ..config import DeltaGridConfig
from ..state import DeltaGridState, DeltaGridLayerState
from ..outputs import DeltaGridLMOutput, DeltaGridContinuousOutput
from ..ops import (
    has_cuda_ext,
    get_cuda_backend,
    _DeltaGridAutogradFunction,
    _SparseDeltaPKMAutogradFunction,
)
from ..modules.memory import DeltaGridLayer, SparseDeltaPKMLayer
from ..modules.block import DeltaGridBlock
from ..models.base import _DeltaGridSequenceModel
from ..models.language_model import DeltaGridLM
from ..models.continuous import DeltaGridContinuous

# Legacy Type Aliases
DeltaState = Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]
BlockStates = List[DeltaState]
ContinuousForwardResult = Union[
    Tuple[torch.Tensor, BlockStates],
    Tuple[torch.Tensor, BlockStates, torch.Tensor],
]

# Legacy Autograd Function Alias
DeltaGridFunction = _DeltaGridAutogradFunction

__all__ = [
    "DeltaGridConfig",
    "DeltaGridState",
    "DeltaGridLayerState",
    "DeltaGridLMOutput",
    "DeltaGridContinuousOutput",
    "DeltaGridLayer",
    "SparseDeltaPKMLayer",
    "DeltaGridBlock",
    "DeltaGridLM",
    "DeltaGridContinuous",
    "DeltaGridFunction",
    "_SparseDeltaPKMAutogradFunction",
    "_DeltaGridSequenceModel",
    "DeltaState",
    "BlockStates",
    "ContinuousForwardResult",
    "has_cuda_ext",
    "get_cuda_backend",
]
