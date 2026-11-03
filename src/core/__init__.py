"""
=============================================================================
HSRU DeltaGrid Core Neural Architecture Components (Backward Compatibility)
=============================================================================

This module provides transparent backward compatibility for legacy imports
pointing to hsru_delta.core.*
=============================================================================
"""

from ..config import DeltaGridConfig
from ..state import DeltaGridState, DeltaGridLayerState
from ..outputs import DeltaGridLMOutput, DeltaGridContinuousOutput
from ..modules.memory import DeltaGridLayer, SparseDeltaPKMLayer
from ..modules.block import DeltaGridBlock
from ..models.language_model import DeltaGridLM
from ..models.continuous import DeltaGridContinuous

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
]
