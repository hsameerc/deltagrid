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
from ..outputs import DeltaGridLMOutput, DeltaGridContinuousOutput, DeltaGridClassificationOutput
from ..modules.memory import DeltaGridLayer, SparseDeltaPKMLayer
from ..modules.block import DeltaGridBlock
from ..models.language_model import DeltaGridLM
from ..models.continuous import DeltaGridContinuous
from ..models.classification import DeltaGridForSequenceClassification

__all__ = [
    "DeltaGridConfig",
    "DeltaGridState",
    "DeltaGridLayerState",
    "DeltaGridLMOutput",
    "DeltaGridContinuousOutput",
    "DeltaGridClassificationOutput",
    "DeltaGridLayer",
    "SparseDeltaPKMLayer",
    "DeltaGridBlock",
    "DeltaGridLM",
    "DeltaGridContinuous",
    "DeltaGridForSequenceClassification",
]
