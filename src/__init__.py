"""
=============================================================================
HSRU DeltaGrid: Standalone Hybrid State Recurrent Engine
=============================================================================

A high-performance recurrent sequence architecture that achieves 100% on-chip
SRAM memory residency (zero DRAM round trips) and strictly constant O(1) RAM
during autoregressive generation.

Public API:
  - DeltaGridConfig: Architectural and hardware configuration engine.
  - DeltaGridLM: Autoregressive zero-KV language model.
  - DeltaGridContinuous: Real-time continuous stream / drone / robotics controller.
  - DeltaGridBlock: 4-tier recurrent neural block.
  - DeltaGridLayer: Associative fast-weight memory layer.
  - DeltaGridState: Encapsulated multi-layer recurrent state container.
  - DeltaGridLayerState: Encapsulated single-layer recurrent state container.
  - DeltaGridLMOutput: Structured output container for DeltaGridLM.
  - DeltaGridContinuousOutput: Structured output container for DeltaGridContinuous.

Author: Sameer Humagain <im@hsameer.com.np>
License: MIT
=============================================================================
"""

from .config import DeltaGridConfig
from .state import DeltaGridState, DeltaGridLayerState
from .outputs import DeltaGridLMOutput, DeltaGridContinuousOutput
from .modules.memory import DeltaGridLayer, SparseDeltaPKMLayer
from .modules.block import DeltaGridBlock
from .models.language_model import DeltaGridLM
from .models.continuous import DeltaGridContinuous
from .cpp.engine.tools.export_deltagrid_binary import export_deltagrid_to_bin

__version__ = "0.2.0"
__author__ = "Sameer Humagain"
__license__ = "MIT"

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
    "export_deltagrid_to_bin",
]
