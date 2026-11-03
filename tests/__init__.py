"""
=============================================================================
HSRU DeltaGrid Automated Unit & Regression Test Suite
=============================================================================

This package contains automated unit, regression, and mathematical parity tests
for the HSRU DeltaGrid architecture.

Test Modules:
  - test_config: Architectural validation, bounds, and JSON/dict persistence.
  - test_state: Recurrent state encapsulation, TBPTT graph detachment, and legacy unpacking.
  - test_models: Standalone language model and continuous control public APIs.
  - test_sparse_pkm: Sparse Delta-PKM Householder isometry, 1-step overwriting, and AMP.
  - test_cuda_parity: Bitwise floating-point numerical parity vs native CUDA kernels.
  - test_continuous: Continuous stream filtering, single-step control, and trajectory forecasting.
  - test_integrity: Causality, chunkwise invariance, gradient health, and determinism.

Run all tests:
  python -m unittest discover tests
=============================================================================
"""

__all__ = [
    "test_config",
    "test_state",
    "test_models",
    "test_sparse_pkm",
    "test_cuda_parity",
    "test_continuous",
    "test_integrity",
]
