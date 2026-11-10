"""
=============================================================================
HSRU DeltaGrid Low-Level Ops & Hardware Dispatch
=============================================================================

Private execution primitives connecting PyTorch to C++/CUDA kernel extensions.
=============================================================================
"""

from ._backend import has_cuda_ext, get_cuda_backend
from ._autograd import _DeltaGridAutogradFunction, _SparseDeltaPKMAutogradFunction

__all__ = [
    "has_cuda_ext",
    "get_cuda_backend",
    "_DeltaGridAutogradFunction",
    "_SparseDeltaPKMAutogradFunction",
]

