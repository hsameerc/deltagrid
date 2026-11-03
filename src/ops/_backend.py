"""
=============================================================================
HSRU DeltaGrid Backend Discovery & Native Kernel Management
=============================================================================

Internal hardware backend discovery engine. Safely locates compiled C++/CUDA
extensions across development workspaces, wheel installations, and build
directories with zero user intervention.
=============================================================================
"""

from __future__ import annotations

import sys
import importlib.util
from pathlib import Path
from typing import Optional, Any

_CUDA_EXT: Optional[Any] = None
_CHECKED: bool = False


def _find_and_load_cuda_extension() -> Optional[Any]:
    """Dynamically locates and imports hsru_delta_cuda binary."""
    global _CUDA_EXT, _CHECKED
    if _CHECKED:
        return _CUDA_EXT

    _CHECKED = True

    # 1. Standard Python module import
    try:
        import hsru_delta_cuda  # type: ignore
        _CUDA_EXT = hsru_delta_cuda
        return _CUDA_EXT
    except ImportError:
        pass

    # 2. Dynamic file-system discovery relative to this file
    try:
        current_dir = Path(__file__).resolve().parent
        pkg_root = current_dir.parent.parent  # hsru-delta repository root

        search_dirs = [
            current_dir.parent,               # src/
            pkg_root,                         # repository root
            pkg_root / "build",               # setuptools build/
            Path(sys.prefix) / "Lib" / "site-packages",
        ]

        pattern = "hsru_delta_cuda*.pyd" if sys.platform == "win32" else "hsru_delta_cuda*.so"

        candidates = []
        for sdir in search_dirs:
            if sdir.is_dir():
                candidates.extend(list(sdir.glob(pattern)))
                candidates.extend(list(sdir.glob(f"**/{pattern}")))

        if candidates:
            # Pick candidate with latest modification time
            best_binary = max(candidates, key=lambda p: p.stat().st_mtime)
            spec = importlib.util.spec_from_file_location("hsru_delta_cuda", str(best_binary))
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                _CUDA_EXT = mod
                return _CUDA_EXT
    except Exception:
        pass

    _CUDA_EXT = None
    return None


def has_cuda_ext() -> bool:
    """Returns True if the fused hsru_delta_cuda C++/CUDA extension is available."""
    return _find_and_load_cuda_extension() is not None


def get_cuda_backend() -> Optional[Any]:
    """Returns the loaded hsru_delta_cuda module or None if not available."""
    return _find_and_load_cuda_extension()

