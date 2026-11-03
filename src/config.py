"""
=============================================================================
HSRU DeltaGrid Configuration Engine
=============================================================================

Strongly-typed, serializable configuration dataclass for the DeltaGrid
architecture. Provides validation, Hugging Face-compatible serialization,
and dynamic hardware sizing properties.
=============================================================================
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union


@dataclass
class DeltaGridConfig:
    """
    Architectural configuration for DeltaGrid models.

    Supports both Language Modeling (DeltaGridLM) and Continuous Stream /
    Robotics / Drone Physics forecasting (DeltaGridContinuous).

    Attributes:
        vocab_size (Optional[int]): Vocabulary size for token-based language models.
        input_dim (Optional[int]): Number of input channels for continuous sensory streams.
        output_dim (Optional[int]): Number of output channels for continuous control / prediction.
        d_model (int): Total hidden representation dimension. Must be divisible by num_heads.
        num_layers (int): Number of stacked 4-tier DeltaGridBlock layers.
        num_heads (int): Number of associative memory heads.
        num_slots (Optional[int]): Number of memory slots per head in dense mode (defaults to head_dim).
        grid_dim (Optional[Tuple[int, ...]]): Factorized grid coordinates (e.g. (64, 64) or (1024, 1024)).
        top_k (Optional[int]): Number of active coordinate bins per axis for Sparse Delta-PKM.
        thinking_cycles (int): Working-memory cognitive scratchpad iterations per step.
        decay_init (float): Initial value for memory retention decay alpha in (0.0, 1.0].
        dynamic_decay (bool): Whether to enable input-dependent dynamic forget gates.
        strict_slots (bool): Force exact num_slots projection even when num_slots <= head_dim.
        use_chunkwise (bool): Use parallel WY chunkwise tensor-core prefill.
        chunk_size (int): Chunk length for WY prefill.
        forecast_horizon (Optional[int]): Direct trajectory forecasting horizon for continuous signals.
        kernel_size (int): Receptive field size for Tier 1 Causal Conv1D.
        max_norm (Optional[float]): Optional gradient/state norm clipping.
        extra_fields (Dict[str, Any]): Additional downstream application-specific settings.
    """
    vocab_size: Optional[int] = 32000
    input_dim: Optional[int] = None
    output_dim: Optional[int] = None
    d_model: int = 128
    num_layers: int = 4
    num_heads: int = 4
    num_slots: Optional[int] = None
    grid_dim: Optional[Tuple[int, ...]] = None
    top_k: Optional[int] = None
    thinking_cycles: int = 0
    decay_init: float = 0.999
    dynamic_decay: bool = True
    strict_slots: bool = False
    use_chunkwise: bool = False
    chunk_size: int = 64
    forecast_horizon: Optional[int] = None
    kernel_size: int = 4
    max_norm: Optional[float] = None
    extra_fields: Dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        vocab_size: Optional[int] = 32000,
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        d_model: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        num_layers: int = 4,
        num_heads: int = 4,
        num_slots: Optional[int] = None,
        grid_dim: Optional[Tuple[int, ...]] = None,
        top_k: Optional[int] = None,
        thinking_cycles: int = 0,
        decay_init: float = 0.999,
        dynamic_decay: bool = True,
        strict_slots: bool = False,
        use_chunkwise: bool = False,
        chunk_size: int = 64,
        forecast_horizon: Optional[int] = None,
        kernel_size: int = 4,
        max_norm: Optional[float] = None,
        **kwargs: Any,
    ):
        actual_d_model = d_model if d_model is not None else (hidden_dim if hidden_dim is not None else 128)
        self.vocab_size = vocab_size
        self.input_dim = input_dim
        self.output_dim = output_dim if output_dim is not None else input_dim
        self.d_model = actual_d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_slots = num_slots
        self.grid_dim = tuple(grid_dim) if grid_dim is not None else None
        self.top_k = top_k
        self.thinking_cycles = thinking_cycles
        self.decay_init = decay_init
        self.dynamic_decay = dynamic_decay
        self.strict_slots = strict_slots
        self.use_chunkwise = use_chunkwise
        self.chunk_size = chunk_size
        self.forecast_horizon = forecast_horizon
        self.kernel_size = kernel_size
        self.max_norm = max_norm
        self.extra_fields = kwargs

        self._validate()

    def _validate(self) -> None:
        """Strict validation of mathematical and architectural constraints."""
        if self.d_model <= 0:
            raise ValueError(f"d_model must be positive, got {self.d_model}")
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {self.num_heads}")
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be cleanly divisible by num_heads ({self.num_heads}). "
                f"Head dimension would be fractional ({self.d_model / self.num_heads})."
            )
        if not (0.0 <= self.decay_init <= 1.0):
            raise ValueError(f"decay_init must be in [0.0, 1.0], got {self.decay_init}")
        if self.grid_dim is not None and self.top_k is not None:
            total_slots = math.prod(self.grid_dim)
            active_slots = self.top_k ** len(self.grid_dim)
            if active_slots > total_slots:
                raise ValueError(
                    f"Active slots ({active_slots} = {self.top_k}^{len(self.grid_dim)}) cannot exceed "
                    f"total virtual slot coordinate space ({total_slots})."
                )

    @property
    def head_dim(self) -> int:
        """Dimension per associative memory head."""
        return self.d_model // self.num_heads

    @property
    def is_pkm(self) -> bool:
        """True if configured for Sparse Product-Key Memory (PKM) mode."""
        return self.grid_dim is not None and self.top_k is not None

    @property
    def virtual_slots(self) -> int:
        """Total virtual slot capacity."""
        if self.grid_dim is not None:
            return math.prod(self.grid_dim)
        return self.num_slots if self.num_slots is not None else self.head_dim

    @property
    def active_slots(self) -> int:
        """Active slot footprint per recurrence step."""
        if self.is_pkm and self.grid_dim is not None and self.top_k is not None:
            return self.top_k ** len(self.grid_dim)
        return self.virtual_slots

    def to_dict(self) -> Dict[str, Any]:
        """Serializes configuration to a JSON-compatible Python dictionary."""
        d = {
            "vocab_size": self.vocab_size,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "num_slots": self.num_slots,
            "grid_dim": list(self.grid_dim) if self.grid_dim is not None else None,
            "top_k": self.top_k,
            "thinking_cycles": self.thinking_cycles,
            "decay_init": self.decay_init,
            "dynamic_decay": self.dynamic_decay,
            "strict_slots": self.strict_slots,
            "use_chunkwise": self.use_chunkwise,
            "chunk_size": self.chunk_size,
            "forecast_horizon": self.forecast_horizon,
            "kernel_size": self.kernel_size,
            "max_norm": self.max_norm,
        }
        if self.extra_fields:
            d.update(self.extra_fields)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DeltaGridConfig:
        """Constructs a DeltaGridConfig from a dictionary."""
        cfg_data = data.copy()
        if "grid_dim" in cfg_data and isinstance(cfg_data["grid_dim"], list):
            cfg_data["grid_dim"] = tuple(cfg_data["grid_dim"])
        return cls(**cfg_data)

    def to_json(self, path: Optional[Union[str, Path]] = None, indent: int = 2) -> Optional[str]:
        """Serializes configuration to a JSON string or file."""
        dumped = json.dumps(self.to_dict(), indent=indent)
        if path is not None:
            Path(path).write_text(dumped, encoding="utf-8")
            return None
        return dumped

    @classmethod
    def from_json(cls, path_or_str: Union[str, Path]) -> DeltaGridConfig:
        """Loads configuration from a JSON file path or raw JSON string."""
        if isinstance(path_or_str, str) and path_or_str.lstrip().startswith("{"):
            data = json.loads(path_or_str)
        else:
            path = Path(path_or_str)
            data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else json.loads(str(path_or_str))
        return cls.from_dict(data)

    def save_pretrained(self, save_directory: Union[str, Path]) -> None:
        """Hugging Face compatible config persistence."""
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        self.to_json(save_path / "config.json")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, Path]) -> DeltaGridConfig:
        """Hugging Face compatible config loading."""
        path = Path(pretrained_model_name_or_path)
        if path.is_dir():
            config_file = path / "config.json"
            if not config_file.exists():
                raise FileNotFoundError(f"No config.json found in {path}")
            return cls.from_json(config_file)
        return cls.from_json(path)

