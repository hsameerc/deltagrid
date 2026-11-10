"""
=============================================================================
HSRU DeltaGrid Real-Time Continuous Signal & Physics Engine
=============================================================================

Continuous sensory, physics, and telemetry control architecture designed for
autonomous robotics, drone flight control, and streaming physical signals.
Supports sub-millisecond single-step control and direct multi-horizon forecasting.
=============================================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, List, Union, Any
import torch
import torch.nn as nn

from ..config import DeltaGridConfig
from ..state import DeltaGridState
from ..outputs import DeltaGridContinuousOutput
from ..modules.block import DeltaGridBlock
from .base import _DeltaGridSequenceModel


class DeltaGridContinuous(_DeltaGridSequenceModel):
    """
    Real-Time Continuous Sensory / Drone / Robotics Controller.

    Supports both:
      1. Single-step streaming inference via .step() in constant O(1) RAM.
      2. Direct multi-horizon trajectory forecasting via .forecast() with zero compounding drift.
    """
    def __init__(
        self,
        input_dim: Optional[Union[int, DeltaGridConfig]] = None,
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
        forecast_horizon: Optional[int] = None,
        dynamic_decay: bool = True,
        strict_slots: bool = False,
        use_chunkwise: bool = False,
        chunk_size: int = 64,
        config: Optional[DeltaGridConfig] = None,
        **kwargs: Any,
    ):
        super().__init__()
        if isinstance(input_dim, DeltaGridConfig):
            self.config = input_dim
        elif config is not None:
            self.config = config
        else:
            actual_in = input_dim if input_dim is not None else kwargs.get("input_dim", 1)
            actual_d_model = d_model if d_model is not None else (hidden_dim if hidden_dim is not None else 128)
            self.config = DeltaGridConfig(
                input_dim=actual_in,
                output_dim=output_dim if output_dim is not None else actual_in,
                d_model=actual_d_model,
                num_layers=num_layers,
                num_heads=num_heads,
                num_slots=num_slots,
                grid_dim=grid_dim,
                top_k=top_k,
                thinking_cycles=thinking_cycles,
                decay_init=decay_init,
                dynamic_decay=dynamic_decay,
                strict_slots=strict_slots,
                use_chunkwise=use_chunkwise,
                chunk_size=chunk_size,
                forecast_horizon=forecast_horizon,
                **kwargs,
            )

        cfg = self.config
        self.input_dim = cfg.input_dim if cfg.input_dim is not None else 1
        self.output_dim = cfg.output_dim if cfg.output_dim is not None else self.input_dim
        self.d_model = cfg.d_model
        self.num_layers = cfg.num_layers
        self.num_heads = cfg.num_heads
        self.num_slots = cfg.num_slots
        self.thinking_cycles = cfg.thinking_cycles
        self.forecast_horizon = cfg.forecast_horizon

        # Continuous feature encoder: projects raw sensory channels to latent dimension d_model
        self.feature_proj = nn.Linear(self.input_dim, self.d_model)

        # Deep recurrent backbone: stacked 4-tier DeltaGrid blocks
        self.blocks = nn.ModuleList([
            DeltaGridBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                num_slots=self.num_slots,
                grid_dim=cfg.grid_dim,
                top_k=cfg.top_k,
                decay_init=cfg.decay_init,
                dynamic_decay=cfg.dynamic_decay,
                strict_slots=cfg.strict_slots,
                use_chunkwise=cfg.use_chunkwise,
                chunk_size=cfg.chunk_size,
            ) for _ in range(self.num_layers)
        ])

        # Pre-head LayerNorm
        self.norm = nn.LayerNorm(self.d_model)

        # Step-wise continuous readout head
        self.head = nn.Linear(self.d_model, self.output_dim)

        # Multi-Horizon Direct Trajectory Forecasting Head
        if self.forecast_horizon is not None:
            self.forecast_head: Optional[nn.Sequential] = nn.Sequential(
                nn.Linear(self.d_model, self.d_model),
                nn.SiLU(),
                nn.Linear(self.d_model, self.forecast_horizon * self.output_dim),
            )
        else:
            self.forecast_head = None

    @property
    def num_parameters(self) -> int:
        """Total trainable parameters in the controller."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        x: torch.Tensor,
        states: Optional[Union[DeltaGridState, List[Any]]] = None,
        seq_offset: int = 0,
        return_forecast: bool = False,
        return_all_forecasts: bool = False,
        **kwargs: Any,
    ) -> Any:
        """
        Forward pass on continuous stream x: [batch_size, seq_len, input_dim].
        """
        b, t, _ = x.shape
        h, new_states = self._forward_blocks(self.feature_proj(x), states=states, seq_offset=seq_offset)

        h_norm = self.norm(h)
        out = self.head(h_norm)

        if return_all_forecasts:
            if self.forecast_head is not None:
                forecast_all = self.forecast_head(h_norm).view(b, t, self.forecast_horizon, self.output_dim)
            else:
                forecast_all = out.unsqueeze(2)
            return DeltaGridContinuousOutput(
                out=out,
                state=new_states,
                forecast=forecast_all,
                legacy_format="out_state_forecast",
            )

        if return_forecast:
            if self.forecast_head is not None:
                forecast_out = self.forecast_head(h_norm[:, -1, :]).view(b, self.forecast_horizon, self.output_dim)
            else:
                forecast_out = out[:, -1:, :]
            return DeltaGridContinuousOutput(
                out=out,
                state=new_states,
                forecast=forecast_out,
                legacy_format="out_state_forecast",
            )

        return DeltaGridContinuousOutput(
            out=out,
            state=new_states,
            legacy_format="out_state",
        )

    def step(
        self,
        x: torch.Tensor,
        states: Optional[Union[DeltaGridState, List[Any]]] = None,
        seq_offset: int = 0,
        return_forecast: bool = False,
        **kwargs: Any,
    ) -> Any:
        """
        Single-step inference for real-time control loops in constant O(1) RAM.
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        b, t, _ = x.shape
        assert t == 1, f"step() expects continuous sample with sequence length 1, got {t}"

        h, new_states = self._step_blocks(self.feature_proj(x), states=states, seq_offset=seq_offset)
        h_norm = self.norm(h)
        out = self.head(h_norm)

        if return_forecast:
            if self.forecast_head is not None:
                forecast_out = self.forecast_head(h_norm.squeeze(1)).view(b, self.forecast_horizon, self.output_dim)
            else:
                forecast_out = out.clone()
            return DeltaGridContinuousOutput(
                out=out,
                state=new_states,
                forecast=forecast_out,
                legacy_format="out_state_forecast",
            )

        return DeltaGridContinuousOutput(
            out=out,
            state=new_states,
            legacy_format="out_state",
        )

    def compile(self, **kwargs):
        """Compiles the continuous model using torch.compile."""
        return torch.compile(self, **kwargs)

    @torch.no_grad()
    def forecast(
        self,
        history: torch.Tensor,
        forecast_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Continuous multistep forecasting with zero KV cache.
        If forecast_horizon was configured at init, uses the direct trajectory head.
        Otherwise, unrolls autoregressively using stateful .step() recurrence.
        """
        self.eval()
        if self.forecast_head is not None:
            res = self.forward(history, return_forecast=True)
            forecast_out = res.forecast if hasattr(res, "forecast") else res[2]
            if forecast_steps is not None and forecast_steps != self.forecast_horizon:
                return forecast_out[:, :forecast_steps, :]
            return forecast_out

        assert self.output_dim == self.input_dim, "Autoregressive forecast requires output_dim == input_dim"
        b, t_hist, _ = history.shape
        steps = forecast_steps if forecast_steps is not None else 20

        out_hist, states = self.forward(history)
        last_val = out_hist[:, -1:, :]

        predictions = [last_val]
        curr_val = last_val

        for step_i in range(steps - 1):
            next_pred, states = self.step(curr_val, states, seq_offset=t_hist + step_i)
            predictions.append(next_pred)
            curr_val = next_pred

        return torch.cat(predictions, dim=1)

    def save_pretrained(self, save_directory: Union[str, Path]) -> None:
        """Saves model weights and configuration."""
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        self.config.save_pretrained(save_path)
        torch.save(self.state_dict(), save_path / "pytorch_model.bin")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, Path], map_location: str = "cpu") -> DeltaGridContinuous:
        """Loads model weights and configuration from a directory."""
        load_path = Path(pretrained_model_name_or_path)
        config = DeltaGridConfig.from_pretrained(load_path)
        model = cls(config)
        weights_file = load_path / "pytorch_model.bin"
        if weights_file.exists():
            state_dict = torch.load(weights_file, map_location=map_location, weights_only=True)
            model.load_state_dict(state_dict)
        return model
