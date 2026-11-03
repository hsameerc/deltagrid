"""
=============================================================================
HSRU DeltaGrid C++ Edge Engine Binary Exporter
=============================================================================

Serializes trained PyTorch DeltaGridContinuous and DeltaGridLM models into
contiguous, endian-aligned .bin files for zero-dependency execution via
the C++ edge runtime in src/cpp/engine.

Supports:
  - Precision Modes: "fp32" (default), "int8" (symmetric Q8), "fxp" (Q16.16 MCU)
  - Memory Modes: Standard Dense, Factorized Kronecker Dense, Sparse Delta-PKM
  - Multi-Horizon Trajectory Forecasting Heads & Cognitive Thinking Cycles
=============================================================================
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path
from typing import Union
import numpy as np
import torch
import torch.nn as nn

DELTAGRID_BIN_MAGIC   = 0x31474C44  # 'DLG1'
DELTAGRID_BIN_VERSION = 1

DG_MODEL_CONTINUOUS = 0
DG_MODEL_LM         = 1

DG_PREC_FP32 = 0
DG_PREC_INT8 = 1
DG_PREC_FXP  = 2

DG_FLAG_DYNAMIC_DECAY = (1 << 0)
DG_FLAG_HAS_SLOT_PROJ = (1 << 1)
DG_FLAG_TIED_LM_HEAD  = (1 << 2)


def _to_flat_f32(tensor_or_val: Union[torch.Tensor, float, np.ndarray]) -> np.ndarray:
    if isinstance(tensor_or_val, torch.Tensor):
        return tensor_or_val.detach().cpu().float().contiguous().numpy().flatten()
    elif isinstance(tensor_or_val, np.ndarray):
        return tensor_or_val.astype(np.float32).flatten()
    else:
        return np.array([float(tensor_or_val)], dtype=np.float32)


def _write_vec(f, tensor_or_val: Union[torch.Tensor, float, np.ndarray], pmode: int) -> None:
    arr = _to_flat_f32(tensor_or_val)
    if pmode == DG_PREC_FXP:
        fxp_arr = np.round(arr * 65536.0).astype(np.int32)
        f.write(fxp_arr.tobytes())
    else:
        f.write(arr.tobytes())


def _write_weight(f, tensor: torch.Tensor, pmode: int) -> None:
    arr = _to_flat_f32(tensor)
    if pmode == DG_PREC_FP32:
        f.write(arr.tobytes())
    elif pmode == DG_PREC_INT8:
        max_abs = float(np.max(np.abs(arr))) if arr.size > 0 else 0.0
        scale = (max_abs / 127.0) if max_abs > 1e-12 else 1.0
        q8 = np.clip(np.round(arr / scale), -127, 127).astype(np.int8)
        f.write(struct.pack("<f", scale))
        f.write(q8.tobytes())
    elif pmode == DG_PREC_FXP:
        fxp_arr = np.round(arr * 65536.0).astype(np.int32)
        f.write(fxp_arr.tobytes())
    else:
        raise ValueError(f"Unsupported precision mode: {pmode}")


def export_deltagrid_to_bin(
    model: nn.Module,
    save_path: Union[str, Path],
    mode: str = "fp32",
) -> Path:
    """
    Exports a DeltaGridContinuous or DeltaGridLM PyTorch model to a .bin file
    compatible with deltagrid_init_context(), deltagrid_init_context_fxp(),
    and deltagrid_llm_init_context().

    Args:
        model (nn.Module): Instance of DeltaGridContinuous or DeltaGridLM.
        save_path (Union[str, Path]): Output binary file path.
        mode (str): Precision mode: "fp32" ("float32"), "int8" ("q8"), or "fxp".

    Returns:
        Path: Resolved path to the written binary file.
    """
    model.eval()
    mode_norm = mode.strip().lower()
    if mode_norm in ("fp32", "float32", "float"):
        pmode = DG_PREC_FP32
    elif mode_norm in ("int8", "q8"):
        pmode = DG_PREC_INT8
    elif mode_norm in ("fxp", "q16.16", "fixed"):
        pmode = DG_PREC_FXP
    else:
        raise ValueError(f"Unknown precision mode '{mode}'. Choose from 'fp32', 'int8', 'fxp'.")

    is_lm = hasattr(model, "embed") and hasattr(model, "lm_head")
    model_type = DG_MODEL_LM if is_lm else DG_MODEL_CONTINUOUS

    num_layers      = len(model.blocks)
    d_model         = int(model.d_model)
    num_heads       = int(model.num_heads)
    kernel_size     = int(model.blocks[0].kernel_size)
    thinking_cycles = int(getattr(model, "thinking_cycles", 0) or 0)

    if is_lm:
        input_or_vocab   = int(model.vocab_size)
        output_dim       = int(model.vocab_size)
        forecast_horizon = 0
    else:
        input_or_vocab   = int(model.input_dim)
        output_dim       = int(model.output_dim)
        forecast_horizon = int(model.forecast_horizon or 0)

    mem0 = next((b.memory for b in model.blocks if b.has_memory and b.memory is not None), None)
    grid_dim     = mem0.grid_dim if mem0 is not None else None
    num_axes     = len(grid_dim) if grid_dim is not None else 0
    top_k        = int(mem0.top_k or 0) if mem0 is not None else 0
    actual_slots = int(mem0.actual_slots) if mem0 is not None else (d_model // num_heads)

    flags = 0
    if mem0 is not None and mem0.proj_alpha is not None:
        flags |= DG_FLAG_DYNAMIC_DECAY
    if mem0 is not None and mem0.slot_proj_k is not None:
        flags |= DG_FLAG_HAS_SLOT_PROJ
    if is_lm and model.lm_head.weight.data_ptr() == model.embed.weight.data_ptr():
        flags |= DG_FLAG_TIED_LM_HEAD

    out_path = Path(save_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "wb") as f:
        # 1. Write 64-byte Header (16 x int32)
        header_bytes = struct.pack(
            "<16i",
            DELTAGRID_BIN_MAGIC,
            DELTAGRID_BIN_VERSION,
            model_type,
            pmode,
            num_layers,
            d_model,
            num_heads,
            kernel_size,
            thinking_cycles,
            input_or_vocab,
            output_dim,
            forecast_horizon,
            num_axes,
            top_k,
            actual_slots,
            flags,
        )
        f.write(header_bytes)

        if num_axes > 0 and grid_dim is not None:
            for ax_dim in grid_dim:
                f.write(struct.pack("<i", int(ax_dim)))

        # 2. Input Feature Encoder / Token Embedding
        if is_lm:
            _write_weight(f, model.embed.weight, pmode)
        else:
            _write_weight(f, model.feature_proj.weight, pmode)
            _write_vec(f, model.feature_proj.bias, pmode)

        # 3. Stacked 4-Tier DeltaGrid Blocks
        for block in model.blocks:
            has_mem = 1 if (block.has_memory and block.memory is not None) else 0
            f.write(struct.pack("<i", has_mem))

            # Tier 0: Biological Harmonic Clock
            _write_vec(f, block.div_term, pmode)
            _write_vec(f, block.clock_scale, pmode)

            # Tier 1: Causal Depthwise Conv1D (weight shape [d_model, 1, kernel_size] -> [d_model, kernel_size])
            _write_vec(f, block.conv1d.weight.squeeze(1), pmode)
            _write_vec(f, block.conv1d.bias, pmode)
            _write_vec(f, block.norm_conv.weight, pmode)
            _write_vec(f, block.norm_conv.bias, pmode)

            # Tier 2: Associative DeltaGrid Memory Layer
            if has_mem:
                mem = block.memory
                _write_vec(f, block.scale_mem, pmode)
                _write_weight(f, mem.proj_qkv_beta.weight, pmode)
                _write_vec(f, mem.q_scale, pmode)
                _write_vec(f, mem.beta_bias.view(-1), pmode)
                _write_vec(f, mem.decay.view(-1), pmode)

                if flags & DG_FLAG_DYNAMIC_DECAY:
                    _write_weight(f, mem.proj_alpha.weight, pmode)
                    _write_vec(f, mem.proj_alpha.bias, pmode)

                if num_axes > 0:
                    for pk, pq in zip(mem.axes_proj_k, mem.axes_proj_q):
                        _write_vec(f, pk.weight, pmode)
                        _write_vec(f, pq.weight, pmode)
                elif flags & DG_FLAG_HAS_SLOT_PROJ:
                    _write_vec(f, mem.slot_proj_k.weight, pmode)
                    _write_vec(f, mem.slot_proj_q.weight, pmode)

                _write_weight(f, mem.out_proj.weight, pmode)
                _write_vec(f, mem.norm.weight, pmode)
                _write_vec(f, mem.norm.bias, pmode)

            # Tier 3: GLU Logic Reasoning Engine
            _write_vec(f, block.scale_logic, pmode)
            _write_weight(f, block.ffn[0].weight, pmode)
            _write_vec(f, block.ffn[0].bias, pmode)
            _write_weight(f, block.ffn[2].weight, pmode)
            _write_vec(f, block.ffn[2].bias, pmode)
            _write_vec(f, block.norm_ffn.weight, pmode)
            _write_vec(f, block.norm_ffn.bias, pmode)

        # 4. Final LayerNorm & Readout Head
        _write_vec(f, model.norm.weight, pmode)
        _write_vec(f, model.norm.bias, pmode)

        if is_lm:
            if not (flags & DG_FLAG_TIED_LM_HEAD):
                _write_weight(f, model.lm_head.weight, pmode)
        else:
            _write_weight(f, model.head.weight, pmode)
            _write_vec(f, model.head.bias, pmode)

            if forecast_horizon > 0 and model.forecast_head is not None:
                _write_weight(f, model.forecast_head[0].weight, pmode)
                _write_vec(f, model.forecast_head[0].bias, pmode)
                _write_weight(f, model.forecast_head[2].weight, pmode)
                _write_vec(f, model.forecast_head[2].bias, pmode)

    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export DeltaGrid PyTorch checkpoint to C++ Edge .bin")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to pretrained directory or .pt file")
    parser.add_argument("--output", type=str, required=True, help="Output .bin path")
    parser.add_argument("--mode", type=str, default="fp32", choices=["fp32", "int8", "fxp"])
    parser.add_argument("--type", type=str, default="continuous", choices=["continuous", "lm"])
    args = parser.parse_args()

    from src.models import DeltaGridContinuous, DeltaGridLM
    if args.type == "lm":
        m = DeltaGridLM.from_pretrained(args.checkpoint)
    else:
        m = DeltaGridContinuous.from_pretrained(args.checkpoint)
    export_deltagrid_to_bin(m, args.output, mode=args.mode)
    print(f"[SUCCESS] Exported {args.type} model ({args.mode}) -> {args.output}")

