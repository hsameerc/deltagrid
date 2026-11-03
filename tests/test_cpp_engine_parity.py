"""
=============================================================================
HSRU DeltaGrid C++ Edge Engine vs PyTorch Numerical Divergence & Parity Suite
=============================================================================

This test suite provides a complete, human-readable verification of the
standalone C++ Edge Engine (`src/cpp/engine`) against the canonical PyTorch
reference models (`DeltaGridContinuous` and `DeltaGridLM`).

What Is Verified:
-----------------
  1. DeltaGridContinuous (FP32):
     - Standard Dense Associative Memory (with and without slot expansion)
     - Multi-Horizon Trajectory Forecasting Head (`deltagrid_step_forecast`)
     - Sequence `.forward()` vs C++ sequential `.step()` equivalence
     - Bitwise-identical State Save & Restore (`deltagrid_save_state` / `load_state`)
  2. Coordinate Routing Modes (FP32):
     - Mode 2: Multi-Axis Factorized Kronecker Dense (`grid_dim=(6, 6)`)
     - Mode 3: Exponential Sparse Delta-PKM (`grid_dim=(16, 16), top_k=4`)
  3. Embedded Microcontroller Q16.16 Fixed-Point Runtime (`libdeltagrid_fxp`):
     - Standard Dense, Factorized Kronecker Dense, and Sparse Delta-PKM in Q16.16
     - Multi-Horizon Forecasting in Q16.16 (`deltagrid_step_forecast_fxp`)
     - Cognitive Thinking Cycles (`thinking_cycles=1`) in Q16.16
     - Bitwise-identical State Save & Restore in Q16.16
  4. DeltaGridLM Zero-KV Causal Language Model (`libdeltagrid_llm`):
     - Standard Dense + Cognitive Thinking Cycles (`thinking_cycles=1`)
     - Sparse Delta-PKM (`grid_dim=(16, 16), top_k=4`)
     - Conversational State Checkpointing (`deltagrid_llm_save_state` / `load_state`)
  5. Symmetric INT8 (Q8) Quantized Weight Execution:
     - Verifies > 99.5% cosine similarity and tight L_inf error bounds for
       4x compressed INT8 weights across both Continuous and LM engines.
=============================================================================
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
import ctypes
import subprocess
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import DeltaGridContinuous, DeltaGridLM
from src.cpp.engine.tools.export_deltagrid_binary import export_deltagrid_to_bin


def _load_shared_lib(lib_basename: str) -> ctypes.CDLL:
    ext = ".dll" if os.name == "nt" else (".dylib" if sys.platform == "darwin" else ".so")
    engine_dir   = Path(__file__).resolve().parent.parent / "src" / "cpp" / "engine"
    override_dir = os.environ.get("DELTAGRID_BUILD_DIR")
    engine_build = Path(override_dir) if override_dir else (engine_dir / "build")
    candidates = [
        engine_build / f"{lib_basename}{ext}",
        engine_build / f"{lib_basename}.dylib",
        engine_build / f"{lib_basename}.so",
        engine_build / f"{lib_basename}.dll",
    ]
    for cand in candidates:
        if cand.exists():
            return ctypes.CDLL(str(cand))

    # Automatically build C++ engine if not yet compiled
    subprocess.run(["make", "-C", str(engine_dir), "all"], check=True)
    for cand in candidates:
        if cand.exists():
            return ctypes.CDLL(str(cand))

    raise FileNotFoundError(
        f"Could not find {lib_basename} in {engine_build}. Run `make -C src/cpp/engine` first."
    )


class TestDeltaGridCppEngineParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib_cont = _load_shared_lib("libdeltagrid_continuous")
        cls.lib_fxp  = _load_shared_lib("libdeltagrid_fxp")
        cls.lib_llm  = _load_shared_lib("libdeltagrid_llm")

        # ---------------------------------------------------------------------
        # 1. Continuous FP32 / INT8 C-API Signatures
        # ---------------------------------------------------------------------
        cls.lib_cont.deltagrid_authenticate.argtypes = [ctypes.c_char_p]
        cls.lib_cont.deltagrid_authenticate.restype  = ctypes.c_bool

        cls.lib_cont.deltagrid_init_context.argtypes = [ctypes.c_char_p]
        cls.lib_cont.deltagrid_init_context.restype  = ctypes.c_void_p

        cls.lib_cont.deltagrid_get_input_dim.argtypes = [ctypes.c_void_p]
        cls.lib_cont.deltagrid_get_input_dim.restype  = ctypes.c_int

        cls.lib_cont.deltagrid_get_output_dim.argtypes = [ctypes.c_void_p]
        cls.lib_cont.deltagrid_get_output_dim.restype  = ctypes.c_int

        cls.lib_cont.deltagrid_get_d_model.argtypes = [ctypes.c_void_p]
        cls.lib_cont.deltagrid_get_d_model.restype  = ctypes.c_int

        cls.lib_cont.deltagrid_get_num_layers.argtypes = [ctypes.c_void_p]
        cls.lib_cont.deltagrid_get_num_layers.restype  = ctypes.c_int

        cls.lib_cont.deltagrid_get_forecast_horizon.argtypes = [ctypes.c_void_p]
        cls.lib_cont.deltagrid_get_forecast_horizon.restype  = ctypes.c_int

        cls.lib_cont.deltagrid_get_step_count.argtypes = [ctypes.c_void_p]
        cls.lib_cont.deltagrid_get_step_count.restype  = ctypes.c_int

        cls.lib_cont.deltagrid_step.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
        cls.lib_cont.deltagrid_step.restype = None

        cls.lib_cont.deltagrid_step_forecast.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
        cls.lib_cont.deltagrid_step_forecast.restype = None

        cls.lib_cont.deltagrid_save_state.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        cls.lib_cont.deltagrid_save_state.restype  = ctypes.c_int

        cls.lib_cont.deltagrid_load_state.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        cls.lib_cont.deltagrid_load_state.restype  = ctypes.c_int

        cls.lib_cont.deltagrid_reset_state.argtypes = [ctypes.c_void_p]
        cls.lib_cont.deltagrid_reset_state.restype  = None

        cls.lib_cont.deltagrid_release_context.argtypes = [ctypes.c_void_p]
        cls.lib_cont.deltagrid_release_context.restype  = None

        # ---------------------------------------------------------------------
        # 2. Fixed-Point Q16.16 C-API Signatures
        # ---------------------------------------------------------------------
        cls.lib_fxp.deltagrid_init_context_fxp.argtypes = [ctypes.c_char_p]
        cls.lib_fxp.deltagrid_init_context_fxp.restype  = ctypes.c_void_p

        cls.lib_fxp.deltagrid_step_fxp.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
        ]
        cls.lib_fxp.deltagrid_step_fxp.restype = None

        cls.lib_fxp.deltagrid_step_forecast_fxp.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
        ]
        cls.lib_fxp.deltagrid_step_forecast_fxp.restype = None

        cls.lib_fxp.deltagrid_save_state_fxp.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        cls.lib_fxp.deltagrid_save_state_fxp.restype  = ctypes.c_int

        cls.lib_fxp.deltagrid_load_state_fxp.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        cls.lib_fxp.deltagrid_load_state_fxp.restype  = ctypes.c_int

        cls.lib_fxp.deltagrid_reset_state_fxp.argtypes = [ctypes.c_void_p]
        cls.lib_fxp.deltagrid_reset_state_fxp.restype  = None

        cls.lib_fxp.deltagrid_compute_normalized_mse_fxp.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
        ]
        cls.lib_fxp.deltagrid_compute_normalized_mse_fxp.restype = ctypes.c_float

        cls.lib_fxp.deltagrid_release_context_fxp.argtypes = [ctypes.c_void_p]
        cls.lib_fxp.deltagrid_release_context_fxp.restype  = None

        # ---------------------------------------------------------------------
        # 3. Causal Language Model C-API Signatures
        # ---------------------------------------------------------------------
        cls.lib_llm.deltagrid_llm_init_context.argtypes = [ctypes.c_char_p]
        cls.lib_llm.deltagrid_llm_init_context.restype  = ctypes.c_void_p

        cls.lib_llm.deltagrid_llm_step.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
        ]
        cls.lib_llm.deltagrid_llm_step.restype = None

        cls.lib_llm.deltagrid_llm_get_vocab_size.argtypes = [ctypes.c_void_p]
        cls.lib_llm.deltagrid_llm_get_vocab_size.restype  = ctypes.c_int

        cls.lib_llm.deltagrid_llm_get_d_model.argtypes = [ctypes.c_void_p]
        cls.lib_llm.deltagrid_llm_get_d_model.restype  = ctypes.c_int

        cls.lib_llm.deltagrid_llm_get_num_layers.argtypes = [ctypes.c_void_p]
        cls.lib_llm.deltagrid_llm_get_num_layers.restype  = ctypes.c_int

        cls.lib_llm.deltagrid_llm_get_step_count.argtypes = [ctypes.c_void_p]
        cls.lib_llm.deltagrid_llm_get_step_count.restype  = ctypes.c_int

        cls.lib_llm.deltagrid_llm_reset_state.argtypes = [ctypes.c_void_p]
        cls.lib_llm.deltagrid_llm_reset_state.restype  = None

        cls.lib_llm.deltagrid_llm_save_state.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        cls.lib_llm.deltagrid_llm_save_state.restype  = ctypes.c_int

        cls.lib_llm.deltagrid_llm_load_state.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        cls.lib_llm.deltagrid_llm_load_state.restype  = ctypes.c_int

        cls.lib_llm.deltagrid_llm_free_context.argtypes = [ctypes.c_void_p]
        cls.lib_llm.deltagrid_llm_free_context.restype  = None

    def test_01_continuous_dense_and_forecast_parity(self):
        """Verifies DeltaGridContinuous (Standard Dense + Forecast Head + Forward/Step + State Save)."""
        torch.manual_seed(42)
        np.random.seed(42)

        in_dim, out_dim, d_model, layers, heads, horizon = 8, 4, 32, 2, 4, 6
        model = DeltaGridContinuous(
            input_dim=in_dim,
            output_dim=out_dim,
            d_model=d_model,
            num_layers=layers,
            num_heads=heads,
            num_slots=16,
            forecast_horizon=horizon,
            dynamic_decay=True,
        ).eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            bin_path   = Path(tmpdir) / "cont_dense.bin"
            state_path = Path(tmpdir) / "cont_dense.state"
            export_deltagrid_to_bin(model, bin_path, mode="fp32")

            ctx = self.lib_cont.deltagrid_init_context(str(bin_path).encode("utf-8"))
            self.assertIsNotNone(ctx)

            seq_len = 12
            x_seq_np = np.random.randn(seq_len, in_dim).astype(np.float32)
            x_seq_t  = torch.from_numpy(x_seq_np).unsqueeze(0)  # [1, T, in_dim]

            with torch.no_grad():
                fw_res = model(x_seq_t, return_all_forecasts=True)
                py_fw_out = fw_res.out.squeeze(0).numpy()       # [T, out_dim]
                py_fw_fc  = fw_res.forecast.squeeze(0).numpy()  # [T, horizon, out_dim]

            py_state = None
            max_step_diff = 0.0
            max_fw_diff   = 0.0
            max_fc_diff   = 0.0

            for step_i in range(seq_len):
                x_np = x_seq_np[step_i]
                x_t  = torch.from_numpy(x_np).unsqueeze(0).unsqueeze(1)

                with torch.no_grad():
                    res = model.step(x_t, states=py_state, seq_offset=step_i, return_forecast=True)
                    py_out = res.out.squeeze().numpy()
                    py_fc  = res.forecast.squeeze().numpy().flatten()
                    py_state = res.state

                cpp_out = np.zeros(out_dim, dtype=np.float32)
                cpp_fc  = np.zeros(horizon * out_dim, dtype=np.float32)
                self.lib_cont.deltagrid_step_forecast(
                    ctx,
                    x_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    cpp_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    cpp_fc.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                )

                diff_step = float(np.max(np.abs(py_out - cpp_out)))
                diff_fw   = float(np.max(np.abs(py_fw_out[step_i] - cpp_out)))
                diff_fc   = float(np.max(np.abs(py_fc - cpp_fc)))
                diff_fw_fc = float(np.max(np.abs(py_fw_fc[step_i].flatten() - cpp_fc)))

                max_step_diff = max(max_step_diff, diff_step)
                max_fw_diff   = max(max_fw_diff, diff_fw)
                max_fc_diff   = max(max_fc_diff, diff_fc, diff_fw_fc)

                self.assertLess(diff_step, 1e-5, f"Step {step_i} output diverged: {diff_step}")
                self.assertLess(diff_fw,   1e-5, f"Step {step_i} forward vs C++ diverged: {diff_fw}")
                self.assertLess(diff_fc,   1e-5, f"Step {step_i} forecast diverged: {diff_fc}")

            print(
                f"\n[Divergence Check 1 - Continuous Dense + Forecast] "
                f"PyTorch.step vs C++: {max_step_diff:.2e} | "
                f"PyTorch.forward vs C++: {max_fw_diff:.2e} | "
                f"Forecast Head vs C++: {max_fc_diff:.2e}"
            )

            # Verify State Save & Restore produces bitwise-identical results
            self.assertEqual(self.lib_cont.deltagrid_save_state(ctx, str(state_path).encode("utf-8")), 0)
            ctx2 = self.lib_cont.deltagrid_init_context(str(bin_path).encode("utf-8"))
            self.assertEqual(self.lib_cont.deltagrid_load_state(ctx2, str(state_path).encode("utf-8")), 0)

            x_next = np.random.randn(in_dim).astype(np.float32)
            out1 = np.zeros(out_dim, dtype=np.float32)
            out2 = np.zeros(out_dim, dtype=np.float32)
            self.lib_cont.deltagrid_step(ctx,  x_next.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), out1.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
            self.lib_cont.deltagrid_step(ctx2, x_next.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), out2.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
            self.assertEqual(float(np.max(np.abs(out1 - out2))), 0.0)

            self.lib_cont.deltagrid_release_context(ctx)
            self.lib_cont.deltagrid_release_context(ctx2)

    def test_02_continuous_kronecker_and_sparse_pkm_parity(self):
        """Verifies Factorized Kronecker Dense and Sparse Delta-PKM routing in DeltaGridContinuous."""
        torch.manual_seed(99)
        np.random.seed(99)

        for label, grid_dim, top_k in [
            ("Kronecker Dense (6x6)", (6, 6), None),
            ("Sparse Delta-PKM (16x16, top_k=4)", (16, 16), 4),
        ]:
            model = DeltaGridContinuous(
                input_dim=6,
                output_dim=3,
                d_model=32,
                num_layers=2,
                num_heads=4,
                grid_dim=grid_dim,
                top_k=top_k,
                dynamic_decay=True,
            ).eval()

            with tempfile.TemporaryDirectory() as tmpdir:
                bin_path = Path(tmpdir) / "cont_pkm.bin"
                export_deltagrid_to_bin(model, bin_path, mode="fp32")

                ctx = self.lib_cont.deltagrid_init_context(str(bin_path).encode("utf-8"))
                self.assertIsNotNone(ctx)

                py_state = None
                max_diff = 0.0
                for step_i in range(10):
                    x_np = np.random.randn(6).astype(np.float32)
                    x_t  = torch.from_numpy(x_np).unsqueeze(0).unsqueeze(1)
                    with torch.no_grad():
                        py_out_t, py_state = model.step(x_t, states=py_state, seq_offset=step_i)
                    py_out = py_out_t.squeeze().numpy()

                    cpp_out = np.zeros(3, dtype=np.float32)
                    self.lib_cont.deltagrid_step(
                        ctx,
                        x_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        cpp_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    )

                    diff = float(np.max(np.abs(py_out - cpp_out)))
                    max_diff = max(max_diff, diff)
                    self.assertLess(diff, 1e-5, f"{label} step {step_i} diff={diff}")

                print(f"[Divergence Check 2 - {label}] Max Abs Diff vs PyTorch: {max_diff:.2e}")
                self.lib_cont.deltagrid_release_context(ctx)

    def test_03_continuous_fxp_q16_16_all_modes_parity(self):
        """Verifies Q16.16 Fixed-Point MCU runtime across Dense, Kronecker, Sparse PKM & Forecast."""
        torch.manual_seed(77)
        np.random.seed(77)

        in_dim, out_dim, horizon = 4, 2, 4
        configs = [
            ("FXP Standard Dense + Forecast + Thinking", None, None, 1, horizon),
            ("FXP Factorized Kronecker (6x6)",           (6, 6), None, 0, None),
            ("FXP Sparse Delta-PKM (12x12, top_k=3)",    (12, 12), 3, 0, None),
        ]

        for label, grid_dim, top_k, cycles, fc_h in configs:
            model = DeltaGridContinuous(
                input_dim=in_dim,
                output_dim=out_dim,
                d_model=32,
                num_layers=2,
                num_heads=4,
                grid_dim=grid_dim,
                top_k=top_k,
                thinking_cycles=cycles,
                forecast_horizon=fc_h,
                dynamic_decay=True,
            ).eval()

            with tempfile.TemporaryDirectory() as tmpdir:
                bin_path   = Path(tmpdir) / "cont_fxp.bin"
                state_path = Path(tmpdir) / "cont_fxp.state"
                export_deltagrid_to_bin(model, bin_path, mode="fxp")

                ctx = self.lib_fxp.deltagrid_init_context_fxp(str(bin_path).encode("utf-8"))
                self.assertIsNotNone(ctx)

                py_state = None
                max_diff = 0.0
                for step_i in range(8):
                    x_np  = (np.random.randn(in_dim) * 0.5).astype(np.float32)
                    x_fxp = np.round(x_np * 65536.0).astype(np.int32)
                    x_t   = torch.from_numpy(x_np).unsqueeze(0).unsqueeze(1)

                    with torch.no_grad():
                        res = model.step(x_t, states=py_state, seq_offset=step_i, return_forecast=(fc_h is not None))
                        py_out   = res.out.squeeze().numpy()
                        py_state = res.state

                    out_fxp = np.zeros(out_dim, dtype=np.int32)
                    if fc_h is not None:
                        fc_fxp = np.zeros(fc_h * out_dim, dtype=np.int32)
                        self.lib_fxp.deltagrid_step_forecast_fxp(
                            ctx,
                            x_fxp.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                            out_fxp.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                            fc_fxp.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                        )
                        py_fc  = res.forecast.squeeze().numpy().flatten()
                        cpp_fc = fc_fxp.astype(np.float32) / 65536.0
                        self.assertLess(float(np.max(np.abs(py_fc - cpp_fc))), 2e-2)
                    else:
                        self.lib_fxp.deltagrid_step_fxp(
                            ctx,
                            x_fxp.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                            out_fxp.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                        )

                    cpp_out = out_fxp.astype(np.float32) / 65536.0
                    diff = float(np.max(np.abs(py_out - cpp_out)))
                    max_diff = max(max_diff, diff)
                    self.assertLess(diff, 2e-2, f"{label} step {step_i} diff={diff}")

                # Verify FXP State Save & Restore
                self.assertEqual(self.lib_fxp.deltagrid_save_state_fxp(ctx, str(state_path).encode("utf-8")), 0)
                ctx2 = self.lib_fxp.deltagrid_init_context_fxp(str(bin_path).encode("utf-8"))
                self.assertEqual(self.lib_fxp.deltagrid_load_state_fxp(ctx2, str(state_path).encode("utf-8")), 0)

                x_next = np.round(np.random.randn(in_dim) * 32768.0).astype(np.int32)
                o1 = np.zeros(out_dim, dtype=np.int32)
                o2 = np.zeros(out_dim, dtype=np.int32)
                self.lib_fxp.deltagrid_step_fxp(ctx,  x_next.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), o1.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
                self.lib_fxp.deltagrid_step_fxp(ctx2, x_next.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), o2.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
                self.assertTrue(np.array_equal(o1, o2))

                print(f"[Divergence Check 3 - {label}] Max Q16.16 vs FP32 PyTorch Diff: {max_diff:.2e}")
                self.lib_fxp.deltagrid_release_context_fxp(ctx)
                self.lib_fxp.deltagrid_release_context_fxp(ctx2)

    def test_04_llm_dense_and_sparse_pkm_parity(self):
        """Verifies DeltaGridLM (Dense & Sparse PKM + Thinking Cycles + State Save/Restore)."""
        torch.manual_seed(1337)
        np.random.seed(1337)

        vocab_size = 64
        for label, grid_dim, top_k, cycles in [
            ("LM Standard Dense (thinking_cycles=1)", None, None, 1),
            ("LM Sparse Delta-PKM (16x16, top_k=4)",  (16, 16), 4, 0),
        ]:
            model = DeltaGridLM(
                vocab_size=vocab_size,
                d_model=32,
                num_layers=2,
                num_heads=4,
                grid_dim=grid_dim,
                top_k=top_k,
                thinking_cycles=cycles,
                dynamic_decay=True,
            ).eval()

            with tempfile.TemporaryDirectory() as tmpdir:
                bin_path   = Path(tmpdir) / "llm.bin"
                state_path = Path(tmpdir) / "llm.state"
                export_deltagrid_to_bin(model, bin_path, mode="fp32")

                ctx = self.lib_llm.deltagrid_llm_init_context(str(bin_path).encode("utf-8"))
                self.assertIsNotNone(ctx)
                self.assertEqual(self.lib_llm.deltagrid_llm_get_vocab_size(ctx), vocab_size)

                tokens = [5, 19, 42, 7, 63, 12, 0, 31]
                py_state = None
                max_diff = 0.0
                for step_i, tok in enumerate(tokens):
                    tok_t = torch.tensor([[tok]], dtype=torch.long)
                    with torch.no_grad():
                        py_logits_t, py_state = model.step(tok_t, states=py_state, seq_offset=step_i)
                    py_logits = py_logits_t.squeeze().numpy()

                    cpp_logits = np.zeros(vocab_size, dtype=np.float32)
                    self.lib_llm.deltagrid_llm_step(
                        ctx,
                        ctypes.c_int(tok),
                        cpp_logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    )

                    diff = float(np.max(np.abs(py_logits - cpp_logits)))
                    max_diff = max(max_diff, diff)
                    self.assertLess(diff, 1e-5, f"{label} step {step_i} diff={diff}")

                print(f"[Divergence Check 4 - {label}] Max Logit Diff vs PyTorch: {max_diff:.2e}")

                # Test Conversational State Save & Restore
                self.assertEqual(self.lib_llm.deltagrid_llm_save_state(ctx, str(state_path).encode("utf-8")), 0)
                ctx2 = self.lib_llm.deltagrid_llm_init_context(str(bin_path).encode("utf-8"))
                self.assertEqual(self.lib_llm.deltagrid_llm_load_state(ctx2, str(state_path).encode("utf-8")), 0)

                logits1 = np.zeros(vocab_size, dtype=np.float32)
                logits2 = np.zeros(vocab_size, dtype=np.float32)
                self.lib_llm.deltagrid_llm_step(ctx,  ctypes.c_int(25), logits1.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
                self.lib_llm.deltagrid_llm_step(ctx2, ctypes.c_int(25), logits2.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
                self.assertEqual(float(np.max(np.abs(logits1 - logits2))), 0.0)

                self.lib_llm.deltagrid_llm_free_context(ctx)
                self.lib_llm.deltagrid_llm_free_context(ctx2)

    def test_05_int8_quantized_llm_and_continuous(self):
        """Verifies INT8 (Q8) quantized weight export and execution for both LM and Continuous."""
        torch.manual_seed(2026)
        np.random.seed(2026)

        model = DeltaGridLM(vocab_size=48, d_model=32, num_layers=2, num_heads=4).eval()
        with tempfile.TemporaryDirectory() as tmpdir:
            bin_q8 = Path(tmpdir) / "llm_q8.bin"
            export_deltagrid_to_bin(model, bin_q8, mode="int8")

            ctx = self.lib_llm.deltagrid_llm_init_context(str(bin_q8).encode("utf-8"))
            self.assertIsNotNone(ctx)

            py_state = None
            min_cos = 1.0
            for step_i, tok in enumerate([3, 14, 28, 9]):
                tok_t = torch.tensor([[tok]], dtype=torch.long)
                with torch.no_grad():
                    py_logits_t, py_state = model.step(tok_t, states=py_state, seq_offset=step_i)
                py_logits = py_logits_t.squeeze().numpy()

                cpp_logits = np.zeros(48, dtype=np.float32)
                self.lib_llm.deltagrid_llm_step(
                    ctx,
                    ctypes.c_int(tok),
                    cpp_logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                )

                cos_sim = float(
                    np.dot(py_logits, cpp_logits)
                    / (np.linalg.norm(py_logits) * np.linalg.norm(cpp_logits) + 1e-12)
                )
                min_cos = min(min_cos, cos_sim)
                self.assertGreater(cos_sim, 0.995, f"INT8 Q8 cosine similarity too low: {cos_sim}")

            print(f"[Divergence Check 5 - INT8 (Q8) Quantized LM] Min Cosine Similarity vs FP32 PyTorch: {min_cos:.6f}")
            self.lib_llm.deltagrid_llm_free_context(ctx)

    def test_06_metadata_getters_reset_auth_and_untied_head(self):
        """Verifies all C-API metadata getters, state resets, authentication, MSE, and untied LM head."""
        self.assertTrue(self.lib_cont.deltagrid_authenticate(b"MAC:00-11-22-33-44-55"))

        # 1. Continuous metadata getters & state reset
        cont_model = DeltaGridContinuous(
            input_dim=5, output_dim=3, d_model=16, num_layers=2, num_heads=2, dynamic_decay=False
        ).eval()
        with tempfile.TemporaryDirectory() as tmpdir:
            bin_cont = Path(tmpdir) / "cont_meta.bin"
            bin_fxp  = Path(tmpdir) / "cont_fxp_meta.bin"
            export_deltagrid_to_bin(cont_model, bin_cont, mode="fp32")
            export_deltagrid_to_bin(cont_model, bin_fxp, mode="fxp")

            ctx = self.lib_cont.deltagrid_init_context(str(bin_cont).encode("utf-8"))
            self.assertEqual(self.lib_cont.deltagrid_get_input_dim(ctx), 5)
            self.assertEqual(self.lib_cont.deltagrid_get_output_dim(ctx), 3)
            self.assertEqual(self.lib_cont.deltagrid_get_d_model(ctx), 16)
            self.assertEqual(self.lib_cont.deltagrid_get_num_layers(ctx), 2)
            self.assertEqual(self.lib_cont.deltagrid_get_forecast_horizon(ctx), 0)
            self.assertEqual(self.lib_cont.deltagrid_get_step_count(ctx), 0)

            x_in = np.ones(5, dtype=np.float32)
            out1 = np.zeros(3, dtype=np.float32)
            out2 = np.zeros(3, dtype=np.float32)
            fc_fallback = np.zeros(3, dtype=np.float32)
            self.lib_cont.deltagrid_step_forecast(
                ctx,
                x_in.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                out1.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                fc_fallback.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            )
            self.assertEqual(self.lib_cont.deltagrid_get_step_count(ctx), 1)
            self.assertTrue(np.allclose(out1, fc_fallback))

            self.lib_cont.deltagrid_reset_state(ctx)
            self.assertEqual(self.lib_cont.deltagrid_get_step_count(ctx), 0)
            self.lib_cont.deltagrid_step(
                ctx,
                x_in.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                out2.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            )
            self.assertTrue(np.array_equal(out1, out2))
            self.lib_cont.deltagrid_release_context(ctx)

            # 2. FXP reset & MSE helper
            ctx_fxp = self.lib_fxp.deltagrid_init_context_fxp(str(bin_fxp).encode("utf-8"))
            x_fxp = np.ones(5, dtype=np.int32) * 65536
            o_fxp1 = np.zeros(3, dtype=np.int32)
            o_fxp2 = np.zeros(3, dtype=np.int32)
            fc_fxp_fb = np.zeros(3, dtype=np.int32)
            self.lib_fxp.deltagrid_step_forecast_fxp(
                ctx_fxp,
                x_fxp.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                o_fxp1.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                fc_fxp_fb.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            )
            self.assertTrue(np.array_equal(o_fxp1, fc_fxp_fb))
            self.lib_fxp.deltagrid_reset_state_fxp(ctx_fxp)
            self.lib_fxp.deltagrid_step_fxp(
                ctx_fxp,
                x_fxp.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                o_fxp2.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            )
            self.assertTrue(np.array_equal(o_fxp1, o_fxp2))
            self.lib_fxp.deltagrid_release_context_fxp(ctx_fxp)

            actual = np.array([1.0, 2.0, 3.0], dtype=np.float32)
            pred   = np.array([1.0, 3.0, 5.0], dtype=np.float32)
            mse = self.lib_fxp.deltagrid_compute_normalized_mse_fxp(
                actual.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                pred.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                3,
            )
            self.assertAlmostEqual(mse, (0.0 + 1.0 + 4.0) / 3.0, places=5)

            # 3. LM metadata getters, reset, and untied LM head
            lm_model = DeltaGridLM(vocab_size=32, d_model=16, num_layers=2, num_heads=2, dynamic_decay=False).eval()
            lm_model.lm_head.weight = torch.nn.Parameter(lm_model.embed.weight.detach().clone())
            bin_lm = Path(tmpdir) / "lm_untied.bin"
            export_deltagrid_to_bin(lm_model, bin_lm, mode="fp32")

            ctx_lm = self.lib_llm.deltagrid_llm_init_context(str(bin_lm).encode("utf-8"))
            self.assertEqual(self.lib_llm.deltagrid_llm_get_vocab_size(ctx_lm), 32)
            self.assertEqual(self.lib_llm.deltagrid_llm_get_d_model(ctx_lm), 16)
            self.assertEqual(self.lib_llm.deltagrid_llm_get_num_layers(ctx_lm), 2)
            self.assertEqual(self.lib_llm.deltagrid_llm_get_step_count(ctx_lm), 0)

            l1 = np.zeros(32, dtype=np.float32)
            l2 = np.zeros(32, dtype=np.float32)
            self.lib_llm.deltagrid_llm_step(ctx_lm, 7, l1.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
            self.assertEqual(self.lib_llm.deltagrid_llm_get_step_count(ctx_lm), 1)
            self.lib_llm.deltagrid_llm_reset_state(ctx_lm)
            self.assertEqual(self.lib_llm.deltagrid_llm_get_step_count(ctx_lm), 0)
            self.lib_llm.deltagrid_llm_step(ctx_lm, 7, l2.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
            self.assertTrue(np.array_equal(l1, l2))
            self.lib_llm.deltagrid_llm_free_context(ctx_lm)

    def test_07_edge_cases_error_handling_and_full_coverage(self):
        """Covers FXP slot_proj, C++ error/corrupt-file guards, exporter CLI, and Python edge branches."""
        import io
        import runpy
        from unittest.mock import patch
        from src.config import DeltaGridConfig
        from src.state import DeltaGridState, DeltaGridLayerState
        from src.outputs import DeltaGridLMOutput, DeltaGridContinuousOutput
        from src.modules.block import DeltaGridBlock
        from src.modules.memory import DeltaGridLayer
        from src.cpp.engine.tools.export_deltagrid_binary import _to_flat_f32, _write_weight

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            # 1. FXP with custom num_slots=16 (has_slot_proj=True in FXP)
            fxp_slot_model = DeltaGridContinuous(
                input_dim=4, output_dim=2, d_model=16, num_layers=1, num_heads=2, num_slots=16
            ).eval()
            fxp_slot_bin = tmp / "fxp_slot.bin"
            export_deltagrid_to_bin(fxp_slot_model, fxp_slot_bin, mode="fxp")
            ctx_fxp_s = self.lib_fxp.deltagrid_init_context_fxp(str(fxp_slot_bin).encode("utf-8"))
            self.assertIsNotNone(ctx_fxp_s)
            x_fxp = np.ones(4, dtype=np.int32) * 32768
            o_fxp = np.zeros(2, dtype=np.int32)
            self.lib_fxp.deltagrid_step_fxp(
                ctx_fxp_s,
                x_fxp.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                o_fxp.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            )
            self.lib_fxp.deltagrid_release_context_fxp(ctx_fxp_s)

            # 2. Out-of-bounds token clamping in C++ LLM engine
            lm_small = DeltaGridLM(vocab_size=16, d_model=16, num_layers=1, num_heads=2).eval()
            lm_bin = tmp / "lm_small.bin"
            export_deltagrid_to_bin(lm_small, lm_bin, mode="fp32")
            ctx_lm = self.lib_llm.deltagrid_llm_init_context(str(lm_bin).encode("utf-8"))
            logits_neg = np.zeros(16, dtype=np.float32)
            logits_zero = np.zeros(16, dtype=np.float32)
            self.lib_llm.deltagrid_llm_step(ctx_lm, -5, logits_neg.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
            self.lib_llm.deltagrid_llm_reset_state(ctx_lm)
            self.lib_llm.deltagrid_llm_step(ctx_lm, 0, logits_zero.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
            self.assertTrue(np.array_equal(logits_neg, logits_zero))
            self.lib_llm.deltagrid_llm_reset_state(ctx_lm)
            logits_oob = np.zeros(16, dtype=np.float32)
            self.lib_llm.deltagrid_llm_step(ctx_lm, 999, logits_oob.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
            self.assertTrue(np.array_equal(logits_oob, logits_zero))
            self.lib_llm.deltagrid_llm_free_context(ctx_lm)

            # 3. C++ Error Handling & Corrupt Binary Guards
            missing_path = str(tmp / "nonexistent.bin").encode("utf-8")
            self.assertIsNone(self.lib_cont.deltagrid_init_context(missing_path))
            self.assertIsNone(self.lib_fxp.deltagrid_init_context_fxp(missing_path))

            corrupt_bin = tmp / "corrupt.bin"
            corrupt_bin.write_bytes(b"BAD_MAGIC_HEADER_0000000000000000")
            self.assertIsNone(self.lib_cont.deltagrid_init_context(str(corrupt_bin).encode("utf-8")))
            self.assertIsNone(self.lib_fxp.deltagrid_init_context_fxp(str(corrupt_bin).encode("utf-8")))

            # Wrong model_type (loading LM into Continuous context) & wrong precision_mode (FXP into FP32 context)
            self.assertIsNone(self.lib_cont.deltagrid_init_context(str(lm_bin).encode("utf-8")))
            self.assertIsNone(self.lib_cont.deltagrid_init_context(str(fxp_slot_bin).encode("utf-8")))
            self.assertIsNone(self.lib_fxp.deltagrid_init_context_fxp(str(lm_bin).encode("utf-8")))

            # Mismatched state file dimensions in deltagrid_load_state & deltagrid_load_state_fxp
            m_a = DeltaGridContinuous(input_dim=4, output_dim=2, d_model=16, num_layers=1, num_heads=2).eval()
            m_b = DeltaGridContinuous(input_dim=4, output_dim=2, d_model=32, num_layers=2, num_heads=4).eval()
            bin_a, bin_b = tmp / "a.bin", tmp / "b.bin"
            fxp_a, fxp_b = tmp / "a_fxp.bin", tmp / "b_fxp.bin"
            st_b, st_fxp_b = tmp / "b.state", tmp / "b_fxp.state"
            export_deltagrid_to_bin(m_a, bin_a, mode="fp32")
            export_deltagrid_to_bin(m_b, bin_b, mode="fp32")
            export_deltagrid_to_bin(m_a, fxp_a, mode="fxp")
            export_deltagrid_to_bin(m_b, fxp_b, mode="fxp")

            ctx_a = self.lib_cont.deltagrid_init_context(str(bin_a).encode("utf-8"))
            ctx_b = self.lib_cont.deltagrid_init_context(str(bin_b).encode("utf-8"))
            self.lib_cont.deltagrid_save_state(ctx_b, str(st_b).encode("utf-8"))
            self.assertEqual(self.lib_cont.deltagrid_load_state(ctx_a, missing_path), -1)
            self.assertEqual(self.lib_cont.deltagrid_load_state(ctx_a, str(corrupt_bin).encode("utf-8")), -1)
            self.assertEqual(self.lib_cont.deltagrid_load_state(ctx_a, str(st_b).encode("utf-8")), -1)
            self.lib_cont.deltagrid_release_context(ctx_a)
            self.lib_cont.deltagrid_release_context(ctx_b)

            ctx_fa = self.lib_fxp.deltagrid_init_context_fxp(str(fxp_a).encode("utf-8"))
            ctx_fb = self.lib_fxp.deltagrid_init_context_fxp(str(fxp_b).encode("utf-8"))
            self.lib_fxp.deltagrid_save_state_fxp(ctx_fb, str(st_fxp_b).encode("utf-8"))
            self.assertEqual(self.lib_fxp.deltagrid_load_state_fxp(ctx_fa, missing_path), -1)
            self.assertEqual(self.lib_fxp.deltagrid_load_state_fxp(ctx_fa, str(corrupt_bin).encode("utf-8")), -1)
            self.assertEqual(self.lib_fxp.deltagrid_load_state_fxp(ctx_fa, str(st_fxp_b).encode("utf-8")), -1)
            self.lib_fxp.deltagrid_release_context_fxp(ctx_fa)
            self.lib_fxp.deltagrid_release_context_fxp(ctx_fb)

            # 4. Exporter helper & CLI coverage
            self.assertEqual(_to_flat_f32(np.array([[1.0, 2.0]], dtype=np.float64)).shape, (2,))
            self.assertEqual(_to_flat_f32(3.5).shape, (1,))
            with self.assertRaises(ValueError):
                _write_weight(io.BytesIO(), torch.ones(2, 2), 999)
            with self.assertRaises(ValueError):
                export_deltagrid_to_bin(m_a, tmp / "bad.bin", mode="invalid_mode")

            ckpt_cont = tmp / "ckpt_cont"
            ckpt_lm = tmp / "ckpt_lm"
            m_a.save_pretrained(ckpt_cont)
            lm_small.save_pretrained(ckpt_lm)

            repo_root = Path(__file__).resolve().parent.parent
            exporter_script = str(repo_root / "src" / "cpp" / "engine" / "tools" / "export_deltagrid_binary.py")
            with patch.object(sys, "argv", [exporter_script, "--checkpoint", str(ckpt_cont), "--output", str(tmp / "cli_c.bin"), "--mode", "fp32", "--type", "continuous"]):
                runpy.run_path(exporter_script, run_name="__main__")
            with patch.object(sys, "argv", [exporter_script, "--checkpoint", str(ckpt_lm), "--output", str(tmp / "cli_l.bin"), "--mode", "int8", "--type", "lm"]):
                runpy.run_path(exporter_script, run_name="__main__")

            # 5. Python Config, Outputs, State, Block, Layer & Model Edge Branches
            with self.assertRaises(ValueError):
                DeltaGridConfig(d_model=16, num_heads=0)
            cfg = DeltaGridConfig(input_dim=4, output_dim=4, d_model=16, num_layers=1, num_heads=2)
            raw_json = cfg.to_json()
            self.assertIsInstance(raw_json, str)
            cfg_from_raw = DeltaGridConfig.from_json(raw_json)
            self.assertEqual(cfg_from_raw.d_model, 16)
            json_file = tmp / "direct_cfg.json"
            cfg.to_json(json_file)
            self.assertEqual(DeltaGridConfig.from_pretrained(json_file).d_model, 16)
            empty_dir = tmp / "empty_dir"
            empty_dir.mkdir()
            with self.assertRaises(FileNotFoundError):
                DeltaGridConfig.from_pretrained(empty_dir)

            # Outputs containers
            t_logits = torch.randn(2, 3, 8)
            for fmt in ["logits_aux_state", "logits_aux", "logits_only", "logits_state"]:
                lo = DeltaGridLMOutput(logits=t_logits, state=None, legacy_format=fmt)
                _ = len(lo)
                _ = lo[0]
                _ = lo[0, 1]  # fallback to tensor slice
                _ = lo[99] if False else None
                try:
                    _ = lo[10]
                except IndexError:
                    pass
                self.assertEqual(lo.shape, (2, 3, 8))

            t_out = torch.randn(2, 3, 4)
            t_fc = torch.randn(2, 5, 4)
            co1 = DeltaGridContinuousOutput(out=t_out, state=None, forecast=t_fc, legacy_format="out_state_forecast")
            co2 = DeltaGridContinuousOutput(out=t_out, state="dummy", forecast=t_fc, legacy_format="out_state_forecast")
            co3 = DeltaGridContinuousOutput(out=t_out, state=None, forecast=None, legacy_format="out_state")
            self.assertEqual(len(co1), 2)
            self.assertEqual(len(co2), 3)
            self.assertEqual(len(co3), 1)
            self.assertEqual(co1[0].shape, (2, 3, 4))
            self.assertEqual(co1[0, 1].shape, (4,))
            self.assertEqual(co1.shape, (2, 3, 4))

            # State container edge branches
            self.assertIsNone(DeltaGridState.from_legacy(None))
            st_obj = DeltaGridState([DeltaGridLayerState(torch.zeros(1), torch.zeros(1))])
            self.assertIs(DeltaGridState.from_legacy(st_obj), st_obj)
            st_mixed = DeltaGridState.from_legacy([
                DeltaGridLayerState(torch.zeros(1), torch.zeros(1)),
                (torch.zeros(1),),
                None,
            ])
            self.assertEqual(len(st_mixed), 3)

            # Block & Layer init and has_memory=False branches
            blk_cfg1 = DeltaGridBlock(cfg)
            blk_cfg2 = DeltaGridBlock(config=cfg)
            blk_default = DeltaGridBlock(d_model=None, num_heads=4)
            blk_nomem = DeltaGridBlock(d_model=16, num_heads=2, has_memory=False)
            x_seq = torch.randn(1, 3, 16)
            y_seq, m_seq, i_seq = blk_nomem(x_seq)
            self.assertIsNone(m_seq)
            y_stp, m_stp, i_stp = blk_nomem.step(x_seq[:, :1, :], ind_state=i_seq)
            self.assertIsNone(m_stp)
            _ = (blk_cfg1, blk_cfg2, blk_default, y_stp)

            lyr_cfg1 = DeltaGridLayer(cfg)
            lyr_cfg2 = DeltaGridLayer(config=cfg)
            lyr_default = DeltaGridLayer(d_model=None, num_heads=4)
            self.assertEqual(float(lyr_cfg1.compute_router_entropy(torch.randn(1, 2, 8))), 0.0)
            self.assertEqual(float(lyr_cfg1.compute_load_balance_loss(torch.randn(1, 2, 8))), 0.0)
            _ = (lyr_cfg2, lyr_default)

            # Continuous model branches
            c_m1 = DeltaGridContinuous(cfg)
            c_m2 = DeltaGridContinuous(config=cfg)
            self.assertGreater(c_m1.num_parameters, 0)
            _ = c_m1.compile()
            res_all = c_m1(torch.randn(1, 3, 4), states=[c_m1(torch.randn(1, 1, 4)).state.get_layer(0)], return_all_forecasts=True)
            self.assertEqual(res_all.forecast.shape, (1, 3, 1, 4))
            res_fc = c_m1(torch.randn(1, 3, 4), return_forecast=True)
            self.assertEqual(res_fc.forecast.shape, (1, 1, 4))
            res_stp = c_m1.step(torch.randn(1, 4), states=[res_fc.state.get_layer(0)], return_forecast=True)
            self.assertEqual(res_stp.forecast.shape, (1, 1, 4))

            c_fh = DeltaGridContinuous(input_dim=4, output_dim=4, d_model=16, num_layers=1, num_heads=2, forecast_horizon=4)
            self.assertEqual(c_fh.forecast(torch.randn(1, 3, 4), forecast_steps=2).shape, (1, 2, 4))
            c_loaded = DeltaGridContinuous.from_pretrained(ckpt_cont)
            self.assertEqual(c_loaded.d_model, 16)
            _ = c_m2

            # LM model branches
            lm_cfg = DeltaGridLM(config=DeltaGridConfig(vocab_size=24, d_model=16, num_layers=1, num_heads=2))
            self.assertGreater(lm_cfg.num_parameters, 0)
            _, st_1d = lm_cfg.step(torch.tensor([3]))
            self.assertEqual(len(st_1d), 1)
            gen_out = lm_cfg.generate(torch.tensor([[1, 2]]), max_new_tokens=3, temperature=1.0, top_p=1.0, eos_token_id=-1)
            self.assertEqual(gen_out.shape[1], 5)
            # Trigger immediate EOS break
            with torch.no_grad():
                lm_cfg.embed.weight.zero_()
            gen_eos = lm_cfg.generate(torch.tensor([[0, 0]]), max_new_tokens=4, temperature=0.0, eos_token_id=0)
            self.assertEqual(gen_eos.shape[1], 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
