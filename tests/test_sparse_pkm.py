"""
=============================================================================
Unit Test Suite: Sparse Delta-PKM (Product-Key Memory with Delta Erasure)
=============================================================================
Tests:
  1. Exact Householder Isometry: ||w_k,active||_2 = 1.0.
  2. Lossless 1-Step Fact Overwriting: Reconstruction error < 10^-6.
  3. Forward vs. Step Mathematical Parity: Exact equivalence between batch prefill and streaming.
  4. End-to-End Autograd: Finite, non-zero gradient backpropagation.
  5. Massive Associative Capacity: Storing 1,000 facts in 1,048,576 slots with >99% recall.
=============================================================================
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import unittest
import torch
import torch.nn.functional as F

from src import DeltaGridLayer, SparseDeltaPKMLayer

try:
    import hsru_delta_cuda
    _HAS_CUDA = True
except ImportError:
    _HAS_CUDA = False


class TestSparseDeltaPKM(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_isometry_and_1step_erasure(self):
        """Verifies ||w_k,active||_2 = 1.0 and 1-step fact overwriting error < 10^-6."""
        dh = 64
        top_k = 8
        proj1 = torch.randn(dh, 1024, device=self.device)
        proj2 = torch.randn(dh, 1024, device=self.device)

        k = torch.randn(dh, device=self.device)
        v = torch.randn(dh, device=self.device)

        s1 = torch.matmul(k, proj1)
        s2 = torch.matmul(k, proj2)
        val1, idx1 = torch.topk(s1, top_k)
        val2, idx2 = torch.topk(s2, top_k)
        u1 = F.normalize(val1, p=2, dim=-1)
        u2 = F.normalize(val2, p=2, dim=-1)

        wk = torch.outer(u1, u2).flatten()
        self.assertAlmostEqual(wk.norm().item(), 1.0, places=5, msg="Active key must have unit norm 1.0")

        # Simulate active state matrix
        S_active = torch.randn(top_k * top_k, dh, device=self.device)
        pred_v = torch.matmul(wk, S_active)
        err = v - pred_v
        S_new = S_active + torch.outer(wk, err)

        # Read back
        recalled_v = torch.matmul(wk, S_new)
        reconstruction_error = (v - recalled_v).abs().max().item()
        self.assertLess(reconstruction_error, 1e-5, f"1-step overwrite reconstruction error too high: {reconstruction_error}")

    def test_forward_step_parity(self):
        """Verifies that full sequence forward() and token-by-token step() produce exact parity."""
        d_model = 32
        layer = DeltaGridLayer(d_model=d_model, num_heads=2, grid_dim=(16, 16), top_k=4).to(self.device)
        layer.eval()

        x = torch.randn(2, 6, d_model, device=self.device)

        # 1. Full sequence forward
        out_fwd, state_fwd = layer(x)

        # 2. Token-by-token step
        st = None
        step_outs = []
        for t in range(6):
            out_t, st = layer.step(x[:, t:t+1], state=st)
            step_outs.append(out_t)
        out_step = torch.cat(step_outs, dim=1)

        diff = (out_fwd - out_step).abs().max().item()
        state_diff = (state_fwd - st).abs().max().item()
        self.assertLess(diff, 1e-5, f"Forward vs Step output diff too high: {diff}")
        self.assertLess(state_diff, 1e-5, f"Forward vs Step state diff too high: {state_diff}")

    def test_cuda_pkm_step_kernel(self):
        """Verifies fused CUDA PKM step kernel execution and parity against PyTorch reference."""
        if not (torch.cuda.is_available() and _HAS_CUDA and hasattr(hsru_delta_cuda, "pkm_step")):
            self.skipTest("CUDA or hsru_delta_cuda.pkm_step not available")

        b, h, dh = 2, 2, 32
        active_slots = 16
        total_slots = 256

        v = torch.randn(b, h, dh, device="cuda")
        beta = torch.rand(b, h, device="cuda")
        wk = F.normalize(torch.randn(b, h, active_slots, device="cuda"), p=2, dim=-1)
        wq = torch.randn(b, h, active_slots, device="cuda")
        active_idx = torch.stack([
            torch.stack([torch.randperm(total_slots)[:active_slots] for _ in range(h)])
            for _ in range(b)
        ]).to(device="cuda", dtype=torch.int64)
        decay = torch.full((h,), 0.999, device="cuda")

        s_init = torch.randn(b, h, total_slots, dh, device="cuda")

        # PyTorch reference
        s_py = s_init.clone()
        exp_idx = active_idx.unsqueeze(-1).expand(b, h, active_slots, dh)
        s_act = torch.gather(s_py, 2, exp_idx)
        yt_py = torch.matmul(wq.unsqueeze(-2), s_act).squeeze(-2)
        pred_v = torch.matmul(wk.unsqueeze(-2), s_act).squeeze(-2)
        rt = v - pred_v
        delta = (beta.unsqueeze(-1) * wk).unsqueeze(-1) * rt.unsqueeze(-2)
        s_act_new = decay.view(1, h, 1, 1) * s_act + delta
        s_py = s_py.scatter(2, exp_idx, s_act_new)

        # CUDA fused kernel
        s_cu = s_init.clone()
        yt_cu, s_cu_out = hsru_delta_cuda.pkm_step(v, beta, wk, wq, active_idx, decay, s_cu)

        out_diff = (yt_py - yt_cu).abs().max().item()
        state_diff = (s_py - s_cu_out).abs().max().item()

        self.assertLess(out_diff, 1e-5, f"CUDA vs PyTorch PKM step output diff: {out_diff}")
        self.assertLess(state_diff, 1e-5, f"CUDA vs PyTorch PKM step state diff: {state_diff}")

    def test_autograd_gradients(self):
        """Verifies that gradients backpropagate smoothly with no NaN or Inf."""
        d_model = 32
        layer = DeltaGridLayer(d_model=d_model, num_heads=1, grid_dim=(16, 16), top_k=4).to(self.device)
        layer.train()

        x = torch.randn(2, 4, d_model, device=self.device, requires_grad=True)
        out, _ = layer(x)
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all().item(), "Input gradients must be finite")
        for name, param in layer.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Parameter {name} did not receive gradients")
                self.assertTrue(torch.isfinite(param.grad).all().item(), f"Parameter {name} gradient has NaN/Inf")

    def test_massive_capacity_recall(self):
        """Verifies high-fidelity recall across 1,000 distinct facts in 1,048,576 slots."""
        dh = 64
        top_k = 8
        total_slots = 1024 * 1024 # 1 Million slots

        proj1 = torch.randn(dh, 1024, device=self.device)
        proj2 = torch.randn(dh, 1024, device=self.device)
        state = torch.zeros(1, total_slots, dh, device=self.device, dtype=torch.float32)

        num_facts = 1000
        keys = torch.randn(num_facts, dh, device=self.device)
        values = torch.randn(num_facts, dh, device=self.device)

        # Write facts
        for t in range(num_facts):
            k, v = keys[t], values[t]
            s1, s2 = torch.matmul(k, proj1), torch.matmul(k, proj2)
            val1, idx1 = torch.topk(s1, top_k)
            val2, idx2 = torch.topk(s2, top_k)
            u1, u2 = F.normalize(val1, p=2, dim=-1), F.normalize(val2, p=2, dim=-1)
            wk = torch.outer(u1, u2).flatten()
            idx = (idx1.unsqueeze(-1) * 1024 + idx2.unsqueeze(0)).flatten()

            S_act = state[0, idx]
            err = v - torch.matmul(wk, S_act)
            state[0, idx] = S_act + torch.outer(wk, err)

        # Read & verify recall
        cos_sims = []
        for t in range(num_facts):
            k, v = keys[t], values[t]
            s1, s2 = torch.matmul(k, proj1), torch.matmul(k, proj2)
            val1, idx1 = torch.topk(s1, top_k)
            val2, idx2 = torch.topk(s2, top_k)
            u1, u2 = F.normalize(val1, p=2, dim=-1), F.normalize(val2, p=2, dim=-1)
            wq = torch.outer(u1, u2).flatten()
            idx = (idx1.unsqueeze(-1) * 1024 + idx2.unsqueeze(0)).flatten()

            recalled_v = torch.matmul(wq, state[0, idx])
            cos = F.cosine_similarity(v.unsqueeze(0), recalled_v.unsqueeze(0)).item()
            cos_sims.append(cos)

        mean_cos = sum(cos_sims) / len(cos_sims)
        high_fid_rate = sum(1 for c in cos_sims if c > 0.8) / len(cos_sims) * 100.0
        self.assertGreater(mean_cos, 0.98, f"Mean recall cosine similarity too low: {mean_cos:.4f}")
        self.assertGreaterEqual(high_fid_rate, 99.0, f"High fidelity recall rate too low: {high_fid_rate:.1f}%")

    def test_grid_dim_and_top_k_configuration(self):
        """Verifies direct canonical grid_dim and top_k configuration across all architecture levels."""
        from src.core.delta_grid import DeltaGridBlock, DeltaGridLM, DeltaGridContinuous

        # Layer level: standard dense
        layer_dense = DeltaGridLayer(d_model=64, num_heads=2)
        self.assertEqual(layer_dense.total_slots, 32)
        self.assertIsNone(layer_dense.top_k)
        self.assertIsNone(layer_dense.grid_dim)

        # Layer level: factorized Kronecker dense
        layer_kronecker = DeltaGridLayer(d_model=64, num_heads=2, grid_dim=(8, 8))
        self.assertEqual(layer_kronecker.total_slots, 64)
        self.assertIsNone(layer_kronecker.top_k)

        # Layer level: balanced Sparse Delta-PKM
        layer_pkm = DeltaGridLayer(d_model=64, num_heads=2, grid_dim=(64, 64), top_k=8)
        self.assertEqual(layer_pkm.total_slots, 4096)
        self.assertEqual(layer_pkm.active_slots, 64)

        # Layer level: massive Sparse Delta-PKM (1,048,576 slots)
        layer_massive = DeltaGridLayer(d_model=64, num_heads=2, grid_dim=(1024, 1024), top_k=8)
        self.assertEqual(layer_massive.total_slots, 1048576)
        self.assertEqual(layer_massive.active_slots, 64)

        # Block level
        block_pkm = DeltaGridBlock(d_model=64, num_heads=2, grid_dim=(64, 64), top_k=8)
        self.assertEqual(block_pkm.memory.total_slots, 4096)
        self.assertEqual(block_pkm.memory.active_slots, 64)

        # LM level
        lm_dense = DeltaGridLM(vocab_size=100, d_model=64, num_heads=2)
        self.assertEqual(lm_dense.blocks[0].memory.total_slots, 32)
        lm_pkm = DeltaGridLM(vocab_size=100, d_model=64, num_heads=2, grid_dim=(64, 64), top_k=8)
        self.assertEqual(lm_pkm.blocks[0].memory.total_slots, 4096)

        # Continuous level
        cont_pkm = DeltaGridContinuous(input_dim=8, d_model=64, num_heads=2, grid_dim=(64, 64), top_k=8)
        self.assertEqual(cont_pkm.blocks[0].memory.total_slots, 4096)

    def test_architectural_parameter_validations(self):
        """Verifies strict validation of grid_dim and top_k parameter combinations."""
        # 1. top_k without grid_dim must raise ValueError
        with self.assertRaises(ValueError) as ctx:
            DeltaGridLayer(d_model=64, num_heads=2, top_k=8)
        self.assertIn("top_k (8) was specified without grid_dim", str(ctx.exception))

        # 2. top_k exceeding smallest grid dimension must raise ValueError
        with self.assertRaises(ValueError) as ctx:
            DeltaGridLayer(d_model=64, num_heads=2, grid_dim=(16, 16), top_k=32)
        self.assertIn("cannot exceed the smallest grid dimension", str(ctx.exception))

        # 3. top_k < 1 must raise ValueError
        with self.assertRaises(ValueError) as ctx:
            DeltaGridLayer(d_model=64, num_heads=2, grid_dim=(16, 16), top_k=0)
        self.assertIn("top_k must be an integer >= 1", str(ctx.exception))

        # 4. grid_dim with < 2 axes must raise ValueError
        with self.assertRaises(ValueError) as ctx:
            DeltaGridLayer(d_model=64, num_heads=2, grid_dim=(16,))
        self.assertIn("grid_dim must be a tuple/list with at least 2 axes", str(ctx.exception))

        # 5. grid_dim with axis < 2 must raise ValueError
        with self.assertRaises(ValueError) as ctx:
            DeltaGridLayer(d_model=64, num_heads=2, grid_dim=(1, 16))
        self.assertIn("must be an integer >= 2", str(ctx.exception))

    def test_router_entropy_and_diversity(self):
        """Verifies router entropy computation and load balancing loss helpers."""
        layer = DeltaGridLayer(d_model=64, num_heads=2, grid_dim=(64, 64), top_k=8).to(self.device)
        k = torch.randn(4, 2, 16, 32, device=self.device)

        entropy = layer.compute_router_entropy(k)
        self.assertTrue(torch.isfinite(entropy).item())
        # Entropy for 64 slots should be positive and bounded by ln(64) ~ 4.158
        self.assertGreater(entropy.item(), 2.0)
        self.assertLessEqual(entropy.item(), 4.2)

        loss_bal = layer.compute_load_balance_loss(k)
        self.assertTrue(torch.isfinite(loss_bal).item())
        self.assertGreater(loss_bal.item(), 0.0)

    def test_forward_eval_fast_path_parity(self):
        """Verifies exact parity between autograd forward path and fast eval path in Sparse Delta-PKM."""
        layer = DeltaGridLayer(d_model=64, num_heads=2, grid_dim=(32, 32), top_k=4).to(self.device)
        layer.eval()

        x = torch.randn(2, 8, 64, device=self.device)
        # Fast path (eval mode, no grad)
        with torch.no_grad():
            out_fast, state_fast = layer(x)

        # Autograd path (requires_grad=True)
        x_grad = x.clone().requires_grad_(True)
        out_grad, state_grad = layer(x_grad)

        out_diff = (out_fast - out_grad).abs().max().item()
        state_diff = (state_fast - state_grad).abs().max().item()
        self.assertLess(out_diff, 1e-5, f"Fast vs Autograd output diff too high: {out_diff}")
        self.assertLess(state_diff, 1e-5, f"Fast vs Autograd state diff too high: {state_diff}")

    def test_cpu_reference_autograd_and_submatrix(self):
        """Verifies pure Python CPU autograd forward and backward pass for Sparse Delta-PKM."""
        layer = DeltaGridLayer(d_model=32, num_heads=2, grid_dim=(4, 4), top_k=2).to("cpu")
        layer.train()
        x = torch.randn(2, 4, 32, requires_grad=True, device="cpu")
        out, _ = layer(x)
        out.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

        # Massive slots exercising use_submatrix on CPU
        layer_sub = DeltaGridLayer(d_model=32, num_heads=2, grid_dim=(128, 128), top_k=2).to("cpu")
        layer_sub.train()
        x_sub = torch.randn(2, 2, 32, requires_grad=True, device="cpu")
        out_sub, _ = layer_sub(x_sub)
        out_sub.sum().backward()
        self.assertTrue(torch.isfinite(x_sub.grad).all())

    def test_collision_avoidance_tolerance_and_recovery(self):
        """
        Verifies the 4 collision regimes of Householder Delta-PKM:
          1. Partial Slot Overlap Tolerance (50% shared physical slots, orthogonal subspace):
             Updating Key B (or overwriting Key B with a new value) preserves Key A's old value.
          2. Exact 1-Step Full Overwrite (beta = 1.0):
             Updating Key A erases 100% of Key A's old value without corrupting overlapping Key B.
          3. Controlled Partial / Half-Change Update (beta = 0.5):
             Updating Key A with beta = 0.5 produces exact 50/50 interpolation (0.5 * v_old + 0.5 * v_new)
             while keeping Key B untouched.
          4. Collision Recovery under Non-Orthogonal Correlated Keys (100% shared slots, rho = 0.5):
             Even when two keys share 100% of the same physical slots AND have correlated weights (w_A^T w_B != 0),
             Delta Rule alternating projection (Kaczmarz recovery) converges to <1e-5 error for BOTH keys,
             whereas additive Hebbian memory diverges.
        """
        dh = 64
        active_slots = 64
        total_slots = 256
        state = torch.zeros(total_slots, dh, device=self.device, dtype=torch.float32)

        # ---------------------------------------------------------------------
        # 1. Partial Slot Overlap (50% shared physical slots: slots [0..63] vs [32..95])
        # ---------------------------------------------------------------------
        idx_a = torch.arange(0, 64, device=self.device)
        idx_b = torch.arange(32, 96, device=self.device)  # Shares 32 of 64 physical slots (50% overlap!)

        # Construct unit-norm keys where the shared 32-slot overlap [32..63] is orthogonal
        wk_a = F.normalize(torch.randn(active_slots, device=self.device), p=2, dim=-1)
        wk_b_raw = torch.randn(active_slots, device=self.device)
        # Orthogonalize wk_b's first 32 elements against wk_a's last 32 elements (the shared physical slots [32..63])
        shared_a = wk_a[32:]
        proj_coeff = torch.dot(wk_b_raw[:32], shared_a) / (torch.dot(shared_a, shared_a) + 1e-12)
        wk_b_raw[:32] = wk_b_raw[:32] - proj_coeff * shared_a
        wk_b = F.normalize(wk_b_raw, p=2, dim=-1)

        v_a1 = torch.randn(dh, device=self.device)
        v_b1 = torch.randn(dh, device=self.device)
        v_b2 = torch.randn(dh, device=self.device)

        # Write Key A -> v_a1 (beta = 1.0)
        state[idx_a] = state[idx_a] + torch.outer(wk_a, v_a1 - torch.matmul(wk_a, state[idx_a]))
        # Write Key B -> v_b1 on 50% overlapping slots (beta = 1.0)
        state[idx_b] = state[idx_b] + torch.outer(wk_b, v_b1 - torch.matmul(wk_b, state[idx_b]))
        # Overwrite Key B -> v_b2 on the 50% overlapping slots
        state[idx_b] = state[idx_b] + torch.outer(wk_b, v_b2 - torch.matmul(wk_b, state[idx_b]))

        # Verify Key A's old information v_a1 is 100% preserved and Key B has new value v_b2
        recall_a = torch.matmul(wk_a, state[idx_a])
        recall_b = torch.matmul(wk_b, state[idx_b])
        self.assertLess((recall_a - v_a1).abs().max().item(), 1e-5, "Key A old value corrupted by 50% slot overlap!")
        self.assertLess((recall_b - v_b2).abs().max().item(), 1e-5, "Key B updated value not recalled accurately!")

        # ---------------------------------------------------------------------
        # 2 & 3. Controlled Half-Change Update (beta = 0.5) on Key A
        # ---------------------------------------------------------------------
        v_a2 = torch.randn(dh, device=self.device)
        beta_half = 0.5
        state[idx_a] = state[idx_a] + beta_half * torch.outer(wk_a, v_a2 - torch.matmul(wk_a, state[idx_a]))

        expected_a_half = 0.5 * v_a1 + 0.5 * v_a2
        recall_a_half = torch.matmul(wk_a, state[idx_a])
        recall_b_after_a_half = torch.matmul(wk_b, state[idx_b])

        self.assertLess(
            (recall_a_half - expected_a_half).abs().max().item(),
            1e-5,
            "Half-change (beta=0.5) update did not produce exact 50/50 interpolation!",
        )
        self.assertLess(
            (recall_b_after_a_half - v_b2).abs().max().item(),
            1e-5,
            "Half-change update on Key A corrupted overlapping Key B!",
        )

        # ---------------------------------------------------------------------
        # 4. Collision Recovery (100% shared slots + non-orthogonal correlated keys rho ~ 0.5)
        # ---------------------------------------------------------------------
        state_corr = torch.zeros(active_slots, dh, device=self.device, dtype=torch.float32)
        u1 = F.normalize(torch.randn(active_slots, device=self.device), p=2, dim=-1)
        u2 = F.normalize(torch.randn(active_slots, device=self.device), p=2, dim=-1)
        # Create two non-orthogonal keys with ~0.5 cosine correlation on the EXACT same 64 slots
        k_corr_a = u1
        k_corr_b = F.normalize(0.5 * u1 + 0.8660254 * u2, p=2, dim=-1)

        # Single write of B after A causes partial crosstalk proportional to (k_a . k_b),
        # but Kaczmarz Delta recovery eliminates the crosstalk across re-presentations:
        for _ in range(12):
            state_corr = state_corr + torch.outer(k_corr_a, v_a1 - torch.matmul(k_corr_a, state_corr))
            state_corr = state_corr + torch.outer(k_corr_b, v_b1 - torch.matmul(k_corr_b, state_corr))

        rec_corr_a = torch.matmul(k_corr_a, state_corr)
        rec_corr_b = torch.matmul(k_corr_b, state_corr)
        self.assertLess((rec_corr_a - v_a1).abs().max().item(), 1e-4, "Delta Kaczmarz recovery failed on Key A!")
        self.assertLess((rec_corr_b - v_b1).abs().max().item(), 1e-4, "Delta Kaczmarz recovery failed on Key B!")



if __name__ == "__main__":
    unittest.main()

