"""
=============================================================================
Unit Test Suite: Mathematical Integrity, Causality & Invariant Auditing
=============================================================================
Audits:
  1. LM & Continuous Causality (Future tokens do not leak into past states).
  2. Chunkwise WY Prefill == Token-by-Token Recurrence.
  3. Chunk-size Invariance across different chunk lengths C.
  4. Parallel Forward Prefill Logits == Streaming step() Decoding.
  5. Stateful Split Context Carryover: forward(A + B) == forward(A) then forward(B, state).
  6. Gradient Health & Long-Range Backward Reach across sequence depth.
  7. Exact Determinism given fixed random seed.
=============================================================================
"""

import unittest
import torch
import torch.nn as nn

from hsru_delta import DeltaGridLM, DeltaGridContinuous, DeltaGridLayer


class TestIntegrityAudit(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float32)

    def _make_lm(self, **kw):
        torch.manual_seed(0)
        cfg = dict(vocab_size=50, d_model=64, num_layers=2, num_heads=2)
        cfg.update(kw)
        return DeltaGridLM(**cfg).eval()

    @torch.no_grad()
    def test_lm_causality(self):
        """Verifies that future token modifications do not leak into past output predictions."""
        m = self._make_lm()
        x = torch.randint(0, 50, (2, 48))
        base, _ = m(x)
        worst = 0.0
        for cut in (5, 20, 40):
            x2 = x.clone()
            x2[:, cut + 1:] = torch.randint(0, 50, x2[:, cut + 1:].shape)
            out, _ = m(x2)
            worst = max(worst, (out[:, :cut + 1] - base[:, :cut + 1]).abs().max().item())
        self.assertLess(worst, 1e-5, f"LM causality leakage detected: {worst}")

    @torch.no_grad()
    def test_continuous_causality(self):
        """Verifies continuous stream causality without future signal leakage."""
        torch.manual_seed(0)
        c = DeltaGridContinuous(input_dim=3, output_dim=3, d_model=64, num_layers=2, num_heads=2).eval()
        x = torch.randn(2, 48, 3)
        base, _ = c(x)
        x2 = x.clone()
        x2[:, 25:] = torch.randn_like(x2[:, 25:])
        out, _ = c(x2)
        worst = (out[:, :25] - base[:, :25]).abs().max().item()
        self.assertLess(worst, 1e-5, f"Continuous causality leakage detected: {worst}")

    @torch.no_grad()
    def test_chunk_vs_recurrent_and_chunk_invariance(self):
        """Verifies WY chunked prefill matches explicit recurrence and chunk-size invariance."""
        torch.manual_seed(1)
        for dyn in (False, True):
            layer = DeltaGridLayer(d_model=64, num_heads=2, dynamic_decay=dyn).eval()
            x = torch.randn(2, 100, 64)
            y_chunk, s_chunk = layer(x, use_chunkwise=True)
            state, ys = None, []
            for t in range(x.size(1)):
                y, state = layer.step(x[:, t:t + 1], state)
                ys.append(y)
            y_step = torch.cat(ys, dim=1)
            d_y = (y_chunk - y_step).abs().max().item()
            d_s = (s_chunk - state).abs().max().item()
            self.assertLess(d_y, 1e-3, f"WY chunkwise diff high (dyn={dyn}): {d_y}")
            self.assertLess(d_s, 1e-3, f"WY state diff high (dyn={dyn}): {d_s}")

            # Chunk size invariance
            outs = []
            for cs in (8, 33, 128):
                layer.chunk_size = cs
                outs.append(layer(x, use_chunkwise=True)[0])
            inv = max((outs[0] - o).abs().max().item() for o in outs[1:])
            self.assertLess(inv, 1e-3, f"Chunk size invariance failed (dyn={dyn}): {inv}")

    @torch.no_grad()
    def test_prefill_vs_step_lm_and_continuous(self):
        """Verifies parallel prefill matches sequential step() unrolling."""
        # LM
        m = self._make_lm()
        x = torch.randint(0, 50, (2, 40))
        logits_par, _ = m(x)
        states, outs = None, []
        for t in range(x.size(1)):
            lg, states = m.step(x[:, t:t + 1], states, seq_offset=t)
            outs.append(lg)
        logits_seq = torch.cat(outs, dim=1)
        diff_lm = (logits_par - logits_seq).abs().max().item()
        agree = (logits_par.argmax(-1) == logits_seq.argmax(-1)).float().mean().item()
        self.assertLess(diff_lm, 1e-3, f"LM prefill vs step diff: {diff_lm}")
        self.assertGreaterEqual(agree, 0.99, f"LM prefill vs step argmax agreement too low: {agree}")

        # Continuous
        torch.manual_seed(0)
        c = DeltaGridContinuous(input_dim=2, output_dim=2, d_model=64, num_layers=2, num_heads=2).eval()
        xc = torch.randn(2, 40, 2)
        par, _ = c(xc)
        st, outs = None, []
        for t in range(40):
            o, st = c.step(xc[:, t:t + 1], st, seq_offset=t)
            outs.append(o)
        diff_cont = (par - torch.cat(outs, 1)).abs().max().item()
        self.assertLess(diff_cont, 1e-3, f"Continuous prefill vs step diff: {diff_cont}")

    @torch.no_grad()
    def test_state_carryover(self):
        """Verifies forward(A + B) == forward(A) then forward(B, state)."""
        m = self._make_lm()
        x = torch.randint(0, 50, (2, 64))
        full, _ = m(x)
        a, st = m(x[:, :37])
        b, _ = m(x[:, 37:], states=st, seq_offset=37)
        diff = (full[:, 37:] - b).abs().max().item()
        self.assertLess(diff, 1e-3, f"State carryover divergence: {diff}")

    def test_gradient_health_and_long_range_flow(self):
        """Verifies finite gradients across all weights and non-zero flow back to token 0."""
        m = self._make_lm().train()
        x = torch.randint(0, 50, (4, 64))
        logits, _ = m(x)
        logits[:, -1].sum().backward()
        for name, p in m.named_parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad, f"Missing gradient for {name}")
                self.assertTrue(torch.isfinite(p.grad).all(), f"Non-finite gradient in {name}")

        # Long range flow
        m.zero_grad()
        emb = m.embed(x).detach().requires_grad_(True)
        h, _ = m._forward_blocks(emb)
        m.lm_head(m.norm(h))[:, -1].sum().backward()
        g = emb.grad.abs().sum(-1)
        g0 = g[:, 0].mean().item()
        self.assertGreater(g0, 0.0, "Gradient died before reaching token 0")
        self.assertTrue(torch.isfinite(g).all(), "Non-finite gradient across sequence")

    @torch.no_grad()
    def test_determinism(self):
        """Verifies deterministic output reproduction given fixed seed."""
        a = self._make_lm()
        b = self._make_lm()
        x = torch.randint(0, 50, (2, 32), generator=torch.Generator().manual_seed(3))
        diff = (a(x).logits - b(x).logits).abs().max().item()
        self.assertEqual(diff, 0.0, f"Non-deterministic execution: diff={diff}")


if __name__ == "__main__":
    unittest.main()

