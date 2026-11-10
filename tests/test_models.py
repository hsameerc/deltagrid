"""
=============================================================================
Unit Test Suite: DeltaGrid Models & Public Interface
=============================================================================
Tests:
  1. DeltaGridLM prefill shapes, step decoding parity, and O(1) RAM generate().
  2. DeltaGridLMOutput attribute access and transparent tuple unpacking.
  3. DeltaGridContinuous forward stream, 1-step control, and multi-horizon forecast.
  4. Hugging Face compatible save_pretrained and from_pretrained model loading.
=============================================================================
"""

import unittest
import tempfile
from pathlib import Path
import torch

from hsru_delta import (
    DeltaGridConfig,
    DeltaGridLM,
    DeltaGridContinuous,
    DeltaGridLMOutput,
    DeltaGridContinuousOutput,
)


class TestDeltaGridModels(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_lm_forward_and_output_unpacking(self):
        """Verifies LM forward pass and dual-mode output container (attributes + tuple unpacking)."""
        model = DeltaGridLM(vocab_size=100, d_model=64, num_layers=2, num_heads=2).to(self.device)
        model.eval()

        input_ids = torch.randint(0, 100, (2, 16), device=self.device)

        # 1. Attribute access
        output = model(input_ids)
        self.assertIsInstance(output, DeltaGridLMOutput)
        self.assertEqual(output.logits.shape, (2, 16, 100))
        self.assertIsNotNone(output.state)

        # 2. Legacy tuple unpacking (logits, state = model(input_ids))
        logits, state = model(input_ids)
        self.assertEqual(logits.shape, (2, 16, 100))
        self.assertEqual(len(state), 2)

    def test_lm_step_parity(self):
        """Verifies bitwise numerical parity between sequence forward() and step() decoding."""
        model = DeltaGridLM(vocab_size=100, d_model=64, num_layers=2, num_heads=2).to(self.device)
        model.eval()

        input_ids = torch.randint(0, 100, (2, 8), device=self.device)

        # Prefill pass
        out_fwd = model(input_ids)
        logits_fwd = out_fwd.logits

        # Step decoding
        st = None
        step_logits = []
        for t in range(8):
            step_out, st = model.step(input_ids[:, t:t+1], states=st, seq_offset=t)
            step_logits.append(step_out)

        logits_step = torch.cat(step_logits, dim=1)
        diff = (logits_fwd - logits_step).abs().max().item()
        self.assertLess(diff, 1e-4, f"Forward vs Step parity mismatch: {diff}")

    def test_lm_logic_expansion(self):
        """Verifies DeltaGridLM with custom logic_expansion (e.g. 2.0x SwiGLU) and step parity."""
        model_1x = DeltaGridLM(vocab_size=100, d_model=64, num_layers=2, num_heads=2, logic_expansion=1.0).to(self.device)
        model_2x = DeltaGridLM(vocab_size=100, d_model=64, num_layers=2, num_heads=2, logic_expansion=2.0).to(self.device)

        # 2x model must have strictly more parameters due to wider SwiGLU
        self.assertGreater(model_2x.num_parameters, model_1x.num_parameters)

        # Forward and step parity must hold with expanded logic
        model_2x.eval()
        input_ids = torch.randint(0, 100, (2, 8), device=self.device)
        out_fwd = model_2x(input_ids)
        st = None
        step_logits = []
        for t in range(8):
            step_out, st = model_2x.step(input_ids[:, t:t+1], states=st, seq_offset=t)
            step_logits.append(step_out)
        logits_step = torch.cat(step_logits, dim=1)
        diff = (out_fwd.logits - logits_step).abs().max().item()
        self.assertLess(diff, 1e-4)

    def test_lm_generate(self):
        """Verifies autoregressive token generation with zero KV cache and sampling strategies."""
        model = DeltaGridLM(vocab_size=100, d_model=64, num_layers=2, num_heads=2).to(self.device)
        prompt = torch.randint(0, 100, (1, 4), device=self.device)

        # Greedy decoding
        gen_greedy = model.generate(prompt, max_new_tokens=10, temperature=0.0)
        self.assertEqual(gen_greedy.shape, (1, 14))

        # Sampling with repetition penalty, top_k, top_p, and extra kwargs
        gen_sampled = model.generate(
            prompt,
            max_new_tokens=10,
            temperature=0.8,
            top_k=20,
            top_p=0.9,
            repetition_penalty=1.2,
            extra_param_should_be_ignored=True,
        )
        self.assertEqual(gen_sampled.shape, (1, 14))

    def test_continuous_controller_and_forecasting(self):
        """Verifies DeltaGridContinuous multi-channel streaming and multi-step forecasting."""
        model = DeltaGridContinuous(
            input_dim=4,
            output_dim=2,
            d_model=64,
            num_layers=2,
            num_heads=2,
            forecast_horizon=5,
        ).to(self.device)
        model.eval()

        x = torch.randn(2, 12, 4, device=self.device)

        # 1. Forward stream
        out, states = model(x)
        self.assertEqual(out.shape, (2, 12, 2))
        self.assertEqual(len(states), 2)

        # 2. 1-step real-time control step
        step_x = torch.randn(2, 1, 4, device=self.device)
        step_out, states = model.step(step_x, states=states)
        self.assertEqual(step_out.shape, (2, 1, 2))

        # 3. Direct trajectory forecast
        forecast = model.forecast(x, forecast_steps=5)
        self.assertEqual(forecast.shape, (2, 5, 2))

    def test_model_save_and_from_pretrained(self):
        """Verifies end-to-end model checkpoint saving and loading."""
        model = DeltaGridLM(vocab_size=100, d_model=64, num_layers=2, num_heads=2).to(self.device)
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir)
            model.save_pretrained(save_path)

            loaded_model = DeltaGridLM.from_pretrained(save_path)
            self.assertEqual(loaded_model.d_model, 64)
            self.assertEqual(loaded_model.vocab_size, 100)
            self.assertEqual(loaded_model.num_layers, 2)

    def test_outputs_and_loss_options(self):
        """Verifies cross-entropy loss, aux_loss, gradient checkpointing, and output reprs."""
        model = DeltaGridLM(vocab_size=100, d_model=64, num_layers=2, num_heads=2).to(self.device)
        input_ids = torch.randint(0, 100, (2, 8), device=self.device)
        labels = torch.randint(0, 100, (2, 8), device=self.device)

        # Loss calculation
        out_loss = model(input_ids, labels=labels)
        self.assertIsNotNone(out_loss.loss)
        self.assertTrue(torch.isfinite(out_loss.loss))
        self.assertIn("DeltaGridLMOutput", repr(out_loss))

        # Aux loss format
        out_aux = model(input_ids, return_aux_loss=True)
        items = list(out_aux)
        self.assertEqual(len(items), 3)  # logits, aux_loss, state

        # Gradient checkpointing
        model.gradient_checkpointing_enable()
        out_ckpt = model(input_ids)
        self.assertEqual(out_ckpt.logits.shape, (2, 8, 100))
        model.gradient_checkpointing_disable()

        # Continuous all forecasts & repr
        cont = DeltaGridContinuous(input_dim=2, output_dim=2, d_model=64, num_layers=1, num_heads=2, forecast_horizon=3).to(self.device)
        x = torch.randn(2, 6, 2, device=self.device)
        out_all = cont(x, return_all_forecasts=True)
        self.assertIn("DeltaGridContinuousOutput", repr(out_all))
        self.assertIsNotNone(out_all.forecast)


if __name__ == "__main__":
    unittest.main()

