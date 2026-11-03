"""
=============================================================================
Unit Test Suite: DeltaGridConfig Architectural & Serialization Engine
=============================================================================
Tests:
  1. Default parameters, property computation (head_dim, virtual_slots, active_slots).
  2. Strict architectural validation (divisibility, bounds, coordinate space).
  3. Serialization parity (to_dict / from_dict, to_json / from_json).
  4. Hugging Face compatible persistence (save_pretrained / from_pretrained).
  5. Preservation of downstream application-specific extra fields.
=============================================================================
"""

import unittest
import tempfile
import json
from pathlib import Path

from hsru_delta import DeltaGridConfig


class TestDeltaGridConfig(unittest.TestCase):
    def test_default_initialization(self):
        """Verifies default initialization and computed mathematical properties."""
        cfg = DeltaGridConfig(vocab_size=32000, d_model=128, num_heads=4)
        self.assertEqual(cfg.vocab_size, 32000)
        self.assertEqual(cfg.d_model, 128)
        self.assertEqual(cfg.num_heads, 4)
        self.assertEqual(cfg.head_dim, 32)
        self.assertFalse(cfg.is_pkm)
        self.assertEqual(cfg.virtual_slots, 32)
        self.assertEqual(cfg.active_slots, 32)

    def test_pkm_properties(self):
        """Verifies Sparse Delta-PKM mathematical coordinate space calculations."""
        cfg = DeltaGridConfig(
            d_model=256,
            num_heads=4,
            grid_dim=(64, 64),
            top_k=4,
        )
        self.assertTrue(cfg.is_pkm)
        self.assertEqual(cfg.head_dim, 64)
        self.assertEqual(cfg.virtual_slots, 4096)
        self.assertEqual(cfg.active_slots, 16)  # top_k^2 = 4^2 = 16

    def test_validation_errors(self):
        """Verifies that invalid hyperparameters raise descriptive ValueErrors immediately."""
        # Non-divisible d_model / num_heads
        with self.assertRaises(ValueError):
            DeltaGridConfig(d_model=128, num_heads=5)

        # Negative dimension
        with self.assertRaises(ValueError):
            DeltaGridConfig(d_model=-64, num_heads=2)

        # Invalid decay range
        with self.assertRaises(ValueError):
            DeltaGridConfig(d_model=64, num_heads=2, decay_init=1.5)

        # Active slots exceeding coordinate space
        with self.assertRaises(ValueError):
            DeltaGridConfig(d_model=64, num_heads=2, grid_dim=(4, 4), top_k=8)

    def test_dict_serialization(self):
        """Verifies lossless dictionary serialization and deserialization."""
        cfg = DeltaGridConfig(
            vocab_size=1000,
            d_model=64,
            num_layers=3,
            num_heads=2,
            grid_dim=(16, 16),
            top_k=4,
            decay_init=0.98,
            custom_learning_rate=1e-4,  # extra downstream field
        )
        d = cfg.to_dict()
        self.assertEqual(d["vocab_size"], 1000)
        self.assertEqual(d["d_model"], 64)
        self.assertEqual(d["grid_dim"], [16, 16])
        self.assertEqual(d["custom_learning_rate"], 1e-4)

        restored = DeltaGridConfig.from_dict(d)
        self.assertEqual(restored.d_model, cfg.d_model)
        self.assertEqual(restored.grid_dim, cfg.grid_dim)
        self.assertEqual(restored.extra_fields["custom_learning_rate"], 1e-4)

    def test_json_and_pretrained_persistence(self):
        """Verifies Hugging Face style JSON persistence (save_pretrained / from_pretrained)."""
        cfg = DeltaGridConfig(
            vocab_size=5000,
            d_model=128,
            num_layers=2,
            num_heads=4,
            grid_dim=(32, 32),
            top_k=4,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir)
            cfg.save_pretrained(save_path)

            config_file = save_path / "config.json"
            self.assertTrue(config_file.exists())

            loaded = DeltaGridConfig.from_pretrained(save_path)
            self.assertEqual(loaded.vocab_size, 5000)
            self.assertEqual(loaded.d_model, 128)
            self.assertEqual(loaded.grid_dim, (32, 32))
            self.assertEqual(loaded.top_k, 4)


if __name__ == "__main__":
    unittest.main()

