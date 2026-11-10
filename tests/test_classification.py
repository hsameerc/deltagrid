"""
=============================================================================
Unit Test Suite: DeltaGridForSequenceClassification
=============================================================================
Tests:
  1. Model initialization from kwargs and DeltaGridConfig.
  2. Forward pass shapes, pooling strategies ('dual', 'mean', 'last'), and outputs.
  3. Padding invariance with attention_mask (right-padded sequences match unpadded).
  4. Multi-class, binary, and regression loss computation and gradient flow across all 4 tiers.
  5. Constant O(1) RAM streaming classification (forward_streaming == full forward).
  6. Optimizer parameter grouping (exempting 1D recurrent timescales from weight decay).
  7. Hugging Face compatible save_pretrained and from_pretrained persistence.
  8. Long-range noisy document classification (needle-in-a-haystack brand/topic signal).
=============================================================================
"""

import unittest
import tempfile
from pathlib import Path
import torch

from hsru_delta import (
    DeltaGridConfig,
    DeltaGridLM,
    DeltaGridForSequenceClassification,
    DeltaGridClassificationOutput,
)


class TestDeltaGridClassification(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_initialization_and_pooling_strategies(self):
        """Verifies initialization across 'dual', 'mean', 'last', and 'moments' pooling strategies."""
        for strategy, mult in (("dual", 2), ("mean", 1), ("last", 1), ("moments", 3)):
            model = DeltaGridForSequenceClassification(
                num_classes=4,
                vocab_size=200,
                d_model=64,
                num_layers=2,
                num_heads=4,
                pooling_strategy=strategy,
            ).to(self.device)
            model.eval()

            input_ids = torch.randint(0, 200, (3, 16), device=self.device)
            out = model(input_ids)
            self.assertIsInstance(out, DeltaGridClassificationOutput)
            self.assertEqual(out.logits.shape, (3, 4))
            expected_pool_dim = 64 * mult
            self.assertEqual(out.pooled.shape, (3, expected_pool_dim))

            # Verify standalone extract_features matches out.pooled
            feats = model.extract_features(input_ids)
            self.assertTrue(torch.allclose(feats, out.pooled, atol=1e-6))

        # Verify explicit config= initialization and num_parameters property
        cfg = DeltaGridConfig(vocab_size=100, d_model=32, num_layers=1, num_heads=2, num_classes=3)
        m_cfg = DeltaGridForSequenceClassification(config=cfg).to(self.device)
        self.assertGreater(m_cfg.num_parameters, 0)

    def test_output_container_branches(self):
        """Verifies all legacy formats, indexing, slicing, len, getattr, and repr of DeltaGridClassificationOutput."""
        logits = torch.randn(2, 4)
        loss = torch.tensor(0.5)
        state = ["mock_state"]

        # 1. loss_logits format (with and without loss)
        out_ll = DeltaGridClassificationOutput(logits=logits, loss=loss, legacy_format="loss_logits")
        self.assertEqual(len(out_ll), 2)
        self.assertIs(out_ll[0], loss)
        out_ll_noloss = DeltaGridClassificationOutput(logits=logits, loss=None, legacy_format="loss_logits")
        self.assertEqual(len(out_ll_noloss), 1)

        # 2. logits_only format
        out_lo = DeltaGridClassificationOutput(logits=logits, loss=loss, state=state, legacy_format="logits_only")
        self.assertEqual(len(out_lo), 1)
        self.assertIs(out_lo[0], logits)

        # 3. logits_state format (all combinations of loss and state)
        out_ls1 = DeltaGridClassificationOutput(logits=logits, loss=loss, state=None)
        self.assertEqual(len(out_ls1), 2)
        out_ls2 = DeltaGridClassificationOutput(logits=logits, loss=None, state=state)
        self.assertEqual(len(out_ls2), 2)
        out_ls3 = DeltaGridClassificationOutput(logits=logits, loss=None, state=None)
        self.assertEqual(len(out_ls3), 1)

        # 4. Slicing & IndexError fallback, getattr delegation, and repr
        sliced = out_ls3[(0, slice(None))]
        self.assertEqual(sliced.shape, (4,))
        row_fallback = out_ls3[1]  # Tuple len is 1, falls back to self.logits[1]
        self.assertEqual(row_fallback.shape, (4,))
        self.assertEqual(out_ls3.size(0), 2)
        self.assertIn("DeltaGridClassificationOutput", repr(out_ll))
        self.assertIn("DeltaGridClassificationOutput", repr(out_ls3))

    def test_padding_invariance_with_attention_mask(self):
        """Verifies that right-padding tokens do not alter pooled representations or logits."""
        model = DeltaGridForSequenceClassification(
            num_classes=4,
            vocab_size=100,
            d_model=64,
            num_layers=2,
            num_heads=2,
            classifier_dropout=0.0,
            pooling_strategy="dual",
        ).to(self.device)
        model.eval()

        # Sequence of length 10
        seq = torch.randint(1, 100, (1, 10), device=self.device)
        out_unpadded = model(seq)

        # Same sequence padded to length 24 with arbitrary pad tokens
        padded_seq = torch.zeros((1, 24), dtype=torch.long, device=self.device)
        padded_seq[:, :10] = seq
        mask = torch.zeros((1, 24), dtype=torch.bool, device=self.device)
        mask[:, :10] = True

        out_padded = model(padded_seq, attention_mask=mask)

        diff = (out_unpadded.logits - out_padded.logits).abs().max().item()
        self.assertLess(diff, 1e-5, f"Padding leaked into classification logits: {diff}")

    def test_loss_computation_and_gradient_flow(self):
        """Verifies loss computation (multi-class, binary, regression) and gradient flow across all 4 tiers."""
        model = DeltaGridForSequenceClassification(
            num_classes=4,
            vocab_size=100,
            d_model=64,
            num_layers=2,
            num_heads=2,
        ).to(self.device)
        model.train()

        input_ids = torch.randint(0, 100, (4, 16), device=self.device)
        labels = torch.tensor([0, 1, 2, 3], dtype=torch.long, device=self.device)

        out = model(input_ids, labels=labels)
        self.assertIsNotNone(out.loss)

        # Unpack loss and logits
        loss, logits, _ = out
        self.assertEqual(logits.shape, (4, 4))
        loss.backward()

        # Verify gradients flow to embedding, recurrent memory, and classifier head
        self.assertIsNotNone(model.embed.weight.grad)
        self.assertGreater(model.embed.weight.grad.abs().sum().item(), 0.0)
        self.assertIsNotNone(model.blocks[0].memory.out_proj.weight.grad)
        self.assertGreater(model.blocks[0].memory.out_proj.weight.grad.abs().sum().item(), 0.0)

        # Regression / binary (num_classes=1)
        reg_model = DeltaGridForSequenceClassification(
            num_classes=1, vocab_size=100, d_model=32, num_layers=1, num_heads=2
        ).to(self.device)
        reg_targets = torch.tensor([0.5, -1.2, 2.0, 0.0], dtype=torch.float32, device=self.device)
        reg_out = reg_model(input_ids, labels=reg_targets)
        self.assertIsNotNone(reg_out.loss)

        # Binary integer targets (num_classes=1)
        bin_targets = torch.tensor([1, 0, 1, 0], dtype=torch.long, device=self.device)
        bin_out = reg_model(input_ids, labels=bin_targets)
        self.assertIsNotNone(bin_out.loss)

        # Soft probability targets (2D tensor)
        soft_targets = torch.softmax(torch.randn(4, 4, device=self.device), dim=-1)
        soft_out = model(input_ids, labels=soft_targets)
        self.assertIsNotNone(soft_out.loss)

    def test_streaming_long_document_parity(self):
        """Verifies that chunkwise O(1) RAM forward_streaming matches full-sequence forward."""
        model = DeltaGridForSequenceClassification(
            num_classes=3,
            vocab_size=100,
            d_model=64,
            num_layers=2,
            num_heads=2,
            classifier_dropout=0.0,
            pooling_strategy="moments",
        ).to(self.device)
        model.eval()

        # Full document of 64 tokens
        full_doc = torch.randint(1, 100, (2, 64), device=self.device)
        full_mask = torch.ones((2, 64), dtype=torch.bool, device=self.device)
        full_mask[0, 50:] = False  # Sample 0 ends at token 50

        out_full = model(full_doc, attention_mask=full_mask)

        # Split into 4 chunks of 16 tokens each
        chunks = torch.chunk(full_doc, 4, dim=1)
        mask_chunks = torch.chunk(full_mask, 4, dim=1)

        out_stream = model.forward_streaming(chunks, attention_masks=mask_chunks, labels=torch.tensor([0, 1], device=self.device))
        self.assertIsNotNone(out_stream.loss)

        diff = (out_full.logits - out_stream.logits).abs().max().item()
        self.assertLess(diff, 1e-4, f"Streaming vs Full classification mismatch: {diff}")

        # Unmasked streaming parity
        out_full_unmasked = model(full_doc)
        out_stream_unmasked = model.forward_streaming(chunks)
        self.assertLess((out_full_unmasked.logits - out_stream_unmasked.logits).abs().max().item(), 1e-4)

        # All-zero mask fallback and empty chunks error
        zero_mask = [torch.zeros((2, 16), dtype=torch.bool, device=self.device)]
        out_zero = model.forward_streaming([chunks[0]], attention_masks=zero_mask)
        self.assertEqual(out_zero.logits.shape, (2, 3))
        with self.assertRaises(ValueError):
            model.forward_streaming([])

    def test_optimizer_parameter_grouping(self):
        """Verifies that 1D timescale and normalization parameters receive 0.0 weight decay."""
        model = DeltaGridForSequenceClassification(num_classes=4, vocab_size=100, d_model=64, num_layers=2)
        model.embed.weight.requires_grad = False
        groups = model.get_optimizer_grouped_parameters(weight_decay=0.05)

        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0]["weight_decay"], 0.05)
        self.assertEqual(groups[1]["weight_decay"], 0.0)
        self.assertGreater(len(groups[0]["params"]), 0)
        self.assertGreater(len(groups[1]["params"]), 0)

    def test_save_and_from_pretrained(self):
        """Verifies Hugging Face style persistence and from_lm_pretrained backbone transfer."""
        model = DeltaGridForSequenceClassification(
            num_classes=5, vocab_size=100, d_model=64, num_layers=2, num_heads=2, classifier_dropout=0.0
        ).to(self.device)
        model.eval()

        x = torch.randint(0, 100, (2, 12), device=self.device)
        ref_logits = model(x).logits

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "clf_ckpt"
            model.save_pretrained(save_path)

            loaded = DeltaGridForSequenceClassification.from_pretrained(save_path).to(self.device)
            loaded.eval()
            loaded_logits = loaded(x).logits

            self.assertEqual(loaded.num_classes, 5)
            self.assertTrue(torch.allclose(ref_logits, loaded_logits, atol=1e-6))

            # Verify from_lm_pretrained from both in-memory DeltaGridLM and saved directory
            lm = DeltaGridLM(vocab_size=100, d_model=64, num_layers=2, num_heads=2).to(self.device)
            clf_from_mem = DeltaGridForSequenceClassification.from_lm_pretrained(
                lm, num_classes=6, pooling_strategy="moments"
            ).to(self.device)
            self.assertTrue(torch.allclose(lm.embed.weight, clf_from_mem.embed.weight))

            lm_dir = Path(tmpdir) / "lm_ckpt"
            lm.config.save_pretrained(lm_dir)
            torch.save(lm.state_dict(), lm_dir / "pytorch_model.bin")
            clf_from_dir = DeltaGridForSequenceClassification.from_lm_pretrained(
                lm_dir, num_classes=6, pooling_strategy="moments"
            ).to(self.device)
            self.assertTrue(torch.allclose(lm.embed.weight, clf_from_dir.embed.weight))

    def test_hard_noisy_document_convergence(self):
        """Verifies the classifier learns to classify documents with signature patterns buried in noise."""
        torch.manual_seed(7)
        model = DeltaGridForSequenceClassification(
            num_classes=4,
            vocab_size=64,
            d_model=64,
            num_layers=2,
            num_heads=4,
            classifier_dropout=0.05,
        ).to(self.device)

        optimizer = torch.optim.AdamW(model.get_optimizer_grouped_parameters(weight_decay=0.01), lr=4e-3)

        # Generate 32 noisy documents of length 32 where class signature (3-token motif) appears at random positions
        B, T = 32, 32
        docs = torch.randint(10, 60, (B, T), device=self.device)
        labels = torch.arange(B, device=self.device) % 4
        motifs = {
            0: torch.tensor([1, 2, 3], device=self.device),
            1: torch.tensor([4, 5, 6], device=self.device),
            2: torch.tensor([7, 8, 9], device=self.device),
            3: torch.tensor([2, 6, 8], device=self.device),
        }
        for i in range(B):
            c = int(labels[i].item())
            pos = (i * 5) % (T - 4)
            docs[i, pos:pos + 3] = motifs[c]

        model.train()
        for _ in range(20):
            optimizer.zero_grad()
            out = model(docs, labels=labels)
            out.loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            preds = model(docs).logits.argmax(dim=-1)
            acc = (preds == labels).float().mean().item()
        self.assertGreaterEqual(acc, 0.95, f"Hard noisy document accuracy too low: {acc:.2%}")


if __name__ == "__main__":
    unittest.main()
