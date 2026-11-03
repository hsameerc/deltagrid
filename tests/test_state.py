"""
=============================================================================
Unit Test Suite: DeltaGridState Recurrent State Encapsulation Container
=============================================================================
Tests:
  1. DeltaGridLayerState tensor isolation and tuple unpacking.
  2. Severing autograd graph history across TBPTT boundaries via .detach().
  3. DeltaGridState multi-layer container indexing, cloning, and reset.
  4. 100% backward compatibility with legacy List[Tuple[Tensor, Tensor]] format.
=============================================================================
"""

import unittest
import torch

from hsru_delta import DeltaGridState, DeltaGridLayerState


class TestDeltaGridState(unittest.TestCase):
    def test_layer_state_unpacking_and_indexing(self):
        """Verifies that DeltaGridLayerState supports transparent tuple unpacking and indexing."""
        mem = torch.randn(2, 4, 16, 32)
        ind = torch.randn(2, 128, 3)
        layer_st = DeltaGridLayerState(memory=mem, indicator=ind)

        # Tuple indexing
        self.assertIs(layer_st[0], mem)
        self.assertIs(layer_st[1], ind)
        self.assertEqual(len(layer_st), 2)

        # Unpacking
        m_unpacked, i_unpacked = layer_st
        self.assertIs(m_unpacked, mem)
        self.assertIs(i_unpacked, ind)

    def test_layer_state_detach(self):
        """Verifies that .detach() severs the autograd graph without affecting the source."""
        x = torch.randn(2, 4, 16, 32, requires_grad=True)
        mem = x * 2.0
        self.assertTrue(mem.requires_grad)

        layer_st = DeltaGridLayerState(memory=mem, indicator=None)
        detached = layer_st.detach()

        self.assertFalse(detached.memory.requires_grad)
        self.assertTrue(layer_st.memory.requires_grad)

    def test_grid_state_container(self):
        """Verifies multi-layer recurrent state aggregation and legacy access."""
        state = DeltaGridState()
        for _ in range(4):
            mem = torch.randn(2, 2, 8, 16)
            ind = torch.randn(2, 32, 3)
            state.append(mem, ind)

        self.assertEqual(len(state), 4)

        # Indexing returns legacy tuple (mem, ind)
        m0, i0 = state[0]
        self.assertEqual(m0.shape, (2, 2, 8, 16))
        self.assertEqual(i0.shape, (2, 32, 3))

        # Iteration across layers
        count = 0
        for m, i in state:
            self.assertIsNotNone(m)
            self.assertIsNotNone(i)
            count += 1
        self.assertEqual(count, 4)

    def test_grid_state_tbptt_detach(self):
        """Verifies that .detach() across TBPTT boundary prevents graph retention."""
        state = DeltaGridState()
        for _ in range(3):
            x = torch.randn(2, 2, 8, 16, requires_grad=True)
            state.append(memory=x * 3.0, indicator=None)

        self.assertTrue(state[0][0].requires_grad)
        tbptt_state = state.detach()

        for mem, ind in tbptt_state:
            self.assertFalse(mem.requires_grad)

    def test_legacy_format_conversion(self):
        """Verifies seamless conversion between legacy list of tuples and DeltaGridState."""
        legacy_list = [
            (torch.randn(2, 2, 8, 16), torch.randn(2, 32, 3)),
            (torch.randn(2, 2, 8, 16), None),
        ]
        state = DeltaGridState.from_legacy(legacy_list)
        self.assertIsInstance(state, DeltaGridState)
        self.assertEqual(len(state), 2)

        converted_back = state.to_legacy()
        self.assertEqual(len(converted_back), 2)
        self.assertEqual(converted_back[0][0].shape, legacy_list[0][0].shape)
        self.assertIsNone(converted_back[1][1])

    def test_state_methods_and_repr(self):
        """Verifies clone, to, reset, get_layer, repr, and boundary exceptions."""
        layer_st = DeltaGridLayerState(torch.randn(2, 4), torch.randn(2, 4))
        self.assertIn("DeltaGridLayerState", repr(layer_st))
        
        # Test to() and clone()
        cloned_layer = layer_st.clone()
        self.assertEqual(cloned_layer.memory.shape, layer_st.memory.shape)
        moved_layer = layer_st.to(dtype=torch.float64)
        self.assertEqual(moved_layer.memory.dtype, torch.float64)

        # Index bounds
        with self.assertRaises(IndexError):
            _ = layer_st[2]

        state = DeltaGridState([layer_st])
        self.assertIn("DeltaGridState", repr(state))
        self.assertIs(state.get_layer(0), layer_st)

        cloned_state = state.clone()
        self.assertEqual(len(cloned_state), 1)

        moved_state = state.to(dtype=torch.float64)
        self.assertEqual(moved_state.get_layer(0).memory.dtype, torch.float64)

        state.reset()
        self.assertEqual(len(state), 0)


if __name__ == "__main__":
    unittest.main()

