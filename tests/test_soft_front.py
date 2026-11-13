"""Checks for the opt-in, irregular partial-clearing shape."""
import unittest
import numpy as np
from roofcheck.synth import soft_front_mask, apply_state


class SoftFrontTests(unittest.TestCase):
    def test_front_is_seeded_varied_and_eave_connected(self):
        a = soft_front_mask((200, 320), np.random.default_rng(31))
        b = soft_front_mask((200, 320), np.random.default_rng(31))
        np.testing.assert_array_equal(a, b)
        self.assertEqual(a.shape, (200, 320))
        self.assertTrue(np.all(a[-1]))
        self.assertFalse(np.any(a[0]))
        front = np.argmax(a, axis=0)
        self.assertGreater(np.ptp(front), 4)
        self.assertTrue(np.all(np.diff(a.astype(np.int8), axis=0) >= 0))
        self.assertGreater(a.mean(), .05)
        self.assertLess(a.mean(), .85)

    def test_partial_depth_and_label_are_derived_from_same_field(self):
        h, w = 120, 160
        depth = np.full((h, w), 50., np.float32)
        layers = {'structure': np.zeros((h, w), bool), 'surface': np.ones((h, w), bool),
                  'seams': np.zeros((h, w), bool)}
        out, wet, albedo, info = apply_state(depth, 'partially_cleared', layers, {},
                                            layers['surface'], np.random.default_rng(91), 2, partial_style='soft_front')
        self.assertEqual(info['clearing'], 'soft_front')
        self.assertEqual(out.shape, depth.shape)
        self.assertTrue(np.isfinite(out).all() and np.all(out >= 0))
        self.assertGreater(np.count_nonzero(out < 20), 0)
        self.assertGreater(np.count_nonzero(out > 45), 0)
        self.assertLess(float(out[info['_cut']].mean()), float(out[~info['_cut']].mean()))
        self.assertEqual(wet.shape, out.shape)
        self.assertEqual(albedo.shape, out.shape)


if __name__ == '__main__': unittest.main()
