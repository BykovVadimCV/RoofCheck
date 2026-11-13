"""Unit tests for protected pixels, feathering and uncertainty labels."""
import unittest
import numpy as np
from roofcheck.local_composite import ignore_low_change_snow, local_composite


class LocalCompositeTests(unittest.TestCase):
    def test_protection_and_transition_labels(self):
        source = np.full((24, 32, 3), 40, np.uint8)
        snow = np.full_like(source, 220)
        label = np.ones(source.shape[:2], np.uint8)
        roi = np.zeros(label.shape, bool); roi[3:21, 3:29] = True
        protected = np.zeros_like(roi); protected[9:12, 10:13] = True
        output, target, alpha = local_composite(source, snow, label, roi, protected, feather_px=3)
        np.testing.assert_array_equal(output[~roi | protected], source[~roi | protected])
        self.assertTrue(np.all(target[~roi | protected] == 255))
        blend = (alpha > 0) & (alpha < 1)
        self.assertGreater(blend.sum(), 0)
        self.assertTrue(np.all(target[blend] == 255))
        self.assertGreater((alpha == 1).sum(), 0)
        self.assertTrue(np.all(target[alpha == 1] == 1))
        self.assertTrue(np.all(output[alpha == 1] == 220))

    def test_preserve_unknown_label_and_no_feather(self):
        source = np.zeros((6, 6, 3), np.uint8)
        rendered = np.full_like(source, 80)
        mask = np.zeros((6, 6), np.uint8); mask[2, 2] = 255
        roi = np.zeros((6, 6), np.uint8); roi[1:5, 1:5] = 1
        output, label, alpha = local_composite(source, rendered, mask, roi, feather_px=0)
        self.assertEqual(label[2, 2], 255)
        self.assertEqual(output[0, 0, 0], 0)
        self.assertEqual(output[2, 2, 0], 80)
        self.assertEqual(set(np.unique(alpha)), {0., 1.})

    def test_reject_bad_inputs(self):
        source = np.zeros((3, 3, 3), np.uint8)
        label = np.full((3, 3), 3, np.uint8)
        with self.assertRaises(ValueError): local_composite(source, source, label, np.ones((3, 3)))
        with self.assertRaises(ValueError): local_composite(source, source, label*0, np.ones((3, 3)), feather_px=-1)

    def test_low_change_snow_becomes_ignore_only(self):
        source = np.full((2, 3, 3), 100, np.uint8)
        rendered = source.copy()
        rendered[0, 1] = 109
        rendered[0, 2] = 104
        rendered[1, 0] = 105
        label = np.array([[1, 1, 0], [255, 1, 1]], np.uint8)
        out, count = ignore_low_change_snow(source, rendered, label, threshold=8)
        np.testing.assert_array_equal(out, [[255, 1, 0], [255, 255, 255]])
        np.testing.assert_array_equal(label, [[1, 1, 0], [255, 1, 1]])
        self.assertEqual(count, 3)


if __name__ == '__main__': unittest.main()
