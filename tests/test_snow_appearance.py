"""Snow appearance controls must preserve geometry and produce distinguishable variations."""
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
from roofcheck import synth
from tools.batch_existing import digest, snow_appearance, verify_code_hashes


class SnowAppearanceTests(unittest.TestCase):
    def test_bright_smooth_keeps_geometry_and_changes_appearance(self):
        yy, xx = np.mgrid[:160, :240]
        image = np.stack([70 + xx // 4, 80 + yy // 5, 90 + xx // 5], axis=2).astype(np.uint8)
        quad = np.array([[20, 140], [220, 140], [200, 30], [40, 30]], dtype=np.float32)
        geom = {'length_m': 5., 'depth_m': 4., 'pitch_deg': 20.}
        def render(brightness, detail):
            return synth.make_sample(image, quad, quad, np.random.default_rng(117), geom,
                    state='not_cleared', fence_mode='none', structure_mode='manual',
                    texture_policy='isolated', snow_brightness=brightness, snow_detail=detail)
        normal = render(1, 1)
        bright = render(1.08, .2)
        smooth = render(1.08, 0)
        for name in ('depth_mm', 'classes', 'roi', 'seams'):
            np.testing.assert_array_equal(getattr(normal, name), getattr(bright, name))
            np.testing.assert_array_equal(getattr(normal, name), getattr(smooth, name))
        mask = (normal.depth_mm > 10) & normal.roi
        mask = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        self.assertGreater(mask.sum(), 100)
        def metrics(im):
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
            fine = np.abs(gray - cv2.GaussianBlur(gray, (0, 0), 1.4))
            return float(gray[mask].mean()), float(fine[mask].mean())
        lum_a, detail_a = metrics(normal.image)
        lum_b, detail_b = metrics(bright.image)
        _, detail_c = metrics(smooth.image)
        self.assertGreater(lum_b, lum_a)
        self.assertLess(detail_b, detail_a)
        self.assertLess(detail_c, detail_b)

    def test_mixed_profiles_are_deterministic_and_diverse(self):
        profiles = [snow_appearance(i, 'mixed') for i in range(100)]
        self.assertEqual(profiles, [snow_appearance(i, 'mixed') for i in range(100)])
        self.assertGreater(len(set(profiles)), 90)
        self.assertTrue(any(detail >= 0.7 for _, detail in profiles))
        self.assertTrue(any(0.12 <= detail <= 0.35 for _, detail in profiles))
        self.assertTrue(any(detail == 0.0 for _, detail in profiles))
        self.assertTrue(all(0.97 <= brightness <= 1.10 and 0 <= detail <= 1 for brightness, detail in profiles))
        self.assertEqual(snow_appearance(1, 'fixed', 1.1, .3), (1.1, .3))

    def test_invalid_parameters_are_rejected(self):
        image = np.full((12, 12, 3), 90, np.uint8)
        quad = np.array([[0, 11], [11, 11], [11, 0], [0, 0]], dtype=np.float32)
        geom = {'length_m': 2, 'depth_m': 2, 'pitch_deg': 20}
        with self.assertRaises(ValueError):
            synth.make_sample(image, quad, quad, np.random.default_rng(1), geom, snow_brightness=1.5)
        with self.assertRaises(ValueError):
            synth.make_sample(image, quad, quad, np.random.default_rng(1), geom, snow_detail=-.1)

    def test_batch_detects_code_changed_mid_run(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); code = root / 'module.py'; code.write_text('old')
            original = {'module.py': digest(code)}
            with patch('tools.batch_existing.ROOT', root):
                verify_code_hashes(original)
                code.write_text('new')
                with self.assertRaisesRegex(ValueError, 'Code changed'):
                    verify_code_hashes(original)


if __name__ == '__main__': unittest.main()
