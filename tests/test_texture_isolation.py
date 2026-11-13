"""Texture comparisons must keep simulated geometry and labels fixed."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image

from roofcheck import synth


class TextureIsolationTests(unittest.TestCase):
    def setUp(self):
        yy, xx = np.mgrid[:160, :240]
        self.image = np.stack([70 + xx // 4, 80 + yy // 5, 90 + xx // 5], axis=2).astype(np.uint8)
        self.quad = np.array([[20, 140], [220, 140], [200, 30], [40, 30]], dtype=np.float32)
        self.geom = {'length_m': 5., 'depth_m': 4., 'pitch_deg': 20.}
        gy, gx = np.mgrid[:128, :128]
        self.bank = (np.sin(gx / 4) + np.cos(gy / 7)).astype(np.float32)

    def render(self, state, material, partial_style='legacy', texture_seed=991):
        return synth.make_sample(self.image, self.quad, self.quad, np.random.default_rng(117), self.geom,
                                 state=state, fence_mode='none', structure_mode='manual',
                                 texture_policy='isolated', texture_seed=texture_seed, apply_material=material,
                                 texture_strength=0.16, partial_style=partial_style, scene=None)

    def test_texture_changes_appearance_without_changing_geometry(self):
        with patch.object(synth, '_TEX', [self.bank]):
            for state, style in (('not_cleared', 'legacy'), ('partially_cleared', 'legacy'),
                                 ('partially_cleared', 'soft_front'), ('cleared', 'legacy'), ('thaw', 'legacy')):
                with self.subTest(state=state, style=style):
                    plain = self.render(state, False, style)
                    textured = self.render(state, True, style)
                    for name in ('depth_mm', 'classes', 'roi', 'seams'):
                        np.testing.assert_array_equal(getattr(plain, name), getattr(textured, name), err_msg=name)
                    np.testing.assert_array_equal(synth.labels_at(plain.depth_mm, plain.classes, 2),
                                                  synth.labels_at(textured.depth_mm, textured.classes, 2))
                    self.assertGreater(np.count_nonzero(plain.image != textured.image), 0)

    def test_texture_seed_changes_appearance_only(self):
        with patch.object(synth, '_TEX', [self.bank]):
            a = self.render('not_cleared', True, texture_seed=991)
            b = self.render('not_cleared', True, texture_seed=992)
        np.testing.assert_array_equal(a.depth_mm, b.depth_mm)
        np.testing.assert_array_equal(a.classes, b.classes)
        self.assertGreater(np.count_nonzero(a.image != b.image), 0)

    def test_requires_real_bank_and_planar_mode(self):
        with patch.object(synth, '_TEX', []):
            with self.assertRaisesRegex(ValueError, 'nonempty texture bank'):
                self.render('cleared', True)
        with self.assertRaisesRegex(ValueError, 'planar'):
            synth.make_sample(self.image, self.quad, self.quad, np.random.default_rng(1), self.geom,
                              texture_policy='isolated', scene=object())

    def test_batch_comparison_detects_tampering(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            Image.fromarray(self.image).save(tmp / 'frame.png')
            ref = {'object_id': 'test_roof', 'vantage_id': 'v1', 'image': 'frame.png',
                   'width': 240, 'height': 160,
                   'polygons': [{'role': 'roof', 'points': self.quad.tolist()}, {'role': 'slope', 'points': self.quad.tolist()}],
                   'geometry': self.geom}
            (tmp / 'reference.json').write_text(json.dumps(ref))
            row = {'id': 'test_roof', 'group_id': 'test_building', 'image': 'frame.png', 'reference': 'reference.json',
                   'split': 'train', 'source': 'generated fixture', 'license': 'fixture', 'snow_free_reviewed': 'yes'}
            def write_csv(path, item):
                with path.open('w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=list(item)); writer.writeheader(); writer.writerow(item)
            write_csv(tmp / 'input.csv', row)
            tex_dir = tmp / 'textures'; tex_path = tex_dir / 'ambientcg' / 'Snow001' / 'Snow001_color_2k.jpg'
            tex_path.parent.mkdir(parents=True)
            texture = np.clip(127 + 70 * self.bank, 0, 255).astype(np.uint8)
            Image.fromarray(texture).save(tex_path)
            write_csv(tmp / 'texture.csv', {'path': 'ambientcg/Snow001/Snow001_color_2k.jpg',
                        'source': 'fixture', 'license': 'fixture', 'donor_group': 'fixture', 'donor_split': 'external'})
            base = [sys.executable, str(root / 'tools/batch_existing.py'), '--input', str(tmp / 'input.csv'),
                    '--texture-policy', 'isolated', '--texture-strength', '0.16', '--context', 'local',
                    '--fence-mode', 'none', '--structure-mode', 'manual']
            env = os.environ.copy()
            subprocess.run(base + ['--output', str(tmp / 'plain')], check=True, capture_output=True, text=True, env=env)
            self.assertEqual(json.loads((tmp / 'plain' / 'run.json').read_text())['config']['snow_variety'], 'mixed')
            subprocess.run(base + ['--output', str(tmp / 'tex'), '--texture-dir', str(tex_dir), '--texture-manifest', str(tmp / 'texture.csv')],
                           check=True, capture_output=True, text=True, env=env)
            compare = [sys.executable, str(root / 'tools/compare_texture_batches.py'), '--baseline', str(tmp / 'plain'),
                       '--textured', str(tmp / 'tex'), '--output', str(tmp / 'comparison')]
            subprocess.run(compare, check=True, capture_output=True, text=True, env=env)
            result = json.loads((tmp / 'comparison' / 'comparison.json').read_text())['samples']
            self.assertEqual(len(result), 4)
            self.assertTrue(all(r['mask_identical'] and r['roi_identical'] for r in result))
            self.assertTrue(any(r['roi_pixels_changed'] > 0 for r in result))
            mask = tmp / 'tex' / 'samples' / result[0]['id'] / 'snow_threshold_2mm_experimental.png'
            with mask.open('ab') as f: f.write(b'tampered')
            rejected = subprocess.run(compare[:-1] + [str(tmp / 'other')], capture_output=True, text=True, env=env)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn('checksum mismatch', rejected.stderr)


if __name__ == '__main__': unittest.main()
