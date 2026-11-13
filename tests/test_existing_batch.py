"""Batch integration tests use invented images, never client photographs or weights."""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.batch_existing import check_sample, read_input, read_textures, snow_appearance


class ExistingBatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        y, x = np.mgrid[:160, :240]
        rgb = np.stack([100 + x // 6, 105 + y // 6, 110 + x // 8], axis=2).astype(np.uint8)
        Image.fromarray(rgb).save(self.root / 'clean.png')
        ref = {'object_id': 'toy', 'vantage_id': 'v1', 'image': 'clean.png', 'width': 240, 'height': 160,
               'polygons': [{'role': 'roof', 'points': [[15, 145], [225, 145], [200, 30], [40, 30]]},
                            {'role': 'slope', 'points': [[15, 145], [225, 145], [200, 30], [40, 30]]},
                            {'role': 'exclude', 'points': [[80, 70], [100, 70], [100, 90], [80, 90]]},
                            {'role': 'structure', 'points': [[120, 60], [135, 60], [135, 75], [120, 75]]}],
               'geometry': {'length_m': 5, 'depth_m': 4, 'pitch_deg': 20}}
        (self.root / 'reference.json').write_text(json.dumps(ref))
        self.csv = self.root / 'input.csv'
        self.row = dict(id='toy', group_id='toy_building', image='clean.png', reference='reference.json',
                        split='train', source='generated test fixture', license='not a real image', snow_free_reviewed='yes')
        self.write_csv()

    def write_csv(self):
        with self.csv.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(self.row))
            writer.writeheader(); writer.writerow(self.row)

    def test_reject_non_train_and_unreviewed(self):
        self.row['split'] = 'test'; self.write_csv()
        with self.assertRaisesRegex(ValueError, 'Only train'): read_input(self.csv)
        self.row['split'] = 'train'; self.row['snow_free_reviewed'] = 'no'; self.write_csv()
        with self.assertRaisesRegex(ValueError, 'reviewed'): read_input(self.csv)

    def test_reject_invalid_roi_and_mask(self):
        p = self.root / 'bad'; p.mkdir()
        Image.new('RGB', (10, 10)).save(p / 'image.png')
        Image.fromarray(np.zeros((10, 10), dtype=np.uint8)).save(p / 'snow_threshold_2mm_experimental.png')
        Image.fromarray(np.zeros((10, 10), dtype=np.uint8)).save(p / 'roi.png')
        with self.assertRaises(ValueError): check_sample(p)

    def test_low_contrast_is_reported_without_rejecting(self):
        p = self.root / 'lowcontrast'; p.mkdir()
        source = np.full((10, 10, 3), 240, dtype=np.uint8)
        Image.fromarray(source).save(self.root / 'light.png')
        rendered = source.copy(); rendered[2:4, 2:4] += 4
        Image.fromarray(rendered).save(p / 'image.png')
        label = np.full((10, 10), 255, dtype=np.uint8); label[2:4, 2:4] = 1
        roi = np.zeros_like(label); roi[2:4, 2:4] = 1
        Image.fromarray(label).save(p / 'snow_threshold_2mm_experimental.png')
        Image.fromarray(roi).save(p / 'roi.png')
        qa = check_sample(p, self.root / 'light.png')
        self.assertEqual(qa['snow_fraction_rgb_change_le8'], 1.0)

    def test_texture_provenance_and_split(self):
        root = self.root / 'textures'; asset = root / 'ambientcg' / 'Snow002' / 'Snow002_color_2k.jpg'
        asset.parent.mkdir(parents=True)
        Image.new('RGB', (12, 12), (180, 190, 200)).save(asset)
        manifest = self.root / 'textures.csv'
        row = {'path': 'ambientcg/Snow002/Snow002_color_2k.jpg', 'source': 'https://example.org/material',
               'license': 'CC0', 'donor_group': 'source_1', 'donor_split': 'external'}
        def write():
            with manifest.open('w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(row)); w.writeheader(); w.writerow(row)
        write()
        checked = read_textures(root, manifest, {'train_group'})
        self.assertIn(row['path'], checked['assets'])
        row['donor_split'] = 'test'; write()
        with self.assertRaisesRegex(ValueError, 'donors'): read_textures(root, manifest, {'train_group'})
        row['donor_split'] = 'train'; write()
        with self.assertRaisesRegex(ValueError, 'donor_group'): read_textures(root, manifest, {'train_group'})
        row['donor_group'] = 'train_group'; write()
        self.assertEqual(len(read_textures(root, manifest, {'train_group'})['assets']), 1)
        Image.new('RGB', (12, 12)).save(asset.parent / 'extra_color_2k.jpg')
        with self.assertRaisesRegex(ValueError, 'Every texture'): read_textures(root, manifest, {'train_group'})

    def test_batch_resume_and_integrity(self):
        out = self.root / 'output'
        cmd = [sys.executable, str(ROOT / 'tools/batch_existing.py'), '--input', str(self.csv), '--output', str(out), '--context', 'local', '--fence-mode', 'none', '--structure-mode', 'manual', '--visibility-threshold', '8', '--snow-variety', 'mixed']
        subprocess.run(cmd, check=True, capture_output=True, text=True, env=os.environ.copy())
        report = json.loads((out / 'qa_report.json').read_text())
        self.assertEqual(len(report['samples']), 4)
        for row in report['samples']:
            self.assertEqual((row['snow_brightness'], row['snow_detail']), snow_appearance(row['seed'], 'mixed'))
            meta = json.loads((out / row['path'] / 'metadata.json').read_text())
            self.assertEqual((meta['snow_brightness'], meta['snow_detail']), (row['snow_brightness'], row['snow_detail']))
        sample = out / report['samples'][0]['path']
        label = np.asarray(Image.open(sample / 'snow_threshold_2mm_experimental.png'))
        self.assertTrue(np.all(label[70:90, 80:100] == 255))
        self.assertTrue(np.all(label[60:75, 120:135] == 255))
        rendered = np.asarray(Image.open(sample / 'image.png').convert('RGB'))
        original = np.asarray(Image.open(self.root / 'clean.png').convert('RGB'))
        np.testing.assert_array_equal(rendered[60:75, 120:135], original[60:75, 120:135])
        self.assertEqual(report['samples'][0]['qa']['outside_roi_mean_abs_change'], 0)
        self.assertEqual(report['samples'][0]['qa']['outside_roi_fraction_changed_gt8'], 0)
        metadata = json.loads((sample / 'metadata.json').read_text())
        self.assertEqual(metadata['info']['fence_mode'], 'none')
        self.assertEqual(metadata['info']['fence_drift_m'], 0)
        self.assertEqual(metadata['info']['structure_mode'], 'manual')
        self.assertEqual(metadata['visibility_threshold'], 8)
        self.assertGreaterEqual(metadata['visibility_ignore_pixels'], 0)
        self.assertIn(report['samples'][0]['qa']['snow_fraction_rgb_change_le8'], (0.0, None))
        subprocess.run(cmd + ['--resume'], check=True, capture_output=True, text=True, env=os.environ.copy())
        review = [sys.executable, str(ROOT / 'tools/review_batch.py'), '--run', str(out)]
        subprocess.run(review + ['--template'], check=True, capture_output=True, text=True)
        pending = subprocess.run(review + ['--export'], capture_output=True, text=True)
        self.assertNotEqual(pending.returncode, 0)
        with (out / 'review.csv').open(newline='') as f:
            reader = csv.DictReader(f); fields = reader.fieldnames; decisions = list(reader)
        for i, r in enumerate(decisions):
            r.update(decision='accept' if i == 0 else 'reject', reviewer='test', note='synthetic fixture')
        with (out / 'review.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(decisions)
        subprocess.run(review + ['--export'], check=True, capture_output=True, text=True)
        with (out / 'approved_manifest.csv').open(newline='') as f:
            approved = list(csv.DictReader(f))
        self.assertEqual([r['id'] for r in approved], [report['samples'][0]['id']])
        audit = json.loads((out / 'approved_manifest.csv.review.json').read_text())
        self.assertEqual(audit['manifest_sha256'], hashlib.sha256((out / 'approved_manifest.csv').read_bytes()).hexdigest())
        self.assertEqual(audit['review_csv_sha256'], hashlib.sha256((out / 'review.csv').read_bytes()).hexdigest())
        with (sample / 'image.png').open('ab') as f: f.write(b'changed')
        rejected_review = subprocess.run(review + ['--export', '--output-name', 'second.csv'], capture_output=True, text=True)
        self.assertNotEqual(rejected_review.returncode, 0)
        self.assertIn('checksum mismatch', rejected_review.stderr)
        rejected = subprocess.run(cmd + ['--resume'], capture_output=True, text=True, env=os.environ.copy())
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn('checksum mismatch', rejected.stderr)


if __name__ == '__main__': unittest.main()
