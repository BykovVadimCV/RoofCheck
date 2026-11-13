"""Integration checks for preparing real image archives and simulated depth batches."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / 'tools' / 'prepare_roof_dataset.py'
BATCH = ROOT / 'tools' / 'batch_existing.py'


class RoofDatasetPipelineTests(unittest.TestCase):
    def test_inventory_cvat_import_and_depth_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'archive' / 'Flug_100' / 'RGB'
            source.mkdir(parents=True)
            y, x = np.mgrid[:160, :240]
            rgb = np.stack([100 + x // 6, 105 + y // 6, 110 + x // 8], axis=2).astype('uint8')
            Image.fromarray(rgb).save(source / 'DJI_0001.jpg')
            thermal = source.parent / 'Thermal'
            thermal.mkdir()
            Image.fromarray(rgb).save(thermal / 'DJI_0001.jpg')
            inventory = root / 'inventory'
            subprocess.run([sys.executable, str(PREP), 'init', '--root', str(root / 'archive'), '--karlsruhe', '--stage', '--output', str(inventory)], check=True, capture_output=True)
            with (inventory / 'inventory.csv').open(newline='') as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual((source / 'DJI_0001.jpg').read_bytes(), (inventory / 'cvat_images' / row['cvat_image']).read_bytes())
            row.update(group_id='building_1', split='train', snow_free_reviewed='yes', selected='yes')
            with (inventory / 'inventory.csv').open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                writer.writeheader(); writer.writerow(row)
            annotations = ET.Element('annotations')
            image = ET.SubElement(annotations, 'image', name=row['cvat_image'], width='240', height='160')
            pts = '15,145;225,145;200,30;40,30'
            ET.SubElement(image, 'polygon', label='roof', points=pts)
            ET.SubElement(image, 'polygon', label='slope', points=pts)
            xml = root / 'annotations.xml'
            ET.ElementTree(annotations).write(xml)
            prepared = root / 'prepared'
            subprocess.run([sys.executable, str(PREP), 'import-cvat', '--inventory', str(inventory / 'inventory.csv'), '--xml', str(xml), '--output', str(prepared)], check=True, capture_output=True)
            out = root / 'batch'
            cmd = [sys.executable, str(BATCH), '--input', str(prepared / 'input.csv'), '--output', str(out), '--per-state', '2', '--ground-snow-levels', '10', '200', '--save-depth', '--context', 'local-volume', '--fence-mode', 'none', '--structure-mode', 'manual']
            subprocess.run(cmd, check=True, capture_output=True)
            report = json.loads((out / 'qa_report.json').read_text())
            self.assertEqual(len(report['samples']), 8)
            self.assertEqual({r['ground_snow_mm'] for r in report['samples']}, {10, 200})
            for row in report['samples']:
                sample = out / row['path']
                arr = np.load(sample / 'depth_mm_experimental.npy', allow_pickle=False)
                self.assertEqual(arr.shape, (160, 240))
                self.assertTrue(np.isfinite(arr).all())
                self.assertIn('simulated_roof_depth_p95_mm', row['qa'])
            subprocess.run(cmd + ['--resume'], check=True, capture_output=True)
            review = [sys.executable, str(ROOT / 'tools' / 'review_batch.py'), '--run', str(out)]
            subprocess.run(review + ['--template'], check=True, capture_output=True)
            with (out / 'review.csv').open(newline='') as f:
                rows = list(csv.DictReader(f))
            for row in rows:
                row.update(decision='accept', reviewer='tester', note='fixture only')
            with (out / 'review.csv').open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
            subprocess.run(review + ['--export'], check=True, capture_output=True)
            self.assertTrue((out / 'approved_manifest.csv').exists())
            clean = root / 'clean'
            subprocess.run([sys.executable, str(ROOT / 'tools' / 'export_clean_roofs.py'), '--input', str(prepared / 'input.csv'), '--output', str(clean)], check=True, capture_output=True)
            with (clean / 'real_clean_manifest.csv').open(newline='') as f:
                clean_row = next(csv.DictReader(f))
            clean_mask = np.asarray(Image.open(clean / clean_row['mask']))
            self.assertEqual(set(np.unique(clean_mask)), {0, 255})
            combined = root / 'combined'
            assemble = [sys.executable, str(ROOT / 'tools' / 'assemble_roof_dataset.py'), '--manifest', str(clean / 'real_clean_manifest.csv'), '--manifest', str(out / 'approved_manifest.csv'), '--output', str(combined)]
            subprocess.run(assemble, check=True, capture_output=True)
            with (combined / 'manifest.csv').open(newline='') as f:
                self.assertEqual(len(list(csv.DictReader(f))), 9)


if __name__ == '__main__':
    unittest.main()
