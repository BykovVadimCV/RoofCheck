#!/usr/bin/env python3
"""Create an entirely invented roof fixture for pipeline checks, not model training."""
import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
image = Image.new('RGB', (480, 320), (145, 169, 184))
d = ImageDraw.Draw(image)
d.rectangle((0, 245, 480, 320), fill=(90, 91, 89))
roof = [(25, 260), (450, 260), (384, 92), (104, 92)]
d.polygon(roof, fill=(100, 109, 111))
for i in range(1, 10):
    d.line((25 + 425 * i / 10, 260, 104 + 280 * i / 10, 92), fill=(76, 85, 88), width=2)
d.polygon([(240, 110), (270, 110), (270, 145), (240, 145)], fill=(64, 65, 65))
image.save(a.output / 'invented_roof.png')
reference = {'object_id': 'invented_roof', 'vantage_id': 'mock', 'image': 'invented_roof.png',
             'width': 480, 'height': 320,
             'polygons': [{'role': 'roof', 'points': roof}, {'role': 'slope', 'points': roof},
                          {'role': 'structure', 'points': [[240, 110], [270, 110], [270, 145], [240, 145]]}],
             'geometry': {'length_m': 6, 'depth_m': 4, 'pitch_deg': 20},
             'notes': 'Artificial illustration for software checks; never a real dataset or validation example.'}
(a.output / 'reference.json').write_text(json.dumps(reference, ensure_ascii=False, indent=2))
row = {'id': 'invented_roof', 'group_id': 'invented_building', 'image': 'invented_roof.png',
       'reference': 'reference.json', 'split': 'train', 'source': 'artificial software test illustration',
       'license': 'generated test illustration', 'snow_free_reviewed': 'yes'}
with (a.output / 'input.csv').open('w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(row)); w.writeheader(); w.writerow(row)
print('Artificial fixture only:', a.output)
