#!/usr/bin/env python3
"""Combine segmentation manifests into a self-contained dataset directory."""
import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

FIELDS = ('id', 'image', 'mask', 'group_id', 'split', 'kind', 'roi')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', action='append', required=True, type=Path, help='manifest path; repeat for each source')
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    records, ids, groups = [], set(), {}
    for manifest in a.manifest:
        manifest = manifest.resolve()
        with manifest.open(newline='', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or not set(FIELDS).issubset(reader.fieldnames):
                raise ValueError(f'Missing required columns: {manifest}')
            for row in reader:
                sid = row['id']
                if not sid or sid in ids or not row['group_id']:
                    raise ValueError(f'Duplicate/empty id or group: {sid}')
                ids.add(sid)
                split, kind, group = row['split'], row['kind'], row['group_id']
                if split not in ('train', 'val', 'test') or kind not in ('real', 'synthetic') or (kind == 'synthetic' and split != 'train'):
                    raise ValueError(f'Invalid split or kind: {sid}')
                if group in groups and groups[group] != split:
                    raise ValueError(f'Group appears in multiple splits: {group}')
                groups[group] = split
                paths = {}
                for key in ('image', 'mask', 'roi'):
                    raw = row[key]
                    if not raw and key == 'roi':
                        continue
                    path = (manifest.parent / raw).resolve()
                    if not raw or not path.is_file() or not path.is_relative_to(manifest.parent):
                        raise ValueError(f'{sid}: invalid {key} path')
                    paths[key] = path
                with Image.open(paths['image']) as im: size = im.size
                with Image.open(paths['mask']) as im: mask = np.asarray(im)
                if mask.ndim != 2 or mask.shape != (size[1], size[0]) or not set(np.unique(mask)) <= {0, 1, 255} or not (mask != 255).any():
                    raise ValueError(f'{sid}: invalid mask')
                if 'roi' in paths:
                    with Image.open(paths['roi']) as im: roi = np.asarray(im)
                    if roi.shape != mask.shape or not set(np.unique(roi)) <= {0, 1} or np.any(mask[roi == 0] != 255):
                        raise ValueError(f'{sid}: invalid ROI')
                records.append((row, paths, bool((mask == 1).any())))
    if not records:
        raise ValueError('No records')
    real_train = {r['group_id'] for r, _, _ in records if r['kind'] == 'real' and r['split'] == 'train'}
    if any(r['group_id'] not in real_train for r, _, _ in records if r['kind'] == 'synthetic'):
        raise ValueError('Each synthetic group needs a real train sample')
    if a.output.exists():
        raise FileExistsError(a.output)
    out = a.output.resolve(); out.mkdir(parents=True)
    (out / 'samples').mkdir()
    result = []
    for i, (row, paths, _) in enumerate(records):
        sample = out / 'samples' / f'{i:06d}'; sample.mkdir()
        new = {key: row.get(key, '') for key in FIELDS}
        for key, path in paths.items():
            name = key + path.suffix.lower()
            shutil.copy2(path, sample / name)
            new[key] = f'samples/{i:06d}/{name}'
        result.append(new)
    with (out / 'manifest.csv').open('x', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(result)
    print(f'Assembled {len(result)} samples in {out}')
    for split in ('train', 'val', 'test'):
        subset = [(row, snow) for row, _, snow in records if row['split'] == split]
        print(f'{split}: {len(subset)} total, {sum(snow for _, snow in subset)} with snow')
        if not subset or not any(snow for _, snow in subset) or not any(not snow for _, snow in subset):
            print(f'WARNING: {split} lacks snow-positive or clean examples; training notebook cannot run yet')


if __name__ == '__main__': main()
