#!/usr/bin/env python3
"""Check paired planar texture experiments and generate an honest visual comparison."""
import argparse
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from batch_existing import check_sample, digest

CONFIG_KEYS = ('inputs', 'states', 'per_state', 'seed', 'scale', 'context', 'ground_snow_mm', 'ground_snow_levels', 'save_depth', 'feather_px',
               'fence_mode', 'structure_mode', 'partial_style', 'visibility_threshold', 'texture_policy',
               'texture_strength', 'snow_variety', 'snow_brightness', 'snow_detail', 'code_sha256', 'python', 'numpy', 'opencv', 'pillow', 'scipy')
FILES = {'image.png', 'snow_threshold_2mm_experimental.png', 'roi.png', 'metadata.json'}


def load(root):
    run = json.loads((root / 'run.json').read_text())
    qa = json.loads((root / 'qa_report.json').read_text())
    if run['fingerprint'] != qa['fingerprint']:
        raise ValueError('Run and QA fingerprints differ')
    records = qa['samples']
    if not records or len({r['id'] for r in records}) != len(records):
        raise ValueError('Empty or duplicate sample IDs')
    return run['config'], {r['id']: r for r in records}


def read_sample(root, row):
    sid, rel = row['id'], row['path']
    if rel != f'samples/{sid}' or (root / rel).resolve().parent != (root / 'samples').resolve():
        raise ValueError(f'{sid}: invalid sample path')
    folder = root / rel
    if json.loads((folder / 'batch_record.json').read_text()) != row or set(row['files_sha256']) not in (FILES, FILES | {'depth_mm_experimental.npy'}):
        raise ValueError(f'{sid}: QA record differs from sample record')
    for name, expected in row['files_sha256'].items():
        if digest(folder / name) != expected:
            raise ValueError(f'{sid}: checksum mismatch for {name}')
    checked = check_sample(folder)
    if any(checked[k] != row['qa'][k] for k in checked):
        raise ValueError(f'{sid}: mask or ROI failed QA')
    with Image.open(folder / 'image.png') as f: image = np.asarray(f.convert('RGB'))
    with Image.open(folder / 'snow_threshold_2mm_experimental.png') as f: mask = np.asarray(f)
    with Image.open(folder / 'roi.png') as f: roi = np.asarray(f)
    return image, mask, roi, json.loads((folder / 'metadata.json').read_text())


def compare(baseline, textured):
    cfg_a, rows_a = load(baseline); cfg_b, rows_b = load(textured)
    if cfg_a.get('texture_policy') != 'isolated' or cfg_b.get('texture_policy') != 'isolated':
        raise ValueError('Both runs require --texture-policy isolated')
    if cfg_a.get('texture_mode') != 'procedural_only' or cfg_b.get('texture_mode') != 'exemplar':
        raise ValueError('First run must be procedural, second must have a texture manifest')
    if cfg_a.get('visibility_threshold') is not None or cfg_b.get('visibility_threshold') is not None:
        raise ValueError('Disable visibility-threshold for a strict mask comparison')
    for key in CONFIG_KEYS:
        if cfg_a.get(key) != cfg_b.get(key): raise ValueError(f'Run configurations differ: {key}')
    if rows_a.keys() != rows_b.keys(): raise ValueError('Sample IDs differ')
    result = []
    for sid in sorted(rows_a):
        a, ma, ra, meta_a = read_sample(baseline, rows_a[sid])
        b, mb, rb, meta_b = read_sample(textured, rows_b[sid])
        if a.shape != b.shape or not np.array_equal(ma, mb) or not np.array_equal(ra, rb):
            raise ValueError(f'{sid}: image dimensions, labels or ROI differ')
        for key in ('input_sha256', 'reference_sha256', 'seed', 'texture_seed', 'scale', 'snow_brightness', 'snow_detail', 'geometry', 'info', 'geometry_arrays_sha256'):
            if meta_a.get(key) != meta_b.get(key): raise ValueError(f'{sid}: simulation metadata differs: {key}')
        if not meta_a.get('geometry_arrays_sha256'):
            raise ValueError(f'{sid}: geometry hash missing')
        delta = np.abs(a.astype(np.int16) - b.astype(np.int16))
        inside = ra == 1
        result.append({'id': sid, 'state': rows_a[sid]['state'], 'mask_identical': True,
                       'roi_identical': True, 'geometry_arrays_sha256': meta_a['geometry_arrays_sha256'],
                       'rgb_channels_different': int((delta > 0).sum()),
                       'roi_pixels_changed': int((delta.max(axis=2)[inside] > 0).sum()),
                       'roi_mean_absolute_rgb_difference': round(float(delta[inside].mean()), 4),
                       'roi_max_rgb_difference': int(delta[inside].max()),
                       '_left': a, '_right': b, '_roi': ra})
    return result


def sheet(records, output):
    card_w, image_h, header_h = 500, 280, 30
    result = Image.new('RGB', (card_w * 2, (image_h + header_h) * len(records)), 'white')
    draw = ImageDraw.Draw(result)
    for i, r in enumerate(records):
        ys, xs = np.nonzero(r['_roi'])
        pad = max(12, int(max(xs.max()-xs.min(), ys.max()-ys.min()) * 0.12))
        x0, x1 = max(0, xs.min()-pad), min(r['_roi'].shape[1], xs.max()+pad+1)
        y0, y1 = max(0, ys.min()-pad), min(r['_roi'].shape[0], ys.max()+pad+1)
        top = i * (image_h + header_h)
        draw.text((8, top+6), f"{r['state']}: procedural", fill='black')
        draw.text((card_w+8, top+6), f"{r['state']}: material", fill='black')
        for j, key in enumerate(('_left', '_right')):
            im = Image.fromarray(r[key][y0:y1, x0:x1])
            im = ImageOps.contain(im, (card_w-10, image_h-5), Image.Resampling.LANCZOS)
            result.paste(im, (j*card_w+5, top+header_h))
    result.save(output, quality=92)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--textured', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    baseline, textured, output = a.baseline.resolve(), a.textured.resolve(), a.output.resolve()
    if output.exists(): raise FileExistsError(output)
    records = compare(baseline, textured)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix='.texture-comparison-', dir=output.parent))
    try:
        sheet(records, tmp / 'comparison.jpg')
        summary = {'baseline': str(baseline), 'textured': str(textured), 'scope': 'planar; visual analysis, not a quality metric',
                   'samples': [{k: v for k, v in r.items() if not k.startswith('_')} for r in records]}
        (tmp / 'comparison.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        tmp.rename(output)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    print(f'{len(records)} paired samples checked:', output)


if __name__ == '__main__': main()
