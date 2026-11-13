#!/usr/bin/env python3
"""Prepare original roof photographs and import manually reviewed CVAT polygons.

Images are staged only by explicit request; originals are never modified.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from roofcheck.cvat import write_label_spec

COLUMNS = ['id', 'image', 'cvat_image', 'width', 'height', 'group_id', 'split',
           'snow_free_reviewed', 'selected', 'source', 'license']
SOURCE = 'https://zenodo.org/records/7360996'
LICENSE = 'CC BY 4.0: Kahn, Mayer, Hou, Beiersdoerfer, Goetz, Volk'


def oriented_size(im):
    # Read only the JPEG header and EXIF, not all decoded pixels.
    width, height = im.size
    return (height, width) if im.getexif().get(274, 1) in (5, 6, 7, 8) else (width, height)


def write_csv(path, fields, rows):
    with path.open('x', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def init(args):
    source = args.root.expanduser().resolve()
    if not source.is_dir():
        raise ValueError(f'Not a directory: {source}')
    if args.karlsruhe:
        roots = [p / 'RGB' for p in source.iterdir() if p.is_dir() and re.fullmatch(r'flug_10[0-5]', p.name, re.I)]
        if re.fullmatch(r'flug_10[0-5]', source.name, re.I):
            roots.append(source / 'RGB')
        candidates = sorted(p for d in roots if d.is_dir() for p in d.iterdir() if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
    else:
        candidates = sorted(p for p in source.rglob('*') if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
    if not candidates:
        raise ValueError('No matching images found')
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    output.mkdir(parents=True)
    if args.stage:
        (output / 'cvat_images').mkdir()
    rows = []
    for path in candidates:
        rel = path.relative_to(source).as_posix()
        sid = 'img_' + hashlib.sha256(rel.encode('utf-8')).hexdigest()[:20]
        name = sid + path.suffix.lower()
        with Image.open(path) as im:
            width, height = oriented_size(im)
        if args.stage:
            try:
                shutil.copy2(path, output / 'cvat_images' / name)
            except OSError as exc:
                raise RuntimeError(f'Could not copy image {path}') from exc
        rows.append(dict(id=sid, image=str(path), cvat_image=name, width=width, height=height,
                         group_id='', split='', snow_free_reviewed='no', selected='no',
                         source=SOURCE if args.karlsruhe else '', license=LICENSE if args.karlsruhe else ''))
    write_csv(output / 'inventory.csv', COLUMNS, rows)
    write_label_spec(output / 'cvat_labels.json')
    print(f'Indexed {len(rows)} RGB images: {output / "inventory.csv"}')
    if args.stage:
        print(f'Copied CVAT images: {output / "cvat_images"}')


def stage_selected(args):
    rows = [r for r in read_csv(args.inventory) if r.get('selected', '').lower() == 'yes']
    if not rows:
        raise ValueError('No selected rows; set selected=yes in inventory first')
    if len({r['cvat_image'] for r in rows}) != len(rows):
        raise ValueError('Duplicate CVAT image name')
    for row in rows:
        if Path(row['cvat_image']).name != row['cvat_image'] or not Path(row['image']).is_file():
            raise ValueError(f"Invalid image path: {row['id']}")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    for row in rows:
        shutil.copy2(row['image'], output / row['cvat_image'])
    print(f'Copied {len(rows)} selected originals to {output}')


def import_cvat(args):
    rows = read_csv(args.inventory.resolve())
    selected = [r for r in rows if r.get('selected', '').lower() == 'yes']
    if not selected:
        raise ValueError('No selected rows; set selected=yes after review')
    if len({r['id'] for r in rows}) != len(rows) or len({r['cvat_image'] for r in rows}) != len(rows):
        raise ValueError('Duplicate inventory ID or CVAT image name')
    xml = ET.parse(args.xml).getroot()
    images = {}
    for im in xml.iter('image'):
        name = Path(im.get('name', '')).name
        if name in images:
            raise ValueError(f'Duplicate image name in CVAT XML: {name}')
        images[name] = im
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(f'Choose a new output directory: {out}')
    prepared = []
    for row in selected:
        sid = row['id']
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}', row.get('group_id', '')):
            raise ValueError(f'{sid}: group_id must be manually assigned, using ASCII letters, digits, _, . or -')
        if row.get('split') != 'train' or row.get('snow_free_reviewed', '').lower() != 'yes':
            raise ValueError(f'{sid}: requires split=train and snow_free_reviewed=yes')
        if not row.get('source', '').strip() or not row.get('license', '').strip():
            raise ValueError(f'{sid}: source and license are required')
        path = Path(row['image'])
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as im:
            size = oriented_size(im)
        if size != (int(row['width']), int(row['height'])):
            raise ValueError(f'{sid}: image dimensions changed')
        im = images.get(row['cvat_image'])
        if im is None:
            raise ValueError(f'{sid}: image absent from CVAT export')
        if (int(im.get('width')), int(im.get('height'))) != size:
            raise ValueError(f'{sid}: CVAT and image dimensions differ; check EXIF orientation')
        polys = []
        for poly in im.findall('polygon'):
            role = poly.get('label')
            if role not in {'roof', 'slope', 'structure', 'exclude', 'overlay'}:
                continue
            points = [[float(v) for v in pair.split(',')] for pair in poly.get('points', '').split(';')]
            if len(points) < 3 or any(len(p) != 2 or not all(math.isfinite(v) for v in p) or not (0 <= p[0] < size[0] and 0 <= p[1] < size[1]) for p in points):
                raise ValueError(f'{sid}: invalid {role} coordinates')
            pts = np.asarray(points, np.float32)
            if abs(cv2.contourArea(pts)) < 4:
                raise ValueError(f'{sid}: degenerate {role} polygon')
            if role == 'slope' and (len(points) != 4 or not cv2.isContourConvex(pts)):
                raise ValueError(f'{sid}: slope needs four ordered convex corners')
            polys.append({'role': role, 'points': points})
        if sum(p['role'] == 'roof' for p in polys) != 1 or sum(p['role'] == 'slope' for p in polys) != 1:
            raise ValueError(f'{sid}: annotate exactly one roof and one slope polygon')
        roof = np.array(next(p['points'] for p in polys if p['role'] == 'roof'), np.float32)
        slope = next(p['points'] for p in polys if p['role'] == 'slope')
        if any(cv2.pointPolygonTest(roof, tuple(pt), False) < 0 for pt in slope):
            raise ValueError(f'{sid}: slope corners must be inside the roof polygon')
        ref = {'object_id': row['group_id'], 'vantage_id': sid, 'image': path.name,
               'width': size[0], 'height': size[1], 'polygons': polys,
               'source': 'Manual CVAT annotations; roof geometry defaults to an uncalibrated prior'}
        prepared.append((row, ref))
    out.mkdir(parents=True)
    (out / 'references').mkdir()
    manifest = []
    for row, ref in prepared:
        ref_path = out / 'references' / (row['id'] + '.json')
        ref_path.write_text(json.dumps(ref, ensure_ascii=False, indent=2), encoding='utf-8')
        manifest.append(dict(id=row['id'], group_id=row['group_id'], image=row['image'], reference=str(ref_path),
                             split='train', source=row['source'], license=row['license'], snow_free_reviewed='yes'))
    write_csv(out / 'input.csv', ['id', 'group_id', 'image', 'reference', 'split', 'source', 'license', 'snow_free_reviewed'], manifest)
    print(f'Prepared {len(manifest)} manually reviewed roof photographs: {out / "input.csv"}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    init_p = sub.add_parser('init')
    init_p.add_argument('--root', type=Path, required=True)
    init_p.add_argument('--output', type=Path, required=True)
    init_p.add_argument('--karlsruhe', action='store_true', help='only Flug_100..105/RGB directories')
    init_p.add_argument('--stage', action=argparse.BooleanOptionalAction, default=False, help='copy all originals under unique names for CVAT')
    init_p.set_defaults(func=init)
    stage = sub.add_parser('stage-selected', help='copy only inventory rows with selected=yes')
    stage.add_argument('--inventory', type=Path, required=True)
    stage.add_argument('--output', type=Path, required=True)
    stage.set_defaults(func=stage_selected)
    imp = sub.add_parser('import-cvat')
    imp.add_argument('--inventory', type=Path, required=True)
    imp.add_argument('--xml', type=Path, required=True, help='CVAT for images 1.1 XML')
    imp.add_argument('--output', type=Path, required=True)
    imp.set_defaults(func=import_cvat)
    args = p.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
