#!/usr/bin/env python3
"""Export manually checked snow-free roof photos as real negative segmentation samples."""
import argparse
import csv
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.batch_existing import read_input
from roofcheck.schema import Reference


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--scale', type=float, default=1)
    p.add_argument('--boundary-px', type=int, default=4, help='ignore this many output pixels at the ROI and occlusion boundaries')
    args = p.parse_args()
    if not 0 < args.scale <= 1 or not 0 <= args.boundary_px <= 100:
        p.error('scale must be >0 and <=1; boundary-px must be 0..100')
    rows = read_input(args.input.resolve())
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    (out / 'samples').mkdir()
    result = []
    for row in rows:
        ref = Reference.load(row['reference'])
        with Image.open(row['image']) as im:
            rgb = np.asarray(ImageOps.exif_transpose(im).convert('RGB'))
        if (rgb.shape[1], rgb.shape[0]) != (ref.width, ref.height):
            raise ValueError(f"{row['id']}: reference dimensions differ")
        if len(ref.by_role('roof')) != 1 or len(ref.by_role('slope')) != 1:
            raise ValueError(f"{row['id']}: expected one roof and one slope")
        if args.scale != 1:
            rgb = cv2.resize(rgb, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA)
        shape = rgb.shape[:2]
        def raster(roles):
            arr = np.zeros(shape, dtype=np.uint8)
            for role in roles:
                for poly in ref.by_role(role):
                    pts = np.rint(np.asarray(poly.points) * args.scale).astype(np.int32)
                    if len(pts) < 3: raise ValueError(f"{row['id']}: invalid polygon")
                    cv2.fillPoly(arr, [pts], 1)
            return arr
        area = raster(['roof']) & raster(['slope'])
        forbidden = raster(['structure', 'exclude', 'overlay'])
        if args.boundary_px:
            n = 2 * args.boundary_px + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))
            area = cv2.erode(area, kernel)
            forbidden = cv2.dilate(forbidden, kernel)
        roi = (area & (1 - forbidden)).astype('uint8')
        if not roi.any():
            raise ValueError(f"{row['id']}: no valid roof pixels")
        mask = np.where(roi, 0, 255).astype('uint8')
        sid = row['id'] + '__real_clean'
        folder = out / 'samples' / sid
        folder.mkdir()
        Image.fromarray(rgb).save(folder / 'image.png')
        Image.fromarray(mask).save(folder / 'mask.png')
        Image.fromarray(roi).save(folder / 'roi.png')
        (folder / 'provenance.json').write_text(json.dumps({'source_id': row['id'], 'image_sha256': row['image_sha256'], 'reference_sha256': row['reference_sha256'], 'source': row['source'], 'license': row['license'], 'snow_free_reviewed': row['snow_free_reviewed'], 'boundary_px': args.boundary_px}, ensure_ascii=False, indent=2))
        base = f'samples/{sid}'
        result.append({'id': sid, 'image': f'{base}/image.png', 'mask': f'{base}/mask.png', 'group_id': row['group_id'], 'split': 'train', 'kind': 'real', 'roi': f'{base}/roi.png'})
    with (out / 'real_clean_manifest.csv').open('x', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['id', 'image', 'mask', 'group_id', 'split', 'kind', 'roi'])
        w.writeheader(); w.writerows(result)
    print(f'Exported {len(result)} reviewed clean roof negatives: {out}')


if __name__ == '__main__':
    main()
