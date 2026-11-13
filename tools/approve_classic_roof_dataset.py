#!/usr/bin/env python3
"""Export only manually accepted classical roof composites to a self-contained TRAIN set.

No automatic acceptance is supported. Train/val/test splitting requires real,
held-out images and human labels; this tool creates TRAIN inputs only.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def read_csv(path):
    with path.open(newline='',encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True,help='output of build_classic_roof_dataset.py')
    p.add_argument('--output',type=Path,required=True,help='new destination for approved training set')
    args=p.parse_args(argv)
    root=args.run.resolve();out=args.output.resolve()
    if root==out or root in out.parents:p.error('Output must be outside the unreviewed run folder')
    manifest=root/'manifest.csv';review=root/'review.csv'
    if not manifest.is_file() or not review.is_file():p.error('manifest.csv/review.csv missing')
    rows=read_csv(manifest)
    decisions={}
    for row in read_csv(review):
        ident=row.get('sample_id','')
        if not ident or ident in decisions:p.error('Duplicate or missing sample_id in review.csv')
        if row.get('decision','').strip().lower() not in {'','accept','reject'}:
            p.error('Only decision=accept or reject is allowed')
        decisions[ident]=row
    accepted=[]
    for row in rows:
        rev=decisions.get(row['sample_id'])
        if rev and rev['decision'].strip().lower()=='accept':
            if not rev.get('reviewer','').strip():p.error(f'Reviewer required: {row["sample_id"]}')
            accepted.append((row,rev))
    if not accepted:p.error('No accepted samples: review image/mask, fill review.csv with accept and reviewer')
    if out.exists():p.error('Output already exists; refuse to overwrite')
    # Validate every sample before creating the destination (no partial approved set).
    for row,_ in accepted:
        paths=[]
        for key in ('image','mask','roi'):
            candidate=(root/row[key]).resolve()
            if root not in candidate.parents or not candidate.is_file():p.error(f'Unsafe/missing {key}: {row[key]}')
            paths.append(candidate)
        with Image.open(paths[0]) as im:shape=(im.height,im.width)
        with Image.open(paths[1]) as im:mask=np.asarray(im)
        with Image.open(paths[2]) as im:roi=np.asarray(im)
        if mask.shape!=shape or mask.dtype!=np.uint8 or not set(np.unique(mask))<={0,1,255}:
            p.error(f'Invalid 0/1/255 mask for {row["sample_id"]}')
        if roi.shape!=shape or roi.ndim!=2:p.error(f'Bad ROI for {row["sample_id"]}')
    (out/'train/images').mkdir(parents=True)
    (out/'train/masks').mkdir(parents=True)
    (out/'train/roi').mkdir(parents=True)
    exported=[]
    for row,review in accepted:
        sid=row['sample_id']
        if not sid.replace('_','').replace('-','').isalnum():
            raise ValueError('Unsafe sample_id')
        destinations={key:Path('train')/folder/f'{sid}.png' for key,folder in
            (('image','images'),('mask','masks'),('roi','roi'))}
        for key,rel in destinations.items():
            shutil.copy2(root/row[key],out/rel)
        exported.append(dict(row,image=destinations['image'].as_posix(),
                         mask=destinations['mask'].as_posix(),roi=destinations['roi'].as_posix(),
                         review_status='accepted',split='train',reviewer=review['reviewer'],note=review.get('note','')))
    fields=list(rows[0].keys()) if rows else []
    fields.extend(['reviewer','note'])
    with (out/'manifest.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader();w.writerows(exported)
    (out/'dataset_card.json').write_text(json.dumps({
        'samples':len(exported),'split':'train_only',
        'type':'human-reviewed synthetic candidates',
        'source_run':str(root),
        'limitations':'Classical single-image detector has not been independently validated; real val/test required.'
    },indent=2,ensure_ascii=False),encoding='utf-8')
    print(f'Exported {len(exported)} manually accepted training samples to {out}')
    return 0


if __name__=='__main__':raise SystemExit(main())
