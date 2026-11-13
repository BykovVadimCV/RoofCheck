#!/usr/bin/env python3
"""Weight-free automatic roof proposals + RoofCheck masks, one sample per candidate.

Works without PyTorch, network access, or neural model weights. As with every
single-image colour/geometry heuristic, generated polygons require validation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image, ImageOps

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from roofcheck.roof_detection.classical import ClassicalRoofBackend
from roofcheck.roof_detection import build_detection, select, write_detection
from roofcheck.roof_detection.types import Selection


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-side',type=int,default=1500)
    p.add_argument('--min-area',type=int,default=850,help='minimum candidate pixels in full-resolution photo')
    p.add_argument('--min-confidence',type=float,default=.68,
                   help='heuristic rank floor for auto-export (NOT a probability)')
    p.add_argument('--max-roofs',type=int,default=30)
    p.add_argument('--include-neutral',action='store_true',help='riskier closed-quadrilateral neutral proposals')
    p.add_argument('--no-ridges',action='store_true',help='disable speculative facet-line detection')
    selection=p.add_mutually_exclusive_group()
    selection.add_argument('--target-point',type=int,nargs=2,metavar=('X','Y'))
    selection.add_argument('--target-bbox',type=int,nargs=4,metavar=('X1','Y1','X2','Y2'))
    selection.add_argument('--reference-mask',type=Path,help='EXIF-oriented binary known target roof')
    selection.add_argument('--all',action='store_true',help='explicitly make a single sample of ALL roofs')
    p.add_argument('--image-id',default='frame_001')
    p.add_argument('--group-id',default='scene_001',help='same group for all candidates of one photo')
    p.add_argument('--verified-clean',action='store_true',
                   help='human assertion that the source roof has NO snow; detector cannot verify this')
    p.add_argument('--snow-mask',type=Path,help='expert-labelled existing snow 0/1/255')
    p.add_argument('--scenario',default='partial_cover')
    p.add_argument('--coverage',type=float,default=.45)
    p.add_argument('--render',action='store_true',help='also call installed roofcheck.roof generator')
    args=p.parse_args(argv)

    if not 0 <= args.min_confidence <= 1 or args.max_roofs<1 or args.min_area<40:
        p.error('Invalid confidence, max-roofs or min-area')
    if args.verified_clean and args.snow_mask:
        p.error('Choose --verified-clean OR --snow-mask, not both')
    if args.render and not (args.verified_clean or args.snow_mask):
        p.error('Cannot render unknown snow-free status; pass --verified-clean or expert --snow-mask')
    if not args.image.is_file():
        p.error(f'Image not found: {args.image}')
    with Image.open(args.image) as f:
        image=np.array(ImageOps.exif_transpose(f).convert('RGB'))

    backend=ClassicalRoofBackend(max_side=args.max_side,min_area=args.min_area,
                include_neutral=args.include_neutral,detect_ridges=not args.no_ridges)
    maps=backend.predict(image)
    det=build_detection(maps,min_roof_area=args.min_area,instance_min_confidence=0)
    root=args.output.resolve()
    if root.exists():
        raise FileExistsError(f'Output exists: {root}')
    root.mkdir(parents=True)
    records=[]
    auto_each=not any((args.target_point,args.target_bbox,args.reference_mask,args.all))
    if auto_each:
        roofs=sorted((r for r in det.roofs if r.confidence>=args.min_confidence),
                      key=lambda r:r.pixels,reverse=True)[:args.max_roofs]
        requests=[([r.roof_id],r.roof_id) for r in roofs]
    else:
        ref=None
        if args.reference_mask:
            with Image.open(args.reference_mask) as f:
                ref=np.asarray(f)
            if ref.ndim!=2 or not set(np.unique(ref))<={0,1,255}:
                raise ValueError('Reference mask must be single-channel 0/1/255')
            ref=ref>0
        det.selection=select(det.roofs,point=args.target_point,bbox=args.target_bbox,
                             reference=ref,all_roofs=args.all)
        requests=[(det.selection.roof_ids,'selected')] if det.selection.status=='selected' else []
    for ids,name in requests:
        det.selection=Selection(ids,'selected',
            'Classical automatic proposal: unreviewed, not a ground-truth roof')
        folder=root/name
        sample=write_detection(image,args.image,det,folder,
                   image_id=f'{args.image_id}_{name}',group_id=args.group_id,
                   verified_clean=args.verified_clean,snow_mask=args.snow_mask,
                   scenario=args.scenario,target_coverage=args.coverage)
        records.append(dict(roof_ids=ids,sample=str(sample.relative_to(root)),
                            folder=name,pixels=sum(r.pixels for r in det.roofs if r.roof_id in ids)))
        if args.render:
            subprocess.run([sys.executable,'-m','roofcheck.roof',str(sample.resolve()),
                            '--output',str((folder/'rendered').resolve())],
                            check=True,cwd=ROOT)
    summary=dict(backend='classical-opencv',weights='none',image=str(args.image.resolve()),
                 detected=len(det.roofs),exported=len(records),samples=records,
                 note='UNREVIEWED proposals; NOT safe as expert training labels',
                 limitations=maps.model_info['limitations'])
    (root/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    print(f'OpenCV: {len(det.roofs)} proposals, exported {len(records)} candidate(s); {root}')
    if not records:
        print('No qualifying candidates. Try --min-confidence 0.4, not an arbitrary roof selection.',file=sys.stderr)
        return 2
    return 0


if __name__=='__main__':
    raise SystemExit(main())
