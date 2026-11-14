#!/usr/bin/env python3
"""Build a resumable DRAFT snow-segmentation dataset from a folder, with no neural nets.

The detector's masks are heuristic *proposals*. Rendered samples remain UNREVIEWED;
never use this output as expert ground truth or as a validation/test dataset.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from roofcheck.roof_detection.classical import ClassicalRoofBackend
from roofcheck.roof_detection import build_detection, write_detection
from roofcheck.roof_detection.types import Selection, Detection, Roof, Facet

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp', '.bmp'}
FIELDS = ('sample_id', 'image', 'mask', 'roi', 'source_image', 'source_key',
          'group_id', 'roof_id', 'variant', 'scenario', 'target_coverage',
          'roof_confidence_heuristic', 'review_status', 'split', 'license',
          'original_width', 'original_height', 'work_width', 'work_height')


def file_key(root: Path, path: Path) -> str:
    rel = path.relative_to(root).as_posix()
    return hashlib.sha256(rel.encode('utf-8')).hexdigest()[:20]


def discover(root: Path, recursive: bool) -> list[Path]:
    files = root.rglob('*') if recursive else root.iterdir()
    return sorted(p for p in files if p.is_file() and p.suffix.lower() in EXTENSIONS)


def parse_float_list(value: str) -> list[float]:
    numbers = [float(s.strip()) for s in value.split(',') if s.strip()]
    if not numbers or any(not np.isfinite(x) or not 0 <= x <= 1 for x in numbers):
        raise argparse.ArgumentTypeError('coverages must be a nonempty comma-separated list within [0,1]')
    return numbers


def parse_scenarios(value: str) -> list[str]:
    supported = {'sparse_patches', 'scattered_patches', 'partial_cover',
                 'near_full_cover', 'edge_residue', 'cleared_irregular',
                 'thin_streaks', 'shadowed_snow'}
    values = [s.strip() for s in value.split(',') if s.strip()]
    if not values or any(x not in supported for x in values):
        raise argparse.ArgumentTypeError(f'scenarios must be from: {", ".join(sorted(supported))}')
    return values


def write_json(path: Path, payload: dict):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def mask_from(path: Path, shape: tuple[int,int]) -> np.ndarray:
    with Image.open(path) as im:
        arr = np.array(im)
    if arr.dtype != np.uint8 or arr.shape != shape or not set(np.unique(arr)) <= {0, 1, 255}:
        raise ValueError(f'Unexpected snow mask from renderer: {path}')
    return arr


def copy_sample(src_image: Path, src_mask: Path, src_roi: Path, dest: Path,
                row: dict, *, shape: tuple[int,int]):
    dest.mkdir(parents=True, exist_ok=False)
    dst_image, dst_mask, dst_roi = dest / 'image.png', dest / 'mask.png', dest / 'roi.png'
    if src_image.suffix.lower() == '.png':
        shutil.copy2(src_image, dst_image)
    else:
        with Image.open(src_image) as im:
            ImageOps.exif_transpose(im).convert('RGB').save(dst_image)
    mask_from(src_mask, shape)
    shutil.copy2(src_mask, dst_mask)
    shutil.copy2(src_roi, dst_roi)
    write_json(dest/'metadata.json', row)


def collect_rows(samples: Path) -> list[dict]:
    rows = []
    for file in sorted(samples.glob('*/records.json')):
        rows.extend(json.loads(file.read_text(encoding='utf-8')))
    return rows


def rebuild_manifest(output: Path):
    rows = collect_rows(output/'samples')
    tmp = output/'manifest.csv.tmp'
    with tmp.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows([{k:row.get(k,'') for k in FIELDS} for row in rows])
    tmp.replace(output/'manifest.csv')
    # CVAT-style human review manifest. Keep accepted/rejected state when resuming.
    review = output/'review.csv'
    previous = {}
    if review.exists():
        with review.open(newline='', encoding='utf-8') as f:
            previous = {r['sample_id']:r for r in csv.DictReader(f)}
    tmp = output/'review.csv.tmp'
    with tmp.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=['sample_id','decision','reviewer','note'])
        w.writeheader()
        for row in rows:
            old=previous.get(row['sample_id'],{})
            w.writerow({'sample_id':row['sample_id'],'decision':old.get('decision',''),
                        'reviewer':old.get('reviewer',''),'note':old.get('note','')})
    tmp.replace(review)
    return len(rows)


def prepare_work_image(source: Path, work: Path, max_side: int, jpeg_quality: int):
    """Return the exact RGB pixels that the renderer will later see.

    Resizing is opt-in and applies to *all* masks, geometry and snow output.
    The original file is never modified; manifest rows retain its path.
    """
    with Image.open(source) as im:
        oriented = ImageOps.exif_transpose(im).convert('RGB')
        original_size = oriented.size
        if max_side and max(original_size) > max_side:
            oriented.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            work.mkdir(parents=True, exist_ok=True)
            path = work/'working_source.jpg'
            # Re-open the JPEG: roof masks must match exactly the rendered RGB.
            oriented.save(path, quality=jpeg_quality, subsampling=0)
            with Image.open(path) as check:
                return np.array(check.convert('RGB')), path, original_size
        return np.array(oriented), source, original_size


def crop_detection(det: Detection, roof: Roof, bounds: tuple[int,int,int,int]):
    """Use selected roof only and move facet polygons into crop coordinates."""
    x0,y0,x1,y1 = bounds
    h,w = y1-y0,x1-x0
    facets = [Facet(f.facet_id, [[float(x)-x0, float(y)-y0] for x,y in f.polygon],
                    f.pixels, f.confidence, f.mode, f.facet_type) for f in roof.facets]
    selected = Roof(roof.roof_id, roof.mask[y0:y1,x0:x1].copy(), roof.confidence, facets)
    model_info=dict(det.model_info, crop_xyxy_processing=[x0,y0,x1,y1])
    return Detection([selected], Selection([roof.roof_id], 'selected',
        'Automatically proposed roof, human review required'),
        det.obstacle_mask[y0:y1,x0:x1].copy(),(h,w),model_info,det.diagnostics)


def process_photo(source: Path, src_root: Path, working: Path, staging: Path,
                  args, backend) -> tuple[dict, list[dict]]:
    photo_id = 'img_' + file_key(src_root,source)
    rel = source.relative_to(src_root).as_posix()
    image, work_source, original_size = prepare_work_image(
        source, working, args.work_max_side, args.work_jpeg_quality)
    if min(image.shape[:2]) < 64:
        raise ValueError('Image smaller than 64 pixels')
    processed_size = [image.shape[1], image.shape[0]]
    # min_area is specified in ORIGINAL-photo pixels. Scale it with the
    # processing resolution to avoid rejecting small roofs merely because
    # --work-max-side was enabled.
    area_ratio = image.shape[0]*image.shape[1] / (original_size[0]*original_size[1])
    effective_min_area = max(40, int(round(args.min_area*area_ratio)))
    detector = backend if backend.min_area == effective_min_area else replace(
        backend, min_area=effective_min_area)
    maps=detector.predict(image)
    det=build_detection(maps, min_roof_area=effective_min_area, instance_min_confidence=0.0)
    qualifying=sorted((r for r in det.roofs if r.confidence>=args.min_confidence),
                      key=lambda r:r.pixels, reverse=True)[:args.max_roofs]
    records=[]
    working.mkdir(parents=True,exist_ok=True)
    staging.mkdir(parents=True,exist_ok=True)
    for roof in qualifying:
        # Passing only the selected roof avoids repeated full-scene export of
        # every other candidate. Optional crop further reduces render pixels.
        roi_image, roi_source = image, work_source
        bounds = [0, 0, image.shape[1], image.shape[0]]
        if args.roof_crop_pad:
            x0,y0,x1,y1 = roof.bbox
            pad = args.roof_crop_pad
            bounds = [max(0,x0-pad), max(0,y0-pad),
                      min(image.shape[1],x1+pad), min(image.shape[0],y1+pad)]
            xa,ya,xb,yb=bounds
            path=working/(roof.roof_id+'_cropped.jpg')
            Image.fromarray(image[ya:yb,xa:xb]).save(
                path, quality=args.work_jpeg_quality, subsampling=0)
            with Image.open(path) as check:
                roi_image=np.array(check.convert('RGB'))
            roi_source=path
        selected = crop_detection(det,roof,tuple(bounds))
        selected.model_info.update(original_source=str(source.resolve()),
            original_image_size=list(original_size),working_image_size=processed_size)
        candidate=working/roof.roof_id
        sample_path=write_detection(roi_image,roi_source,selected,candidate,
            image_id=photo_id+'_'+roof.roof_id,group_id=photo_id,
            verified_clean=args.verified_clean,scenario=args.scenarios[0],
            target_coverage=args.coverages[0])
        if args.keep_proposals:
            # Retain the full CVAT proposal, all roof/facet ID maps and diagnostic overlay.
            shutil.copytree(candidate, staging/'proposals'/roof.roof_id)
        if not args.render:
            continue
        roi=candidate/'inspection_roi.png'
        for v in range(args.variants):
            scenario=args.scenarios[v%len(args.scenarios)]
            coverage=args.coverages[v%len(args.coverages)]
            # Avoid running full roof detection again for every variant.
            payload=json.loads(sample_path.read_text(encoding='utf-8'))
            payload['generation'].update(scenario=scenario,target_coverage=coverage)
            input_json=candidate/f'variant_{v:03}.json'
            write_json(input_json,payload)
            rendered=candidate/f'render_{v:03}'
            seed=args.seed+int(file_key(src_root,source),16)%1000000+v
            proc=subprocess.run([sys.executable,'-m','roofcheck.roof',str(input_json.resolve()),
                          '--output',str(rendered.resolve()),'--seed',str(seed)],
                          cwd=ROOT, capture_output=True,text=True,timeout=args.render_timeout)
            if proc.returncode:
                raise RuntimeError(f'render failed on {rel} {roof.roof_id} v{v}: {proc.stderr[-1400:]}')
            sample_id=f'{photo_id}_{roof.roof_id}_v{v:03}'
            dest=staging/roof.roof_id/f'v{v:03}'
            mask=mask_from(rendered/'mask.png',roi_image.shape[:2])
            row=dict(sample_id=sample_id,
                image=(Path('samples')/photo_id/roof.roof_id/f'v{v:03}'/'image.png').as_posix(),
                mask=(Path('samples')/photo_id/roof.roof_id/f'v{v:03}'/'mask.png').as_posix(),
                roi=(Path('samples')/photo_id/roof.roof_id/f'v{v:03}'/'roi.png').as_posix(),
                source_image=str(source.resolve()),source_key=rel,
                group_id=photo_id,roof_id=roof.roof_id,variant=v,
                scenario=scenario,target_coverage=coverage,
                roof_confidence_heuristic=round(roof.confidence,5),
                review_status='unreviewed',split='train_candidate',license=args.license,
                original_width=original_size[0],original_height=original_size[1],
                work_width=roi_image.shape[1],work_height=roi_image.shape[0])
            copy_sample(rendered/'image.png',rendered/'mask.png',roi,dest,row,shape=roi_image.shape[:2])
            records.append(row)
        if args.include_clean_controls:
            sample_id=f'{photo_id}_{roof.roof_id}_clean'
            roof_mask=np.asarray(Image.open(candidate/'roof_mask.png'))>0
            occluder=np.asarray(Image.open(candidate/'occluder_mask.png'))>0
            clean_mask=np.full(roof_mask.shape,255,np.uint8)
            clean_mask[roof_mask & ~occluder] = 0
            clean_path=working/roof.roof_id/'clean_mask.png'
            Image.fromarray(clean_mask).save(clean_path)
            row=dict(sample_id=sample_id,
                image=(Path('samples')/photo_id/roof.roof_id/'clean'/'image.png').as_posix(),
                mask=(Path('samples')/photo_id/roof.roof_id/'clean'/'mask.png').as_posix(),
                roi=(Path('samples')/photo_id/roof.roof_id/'clean'/'roi.png').as_posix(),
                source_image=str(source.resolve()),source_key=rel,
                group_id=photo_id,roof_id=roof.roof_id,variant='clean',
                scenario='clean_control',target_coverage=0,
                roof_confidence_heuristic=round(roof.confidence,5),
                review_status='unreviewed',split='train_candidate',license=args.license,
                original_width=original_size[0],original_height=original_size[1],
                work_width=roi_image.shape[1],work_height=roi_image.shape[0])
            copy_sample(candidate/('source'+roi_source.suffix.lower()),clean_path,roi,
                        staging/roof.roof_id/'clean',row,shape=roi_image.shape[:2])
            records.append(row)
    write_json(staging/'records.json',records)
    return {'source':rel,'photo_id':photo_id,'candidates':len(det.roofs),
            'selected_roofs':len(qualifying),'samples':len(records),
            'original_size':list(original_size),'processing_size':processed_size,
            'crop_padding_px':args.roof_crop_pad,
            'effective_min_area':effective_min_area},records


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True,help='folder with input photographs')
    p.add_argument('--output',type=Path,required=True,help='NEW output directory; can resume with --resume')
    p.add_argument('--no-recursive',action='store_true')
    p.add_argument('--max-images',type=int,default=0,help='0 means all, useful for a smoke test')
    p.add_argument('--max-roofs',type=int,default=6)
    p.add_argument('--max-side',type=int,default=1500,
                   help='internal detector resolution; cannot exceed working image size')
    p.add_argument('--work-max-side',type=int,default=0,
                   help='resize photograph BEFORE detection and rendering; 0=original, 1024=fast')
    p.add_argument('--roof-crop-pad',type=int,default=0,
                   help='render a crop around EACH roof with this pixel padding; 0=full working image')
    p.add_argument('--work-jpeg-quality',type=int,default=90,
                   help='JPEG quality for resized/cropped working images (1..100)')
    p.add_argument('--min-area',type=int,default=850)
    p.add_argument('--min-confidence',type=float,default=.68,help='heuristic score, NOT a probability')
    p.add_argument('--include-neutral',action='store_true',help='EXPERIMENTAL; many false positives')
    p.add_argument('--no-ridges',action='store_true')
    p.add_argument('--verified-clean',action='store_true',help='operator confirms every image is snow-free')
    p.add_argument('--render',action='store_true',help='generate synthetic snow images+mask with roofcheck.roof')
    p.add_argument('--variants',type=int,default=4,help='render variants per detected roof')
    p.add_argument('--coverages',type=parse_float_list,default=[.08,.35,.65,.92])
    p.add_argument('--scenarios',type=parse_scenarios,default=['sparse_patches','partial_cover','cleared_irregular','near_full_cover'])
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--render-timeout',type=int,default=300,help='seconds per rendered sample')
    p.add_argument('--include-clean-controls',action='store_true',help='one original clean example per roof')
    p.add_argument('--keep-proposals',action='store_true',help='retain masks+CVAT suggestions; consumes more disk')
    p.add_argument('--license',default='unknown',help='provenance label; you are responsible for input rights')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--dry-run',action='store_true',help='count images without writing output')
    args=p.parse_args(argv)
    src=args.input.expanduser().resolve()
    out=args.output.expanduser().resolve()
    if not src.is_dir():p.error(f'Input directory does not exist: {src}')
    if out==src or src in out.parents:
        p.error('Output must be OUTSIDE input to avoid recursively processing generated images')
    if (args.max_images<0 or args.max_roofs<1 or args.variants<1 or
            args.render_timeout<1 or args.min_area<40 or
            args.work_max_side<0 or (args.work_max_side and args.work_max_side<256) or
            args.roof_crop_pad<0 or not 1<=args.work_jpeg_quality<=100):
        p.error('Invalid max-images, max-roofs, variants, timeout or min-area')
    if not 0<=args.min_confidence<=1:p.error('min-confidence must be in [0,1]')
    if not args.render:
        args.keep_proposals = True  # proposals-only mode must retain useful outputs
    if args.render and not args.verified_clean:
        p.error('--render requires --verified-clean: never infer that snow-free source photos are clean')
    if args.include_clean_controls and not (args.render and args.verified_clean):
        p.error('Clean controls require --render --verified-clean')
    images=discover(src,recursive=not args.no_recursive)
    if args.max_images:images=images[:args.max_images]
    if not images:p.error('No images found')
    print(f'Found {len(images)} image(s) under {src}',flush=True)
    if args.dry_run:return 0
    # Lock the creation-time settings for safe --resume.
    config=dict(input=str(src),recursive=not args.no_recursive,
        max_roofs=args.max_roofs,max_side=args.max_side,min_area=args.min_area,
        work_max_side=args.work_max_side,roof_crop_pad=args.roof_crop_pad,
        work_jpeg_quality=args.work_jpeg_quality,
        min_confidence=args.min_confidence,include_neutral=args.include_neutral,
        ridges=not args.no_ridges,verified_clean=args.verified_clean,
        render=args.render,variants=args.variants,coverages=args.coverages,
        scenarios=args.scenarios,seed=args.seed,clean=args.include_clean_controls,
        keep_proposals=args.keep_proposals,license=args.license)
    if args.resume:
        if not (out/'config.json').exists():p.error('Nothing to resume: config.json not found')
        if json.loads((out/'config.json').read_text(encoding='utf-8'))!=config:
            p.error('Settings differ from original run; use the same settings or a new output directory')
    else:
        if out.exists():p.error(f'Output already exists: {out}; use --resume or another path')
        out.mkdir(parents=True)
        write_json(out/'config.json',config)
    (out/'samples').mkdir(exist_ok=True)
    (out/'.working').mkdir(exist_ok=True)
    backend=ClassicalRoofBackend(max_side=args.max_side,min_area=args.min_area,
                 include_neutral=args.include_neutral,detect_ridges=not args.no_ridges)
    attempts=0;success=0;failed=0;skipped=0
    with (out/'progress.jsonl').open('a',encoding='utf-8') as log:
        for idx,source in enumerate(images,1):
            key='img_'+file_key(src,source)
            dest=out/'samples'/key
            if dest.exists() and (dest/'records.json').is_file():
                skipped+=1
                continue
            attempts+=1
            work=out/'.working'/key
            staged=out/'samples'/('.tmp-'+key)
            shutil.rmtree(work,ignore_errors=True)
            shutil.rmtree(staged,ignore_errors=True)
            try:
                stats,_=process_photo(source,src,work,staged,args,backend)
                staged.rename(dest)
                success+=1
                entry={'status':'done',**stats}
                print(f'[{idx}/{len(images)}] {stats["source"]}: {stats["selected_roofs"]} roofs, {stats["samples"]} samples',flush=True)
            except Exception as exc:
                failed+=1
                entry={'status':'error','source':source.relative_to(src).as_posix(),
                       'photo_id':key,'error':str(exc)[:1600]}
                print(f'[{idx}/{len(images)}] ERROR {entry["source"]}: {exc}',file=sys.stderr,flush=True)
                shutil.rmtree(staged,ignore_errors=True)
            finally:
                shutil.rmtree(work,ignore_errors=True)
            log.write(json.dumps(entry,ensure_ascii=False)+'\n')
            log.flush()
            # Periodic manifest checkpoint allows consuming intermediate results.
            if attempts%10==0:rebuild_manifest(out)
    total=rebuild_manifest(out)
    report={'found':len(images),'processed_ok_this_run':success,'errors_this_run':failed,
            'skipped_existing':skipped,'draft_samples_total':total,
            'split':'train_candidate','review_status':'unreviewed',
            'warning':'Automated OpenCV proposals are NOT independently verified ground truth.'}
    write_json(out/'summary.json',report)
    print(f'Done. {total} DRAFT records: {out/"manifest.csv"}, QA: {out/"review.csv"}',flush=True)
    return 0 if failed==0 else 1


if __name__=='__main__':
    raise SystemExit(main())
