"""Run the existing depth-first roof renderer on an image aligned to a reference."""
import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from roofcheck import hostmatch, overlays, synth
from roofcheck.local_composite import ignore_low_change_snow, local_composite
from roofcheck.schema import Reference


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image',type=Path,required=True,help='snow-free photo aligned to reference coordinates')
    p.add_argument('--reference',type=Path,required=True,help='reference.json with roof, slope and structure polygons')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--state',choices=synth.STATES,default='partially_cleared')
    p.add_argument('--ground-snow-mm',type=float,help='optional simulated ground snow depth in mm; not a measured roof depth')
    p.add_argument('--save-depth',action='store_true',help='save experimental float32 depth map in millimetres')
    p.add_argument('--partial-style',choices=('legacy','soft_front'),default='legacy',help='shape of partially cleared snow')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--scale',type=float,default=1.)
    p.add_argument('--texture-dir',type=Path,help='optional existing CC0 texture library root')
    p.add_argument('--texture-policy',choices=('legacy','isolated'),default='legacy',help='isolated keeps geometry and labels independent of material texture')
    p.add_argument('--texture-seed',type=int,help='isolated material seed; defaults to a stable derivation from --seed')
    p.add_argument('--texture-strength',type=float,default=0.02,help='isolated material contrast, from 0 to 0.25')
    p.add_argument('--snow-brightness',type=float,default=1.0,help='snow-layer brightness multiplier: 0.7..1.4')
    p.add_argument('--snow-detail',type=float,default=1.0,help='snow high-frequency detail: 0..1')
    p.add_argument('--context',choices=('scene','local','local-volume'),default='scene',help='local-volume also retains rendered snow outside the ROI')
    p.add_argument('--feather-px',type=int,default=4,help='local blending width in output pixels; transition labels become ignore')
    p.add_argument('--fence-mode',choices=('auto','none'),default='auto',help='none only if no fence/snow guard is present')
    p.add_argument('--structure-mode',choices=('auto','manual'),default='auto',help='manual uses only explicitly drawn structure polygons')
    p.add_argument('--visibility-threshold',type=int,help='optional RGB-change threshold (0..255); low-change snow becomes ignore, local mode only')
    args=p.parse_args()
    if not 0 < args.scale <= 1:raise ValueError('scale must be >0 and <=1')
    if not 0 <= args.feather_px <= 100:raise ValueError('feather-px must be 0..100')
    if not 0 <= args.texture_strength <= 0.25:raise ValueError('texture-strength must be 0..0.25')
    if not 0.7 <= args.snow_brightness <= 1.4 or not 0 <= args.snow_detail <= 1:raise ValueError('invalid snow brightness or detail')
    if args.texture_seed is not None and not 0 <= args.texture_seed < 2**64:raise ValueError('texture-seed must be a nonnegative 64-bit integer')
    if args.visibility_threshold is not None and (args.context not in ('local','local-volume') or not 0<=args.visibility_threshold<=255):
        raise ValueError('visibility-threshold requires local context and must be 0..255')
    if args.output.exists():raise FileExistsError(args.output)
    ref=Reference.load(args.reference)
    with Image.open(args.image) as im:rgb=np.array(ImageOps.exif_transpose(im).convert('RGB'))
    if rgb.shape[:2]!=(ref.height,ref.width):
        raise ValueError('Image dimensions must match reference; register other photos first')
    if len(ref.by_role('slope'))!=1 or len(ref.by_role('roof'))!=1:
        raise ValueError('Reference must contain one target roof and one four-point slope')
    def polys(role):return [np.asarray(v.points,np.float32)*args.scale for v in ref.by_role(role)]
    q=polys('slope')[0];roof=polys('roof')[0]
    if q.shape!=(4,2):raise ValueError('Slope must contain four ordered points')
    img=cv2.resize(rgb[:,:,::-1].copy(),None,fx=args.scale,fy=args.scale,interpolation=cv2.INTER_AREA) if args.scale!=1 else rgb[:,:,::-1].copy()
    if args.texture_dir and not args.texture_dir.is_dir():raise FileNotFoundError(args.texture_dir)
    # Never pick up a machine-specific texture bank implicitly.
    synth.TEXTURE_DIRS=[str(args.texture_dir)] if args.texture_dir else []
    synth._TEX=None
    texture_files=synth.texture_paths(synth.TEXTURE_DIRS)
    if args.texture_dir and not texture_files:raise ValueError('No supported texture files in texture-dir')
    if any(cv2.imread(str(f),cv2.IMREAD_GRAYSCALE) is None for f in texture_files):
        raise ValueError('A texture cannot be read')
    texture_hashes={str(Path(f).resolve().relative_to(args.texture_dir.resolve())):
                    hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in texture_files} if args.texture_dir else {}
    bbox=(q[:,0].min(),q[:,1].min(),q[:,0].max(),q[:,1].max())
    host=hostmatch.estimate(img,bbox,ignore=overlays.detect(img).mask)
    geom=synth.roof_geometry(ref.object_id,ref.geometry or None)
    texture_seed=args.texture_seed if args.texture_seed is not None else int.from_bytes(hashlib.sha256(f'roofcheck-material-v1:{args.seed}'.encode()).digest()[:8],'big')
    result=synth.make_sample(img,q,roof,np.random.default_rng(args.seed),geom,host=host,
        structures=polys('structure'),other_roofs=[],state=args.state,scene=None,fence_mode=args.fence_mode,structure_mode=args.structure_mode,
        texture_policy=args.texture_policy,texture_seed=texture_seed,
        apply_material=args.texture_policy=='isolated' and bool(args.texture_dir),texture_strength=args.texture_strength,
        partial_style=args.partial_style,snow_brightness=args.snow_brightness,snow_detail=args.snow_detail,ground_snow_mm=args.ground_snow_mm)
    # Legacy depth-based labels are useful for diagnostics only: outside the
    # explicitly annotated ROI and on ice/fences/structures they are unknown.
    label=synth.labels_at(result.depth_mm,result.classes,2)
    label[~result.roi]=255
    label[np.isin(result.classes,[synth.CLS['ice'],synth.CLS['structure'],synth.CLS['fence']])]=255
    protected=np.zeros(label.shape,np.uint8)
    for poly in polys('exclude') + polys('overlay') + polys('structure'):
        cv2.fillPoly(protected,[np.rint(poly).astype(np.int32)],1)
    label[protected>0]=255
    rendered=result.image
    alpha=None
    if args.context in ('local','local-volume'):
        protected |= (overlays.detect(img).mask>0).astype(np.uint8)
        visible_region = result.roi if args.context=='local' else (result.roi | (result.depth_mm > 0))
        rendered,label,alpha=local_composite(img,result.image,label,visible_region,protected,args.feather_px)
    visibility_ignored=0
    if args.visibility_threshold is not None:
        label,visibility_ignored=ignore_low_change_snow(img,rendered,label,args.visibility_threshold)
    if not set(np.unique(label)) <= {0,1,255}:raise ValueError('Unexpected label values')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    tmp=Path(tempfile.mkdtemp(prefix='.existing-render-',dir=args.output.parent))
    try:
        Image.fromarray(rendered[:,:,::-1]).save(tmp/'image.png')
        Image.fromarray(label).save(tmp/'snow_threshold_2mm_experimental.png')
        Image.fromarray(result.roi.astype(np.uint8)).save(tmp/'roi.png')
        if args.save_depth:np.save(tmp/'depth_mm_experimental.npy',result.depth_mm.astype(np.float32))
        metadata=dict(renderer='roofcheck.synth.make_sample',scene=None,render=result.info.get('render'),
            input_sha256=hashlib.sha256(args.image.read_bytes()).hexdigest(),
            reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),seed=args.seed,scale=args.scale,
            geometry=geom,info=result.info,context_mode=args.context,
            feather_px=args.feather_px if args.context in ('local','local-volume') else None,texture_sha256=texture_hashes,
            texture_policy=args.texture_policy,texture_seed=texture_seed if args.texture_policy=='isolated' else None,
            texture_strength=args.texture_strength if args.texture_policy=='isolated' else None,
            snow_brightness=args.snow_brightness,snow_detail=args.snow_detail,
            geometry_arrays_sha256={name:hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
                for name,a in [('depth_mm',result.depth_mm),('classes',result.classes),('roi',result.roi)]},
            transition_ignore_pixels=int(((alpha>0)&(alpha<1)).sum()) if alpha is not None else 0,
            visibility_threshold=args.visibility_threshold,visibility_ignore_pixels=visibility_ignored,
            caveats=['Planar fallback; no calibrated 3D scene','Assumes input image is aligned to reference and snow-free',
                     'Snow thickness uses uncalibrated generator priors; masks are experimental, not expert ground truth',
                     'Local mode removes overhangs outside ROI; local-volume retains rendered snow outside ROI but labels it ignore',
                     'The planar height-field extrusion and roof dimensions are uncalibrated approximations'])
        (tmp/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2,default=float))
        tmp.rename(args.output)
    except Exception:
        shutil.rmtree(tmp,ignore_errors=True)
        raise
    print('Output:',args.output,'| render:',result.info.get('render'))


if __name__=='__main__':main()
