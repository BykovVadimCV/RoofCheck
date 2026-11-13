"""Single-sample generation entry point. Batch/export systems can call this contract."""
from dataclasses import asdict, dataclass, field
from pathlib import Path
import hashlib
import json
import shutil
import tempfile
import numpy as np
from PIL import Image
from .inputs import RoofSample, load_sample
from .geometry import surface_fields
from .generator import GenerationConfig, generate_alpha
from .renderer import render
from .masks import coverage, merge_snow, save_mask, validate_mask
from .scenarios import scenario_field
from .volume import build_volume
from .textures import projected_material

VERSION = '0.4.0'


@dataclass
class RoofResult:
    image: np.ndarray
    mask: np.ndarray
    alpha: np.ndarray
    metadata: dict
    diagnostics: dict = field(default_factory=dict)


def generate(spec: RoofSample, config: GenerationConfig | None = None, seed: int = 0):
    if spec.split != 'train':
        raise ValueError('Synthetic derivatives are allowed only for train')
    config = config or GenerationConfig()
    config.validate()
    loaded = load_sample(spec)
    if config.texture_path and config.render_mode != 'volume':
        raise ValueError('Real textures require render_mode=volume')
    rng = np.random.default_rng(seed)
    base, texture, owner, geom = surface_fields(loaded,rng)
    indices=np.flatnonzero(loaded.allowed)
    count=len(indices)
    if count:
        score=scenario_field(config.scenario,base,loaded,rng,owner,geom)
        order=indices[np.argsort(score.ravel()[indices],kind='stable')]
    else:order=indices
    target=config.validate()
    lo,hi=0,count
    k=round(target*count)
    best=None
    steps=0
    # Reuse the same ranked field and relief seed at every trial. Only the footprint budget changes.
    for _ in range(16 if config.render_mode=='volume' else 1):
        alpha,metrics=generate_alpha(loaded,base,config,rng,order,k)
        geometry=build_volume(loaded,alpha,texture,config,owner,seed) if config.render_mode=='volume' else None
        trial=geometry.visibility if geometry is not None else alpha
        fraction=float(((trial>=config.alpha_high)&loaded.allowed).sum()/count) if count else 0.
        error=abs(fraction-target);steps+=1
        if best is None or error<best[0]:best=(error,alpha,metrics,geometry)
        if error<=min(.0005,config.coverage_tolerance) or lo>=hi:break
        if fraction>target:hi=k-1
        else:lo=k+1
        if lo>hi:break
        k=(lo+hi)//2
    _,alpha,metrics,geometry=best
    metrics['coverage_calibration_trials']=steps
    real_texture,texture_info=None,None
    if config.texture_path:
        real_texture,texture_info=projected_material(config.texture_path,loaded.roof.shape,owner,geom,seed,config.texture_tile_px)
    image,visible,diagnostics=render(loaded,alpha,texture,config,geometry,owner,seed,return_diagnostics=True,real_texture=real_texture)
    if texture_info is not None:diagnostics['real_texture']=texture_info
    mask = merge_snow(loaded.snow,visible,loaded.allowed,config.alpha_low,config.alpha_high)
    validate_mask(mask,loaded.roof.shape)
    if not (mask != 255).any():
        raise ValueError('No known pixels in the final training mask')
    added = (visible >= config.alpha_high) & loaded.allowed
    denom = metrics['eligible_pixels']
    actual = float(added.sum()/denom) if denom else 0.
    if abs(actual-metrics['target_coverage']) > config.coverage_tolerance:
        raise ValueError('Final visible synthetic coverage outside tolerance (including pixel quantization)')
    if not np.array_equal(image[~loaded.allowed],loaded.image[~loaded.allowed]):
        raise RuntimeError('Protected pixels changed')
    if not np.array_equal(mask[~loaded.allowed],loaded.snow[~loaded.allowed]):
        raise RuntimeError('Protected labels changed')
    surface = loaded.roof & ~loaded.occluders
    metrics.update(final_visible_coverage=actual,
                   roof_total_snow=coverage(mask,surface),inspection_total_snow=coverage(mask,loaded.roi & ~loaded.occluders),
                   added_snow_pixels=int(added.sum()),
                   added_snow_in_inspection_pixels=int((added & loaded.roi).sum()))
    hashes = {}
    for key in ('image_path','roof_mask','snow_mask','inspection_roi','occluder_mask'):
        path = getattr(spec,key)
        if path:
            hashes[key] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    code_hashes = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    rendering={k:v for k,v in diagnostics.items() if not isinstance(v,np.ndarray)}
    return RoofResult(image,mask,visible,dict(version=VERSION,seed=seed,sample=asdict(spec),config=asdict(config),
        geometry=geom,coverage=metrics,input_sha256=hashes,implementation_sha256=code_hashes,
        rendering=rendering,
        limitations=['Approximate appearance; visual pixel height is not physical snow depth','Occlusion depends on provided annotation']),diagnostics)


def save_result(spec, result, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    output.parent.mkdir(parents=True,exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix='.roof-',dir=output.parent))
    try:
        Image.fromarray(result.image).save(temp/'image.png')
        save_mask(temp/'mask.png',result.mask)
        Image.fromarray(np.round(result.alpha*65535).astype(np.uint16)).save(temp/'alpha.png')
        if result.diagnostics:
            np.save(temp/'height_visual_px.npy',result.diagnostics['height_visual_px'])
            Image.fromarray(np.round(np.clip(result.diagnostics['height_visual_px']*1000,0,65535)).astype(np.uint16)).save(temp/'height_visual_px.png')
            Image.fromarray(result.diagnostics['face_mask']).save(temp/'face_mask.png')
            n=result.diagnostics['normals']
            Image.fromarray(np.round(np.clip((n+1)*127.5,0,255)).astype(np.uint8)).save(temp/'normals.png')
            Image.fromarray(np.round(result.diagnostics['shadow']*65535).astype(np.uint16)).save(temp/'shadow.png')
        # Keep the exact original file (including EXIF), not a decoded/recompressed substitute.
        shutil.copy2(spec.image_path,temp/('source'+Path(spec.image_path).suffix))
        (temp/'metadata.json').write_text(json.dumps(result.metadata,ensure_ascii=False,indent=2,default=str))
        temp.rename(output)
    except Exception:
        shutil.rmtree(temp,ignore_errors=True)
        raise
    return output
