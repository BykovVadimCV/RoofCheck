"""Optional real snow microtexture; geometry and masks remain independent of the asset."""
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageOps
from .geometry import warp_texture


def projected_material(path, shape, owner, surfaces, seed, tile_px):
    path=Path(path)
    if not path.is_file():raise FileNotFoundError(f'Snow texture not found: {path}')
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    sidecar=path.with_suffix('.json')
    provenance=json.loads(sidecar.read_text()) if sidecar.is_file() else None
    if provenance and provenance.get('split') in {'val','test'}:
        raise ValueError('Texture from validation/test data cannot be used for training')
    with Image.open(path) as im:
        im=ImageOps.exif_transpose(im).convert('RGB')
        im.thumbnail((1024,1024),Image.Resampling.LANCZOS)
        rgb=np.asarray(im,dtype=np.float32)/255
    if min(rgb.shape[:2])<16:raise ValueError('Snow texture must be at least 16 pixels per side')
    gray=rgb@np.array([.2126,.7152,.0722],np.float32)
    # Remove baked-in scene illumination and preserve only bounded material detail.
    hp=gray-cv2.GaussianBlur(gray,(0,0),max(2.,min(gray.shape)*.04))
    deviation=float(hp.std())
    if deviation<1e-5:raise ValueError('Snow texture has no visible detail')
    hp=np.clip(hp/deviation,-2.5,2.5).astype(np.float32)
    rng=np.random.default_rng(np.random.SeedSequence([int(seed), 74291]))
    output=np.zeros(shape,np.float32)
    for i,g in enumerate(surfaces):
        region=owner==i
        if not region.any():continue
        tile=np.rot90(hp,int(rng.integers(4)))
        if rng.integers(2):tile=tile[:,::-1]
        tile=cv2.resize(np.ascontiguousarray(tile),(tile_px,tile_px),interpolation=cv2.INTER_AREA)
        tile=np.roll(tile,(int(rng.integers(tile_px)),int(rng.integers(tile_px))),axis=(0,1))
        if g['grid_to_image'] is None:
            h,w=shape
            detail=np.tile(tile,(h//tile_px+1,w//tile_px+1))[:h,:w]
        else:
            w,h=g['grid_size']
            grid=np.tile(tile,(h//tile_px+1,w//tile_px+1))[:h,:w]
            detail=warp_texture(grid,np.asarray(g['grid_to_image'],np.float64),shape)
        output[region]=detail[region]
    return output,dict(path=str(path.resolve()),sha256=digest,width=int(rgb.shape[1]),height=int(rgb.shape[0]),
        preprocessing='luminance, low-frequency removal, bounded contrast, random rotation/offset, mip projection',
        scale='canonical pixels; physical scale not calibrated',provenance=provenance)
