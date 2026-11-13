"""File contract: masks are explicitly in raw or EXIF-oriented photo coordinates."""
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
from .masks import validate_mask


@dataclass
class RoofSample:
    image_id: str
    image_path: str | Path
    group_id: str
    roof_mask: str | Path
    snow_mask: str | Path | None = None
    split: str = 'train'
    inspection_roi: str | Path | None = None
    occluder_mask: str | Path | None = None
    annotation_mode: str = 'partial'  # partial, full, verified_clean
    source: str = ''
    mask_coordinates: str = 'oriented'  # raw or oriented; do not guess from dimensions
    surfaces: list[dict] = field(default_factory=list)


@dataclass
class LoadedSample:
    spec: RoofSample
    image: np.ndarray  # RGB uint8
    roof: np.ndarray
    snow: np.ndarray
    roi: np.ndarray
    occluders: np.ndarray
    allowed: np.ndarray


def load_sample(spec):
    if not spec.image_id or not spec.group_id or spec.split not in {'train', 'val', 'test'}:
        raise ValueError('image_id, group_id and a valid split are required')
    if spec.annotation_mode not in {'full', 'partial', 'verified_clean'}:
        raise ValueError('Unknown annotation_mode')
    if spec.mask_coordinates not in {'raw', 'oriented'}:
        raise ValueError('mask_coordinates must be raw or oriented')
    with Image.open(spec.image_path) as im:
        raw_size = im.size
        orientation = im.getexif().get(274, 1)
        image = np.array(ImageOps.exif_transpose(im).convert('RGB'))
    shape = image.shape[:2]
    if min(shape) < 8:
        raise ValueError('Photo dimensions must be at least 8 pixels')

    def read(path, semantic=False):
        with Image.open(path) as im:
            if im.mode not in {'L', 'P', '1', 'I', 'I;16'}:
                raise ValueError(f'Mask must be single-channel: {path}')
            if spec.mask_coordinates == 'raw':
                if im.size != raw_size:
                    raise ValueError('Raw mask dimensions differ from raw photo')
                operation = {2: Image.Transpose.FLIP_LEFT_RIGHT, 3: Image.Transpose.ROTATE_180,
                             4: Image.Transpose.FLIP_TOP_BOTTOM, 5: Image.Transpose.TRANSPOSE,
                             6: Image.Transpose.ROTATE_270, 7: Image.Transpose.TRANSVERSE,
                             8: Image.Transpose.ROTATE_90}.get(orientation)
                if operation is not None:
                    im = im.transpose(operation)
            arr = np.array(im)
        if arr.shape != shape:
            raise ValueError(f'Mask dimensions differ from oriented photo: {path}')
        if semantic:
            return validate_mask(arr, shape).copy()
        if not set(np.unique(arr)) <= {0, 1, 255}:
            raise ValueError(f'Region mask must be 0/1 or 0/255: {path}')
        return arr > 0

    roof = read(spec.roof_mask)
    if not roof.any():
        raise ValueError('Empty roof surface')
    occ = read(spec.occluder_mask) if spec.occluder_mask else np.zeros(shape, bool)
    roi = read(spec.inspection_roi) if spec.inspection_roi else roof.copy()
    if spec.snow_mask:
        snow = read(spec.snow_mask, semantic=True)
        if spec.annotation_mode == 'full' and (snow == 255).any():
            raise ValueError('Full annotation cannot contain unknown pixels')
        if spec.annotation_mode == 'verified_clean' and (snow == 1).any():
            raise ValueError('verified_clean conflicts with existing snow')
    elif spec.annotation_mode == 'verified_clean':
        # Clean assertion applies to the annotated visible roof, not the whole scene.
        snow = np.full(shape, 255, np.uint8)
        snow[roof & ~occ] = 0
    elif spec.annotation_mode == 'partial':
        snow = np.full(shape, 255, np.uint8)
    else:
        raise ValueError('Full annotation requires snow_mask')
    return LoadedSample(spec, image, roof, snow, roi, occ, roof & ~occ & (snow != 1))
