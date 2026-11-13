"""Explicit binary snow contract, independent of internal material/depth labels."""
from pathlib import Path
import numpy as np
from PIL import Image

BACKGROUND, SNOW, IGNORE = 0, 1, 255


def validate_mask(mask, shape=None):
    if not isinstance(mask, np.ndarray) or mask.dtype != np.uint8 or mask.ndim != 2:
        raise ValueError('Snow mask must be a 2D uint8 array')
    if shape is not None and mask.shape != shape:
        raise ValueError(f'Mask shape {mask.shape} != image shape {shape}')
    if not set(np.unique(mask)) <= {BACKGROUND, SNOW, IGNORE}:
        raise ValueError('Snow mask supports only 0=background, 1=snow, 255=unknown')
    return mask


def save_mask(path, mask):
    validate_mask(mask)
    path = Path(path)
    if path.suffix.lower() != '.png':
        raise ValueError('Training masks must be PNG')
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(path)
    loaded = np.array(Image.open(path))
    validate_mask(loaded, mask.shape)
    if not np.array_equal(mask, loaded):
        raise IOError(f'Mask round-trip failed: {path}')


def materials_to_binary(material, snow_ids=(1, 2, 4), uncertain_ids=(3,)):
    """Legacy none/fresh/packed/slush_ice/pile; ambiguous slush is not snow."""
    if material.ndim != 2 or not set(np.unique(material)) <= {0, 1, 2, 3, 4}:
        raise ValueError('Unexpected material IDs')
    result = np.zeros(material.shape, np.uint8)
    result[np.isin(material, snow_ids)] = SNOW
    result[np.isin(material, uncertain_ids)] = IGNORE
    return result


def merge_snow(source, alpha, allowed, low=.01, high=.85):
    validate_mask(source)
    if not 0 <= low < high <= 1 or alpha.shape != source.shape or allowed.shape != source.shape:
        raise ValueError('Invalid alpha thresholds or shapes')
    if not np.isfinite(alpha).all() or np.any((alpha < 0) | (alpha > 1)):
        raise ValueError('Alpha outside [0,1]')
    allowed = allowed.astype(bool)
    result = source.copy()
    # Source snow is protected in both renderer and labels.
    active = allowed & (source != SNOW)
    result[active & (alpha > low) & (alpha < high)] = IGNORE
    result[active & (alpha >= high)] = SNOW
    return result


def coverage(mask, region):
    validate_mask(mask, region.shape)
    valid = region & (mask != IGNORE)
    n = int(valid.sum())
    return dict(snow_pixels=int(((mask == SNOW) & valid).sum()), valid_pixels=n,
                unknown_pixels=int((region & (mask == IGNORE)).sum()),
                fraction=float(((mask == SNOW) & valid).sum() / n) if n else None)
