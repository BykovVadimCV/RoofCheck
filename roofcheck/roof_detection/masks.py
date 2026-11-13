"""Conservative raster-to-facet decomposition; preserves the full roof mask."""
from __future__ import annotations
import cv2
import numpy as np
from scipy import ndimage as ndi
from .types import Facet, Roof, ProbabilityMaps


def _polygons(mask: np.ndarray, minimum: int = 4) -> list[list[list[float]]]:
    # CHAIN_APPROX_SIMPLE retains raster boundary turns. Avoid approxPolyDP:
    # large epsilon can push snowable pixels outside the actual roof.
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(contour) >= minimum and len(contour) >= 3:
            result.append(contour[:, 0, :].astype(float).tolist())
    return result


def polygon(mask: np.ndarray) -> list[list[float]]:
    polys = _polygons(mask)
    if not polys:
        raise ValueError('Cannot polygonize an empty or tiny region')
    return polys[0]


def _components(mask: np.ndarray, min_area: int) -> list[np.ndarray]:
    n, ids, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    comps = [ids == i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    return sorted(comps, key=lambda m: int(m.sum()), reverse=True)


def find_roofs(maps: ProbabilityMaps, threshold: float = 0.55, min_area: int = 700,
               instance_confidence: float = 0.20) -> list[Roof]:
    if not 0 < threshold < 1 or not 0 <= instance_confidence <= 1 or min_area < 4:
        raise ValueError('Invalid segmentation threshold or area')
    shape = maps.roof.shape
    if maps.instances:
        proposals = sorted(maps.instances, key=lambda m: m[1], reverse=True)
        assigned = np.zeros(shape, bool)
        regions = []
        for mask, confidence in proposals:
            # GroundingDINO * SAM quality scores are NOT comparable to
            # semantic roof probability. Keep independent thresholds.
            if confidence < instance_confidence:
                continue
            # Prevent double ownership across overlapping SAM proposals.
            available = mask & ~assigned
            for component in _components(available, min_area):
                regions.append((component, float(confidence)))
                assigned |= component
    else:
        regions = [(m, float(np.mean(maps.roof[m]))) for m in
                   _components(maps.roof >= threshold, min_area)]
    return [Roof(f'roof_{idx:03}', mask, conf)
            for idx, (mask, conf) in enumerate(regions, start=1)]


def split_facets(roof: Roof, boundary: np.ndarray, threshold: float = 0.65,
                 min_fraction: float = 0.06, min_area: int = 100) -> list[Facet]:
    """Split only when a *complete* learned boundary separates regions.

    Incomplete ridge guesses do not produce invented facets. The raster is
    partitioned exactly, using nearest-seed assignment for boundary pixels.
    """
    mask = roof.mask
    min_seed = max(min_area, int(mask.sum() * min_fraction))
    cut = (boundary >= threshold) & mask
    cut = cv2.dilate(cut.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    regions = _components(mask & ~cut, min_seed)
    if len(regions) < 2:
        regions = [mask]
        labels = mask.astype(np.int32)
    else:
        # Small pieces/boundary pixels inherit ownership from the nearest large facet.
        labels = np.zeros(mask.shape, np.int32)
        for i, region in enumerate(regions, 1):
            labels[region] = i
        missing = mask & (labels == 0)
        _, idx = ndi.distance_transform_edt(labels == 0, return_indices=True)
        labels[missing] = labels[tuple(k[missing] for k in idx)]
        if np.any(mask & (labels == 0)):
            raise RuntimeError('Facet decomposition left unlabeled roof pixels')

    facets = []
    for idx in range(1, int(labels.max()) + 1):
        region = labels == idx
        if not region.any():
            continue
        fac = Facet(f'{roof.roof_id}_f{idx:02}', polygon(region), int(region.sum()),
                    roof.confidence,
                    mode='approximate')
        facets.append(fac)
    if not np.array_equal(labels > 0, mask):
        raise RuntimeError('Facets do not cover exactly the selected roof')
    return facets


def build_detection(maps: ProbabilityMaps, threshold: float = .55,
                    obstacle_threshold: float = .55, boundary_threshold: float = .65,
                    min_roof_area: int = 700, instance_min_confidence: float = .20):
    from .types import Detection, Selection
    shape = maps.roof.shape
    roofs = find_roofs(maps, threshold=threshold, min_area=min_roof_area,
                       instance_confidence=instance_min_confidence)
    for roof in roofs:
        roof.facets = split_facets(roof, maps.boundary, threshold=boundary_threshold)
    obstacles = maps.obstacle >= obstacle_threshold
    return Detection(roofs, Selection([], 'ambiguous', 'Target not selected'), obstacles,
                     shape, maps.model_info,
                     diagnostics={'roof_count': len(roofs),
                                  'facet_counts': {r.roof_id: len(r.facets) for r in roofs},
                                  'warning': 'Facet count is only a hypothesis unless a facet-trained model was used.'})
