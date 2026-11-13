"""Deterministic target selection. Never silently select the largest roof."""
from __future__ import annotations
import numpy as np
from .types import Roof, Selection


def select(roofs: list[Roof], *, point: tuple[int, int] | None = None,
           bbox: tuple[int, int, int, int] | None = None,
           reference: np.ndarray | None = None, all_roofs: bool = False) -> Selection:
    count = sum(x is not None for x in (point, bbox, reference)) + int(all_roofs)
    if count > 1:
        raise ValueError('Choose one selection mode: point, bbox, reference, or all')
    if not roofs:
        return Selection([], 'no_roofs', 'No roof proposals were found')
    if all_roofs:
        return Selection([r.roof_id for r in roofs], 'selected', 'Explicit all-roofs selection')
    if point is not None:
        x, y = map(int, point)
        matches = [r for r in roofs if 0 <= y < r.mask.shape[0] and
                   0 <= x < r.mask.shape[1] and r.mask[y, x]]
        if len(matches) == 1:
            return Selection([matches[0].roof_id], 'selected', 'Target point inside roof')
        return Selection([], 'ambiguous', 'Target point missed roofs or hit overlapping roof proposals')
    if bbox is not None or reference is not None:
        region = np.zeros(roofs[0].mask.shape, bool)
        if reference is not None:
            if reference.shape != region.shape:
                raise ValueError('Reference mask shape must equal oriented image size')
            region = reference.astype(bool)
        else:
            x1, y1, x2, y2 = map(int, bbox)
            if not (0 <= x1 < x2 <= region.shape[1] and 0 <= y1 < y2 <= region.shape[0]):
                raise ValueError('Invalid target bbox')
            region[y1:y2, x1:x2] = True
        scores = []
        for roof in roofs:
            inter = np.count_nonzero(roof.mask & region)
            # IoU as a target match, not confidence of roof segmentation.
            union = np.count_nonzero(roof.mask | region)
            scores.append(inter / union if union else 0.)
        order = np.argsort(scores)[::-1]
        best = float(scores[order[0]])
        runnerup = float(scores[order[1]]) if len(roofs) > 1 else 0.0
        if best < .20 or (len(roofs) > 1 and best - runnerup < .10):
            return Selection([], 'ambiguous', f'Target overlap is weak or tied ({best:.3f} vs {runnerup:.3f})')
        return Selection([roofs[int(order[0])].roof_id], 'selected', f'Target overlap IoU={best:.3f}')
    if len(roofs) == 1:
        return Selection([roofs[0].roof_id], 'selected', 'Only one detected roof')
    return Selection([], 'ambiguous', f'{len(roofs)} roofs found; specify --target-point/--target-bbox/--reference-mask/--all')
