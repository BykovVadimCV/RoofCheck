"""Typed outputs for roof-surface detection. Coordinates are EXIF-oriented pixels."""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class ProbabilityMaps:
    roof: np.ndarray
    boundary: np.ndarray
    obstacle: np.ndarray
    # Optional instance masks from proposal+refinement backends (e.g. GroundingDINO+SAM2).
    instances: list[tuple[np.ndarray, float]] = field(default_factory=list)
    model_info: dict = field(default_factory=dict)

    def validate(self, shape: tuple[int, int]):
        for name in ('roof', 'boundary', 'obstacle'):
            arr = getattr(self, name)
            if arr.shape != shape or arr.ndim != 2 or not np.isfinite(arr).all():
                raise ValueError(f'Invalid {name} probabilities: expected {shape}')
            if np.any((arr < 0) | (arr > 1)):
                raise ValueError(f'{name} probabilities must lie in [0,1]')
        for mask, conf in self.instances:
            if mask.shape != shape or mask.dtype != np.bool_ or not 0 <= conf <= 1:
                raise ValueError('Invalid instance mask or confidence')
        return self


@dataclass
class Facet:
    facet_id: str
    polygon: list[list[float]]
    pixels: int
    confidence: float
    mode: str = 'approximate'
    facet_type: str = 'unknown'   # never guess pitched vs flat from one raster mask

    def to_surface(self) -> dict:
        # Omit quad: RoofCheck geometry.py estimates a canonical quad without
        # clipping an irregular mask to that approximation.
        return dict(surface_id=self.facet_id, polygon=self.polygon,
                    mode=self.mode, grid_size=[256, 256])


@dataclass
class Roof:
    roof_id: str
    mask: np.ndarray
    confidence: float
    facets: list[Facet] = field(default_factory=list)

    @property
    def bbox(self) -> list[int]:
        yy, xx = np.where(self.mask)
        return [int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1]

    @property
    def pixels(self) -> int:
        return int(self.mask.sum())


@dataclass
class Selection:
    roof_ids: list[str]
    status: str  # selected | ambiguous | no_roofs
    reason: str


@dataclass
class Detection:
    roofs: list[Roof]
    selection: Selection
    obstacle_mask: np.ndarray
    shape: tuple[int, int]
    model_info: dict
    diagnostics: dict = field(default_factory=dict)
