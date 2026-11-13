from dataclasses import dataclass
import numpy as np
from scipy import ndimage as ndi
from .scenarios import SCENARIOS, scenario_field


@dataclass(frozen=True)
class GenerationConfig:
    scenario: str = 'partial_cover'
    target_coverage: float | None = None
    coverage_tolerance: float = .02
    alpha_low: float = .01
    alpha_high: float = .85
    edge_px: float = 1.0
    texture_strength: float = .025
    texture_path: str | None = None
    real_texture_mix: float = .5
    texture_tile_px: int = 128
    render_mode: str = 'volume'
    height_px: float = 6.0
    freshness: str = 'fresh'
    camera_blur: float = .6
    light_direction: tuple = (-.5, -.6, .8)
    lift: tuple = (0., -1.)
    directional_strength: float = .20
    ambient_occlusion: float = .12
    shadow_strength: float = .16

    def validate(self):
        if self.scenario not in SCENARIOS:
            raise ValueError(f'Unknown scenario: {self.scenario}')
        values = [self.coverage_tolerance,self.alpha_low,self.alpha_high,self.edge_px,self.texture_strength]
        if not np.isfinite(values).all() or not 0 <= self.alpha_low < self.alpha_high <= 1:
            raise ValueError('Invalid generation parameters')
        if self.edge_px < 0 or not 0 <= self.texture_strength <= .2 or not 0 <= self.coverage_tolerance <= 1:
            raise ValueError('Invalid edge, texture or tolerance parameter')
        if not np.isfinite(self.real_texture_mix) or not 0 <= self.real_texture_mix <= 1:
            raise ValueError('real_texture_mix must be in [0,1]')
        if not isinstance(self.texture_tile_px, int) or not 16 <= self.texture_tile_px <= 1024:
            raise ValueError('texture_tile_px must be an integer in 16..1024')
        if self.render_mode not in {'flat','volume'} or self.freshness not in {'fresh','settled','wet'}:
            raise ValueError('Invalid render_mode or freshness')
        extra=[self.height_px,self.camera_blur,self.directional_strength,self.ambient_occlusion,self.shadow_strength]
        if not np.isfinite(extra).all() or not 0<=self.height_px<=24 or not 0<=self.camera_blur<=2:
            raise ValueError('Invalid height_px or camera_blur')
        if any(not 0<=v<=1 for v in extra[2:]):raise ValueError('Shading strengths must be in [0,1]')
        if len(self.light_direction)!=3 or not np.isfinite(self.light_direction).all() or self.light_direction[2]<=0:
            raise ValueError('light_direction must be finite and point above the surface')
        if len(self.lift)!=2 or not np.isfinite(self.lift).all() or not np.isclose(np.linalg.norm(self.lift),1,atol=1e-4):
            raise ValueError('lift must be a unit 2D vector')
        if self.lift[1]>0:
            raise ValueError('lift must point up-image for the current painter visibility order')
        target = self.target_coverage if self.target_coverage is not None else SCENARIOS[self.scenario]
        if not np.isfinite(target) or not 0 <= target <= 1:
            raise ValueError('target_coverage must be in [0,1]')
        return float(target)


def generate_alpha(loaded, base, config, rng, ordered_indices=None, rank_count=None):
    target = config.validate()
    eligible = loaded.allowed
    count = int(eligible.sum())
    if not count:
        if target:
            raise ValueError('No eligible pixels for added snow')
        return np.zeros(eligible.shape,np.float32), dict(target_coverage=target,generated_coverage=0.,eligible_pixels=0,components=0)
    # Stable rank selection on a fixed field. No RNG, morphology or moving objective inside a threshold search.
    indices = np.flatnonzero(eligible)
    k = int(round(target*count)) if rank_count is None else int(rank_count)
    if not 0<=k<=count:raise ValueError('Invalid rank_count')
    selected = np.zeros(eligible.shape,bool)
    if k:
        if ordered_indices is None:
            score = scenario_field(config.scenario, base, loaded, rng)
            ordered_indices = indices[np.argsort(score.ravel()[indices],kind='stable')]
        selected.ravel()[ordered_indices[-k:]] = True
    alpha = selected.astype(np.float32)
    if k and config.edge_px:
        # Opaque selected cores retain exact area; a weak exterior transition is explicitly ignore.
        distance = ndi.distance_transform_edt(~selected)
        fringe = (~selected) & eligible & (distance <= 2*config.edge_px)
        alpha[fringe] = np.minimum(config.alpha_high*.5, .4*np.exp(-distance[fringe]/config.edge_px))
    alpha[~eligible] = 0
    _, components = ndi.label(selected)
    return alpha, dict(target_coverage=target,generated_coverage=k/count,
                       eligible_pixels=count,components=int(components),
                       target_definition='added confident snow / eligible visible roof pixels; excludes original snow')
