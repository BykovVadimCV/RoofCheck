"""Snow on a pitched metal roof slope, drawn onto a real photo taken from a taller building.

Depth-first, as in the parking-lot generator (snowtex/depth_v4.py): one quantity, snow depth D in
millimetres, is generated on a metric grid of the rectified slope; appearance and labels both follow
from D, so the label is exact by construction.

  slope grid   rectified slope (see rectify.py), resampled so 1 px = 1/PX_PER_M m in both directions.
               Metric width comes from the standing-seam spacing (sheet width ~0.55 m), metric depth
               (eave -> ridge) is a scenario parameter, the photo does not show it.
  scenarios    fresh cover, drifts behind chimneys, cleared eave band, cleared strips, residual film
               after cleaning, ridge melt, trodden path along the fence, snow cornice at the eave
  surface      structures (chimneys, vents, dishes) and the fence are found on the slope and stay on
               top of the snow; seams (~25 mm high) show through snow thinner than that
  neighbours   other metal roofs in the photo get snow too, so a model cannot learn
               "snow only inside the target polygon"
  overlays     the app's timestamp and GPS pin are restored on top

Output per sample: image, depth_mm (photo px), class map (schema.SEG_CLASSES), coverage of the
visible slope area at several thresholds, scenario parameters.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import ndimage as ndi

from . import overlays
from .schema import SEG_CLASSES

CLS = {k: i for i, (k, _) in enumerate(SEG_CLASSES)}
PX_PER_M = 40                     # slope grid resolution
SEAM_SPACING_M = 0.55             # standing-seam sheet width on old Moscow roofs (randomised +-10%)
SEAM_HEIGHT_MM = 25
THRESHOLDS_MM = (1, 2, 5, 20, 60)
SCENARIOS = ["fresh", "drifted", "cleared_eave_band", "cleared_strips", "residual_film",
             "ridge_melt", "trodden_path", "cornice"]


# ------------------------------------------------------------------ exemplar snow textures (CC0)
TEXTURE_DIRS = ["/mnt/user-data/uploads/snow-survey/textures/cc0", "data/textures"]
_TEX = None


def texture_bank() -> list[np.ndarray]:
    """High-pass luminance detail of the CC0 snow scans (Poly Haven snow_*, ambientCG Snow*), unit std."""
    global _TEX
    if _TEX is None:
        import glob
        files = []
        for d in TEXTURE_DIRS:
            files += glob.glob(f"{d}/polyhaven/snow_*/*_color_2k.jpg") + glob.glob(f"{d}/polyhaven/asphalt_snow/*_color_2k.jpg")
            files += [f for f in glob.glob(f"{d}/ambientcg/Snow*/*_color_2k.jpg")]
        _TEX = []
        for f in sorted(set(files)):
            g = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if g is None:
                continue
            g = cv2.resize(g, (1024, 1024), interpolation=cv2.INTER_AREA).astype(np.float32)
            hp = g - cv2.GaussianBlur(g, (0, 0), 48)          # drop lighting gradients of the scan
            _TEX.append((hp - hp.mean()) / (hp.std() + 1e-6))
    return _TEX


def exemplar(shape, rng, tile_px) -> np.ndarray:
    """A random snow scan tiled over the grid at tile_px per 1024-px tile, random flip/rotation."""
    bank = texture_bank()
    H, W = shape
    if not bank:
        return fbm(shape, tile_px / 8, 4, rng)
    t = bank[rng.integers(len(bank))]
    t = np.rot90(t, rng.integers(4))
    if rng.random() < 0.5:
        t = t[:, ::-1]
    t = cv2.resize(np.ascontiguousarray(t), (max(8, int(tile_px)), max(8, int(tile_px))), interpolation=cv2.INTER_AREA)
    reps = (H // t.shape[0] + 2, W // t.shape[1] + 2)
    big = np.tile(t, reps)
    oy, ox = rng.integers(t.shape[0]), rng.integers(t.shape[1])
    out = big[oy:oy + H, ox:ox + W]
    return (out - out.mean()) / (out.std() + 1e-6)


# ------------------------------------------------------------------ noise helpers
def fbm(shape, scale_px, octaves, rng, aniso=(1.0, 1.0)):
    H, W = shape
    out = np.zeros(shape, np.float32)
    amp, tot = 1.0, 0.0
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    for o in range(octaves):
        s = scale_px / 2 ** o
        if s < 1.0:
            break
        g = rng.random((256, 256)).astype(np.float32)
        v = cv2.remap(g, ((xx / (s * aniso[0])) % 256).astype(np.float32), ((yy / (s * aniso[1])) % 256).astype(np.float32),
                      cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
        out += amp * v
        tot += amp
        amp *= 0.55
    out /= tot
    return (out - out.mean()) / (out.std() + 1e-6)


def soft_mask(mask, ramp_px):
    d_in = ndi.distance_transform_edt(mask)
    d_out = ndi.distance_transform_edt(~mask)
    return np.clip(0.5 + (d_in - d_out) / (2 * ramp_px + 1e-6), 0, 1).astype(np.float32)


def warp_mask(mask, rng, amp_px, scale_px):
    H, W = mask.shape
    dx = fbm((H, W), scale_px, 3, rng) * amp_px
    dy = fbm((H, W), scale_px, 3, rng) * amp_px
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    return cv2.remap(mask.astype(np.float32), xx + dx, yy + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) > 0.5


# ------------------------------------------------------------------ geometry
@dataclass
class SlopeGeom:
    quad: np.ndarray            # photo px: eave-l, eave-r, ridge-r, ridge-l
    W: int                      # grid width (px), PX_PER_M per metre
    H: int                      # grid height (px): eave -> ridge depth
    M: np.ndarray               # photo -> grid
    Minv: np.ndarray            # grid -> photo
    length_m: float
    depth_m: float
    seam_spacing_px_rect: float | None


def seam_spacing(rect_gray: np.ndarray) -> float | None:
    """Dominant period of vertical seams in a rectified slope (px), from the column profile."""
    g = cv2.createCLAHE(2.0, (8, 8)).apply(rect_gray)
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, np.ones((1, 9), np.uint8)).astype(np.float32)
    prof = bh.mean(0)
    prof = prof - ndi.uniform_filter1d(prof, 61)
    n = len(prof)
    spec = np.abs(np.fft.rfft(prof * np.hanning(n)))
    freqs = np.fft.rfftfreq(n)
    ok = (freqs > 1 / 80) & (freqs < 1 / 6)          # periods 6..80 px
    if not ok.any() or spec[ok].max() < 3 * np.median(spec[ok]):
        return None
    return float(1 / freqs[ok][np.argmax(spec[ok])])


def slope_geometry(img: np.ndarray, quad: np.ndarray, depth_m: float, rng) -> SlopeGeom:
    q = np.asarray(quad, np.float32)
    w_px = 0.5 * (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3]))
    h_px = 0.5 * (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1]))
    dst0 = np.float32([[0, h_px - 1], [w_px - 1, h_px - 1], [w_px - 1, 0], [0, 0]])
    rect = cv2.warpPerspective(img, cv2.getPerspectiveTransform(q, dst0), (int(w_px), int(h_px)))
    sp = seam_spacing(cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY))
    sheet = SEAM_SPACING_M * rng.uniform(0.9, 1.1)
    # visible seam lines are not always single sheets (old roofs: 0.5-0.7 m sheets, but every 2nd-3rd
    # seam may read in the photo), so the seam estimate is kept as info and the metric length is a
    # randomised prior; it only scales drift / strip sizes, labels are shares and do not depend on it
    length_m = float(rng.uniform(20, 60))
    W = int(round(length_m * PX_PER_M))
    H = int(round(depth_m * PX_PER_M))
    dst = np.float32([[0, H - 1], [W - 1, H - 1], [W - 1, 0], [0, 0]])
    M = cv2.getPerspectiveTransform(q, dst)
    return SlopeGeom(q, W, H, M, np.linalg.inv(M), length_m, depth_m, sp)


# ------------------------------------------------------------------ surface analysis on the grid
def surface_layers(rect: np.ndarray, valid: np.ndarray) -> dict:
    """structures (stand above the slope), fence (in front, near the eave), seams (thin, vertical)."""
    H, W = rect.shape[:2]
    lab = cv2.cvtColor(rect, cv2.COLOR_BGR2LAB).astype(np.float32)
    v = valid > 0
    ref = np.median(lab[v], axis=0)
    dist = np.linalg.norm(lab - ref, axis=2)
    struct = (dist > 22) & v
    struct = ndi.binary_opening(struct, np.ones((5, 5)))
    struct = ndi.binary_closing(struct, np.ones((9, 9)))
    lbl, n = ndi.label(struct)
    keep = []
    for i, sl in enumerate(ndi.find_objects(lbl), start=1):
        if sl is None:
            continue
        hh, ww = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        area = (lbl[sl] == i).sum()
        seam_like = ww < 0.3 * PX_PER_M and hh > 0.8 * PX_PER_M     # a dark seam line, not a chimney
        if area > 0.04 * PX_PER_M ** 2 and not seam_like:
            keep.append(i)
    struct = np.isin(lbl, keep)
    g = cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY)
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    thin = (bh > 10) & v & ~struct
    # seams run the whole way eave -> ridge; fence posts are short and stand in the lower band
    # seams: allow a few degrees of residual tilt by widening before the long vertical opening
    wide = cv2.dilate(thin.astype(np.uint8), np.ones((1, 5), np.uint8))
    seams = (cv2.morphologyEx(wide, cv2.MORPH_OPEN, np.ones((max(9, int(0.45 * H)), 1), np.uint8)) > 0) & thin
    band = np.zeros_like(v)
    band[int(0.55 * H):] = True
    fence = thin & ~seams & band
    fence = ndi.binary_dilation(fence, np.ones((3, 3)))
    # roof surface: metal-coloured area (seams, fence and structures filled in), largest pieces only.
    # A loosely drawn polygon also covers sky, facades behind the ridge and air past the gable.
    metal = (dist < 26) & v
    surf = ndi.binary_closing(metal | fence | seams, np.ones((int(0.3 * PX_PER_M),) * 2))
    surf = ndi.binary_fill_holes(surf | struct) & v
    lbl, n = ndi.label(surf)
    if n:
        sizes = ndi.sum(surf, lbl, range(1, n + 1))
        surf = np.isin(lbl, 1 + np.flatnonzero(sizes > 0.05 * sizes.max()))
    return {"structure": struct & surf, "fence": fence & surf, "seams": seams & surf, "surface": surf, "metal_lab": ref}


# ------------------------------------------------------------------ depth scenarios (mm on the grid)
@dataclass
class Scenario:
    name: str
    params: dict = field(default_factory=dict)


def sample_scenario(rng, name: str | None = None) -> Scenario:
    name = name or rng.choice(SCENARIOS)
    p = {"fall_mm": float(np.exp(rng.uniform(np.log(30), np.log(350)))),
         "wind_dir": float(rng.choice([-1, 1])), "variation": float(rng.uniform(0.15, 0.4))}
    if name == "residual_film":
        p["fall_mm"] = float(rng.uniform(0.5, 6))
    return Scenario(str(name), p)


def depth_field(sc: Scenario, H: int, W: int, layers: dict, rng) -> tuple[np.ndarray, np.ndarray]:
    """Returns D (mm) and a 'wet' map (0..1: melt water on metal near melt edges)."""
    p = sc.params
    P = PX_PER_M
    F = p["fall_mm"]
    base = 1 + p["variation"] * fbm((H, W), 3.0 * P, 4, rng)
    # snow creeps down the slope and is held by the fence / snow guards above the eave
    v = np.linspace(1, 0, H)[:, None]                        # 0 at eave (bottom row), 1 at ridge
    profile = 1 + 0.25 * (0.5 - v) + 0.5 * np.exp(-((v * H / P) / rng.uniform(0.6, 1.4)) ** 2)
    D = F * base * profile
    struct = layers["structure"]
    wet = np.zeros((H, W), np.float32)

    # drifts on the lee side of structures, scour on the windward side
    if sc.name in ("drifted", "fresh", "cornice") or rng.random() < 0.4:
        lbl, n = ndi.label(struct)
        for i in range(1, n + 1):
            ys, xs = np.nonzero(lbl == i)
            cx, cy, wb = xs.mean(), ys.mean(), (xs.max() - xs.min() + 1)
            L = rng.uniform(1.5, 5.0) * P
            dx = (np.arange(W)[None, :] - cx) * p["wind_dir"]
            dy = np.arange(H)[:, None] - cy
            lee = np.exp(-np.clip(dx, 0, None) / L) * (dx > 0) * np.exp(-(dy / (0.6 * wb + 0.3 * P)) ** 2)
            ws = np.exp(-np.clip(-dx, 0, None) / (0.5 * P)) * (dx <= 0) * np.exp(-(dy / (0.5 * wb + 0.2 * P)) ** 2)
            D = D * (1 + rng.uniform(0.6, 1.8) * lee) * (1 - 0.6 * ws)

    if sc.name == "cleared_eave_band":
        band_m = rng.uniform(0.8, 2.2)
        m = (np.arange(H)[:, None] > H - band_m * P) & np.ones((1, W), bool)
        m = warp_mask(m, rng, 0.15 * P, 1.5 * P)
        D = np.where(m, residual(H, W, rng), D)
        D += berm(m, rng, F)
    elif sc.name == "cleared_strips":
        m = np.zeros((H, W), bool)
        x = rng.uniform(0, 2) * P
        while x < W:
            w = rng.uniform(1.0, 3.0) * P
            if rng.random() < 0.65:
                m[:, int(x):int(x + w)] = True
            x += w + rng.uniform(0.3, 2.5) * P
        m &= np.arange(H)[:, None] > rng.uniform(0, 0.5) * H         # strips may stop short of the ridge
        m = warp_mask(m, rng, 0.2 * P, 2 * P)
        D = np.where(m, residual(H, W, rng), D)
        D += berm(m, rng, F)
    elif sc.name == "residual_film":
        patches = fbm((H, W), 1.5 * P, 4, rng) > rng.uniform(0.6, 1.4)
        D = residual(H, W, rng, hi=F) + patches * rng.uniform(5, 40) * soft_mask(patches, 0.2 * P)
    elif sc.name == "ridge_melt":
        melt_m = rng.uniform(0.5, 2.0)
        edge = (np.arange(H)[:, None] < melt_m * P + 0.4 * P * fbm((H, W), 2 * P, 3, rng))
        for_struct = ndi.distance_transform_edt(~struct) < rng.uniform(0.2, 0.7) * P   # warm chimneys
        melt = edge | for_struct
        s = soft_mask(melt, 0.25 * P)
        D = D * (1 - s)
        ring = (s > 0.05) & (s < 0.95)
        wet = np.maximum(wet, ndi.gaussian_filter(ring.astype(np.float32), 0.3 * P) * 2)
    elif sc.name == "trodden_path":
        y0 = H - rng.uniform(0.6, 1.4) * P
        path = np.abs(np.arange(H)[:, None] - y0 - 0.2 * P * fbm((H, W), 4 * P, 2, rng)) < rng.uniform(0.25, 0.45) * P
        steps = fbm((H, W), 0.15 * P, 2, rng) > 0.3
        D = np.where(path, D * rng.uniform(0.3, 0.6) * (1 - 0.4 * steps), D)
    D = np.clip(D, 0, None).astype(np.float32)
    D[struct] = 0
    D[~layers["surface"]] = 0
    # micro unevenness, mm scale
    # cm-scale unevenness and crumbly patch edges: thin snow breaks up, deep snow barely changes
    ex = exemplar((H, W), rng, rng.uniform(3, 8) * P)
    rough = 0.6 * fbm((H, W), 0.15 * P, 3, rng) + 0.4 * ex
    D += np.clip(D, 0, 30) * rng.uniform(0.25, 0.5) * rough + np.clip(D, 0, 400) * 0.03 * ex
    # snow survives in the grooves along the standing seams after clearing / melting
    if sc.name in ("residual_film", "cleared_strips", "cleared_eave_band", "ridge_melt") or rng.random() < 0.3:
        groove = ndi.binary_dilation(layers["seams"], np.ones((1, 3))) ^ layers["seams"]
        groove = ndi.gaussian_filter(groove.astype(np.float32), 0.6) * (fbm((H, W), 0.4 * P, 2, rng) > -0.3)
        D += groove * rng.uniform(2, 10)
    D[struct] = 0
    D[~layers["surface"]] = 0
    return np.clip(D, 0, None), np.clip(wet, 0, 1)


def target_melt(D, vis, tau, target, noise, iters=30):
    """Subtract a spatially varying melt depth M*(1+0.5*noise) so that share(D >= tau) on vis == target."""
    if not vis.any():
        return D
    f = 1 + 0.5 * np.clip(noise, -1.5, 1.5)
    lo, hi = 0.0, float(np.percentile(D[vis], 99.5)) * 2 + 1
    for _ in range(iters):
        M = 0.5 * (lo + hi)
        cov = ((D - M * f)[vis] >= tau).mean()
        if cov > target:
            lo = M
        else:
            hi = M
    return np.clip(D - 0.5 * (lo + hi) * f, 0, None).astype(np.float32)


def residual(H, W, rng, hi=None):
    """What a shovel leaves on metal: a patchy film 0..hi mm."""
    hi = hi if hi is not None else rng.uniform(1, 8)
    f = np.clip(0.5 + 0.35 * fbm((H, W), 0.6 * 40, 4, rng, aniso=(0.5, 2.0)), 0, 1)
    return (hi * f ** 2).astype(np.float32)


def berm(cleared, rng, F):
    """Snow pushed to the edge of a cleared area (small: on roofs most is thrown off)."""
    d = ndi.distance_transform_edt(~cleared) / PX_PER_M
    return (F * rng.uniform(0.1, 0.4) * np.exp(-((d - 0.15) / 0.2) ** 2) * (~cleared)).astype(np.float32)


# ------------------------------------------------------------------ appearance
def snow_rgb(img: np.ndarray, rng, metal_bgr: np.ndarray | None = None) -> np.ndarray:
    """Snow colour under the photo's light: the bright end of the frame, slightly blue, and clearly
    brighter than the roof metal (real winter frames: snow ~1.5-1.9x the exposed metal)."""
    # white point from bright, low-saturation pixels (sky, white walls), not from yellow facades
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lum = img.mean(2)
    sel = (hsv[..., 1] < 35) & (lum >= np.percentile(lum, 95))
    pix = img[sel] if sel.sum() > 100 else img.reshape(-1, 3)
    white = np.percentile(pix, 90, axis=0).astype(np.float32)
    white = white.mean() * (0.85 + 0.15 * white / white.mean())        # keep it near neutral
    k = rng.uniform(0.92, 1.0)
    tint = np.array([1.03, 1.0, 0.97]) * rng.uniform(0.985, 1.015, 3)    # BGR: a touch of blue
    snow = white * k * tint
    if metal_bgr is not None:
        lum_m, lum_s = float(np.mean(metal_bgr)), float(np.mean(snow))
        if lum_s < 1.35 * lum_m:
            snow *= 1.35 * lum_m / max(lum_s, 1)
    return np.clip(snow, 0, 255).astype(np.float32)


def render_grid(D, wet, layers, rect, rng, d0=None):
    """Snow layer on the grid: colour (float BGR) and alpha."""
    P = PX_PER_M
    d0 = d0 or rng.uniform(2.0, 4.0)
    Deff = np.where(layers["seams"], np.clip(D - SEAM_HEIGHT_MM, 0, None), D)
    alpha = 1 - np.exp(-Deff / d0)
    # shading: overcast sky light is soft; relief shows only weakly (drift faces, hollows)
    Dm = ndi.gaussian_filter(D, 0.08 * P) / 1000.0            # metres
    gy, gx = np.gradient(Dm * P)
    shade = 1 - 0.08 * np.clip(gy * 3, -1, 1) - 0.04 * np.clip(gx * rng.uniform(-3, 3), -1, 1)
    ao = np.clip((Dm - ndi.gaussian_filter(Dm, 0.5 * P)) * 6, -0.06, 0.03)
    tex = exemplar(D.shape, rng, rng.uniform(2.5, 7) * P)
    grain = rng.uniform(0.015, 0.045) * tex + 0.008 * fbm(D.shape, 0.05 * P, 2, rng)
    # old snow collects soot and debris: sparse darker specks and smudges
    if rng.random() < 0.35:
        specks = (fbm(D.shape, 0.06 * P, 2, rng) > rng.uniform(2.6, 3.2)).astype(np.float32)
        grain -= 0.08 * ndi.gaussian_filter(specks, 0.7)
    # snow drapes over the seams: faint stripes that fade as the layer gets deeper
    drape = ndi.gaussian_filter(layers["seams"].astype(np.float32), 0.04 * P) * np.exp(-D / 60.0)
    s = np.clip(shade + ao + grain - 0.05 * drape, 0.72, 1.06)[..., None]
    wet_dark = (1 - 0.35 * wet)[..., None]
    return s, alpha.astype(np.float32), wet_dark


def neighbour_roof_mask(img, target_poly, metal_lab, exclude=None):
    """Other metal roofs: colour close to the target roof metal, outside the target polygon."""
    H, W = img.shape[:2]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    dist = np.linalg.norm(lab - metal_lab, axis=2)
    m = (dist < 14).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    t = np.zeros((H, W), np.uint8)
    cv2.fillPoly(t, [np.round(target_poly).astype(np.int32)], 1)
    m[cv2.dilate(t, np.ones((15, 15), np.uint8)) > 0] = 0
    if exclude is not None:
        m[exclude > 0] = 0
    n, lbl, st, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros((H, W), bool)
    for i in range(1, n):
        if st[i, 4] > 0.0015 * H * W:
            keep |= lbl == i
    return keep


# ------------------------------------------------------------------ main entry
@dataclass
class Sample:
    image: np.ndarray
    depth_mm: np.ndarray        # float32, photo px; 0 outside snow-bearing surfaces
    classes: np.ndarray         # uint8, schema.SEG_CLASSES index
    coverage: dict              # threshold mm -> share of visible slope area
    info: dict


def neighbour_from_polygons(img, polys, metal_lab):
    """Other roofs marked once on the reference (role other_roof), restricted to roof-like colour
    (low saturation grey/blue metal) so facades inside a rough polygon stay untouched."""
    H, W = img.shape[:2]
    m = np.zeros((H, W), np.uint8)
    for p in polys:
        cv2.fillPoly(m, [np.round(p).astype(np.int32)], 1)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    roofish = (hsv[..., 1] < 70) & (np.linalg.norm(lab - metal_lab, axis=2) < 40)
    k = (m > 0) & roofish
    k = ndi.binary_opening(k, np.ones((3, 3)))
    return ndi.binary_closing(k, np.ones((7, 7))) & (m > 0)


def make_sample(img: np.ndarray, quad: np.ndarray, roof_poly: np.ndarray, rng,
                scenario: str | None = None, neighbours: bool = True, label_tau_mm: float = 2.0,
                other_roofs: list | None = None) -> Sample:
    H0, W0 = img.shape[:2]
    ov = overlays.detect(img)
    sc = sample_scenario(rng, scenario)
    depth_m = rng.uniform(4.0, 8.0)
    geo = slope_geometry(img, quad, depth_m, rng)
    Hg, Wg = geo.H, geo.W
    rect = cv2.warpPerspective(img, geo.M, (Wg, Hg), flags=cv2.INTER_LINEAR)
    ok_src = np.full((H0, W0), 255, np.uint8)
    ok_src[ov.mask > 0] = 0
    valid_src = cv2.warpPerspective(np.full((H0, W0), 255, np.uint8), geo.M, (Wg, Hg), flags=cv2.INTER_NEAREST)
    layers = surface_layers(rect, valid_src)
    D, wet = depth_field(sc, Hg, Wg, layers, rng)
    vis0 = (cv2.warpPerspective(ok_src, geo.M, (Wg, Hg), flags=cv2.INTER_NEAREST) > 0) & (valid_src > 0) \
        & ~layers["structure"] & layers["surface"]
    # balance the coverage distribution: thin out the cover to a target share (thin areas go first)
    target = None
    if sc.name in ("fresh", "drifted", "trodden_path", "cornice", "cleared_strips") and rng.random() < 0.6:
        target = float(rng.uniform(0.05, 0.95))
        D = target_melt(D, vis0, label_tau_mm, target, fbm((Hg, Wg), 2.0 * PX_PER_M, 3, rng))

    # cornice: snow pushed past the eave; the grid is extended below the eave line
    ext = 0
    if sc.name == "cornice" or (sc.params["fall_mm"] > 150 and rng.random() < 0.5):
        ext = int(rng.uniform(0.15, 0.45) * PX_PER_M)
        lip = np.repeat(D[-1:, :], ext, 0)
        prof = np.linspace(1, 0, ext)[:, None] ** rng.uniform(0.6, 1.5)
        edge = fbm((ext, Wg), 0.3 * PX_PER_M, 3, rng) * 0.25
        lip = lip * np.clip(prof + edge, 0, 1) * (1.0 + 0.3 * rng.random())
        lip[lip < 30] = 0
        D = np.vstack([D, lip.astype(np.float32)])
        wet = np.vstack([wet, np.zeros((ext, Wg), np.float32)])
        for k in ("structure", "fence", "seams"):
            layers[k] = np.vstack([layers[k], np.zeros((ext, Wg), bool)])
        layers["surface"] = np.vstack([layers["surface"], lip > 0])

    shade, alpha, wet_dark = render_grid(D, wet, layers, rect, rng)
    metal_bgr = cv2.cvtColor(np.uint8([[layers['metal_lab']]]), cv2.COLOR_LAB2BGR)[0, 0].astype(np.float32)
    snow = snow_rgb(img, rng, metal_bgr)

    # grid -> photo
    Minv = geo.Minv
    to_photo = lambda a, interp=cv2.INTER_LINEAR: cv2.warpPerspective(a, Minv, (W0, H0), flags=interp,
                                                                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    A = to_photo(alpha)
    illum = 1 + rng.uniform(-0.06, 0.06) * np.linspace(-1, 1, shade.shape[1])[None, :] \
        + rng.uniform(-0.05, 0.05) * np.linspace(-1, 1, shade.shape[0])[:, None]
    S = cv2.warpPerspective((shade[..., 0] * illum).astype(np.float32), Minv, (W0, H0), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=1.0)
    Wd = cv2.warpPerspective(wet_dark[..., 0].astype(np.float32), Minv, (W0, H0), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=1.0)
    Dp = to_photo(D)
    on_grid = to_photo(layers["surface"].astype(np.float32), cv2.INTER_NEAREST) > 0.5
    struct_p = to_photo(layers["structure"].astype(np.float32), cv2.INTER_NEAREST) > 0.5
    fence_p = to_photo(layers["fence"].astype(np.float32), cv2.INTER_LINEAR) > 0.3

    # match sharpness: slight blur of the snow layer, then photo noise
    sig = rng.uniform(0.3, 0.7)
    A = cv2.GaussianBlur(A, (0, 0), sig)
    S = cv2.GaussianBlur(S, (0, 0), sig)
    img_f = img.astype(np.float32)
    # a phone exposes a snowy scene darker: the background comes down, the snow stays near white
    snow_frac = float(A.mean())
    e = 1 - min(0.25, rng.uniform(2.0, 5.0) * snow_frac) if snow_frac > 0.005 else 1.0
    img_f = img_f * e
    snow_layer = snow[None, None, :] * S[..., None]
    out = img_f * Wd[..., None]
    out = out * (1 - A[..., None]) + snow_layer * A[..., None]

    # neighbouring roofs
    neigh = np.zeros((H0, W0), bool)
    if neighbours:
        neigh = (neighbour_from_polygons(img, other_roofs, layers["metal_lab"]) if other_roofs
                 else neighbour_roof_mask(img, roof_poly, layers["metal_lab"], ov.mask))
        neigh &= ov.mask == 0
        if neigh.any():
            nb = np.clip(rng.uniform(0.5, 1.2) + 0.3 * fbm((H0, W0), 120, 3, rng), 0, 1)
            nb = cv2.GaussianBlur(nb * soft_mask(neigh, 3), (0, 0), 1.0)
            out = out * (1 - nb[..., None]) + (snow * rng.uniform(0.85, 1.0))[None, None, :] * nb[..., None]

    # things in front of the snow: fence, structures (with a little snow on top), overlays
    keep = fence_p | struct_p
    out[keep] = img_f[keep]
    # phone processing: the same sharpening halo and grain as the rest of the frame
    snowy = cv2.GaussianBlur((A > 0.05).astype(np.float32), (0, 0), 1.5)[..., None]
    k = rng.uniform(0.2, 0.6)
    sharp = out + k * (out - cv2.GaussianBlur(out, (0, 0), 1.2))
    out = out * (1 - snowy) + sharp * snowy
    sigma_n = photo_noise(img)
    n = rng.normal(0, 1, out.shape[:2]).astype(np.float32)
    n = cv2.GaussianBlur(n, (0, 0), 0.6)
    n *= sigma_n / (n.std() + 1e-6)
    chroma = cv2.GaussianBlur(rng.normal(0, 0.35 * sigma_n, out.shape).astype(np.float32), (0, 0), 1.5)
    out = np.clip(out + (n[..., None] + chroma) * snowy, 0, 255)
    out[ov.mask > 0] = img.astype(np.float32)[ov.mask > 0]      # the app stamps after exposure
    ok, enc = cv2.imencode(".jpg", out.astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(70, 93))])
    out = cv2.imdecode(enc, 1)

    # labels
    cls = np.zeros((H0, W0), np.uint8)
    slope_p = on_grid
    cls[slope_p] = CLS["metal_clean"]
    cls[slope_p & (Wd < 0.9)] = CLS["metal_wet"]
    cls[slope_p & (Dp >= label_tau_mm)] = CLS["snow"]
    q = np.asarray(quad)
    below_eave = _below_line(q[0], q[1], (H0, W0)) & on_grid
    cls[below_eave & (Dp >= label_tau_mm)] = CLS["overhang"]
    cls[struct_p] = CLS["structure"]
    cls[fence_p & slope_p] = CLS["fence"]
    cls[ov.mask > 0] = CLS["ignore"]
    Dp[~slope_p] = 0
    Dp[struct_p] = 0

    # coverage on the slope, by area: grid pixels are equal areas
    vis = cv2.warpPerspective(ok_src, geo.M, (Wg, Hg), flags=cv2.INTER_NEAREST) > 0
    vis &= valid_src > 0
    vis &= ~layers["structure"][:Hg] & layers["surface"][:Hg]
    Dg = D[:Hg]
    cov = {t: float((Dg[vis] >= t).mean()) if vis.any() else 0.0 for t in THRESHOLDS_MM}
    info = {"scenario": sc.name, **{k: round(v, 3) if isinstance(v, float) else v for k, v in sc.params.items()},
            "slope_length_m": round(geo.length_m, 1), "slope_depth_m": round(geo.depth_m, 1),
            "seam_period_px": round(geo.seam_spacing_px_rect, 1) if geo.seam_spacing_px_rect else None,
            "mean_depth_mm": round(float(Dg[vis].mean()), 1) if vis.any() else 0,
            "p95_depth_mm": round(float(np.percentile(Dg[vis], 95)), 1) if vis.any() else 0,
            "visible_slope_frac": round(float(vis.mean()), 3), "cornice_px": ext,
            "surface_frac_of_quad": round(float(layers["surface"][:Hg].mean()), 3),
            "neighbour_roof_frac": round(float(neigh.mean()), 3), "label_tau_mm": label_tau_mm,
            "target_coverage": round(target, 3) if target is not None else None, "exposure": round(e, 3)}
    return Sample(out, Dp.astype(np.float32), cls, cov, info)


def photo_noise(img: np.ndarray) -> float:
    """Grain of the photo itself: std of the fine residual in flat areas (sky, plain walls)."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hp = g - cv2.GaussianBlur(g, (0, 0), 1.0)
    grad = cv2.GaussianBlur(np.abs(cv2.Laplacian(cv2.GaussianBlur(g, (0, 0), 2.0), cv2.CV_32F)), (0, 0), 3)
    flat = grad < np.percentile(grad, 20)
    return float(np.clip(hp[flat].std() * 1.4, 1.0, 8.0))


def _below_line(a, b, shape):
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W]
    # sign of cross product: points below the eave line (larger y)
    return ((b[0] - a[0]) * (yy - a[1]) - (b[1] - a[1]) * (xx - a[0])) > 0
