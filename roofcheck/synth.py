"""Snow on a pitched metal roof, drawn onto a real photo taken from a taller building.

Depth-first: snow depth D (mm) is generated on a metric grid of the rectified slope; appearance and
labels follow from D, so labels are exact by construction. Physics and parameters are documented in
docs/research-notes.md; the short version:

  amount       ground snow since the last clearing, drawn around the 30 cm clearing trigger
               (Gosstroy No. 170, cl. 4.6.1.23), times the roof shape coefficient mu(pitch) of
               SP 20.13330.2016 (1 up to 30 deg, 0 from 60 deg), wind exposure and thermal factors
  wind         unbalanced load between slopes, triangular drifts behind chimneys and against the
               fence (length 2h, total mu capped at 2.4), all mass-conserving: drifted snow is taken
               from the upwind part of the same roof
  state        not cleared / partially cleared (strips, eave band) / cleared (residue) / thaw
               (ridge and chimney melt, patchy). Coverage-balancing melt only happens in "thaw".
  optics       two-flux opacity alpha = bD / (1 + bD), b = 0.35-0.6 /mm (wet snow higher), fitted to
               Perovich (2007): substrate visible at 5 mm, near-white at 30 mm, white at 80 mm
  volume       visible slab at the eave, snow caps on chimney tops, dishes and the railing top rail,
               cornice lip; seams print through only while the layer is thinner than the seam
  scene        the same weather on neighbouring roofs and the photographer's own roof; foliage graded
               toward winter; all snow in the frame is labelled snow, the target roof is a separate ROI
  camera       blur, sharpening, haze, noise, clipping and JPEG quality measured on the host photo
               (hostmatch.py)

Per sample: image, depth_mm (photo px, all snow-bearing surfaces), class map, ROI mask of the target
slope, coverage of the visible target slope at several thresholds, parameters.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import ndimage as ndi

from . import hostmatch, overlays
from .schema import SEG_CLASSES

CLS = {k: i for i, (k, _) in enumerate(SEG_CLASSES)}
PX_PER_M = 40                     # slope grid resolution
SEAM_HEIGHT_MM = 25
THRESHOLDS_MM = (1, 2, 5, 20, 300)        # 300 mm: clearing trigger for pitched roofs with external drainage
IGNORE_BAND_MM = 1.0
STATES = ["not_cleared", "partially_cleared", "cleared", "thaw"]
STATE_P = [0.35, 0.25, 0.25, 0.15]


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
            files += glob.glob(f"{d}/ambientcg/Snow*/*_color_2k.jpg")
        _TEX = []
        for f in sorted(set(files)):
            g = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if g is None:
                continue
            g = cv2.resize(g, (1024, 1024), interpolation=cv2.INTER_AREA).astype(np.float32)
            hp = g - cv2.GaussianBlur(g, (0, 0), 48)
            _TEX.append((hp - hp.mean()) / (hp.std() + 1e-6))
    return _TEX


def exemplar(shape, rng, tile_px) -> np.ndarray:
    bank = texture_bank()
    H, W = shape
    if not bank:
        return fbm(shape, tile_px / 8, 4, rng)
    t = np.rot90(bank[rng.integers(len(bank))], rng.integers(4))
    if rng.random() < 0.5:
        t = t[:, ::-1]
    t = cv2.resize(np.ascontiguousarray(t), (max(8, int(tile_px)),) * 2, interpolation=cv2.INTER_AREA)
    big = np.tile(t, (H // t.shape[0] + 2, W // t.shape[1] + 2))
    oy, ox = rng.integers(t.shape[0]), rng.integers(t.shape[1])
    out = big[oy:oy + H, ox:ox + W]
    return ((out - out.mean()) / (out.std() + 1e-6)).astype(np.float32)


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


# ------------------------------------------------------------------ per-roof geometry (fixed per roof)
def roof_geometry(object_id: str, stored: dict | None = None) -> dict:
    """Metric size and pitch of a roof. Taken from the reference when known (client data, cadastre),
    otherwise drawn once from a prior seeded by the object id, so every sample of a roof agrees."""
    if stored and {"length_m", "depth_m", "pitch_deg"} <= set(stored):
        return dict(stored, source=stored.get("source", "reference"))
    seed = int(hashlib.sha256(object_id.encode()).hexdigest()[:8], 16)
    r = np.random.default_rng(seed)
    return {"length_m": round(float(r.uniform(20, 60)), 1), "depth_m": round(float(r.uniform(4, 8)), 1),
            "pitch_deg": round(float(r.uniform(15, 35)), 1), "source": "prior (seeded by object id)"}


@dataclass
class SlopeGeom:
    quad: np.ndarray
    W: int
    H: int
    M: np.ndarray               # photo -> grid
    Minv: np.ndarray            # grid -> photo
    length_m: float
    depth_m: float
    px_per_m_photo: float       # photo px per metre along the eave (also used for vertical extents)


def slope_geometry(quad: np.ndarray, geom: dict) -> SlopeGeom:
    q = np.asarray(quad, np.float32)
    W = int(round(geom["length_m"] * PX_PER_M))
    H = int(round(geom["depth_m"] * PX_PER_M))
    dst = np.float32([[0, H - 1], [W - 1, H - 1], [W - 1, 0], [0, 0]])
    M = cv2.getPerspectiveTransform(q, dst)
    eave_px = float(np.linalg.norm(q[1] - q[0]))
    return SlopeGeom(q, W, H, M, np.linalg.inv(M), geom["length_m"], geom["depth_m"], eave_px / geom["length_m"])


# ------------------------------------------------------------------ surface analysis on the grid
def surface_layers(rect: np.ndarray, valid: np.ndarray, known: np.ndarray, struct_known: np.ndarray | None = None) -> dict:
    """structure (stands above the slope: chimneys, dishes, dormers), fence (in front, lower band),
    seams (thin, eave->ridge), surface (snow-bearing metal), illum (shading of the metal, kept on snow).

    Structures are told apart from metal by colour (chroma) or by being much brighter, not by being
    darker: shadows, seams and wet patches are darker metal and must carry snow."""
    H, W = rect.shape[:2]
    lab = cv2.cvtColor(rect, cv2.COLOR_BGR2LAB).astype(np.float32)
    v = (valid > 0) & (known > 0)
    ref = np.median(lab[v], axis=0)
    dab = np.linalg.norm(lab[..., 1:] - ref[1:], axis=2)
    dL = lab[..., 0] - ref[0]
    # with structures outlined in the reference, only strong colour is taken (rust stains carry snow)
    struct = ((dab > (25 if struct_known is not None and struct_known.any() else 10)) | (dL > 30)) & v
    struct = ndi.binary_opening(struct, np.ones((5, 5)))
    struct = ndi.binary_closing(struct, np.ones((7, 7)))
    lbl, _ = ndi.label(struct)
    keep, outside = [], []
    for i, sl in enumerate(ndi.find_objects(lbl), start=1):
        if sl is None:
            continue
        hh, ww = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        area = (lbl[sl] == i).sum()
        on_border = sl[0].start == 0 or sl[1].start == 0 or sl[1].stop == W
        if on_border:
            outside.append(i)               # beyond the ridge or the gable: background, not a structure
        elif area > 0.05 * PX_PER_M ** 2 and not (ww < 0.25 * PX_PER_M and hh > 0.6 * PX_PER_M):
            keep.append(i)
    outside = np.isin(lbl, outside)
    struct = np.isin(lbl, keep)
    if struct_known is not None:                # structures outlined once per vantage (reference role 'structure')
        struct |= struct_known > 0

    g = cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY)
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    thin = (bh > 9) & v & ~struct
    wide = cv2.dilate(thin.astype(np.uint8), np.ones((1, 5), np.uint8))
    seams = (cv2.morphologyEx(wide, cv2.MORPH_OPEN, np.ones((max(9, int(0.25 * H)), 1), np.uint8)) > 0) & thin
    band = np.zeros_like(v)
    band[int(0.55 * H):] = True
    fence = ndi.binary_dilation(thin & ~seams & band, np.ones((3, 3)))

    metal = (dab < 12) & (dL < 30) & (valid > 0)
    surf = ndi.binary_closing(metal | fence | seams | ((known == 0) & (valid > 0)), np.ones((int(0.3 * PX_PER_M),) * 2),
                             border_value=1)
    surf = ndi.binary_fill_holes(surf | struct) & (valid > 0) & ~outside
    lbl, n = ndi.label(surf)
    if n:
        sizes = ndi.sum(surf, lbl, range(1, n + 1))
        surf = np.isin(lbl, 1 + np.flatnonzero(sizes > 0.05 * sizes.max()))

    # shading of the metal (shadows of chimneys, darker panels) carried over to the snow, softened
    Lm = np.where(metal & ~struct, lab[..., 0], np.nan)
    med = np.nanmedian(Lm)
    Lf = np.where(np.isnan(Lm), med, Lm).astype(np.float32)
    w = (metal & ~struct).astype(np.float32)
    num = ndi.gaussian_filter(Lf * w, 0.25 * PX_PER_M)
    den = ndi.gaussian_filter(w, 0.25 * PX_PER_M) + 1e-3
    illum = np.clip((num / den) / med, 0.72, 1.06)
    illum = 1 + 0.7 * (illum - 1)            # snow scatters more light into shadow than metal does
    return {"structure": struct & surf, "fence": fence & surf, "seams": seams & surf, "surface": surf,
            "illum": illum.astype(np.float32), "metal_lab": ref}


# ------------------------------------------------------------------ snow amount and redistribution
def mu_pitch(pitch_deg: float) -> float:
    """SP 20.13330.2016 App. B: 1 up to 30 deg, 0 from 60 deg, linear in between."""
    return float(np.clip((60 - pitch_deg) / 30, 0, 1))


def sample_weather(rng) -> dict:
    """Ground snow accumulated since the roof was last cleared, drawn around the 30 cm trigger."""
    k = rng.choice(3, p=[0.4, 0.35, 0.25])
    ground = [rng.uniform(10, 80), rng.uniform(80, 300), rng.uniform(300, 650)][k]
    return {"ground_mm": float(ground), "c_e": float(rng.uniform(0.7, 1.0)), "c_t": float(rng.uniform(0.8, 1.0)),
            "wind_u": float(rng.choice([-1, 1])), "leeward_slope": bool(rng.random() < 0.5),
            "wetness": float(rng.beta(1.5, 4)), "variation": float(rng.uniform(0.1, 0.3))}


def triangular_drift(dist_m, length_m, peak_mm):
    return np.clip(1 - dist_m / max(length_m, 1e-3), 0, 1) * peak_mm


def accumulate(Hg, Wg, layers, weather, pitch_deg, rng) -> tuple[np.ndarray, dict]:
    """Balanced load x unbalance between slopes + fence and obstruction drifts, mass-conserving."""
    P = PX_PER_M
    surf, struct = layers["surface"], layers["structure"]
    mu = mu_pitch(pitch_deg)
    unbalance = 1.0
    if 15 <= pitch_deg <= 40:                 # gable roofs: windward slope loses, leeward gains
        unbalance = 1.25 if weather["leeward_slope"] else 0.75
    base = weather["ground_mm"] * mu * weather["c_e"] * weather["c_t"] * unbalance
    D = base * (1 + weather["variation"] * fbm((Hg, Wg), 3.0 * P, 4, rng))
    D = np.clip(D, 0, None) * surf * ~struct
    total0 = float(D.sum())
    info = {"mu": round(mu, 3), "unbalance": unbalance, "base_mm": round(base, 1)}

    # fence / snow guards above the eave hold creeping snow: triangle 0.6-1.5 m up from the eave
    yy = np.arange(Hg)[:, None].astype(np.float32)
    up_m = (Hg - 1 - yy) / P
    l_f = rng.uniform(0.6, 1.5)
    k_f = rng.uniform(0.2, 0.7)
    drift = triangular_drift(up_m, l_f, k_f * base) * np.ones((1, Wg), np.float32)

    # obstruction drifts on the lee side along the eave direction (wind_u), length 2h, mu_total <= 2.4
    lbl, n = ndi.label(struct)
    n_obs = 0
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lbl == i)
        if len(ys) < 0.1 * P * P:
            continue
        n_obs += 1
        h_obs = rng.uniform(0.6, 2.0)
        L = min(2 * h_obs, 5.0)
        peak = base * min(rng.uniform(0.5, 1.4), 2.4 - 1.0)
        edge_x = xs.max() if weather["wind_u"] > 0 else xs.min()
        dx_m = (np.arange(Wg)[None, :] - edge_x) * weather["wind_u"] / P
        lat_c, lat_w = ys.mean(), (ys.max() - ys.min()) / 2 + 0.4 * P
        lat = np.exp(-((np.arange(Hg)[:, None] - lat_c) / lat_w) ** 4)
        drift += np.where(dx_m >= 0, triangular_drift(dx_m, L, peak), 0) * lat
    drift *= surf * ~struct
    # the drifted volume comes from the upwind half of the roof (scour), so the total is conserved
    V = float(drift.sum())
    ux = np.arange(Wg)[None, :] / max(Wg - 1, 1)
    upwind = (ux < 0.5) if weather["wind_u"] > 0 else (ux > 0.5)
    src = D * (upwind & ~(drift > 0))
    if src.sum() > 0 and V > 0:
        frac = min(V / src.sum(), 0.6)
        D = D - src * frac + drift * (frac * src.sum() / V)
    info.update({"fence_drift_m": round(l_f, 2), "obstruction_drifts": n_obs,
                 "mass_ratio": round(float(D.sum()) / max(total0, 1e-6), 3)})
    return np.clip(D, 0, None).astype(np.float32), info


def residual(H, W, rng, hi=None):
    """What a shovel leaves on metal: a patchy film 0..hi mm, streaky along the slope."""
    hi = hi if hi is not None else rng.uniform(1, 8)
    f = np.clip(0.5 + 0.35 * fbm((H, W), 0.6 * PX_PER_M, 4, rng, aniso=(0.5, 2.0)), 0, 1)
    return (hi * f ** 2).astype(np.float32)


def target_melt(D, vis, tau, target, noise, iters=30):
    if not vis.any():
        return D
    f = 1 + 0.5 * np.clip(noise, -1.5, 1.5)
    lo, hi = 0.0, float(np.percentile(D[vis], 99.5)) * 2 + 1
    for _ in range(iters):
        Mm = 0.5 * (lo + hi)
        if ((D - Mm * f)[vis] >= tau).mean() > target:
            lo = Mm
        else:
            hi = Mm
    return np.clip(D - 0.5 * (lo + hi) * f, 0, None).astype(np.float32)


def apply_state(D, state, layers, weather, vis, rng, tau) -> tuple[np.ndarray, np.ndarray, dict]:
    P = PX_PER_M
    Hg, Wg = D.shape
    struct = layers["structure"]
    wet = np.zeros_like(D)
    info = {}
    if state == "partially_cleared":
        how = rng.choice(["strips", "eave_band"])
        if how == "strips":
            m = np.zeros((Hg, Wg), bool)
            x = rng.uniform(0, 2) * P
            while x < Wg:
                w = rng.uniform(1.0, 3.0) * P
                if rng.random() < 0.65:
                    m[:, int(x):int(x + w)] = True
                x += w + rng.uniform(0.3, 2.5) * P
            m &= np.arange(Hg)[:, None] > rng.uniform(0, 0.5) * Hg
        else:
            m = (np.arange(Hg)[:, None] > Hg - rng.uniform(0.8, 2.2) * P) & np.ones((1, Wg), bool)
        m = warp_mask(m, rng, 0.2 * P, 2 * P)
        D = np.where(m, residual(Hg, Wg, rng), D)
        info["clearing"] = how
    elif state == "cleared":
        D = residual(Hg, Wg, rng, hi=rng.uniform(0.5, 8))
        leftovers = fbm((Hg, Wg), 1.2 * P, 4, rng) > rng.uniform(1.4, 2.4)       # missed lumps
        D += leftovers * rng.uniform(10, 80) * soft_mask(leftovers, 0.15 * P)
        info["clearing"] = "full"
    elif state == "thaw":
        melt_m = rng.uniform(0.4, 2.0)
        ridge = np.arange(Hg)[:, None] < melt_m * P + 0.4 * P * fbm((Hg, Wg), 2 * P, 3, rng)
        warm = ndi.distance_transform_edt(~struct) < rng.uniform(0.2, 0.8) * P
        s = soft_mask(ridge | warm, 0.25 * P)
        D = D * (1 - s)
        target = float(rng.uniform(0.05, 0.95))
        D = target_melt(D, vis, tau, target, fbm((Hg, Wg), 2.0 * P, 3, rng))
        edge = (D > 0) & (D < 3)
        wet = np.clip(ndi.gaussian_filter(edge.astype(np.float32), 0.3 * P) * 2.5, 0, 1)
        info.update({"thaw_target_coverage": round(target, 3)})
    if state in ("not_cleared", "partially_cleared") and rng.random() < 0.3:
        y0 = Hg - rng.uniform(0.6, 1.4) * P
        path = np.abs(np.arange(Hg)[:, None] - y0 - 0.2 * P * fbm((Hg, Wg), 4 * P, 2, rng)) < rng.uniform(0.25, 0.45) * P
        steps = fbm((Hg, Wg), 0.15 * P, 2, rng) > 0.3
        D = np.where(path, D * rng.uniform(0.3, 0.6) * (1 - 0.4 * steps), D)
        info["trodden_path"] = True
    return D.astype(np.float32), wet.astype(np.float32), info


def roughen(D, layers, state, rng):
    """cm-scale unevenness and crumbly edges (thin snow breaks up, deep snow barely changes),
    snow kept in the grooves along the seams after clearing or melt."""
    P = PX_PER_M
    Hg, Wg = D.shape
    ex = exemplar((Hg, Wg), rng, rng.uniform(3, 8) * P)
    rough = 0.6 * fbm((Hg, Wg), 0.15 * P, 3, rng) + 0.4 * ex
    D = D + np.clip(D, 0, 30) * rng.uniform(0.25, 0.5) * rough + np.clip(D, 0, 400) * 0.03 * ex
    if state in ("cleared", "partially_cleared", "thaw"):
        groove = ndi.binary_dilation(layers["seams"], np.ones((1, 3))) ^ layers["seams"]
        groove = ndi.gaussian_filter(groove.astype(np.float32), 0.6) * (fbm((Hg, Wg), 0.4 * P, 2, rng) > -0.3)
        D = D + groove * rng.uniform(2, 10)
    D = np.clip(D, 0, None) * layers["surface"] * ~layers["structure"]
    return D.astype(np.float32)


# ------------------------------------------------------------------ optics and appearance
def opacity(D, beta):
    """Two-flux (Kubelka-Munk) whiteness of a layer over dark metal, relative to deep snow."""
    bd = beta * np.clip(D, 0, None)
    return (bd / (1 + bd)).astype(np.float32)


def snow_radiance(img, p: hostmatch.HostParams, metal_bgr, rng) -> tuple[np.ndarray, dict]:
    """Snow colour before haze. Under overcast a snow surface sees most of the sky and can be brighter
    than the sky near the horizon (CIE overcast: zenith ~3x horizon); camera clipping caps it."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lum = img.mean(2)
    sel = (hsv[..., 1] < 35) & (lum >= np.percentile(lum, 95))
    pix = img[sel] if sel.sum() > 100 else img.reshape(-1, 3)
    white = np.percentile(pix, 90, axis=0).astype(np.float32)
    white = white.mean() * (0.85 + 0.15 * white / white.mean())
    r = rng.uniform(0.85, 1.15)
    snow = white * r * np.array([1.03, 1.0, 0.97]) * rng.uniform(0.985, 1.015, 3)
    lum_m = float(np.mean(metal_bgr))
    if snow.mean() < 1.3 * lum_m:
        snow *= 1.3 * lum_m / max(float(snow.mean()), 1)
    return snow.astype(np.float32), {"snow_vs_white": round(float(r), 3)}


def shade_field(D, layers, rng):
    P = PX_PER_M
    Dm = ndi.gaussian_filter(D, 0.08 * P) / 1000.0
    gy, gx = np.gradient(Dm * P)
    shade = 1 - 0.10 * np.clip(gy * 3, -1, 1) - 0.05 * np.clip(gx * rng.uniform(-3, 3), -1, 1)
    ao = np.clip((Dm - ndi.gaussian_filter(Dm, 0.5 * P)) * 6, -0.08, 0.03)
    tex = exemplar(D.shape, rng, rng.uniform(2.5, 7) * P)
    grain = rng.uniform(0.015, 0.045) * tex + 0.008 * fbm(D.shape, 0.05 * P, 2, rng)
    if rng.random() < 0.35:
        specks = (fbm(D.shape, 0.06 * P, 2, rng) > rng.uniform(2.6, 3.2)).astype(np.float32)
        grain -= 0.08 * ndi.gaussian_filter(specks, 0.7)
    # seams print through only while the layer is about as thin as the seam is tall
    drape = ndi.gaussian_filter(layers["seams"].astype(np.float32), 0.04 * P) * np.exp(-np.clip(D - 5, 0, None) / 15.0)
    s = (shade + ao + grain - 0.06 * drape) * layers["illum"]
    return np.clip(s, 0.6, 1.08).astype(np.float32)


def photo_field_D(shape, mask, base_mm, rng, scale_px=80):
    """Depth on surfaces outside the slope grid (neighbour roofs), in photo space."""
    D = base_mm * (1 + 0.3 * fbm(shape, scale_px, 4, rng))
    patchy = fbm(shape, scale_px * 0.8, 3, rng) < rng.uniform(-2.5, 0.3)    # partly cleared / melted
    D = np.where(patchy, residual(*shape, rng, hi=rng.uniform(0, 5)), D)
    m = warp_mask(mask, rng, 2.5, 25) & mask
    return (np.clip(D, 0, None) * soft_mask(m, 1.2)).astype(np.float32)


def neighbour_from_polygons(img, polys, metal_lab):
    H, W = img.shape[:2]
    m = np.zeros((H, W), np.uint8)
    for p in polys:
        cv2.fillPoly(m, [np.round(p).astype(np.int32)], 1)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    roofish = (hsv[..., 1] < 70) & (np.linalg.norm(lab - metal_lab, axis=2) < 40)
    k = ndi.binary_opening((m > 0) & roofish, np.ones((3, 3)))
    k = ndi.binary_closing(k, np.ones((9, 9)))
    k = cv2.medianBlur(k.astype(np.uint8) * 255, 7) > 0
    k = ndi.binary_fill_holes(k)
    k = ndi.binary_opening(k, np.ones((5, 5)))          # wires, gutters and ridge lines are not roof area
    return k & (m > 0)


def winter_grade(img_f, rng):
    """Autumn foliage toward winter: desaturate and darken green / yellow / orange vegetation."""
    hsv = cv2.cvtColor(np.clip(img_f, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    veg = (hsv[..., 0] > 12) & (hsv[..., 0] < 95) & (hsv[..., 1] > 70)
    veg = cv2.GaussianBlur(veg.astype(np.float32), (0, 0), 2)
    k = rng.uniform(0.5, 0.85)
    hsv[..., 1] *= 1 - k * veg
    hsv[..., 2] *= 1 - rng.uniform(0.1, 0.3) * veg
    out = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    return out


# ------------------------------------------------------------------ photo-space volume
def eave_slab(geo: SlopeGeom, D_eave_mm: np.ndarray, shape, scale):
    """Vertical face of the snow layer at the eave, seen from above: a band below the eave line whose
    height in px is the depth there times the photo scale. Returns depth-like map (mm) of the face."""
    H0, W0 = shape
    q = geo.quad
    a, b = q[0].astype(np.float64), q[1].astype(np.float64)
    d = b - a
    L2 = float(d @ d)
    maxh = int(np.ceil(D_eave_mm.max() / 1000 * scale)) + 2
    if maxh < 2:
        return np.zeros(shape, np.float32)
    x0, x1 = int(max(0, min(a[0], b[0]) - 2)), int(min(W0, max(a[0], b[0]) + 2))
    y0, y1 = int(max(0, min(a[1], b[1]) - 2)), int(min(H0, max(a[1], b[1]) + maxh + 2))
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
    t = ((xx - a[0]) * d[0] + (yy - a[1]) * d[1]) / L2
    foot_y = a[1] + t * d[1]
    dz = (yy - foot_y) / scale * 1000                     # mm below the eave line
    u = np.clip(t, 0, 1) * (len(D_eave_mm) - 1)
    De = np.interp(u.ravel(), np.arange(len(D_eave_mm)), D_eave_mm).reshape(u.shape)
    face = (t >= 0) & (t <= 1) & (dz >= 0) & (dz < De)
    out = np.zeros(shape, np.float32)
    out[y0:y1, x0:x1] = np.where(face, De, 0)
    return out


def caps(mask: np.ndarray, thick_px: float, rails_only=False, rng=None):
    """Snow resting on the top edges of structures (chimneys, dishes) or of the railing's top rail.
    Structure caps are mounds: thickest in the middle of the top edge, rounded toward the ends."""
    m = (mask > 0).astype(np.uint8)
    if rails_only:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((1, 15), np.uint8))   # horizontal rails only
        n, lab, st, _ = cv2.connectedComponentsWithStats(m)
        m = np.isin(lab, 1 + np.flatnonzero(st[1:, cv2.CC_STAT_WIDTH] >= 40)).astype(np.uint8)
    top = (m > 0) & (np.roll(m, 1, axis=0) == 0)
    top[0] = False
    out = np.zeros(m.shape, bool)
    t = max(1.0, float(thick_px))
    n, lab, st, _ = cv2.connectedComponentsWithStats(top.astype(np.uint8), connectivity=8)
    for i in range(1, n):
        ys, xs = np.nonzero(lab == i)
        if len(xs) < 3:
            continue
        xc, hw = (xs.min() + xs.max()) / 2, (xs.max() - xs.min()) / 2 + 1
        jit = 1.0 if rng is None else rng.uniform(0.8, 1.15)
        for x, y in zip(xs, ys):
            prof = 1.0 if rails_only else np.sqrt(max(0.0, 1 - ((x - xc) / hw) ** 2)) ** 0.6
            h = int(round(t * prof * jit))
            if h > 0:
                out[max(0, y - h):y, x] = True
    return out & (mask == 0)


# ------------------------------------------------------------------ main entry
@dataclass
class Sample:
    image: np.ndarray
    depth_mm: np.ndarray        # float32, photo px, all snow-bearing surfaces
    classes: np.ndarray         # uint8, schema.SEG_CLASSES index
    roi: np.ndarray             # bool, visible target slope (where coverage is computed)
    coverage: dict              # threshold mm -> share of the visible target slope
    info: dict


def make_sample(img: np.ndarray, quad: np.ndarray, roof_poly: np.ndarray, rng, geom: dict,
                host: hostmatch.HostParams | None = None, other_roofs: list | None = None, structures: list | None = None,
                state: str | None = None, label_tau_mm: float = 2.0, null: bool = False) -> Sample:
    """null=True runs the whole compositing chain with zero snow (shortcut test, see tests/validate.py)."""
    H0, W0 = img.shape[:2]
    ov = overlays.detect(img)
    if host is None:
        q = np.asarray(quad)
        host = hostmatch.estimate(img, (q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()), ignore=ov.mask)
    geo = slope_geometry(quad, geom)
    Hg, Wg = geo.H, geo.W
    rect = cv2.warpPerspective(img, geo.M, (Wg, Hg), flags=cv2.INTER_LINEAR)
    valid = cv2.warpPerspective(np.full((H0, W0), 255, np.uint8), geo.M, (Wg, Hg), flags=cv2.INTER_NEAREST)
    known = cv2.warpPerspective((ov.mask == 0).astype(np.uint8) * 255, geo.M, (Wg, Hg), flags=cv2.INTER_NEAREST)
    struct_ph = np.zeros((H0, W0), np.uint8)
    for sp in structures or []:
        cv2.fillPoly(struct_ph, [np.round(np.asarray(sp)).astype(np.int32)], 1)
    struct_known = cv2.warpPerspective(struct_ph, geo.M, (Wg, Hg), flags=cv2.INTER_NEAREST)
    layers = surface_layers(rect, valid, known, struct_known)
    vis = (known > 0) & (valid > 0) & ~layers["structure"] & layers["surface"]

    weather = sample_weather(rng)
    state = state or str(rng.choice(STATES, p=STATE_P))
    D, acc_info = accumulate(Hg, Wg, layers, weather, geom["pitch_deg"], rng)
    D, wet, st_info = apply_state(D, state, layers, weather, vis, rng, label_tau_mm)
    D = roughen(D, layers, state, rng)
    if null:
        D[:] = 0
        wet[:] = 0
    beta = rng.uniform(0.35, 0.6) + 0.4 * weather["wetness"]       # wetter snow whitens faster
    alpha_g = opacity(np.where(layers["seams"], np.clip(D - SEAM_HEIGHT_MM, 0, None), D), beta)
    shade_g = shade_field(D, layers, rng)

    # cornice lip beyond the eave on the leeward side of deep snow
    ext = 0
    if not null and state in ("not_cleared", "partially_cleared") and weather["leeward_slope"] \
            and acc_info["base_mm"] > 120 and rng.random() < 0.5:
        ext = int(rng.uniform(0.15, 0.45) * PX_PER_M)
        lip = np.repeat(D[-1:, :], ext, 0) * np.clip(np.linspace(1, 0, ext)[:, None] ** rng.uniform(0.6, 1.5)
                                                     + 0.25 * fbm((ext, Wg), 0.3 * PX_PER_M, 3, rng), 0, 1)
        lip[lip < 30] = 0
        D = np.vstack([D, lip.astype(np.float32)])
        alpha_g = np.vstack([alpha_g, opacity(lip, beta)])
        shade_g = np.vstack([shade_g, np.full((ext, Wg), 0.9, np.float32)])
        wet = np.vstack([wet, np.zeros((ext, Wg), np.float32)])
    Hx = D.shape[0]

    to_photo = lambda a, interp=cv2.INTER_LINEAR, bv=0.0: cv2.warpPerspective(
        a.astype(np.float32), geo.Minv, (W0, H0), flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=bv)
    surf_x = np.vstack([layers["surface"], D[Hg:] > 0]) if ext else layers["surface"]
    on_slope = to_photo(surf_x, cv2.INTER_LINEAR) > 0.5
    A = to_photo(alpha_g)
    S = to_photo(shade_g, bv=1.0)
    Wd = to_photo(1 - 0.35 * wet, bv=1.0)
    Dp = to_photo(D)
    struct_p = (to_photo(layers["structure"], cv2.INTER_LINEAR) > 0.5) | (struct_ph > 0)
    fence_p = to_photo(layers["fence"], cv2.INTER_LINEAR) > 0.3
    below_eave = _below_line(geo.quad[0], geo.quad[1], (H0, W0))
    lip_p = (to_photo(np.vstack([np.zeros((Hg, Wg)), np.ones((ext, Wg))]) if ext else np.zeros((Hx, Wg)),
                      cv2.INTER_NEAREST) > 0.5) if ext else np.zeros((H0, W0), bool)

    metal_bgr = cv2.cvtColor(np.uint8([[layers["metal_lab"]]]), cv2.COLOR_LAB2BGR)[0, 0].astype(np.float32)
    snow, sn_info = snow_radiance(img, host, metal_bgr, rng)
    base_mm = acc_info["base_mm"] if state != "cleared" else 0.0

    # ---- volume in photo space
    scale = geo.px_per_m_photo * rng.uniform(0.9, 1.1)          # photo px per metre, vertical ~ horizontal
    D_eave = D[Hg - int(0.3 * PX_PER_M):Hg].max(0) if not null else np.zeros(Wg, np.float32)
    face_D = eave_slab(geo, ndi.uniform_filter1d(D_eave, 9), (H0, W0), scale) * ~on_slope
    face = face_D > 0
    cap_struct = caps(struct_ph > 0 if structures else struct_p, min(base_mm, 400) / 1000 * scale * rng.uniform(0.5, 0.9), rng=rng) if base_mm > 20 and not null \
        else np.zeros((H0, W0), bool)
    cap_rail = caps(fence_p, min(base_mm, 60) / 1000 * scale, rails_only=True) if base_mm > 20 and not null \
        else np.zeros((H0, W0), bool)

    # ---- neighbouring roofs and the photographer's own roof: same weather, own clearing
    neigh = np.zeros((H0, W0), bool)
    Dn = np.zeros((H0, W0), np.float32)
    if other_roofs:
        neigh = neighbour_from_polygons(img, other_roofs, layers["metal_lab"]) & (ov.mask == 0) & ~on_slope
        if neigh.any() and not null:
            nbase = weather["ground_mm"] * rng.uniform(0.5, 1.1) * (0.15 if rng.random() < 0.25 else 1.0)
            Dn = photo_field_D((H0, W0), neigh, nbase, rng)

    # ---- composite: background (exposure + winter grade), then premultiplied, blurred snow layers
    img_f = img.astype(np.float32)
    frac = float((A * on_slope).mean() + opacity(Dn, beta).mean())
    e = 1 - min(0.15, 1.5 * frac) if frac > 0.02 and not null else 1.0
    bg = img_f * e
    graded = False
    if not null and weather["ground_mm"] > 30 and rng.random() < 0.7:
        bg = winter_grade(bg, rng)
        graded = True
    bg = bg * Wd[..., None]

    snow_hz = hostmatch.apply_haze(snow[None, None, :], host)                    # (1,1,3)
    tex_p = 1 + 0.03 * fbm((H0, W0), 6, 3, rng)
    alpha = A * on_slope
    col = snow_hz * (S * tex_p)[..., None]
    a_n = opacity(Dn, beta) * neigh
    alpha = np.maximum(alpha, a_n)
    face_a = face.astype(np.float32) * (0.75 + 0.1 * fbm((H0, W0), 4, 2, rng))
    col = np.where(face[..., None], snow_hz * (0.78 * tex_p)[..., None], col)        # vertical face, half sky
    alpha = np.maximum(alpha, face_a)
    capm = (cap_struct | cap_rail).astype(np.float32)
    col = np.where(capm[..., None] > 0, snow_hz * 0.97, col)
    alpha = np.maximum(alpha, capm)
    keep_front = (fence_p | struct_p) & ~(cap_rail | cap_struct)
    alpha = np.where(keep_front, 0, alpha)
    pm, ab = hostmatch.blur_layer(col * alpha[..., None], alpha.astype(np.float32), host)
    out = bg * (1 - ab[..., None]) + pm
    weight = cv2.GaussianBlur((ab > 0.02).astype(np.float32), (0, 0), 2.0)
    out = hostmatch.finish(out, weight, host, rng)
    out[ov.mask > 0] = img_f[ov.mask > 0]                         # the app stamps after processing
    ok, enc = cv2.imencode(".jpg", out.astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, int(host.jpeg_quality)])
    out = cv2.imdecode(enc, 1)

    # ---- labels (location-agnostic: snow is snow anywhere; the target slope is the ROI)
    depth = np.zeros((H0, W0), np.float32)
    depth[on_slope] = Dp[on_slope]
    depth[neigh] = np.maximum(depth[neigh], Dn[neigh])
    depth[face] = np.maximum(depth[face], face_D[face])
    depth[cap_struct | cap_rail] = np.maximum(depth[cap_struct | cap_rail], max(base_mm, label_tau_mm + 2))
    depth[struct_p & ~cap_struct] = 0
    cls = np.zeros((H0, W0), np.uint8)
    roofish = on_slope | neigh
    cls[roofish] = CLS["metal_clean"]
    cls[roofish & (Wd < 0.9)] = CLS["metal_wet"]
    snowy = depth >= label_tau_mm
    cls[snowy] = CLS["snow"]
    cls[lip_p & snowy] = CLS["overhang"]
    cls[struct_p & ~cap_struct] = CLS["structure"]
    cls[fence_p & on_slope & ~cap_rail] = CLS["fence"]
    cls[ov.mask > 0] = CLS["ignore"]
    roi = on_slope & ~below_eave & ~struct_p & (ov.mask == 0)

    # coverage on the target slope by area: grid pixels are equal areas
    Dg = D[:Hg]
    cov = {t: float((Dg[vis] >= t).mean()) if vis.any() else 0.0 for t in THRESHOLDS_MM}
    info = {"state": state, **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in weather.items()},
            **acc_info, **st_info, **sn_info, "pitch_deg": geom["pitch_deg"], "geometry_source": geom.get("source"),
            "slope_length_m": geo.length_m, "slope_depth_m": geo.depth_m, "beta_per_mm": round(float(beta), 3),
            "mean_depth_mm": round(float(Dg[vis].mean()), 1) if vis.any() else 0.0,
            "p95_depth_mm": round(float(np.percentile(Dg[vis], 95)), 1) if vis.any() else 0.0,
            "visible_slope_frac": round(float(vis.mean()), 3), "cornice_px": ext,
            "eave_face_px": int(face.sum()), "cap_px": int((cap_struct | cap_rail).sum()),
            "neighbour_frac": round(float(neigh.mean()), 3), "neighbour_snow_frac": round(float((Dn >= label_tau_mm).mean()), 3),
            "winter_grade": graded, "exposure": round(e, 3), "label_tau_mm": label_tau_mm, "null": null,
            "host": host.as_dict()}
    return Sample(out, depth, cls, roi, cov, info)


def labels_at(depth: np.ndarray, cls: np.ndarray, threshold_mm: float, band_mm: float = IGNORE_BAND_MM) -> np.ndarray:
    """Binary snow target at a client threshold: 1 snow, 0 not, 255 ignore (overlays, structures' bodies
    and the +-band around the threshold, where a pixel is physically ambiguous)."""
    y = (depth >= threshold_mm).astype(np.uint8)
    y[np.abs(depth - threshold_mm) <= band_mm] = 255
    y[cls == CLS["ignore"]] = 255
    return y


def _below_line(a, b, shape):
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W]
    return ((b[0] - a[0]) * (yy - a[1]) - (b[1] - a[1]) * (xx - a[0])) > 0
