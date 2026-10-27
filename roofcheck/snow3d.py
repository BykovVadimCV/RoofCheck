"""Snow as a 3D solid on the roof, rendered by casting a ray through every photo pixel.

1. Roof plane in camera space: plane fit to the depth model's 3D points inside the slope quad, using the
   camera intrinsics estimated by the same model; the scene is rescaled so the eave has its known length.
   A grid cell's base point is where the camera ray through its photo position meets that plane, so the
   3D frame agrees exactly with the photo <-> grid homography used everywhere else.
2. Snow solid: height above the plane (metres) on the slope grid. Every edge of the snow cover gets a
   rounded profile whose radius follows the local depth and the snow's freshness (fresh: puffy, settled:
   flatter, wet: nearly flat); shovel cuts stay sharp; snow banks up against chimneys.
3. Rendering: for each photo pixel the ray is marched from above the highest snow down to the roof; the
   first point inside the solid is what the camera sees. Occlusion, visible walls, rounded noses at the
   eave and the overhang all follow from the geometry. Lighting uses the 3D surface normal against the
   true up direction: overcast sky term x local sky visibility + a weak directional term.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage as ndi

PX_PER_M = 40


@dataclass
class RoofFrame:
    K: np.ndarray
    Kinv: np.ndarray
    n: np.ndarray            # unit plane normal, pointing to the camera side
    d: float                 # plane: n . P = d (scaled scene)
    M: np.ndarray            # photo -> grid homography
    Minv: np.ndarray
    ex: np.ndarray           # unit vector along the eave (grid +u)
    ey: np.ndarray           # unit vector ridge -> eave (grid +v)
    m_per_u: float           # metres per grid px along u
    m_per_v: float           # metres per grid px along v
    pitch_deg: float
    fit_rms: float
    scale: float = 1.0       # model units -> metres


def _base(Kinv, n, d, px, py):
    r = Kinv @ np.vstack([px, py, np.ones_like(px)])
    t = d / (n @ r)
    return r * t


def roof_frame(K, depth, valid, quad, M, Hg, Wg, length_m, up, normal=None, ignore=None, rng=None) -> RoofFrame | None:
    """Metric 3D frame of the slope. The plane's orientation comes from the surface normals (robust on
    roofs: they gave the pitch estimates), its offset from the depth of pixels that agree with it; the
    scale is then fixed by the eave's known length (projection does not depend on scale)."""
    rng = rng or np.random.default_rng(0)
    H0, W0 = depth.shape
    q = np.asarray(quad, np.float64)
    m = np.zeros((H0, W0), np.uint8)
    cv2.fillPoly(m, [np.round(q).astype(np.int32)], 1)
    sel = (m > 0) & valid & (depth > 0)
    if ignore is not None:
        sel &= ~ignore
    if normal is None or sel.sum() < 500:
        return None
    N = normal[sel].astype(np.float64)
    N /= np.linalg.norm(N, axis=1, keepdims=True) + 1e-9
    tilt = np.degrees(np.arccos(np.clip(N @ up, -1, 1)))
    roofish = (tilt > 5) & (tilt < 60)
    if roofish.sum() < 300:
        return None
    n = np.median(N[roofish], axis=0)
    for _ in range(3):                                        # trimmed mean around the mode
        n /= np.linalg.norm(n)
        agree = N @ n > np.cos(np.radians(12))
        if agree.sum() < 200:
            return None
        n = N[agree].mean(0)
    n /= np.linalg.norm(n)
    ys, xs = np.nonzero(sel)
    ys, xs = ys[agree], xs[agree]
    Kinv = np.linalg.inv(K)
    P = ((Kinv @ np.vstack([xs, ys, np.ones_like(xs)])) * depth[ys, xs]).T
    if n @ (-np.median(P, 0)) < 0:                            # normal toward the camera
        n = -n
    dd = P @ n
    d = float(np.median(dd))
    rms = float(np.median(np.abs(dd - d)) / np.median(P[:, 2]))
    B = _base(Kinv, n, d, q[:, 0], q[:, 1])                   # 3 x 4: eave-l, eave-r, ridge-r, ridge-l
    eave = np.linalg.norm(B[:, 1] - B[:, 0])
    if not np.isfinite(eave) or eave <= 0:
        return None
    s = length_m / eave
    d *= s
    B *= s
    ex = (B[:, 1] - B[:, 0]); ex /= np.linalg.norm(ex)
    ey = (B[:, 0] - B[:, 3]); ey /= np.linalg.norm(ey)
    m_u = 0.5 * (np.linalg.norm(B[:, 1] - B[:, 0]) + np.linalg.norm(B[:, 2] - B[:, 3])) / Wg
    m_v = 0.5 * (np.linalg.norm(B[:, 0] - B[:, 3]) + np.linalg.norm(B[:, 1] - B[:, 2])) / Hg
    pitch = float(np.degrees(np.arccos(np.clip(abs(n @ up), -1, 1))))
    if not (3 < pitch < 70):
        return None
    return RoofFrame(K, Kinv, n, d, M, np.linalg.inv(M), ex, ey, float(m_u), float(m_v), pitch, rms, float(s))


# ------------------------------------------------------------------ the snow solid
def _fbm(shape, scale_px, octaves, rng):
    H, W = shape
    out = np.zeros(shape, np.float32); amp = 1.0; tot = 0.0
    for o in range(octaves):
        s = scale_px / 2 ** o
        if s < 2:
            break
        g = rng.standard_normal((int(H / s) + 3, int(W / s) + 3)).astype(np.float32)
        out += amp * cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)[:H, :W]
        tot += amp; amp *= 0.5
    out /= max(tot, 1e-6)
    return (out - out.mean()) / (out.std() + 1e-6)


def smooth_relief(D_mm, seams, freshness, rng, P=PX_PER_M):
    """Top surface of the snow body, mm. Seen from tens of metres, a snow cover is smooth: the body follows
    the depth field low-passed at ~0.3-0.5 m (wind and settling round everything off), plus broad, low
    swells (1.5-5 m, a few mm to 2 cm high) and, only while the layer is thin, the seams printing through.
    No centimetre-scale relief: lit at a grazing angle it reads as crumpled paper."""
    body = ndi.gaussian_filter(D_mm, rng.uniform(0.3, 0.5) * P)
    thin_mix = np.clip(D_mm / 40.0, 0, 1)                      # thin snow keeps its own shape (clearing marks)
    base = D_mm * (1 - thin_mix) + body * thin_mix
    swell = _fbm(D_mm.shape, rng.uniform(1.5, 5.0) * P, 2, rng)
    amp = np.clip(body * {"fresh": 0.04, "settled": 0.03, "wet": 0.02}[freshness], 0, 20)
    ridge = ndi.gaussian_filter(seams.astype(np.float32), 0.06 * P)
    ridge = ridge / (ridge.max() + 1e-6) * 25.0 * np.exp(-body / 60.0) * (D_mm > 1)
    return np.clip(base + swell * amp + ridge, 0, None).astype(np.float32)


def solid_height(D_mm, cut, near_struct, freshness, rng, relief_mm=None, P=PX_PER_M, continues=()):
    """Height of the snow surface above the roof plane, metres, with rounded edges.

    The profile at a free edge is a quarter ellipse of radius R = k x local depth (k: fresh 1.2-2.0,
    settled 0.5-1.0, wet 0.2-0.45), i.e. fresh snow ends in a soft rounded nose, settled snow in a
    shallower one. At shovel cuts R is ~0.15 x depth (near-vertical wall with a softened top). Against
    structures snow banks up (a fillet) instead of rounding away. `continues`: grid sides where the snow
    does not end (e.g. "top" for the ridge: the cover carries on over it to the other slope)."""
    k = {"fresh": rng.uniform(1.2, 2.0), "settled": rng.uniform(0.5, 1.0), "wet": rng.uniform(0.2, 0.45)}[freshness]
    snow = D_mm > 3
    pad = np.pad(snow, 1)
    if "top" in continues:
        pad[0, :] = True
    if "bottom" in continues:
        pad[-1, :] = True
    if "left" in continues:
        pad[:, 0] = True
    if "right" in continues:
        pad[:, -1] = True
    dist_m = ndi.distance_transform_edt(pad)[1:-1, 1:-1] / P
    Ds = ndi.gaussian_filter(D_mm, 0.08 * P)
    Dref = ndi.grey_dilation(Ds, size=int(0.3 * P) | 1)               # thickness of the body behind the edge
    R = np.maximum(k * Dref / 1000.0, 0.01)
    if cut is not None and cut.any():
        near_cut = ndi.distance_transform_edt(~cut) < 0.5 * P
        R = np.where(near_cut & ~cut, np.maximum(0.15 * Dref / 1000.0, 0.01), R)
    x = np.clip(dist_m / R, 0, 1)
    prof = np.sqrt(np.clip(1 - (1 - x) ** 2, 0, 1))
    base = (relief_mm if relief_mm is not None else D_mm) / 1000.0
    H = base * np.where(snow, prof, 1.0)
    if "top" in continues:            # ridge: a lower, rounded crest where wind scours the snow
        rows = np.arange(H.shape[0])[:, None] / P
        w = rng.uniform(0.3, 0.7)
        H = H * (rng.uniform(0.45, 0.7) + (1 - rng.uniform(0.45, 0.7)) * np.sqrt(np.clip(rows / w, 0, 1)))
    if near_struct is not None and near_struct.any():                  # bank against chimneys and walls
        ds = ndi.distance_transform_edt(~near_struct) / P
        bank = np.exp(-ds / rng.uniform(0.1, 0.25)) * rng.uniform(0.15, 0.45) * Dref / 1000.0
        near = (ds < 0.6) & snow                      # only around structures, on top of the smooth body
        H = np.where(near, np.maximum(H, base + bank * (Ds > 20)), H)
        H = np.where(near_struct, 0, H)
    return np.clip(H, 0, None).astype(np.float32), {"freshness": freshness, "round_k": round(float(k), 2)}


def _horizon(h_m, px_m, dirs=12, steps=(1, 2, 3, 4, 6, 8, 11, 15, 20, 27, 36, 48, 64)):
    H, W = h_m.shape
    pad = max(steps) + 1
    hp = np.pad(h_m, pad, mode="edge")
    vis = np.zeros((H, W), np.float32)
    for a in np.linspace(0, 2 * np.pi, dirs, endpoint=False):
        dx, dy = np.cos(a), np.sin(a)
        mx = np.zeros((H, W), np.float32)
        for st in steps:
            sx, sy = int(round(dx * st)), int(round(dy * st))
            sh = hp[pad + sy:pad + sy + H, pad + sx:pad + sx + W]
            mx = np.maximum(mx, (sh - h_m) / (np.hypot(sx, sy) * px_m + 1e-6))
        vis += 1 - np.sin(np.arctan(np.clip(mx, 0, None)))
    return cv2.GaussianBlur((vis / dirs).astype(np.float32), (0, 0), 2.0)      # no step-ghosts of tall objects


def shade3d(H_m, fr: RoofFrame, up, struct, rng, albedo=None, illum=None):
    """Irradiance of the snow surface relative to flat roof snow: overcast sky term from the 3D normal
    (CIE overcast: (1 + 2 cos)/3), visibility of the sky over the relief (structures occlude), and a weak
    light from the brighter part of the cloud."""
    gv, gu = np.gradient(H_m, fr.m_per_v, fr.m_per_u)
    nrm = (-gu[..., None] * fr.ex[None, None, :] - gv[..., None] * fr.ey[None, None, :] + fr.n[None, None, :])
    nrm /= np.linalg.norm(nrm, axis=2, keepdims=True)
    cos_up = nrm @ up
    sky = (1 + 2 * np.clip(cos_up, -1, 1)) / 3
    h_ao = H_m + struct * 1.5
    vis = _horizon(h_ao, 0.5 * (fr.m_per_u + fr.m_per_v))
    horiz = np.cross(up, rng.normal(size=3)); horiz /= np.linalg.norm(horiz)
    el = np.radians(rng.uniform(25, 55))
    l = np.cos(el) * horiz + np.sin(el) * up
    direct = np.clip(nrm @ l, 0, None)
    kd = rng.uniform(0.05, 0.18)
    E = (1 - kd) * sky * (0.35 + 0.65 * vis) + kd * direct
    flat = (1 - kd) * (1 + 2 * float(fr.n @ up)) / 3 * (0.35 + 0.65 * float(np.median(vis))) + kd * max(float(fr.n @ l), 0)
    S = E / max(flat, 1e-3)
    # snow faces stay fairly bright: light scattered inside the snow and bounced off the snow around fills
    # walls and undersides (they read blue-grey, not dark grey)
    S = 1 - (1 - S) * 0.55
    if albedo is not None:
        S = S * albedo
    if illum is not None:
        S = S * illum
    return np.clip(S, 0.35, 1.2).astype(np.float32), nrm


# ------------------------------------------------------------------ rendering
def _samp(G, gu, gv, border="constant"):
    """Bilinear sample of a grid at float grid coords (u = column, v = row)."""
    return ndi.map_coordinates(G, [gv, gu], order=1, mode=border, cval=0.0, prefilter=False)


def raymarch(fr: RoofFrame, H_m, grids: dict, shape, bbox, step_m=0.012, refine=6):
    """For every photo pixel in bbox find the first point of the snow solid along its camera ray.
    grids: name -> grid array to sample at the hit (shade, alpha, depth...). Returns photo-space maps
    and the hit mask. Grid rows beyond the slope (overhang) are valid: the plane extends past the eave."""
    H0, W0 = shape
    x0, y0, x1, y1 = bbox
    ys, xs = np.mgrid[y0:y1, x0:x1]
    px, py = xs.ravel().astype(np.float64), ys.ravel().astype(np.float64)
    r = fr.Kinv @ np.vstack([px, py, np.ones_like(px)])
    nr = fr.n @ r
    hmax = float(H_m.max()) + 0.02
    if hmax <= 0.021:
        return None
    Hf = H_m.astype(np.float32)

    def sample(h):
        t = (fr.d + h) / nr
        P = r * t - np.outer(fr.n, np.full_like(t, h)) if np.ndim(h) == 0 else r * t - fr.n[:, None] * h[None, :]
        q = fr.K @ P
        qx, qy = q[0] / q[2], q[1] / q[2]
        g = fr.M @ np.vstack([qx, qy, np.ones_like(qx)])
        gu, gv = (g[0] / g[2]).astype(np.float32), (g[1] / g[2]).astype(np.float32)
        return _samp(Hf, gu, gv), gu, gv

    nsteps = int(np.ceil(hmax / step_m))
    levels = np.linspace(hmax, 0.0, nsteps + 1)
    found = np.zeros(px.size, bool)
    lo = np.zeros(px.size)
    hi = np.zeros(px.size)
    prev = np.full(px.size, hmax)
    for h in levels:
        Hs, _, _ = sample(h)
        hit = (Hs >= h) & (Hs > 0.0015) & ~found
        lo[hit], hi[hit] = h, prev[hit]
        found |= hit
        prev[:] = h
        if found.all():
            break
    idx = np.flatnonzero(found)
    if idx.size == 0:
        return None
    a, b = lo[idx], hi[idx]
    rr, nrr = r[:, idx], nr[idx]
    for _ in range(refine):                     # first crossing between lo (inside) and hi (outside)
        mid = 0.5 * (a + b)
        t = (fr.d + mid) / nrr
        P = rr * t - fr.n[:, None] * mid[None, :]
        q = fr.K @ P
        g = fr.M @ np.vstack([q[0] / q[2], q[1] / q[2], np.ones(idx.size)])
        gu, gv = (g[0] / g[2]).astype(np.float32), (g[1] / g[2]).astype(np.float32)
        Hs = _samp(Hf, gu, gv)
        inside = Hs >= mid
        a = np.where(inside, mid, a)
        b = np.where(inside, b, mid)
    t = (fr.d + a) / nrr
    P = rr * t - fr.n[:, None] * a[None, :]
    q = fr.K @ P
    g = fr.M @ np.vstack([q[0] / q[2], q[1] / q[2], np.ones(idx.size)])
    gu, gv = (g[0] / g[2]).astype(np.float32), (g[1] / g[2]).astype(np.float32)
    out = {"hit": np.zeros((H0, W0), bool), "height_m": np.zeros((H0, W0), np.float32)}
    ph, pw = y1 - y0, x1 - x0
    full = np.zeros(ph * pw, bool); full[idx] = True
    out["hit"][y0:y1, x0:x1] = full.reshape(ph, pw)
    hv = np.zeros(ph * pw, np.float32); hv[idx] = a
    out["height_m"][y0:y1, x0:x1] = hv.reshape(ph, pw)
    for name, G in grids.items():
        G = np.asarray(G, np.float32)
        if G.ndim == 2:
            vals = _samp(G, gu, gv, "nearest")
            m = np.zeros(ph * pw, np.float32); m[idx] = vals
            arr = np.zeros((H0, W0), np.float32); arr[y0:y1, x0:x1] = m.reshape(ph, pw)
        else:
            arr = np.zeros((H0, W0, G.shape[2]), np.float32)
            for c in range(G.shape[2]):
                vals = _samp(np.ascontiguousarray(G[..., c]), gu, gv, "nearest")
                m = np.zeros(ph * pw, np.float32); m[idx] = vals
                arr[y0:y1, x0:x1, c] = m.reshape(ph, pw)
        out[name] = arr
    return out


def render_bbox(fr: RoofFrame, quad, H_m, shape, margin_px=8):
    """Photo-space box that can contain snow: the quad (with the overhang) lifted by the highest snow."""
    H0, W0 = shape
    q = np.asarray(quad, np.float64)
    hmax = float(H_m.max()) + 0.05
    pts = [q]
    for h in (hmax,):
        B = _base(fr.Kinv, fr.n, fr.d, q[:, 0], q[:, 1]) + fr.n[:, None] * h
        pr = fr.K @ B
        pts.append((pr[:2] / pr[2]).T)
    # overhang rows beyond the eave
    Hx = H_m.shape[0]
    g = np.array([[0, Hx - 1, 1], [H_m.shape[1] - 1, Hx - 1, 1]], np.float64).T
    e = fr.Minv @ g
    pts.append((e[:2] / e[2]).T)
    allp = np.vstack(pts)
    x0, y0 = np.floor(allp.min(0)).astype(int) - margin_px
    x1, y1 = np.ceil(allp.max(0)).astype(int) + margin_px
    return max(0, x0), max(0, y0), min(W0, x1), min(H0, y1)


# ------------------------------------------------------------------ snow on the other roofs, in 3D
def surface_frames(scene, mask, scale, min_lift_px=3.0, D_mm=None, max_frames=12):
    """One plane per connected snow-bearing surface that is near and deep enough for its thickness to show
    (lift > min_lift_px). Plane from the surface normals and depth of its own pixels; grid on the plane at
    roughly photo resolution. Returns a list of (frame, component mask, grid shape)."""
    H0, W0 = mask.shape
    lbl, n = ndi.label(mask)
    if n == 0 or scene.K is None:
        return []
    Kinv = np.linalg.inv(scene.K)
    f = float(scene.K[0, 0])
    out = []
    sizes = ndi.sum(mask, lbl, range(1, n + 1))
    for i in np.argsort(-sizes)[:max_frames * 3]:
        comp = lbl == (i + 1)
        if comp.sum() < 400:
            continue
        ys, xs = np.nonzero(comp)
        z = scene.depth[ys, xs] * scale
        if not np.isfinite(z).all() or (z <= 0).any():
            continue
        ppm = f / np.median(z)                                  # photo px per metre at this distance
        dmax = float(np.percentile(D_mm[comp], 95)) if D_mm is not None else 300.0
        if dmax / 1000.0 * ppm < min_lift_px:
            continue
        N = scene.normal[ys, xs].astype(np.float64)
        N /= np.linalg.norm(N, axis=1, keepdims=True) + 1e-9
        nrm = np.median(N, 0); nrm /= np.linalg.norm(nrm)
        Pt = ((Kinv @ np.vstack([xs, ys, np.ones_like(xs)])) * z).T
        if nrm @ (-np.median(Pt, 0)) < 0:
            nrm = -nrm
        d = float(np.median(Pt @ nrm))
        up = scene.up
        e1 = np.cross(up, nrm)
        if np.linalg.norm(e1) < 0.2:                            # near-flat roof: any horizontal axis
            e1 = np.cross(nrm, np.array([0.0, 0.0, 1.0]))
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(nrm, e1)
        base = Pt - np.outer(Pt @ nrm - d, nrm)                 # pixels projected onto the plane
        c = base.mean(0)
        u, v = (base - c) @ e1, (base - c) @ e2
        res = float(np.clip(ppm, 5, 40))                        # grid px per metre ~ photo resolution
        u0, v0 = u.min() - 0.5, v.min() - 0.5
        Wg = int((u.max() + 0.5 - u0) * res) + 1
        Hg = int((v.max() + 0.5 - v0) * res) + 1
        if Wg * Hg > 4_000_000 or Wg < 8 or Hg < 8:
            continue
        o = c + e1 * u0 + e2 * v0
        G2P = scene.K @ np.column_stack([e1 / res, e2 / res, o])
        M = np.linalg.inv(G2P)
        pitch = float(np.degrees(np.arccos(np.clip(abs(nrm @ up), -1, 1))))
        fr = RoofFrame(scene.K, Kinv, nrm, d, M, G2P, e1, e2, 1 / res, 1 / res, pitch, 0.0, scale)
        out.append((fr, comp, (Hg, Wg)))
        if len(out) >= max_frames:
            break
    return out


def render_surfaces(frames, D_mm, scene, freshness, rng, shape):
    """Ray-march the snow solid of each surface frame. Returns photo maps: hit, shade S, depth D."""
    H0, W0 = shape
    hit = np.zeros(shape, bool)
    S = np.zeros(shape, np.float32)
    Dp = np.zeros(shape, np.float32)
    for fr, comp, (Hg, Wg) in frames:
        M = fr.M / fr.M[2, 2]
        Dg = cv2.warpPerspective((D_mm * comp).astype(np.float32), M, (Wg, Hg), flags=cv2.INTER_LINEAR)
        if Dg.max() < 5:
            continue
        res = 1.0 / fr.m_per_u
        rel = smooth_relief(Dg, np.zeros_like(Dg, bool), freshness, rng, P=res)
        Hm, _ = solid_height(Dg, None, None, freshness, rng, relief_mm=rel, P=res)
        Sg, _ = shade3d(Hm, fr, scene.up, np.zeros_like(Hm), rng)
        ys, xs = np.nonzero(comp)
        lift = float(Hm.max()) * float(scene.K[0, 0]) / max(float(np.median(scene.depth[ys, xs])) * fr.scale, 1e-3)
        bbox = (max(xs.min() - 4, 0), max(int(ys.min() - lift * 1.5) - 4, 0),
                min(xs.max() + 5, W0), min(ys.max() + 5, H0))
        R = raymarch(fr, Hm, {"S": Sg, "D": Dg}, shape, bbox, step_m=max(0.012, fr.m_per_u * 0.5))
        if R is None:
            continue
        new = R["hit"] & ~hit
        hit |= new
        S[new] = R["S"][new]
        Dp[new] = R["D"][new]
    return hit, S, Dp
