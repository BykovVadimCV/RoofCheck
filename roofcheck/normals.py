"""Surface normals of the whole frame (MoGe-2 ViT-B normal, code and weights MIT) and what follows from them:
the gravity direction, the tilt of every surface, and where snow can lie.

    est = NormalEstimator("models/moge-2-vitb-normal.pt")
    g = est.analyse(img)              # cached per image content in out/cache/normals
    g.tilt_deg                        # 0 = horizontal, 90 = wall, nan = sky / no estimate
    g.mu                              # SP 20 shape factor of each pixel: 1 up to 30 deg, 0 from 60 deg

Camera frame is OpenCV (x right, y down, z forward). Up is fitted from the facades: their normals are
perpendicular to gravity, so up is the direction least represented among near-vertical-surface normals.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = ROOT / "models/moge-2-vitb-normal.pt"
CACHE = ROOT / "out/cache/normals"


@dataclass
class Geometry:
    normal: np.ndarray        # H x W x 3, unit, camera frame
    valid: np.ndarray         # bool, model's own validity mask (no sky)
    up: np.ndarray            # 3, unit, camera frame
    tilt_deg: np.ndarray      # H x W, angle between surface normal and up
    mu: np.ndarray            # H x W, snow shape factor by tilt
    depth: np.ndarray         # H x W, metres (relative scale is reliable, absolute is approximate)
    K: np.ndarray | None = None   # 3x3 camera intrinsics in pixels


def fit_up(n: np.ndarray, valid: np.ndarray, iters: int = 6) -> np.ndarray:
    """Up vector from the near-vertical surfaces (facades), IRLS starting from the image 'up'."""
    N = n[valid].reshape(-1, 3)
    if len(N) > 400_000:
        N = N[np.random.default_rng(0).choice(len(N), 400_000, replace=False)]
    u = np.array([0.0, -1.0, 0.0])
    for _ in range(iters):
        c = N @ u
        w = np.exp(-(c / 0.2) ** 2)                    # weight walls (n . up ~ 0)
        C = (N * w[:, None]).T @ N / w.sum()
        vals, vecs = np.linalg.eigh(C)
        u_new = vecs[:, 0]
        if u_new @ u < 0:
            u_new = -u_new
        if np.allclose(u_new, u, atol=1e-5):
            break
        u = u_new
    return u / np.linalg.norm(u)


def mu_from_tilt(tilt_deg: np.ndarray) -> np.ndarray:
    """SP 20.13330.2016 App. B: 1 up to 30 deg, 0 from 60 deg, linear in between."""
    return np.clip((60.0 - tilt_deg) / 30.0, 0.0, 1.0).astype(np.float32)


class NormalEstimator:
    def __init__(self, weights: str | Path = DEFAULT_WEIGHTS, threads: int = 4):
        self.weights = Path(weights)
        self.threads = threads
        self._model = None

    def _load(self):
        if self._model is None:
            import torch
            from moge.model.v2 import MoGeModel
            torch.set_num_threads(self.threads)
            self._model = MoGeModel.from_pretrained(str(self.weights)).eval()
        return self._model

    def raw(self, img_bgr: np.ndarray) -> dict:
        key = hashlib.sha256(img_bgr.tobytes()).hexdigest()[:16]
        f = CACHE / f"{key}.npz"
        if f.exists():
            d = np.load(f)
            if "intrinsics" in d.files:
                return {k: d[k] for k in d.files}
        import torch
        import warnings
        m = self._load()
        x = torch.tensor(img_bgr[:, :, ::-1] / 255.0, dtype=torch.float32).permute(2, 0, 1)
        with torch.no_grad(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            o = m.infer(x, resolution_level=0, use_fp16=False)
        d = {"normal": o["normal"].numpy().astype(np.float16), "mask": o["mask"].numpy().astype(bool),
             "depth": o["depth"].numpy().astype(np.float32),
             "intrinsics": o["intrinsics"].numpy().astype(np.float64)}   # normalised: fx/W, fy/H, cx/W, cy/H
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(f, **d)
        return d

    def analyse(self, img_bgr: np.ndarray, ignore: np.ndarray | None = None) -> Geometry:
        d = self.raw(img_bgr)
        n = d["normal"].astype(np.float32)
        n /= np.linalg.norm(n, axis=2, keepdims=True) + 1e-6
        valid = d["mask"] & np.isfinite(n).all(2)
        if ignore is not None:
            valid &= ignore == 0
        up = fit_up(n, valid)
        tilt = np.degrees(np.arccos(np.clip(n @ up, -1, 1))).astype(np.float32)
        tilt[~valid] = np.nan
        mu = mu_from_tilt(np.nan_to_num(tilt, nan=90.0))
        depth = np.nan_to_num(d["depth"].astype(np.float32), nan=0.0, posinf=0.0)
        K = None
        if "intrinsics" in d:
            Hh, Ww = img_bgr.shape[:2]
            K = np.array(d["intrinsics"], np.float64) * np.array([[Ww], [Hh], [1.0]])
        return Geometry(n, valid, up, tilt, mu, depth, K)


def pitch_of(geo: Geometry, mask: np.ndarray) -> float | None:
    """Median tilt over a region (e.g. the target slope): an estimate of the roof pitch."""
    t = geo.tilt_deg[mask & geo.valid]
    return float(np.nanmedian(t)) if t.size > 200 else None


def roughness(normal: np.ndarray, k: int = 15) -> np.ndarray:
    """1 - |local mean normal|: ~0 on planar surfaces (facades, roofs), large in tree crowns."""
    m = np.stack([cv2.blur(normal[..., i], (k, k)) for i in range(3)], -1)
    return (1 - np.linalg.norm(m, axis=2)).astype(np.float32)


def vegetation(img_bgr: np.ndarray, normal: np.ndarray | None = None) -> np.ndarray:
    """Foliage: leaf colour (green, or strongly saturated autumn yellow/orange/red), leafy texture, and,
    when normals are given, a chaotic surface orientation. Colour + texture alone takes window frames and
    ochre plaster; planar facades have smooth normals, crowns do not."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, sat, val = hsv[..., 0].astype(np.int16), hsv[..., 1], hsv[..., 2]
    green = (h >= 30) & (h < 95) & (sat > 40)
    autumn = ((h < 30) | (h > 165)) & (sat > 80)
    colour = (green | autumn) & (val > 25)
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mu = cv2.blur(g, (9, 9))
    sd = np.sqrt(np.clip(cv2.blur(g * g, (9, 9)) - mu * mu, 0, None))
    v = colour & (sd > 8)
    if normal is not None:
        v &= cv2.blur(roughness(normal).astype(np.float32), (9, 9)) > 0.02
    v = cv2.morphologyEx(v.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(v)
    v = np.isin(lab, 1 + np.flatnonzero(st[1:, cv2.CC_STAT_AREA] >= 600))      # crowns, not window frames
    return cv2.dilate(v.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0


def point_normals(depth: np.ndarray, K: np.ndarray, smooth_px: float = 1.5) -> np.ndarray:
    """Normals from the 3D point map (cross product of its image gradients). Unlike the network's normal
    head they are not smoothed across object boundaries, so a facade stays a wall right up to the roof
    edge."""
    H, W = depth.shape
    d = cv2.GaussianBlur(depth.astype(np.float32), (0, 0), smooth_px)
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    Ki = np.linalg.inv(K)
    X = (Ki[0, 0] * xs + Ki[0, 2]) * d
    Y = (Ki[1, 1] * ys + Ki[1, 2]) * d
    P = np.dstack([X, Y, d])
    du = np.gradient(P, axis=1)
    dv = np.gradient(P, axis=0)
    n = np.cross(du, dv)
    n /= np.linalg.norm(n, axis=2, keepdims=True) + 1e-9
    flip = (n * P).sum(2) > 0                     # face the camera
    n[flip] *= -1
    return n.astype(np.float32)


def roofness(img_bgr: np.ndarray, geo: Geometry, n_segments: int = 5000, min_frac: float = 0.75):
    """Mask of surfaces snow can lie on (roofs, ledges, ground), conservative:
    - both the network normal and the point-map normal say the surface faces up (tilt < 55 deg);
    - snapped to SLIC superpixels: a segment is kept only if >= min_frac of it agrees and its normals are
      consistent (a planar patch), so tree crowns and facade tops with mixed normals drop out;
    - vegetation and sky removed. Returns (mask, mu per pixel)."""
    from skimage.segmentation import slic
    H, W = img_bgr.shape[:2]
    if geo.K is None:
        up_ok = (geo.mu > 0.3) & geo.valid
        pn_tilt = None
    else:
        pn = point_normals(geo.depth, geo.K)
        pn_tilt = np.degrees(np.arccos(np.clip(pn @ geo.up, -1, 1)))
        up_ok = (np.nan_to_num(geo.tilt_deg, nan=90) < 55) & (pn_tilt < 55) & geo.valid
    veg = vegetation(img_bgr, geo.normal)
    up_ok &= ~veg
    s = 1024 / max(H, W)
    small = cv2.resize(img_bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    seg = slic(small[:, :, ::-1], n_segments=n_segments, compactness=10, start_label=0, channel_axis=-1)
    seg = cv2.resize(seg.astype(np.int32), (W, H), interpolation=cv2.INTER_NEAREST)
    n = seg.max() + 1
    cnt = np.bincount(seg.ravel(), minlength=n) + 1e-6
    frac = np.bincount(seg.ravel(), up_ok.ravel().astype(np.float64), minlength=n) / cnt
    nm = np.stack([np.bincount(seg.ravel(), geo.normal[..., k].ravel().astype(np.float64), minlength=n) / cnt
                   for k in range(3)], 1)
    consistency = np.linalg.norm(nm, axis=1)             # 1 = all normals equal, low = rough (crowns)
    vegf = np.bincount(seg.ravel(), veg.ravel().astype(np.float64), minlength=n) / cnt
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, sat = hsv[..., 0].astype(np.int16), hsv[..., 1]
    leafc = (((h >= 15) & (h < 80)) | (h > 170) | (h < 8)) & (sat > 55)
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mu_ = cv2.blur(g, (7, 7)); sd = np.sqrt(np.clip(cv2.blur(g * g, (7, 7)) - mu_ * mu_, 0, None))
    leaf_f = np.bincount(seg.ravel(), leafc.ravel().astype(np.float64), minlength=n) / cnt
    busy = np.bincount(seg.ravel(), sd.ravel().astype(np.float64), minlength=n) / cnt
    treeish = (leaf_f > 0.3) & (busy > 9)
    # a crown is a cluster of such segments: grow the decision to neighbouring leafy segments
    tm = cv2.dilate(treeish[seg].astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    near_tree = np.bincount(seg.ravel(), tm.ravel().astype(np.float64), minlength=n) / cnt
    keep = (frac >= min_frac) & (consistency > 0.9) & (vegf < 0.2) & ~treeish & ~((near_tree > 0.5) & (leaf_f > 0.15))
    m = keep[seg] & geo.valid & ~veg
    mu = mu_from_tilt(np.nan_to_num(geo.tilt_deg, nan=90.0))
    return m, mu
