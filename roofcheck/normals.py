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
