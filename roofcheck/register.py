"""Registration of a report photo to the reference frame of its vantage point.

ALIKED keypoints (BSD-3) + LightGlue matcher (Apache-2.0) via kornia, homography by
OpenCV USAC_MAGSAC. Two passes:
  1. global homography on the whole frame (overlays and the photographer's own roof masked out);
  2. local homography from inliers around the target roof only. The scene is not planar, so a
     homography fitted near the roof is more accurate there than a whole-frame one.

The result carries quality numbers; `ok` is False when the photo must go to the operator.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import warnings

import cv2
import kornia as K
import kornia.feature as KF
import numpy as np
import torch

from . import overlays
from .schema import Reference

warnings.filterwarnings("ignore", category=UserWarning, module="kornia")
warnings.filterwarnings("ignore", category=FutureWarning)
MATCH_LONG_SIDE = 1024          # matching resolution; homography is rescaled to full resolution


@dataclass
class Quality:
    matches: int = 0
    inliers: int = 0
    inlier_ratio: float = 0.0
    spread: float = 0.0          # convex-hull area of inliers / frame area (reference side)
    local_inliers: int = 0
    reproj_px: float = 0.0       # median reprojection error of local inliers, full-res px
    scale: float = 1.0           # zoom of photo vs reference (sqrt det of the affine part)
    rotation_deg: float = 0.0
    visible_frac: float = 0.0    # share of the target roof (by area) that falls inside the photo


@dataclass
class Registration:
    ok: bool
    reason: str
    H: np.ndarray | None                     # reference px -> photo px (full resolution)
    quality: Quality = field(default_factory=Quality)
    polygons: dict = field(default_factory=dict)   # role -> list of N x 2 arrays in photo px


class Registrar:
    # acceptance thresholds (tuned on the pilot; conservative start values)
    MIN_INLIERS = 40
    MIN_LOCAL_INLIERS = 15
    MIN_SPREAD = 0.05
    MAX_REPROJ_PX = 6.0
    MIN_VISIBLE = 0.3            # less of the roof in frame -> operator ("кровля не попала в кадр")

    def __init__(self, max_kp: int = 2048, device: str = "cpu"):
        self.dev = torch.device(device)
        self.det = KF.ALIKED.from_pretrained("aliked-n16", max_num_keypoints=max_kp, device=self.dev).eval()
        self.lg = KF.LightGlue("aliked").eval().to(self.dev)
        self._ref_cache: dict = {}

    # ---------------------------------------------------------------- features
    def _features(self, img_bgr: np.ndarray, ignore: np.ndarray | None):
        H, W = img_bgr.shape[:2]
        s = MATCH_LONG_SIDE / max(H, W)
        small = cv2.resize(img_bgr, (round(W * s), round(H * s)), interpolation=cv2.INTER_AREA)
        x = K.image_to_tensor(small[:, :, ::-1].copy(), False).float().to(self.dev) / 255
        with torch.inference_mode():
            f = self.det(x)[0]
        kp = f.keypoints.cpu().numpy()
        desc = f.descriptors
        if ignore is not None:
            ig = cv2.resize(ignore, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
            xi = np.clip(kp[:, 0].round().astype(int), 0, small.shape[1] - 1)
            yi = np.clip(kp[:, 1].round().astype(int), 0, small.shape[0] - 1)
            keep = ig[yi, xi] == 0
            kp, desc = kp[keep], desc[torch.from_numpy(keep).to(desc.device)]
        return kp, desc, (small.shape[0], small.shape[1]), s

    def _match(self, fa, fb):
        kpa, da, sha, _ = fa
        kpb, db, shb, _ = fb
        data = {
            "image0": {"keypoints": torch.from_numpy(kpa)[None].float().to(self.dev), "descriptors": da[None],
                       "image_size": torch.tensor([[sha[1], sha[0]]], device=self.dev)},
            "image1": {"keypoints": torch.from_numpy(kpb)[None].float().to(self.dev), "descriptors": db[None],
                       "image_size": torch.tensor([[shb[1], shb[0]]], device=self.dev)},
        }
        with torch.inference_mode():
            out = self.lg(data)
        m = out["matches"][0].cpu().numpy()
        return kpa[m[:, 0]], kpb[m[:, 1]]

    # ---------------------------------------------------------------- api
    def reference_features(self, ref: Reference, ref_img: np.ndarray):
        key = (ref.object_id, ref.vantage_id, ref.image)
        if key not in self._ref_cache:
            ignore = overlays.detect(ref_img).mask
            for p in ref.by_role("exclude") + ref.by_role("overlay"):
                cv2.fillPoly(ignore, [np.round(p.points).astype(np.int32)], 255)
            self._ref_cache[key] = self._features(ref_img, ignore)
        return self._ref_cache[key]

    def register(self, photo_bgr: np.ndarray, ref: Reference, ref_img: np.ndarray,
                 exclude_photo: np.ndarray | None = None) -> Registration:
        fr = self.reference_features(ref, ref_img)
        ign = overlays.detect(photo_bgr).mask
        if exclude_photo is not None:
            ign = np.maximum(ign, exclude_photo)
        fp = self._features(photo_bgr, ign)
        pr, pp = self._match(fr, fp)
        q = Quality(matches=len(pr))
        if len(pr) < 8:
            return Registration(False, "мало совпадений с эталоном", None, q)
        # back to full resolution
        pr_f, pp_f = pr / fr[3], pp / fp[3]
        Hg, inl = cv2.findHomography(pr_f, pp_f, cv2.USAC_MAGSAC, 4.0, maxIters=10000, confidence=0.9999)
        if Hg is None:
            return Registration(False, "гомография не найдена", None, q)
        inl = inl.ravel().astype(bool)
        q.inliers, q.inlier_ratio = int(inl.sum()), float(inl.mean())
        hull = cv2.convexHull(pr_f[inl].astype(np.float32)) if inl.sum() >= 3 else None
        q.spread = float(cv2.contourArea(hull) / (ref.width * ref.height)) if hull is not None else 0.0

        # local refinement around the target roof
        H = Hg
        roof = ref.by_role("roof") or ref.by_role("slope")
        if roof:
            zone = _expanded_box(np.array(roof[0].points), ref.width, ref.height, grow=1.0)
            near = inl & _inside(pr_f, zone)
            if near.sum() >= self.MIN_LOCAL_INLIERS:
                Hl, inl2 = cv2.findHomography(pr_f[near], pp_f[near], cv2.USAC_MAGSAC, 3.0,
                                              maxIters=10000, confidence=0.9999)
                if Hl is not None and inl2.sum() >= self.MIN_LOCAL_INLIERS:
                    H = Hl
                    near_idx = np.flatnonzero(near)[inl2.ravel().astype(bool)]
                    q.local_inliers = int(len(near_idx))
                    proj = cv2.perspectiveTransform(pr_f[near_idx][None].astype(np.float64), H)[0]
                    q.reproj_px = float(np.median(np.linalg.norm(proj - pp_f[near_idx], axis=1)))
            else:
                q.local_inliers = int(near.sum())
        A = H[:2, :2] / H[2, 2]
        q.scale = float(np.sqrt(abs(np.linalg.det(A))))
        q.rotation_deg = float(np.degrees(np.arctan2(A[1, 0] - A[0, 1], A[0, 0] + A[1, 1])))

        polys = {}
        for p in ref.polygons:
            pts = cv2.perspectiveTransform(np.array(p.points, np.float64)[None], H)[0]
            polys.setdefault(p.role, []).append(pts)

        reasons = []
        if q.inliers < self.MIN_INLIERS:
            reasons.append(f"мало надёжных совпадений ({q.inliers})")
        if q.spread < self.MIN_SPREAD:
            reasons.append("совпадения сосредоточены в малой части кадра")
        if q.local_inliers < self.MIN_LOCAL_INLIERS:
            reasons.append(f"мало совпадений у проверяемой кровли ({q.local_inliers})")
        if q.reproj_px > self.MAX_REPROJ_PX:
            reasons.append(f"ошибка переноса {q.reproj_px:.1f} px")
        if roof:
            q.visible_frac = visible_fraction(polys[roof[0].role][0], photo_bgr.shape)
            if q.visible_frac < self.MIN_VISIBLE:
                reasons.append(f"в кадре {q.visible_frac:.0%} проверяемой кровли")
        return Registration(not reasons, "; ".join(reasons) or "ok", H, q, polys)


def _expanded_box(pts, W, H, grow=1.0):
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    w, h = x1 - x0, y1 - y0
    g = grow * max(h, 0.15 * w)
    return max(0, x0 - g), max(0, y0 - g), min(W, x1 + g), min(H, y1 + g)


def _inside(p, box):
    x0, y0, x1, y1 = box
    return (p[:, 0] >= x0) & (p[:, 0] <= x1) & (p[:, 1] >= y0) & (p[:, 1] <= y1)


def visible_fraction(poly: np.ndarray, shape) -> float:
    H, W = shape[:2]
    hull = cv2.convexHull(poly.astype(np.float32))
    a = cv2.contourArea(hull)
    if a <= 0:
        return 0.0
    frame = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    inter, _ = cv2.intersectConvexConvex(hull, frame)
    return float(min(1.0, inter / a))


def polygon_iou(a: np.ndarray, b: np.ndarray, shape) -> float:
    H, W = shape[:2]
    ma = np.zeros((H, W), np.uint8)
    mb = np.zeros((H, W), np.uint8)
    cv2.fillPoly(ma, [np.round(a).astype(np.int32)], 1)
    cv2.fillPoly(mb, [np.round(b).astype(np.int32)], 1)
    inter = (ma & mb).sum()
    uni = (ma | mb).sum()
    return float(inter / uni) if uni else 0.0
