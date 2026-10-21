"""Rectification of the roof slope: the 4-corner slope quad (eave-left, eave-right, ridge-right,
ridge-left) is mapped to a rectangle. The slope is planar, so equal pixels of the rectified
image are equal areas of the roof; shares computed there are perspective-correct.

The rectangle's aspect is not known in metres. We take it from the quad's mean side lengths;
this only changes the scale of the output, not the area shares.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Rectified:
    image: np.ndarray            # rectified slope, H_r x W_r
    valid: np.ndarray            # uint8 mask: pixel comes from inside the photo and not from an overlay
    M: np.ndarray                # photo px -> rectified px
    area_photo: np.ndarray       # per-pixel relative area in the photo (sum over the quad == 1)


def rectify(img: np.ndarray, quad: np.ndarray, ignore: np.ndarray | None = None,
            px_per_unit: float | None = None) -> Rectified:
    q = np.asarray(quad, np.float32)
    w = 0.5 * (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3]))
    h = 0.5 * (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1]))
    # the slope is foreshortened in the photo: its depth (eave -> ridge) is shown much shorter than
    # its length. Without metric data we keep the photo's long side and do not invent depth.
    W_r = int(round(px_per_unit * w)) if px_per_unit else int(round(w))
    H_r = max(8, int(round(h)))
    dst = np.float32([[0, H_r - 1], [W_r - 1, H_r - 1], [W_r - 1, 0], [0, 0]])
    M = cv2.getPerspectiveTransform(q, dst)
    out = cv2.warpPerspective(img, M, (W_r, H_r), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    src_ok = np.full(img.shape[:2], 255, np.uint8)
    if ignore is not None:
        src_ok[ignore > 0] = 0
    valid = cv2.warpPerspective(src_ok, M, (W_r, H_r), flags=cv2.INTER_NEAREST, borderValue=0)
    return Rectified(out, valid, M, pixel_area(img.shape, q, M))


def pixel_area(shape, quad: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Relative ground area of each photo pixel inside the quad: |det J| of photo->rectified map,
    normalised so the quad sums to 1. Pixels outside the quad are 0."""
    H, W = shape[:2]
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float64)
    a, b, c = M[0]
    d, e, f = M[1]
    g, h, i = M[2]
    w = g * xs + h * ys + i
    u = a * xs + b * ys + c
    v = d * xs + e * ys + f
    # Jacobian of (u/w, v/w)
    dudx = (a * w - u * g) / w**2
    dudy = (b * w - u * h) / w**2
    dvdx = (d * w - v * g) / w**2
    dvdy = (e * w - v * h) / w**2
    J = np.abs(dudx * dvdy - dudy * dvdx)
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [np.round(quad).astype(np.int32)], 1)
    J *= m
    s = J.sum()
    return (J / s).astype(np.float32) if s > 0 else J.astype(np.float32)


def _segments(gray: np.ndarray) -> np.ndarray:
    lsd = cv2.createLineSegmentDetector()
    s = lsd.detect(gray)[0]
    return np.zeros((0, 4)) if s is None else s.reshape(-1, 4)


def _angle_deg(seg):
    """Direction in 0..180, counter-clockwise from the image x axis (y up)."""
    x1, y1, x2, y2 = seg.T
    return np.degrees(np.arctan2(-(y2 - y1), x2 - x1)) % 180


def seam_vanishing_point(img: np.ndarray, region: np.ndarray, ignore: np.ndarray | None = None,
                         min_len: float = 25) -> tuple[np.ndarray | None, dict]:
    """Vanishing point (homogeneous 3-vector) of the standing seams inside `region`.

    Seams are the dominant oblique family: not horizontal (fence, eave) and not vertical
    (chimneys, fence posts, which stay vertical in the photo). Least-squares point closest
    to all seam lines, weighted by length; it may lie at infinity (seams parallel in the photo).
    """
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    seg = _segments(g)
    if len(seg) == 0:
        return None, {"n": 0}
    mid = ((seg[:, :2] + seg[:, 2:]) / 2).round().astype(int)
    H, W = g.shape
    mid[:, 0] = mid[:, 0].clip(0, W - 1)
    mid[:, 1] = mid[:, 1].clip(0, H - 1)
    ok = region[mid[:, 1], mid[:, 0]] > 0
    if ignore is not None:
        ok &= ignore[mid[:, 1], mid[:, 0]] == 0
    L = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])
    ok &= L >= min_len
    a = _angle_deg(seg)
    oblique = (a > 15) & (a < 165) & (np.abs(a - 90) > 12)
    sel = ok & oblique
    if sel.sum() < 6:
        return None, {"n": int(sel.sum())}
    hist, edges = np.histogram(a[sel], bins=36, range=(0, 180), weights=L[sel])
    peak = edges[np.argmax(hist)] + 2.5
    fam = sel & (np.abs(a - peak) < 15)
    s = seg[fam]
    w = L[fam]
    p1 = np.c_[s[:, :2], np.ones(len(s))]
    p2 = np.c_[s[:, 2:], np.ones(len(s))]
    lines = np.cross(p1, p2)
    lines /= np.linalg.norm(lines[:, :2], axis=1, keepdims=True)
    _, _, vt = np.linalg.svd(lines * w[:, None])
    vp = vt[-1]
    return vp, {"n": int(fam.sum()), "peak_deg": float(peak)}


def refine_quad_by_seams(img: np.ndarray, quad: np.ndarray, region: np.ndarray | None = None,
                         ignore: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Make the quad's side edges follow the seam direction (eave -> ridge).

    A loosely drawn polygon (the customer's red outline) gets eave and ridge roughly right but its
    left/right sides are arbitrary, which shears the rectified slope. The sides are redrawn as lines
    from the old side midpoints toward the seams' vanishing point; eave and ridge lines are kept.
    """
    q = np.asarray(quad, float)
    if region is None:
        region = np.zeros(img.shape[:2], np.uint8)
        cv2.fillPoly(region, [np.round(q).astype(np.int32)], 255)
    vp, info = seam_vanishing_point(img, region, ignore)
    if vp is None:
        return q, dict(info, refined=False)
    h = lambda p: np.array([p[0], p[1], 1.0])
    eave = np.cross(h(q[0]), h(q[1]))
    ridge = np.cross(h(q[3]), h(q[2]))
    out = q.copy()
    for (ie, ir) in [(0, 3), (1, 2)]:
        midp = h((q[ie] + q[ir]) / 2)
        side = np.cross(midp, vp)
        pe = np.cross(side, eave)
        pr = np.cross(side, ridge)
        if abs(pe[2]) < 1e-9 or abs(pr[2]) < 1e-9:
            return q, dict(info, refined=False)
        out[ie] = pe[:2] / pe[2]
        out[ir] = pr[:2] / pr[2]
    return out, dict(info, refined=True)


def seam_parallelism(rect_img: np.ndarray, valid: np.ndarray, near_vertical: bool = True) -> dict:
    """Check of the rectification: standing seams run eave->ridge, so on a correctly rectified
    slope they are vertical and parallel. Returns median deviation from vertical and the
    interquartile spread of seam angles (degrees), length-weighted."""
    g = cv2.cvtColor(rect_img, cv2.COLOR_BGR2GRAY)
    seg = _segments(g)
    if len(seg) == 0:
        return {"n": 0}
    mid = ((seg[:, :2] + seg[:, 2:]) / 2).round().astype(int)
    Hh, Ww = g.shape
    mid[:, 0] = mid[:, 0].clip(0, Ww - 1)
    mid[:, 1] = mid[:, 1].clip(0, Hh - 1)
    ok = cv2.erode(valid, np.ones((5, 5), np.uint8))[mid[:, 1], mid[:, 0]] > 0
    L = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])
    ok &= L >= 0.2 * Hh
    dev = _angle_deg(seg) - 90                      # 0 = vertical
    if near_vertical:
        ok &= np.abs(dev) < 35
    if ok.sum() < 3:
        return {"n": int(ok.sum())}
    # seam family = dominant direction; chimney edges and fence braces fall outside +-12 deg of it
    hist, edges = np.histogram(dev[ok], bins=28, range=(-35, 35), weights=L[ok])
    peak = edges[np.argmax(hist)] + 1.25
    ok &= np.abs(dev - peak) < 12
    if ok.sum() < 3:
        return {"n": int(ok.sum())}
    d, w = dev[ok], L[ok]
    o = np.argsort(d)
    cw = np.cumsum(w[o]) / w.sum()
    q = lambda p: float(d[o][np.searchsorted(cw, p)])
    return {"n": int(ok.sum()), "median_dev_deg": round(q(0.5), 2), "iqr_deg": round(q(0.75) - q(0.25), 2)}


def seam_parallelism_photo(img: np.ndarray, quad: np.ndarray, ignore: np.ndarray | None = None) -> dict:
    """Spread of seam angles in the unrectified photo, for comparison (they converge in perspective)."""
    q = np.asarray(quad, np.float32)
    region = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(region, [np.round(q).astype(np.int32)], 255)
    vp, info = seam_vanishing_point(img, region, ignore)
    if vp is None:
        return info
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    seg = _segments(g)
    a = _angle_deg(seg)
    mid = ((seg[:, :2] + seg[:, 2:]) / 2).round().astype(int).clip(0, [img.shape[1] - 1, img.shape[0] - 1])
    L = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])
    ok = (region[mid[:, 1], mid[:, 0]] > 0) & (L >= 25) & (np.abs(a - info["peak_deg"]) < 15)
    d = a[ok]
    return {"n": int(ok.sum()), "iqr_deg": round(float(np.subtract(*np.percentile(d, [75, 25]))), 2)}
