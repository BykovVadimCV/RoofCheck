"""Import of the customer's markup: a red polygon drawn on top of a report photo.

The customer already circles the target roof in red on report photos. We read that outline back
as a polygon (role "roof") and derive the 4-corner slope quad (role "slope") from its two
longest near-horizontal edges: the lower one is the eave, the upper one the ridge.
"""
from __future__ import annotations

import cv2
import numpy as np


def red_mask(img_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    m = ((h <= 8) | (h >= 170)) & (s >= 150) & (v >= 120)
    return m.astype(np.uint8) * 255


def extract_polygon(img_bgr: np.ndarray, min_area_frac: float = 0.005) -> np.ndarray | None:
    """Largest closed red outline -> polygon (N x 2, float) along the middle of the stroke."""
    m = red_mask(img_bgr)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))  # bridge gaps under text
    cnts, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    H, W = m.shape
    best, best_area = None, 0.0
    for i, c in enumerate(cnts):
        if hier[0][i][3] < 0:                     # outer contour of a stroke; we want the hole inside
            continue
        a = cv2.contourArea(c)
        if a > best_area:
            best, best_area = c, a
    if best is None or best_area < min_area_frac * H * W:
        return None
    # inner edge of the stroke -> grow by half the stroke width to sit on the line centre
    stroke = _stroke_width(m)
    filled = np.zeros_like(m)
    cv2.drawContours(filled, [best], -1, 255, -1)
    filled = cv2.dilate(filled, np.ones((stroke // 2 * 2 + 1,) * 2, np.uint8))
    c2, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c2 = max(c2, key=cv2.contourArea)
    peri = cv2.arcLength(c2, True)
    poly = cv2.approxPolyDP(c2, 0.006 * peri, True)[:, 0, :].astype(float)
    return poly


def _stroke_width(m: np.ndarray) -> int:
    d = cv2.distanceTransform((m > 0).astype(np.uint8), cv2.DIST_L2, 3)
    vals = d[d > 0]
    return int(max(2, round(2 * np.percentile(vals, 90)))) if vals.size else 4


def slope_quad(poly: np.ndarray, max_tilt_deg: float = 35.0) -> np.ndarray:
    """4 corners [eave_l, eave_r, ridge_r, ridge_l] from the two longest near-horizontal edges."""
    n = len(poly)
    edges = []
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        d = b - a
        ang = abs(np.degrees(np.arctan2(d[1], d[0])))
        ang = min(ang, 180 - ang)
        if ang <= max_tilt_deg:
            edges.append((np.hypot(*d), a, b))
    edges.sort(key=lambda e: -e[0])
    if len(edges) < 2:
        raise ValueError("no two near-horizontal edges: draw the slope quad manually")
    e1, e2 = edges[0], edges[1]
    ymid = lambda e: (e[1][1] + e[2][1]) / 2
    eave, ridge = (e1, e2) if ymid(e1) > ymid(e2) else (e2, e1)
    el, er = sorted([eave[1], eave[2]], key=lambda p: p[0])
    rl, rr = sorted([ridge[1], ridge[2]], key=lambda p: p[0])
    return np.array([el, er, rr, rl], float)
