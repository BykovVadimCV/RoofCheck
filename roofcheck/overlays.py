"""Overlays burned into report photos by the camera app: timestamp text and GPS pin.

- mask(): pixels to drop from matching, synthesis and the coverage estimate
- read_timestamp(): date/time from the burned-in text (checked against the cleaning window)
- gps_pin_state(): the app's pin is green with a GPS fix and grey without one
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import cv2
import numpy as np

PIN_ZONE = (0.86, 0.0, 1.0, 0.28)          # x0, y0, x1, y1 as fractions of width/height

# yellow (#ffff00-ish, 2025 app) and orange (2022 app) stamp colours, OpenCV HSV ranges
STAMP_HSV = [((22, 170, 190), (38, 255, 255)), ((8, 170, 190), (21, 255, 255))]

_TS_PATTERNS = [
    # 07/10/2025 13:57:16 +03:00
    (re.compile(r"(\d{1,2})[/.](\d{1,2})[/.](\d{4})\s*(\d{2}):(\d{2})(?::(\d{2}))?\s*(?:GMT)?\s*([+-]\d{2}):?(\d{2})"),
     "dmy"),
]


@dataclass
class OverlayInfo:
    mask: np.ndarray            # uint8 0/255, same size as image
    stamp_box: tuple | None     # x0, y0, x1, y1 of the timestamp text
    pin_box: tuple
    gps_fix: bool | None        # True green pin, False grey pin, None no pin found


def _stamp_pixels(img_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    m = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in STAMP_HSV:
        m |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    return m


def _stamp_box(m: np.ndarray) -> tuple | None:
    """Stamp = row band in the lower part of the frame holding many saturated yellow/orange glyphs."""
    H, W = m.shape
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros_like(m)
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if y > 0.55 * H and 0.03 * H < h < 0.2 * H and a > 0.0004 * H * W:   # glyph-sized, lower part
            keep[lab == i] = 255
    if keep.sum() == 0:
        return None
    ys, xs = np.nonzero(keep)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def gps_pin_state(img_bgr: np.ndarray) -> tuple[bool | None, tuple]:
    H, W = img_bgr.shape[:2]
    x0, y0, x1, y1 = (int(PIN_ZONE[0] * W), int(PIN_ZONE[1] * H), int(PIN_ZONE[2] * W), int(PIN_ZONE[3] * H))
    zone = img_bgr[y0:y1, x0:x1]
    dark = (zone.max(axis=2) < 40)
    if dark.mean() < 0.01:                                    # no black outline -> no pin
        return None, (x0, y0, x1, y1)
    hsv = cv2.cvtColor(zone, cv2.COLOR_BGR2HSV)
    green = (hsv[..., 0] > 45) & (hsv[..., 0] < 85) & (hsv[..., 1] > 120) & (hsv[..., 2] > 120)
    return bool(green.mean() > 0.03), (x0, y0, x1, y1)


def detect(img_bgr: np.ndarray, pad: int = 8) -> OverlayInfo:
    H, W = img_bgr.shape[:2]
    m = np.zeros((H, W), np.uint8)
    sp = _stamp_pixels(img_bgr)
    box = _stamp_box(sp)
    if box:
        x0, y0, x1, y1 = box
        # glyph pixels plus their dark outline/antialiasing; the band itself is not masked entirely
        g = cv2.dilate(sp[y0:y1, x0:x1], np.ones((2 * pad + 1,) * 2, np.uint8))
        m[y0:y1, x0:x1] = np.maximum(m[y0:y1, x0:x1], g)
        m = cv2.dilate(m, np.ones((5, 5), np.uint8))
    fix, pb = gps_pin_state(img_bgr)
    if fix is not None:
        x0, y0, x1, y1 = pb
        zone = img_bgr[y0:y1, x0:x1]
        dark = (zone.max(axis=2) < 60).astype(np.uint8)
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        cnts, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pin = np.zeros_like(dark)
        if cnts:
            cv2.drawContours(pin, [max(cnts, key=cv2.contourArea)], -1, 1, -1)   # outline + interior
        m[y0:y1, x0:x1] = np.maximum(m[y0:y1, x0:x1], cv2.dilate(pin, np.ones((5, 5), np.uint8)) * 255)
    return OverlayInfo(mask=m, stamp_box=box, pin_box=pb, gps_fix=fix)


_ocr = None


def read_timestamp(img_bgr: np.ndarray, info: OverlayInfo | None = None) -> tuple[datetime | None, str]:
    """(aware datetime or None, raw OCR text)."""
    global _ocr
    info = info or detect(img_bgr)
    if not info.stamp_box:
        return None, ""
    if _ocr is None:
        from rapidocr_onnxruntime import RapidOCR
        _ocr = RapidOCR()
    x0, y0, x1, y1 = info.stamp_box
    crop = img_bgr[max(0, y0 - 10):y1 + 10, max(0, x0 - 10):x1 + 10]
    # stamp colour -> black on white: best case for OCR
    sp = _stamp_pixels(crop)
    clean = 255 - cv2.dilate(sp, np.ones((2, 2), np.uint8))
    res, _ = _ocr(cv2.cvtColor(clean, cv2.COLOR_GRAY2BGR))
    text = " ".join(r[1] for r in (res or []))
    t = text.replace("O", "0").replace("o", "0").replace(" :", ":").replace(": ", ":")
    for rx, order in _TS_PATTERNS:
        mt = rx.search(t)
        if mt:
            d, mo, y, hh, mi, ss, tzh, tzm = mt.groups()
            sign = -1 if tzh.startswith("-") else 1
            tz = timezone(sign * timedelta(hours=abs(int(tzh)), minutes=int(tzm)))
            try:
                return datetime(int(y), int(mo), int(d), int(hh), int(mi), int(ss or 0), tzinfo=tz), text
            except ValueError:
                pass
    return None, text


BLANK_VALUE = 114          # same constant for real and synthetic network inputs


def blank(img_bgr: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Network input: overlay pixels set to a constant, so the burned-in date and pin carry no signal
    (an October date next to synthetic snow would otherwise be a free shortcut)."""
    m = detect(img_bgr).mask if mask is None else mask
    out = img_bgr.copy()
    out[m > 0] = BLANK_VALUE
    return out
