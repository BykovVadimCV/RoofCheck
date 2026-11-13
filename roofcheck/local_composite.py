"""Optional conservative compositing for the existing planar renderer.

Keep the original frame outside the marked roof. The feather band is ignored in
training labels because the visible snow there is a mixture of two images.
"""
from __future__ import annotations

import cv2
import numpy as np


def local_composite(original, rendered, label, roi, protected=None, feather_px=4):
    """Return image, conservative label and alpha; arrays stay in input channel order."""
    original = np.asarray(original)
    rendered = np.asarray(rendered)
    label = np.asarray(label)
    roi = np.asarray(roi)
    if original.dtype != np.uint8 or rendered.dtype != np.uint8 or original.shape != rendered.shape:
        raise ValueError('original and rendered must be matching uint8 images')
    if original.ndim != 3 or original.shape[2] != 3 or label.shape != original.shape[:2] or roi.shape != label.shape:
        raise ValueError('image, label and ROI dimensions must match')
    if label.dtype != np.uint8 or not set(np.unique(label).tolist()) <= {0, 1, 255}:
        raise ValueError('label must be uint8 with values 0, 1 or 255')
    if not isinstance(feather_px, int) or not 0 <= feather_px <= 100:
        raise ValueError('feather_px must be an integer in 0..100')
    allowed = roi.astype(bool).copy()
    if protected is not None:
        protected = np.asarray(protected)
        if protected.shape != label.shape:
            raise ValueError('protected mask dimensions must match ROI')
        allowed &= ~protected.astype(bool)
    if feather_px:
        # Padding supplies a real zero boundary even when the ROI reaches the frame.
        pad = np.pad(allowed.astype(np.uint8), 1)
        dist = cv2.distanceTransform(pad, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
        alpha = np.minimum(dist / float(feather_px + 1), 1).astype(np.float32)
    else:
        alpha = allowed.astype(np.float32)
    mixed = original.astype(np.float32) * (1 - alpha[..., None]) + rendered.astype(np.float32) * alpha[..., None]
    image = np.clip(np.rint(mixed), 0, 255).astype(np.uint8)
    # Assign explicitly to avoid any rounding surprises in protected pixels.
    image[~allowed] = original[~allowed]
    result_label = label.copy()
    result_label[(~allowed) | (alpha < 1)] = 255
    return image, result_label, alpha


def ignore_low_change_snow(original, rendered, label, threshold):
    """Ignore positive pixels similar to their snow-free source (an optional heuristic).

    Low RGB change does not prove snow is invisible; thresholds require validation.
    """
    original, rendered, label = map(np.asarray, (original, rendered, label))
    if (original.dtype != np.uint8 or rendered.dtype != np.uint8 or label.dtype != np.uint8
            or original.ndim != 3 or original.shape[2] != 3 or original.shape != rendered.shape
            or label.shape != original.shape[:2] or not set(np.unique(label).tolist()) <= {0, 1, 255}):
        raise ValueError('Images and binary label must have matching shapes and uint8 values')
    if not isinstance(threshold, int) or not 0 <= threshold <= 255:
        raise ValueError('threshold must be an integer in 0..255')
    delta = np.abs(original.astype(np.int16) - rendered.astype(np.int16)).max(axis=2)
    ambiguous = (label == 1) & (delta <= threshold)
    result = label.copy()
    result[ambiguous] = 255
    return result, int(ambiguous.sum())
