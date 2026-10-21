"""Measure how the host photo was imaged, so pasted snow goes through the same degradations.

Only cues that survive the network's ~518-px input are matched (see docs/research-notes.md):
blur and sharpening halo width, haze / contrast, brightness-dependent noise, highlight clipping,
JPEG quality. Sensor-level traces (CFA, PRNU, 8x8 grid alignment) are not emulated.

    p = estimate(img, roi_box)          # once per background photo
    layer = apply_optics(layer, p)      # blur + haze on the snow radiance / alpha
    out = finish(out, weight, p, rng)   # sharpening + noise near the snow, clipping
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import warnings

import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import curve_fit
from scipy.special import erf

# IJG standard luminance quantisation table (quality 50), row-major
from scipy.optimize import OptimizeWarning
warnings.simplefilter('ignore', OptimizeWarning)
_IJG_LUMA = np.array([
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99], np.float64)


@dataclass
class HostParams:
    sigma_psf: float = 0.8        # Gaussian blur of edges (px)
    overshoot: float = 0.05       # sharpening overshoot as a share of the edge step
    usm_amount: float = 0.3       # unsharp-mask amount that reproduces the overshoot
    usm_radius: float = 1.2
    nlf_a: float = 0.02           # noise variance = a * I + c   (I in 0..255)
    nlf_c: float = 1.0
    transmission: float = 0.9     # haze transmission at the roof (1 = clear)
    airlight: tuple = (200.0, 200.0, 200.0)
    clip_hi: tuple = (255.0, 255.0, 255.0)
    jpeg_quality: int = 90

    def as_dict(self):
        d = asdict(self)
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}


# ------------------------------------------------------------------ estimation
def _esf(s, lo, hi, s0, sig):
    return lo + (hi - lo) * 0.5 * (1 + erf((s - s0) / (np.sqrt(2) * max(sig, 0.05))))


def edge_blur(gray: np.ndarray, box, n_max=300) -> tuple[float, float]:
    """Median Gaussian sigma and sharpening overshoot from edge-spread profiles of strong edges."""
    x0, y0, x1, y1 = box
    g = cv2.GaussianBlur(gray, (0, 0), 0.5).astype(np.float32)
    sub = g[y0:y1, x0:x1]
    gx = cv2.Sobel(sub, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(sub, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    edges = cv2.Canny(sub.astype(np.uint8), 40, 120) > 0
    cand = np.argwhere(edges & (mag > np.percentile(mag[edges], 80) if edges.any() else False))
    if len(cand) < 20:
        return 0.8, 0.05
    rng = np.random.default_rng(0)
    cand = cand[rng.choice(len(cand), min(n_max, len(cand)), replace=False)]
    s = np.arange(-7, 7.01, 0.5)
    sigmas, overs = [], []
    for yy, xx in cand:
        nx, ny = gx[yy, xx], gy[yy, xx]
        nrm = np.hypot(nx, ny) + 1e-6
        nx, ny = nx / nrm, ny / nrm
        py, px = yy + y0 + s * ny, xx + x0 + s * nx
        prof = ndi.map_coordinates(gray.astype(np.float32), [py, px], order=1, mode="nearest")
        lo, hi = np.percentile(prof[:6], 50), np.percentile(prof[-6:], 50)
        if hi - lo < 25:
            continue
        try:
            with np.errstate(all='ignore'):
                popt, _ = curve_fit(_esf, s, prof, p0=[lo, hi, 0.0, 1.0], maxfev=400)
        except Exception:
            continue
        sig = abs(popt[3])
        if 0.2 < sig < 5:
            sigmas.append(sig)
            step = popt[1] - popt[0]
            overs.append(max(0.0, (prof.max() - popt[1]) / step) if step > 0 else 0.0)
    if len(sigmas) < 10:
        return 0.8, 0.05
    return float(np.median(sigmas)), float(np.clip(np.median(overs), 0, 0.5))


def usm_for_overshoot(sigma: float, overshoot: float, radius: float = 1.2) -> float:
    """Unsharp-mask amount that gives the measured overshoot on a step blurred with sigma (1-D search)."""
    x = np.arange(-40, 41, dtype=np.float32)
    step = (x > 0).astype(np.float32)
    b = ndi.gaussian_filter1d(step, sigma)
    lo, hi = 0.0, 3.0
    for _ in range(30):
        k = 0.5 * (lo + hi)
        y = b + k * (b - ndi.gaussian_filter1d(b, radius))
        if y.max() - 1 > overshoot:
            hi = k
        else:
            lo = k
    return float(0.5 * (lo + hi))


def noise_level_function(img: np.ndarray, patch=8) -> tuple[float, float]:
    """Fit variance = a*I + c on flat patches (lowest-variance patches per brightness bin)."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    res = g - cv2.GaussianBlur(g, (0, 0), 1.0)
    H, W = g.shape
    H8, W8 = H // patch * patch, W // patch * patch
    m = g[:H8, :W8].reshape(H8 // patch, patch, W8 // patch, patch).mean((1, 3))
    v = res[:H8, :W8].reshape(H8 // patch, patch, W8 // patch, patch).var((1, 3))
    grad = np.hypot(*np.gradient(cv2.GaussianBlur(g, (0, 0), 2)))[:H8, :W8].reshape(
        H8 // patch, patch, W8 // patch, patch).mean((1, 3))
    flat = grad < np.percentile(grad, 30)
    I, V = m[flat].ravel(), v[flat].ravel()
    bins = np.linspace(20, 245, 10)
    xs, ys = [], []
    for a, b in zip(bins[:-1], bins[1:]):
        sel = (I >= a) & (I < b) & (I < 250)
        if sel.sum() > 20:
            xs.append(I[sel].mean())
            ys.append(np.percentile(V[sel], 25))
    if len(xs) < 3:
        return 0.02, 1.0
    A = np.c_[xs, np.ones(len(xs))]
    a, c = np.linalg.lstsq(A, np.array(ys), rcond=None)[0]
    # residual of a 1-px Gaussian high-pass keeps ~0.45 of white-noise variance
    return float(max(a, 0) / 0.45), float(max(c, 0.2) / 0.45)


def haze(img: np.ndarray, box, patch=15) -> tuple[float, tuple]:
    """Dark-channel transmission near the roof. Bright grey roofs break the prior, so the dark end
    (windows, gaps, shadows at the same distance) is used: t = 1 - 0.95 * p10(dark) / A."""
    f = img.astype(np.float32)
    dark = cv2.erode(f.min(axis=2), np.ones((patch, patch), np.uint8))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lum = f.mean(2)
    sky = (hsv[..., 1] < 30) & (lum > np.percentile(lum, 97))
    A = f[sky].mean(0) if sky.sum() > 200 else np.percentile(f.reshape(-1, 3), 99, axis=0)
    x0, y0, x1, y1 = box
    d = dark[y0:y1, x0:x1]
    t = 1 - 0.95 * np.percentile(d, 10) / max(A.max(), 1)
    return float(np.clip(t, 0.35, 1.0)), tuple(float(a) for a in A)


def jpeg_quality_from_file(path) -> int | None:
    """IJG quality that best explains the file's luminance quantisation table."""
    try:
        from PIL import Image
        q = Image.open(path).quantization
    except Exception:
        return None
    if not q:
        return None
    t = np.array(q[0], np.float64)
    best, err = None, 1e18
    for qual in range(30, 101):
        sc = 5000 / qual if qual < 50 else 200 - 2 * qual
        ref = np.clip(np.floor((_IJG_LUMA * sc + 50) / 100), 1, 255)
        e = np.abs(np.sort(ref) - np.sort(t)).sum()
        if e < err:
            best, err = qual, e
    return int(best)


def estimate(img: np.ndarray, roi_box, path=None, ignore: np.ndarray | None = None) -> HostParams:
    H, W = img.shape[:2]
    x0, y0, x1, y1 = [int(v) for v in roi_box]
    pad_x, pad_y = int(0.15 * (x1 - x0)) + 20, int(0.6 * (y1 - y0)) + 20
    box = (max(0, x0 - pad_x), max(0, y0 - pad_y), min(W, x1 + pad_x), min(H, y1 + pad_y))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sig, ov = edge_blur(gray, box)
    a, c = noise_level_function(img)
    t, A = haze(img, box)
    keep = np.ones(img.shape[:2], bool) if ignore is None else ignore == 0      # stamps are not scene light
    clip = tuple(float(np.percentile(img[..., k][keep], 99.9)) for k in range(3))
    q = jpeg_quality_from_file(path) if path else None
    p = HostParams(sigma_psf=sig, overshoot=ov, nlf_a=a, nlf_c=c, transmission=t, airlight=A,
                   clip_hi=clip, jpeg_quality=q or 90)
    p.usm_amount = usm_for_overshoot(max(sig, 0.3), ov, p.usm_radius)
    return p


# ------------------------------------------------------------------ application
def apply_haze(rgb: np.ndarray, p: HostParams) -> np.ndarray:
    A = np.array(p.airlight, np.float32)
    return rgb * p.transmission + A * (1 - p.transmission)


def blur_layer(premult: np.ndarray, alpha: np.ndarray, p: HostParams):
    """Blur the snow layer as the lens did: premultiplied colour and alpha with the same PSF."""
    s = max(p.sigma_psf, 0.3)
    return cv2.GaussianBlur(premult, (0, 0), s), cv2.GaussianBlur(alpha, (0, 0), s)


def finish(out: np.ndarray, weight: np.ndarray, p: HostParams, rng) -> np.ndarray:
    """Sharpening halo and brightness-dependent noise where snow was added (weight 0..1), then clip."""
    w = weight[..., None]
    blur = cv2.GaussianBlur(out, (0, 0), p.usm_radius)
    sharp = out + p.usm_amount * (out - blur)
    out = out * (1 - w) + sharp * w
    lum = out.mean(2)
    var = np.clip(p.nlf_a * lum + p.nlf_c, 0.2, 80)
    n = rng.normal(0, 1, lum.shape).astype(np.float32)
    n = cv2.GaussianBlur(n, (0, 0), 0.6)          # demosaic / denoise leave correlated grain
    n /= n.std() + 1e-6
    chroma = cv2.GaussianBlur(rng.normal(0, 1, out.shape).astype(np.float32), (0, 0), 1.5) * 0.3
    out = out + (n[..., None] + chroma) * np.sqrt(var)[..., None] * w
    hi = np.array(p.clip_hi, np.float32)
    return np.clip(out, 0, hi[None, None, :])
