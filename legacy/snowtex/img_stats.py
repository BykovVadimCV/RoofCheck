"""Realism statistics used as the acceptance test (REALISM-RESEARCH-2 §1)."""
import sys, json
import numpy as np, cv2


def stats(gray):
    g = gray.astype(np.float32)
    lo, hi = np.percentile(g, 2), np.percentile(g, 98)
    t = (g - lo) / (hi - lo + 1e-6)
    mid = float(((t > 0.25) & (t < 0.75)).mean())
    h, w = g.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    F = np.abs(np.fft.fftshift(np.fft.fft2((g - g.mean()) * win))) ** 2
    yy, xx = np.indices(F.shape); r = np.hypot((yy - h / 2) / h, (xx - w / 2) / w)
    bins = np.linspace(0.02, 0.25, 24)
    idx = np.digitize(r.ravel(), bins)
    p = np.array([F.ravel()[idx == i].mean() for i in range(1, len(bins))])
    fc = (bins[1:] + bins[:-1]) / 2
    slope = float(np.polyfit(np.log(fc), np.log(p), 1)[0])
    hf = float(np.abs(cv2.Laplacian(g, cv2.CV_32F)).mean() / g.mean() * 100)
    return dict(mid=round(mid, 2), slope=round(slope, 2), hf=round(hf, 1))


if __name__ == "__main__":
    for spec in sys.argv[1:]:
        path, box = spec.split("@")
        y0, y1, x0, x1 = map(int, box.split(","))
        g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)[y0:y1, x0:x1]
        print(f"{path.split('/')[-1]:40s} {box:18s}", json.dumps(stats(g)))
