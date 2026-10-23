"""Fine-detail check: high-frequency energy of synthetic snow vs the photo's own surfaces.
HF = mean |Laplacian| of luminance. Compared on the same photo: snow (>=20 mm, away from edges) vs
the roof metal it replaced (same pixels, original photo) and vs a plain facade box."""
import sys, glob, json, re
from pathlib import Path
import cv2, numpy as np
ROOT = Path(__file__).resolve().parents[1]
def hf(g, m):
    L = np.abs(cv2.Laplacian(g, cv2.CV_32F, ksize=3)); return float(L[m].mean()) if m.sum() > 300 else np.nan
orig = {"ref_2025-10-09_103538": ROOT/"data/objects/demo_roof_01/vantages/v1/reference.jpg"}
for n in ["2025-10-07_135619", "2025-10-07_135716", "2025-10-07_140441_tilted"]: orig[n] = ROOT/f"data/raw/{n}.jpg"
res = {}
for d in sys.argv[1:]:
    r_snow, r_fac = [], []
    for f in sorted(glob.glob(f"{d}/*_cls.png")):
        sid = Path(f).name[:-8]; bg = re.sub(r"_(\d{3}|neg\d+)_.*$", "", sid)
        im = cv2.imread(f"{d}/{sid}.jpg", 0).astype(np.float32); o = cv2.imread(str(orig[bg]), 0).astype(np.float32)
        cls = cv2.imread(f, 0); dep = cv2.imread(f"{d}/{sid}_depth.png", -1) / 10.0
        m = cv2.erode(((cls == 1) & (dep >= 20)).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        fac = np.zeros_like(m); fac[560:760, 420:900] = True          # plain facade below the roof
        a, b, c = hf(im, m), hf(o, m), hf(im, fac)
        if not np.isnan(a): r_snow.append(a / b); r_fac.append(a / c)
    res[d] = {"n": len(r_snow), "snow_vs_metal_under_it": round(float(np.median(r_snow)), 3), "snow_vs_facade": round(float(np.median(r_fac)), 3)}
print(json.dumps(res, indent=1))
