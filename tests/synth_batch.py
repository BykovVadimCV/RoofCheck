"""Stage 2 batch: synthetic samples on every photo of the demo vantage point.

Backgrounds: the reference and the other autumn photos of the same point (polygons carried over by
registration). Writes per sample: <id>.jpg, <id>_depth.png (uint16, 0.1 mm), <id>_cls.png, <id>.json,
plus out/synth/index.csv and a realism report.
"""
import csv, json, sys, time
from pathlib import Path
import cv2, numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from roofcheck import synth, overlays
from roofcheck.register import Registrar
from roofcheck.schema import Reference
torch.set_num_threads(2)
N_PER_BG = int(sys.argv[1]) if len(sys.argv) > 1 else 8
OUT = ROOT / "out/synth"; OUT.mkdir(parents=True, exist_ok=True)
vdir = ROOT / "data/objects/demo_roof_01/vantages/v1"
ref = Reference.load(vdir / "reference.json"); ref_img = cv2.imread(str(vdir / "reference.jpg"))
reg = Registrar()
bgs = [("ref_2025-10-09_103538", ref_img, {r: [np.array(p.points) for p in ref.by_role(r)] for r in ("roof", "slope", "other_roof")})]
for name in ["2025-10-07_135619", "2025-10-07_135716", "2025-10-07_140441_tilted"]:
    im = cv2.imread(str(ROOT / f"data/raw/{name}.jpg"))
    r = reg.register(im, ref, ref_img)
    if r.ok:
        bgs.append((name, im, {k: r.polygons.get(k, []) for k in ("roof", "slope", "other_roof")}))
rng = np.random.default_rng(2026)
rows = []
for bg_name, im, P in bgs:
    for k in range(N_PER_BG):
        sc = synth.SCENARIOS[k % len(synth.SCENARIOS)]
        t = time.time()
        s = synth.make_sample(im, P["slope"][0], P["roof"][0], rng, scenario=sc, other_roofs=P["other_roof"])
        sid = f"{bg_name}_{k:03d}_{sc}"
        cv2.imwrite(str(OUT / f"{sid}.jpg"), s.image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(OUT / f"{sid}_depth.png"), np.clip(s.depth_mm * 10, 0, 65535).astype(np.uint16))
        cv2.imwrite(str(OUT / f"{sid}_cls.png"), s.classes)
        meta = {"id": sid, "background": bg_name, "coverage": {str(k_): v for k_, v in s.coverage.items()}, **s.info,
                "sec": round(time.time() - t, 2)}
        (OUT / f"{sid}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
        rows.append({"id": sid, "background": bg_name, "scenario": sc, **{f"cov_{k_}mm": round(v, 3) for k_, v in s.coverage.items()},
                     "mean_depth_mm": s.info["mean_depth_mm"], "sec": meta["sec"]})
with open(OUT / "index.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(len(rows), "samples;", "mean sec", round(np.mean([r["sec"] for r in rows]), 2))
cov2 = np.array([r["cov_2mm"] for r in rows])
print("coverage>=2mm histogram (0-1, 5 bins):", np.histogram(cov2, bins=5, range=(0, 1))[0].tolist())
