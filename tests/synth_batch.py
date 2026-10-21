"""Stage 2 batch: synthetic samples on every photo of the demo vantage point.

Backgrounds: the reference and the other autumn photos of the same point (polygons carried over by
registration). Roof geometry is fixed per object (reference.geometry or a prior seeded by the object id);
camera parameters are measured once per background. Per sample:
  <id>.jpg               image (overlays left in place, as the app stamps them)
  <id>_net.jpg           network input: overlays blanked
  <id>_depth.png         uint16, 0.1 mm, all snow-bearing surfaces
  <id>_cls.png           class map (schema.SEG_CLASSES)
  <id>_roi.png           visible target slope
  <id>_t{T}.png          binary label at threshold T mm (0/1, 255 = ignore band / overlays)
  <id>.json              parameters
"""
import csv, json, sys, time
from pathlib import Path
import cv2, numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from roofcheck import synth, overlays, hostmatch
from roofcheck.register import Registrar
from roofcheck.schema import Reference
torch.set_num_threads(2)
N_PER_BG = int(sys.argv[1]) if len(sys.argv) > 1 else 8
OUT = ROOT / "out/synth"; OUT.mkdir(parents=True, exist_ok=True)
for f in OUT.glob("*"):
    f.unlink()
OBJ = "demo_roof_01"
vdir = ROOT / f"data/objects/{OBJ}/vantages/v1"
ref = Reference.load(vdir / "reference.json"); ref_img = cv2.imread(str(vdir / "reference.jpg"))
geom = synth.roof_geometry(OBJ, getattr(ref, "geometry", None))
print("geometry:", geom)
reg = Registrar()
bgs = [("ref_2025-10-09_103538", ref_img, vdir / "reference.jpg",
        {r: [np.array(p.points) for p in ref.by_role(r)] for r in ("roof", "slope", "other_roof", "structure")})]
for name in ["2025-10-07_135619", "2025-10-07_135716", "2025-10-07_140441_tilted"]:
    path = ROOT / f"data/raw/{name}.jpg"
    im = cv2.imread(str(path))
    r = reg.register(im, ref, ref_img)
    if r.ok:
        bgs.append((name, im, path, {k: r.polygons.get(k, []) for k in ("roof", "slope", "other_roof", "structure")}))
rng = np.random.default_rng(2026)
rows = []
for bg_name, im, path, P in bgs:
    q = np.asarray(P["slope"][0])
    host = hostmatch.estimate(im, (q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()), path=path,
                              ignore=overlays.detect(im).mask)
    print(bg_name, host.as_dict())
    for k in range(N_PER_BG):
        st = synth.STATES[k % len(synth.STATES)]
        t = time.time()
        s = synth.make_sample(im, q, P["roof"][0], rng, geom, host=host, other_roofs=P["other_roof"],
                             structures=P["structure"], state=st)
        sid = f"{bg_name}_{k:03d}_{st}"
        cv2.imwrite(str(OUT / f"{sid}.jpg"), s.image, [cv2.IMWRITE_JPEG_QUALITY, 97])
        cv2.imwrite(str(OUT / f"{sid}_net.jpg"), overlays.blank(s.image), [cv2.IMWRITE_JPEG_QUALITY, 97])
        cv2.imwrite(str(OUT / f"{sid}_depth.png"), np.clip(s.depth_mm * 10, 0, 65535).astype(np.uint16))
        cv2.imwrite(str(OUT / f"{sid}_cls.png"), s.classes)
        cv2.imwrite(str(OUT / f"{sid}_roi.png"), s.roi.astype(np.uint8) * 255)
        for T in synth.THRESHOLDS_MM:
            cv2.imwrite(str(OUT / f"{sid}_t{T}.png"), synth.labels_at(s.depth_mm, s.classes, T))
        meta = {"id": sid, "background": bg_name, "coverage": {str(k_): v for k_, v in s.coverage.items()}, **s.info,
                "sec": round(time.time() - t, 2)}
        (OUT / f"{sid}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=float))
        rows.append({"id": sid, "background": bg_name, "state": st,
                     **{f"cov_{k_}mm": round(v, 3) for k_, v in s.coverage.items()},
                     "ground_mm": round(s.info["ground_mm"], 0), "mean_depth_mm": s.info["mean_depth_mm"],
                     "p95_depth_mm": s.info["p95_depth_mm"], "mass_ratio": s.info["mass_ratio"],
                     "eave_face_px": s.info["eave_face_px"], "cap_px": s.info["cap_px"], "sec": meta["sec"]})
with open(OUT / "index.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(len(rows), "samples;", "mean sec", round(np.mean([r["sec"] for r in rows]), 2))
for T in (2, 300):
    c = np.array([r[f"cov_{T}mm"] for r in rows])
    print(f"coverage>={T}mm histogram (5 bins):", np.histogram(c, bins=5, range=(0, 1))[0].tolist())
