"""Stage 1 check: reference from the customer's markup, registration, overlays, rectification.

    python3 tests/stage1.py            -> out/stage1_report.json, out/stage1_*.jpg
"""
import glob
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from roofcheck import cvat, overlays
from roofcheck.customer_markup import extract_polygon, slope_quad
from roofcheck.rectify import (pixel_area, rectify, refine_quad_by_seams, seam_parallelism,
                               seam_parallelism_photo)
from roofcheck.register import Registrar, polygon_iou
from roofcheck.schema import ObjectInfo, Polygon, Reference

torch.set_num_threads(2)
RAW = ROOT / "data/raw"
OUT = ROOT / "out"
OUT.mkdir(exist_ok=True)
UPLOADS = Path("/root/.claude/uploads/242374d3-e5f2-5497-a897-a76cd251210f")
rng = np.random.default_rng(7)
report = {}

# ------------------------------------------------------------------ step 1-2: object, reference
obj_dir = ROOT / "data/objects/demo_roof_01"
vdir = obj_dir / "vantages/v1"
vdir.mkdir(parents=True, exist_ok=True)
ref_img = cv2.imread(str(RAW / "2025-10-09_103538.jpg"))
cv2.imwrite(str(vdir / "reference.jpg"), ref_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
cust = cv2.imread(str(RAW / "2025-10-09_103538_customer_polygon.jpg"))
roof = extract_polygon(cust)
quad0 = slope_quad(roof)
H0, W0 = ref_img.shape[:2]
roof_region = np.zeros((H0, W0), np.uint8)
cv2.fillPoly(roof_region, [np.round(roof).astype(np.int32)], 255)
ref_ign = overlays.detect(ref_img).mask
quad, refine_info = refine_quad_by_seams(ref_img, quad0, roof_region, ref_ign)
seams_before = seam_parallelism(rectify(ref_img, quad0, ref_ign).image, rectify(ref_img, quad0, ref_ign).valid)
seams_after = seam_parallelism(rectify(ref_img, quad, ref_ign).image, rectify(ref_img, quad, ref_ign).valid)
own_roof = [[1210, 1152], [1220, 1040], [1440, 940], [1800, 790], [2048, 720], [2048, 1152]]
ref = Reference(
    object_id="demo_roof_01", vantage_id="v1", image="reference.jpg", width=W0, height=H0,
    polygons=[Polygon("roof", roof.tolist(), "из красного контура заказчика, 09.10.2025"),
              Polygon("slope", quad.tolist(), "карниз-лево, карниз-право, конёк-право, конёк-лево; боковые стороны по фальцам"),
              Polygon("exclude", own_roof, "кровля, с которой ведётся съёмка")],
    source="customer red polygon (auto-extracted) + manual exclude", season="autumn")
# keep hand-drawn roles this script does not produce (structures, other roofs): they cannot be regenerated
if (vdir / "reference.json").exists():
    old = Reference.load(vdir / "reference.json")
    own = {"roof", "slope", "exclude"}
    ref.polygons += [p for p in old.polygons if p.role not in own]
    ref.geometry = getattr(old, "geometry", {}) or {}
ref.save(vdir)
ObjectInfo("demo_roof_01", address="(уточнить у заказчика)", vantages=["v1"]).save(obj_dir)
cvat.write_label_spec(OUT / "cvat_labels.json")
cvat.references_to_cvat_xml([ref], OUT / "cvat_references.xml")
back = cvat.polygons_from_cvat_xml(OUT / "cvat_references.xml")
report["step1_2"] = {"roof_polygon_vertices": len(roof), "slope_quad_customer": quad0.round(1).tolist(),
                     "slope_quad_refined": quad.round(1).tolist(), "seam_refine": refine_info,
                     "seams_rectified_customer_quad": seams_before, "seams_rectified_refined_quad": seams_after,
                     "cvat_roundtrip_polygons": sum(len(v) for v in back.values())}

# ------------------------------------------------------------------ step 4: overlays on all photos
ov = {}
for f in sorted(RAW.glob("2025-*.jpg")):
    if "customer" in f.name:
        continue
    im = cv2.imread(str(f))
    info = overlays.detect(im)
    ts, txt = overlays.read_timestamp(im, info)
    ov[f.name] = {"timestamp": ts.isoformat() if ts else None, "ocr": txt, "gps_fix": info.gps_fix,
                  "masked_frac": round(float((info.mask > 0).mean()), 3)}
report["step4_overlays"] = ov

# ------------------------------------------------------------------ step 3: registration
reg = Registrar()
t0 = time.time()
reg.reference_features(ref, ref_img)
t_ref = time.time() - t0


def draw(img, res, gt=None, title=""):
    v = img.copy()
    if gt is not None:
        cv2.polylines(v, [np.round(gt).astype(np.int32)], True, (0, 0, 255), 4)
    if res.H is not None:
        for p in res.polygons.get("roof", []):
            cv2.polylines(v, [np.round(p).astype(np.int32)], True, (60, 255, 60), 4)
        for p in res.polygons.get("slope", []):
            cv2.polylines(v, [np.round(p).astype(np.int32)], True, (255, 0, 255), 2)
    cv2.putText(v, title, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 6)
    cv2.putText(v, title, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 2)
    return cv2.resize(v, (1024, 576))


real = {}
tiles = []
cust_1356 = extract_polygon(cv2.imread(str(RAW / "2025-10-07_135619_customer_polygon.jpg")))
for name in ["2025-10-07_135619.jpg", "2025-10-07_135716.jpg", "2025-10-07_140441_tilted.jpg"]:
    im = cv2.imread(str(RAW / name))
    t0 = time.time()
    r = reg.register(im, ref, ref_img)
    dt = time.time() - t0
    q = r.quality
    d = {"ok": r.ok, "reason": r.reason, "sec": round(dt, 2), "inliers": q.inliers, "local_inliers": q.local_inliers,
         "reproj_px": round(q.reproj_px, 2), "scale": round(q.scale, 3), "rotation_deg": round(q.rotation_deg, 1),
         "spread": round(q.spread, 3)}
    gt = None
    if name.startswith("2025-10-07_135619"):
        gt = cust_1356
        d["iou_vs_customer_polygon"] = round(polygon_iou(r.polygons["roof"][0], gt, im.shape), 3)
    real[name] = d
    tiles.append(draw(im, r, gt, f"{name[:17]}  ok={r.ok}  inl={q.inliers}/{q.local_inliers}"))
report["step3_real"] = real
report["step3_ref_features_sec"] = round(t_ref, 2)

# cycle consistency: ref->13:56->13:57 vs ref->13:57 directly (slope corners, px)
im56 = cv2.imread(str(RAW / "2025-10-07_135619.jpg"))
im57 = cv2.imread(str(RAW / "2025-10-07_135716.jpg"))
r56 = reg.register(im56, ref, ref_img)
r57 = reg.register(im57, ref, ref_img)
ref56 = Reference("tmp", "v56", "x", im56.shape[1], im56.shape[0],
                  [Polygon("roof", r56.polygons["roof"][0].tolist()), Polygon("slope", r56.polygons["slope"][0].tolist())])
r56_57 = reg.register(im57, ref56, im56)
via = r56_57.polygons["slope"][0]
direct = r57.polygons["slope"][0]
report["step3_cycle_corner_err_px"] = {"median": round(float(np.median(np.linalg.norm(via - direct, axis=1))), 2),
                                       "max": round(float(np.max(np.linalg.norm(via - direct, axis=1))), 2)}

# negatives: photos of other places must be rejected
neg = {}
for f in ["b7be4569-image.jpg", "ab5e181a-image.jpg", "de08bc81-image.jpg", "9c90ee44-image.jpg"]:
    p = UPLOADS / f
    if not p.exists():
        continue
    im = cv2.imread(str(p))
    r = reg.register(im, ref, ref_img)
    neg[f] = {"ok": r.ok, "reason": r.reason, "inliers": r.quality.inliers, "local": r.quality.local_inliers}
report["step3_negatives"] = neg

# synthetic perturbations with known ground truth
stamp_mask = overlays.detect(ref_img).mask
clean = cv2.inpaint(ref_img, stamp_mask, 5, cv2.INPAINT_TELEA)       # remove the old stamp before warping


def random_view(img):
    H, W = img.shape[:2]
    z = rng.uniform(1.0, 3.0)
    rot = rng.uniform(-30, 30)
    roof_c = np.array(ref.by_role("roof")[0].points).mean(0)
    c = roof_c + rng.normal(0, [60, 40])
    A = cv2.getRotationMatrix2D(tuple(c), rot, z)
    A[:, 2] += np.array([W / 2, H / 2]) - c                      # keep the roof near the centre
    Hm = np.vstack([A, [0, 0, 1]])
    P = np.eye(3)
    P[2, :2] = rng.normal(0, 3e-5, 2)                            # mild perspective change
    Hm = P @ Hm
    out = cv2.warpPerspective(img, Hm, (W, H), borderMode=cv2.BORDER_REFLECT)
    haze = rng.uniform(0, 0.35)
    out = (out * (1 - haze) + 200 * haze).astype(np.uint8)
    out = cv2.convertScaleAbs(out, alpha=rng.uniform(0.85, 1.15), beta=rng.uniform(-15, 15))
    # the app stamps the new photo in the usual place
    txt = "07/10/2025 13:%02d:%02d +03:00" % (rng.integers(0, 60), rng.integers(0, 60))
    cv2.putText(out, txt, (165, 1035), cv2.FONT_HERSHEY_SIMPLEX, 3.6, (0, 255, 255), 9, cv2.LINE_AA)
    ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(40, 80))])
    return cv2.imdecode(enc, 1), Hm, dict(zoom=z, rot=rot, haze=haze)


syn = []
for k in range(40):
    im, Hm, par = random_view(clean)
    gt_roof = cv2.perspectiveTransform(np.array(ref.by_role("roof")[0].points, np.float64)[None], Hm)[0]
    gt_quad = cv2.perspectiveTransform(np.array(quad, np.float64)[None], Hm)[0]
    r = reg.register(im, ref, ref_img)
    rec = dict(par, ok=r.ok, reason=r.reason, inliers=r.quality.inliers, local=r.quality.local_inliers,
               visible=r.quality.visible_frac)
    if r.H is not None:
        rec["iou"] = polygon_iou(r.polygons["roof"][0], gt_roof, im.shape)
        rec["corner_err_px"] = float(np.max(np.linalg.norm(r.polygons["slope"][0] - gt_quad, axis=1)))
    syn.append(rec)
    if k < 3:
        tiles.append(draw(im, r, gt_roof, f"synthetic z={par['zoom']:.1f} rot={par['rot']:.0f} ok={r.ok}"))
ious = np.array([s.get("iou", 0) for s in syn])
oks = np.array([s["ok"] for s in syn])
report["step3_synthetic"] = {
    "n": len(syn), "accepted": int(oks.sum()),
    "iou_accepted_min": round(float(ious[oks].min()), 3) if oks.any() else None,
    "iou_accepted_median": round(float(np.median(ious[oks])), 3) if oks.any() else None,
    "accepted_with_iou_below_0.9": int(((ious < 0.9) & oks).sum()),
    "rejected_but_iou_ok": int(((ious >= 0.9) & ~oks).sum()),
    "corner_err_px_median": round(float(np.median([s.get("corner_err_px", np.nan) for s in syn if s["ok"]])), 2),
    "by_zoom": {f"{a}-{b}": round(float(np.mean([s.get("iou", 0) for s in syn if a <= s["zoom"] < b])), 3)
                for a, b in [(1, 1.7), (1.7, 2.4), (2.4, 3.01)]},
    "rejected_reasons": sorted({s["reason"] for s in syn if not s["ok"]}),
    "rejected_visible_frac": [round(s["visible"], 2) for s in syn if not s["ok"]],
    "iou_partial_roof_accepted": [round(s["iou"], 3) for s in syn if s["ok"] and s["visible"] < 0.99],
}

# ------------------------------------------------------------------ step 5: rectification
rect = {}
rect_tiles = []
for name in ["2025-10-09_103538.jpg", "2025-10-07_135619.jpg", "2025-10-07_140441_tilted.jpg"]:
    im = cv2.imread(str(RAW / name))
    info = overlays.detect(im)
    if name == "2025-10-09_103538.jpg":
        q4 = np.array(quad)
    else:
        q4 = reg.register(im, ref, ref_img).polygons["slope"][0]
    rr = rectify(im, q4, ignore=info.mask)
    sp_r = seam_parallelism(rr.image, rr.valid)
    sp_p = seam_parallelism_photo(im, q4, info.mask)
    area = rr.area_photo
    m = area > 0
    rect[name] = {"rect_size": list(rr.image.shape[:2]), "valid_frac": round(float((rr.valid > 0).mean()), 3),
                  "seams_rectified": sp_r, "seams_photo": sp_p,
                  "area_ratio_far_vs_near_px": round(float(np.percentile(area[m], 95) / np.percentile(area[m], 5)), 2)}
    big = cv2.resize(rr.image, (1024, max(60, int(1024 * rr.image.shape[0] / rr.image.shape[1] * 3))))  # depth x3 for viewing
    lab = np.zeros((40, 1024, 3), np.uint8)
    cv2.putText(lab, f"{name[:17]} rectified slope (depth x3 for display)", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    rect_tiles += [lab, big]
report["step5_rectify"] = rect

# ------------------------------------------------------------------ outputs
(OUT / "stage1_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str))
rows = [np.hstack(tiles[i:i + 2]) if i + 1 < len(tiles) else np.hstack([tiles[i], np.zeros_like(tiles[i])])
        for i in range(0, len(tiles), 2)]
cv2.imwrite(str(OUT / "stage1_registration.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
cv2.imwrite(str(OUT / "stage1_rectified.jpg"), np.vstack(rect_tiles), [cv2.IMWRITE_JPEG_QUALITY, 88])
print(json.dumps(report, ensure_ascii=False, indent=1, default=str))
