import sys, json, time
from pathlib import Path
import cv2, numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from roofcheck import synth
from roofcheck.schema import Reference
ref = Reference.load(ROOT / "data/objects/demo_roof_01/vantages/v1/reference.json")
img = cv2.imread(str(ROOT / "data/objects/demo_roof_01/vantages/v1/reference.jpg"))
quad = np.array(ref.by_role("slope")[0].points); roof = np.array(ref.by_role("roof")[0].points)
rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
tiles = []; infos = []
x0, y0, x1, y1 = 0, 60, 2048, 700          # crop around the target roof for viewing
for sc in synth.SCENARIOS:
    t = time.time(); s = synth.make_sample(img, quad, roof, rng, scenario=sc, other_roofs=[np.array(p.points) for p in ref.by_role('other_roof')]); dt = time.time() - t
    crop = s.image[y0:y1, x0:x1]
    lab = (s.classes[y0:y1, x0:x1] == synth.CLS["snow"])
    ov = crop.copy(); ov[lab] = (0.5 * ov[lab] + [0, 60, 230]).clip(0, 255)
    pane = np.hstack([cv2.resize(crop, (800, 250)), cv2.resize(ov, (800, 250))])
    txt = f"{sc}: fall {s.info['fall_mm']:.0f}mm cov>=2mm {s.coverage[2]:.2f} >=20mm {s.coverage[20]:.2f} ({dt:.1f}s)"
    cv2.rectangle(pane, (0, 0), (1600, 34), (0, 0, 0), -1); cv2.putText(pane, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    tiles.append(pane); infos.append({"coverage": s.coverage, **s.info, "sec": round(dt, 1)})
    if sc == "fresh":
        cv2.imwrite(str(ROOT / "out/synth_full_fresh.jpg"), cv2.resize(s.image, (1024, 576)))
cv2.imwrite(str(ROOT / "out/synth_preview.jpg"), np.vstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 88])
print(json.dumps(infos, ensure_ascii=False, indent=0))
