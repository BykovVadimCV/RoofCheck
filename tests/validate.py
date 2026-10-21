"""Checks of the synthetic data that do not need real snow photos, plus a hook for when they arrive.

1. Null-paste test. Every background goes through the full compositing chain with zero snow. A classifier
   on low-level patch statistics (spectrum, noise, gradients, JPEG blockiness) then tries to tell the
   null output from the untouched photo, leave-one-background-out. Accuracy near 0.5 means the pipeline
   leaves no trace a network could use instead of looking at snow. Run at native resolution and at the
   model input scale (518 px on the long side). A positive control (extra noise + second JPEG at q=80)
   shows the test is sensitive enough to catch a trace of that size.
2. Label checks on out/synth: overlays never labelled snow, the ROI never covers overlays, depth >= T
   matches the binary label outside the ignore band, coverage in the json matches the depth map.
3. --real DIR: the same low-level statistics on real snow-roof crops vs synthetic snow crops (C2ST).
   Without real data this step is skipped; the result is only meaningful against a real-vs-real floor.

    python3 tests/validate.py [--real DIR]
"""
import argparse, json, sys
from pathlib import Path
import cv2, numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from roofcheck import synth, overlays, hostmatch
from roofcheck.schema import Reference
from roofcheck.register import Registrar

PATCH = 64


def patch_features(p: np.ndarray) -> np.ndarray:
    """Low-level statistics of a BGR patch: spectral slope and HF share, high-pass residual moments,
    gradient histogram, 8x8 blockiness, chroma noise."""
    g = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY).astype(np.float32)
    F = np.abs(np.fft.fftshift(np.fft.fft2(g - g.mean()))) + 1e-3
    h, w = g.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(yy - h / 2, xx - w / 2).astype(int)
    rad = np.bincount(r.ravel(), F.ravel()) / np.maximum(np.bincount(r.ravel()), 1)
    k = np.arange(2, min(h, w) // 2)
    slope = np.polyfit(np.log(k), np.log(rad[k]), 1)[0]
    hf = rad[min(h, w) // 4:min(h, w) // 2].sum() / rad[1:min(h, w) // 2].sum()
    res = g - cv2.GaussianBlur(g, (0, 0), 1.0)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    mag = np.hypot(gx, gy)
    hist = np.histogram(np.log1p(mag), bins=6, range=(0, 6))[0] / mag.size
    d = np.abs(np.diff(g, axis=1))
    block = d[:, 7::8].mean() / (d.mean() + 1e-3)
    lab = cv2.cvtColor(p, cv2.COLOR_BGR2LAB).astype(np.float32)
    cres = lab[..., 1:] - cv2.GaussianBlur(lab[..., 1:], (0, 0), 1.0)
    return np.r_[slope, hf, res.std(), np.abs(res).mean(), ((res - res.mean()) ** 4).mean() / (res.var() ** 2 + 1e-6),
                 hist, block, cres.std(), g.mean() / 255, g.std() / 255]


def patches(img, mask, n, rng, size=PATCH):
    H, W = img.shape[:2]
    ys, xs = np.nonzero(mask[size // 2:H - size // 2, size // 2:W - size // 2])
    if len(ys) == 0:
        return []
    idx = rng.choice(len(ys), min(n, len(ys)), replace=False)
    return [img[y:y + size, x:x + size] for y, x in zip(ys[idx], xs[idx])]


def c2st(X, y, groups):
    """Leave-one-group-out logistic regression accuracy (0.5 = indistinguishable)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    accs = []
    for g in np.unique(groups):
        tr, te = groups != g, groups == g
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
        clf.fit(X[tr], y[tr])
        accs.append(clf.score(X[te], y[te]))
    return float(np.mean(accs)) if accs else float("nan"), [round(a, 3) for a in accs]


def resize_long(img, L=518):
    s = L / max(img.shape[:2])
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)


def degrade_control(img, rng):
    """Positive control: a trace the test must catch (extra noise sigma 1.5 and a second JPEG at q=80)."""
    n = img.astype(np.float32) + rng.normal(0, 1.5, img.shape)
    ok, enc = cv2.imencode(".jpg", np.clip(n, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 80])
    return cv2.imdecode(enc, 1)


FEATURE_NAMES = ["spectral_slope", "hf_share", "res_std", "res_mad", "res_kurt"] + [f"grad_h{i}" for i in range(6)] + \
    ["blockiness", "chroma_noise", "mean", "std"]


def null_paste(bgs, geom, reps=3, control=False):
    """Two views of the same question. C2ST: unpaired patches, as a network would see them. Paired: the
    same patch locations before and after, effect = median difference / spread across patches; this is
    far more sensitive and says which statistic moved."""
    out = {}
    for scale_name, f in (("native", lambda a: a), ("518px", resize_long)):
        X, y, grp, D, F0 = [], [], [], [], []
        for gi, (name, im, path, P) in enumerate(bgs):
            q = np.asarray(P["slope"][0])
            ov = overlays.detect(im).mask
            host = hostmatch.estimate(im, (q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()), path=path, ignore=ov)
            m = np.zeros(im.shape[:2], np.uint8)
            cv2.fillPoly(m, [np.round(q).astype(np.int32)], 1)
            m = (m > 0) & (ov == 0)
            size = PATCH if scale_name == "native" else 24
            for r in range(reps):
                s = synth.make_sample(im, q, P["roof"][0], np.random.default_rng(100 + r), geom, host=host,
                                      other_roofs=P["other_roof"], structures=P["structure"], null=True)
                if control:
                    s.image = degrade_control(s.image, np.random.default_rng(200 + r))
                feats = {}
                for lab, img in ((0, im), (1, s.image)):
                    a = f(img)
                    mm = cv2.resize(m.astype(np.uint8), (a.shape[1], a.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
                    feats[lab] = np.array([patch_features(p) for p in patches(a, mm, 60, np.random.default_rng(gi * 10 + r), size)])
                n = min(len(feats[0]), len(feats[1]))
                half = n // 2               # unpaired: original from one half of the locations, processed from the other
                for lab, sl in ((0, slice(0, half)), (1, slice(half, n))):
                    X += list(feats[lab][sl]); y += [lab] * len(feats[lab][sl]); grp += [gi] * len(feats[lab][sl])
                D.append(feats[1][:n] - feats[0][:n]); F0.append(feats[0][:n])
        acc, per = c2st(np.array(X), np.array(y), np.array(grp))
        D, F0 = np.vstack(D), np.vstack(F0)
        # paired: share of locations where the statistic went up (sign test; 0.5 = no systematic change)
        # and the shift relative to the natural spread of that statistic across the roof
        up = (D > 0).mean(0) + 0.5 * (D == 0).mean(0)
        shift = np.median(D, 0) / (F0.std(0) + 1e-9)
        top = np.argsort(-np.abs(up - 0.5))[:3]
        out[scale_name] = {"c2st_accuracy": round(acc, 3), "per_background": per, "n": len(y),
                           "paired_max_sign_bias": round(float(np.abs(up - 0.5).max()), 3),
                           "paired_top": {FEATURE_NAMES[i]: {"share_up": round(float(up[i]), 3), "shift_in_spread": round(float(shift[i]), 3)}
                                          for i in top}}
    return out


def label_checks():
    res = {"samples": 0, "overlay_as_snow_px": 0, "roi_on_overlay_px": 0, "label_mismatch_px": 0}
    for js in sorted((ROOT / "out/synth").glob("*.json")):
        meta = json.loads(js.read_text())
        sid = meta["id"]
        img = cv2.imread(str(ROOT / f"out/synth/{sid}.jpg"))
        cls = cv2.imread(str(ROOT / f"out/synth/{sid}_cls.png"), 0)
        roi = cv2.imread(str(ROOT / f"out/synth/{sid}_roi.png"), 0) > 0
        depth = cv2.imread(str(ROOT / f"out/synth/{sid}_depth.png"), -1) / 10.0
        ov = overlays.detect(img).mask > 0
        res["samples"] += 1
        res["overlay_as_snow_px"] += int((ov & (cls == synth.CLS["snow"])).sum())
        res["roi_on_overlay_px"] += int((ov & roi).sum())
        for T in synth.THRESHOLDS_MM:
            lab = cv2.imread(str(ROOT / f"out/synth/{sid}_t{T}.png"), 0)
            ok = lab != 255
            res["label_mismatch_px"] += int(((depth[ok] >= T) != (lab[ok] == 1)).sum())
    return res


def real_vs_synth(real_dir: Path):
    """Snow crops: real (any jpg in real_dir with a *_snow.png mask, or whole image) vs synthetic snow."""
    rng = np.random.default_rng(3)
    X, y, grp = [], [], []
    files = sorted(real_dir.glob("*.jpg"))
    for i, f in enumerate(files):
        im = cv2.imread(str(f))
        mp = f.with_name(f.stem + "_snow.png")
        m = cv2.imread(str(mp), 0) > 0 if mp.exists() else np.ones(im.shape[:2], bool)
        for p in patches(resize_long(im), cv2.resize(m.astype(np.uint8), resize_long(im).shape[1::-1]) > 0, 30, rng, 24):
            X.append(patch_features(p)); y.append(0); grp.append(i % 5)
    for i, js in enumerate(sorted((ROOT / "out/synth").glob("*.json"))):
        sid = json.loads(js.read_text())["id"]
        im = cv2.imread(str(ROOT / f"out/synth/{sid}.jpg"))
        d = cv2.imread(str(ROOT / f"out/synth/{sid}_depth.png"), -1) / 10.0
        a = resize_long(im)
        m = cv2.resize((d >= 20).astype(np.uint8), a.shape[1::-1], interpolation=cv2.INTER_NEAREST) > 0
        for p in patches(a, m, 30, rng, 24):
            X.append(patch_features(p)); y.append(1); grp.append(i % 5)
    if len(set(y)) < 2:
        return {"skipped": "no patches"}
    acc, per = c2st(np.array(X), np.array(y), np.array(grp))
    return {"accuracy": round(acc, 3), "folds": per, "n_real": int((np.array(y) == 0).sum()), "n_synth": int(np.sum(y)),
            "note": "compare with a real-vs-real split of the same data before reading anything into it"}


def load_backgrounds():
    vdir = ROOT / "data/objects/demo_roof_01/vantages/v1"
    ref = Reference.load(vdir / "reference.json"); ref_img = cv2.imread(str(vdir / "reference.jpg"))
    roles = ("roof", "slope", "other_roof", "structure")
    bgs = [("ref", ref_img, vdir / "reference.jpg", {r: [np.array(p.points) for p in ref.by_role(r)] for r in roles})]
    reg = Registrar()
    for name in ["2025-10-07_135619", "2025-10-07_135716", "2025-10-07_140441_tilted"]:
        path = ROOT / f"data/raw/{name}.jpg"
        im = cv2.imread(str(path))
        r = reg.register(im, ref, ref_img)
        if r.ok:
            bgs.append((name, im, path, {k: r.polygons.get(k, []) for k in roles}))
    return ref, bgs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", type=Path)
    a = ap.parse_args()
    ref, bgs = load_backgrounds()
    geom = synth.roof_geometry(ref.object_id, getattr(ref, "geometry", None))
    report = {"null_paste": null_paste(bgs, geom), "positive_control": null_paste(bgs, geom, reps=2, control=True),
              "labels": label_checks()}
    if a.real:
        report["real_vs_synth"] = real_vs_synth(a.real)
    print(json.dumps(report, indent=1, ensure_ascii=False))
    (ROOT / "out/validate.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
