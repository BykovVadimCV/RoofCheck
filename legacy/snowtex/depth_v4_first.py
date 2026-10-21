"""
v4 — depth-first snow generator.

One quantity drives everything: snow depth D (mm) on the ground, plus a compaction map.
  macro (top-down metric grid, 15 px/m): snowfall + drift, plow passes (residual film with scrape
        striations, removed volume deposited as windrows), edge piles, occupied stalls and the
        car-shaped holes left when cars drive off, melt
  micro (per image pixel, from ground coordinates): cm-scale unevenness, tyre tracks (compaction,
        groove packing, side ridges, ruts in deep snow), footprints
Outputs per frame:
  image, depth_mm (per pixel, ground footprint), compaction, label at threshold tau,
  coverage at several thresholds (ground m2 in the inspection zone), mean / p95 depth.
Coverage is targeted by bisection on the melt amount at the macro level (label threshold tau).
"""
import json
import numpy as np
import cv2
from scipy import ndimage as ndi
import render_snow_v2 as R

G, P, X0, Y0 = R.G, R.P, R.X0, R.Y0
IMG, GX, GY, FOOT = R.IMG, R.GX, R.GY, R.FOOT
H, W = IMG.shape[:2]
SURF = R.surface_mask()
LAY = R.bev_layout(SURF)
BSURF = LAY["surface"]
BH_, BW_ = BSURF.shape
ZONE_B = R.bev_zone(BSURF.shape) & BSURF
zx0, zy0, zx1, zy1 = R.INSPECTION_ZONE
ZONE_I = SURF & (GX > zx0) & (GX < zx1) & (GY > zy0) & (GY < zy1)
AREA = FOOT ** 2
HC = 7.0
_h = R.H_i2m[2]
V_HOR = float(-(_h[0] * (W / 2) + _h[2]) / _h[1])
uu, vv = R.uu, R.vv
img_lin = R.srgb_to_lin(IMG)
m = lambda x: int(round(x * P))
BY, BX = np.mgrid[0:BH_, 0:BW_]
BYm = BY / P + Y0                                        # BEV pixel -> metres
BXm = BX / P + X0


def gnoise(scale_m, octaves, seed):
    """Fractal noise at image pixels, evaluated in ground coordinates, octaves below 1 px dropped."""
    r = np.random.default_rng(seed)
    out = np.zeros(GX.shape, np.float32); amp = 1.0; tot = 0
    for o in range(octaves):
        s = scale_m / 2 ** o
        grid = r.random((256, 256)).astype(np.float32)
        val = cv2.remap(grid, ((GX / s) % 256).astype(np.float32), ((GY / s) % 256).astype(np.float32),
                        cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
        fade = np.clip(1.5 - FOOT / s, 0, 1)
        out += amp * (val * fade + 0.5 * (1 - fade)); tot += amp; amp *= 0.55
    return out / tot


def to_img(a):
    return cv2.remap(a.astype(np.float32), R.BEV_MX, R.BEV_MY, cv2.INTER_LINEAR, borderValue=0)


# ============================================================ MACRO (BEV, mm)
def macro_depth(sc, rng):
    dist_edge = ndi.distance_transform_edt(np.pad(BSURF, 1))[1:-1, 1:-1] / P
    fall = sc["fall_mm"]
    D = fall * (0.75 + 0.5 * G.fbm(BH_, BW_, 8.0, 4, rng))
    D += fall * 0.8 * np.exp(-dist_edge / 2.5) * G.fbm(BH_, BW_, 3.0, 3, rng)   # drift against edges
    comp = np.zeros_like(D)                                                       # compaction 0..1
    cleared = np.zeros(D.shape, bool)
    windrow = np.zeros_like(D)

    if sc.get("plow_frac", 0) > 0:
        # aisles always plowed, stalls until the requested share is reached; occupied stalls skipped
        occupied = rng.random(len(LAY["stalls"])) < sc.get("occupied", 0.3)
        targets = [("aisle", 0, a0, LAY["W"], a1) for (a0, a1) in LAY["aisles"]]
        free = [s for s, o in zip(LAY["stalls"], occupied) if not o]
        rng.shuffle(free)
        n_st = int(sc["plow_frac"] * len(free))
        targets += [("stall",) + tuple(s) for s in free[:n_st]]
        strip = np.zeros(D.shape, bool)
        for t in targets:
            _, x0, y0, x1, y1 = t
            r = (slice(max(int(y0 * P), 0), max(int(y1 * P), 0)), slice(int(x0 * P), int(x1 * P)))
            strip[r] = True
        strip &= BSURF
        removed = np.where(strip, D, 0.0)
        # residual film: blade leaves 0.5-4 mm, with scrape striations along the pass direction (x)
        stri = 0.5 + 0.5 * np.sin(2 * np.pi * BYm / rng.uniform(0.05, 0.09) + 3 * G.fbm(BH_, BW_, 1.0, 2, rng))
        film = sc.get("film_mm", 2.0) * (0.3 + 0.9 * G.fbm(BH_, BW_, 2.0, 3, rng)) * (0.5 + 0.5 * stri)
        D = np.where(strip, film, D)
        comp = np.where(strip, 0.6, comp)
        cleared = strip
        # windrows: removed volume per metre of aisle goes to one side of each aisle (triangle, 0.9 m wide)
        for (a0, a1) in LAY["aisles"]:
            ya, yb = int(a0 * P), int(a1 * P)
            if yb <= 0 or ya >= BH_:
                continue
            vol = removed[max(ya, 0):min(yb, BH_)].sum(0) / P              # mm * m per column
            side = a1 if rng.random() < 0.5 else a0
            wr = 0.9
            prof = np.clip(1 - np.abs(BY / P - (side + (0.45 if side == a1 else -0.45))) / (wr / 2), 0, 1)
            hgt = 2 * vol / wr * 0.5                                     # half goes to the end piles
            windrow += prof * hgt[None, :] * (0.6 + 0.8 * G.fbm(BH_, BW_, 0.6, 3, rng))
        # car-shaped holes: some occupied stalls were vacated after snowfall
        for s, o in zip(LAY["stalls"], occupied):
            if o and rng.random() < sc.get("left", 0.5):
                x0, y0, x1, y1 = s
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                hole = (np.abs(BXm - (cx + X0)) < 0.9) & (np.abs(BYm - (cy + Y0)) < 2.2)
                D = np.where(hole, D * 0.08, D)
        D = D + windrow * BSURF
    # edge piles from the plowed volume
    if sc.get("piles", 0):
        pile = R.edge_piles(LAY, rng, sc["piles"]).astype(np.float32)
        pile = ndi.gaussian_filter(pile, 1.2 * P)
        D = D + 900 * pile * (0.7 + 0.6 * G.fbm(BH_, BW_, 1.0, 3, rng))
    D = D * BSURF
    return D.astype(np.float32), comp.astype(np.float32), cleared


def apply_melt(D, Mmm, noise):
    return np.maximum(D - Mmm * (0.6 + 0.8 * noise), 0)


def target_melt(D, tau, target, noise):
    """Bisection on melt depth so that coverage(D > tau) in the zone hits the target."""
    cov = lambda M: (apply_melt(D, M, noise) > tau)[ZONE_B].mean()
    if cov(0) < target:
        return 0.0, cov(0)
    lo, hi = 0.0, float(np.percentile(D[ZONE_B], 99)) + 50
    for _ in range(30):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if cov(mid) > target else (lo, mid)
    return (lo + hi) / 2, cov((lo + hi) / 2)


# ============================================================ MICRO (per pixel)
def micro(Dpx, comp_px, sc, rng):
    n08 = gnoise(0.8, 6, 2)
    D = Dpx * (0.35 + 1.3 * n08) if sc["fall_mm"] < 15 else Dpx * (0.8 + 0.4 * n08)
    D = D * (0.8 + 0.4 * gnoise(0.04, 3, 5))
    tread = np.zeros_like(D); groove = np.zeros_like(D); ridge = np.zeros_like(D)
    for k in range(sc.get("tracks", 40)):
        aisle = rng.choice([14.0, -4.0, -22.6, -41.3])
        y0 = aisle + rng.uniform(-2.6, 2.6); A = rng.uniform(0.05, 0.6); L = rng.uniform(6, 25); ph = rng.uniform(0, 6.28)
        yc = y0 + A * np.sin(GX / L + ph)
        for side in (-0.78, 0.78):
            d = GY - (yc + side); w = 0.115
            inside = (np.abs(d) < w).astype(np.float32)
            chev = np.mod(GX / 0.034 + 2.2 * np.abs(d) / w * (0.5 if side > 0 else -0.5), 1.0)
            blocks = (chev < 0.62) & (np.abs(np.abs(d) - 0.035) > 0.008)
            aa = np.clip((0.034 / FOOT - 1.5) / 1.5, 0, 1)
            tb = blocks * aa + 0.62 * (1 - aa)
            tread = np.maximum(tread, inside * tb); groove = np.maximum(groove, inside * (1 - tb))
            ridge = np.maximum(ridge, np.clip(1 - np.abs(np.abs(d) - w - 0.04) / 0.06, 0, 1))
    deep = np.clip(D / 40.0, 0, 1)                         # in deep snow a track is a rut, not a print
    D_track = D * (1 - (0.85 - 0.45 * deep) * tread) * (1 + (0.6 - 0.4 * deep) * groove)
    D = np.where(tread + groove > 0, D_track, D)
    D = D + ridge * np.minimum(D, 60) * 0.8                # displaced snow beside the tyres
    comp_px = np.maximum(comp_px, np.clip(tread + 0.5 * groove, 0, 1))
    foot = np.zeros_like(D)
    for k in range(sc.get("walkers", 3)):
        p = np.array([rng.uniform(-5, 12), rng.uniform(4, 16)]); hd = rng.uniform(0, 6.28)
        for i in range(40):
            hd += rng.normal(0, 0.15); p = p + 0.38 * np.array([np.cos(hd), np.sin(hd)])
            c = p + 0.09 * (1 if i % 2 else -1) * np.array([-np.sin(hd), np.cos(hd)])
            dx = GX - c[0]; dy = GY - c[1]
            u = dx * np.cos(hd) + dy * np.sin(hd); v = -dx * np.sin(hd) + dy * np.cos(hd)
            foot = np.maximum(foot, ((u / 0.14) ** 2 + (v / 0.05) ** 2 < 1).astype(np.float32))
    D = D * (1 - 0.5 * foot); comp_px = np.maximum(comp_px, 0.8 * foot)
    return (D * SURF).astype(np.float32), comp_px.astype(np.float32)


# ============================================================ APPEARANCE
hsv = cv2.cvtColor(IMG, cv2.COLOR_BGR2HSV)
paint = cv2.inRange(hsv, (15, 60, 90), (35, 255, 255)) | cv2.inRange(hsv, (0, 0, 175), (180, 50, 255))
clean = cv2.inpaint(IMG, cv2.dilate(paint, np.ones((5, 5), np.uint8)), 6, cv2.INPAINT_TELEA)
ILLUM = np.exp(cv2.GaussianBlur(np.log(R.srgb_to_lin(clean) + 1e-3), (0, 0), 12)) / R.GROUND_ALBEDO
A_GROUND = np.clip(img_lin / ILLUM, 0, 1)
_g = cv2.cvtColor(IMG, cv2.COLOR_BGR2GRAY).astype(np.float32)
_hp = _g - cv2.GaussianBlur(_g, (0, 0), 2.0); _hp = (_hp - _hp.mean()) / (_hp.std() + 1e-6)
H_MM = 1.0 + 0.7 * _hp + 0.8 * (gnoise(0.012, 3, 11) - 0.5) / 0.15     # pavement micro-height, ~1 mm


def remap_v(a, dv):
    return cv2.remap(a.astype(np.float32), uu, vv + dv.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def appearance(D, comp, rng):
    cov = 1 / (1 + np.exp(-(D - H_MM) / 0.3)) * SURF
    a_snow = np.array([0.82, 0.82, 0.81]) * (1 - 0.35 * comp)[..., None]
    d0 = (2.5 + 3.5 * comp)[..., None]
    opt = 1 - np.exp(-np.clip(D, 0, None)[..., None] / d0)
    grain = (0.92 + 0.16 * gnoise(0.015, 2, 21))[..., None]
    A = A_GROUND * (1 - cov[..., None]) + (A_GROUND + (a_snow * grain - A_GROUND) * opt) * cov[..., None]
    # shading from the full depth field, in ground space
    Dm = cv2.GaussianBlur(D, (0, 0), 0.7) / 1000.0
    dDdu, dDdv = np.gradient(Dm, axis=1), np.gradient(Dm, axis=0)
    dXdu, dXdv = np.gradient(GX, axis=1), np.gradient(GX, axis=0)
    dYdu, dYdv = np.gradient(GY, axis=1), np.gradient(GY, axis=0)
    det = dXdu * dYdv - dXdv * dYdu + 1e-9
    dDdX = (dDdu * dYdv - dDdv * dYdu) / det; dDdY = (dDdv * dXdu - dDdu * dXdv) / det
    n = np.dstack([-dDdX, -dDdY, np.ones_like(D)]); n /= np.linalg.norm(n, axis=2, keepdims=True)
    Lg = np.array([0.2, -0.5, 0.84]); Lg /= np.linalg.norm(Lg)
    cav = np.clip(cv2.GaussianBlur(D, (0, 0), 3.0) - D, 0, None) / (np.percentile(D[SURF], 90) + 5)
    shade = (0.75 + 0.25 * np.clip((n * Lg).sum(2), 0, 1) / Lg[2]) * (1 - 0.3 * np.clip(cav, 0, 1))
    col = ILLUM * A * np.where(SURF, shade, 1.0)[..., None]
    col = np.where(SURF[..., None], col, img_lin)
    # thickness parallax: the snow top is drawn shifted up by h*(v - v_hor)/HC; exposed faces darker
    hgt = D / 1000.0
    dv = hgt * np.clip(vv - V_HOR, 0, None) / HC
    dv = remap_v(hgt, dv) * np.clip(vv - V_HOR, 0, None) / HC
    thick = np.clip((D - 15) / 25, 0, 1) * SURF                 # only thick snow gets visible faces
    top = remap_v(thick, dv)
    col_top = np.dstack([remap_v(col[..., c], dv) for c in range(3)])
    wall = np.clip(thick - top, 0, 1)
    out = col * (1 - top[..., None]) + col_top * top[..., None]
    out = out * (1 - wall[..., None]) + (col * 0.62 * np.array([1.05, 1.0, 0.95])) * wall[..., None]
    vis_snow = np.maximum(remap_v(cov, dv) * (dv > 0.5) + cov * (dv <= 0.5), wall > 0.3)
    return out, cov, vis_snow


def camera(out, rng):
    o = R.lin_to_srgb(out).astype(np.float32)
    o = cv2.GaussianBlur(o, (0, 0), 0.5)
    hpI = IMG.astype(np.float32) - cv2.GaussianBlur(IMG.astype(np.float32), (0, 0), 1.5)
    s0 = np.median(np.abs(hpI[450:740, 100:900])) * 1.48
    o += rng.normal(0, 1, o.shape) * s0 * np.sqrt(np.clip(o.mean(2, keepdims=True) / 128, 0.3, 3))
    ok, enc = cv2.imencode(".jpg", np.clip(o, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def depth_vis(D):
    v = np.clip(np.log1p(D) / np.log1p(1000), 0, 1)
    c = cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    c[~SURF] = (IMG[~SURF] * 0.35).astype(np.uint8)
    return c


# ============================================================ RUN
SCENARIOS = [
    dict(name="dusting_in_use", fall_mm=4, tracks=60, walkers=3, target=0.80),
    dict(name="fresh_25mm_in_use", fall_mm=25, tracks=50, walkers=2, target=0.90),
    dict(name="plowed_with_film", fall_mm=60, plow_frac=0.85, occupied=0.3, left=0.6, film_mm=1.6,
         piles=2, tracks=30, walkers=1, target=0.25),
    dict(name="half_plowed", fall_mm=80, plow_frac=0.35, occupied=0.25, left=0.5, film_mm=2.5,
         piles=1, tracks=35, walkers=2, target=0.55),
]
TAU = 2.0                                                   # label threshold, mm
THRESH = [1, 2, 5, 20, 60]

if __name__ == "__main__":
    rows, report = [], {}
    for i, sc in enumerate(SCENARIOS):
        rng = np.random.default_rng(500 + i)
        Dm, cm, cleared = macro_depth(sc, rng)
        noise = G.fbm(BH_, BW_, 4.0, 4, rng)
        M, macro_cov = target_melt(Dm, TAU, sc["target"], noise)
        Dm = apply_melt(Dm, M, noise)
        Dpx, comp = micro(to_img(Dm), to_img(cm), sc, rng)
        out, cov, vis = appearance(Dpx, comp, rng)
        img = camera(out, rng)
        label = (Dpx > TAU) & SURF
        st = {f"cov_ge_{t}mm": float((AREA * ((Dpx > t) & SURF))[ZONE_I].sum() / AREA[ZONE_I].sum()) for t in THRESH}
        dz = Dpx[ZONE_I]; wz = AREA[ZONE_I]
        st.update(melt_mm=round(M, 2), macro_cov=round(float(macro_cov), 3),
                  mean_depth_cm=round(float((dz * wz).sum() / wz.sum() / 10), 2),
                  p95_depth_cm=round(float(np.percentile(dz, 95) / 10), 2))
        report[sc["name"]] = st
        print(sc["name"], json.dumps(st))
        cv2.imwrite(f"v4_{i}_{sc['name']}.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        np.save(f"v4_{i}_{sc['name']}_depth_mm.npy", Dpx.astype(np.float16))
        cv2.imwrite(f"v4_{i}_{sc['name']}_label_tau{int(TAU)}mm.png", (label * 255).astype(np.uint8))
        a = cv2.resize(img, (560, 420)); b = cv2.resize(depth_vis(Dpx), (560, 420))
        ov = img.copy(); ov[label] = (0.45 * ov[label] + 0.55 * np.array([255, 120, 0])).astype(np.uint8)
        c = cv2.resize(ov, (560, 420))
        cv2.putText(a, sc["name"], (10, 28), 0, 0.75, (255, 255, 255), 2)
        cv2.putText(b, "depth (log, 0-1000 mm)", (10, 28), 0, 0.75, (255, 255, 255), 2)
        cv2.putText(c, f">= {TAU:.0f} mm: {st['cov_ge_2mm']:.0%} | mean {st['mean_depth_cm']} cm", (10, 28), 0, 0.65, (255, 255, 255), 2)
        rows.append(np.hstack([a, b, c]))
    cv2.imwrite("v4_contact_sheet.jpg", np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
    json.dump(report, open("v4_report.json", "w"), indent=1)
