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
_hsv0 = cv2.cvtColor(IMG, cv2.COLOR_BGR2HSV)
_paint0 = (cv2.inRange(_hsv0, (15, 60, 90), (35, 255, 255)) | cv2.inRange(_hsv0, (0, 0, 175), (180, 50, 255))) > 0
SURF = SURF | (_paint0 & (cv2.dilate(SURF.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0) & (R.vv >= 300))
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


def gnoise(scale_m, octaves, seed, stretch_x=1.0):
    """Fractal noise at image pixels, evaluated in ground coordinates, octaves below 1 px dropped."""
    r = np.random.default_rng(seed)
    out = np.zeros(GX.shape, np.float32); amp = 1.0; tot = 0
    for o in range(octaves):
        s = scale_m / 2 ** o
        grid = r.random((256, 256)).astype(np.float32)
        val = cv2.remap(grid, ((GX / (s * stretch_x)) % 256).astype(np.float32), ((GY / s) % 256).astype(np.float32),
                        cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
        fade = np.clip(1.5 - FOOT / s, 0, 1)
        out += amp * (val * fade + 0.5 * (1 - fade)); tot += amp; amp *= 0.55
    return out / tot


def to_img(a):
    return cv2.remap(a.astype(np.float32), R.BEV_MX, R.BEV_MY, cv2.INTER_LINEAR, borderValue=0)


# ============================================================ MACRO (BEV, mm)
AISLES_Y = [14.0, -3.85, -22.55, -41.25]                 # aisle centrelines, ground metres
TREADS = [  # pitch m, chevron slant, block share, circumferential groove offsets (m from centre)
    (0.034, 2.2, 0.62, (0.035,)), (0.028, 1.2, 0.55, (0.0, 0.06)), (0.042, 3.0, 0.68, (0.045,)),
    (0.031, 0.0, 0.58, (0.02, 0.07)), (0.038, 1.8, 0.50, ()),
]


def warp(a, rng, amp_m=0.35, scale_m=3.0):
    """Domain-warp a BEV field: real plow / car / pile boundaries are never ruler-straight."""
    dx = (G.fbm(BH_, BW_, scale_m, 3, rng) - 0.5) * 2 * amp_m * P
    dy = (G.fbm(BH_, BW_, scale_m, 3, rng) - 0.5) * 2 * amp_m * P
    mx = (BX + dx).astype(np.float32); my = (BY + dy).astype(np.float32)
    return cv2.remap(a.astype(np.float32), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def soften(mask, ramp_m):
    return np.clip(ndi.gaussian_filter(mask.astype(np.float32), ramp_m * P / 2), 0, 1)


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
        strip = np.zeros(D.shape, np.float32)
        for t in targets:
            _, x0, y0, x1, y1 = t
            r = (slice(max(int(y0 * P), 0), max(int(y1 * P), 0)), slice(int(x0 * P), int(x1 * P)))
            strip[r] = 1
        # blade path wobbles; edge is a short ramp, not a cut
        s_f = soften(warp(strip, rng, 0.25, 4.0) > 0.5, 0.12) * BSURF
        removed = s_f * D
        # residual film: blade leaves 0.5-4 mm, with scrape striations along the pass direction (x)
        film = sc.get("film_mm", 2.0) * (0.3 + 0.9 * G.fbm(BH_, BW_, 2.0, 3, rng))
        D = D * (1 - s_f) + film * s_f
        comp = np.maximum(comp, 0.6 * s_f)
        cleared = s_f > 0.5
        # spill berm: snow rolling off the blade end piles up along the strip edge (outside it)
        out_d = ndi.distance_transform_edt(s_f < 0.5) / P
        berm = np.exp(-((out_d - 0.25) / 0.18) ** 2) * (s_f < 0.5)
        D = D + berm * 0.5 * sc["fall_mm"] * (0.4 + 1.2 * G.fbm(BH_, BW_, 0.8, 3, rng))
        # windrows: removed volume per metre of aisle goes to one side of each aisle (chunky, wobbly)
        chunks = (0.25 + 1.3 * G.fbm(BH_, BW_, 0.35, 3, rng)) ** 1.5
        for (a0, a1) in LAY["aisles"]:
            ya, yb = int(a0 * P), int(a1 * P)
            if yb <= 0 or ya >= BH_:
                continue
            vol = ndi.gaussian_filter1d(removed[max(ya, 0):min(yb, BH_)].sum(0) / P, 1.5 * P)   # mm*m per column, tapered ends
            side = a1 if rng.random() < 0.5 else a0
            wr = rng.uniform(0.7, 1.2)
            yc = side + (0.4 if side == a1 else -0.4) + 0.25 * (G.fbm(1, BW_, 5.0, 3, rng)[0] - 0.5)
            prof = np.clip(1 - np.abs(BY / P - yc[None, :]) / (wr / 2), 0, 1) ** 1.3
            hgt = 2 * vol / wr * 0.5                                     # half goes to the end piles
            windrow += prof * hgt[None, :] * chunks
        # car-shaped holes: stalls that were occupied during snowfall and are now empty
        holes = np.zeros(D.shape, np.float32)
        for s_, o in zip(LAY["stalls"], occupied):
            if o and rng.random() < sc.get("left", 1.0):
                x0, y0, x1, y1 = s_
                cx, cy = (x0 + x1) / 2 + rng.normal(0, 0.15), (y0 + y1) / 2 + rng.normal(0, 0.3)
                hw, hl = rng.uniform(0.85, 0.95), rng.uniform(2.1, 2.4)
                q = (np.abs(BX / P - cx) / hw) ** 5 + (np.abs(BY / P - cy) / hl) ** 5   # rounded rectangle
                holes = np.maximum(holes, (q < 1).astype(np.float32))
                ax, ay = cx + X0, cy + Y0
                aisle = min(AISLES_Y, key=lambda a: abs(a - ay))
                R_ = rng.uniform(4.0, 5.5); dr = rng.choice([-1, 1])
                sc.setdefault("exits", []).append(dict(kind="turn", y=aisle + rng.uniform(-1, 1), xt=ax - dr * R_, R=R_,
                    dir=dr, side=int(np.sign(ay - aisle)), run=rng.uniform(10, 40),
                    tread=TREADS[rng.integers(len(TREADS))], age=rng.uniform(0, 0.3)))
        h_f = soften(warp(holes, rng, 0.08, 0.8) > 0.5, 0.15)
        D = D * (1 - 0.93 * h_f)
        # snow dropped from the car body when it left: a few clumps inside / at the edge of the hole
        drop = (G.fbm(BH_, BW_, 0.3, 2, rng) > 0.78) * h_f * (1 - soften(h_f > 0.9, 0.6))
        D = D + drop * 0.6 * sc["fall_mm"]
        D = D + windrow * BSURF
    # edge piles from the plowed volume
    if sc.get("piles", 0):
        pile = R.edge_piles(LAY, rng, sc["piles"]).astype(np.float32)
        pile = ndi.gaussian_filter(pile, 1.2 * P)
        pile = warp(pile, rng, 0.5, 2.0)
        D = D + 900 * pile * (0.4 + 1.2 * G.fbm(BH_, BW_, 0.5, 3, rng)) ** 1.3
    D = D * BSURF
    return D.astype(np.float32), comp.astype(np.float32), cleared


def apply_melt(D, Mmm, noise):
    return np.maximum(D - Mmm * (0.6 + 0.8 * noise), 0)


def target_melt(D, tau, target, noise):
    """Bisection on melt depth (mm) so that ground-area coverage(D > tau) in the inspection zone hits the target.
    Runs on the final per-pixel depth, after tracks and wear, so the label matches the target exactly."""
    wz, Dz, nz = AREA[ZONE_I], D[ZONE_I], noise[ZONE_I]
    cov = lambda M: float((wz * (np.maximum(Dz - M * (0.6 + 0.8 * nz), 0) > tau)).sum() / wz.sum())
    if cov(0) < target:
        return 0.0, cov(0)
    lo, hi = 0.0, float(np.percentile(Dz, 99)) + 50
    for _ in range(30):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if cov(mid) > target else (lo, mid)
    return (lo + hi) / 2, cov((lo + hi) / 2)


# ============================================================ MICRO (per pixel)


def smoothstep(x):
    x = np.clip(x, 0, 1); return x * x * (3 - 2 * x)


def wheel_pieces(car, lat, front, GX, GY):
    """Yield (d, s, valid) per path piece for one wheel: lateral offset d (m), arclength s (m), mask."""
    kind = car["kind"]
    if kind == "through":
        yc = car["y"] + car["A"] * np.sin(GX / car["L"] + car["ph"]) + lat
        yield GY - yc, GX * car["dir"], np.ones(GX.shape, bool)
        return
    xt, ya, R0, dr, sd = car["xt"], car["y"], car["R"], car["dir"], car["side"]
    cx, cy = xt, ya + sd * R0
    Rr = R0 - sd * 0 - lat                           # lat > 0 = towards the turn centre
    Rw = np.hypot(Rr, 2.7) if front else Rr          # front wheels sweep wider (off-tracking)
    ramp = 3.0
    # approach along the aisle; front wheels drift outward over the last wheelbase before the turn
    t1 = (GX - xt) * dr
    off1 = (Rw - Rr) * smoothstep((t1 + ramp) / ramp) if front else 0.0
    y_line = cy - sd * (Rr + off1)
    yield GY - y_line, GX * dr, (t1 < 0) & (t1 > -car["run"])
    # quarter arc into the stall
    ux, uy = (GX - cx) * dr, (cy - GY) * sd
    ok = (ux >= 0) & (uy >= 0)
    r = np.hypot(GX - cx, GY - cy)
    ang = np.arctan2(ux, uy)
    yield r - Rw, Rw * ang + xt * dr, ok
    # straight into the stall; front wheels converge back onto the rear track
    t2 = (GY - cy) * sd
    off2 = (Rw - Rr) * (1 - smoothstep(t2 / ramp)) if front else 0.0
    x_line = cx + dr * (Rr + off2)
    yield (GX - x_line) * dr, GY * sd + 10, (t2 > 0) & (t2 < 5.0)


def make_cars(sc, rng):
    cars = []
    for k in range(sc.get("tracks", 40)):
        ya = rng.choice(AISLES_Y) + rng.uniform(-1.6, 1.6)
        tr = TREADS[rng.integers(len(TREADS))]
        age = rng.uniform(0, 1) ** 2.5                  # 0 = just driven, 1 = mostly refilled by new snow
        if rng.random() < sc.get("turn_share", 0.45):
            cars.append(dict(kind="turn", y=ya, xt=rng.uniform(-20, 55), R=rng.uniform(4.0, 6.0),
                             dir=rng.choice([-1, 1]), side=rng.choice([-1, 1]), run=rng.uniform(8, 60),
                             tread=tr, age=age))
        else:
            cars.append(dict(kind="through", y=ya, A=rng.uniform(0.05, 0.5), L=rng.uniform(6, 25),
                             ph=rng.uniform(0, 6.28), dir=rng.choice([-1, 1]), tread=tr, age=age))
    return cars


def micro(Dpx, comp_px, sc, rng, cl_px=None):
    if sc["fall_mm"] < 15:      # a dusting: unevenness lives at the dm-cm scale, not in metre blotches
        D = Dpx * (0.55 + 0.9 * gnoise(0.25, 5, 2)) * (0.6 + 0.8 * gnoise(0.03, 3, 7))
    else:
        D = Dpx * (0.8 + 0.4 * gnoise(0.8, 6, 2)) * (0.85 + 0.3 * gnoise(0.04, 3, 5))
    if cl_px is not None:     # plow-blade scrape striations in the residual film, along the pass (x)
        stri = gnoise(0.04, 3, 41, stretch_x=40.0)     # irregular scrape lines, long along the pass
        D = D * (1 - cl_px * 0.6 * (1 - stri))
    # small wind-sculpted relief on deeper snow (gives the surface shading something to work with)
    deep_r = np.clip(D / 20.0, 0, 1)
    D = D + deep_r * (gnoise(0.5, 4, 31) - 0.5) * 2 * np.minimum(D * 0.12, 6)
    D = np.maximum(D, 0)
    # traffic wear: drive aisles churned by many cars -> thinner, mottled band
    if sc.get("wear", 0) > 0:
        traffic = np.zeros_like(D)
        for yc_ in AISLES_Y:
            traffic = np.maximum(traffic, np.clip(1 - (np.abs(GY - yc_) - 2.6) / 1.5, 0, 1))
        D = D * (1 - sc["wear"] * traffic * (0.5 + 1.0 * gnoise(1.2, 4, 9)) * (0.7 + 0.6 * gnoise(0.12, 3, 19)))
        D = np.maximum(D, 0)
    tread = np.zeros_like(D); groove = np.zeros_like(D); ridge = np.zeros_like(D)
    w = 0.11
    tread_f, groove_f, ridge_f = tread.reshape(-1), groove.reshape(-1), ridge.reshape(-1)
    GXf, GYf, FOOTf = GX.reshape(-1), GY.reshape(-1), FOOT.reshape(-1)
    for car in make_cars(sc, rng) + sc.get("exits", []):
        pitch, slant, share, grooves = car["tread"]
        fresh = 1 - 0.75 * car["age"]
        if car["kind"] == "through":
            box = np.abs(GYf - car["y"]) < car["A"] + 1.2
        else:
            xa, xb = sorted([car["xt"] - car["dir"] * car["run"], car["xt"] + car["dir"] * (car["R"] + 1.5)])
            ya_, yb_ = sorted([car["y"] - car["side"] * 1.5, car["y"] + car["side"] * (car["R"] + 6.5)])
            box = (GXf > xa - 1.5) & (GXf < xb + 1.5) & (GYf > ya_) & (GYf < yb_)
        sel = np.nonzero(box & SURF.reshape(-1))[0]
        if sel.size == 0:
            continue
        X, Y, F = GXf[sel], GYf[sel], FOOTf[sel]
        t_acc = tread_f[sel]; g_acc = groove_f[sel]; r_acc = ridge_f[sel]
        for lat in (-0.78, 0.78):
            for front in (False, True):
                for d, sarc, ok in wheel_pieces(car, lat, front, X, Y):
                    near = ok & (np.abs(d) < w + 0.15)
                    if not near.any():
                        continue
                    ad = np.abs(d)
                    inside = (ad < w) & near
                    chev = np.mod(sarc / pitch + slant * ad / w * np.sign(lat), 1.0)
                    blocks = chev < share
                    for g0 in grooves:
                        blocks &= np.abs(ad - g0) > 0.007
                    aa = np.clip((pitch / F - 1.5) / 1.5, 0, 1) * (1 - 0.6 * car["age"])   # refill blurs tread
                    tb = blocks * aa + share * (1 - aa)
                    t_acc = np.maximum(t_acc, inside * tb * fresh)
                    g_acc = np.maximum(g_acc, inside * (1 - tb) * fresh)
                    r_acc = np.maximum(r_acc, near * np.clip(1 - np.abs(ad - w - 0.04) / 0.06, 0, 1) * fresh)
        tread_f[sel], groove_f[sel], ridge_f[sel] = t_acc, g_acc, r_acc
    deep = np.clip(D / 80.0, 0, 1)                         # in deep snow a track is a rut, not a print
    D_track = D * (1 - (0.85 - 0.1 * deep) * tread) * (1 + (0.6 - 0.4 * deep) * groove)
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
            foot = np.maximum(foot, np.clip(1.6 * (1 - (u / 0.14) ** 2 - (v / 0.05) ** 2), 0, 1))
    D = D * (1 - 0.45 * foot); comp_px = np.maximum(comp_px, 0.5 * foot)
    return (D * SURF).astype(np.float32), comp_px.astype(np.float32)


# ============================================================ APPEARANCE
hsv = cv2.cvtColor(IMG, cv2.COLOR_BGR2HSV)
paint = cv2.inRange(hsv, (15, 60, 90), (35, 255, 255)) | cv2.inRange(hsv, (0, 0, 175), (180, 50, 255))
clean = cv2.inpaint(IMG, cv2.dilate(paint, np.ones((5, 5), np.uint8)), 6, cv2.INPAINT_TELEA)
_logc = np.log(R.srgb_to_lin(clean) + 1e-3)
# lighting = slow field; asphalt patches / stains must stay in albedo, or they print as grey clouds on snow
_Ls, _Lb = np.exp(cv2.GaussianBlur(_logc, (0, 0), 30)), np.exp(cv2.GaussianBlur(_logc, (0, 0), 90))
ILLUM = _Lb * np.clip(_Ls / _Lb, 0.85, 1.18) / R.GROUND_ALBEDO
A_GROUND = np.clip(img_lin / ILLUM, 0, 1)
_g = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY).astype(np.float32)
_hp = _g - cv2.GaussianBlur(_g, (0, 0), 2.0); _hp = (_hp - _hp.mean()) / (_hp.std() + 1e-6)
H_MM = 1.0 + 0.7 * _hp + 0.8 * (gnoise(0.012, 3, 11) - 0.5) / 0.15     # pavement micro-height, ~1 mm


def remap_v(a, dv):
    return cv2.remap(a.astype(np.float32), uu, vv + dv.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def appearance(D, comp, rng):
    cov = 1 / (1 + np.exp(-(D - H_MM) / 0.3)) * SURF
    a_snow = np.array([0.82, 0.82, 0.81]) * (1 - 0.35 * comp)[..., None]
    d0 = (2.5 + 3.5 * comp)[..., None]
    opt = 1 - np.exp(-np.clip(D, 0, None)[..., None] / d0)
    grain = ((0.92 + 0.16 * gnoise(0.015, 2, 21)) * (0.95 + 0.10 * gnoise(0.15, 3, 22)))[..., None]
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
    shade = np.clip(1 + 1.5 * (np.clip((n * Lg).sum(2), 0, 1) / Lg[2] - 1), 0.55, 1.2) * (1 - 0.3 * np.clip(cav, 0, 1))
    col = ILLUM * A * np.where(SURF, shade, 1.0)[..., None]
    col = np.where(SURF[..., None], col, img_lin)
    # thickness parallax: the snow top is drawn shifted up by h*(v - v_hor)/HC; exposed faces darker
    hgt = np.minimum(D, 250) / 1000.0                          # piles: cap parallax until they get real exemplars
    gain = np.clip(vv - V_HOR, 0, None) / HC
    dv = remap_v(hgt, hgt * gain) * gain                         # one fixed-point step
    dvi = np.round(dv)                                           # integer shifts: no resampling blur
    out = np.dstack([cv2.remap(col[..., c].astype(np.float32), uu, vv + dvi.astype(np.float32), cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_REPLICATE) for c in range(3)])
    # a camera-facing snow face shows where the displaced height drops toward the camera
    face = np.clip(-np.gradient(dv, axis=0), 0, 1) * (dv > 0.3)
    out = out * (1 - 0.22 * face[..., None] * np.array([0.95, 1.0, 1.05]))
    vis_snow = remap_v(cov, dv)
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
    dict(name="dusting_in_use", fall_mm=4, tracks=60, walkers=3, wear=0.55, target=0.80),
    dict(name="fresh_25mm_in_use", fall_mm=25, tracks=50, walkers=2, wear=0.35, target=None),
    dict(name="plowed_with_film", fall_mm=60, plow_frac=0.85, occupied=0.3, film_mm=1.6,
         piles=2, tracks=30, walkers=1, target=0.25),
    dict(name="half_plowed", fall_mm=80, plow_frac=0.35, occupied=0.25, film_mm=2.5,
         piles=1, tracks=35, walkers=2, target=0.45),
]
TAU = 2.0                                                   # label threshold, mm
THRESH = [1, 2, 5, 20, 60]

if __name__ == "__main__":
    rows, report = [], {}
    for i, sc in enumerate(SCENARIOS):
        rng = np.random.default_rng(500 + i)
        sc.pop("exits", None)
        Dm, cm, cleared = macro_depth(sc, rng)
        Dpx, comp = micro(to_img(Dm), to_img(cm), sc, rng, to_img(cleared.astype(np.float32)))
        noise = gnoise(4.0, 4, 600 + i)
        if sc.get("target") is None:
            M, macro_cov = 0.0, float((AREA * (Dpx > TAU))[ZONE_I].sum() / AREA[ZONE_I].sum())
        else:
            M, macro_cov = target_melt(Dpx, TAU, sc["target"], noise)
        Dpx = (np.maximum(Dpx - M * (0.6 + 0.8 * noise), 0) * SURF).astype(np.float32)
        out, cov, vis = appearance(Dpx, comp, rng)
        img = camera(out, rng)
        label = (Dpx > TAU) & SURF
        st = {f"cov_ge_{t}mm": float((AREA * ((Dpx > t) & SURF))[ZONE_I].sum() / AREA[ZONE_I].sum()) for t in THRESH}
        dz = Dpx[ZONE_I]; wz = AREA[ZONE_I]
        st.update(melt_mm=round(M, 2), target=sc.get("target"),
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
