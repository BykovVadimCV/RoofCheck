"""
First-shot snow renderer: procedural snow masks (metric, top-down) -> photoreal-ish
composite on a real snow-free parking-lot photo, with exact labels.

Pipeline
  1. Ground plane: homography image <-> metres, fitted from 4 stall-marking points.
  2. Surface mask: lot polygon minus posts / cones / signs (colour heuristics).
  3. Snow material map in a top-down grid (generator from snowmask_proto.py),
     coverage targeted inside an inspection zone, measured in ground area (m²).
  4. Rendering in LINEAR light:
       illumination S = blur(inpainted ground) / ground_albedo   (shadows, sky tint, lamps)
       snow = S * albedo(material) * texture_detail * relief_shading
       texture sampled perspective-correctly from ground coords, mip-mapped by footprint
     plus contact darkening and wet rims on the bare ground around snow.
  5. Camera matching: slight blur, signal-dependent noise, JPEG re-encode.

Textures: Poly Haven snow_02 (fresh) and snow_05 (dirty, road) — CC0.
"""
import sys
import numpy as np
import cv2
from scipy import ndimage as ndi
import snowmask_proto as G

rng_master = np.random.default_rng(7)

# ---------------------------------------------------------------- inputs ----
IMG = cv2.imread("lot.png")                          # BGR uint8, 1000x750
H_IMG, W_IMG = IMG.shape[:2]
SRC_PTS = np.float32([[274, 392], [632, 385], [862, 549], [248, 597]])   # stall-line endpoints
DST_M = np.float32([[0, 0], [10.8, 0], [10.8, 11.0], [0, 11.0]])        # 4 stalls x 2.7 m, 11 m double row
H_i2m = cv2.getPerspectiveTransform(SRC_PTS, DST_M)  # image px -> ground metres

# BEV grid (metres -> px), origin shift so all coords are positive
P = 15                                                # px per metre in the top-down grid
X0, Y0 = -30.0, -105.0                                # metric coord of BEV pixel (0,0)
BW, BH = 125.0, 122.0                                 # BEV extent in metres
G.PX_PER_M = P                                        # generator works at the same resolution

GROUND_ALBEDO = 0.25                                  # this lot is light (concrete-like) pavement
SNOW_ALBEDO = np.array([0.86, 0.84, 0.82])            # BGR, fresh snow ~0.8-0.9, faintly blue
INSPECTION_ZONE = (-25.0, -40.0, 60.0, 17.0)          # x0,y0,x1,y1 metres: where coverage is scored


def srgb_to_lin(x):
    x = x.astype(np.float32) / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def lin_to_srgb(x):
    x = np.clip(x, 0, 1)
    y = np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)
    return (y * 255 + 0.5).astype(np.uint8)


# --------------------------------------------------------- ground coords ----
uu, vv = np.meshgrid(np.arange(W_IMG, dtype=np.float32), np.arange(H_IMG, dtype=np.float32))
pts = np.stack([uu, vv], -1).reshape(-1, 1, 2)
gm = cv2.perspectiveTransform(pts, H_i2m).reshape(H_IMG, W_IMG, 2)   # metres per image pixel
GX, GY = gm[..., 0], gm[..., 1]
# footprint: metres covered by one image pixel (sqrt of Jacobian determinant)
dxu, dxv = np.gradient(GX, axis=1), np.gradient(GX, axis=0)
dyu, dyv = np.gradient(GY, axis=1), np.gradient(GY, axis=0)
FOOT = np.sqrt(np.abs(dxu * dyv - dxv * dyu)) + 1e-6
BEV_MX = ((GX - X0) * P).astype(np.float32)           # image pixel -> BEV pixel (for remap)
BEV_MY = ((GY - Y0) * P).astype(np.float32)

# ---------------------------------------------------------- surface mask ----
def surface_mask():
    m = np.zeros((H_IMG, W_IMG), np.uint8)
    poly = np.int32([[0, 749], [999, 749], [999, 216], [770, 212], [620, 209], [300, 207],
                     [140, 206], [100, 199], [0, 203]])
    cv2.fillPoly(m, [poly], 1)
    hsv = cv2.cvtColor(IMG, cv2.COLOR_BGR2HSV)
    orange = cv2.inRange(hsv, (0, 110, 90), (16, 255, 255)) | cv2.inRange(hsv, (165, 110, 90), (180, 255, 255))
    yellow = cv2.inRange(hsv, (15, 80, 120), (35, 255, 255))
    n, lab, st, _ = cv2.connectedComponentsWithStats(yellow)
    posts = np.zeros_like(yellow)
    for i in range(1, n):                              # compact blobs = posts/blocks; long thin = paint
        x, y, w, h, a = st[i]
        if y < 330 and a < 600 and h >= 7 and h >= w * 1.3 and max(w, h) < 40:   # upright posts only
            posts[lab == i] = 255
    orange[330:] = 0                                   # cones/signs live at the far edge; paint is near
    obstacles = cv2.dilate(orange | posts, np.ones((5, 5), np.uint8))
    # dark vertical clutter at the far edge (pole base, sign, bollards): very dark pixels near horizon
    dark = ((IMG.mean(2) < 45) & (vv < 260)).astype(np.uint8) * 255
    obstacles |= cv2.dilate(dark, np.ones((3, 3), np.uint8))
    m[obstacles > 0] = 0
    return m.astype(bool)


# --------------------------------------------------------- BEV layout -------
def bev_layout(img_surface):
    w, h = G.m(BW), G.m(BH)
    Hm = np.array([[P, 0, -X0 * P], [0, P, -Y0 * P], [0, 0, 1]], np.float64) @ H_i2m
    surf = cv2.warpPerspective(img_surface.astype(np.uint8), Hm, (w, h), flags=cv2.INTER_NEAREST) > 0
    surf = ndi.binary_closing(surf, iterations=2)
    period, depth, aisle = 18.7, 11.0, 7.7               # measured from the photo
    rows, aisles = [], [(11.0 - Y0, 17.0 - Y0)]
    y = 0.0
    while y > -100:
        rows += [(y - Y0, y + 5.5 - Y0), (y + 5.5 - Y0, y + depth - Y0)]
        aisles.append((y - aisle - Y0, y - Y0))
        y -= period
    stalls = [(x - X0, a, x + 2.7 - X0, b) for (a, b) in rows for x in np.arange(-29.7, 94, 2.7)]
    return dict(W=BW, H=BH, surface=surf, aisles=aisles, stalls=stalls, stall_w=2.7)


def bev_zone(shape):
    zx0, zy0, zx1, zy1 = INSPECTION_ZONE
    z = np.zeros(shape, bool)
    z[G.m(zy0 - Y0):G.m(zy1 - Y0), G.m(zx0 - X0):G.m(zx1 - X0)] = True
    return z


def edge_piles(lay, rng, n):
    """Piles pushed against the real lot boundary, inside the visible zone."""
    surf = lay['surface']
    zone = bev_zone(surf.shape)
    edge = surf & ~ndi.binary_erosion(surf, iterations=2) & zone
    ys, xs = np.nonzero(edge)
    out = np.zeros_like(surf)
    if len(xs) == 0 or n == 0:
        return out
    hh, ww = surf.shape
    yy, xx = np.mgrid[0:hh, 0:ww]
    for _ in range(n):
        k = rng.integers(len(xs))
        r = rng.uniform(2.0, 4.0) * P
        noise = G.fbm(hh, ww, 2.0, 3, rng)
        out |= np.hypot(xx - xs[k], yy - ys[k]) < r * (0.7 + 0.6 * noise)
    return out & surf


def make_material(lay, target, scenario, rng):
    """Same layer logic as the generator, coverage scored in the inspection zone (ground m²)."""
    surf = lay['surface']
    hh, ww = surf.shape
    zone = bev_zone(surf.shape) & surf
    dist_edge = ndi.distance_transform_edt(np.pad(surf, 1))[1:-1, 1:-1] / P
    for _ in range(25):
        depth = G.fbm(hh, ww, base_m=rng.uniform(4, 10), octaves=5, rng=rng)
        depth += 0.35 * np.exp(-dist_edge / rng.uniform(1.5, 4.0))
        snow, ridges = surf.copy(), np.zeros_like(surf)
        if scenario != 'uncleared':
            frac = rng.uniform(0.2, 0.45) if scenario == 'partial' else rng.uniform(0.7, 0.95)
            snow, ridges = G.plow(snow, lay, rng, frac, rng.uniform(2.5, 3.5))
        pile = edge_piles(lay, rng, rng.integers(1, 3)) if scenario != 'uncleared' else np.zeros_like(surf)
        tracks = G.tire_tracks(lay, rng, rng.integers(6, 16))
        paths = G.footpaths(lay, rng, rng.integers(1, 4))
        mat = np.full((hh, ww), G.BARE, np.uint8)
        mat[snow] = G.FRESH
        mat[ridges] = G.FRESH
        mat[(tracks | paths) & snow] = G.PACKED
        mat[pile] = G.PILE
        mat[~surf] = G.BARE
        cov = lambda mm: np.isin(mm, G.COUNTED)[zone].mean()

        def melt(t):
            mm = mat.copy()
            mlt = (depth < t) & np.isin(mm, (G.FRESH, G.PACKED))
            mm[mlt] = G.BARE
            rim = ndi.binary_dilation(mlt, iterations=G.m(0.3)) & np.isin(mm, (G.FRESH, G.PACKED))
            mm[rim & (rng.random(mm.shape) < 0.5)] = G.SLUSH
            return mm
        if cov(mat) < target:
            continue
        lo, hi = 0.0, 1.0
        for _ in range(18):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if cov(melt(mid)) > target else (lo, mid)
        mm = G.roughen(melt((lo + hi) / 2), rng, amp_m=0.12, scale_m=0.5)
        mm[~surf] = G.BARE
        if abs(cov(mm) - target) < 0.03:
            return mm, depth, cov(mm)
    raise RuntimeError("target coverage not reachable")


# ------------------------------------------------------------ textures ------
def load_tile(path, crop=None):
    t = cv2.imread(path).astype(np.float32)
    if crop:
        x0, x1 = crop
        t = t[:, x0:x1]
    t = srgb_to_lin(t.astype(np.uint8))
    ref = np.percentile(t.reshape(-1, 3), 90, axis=0)     # "clean snow" reference of this photo
    return np.clip(t / ref, 0, 1.25)                     # relative albedo detail, keeps dirt chroma


FRESH_T = load_tile("s02_diff.jpg")                       # 256 px over 2.0 m
DIRTY_T = load_tile("s05_diff.jpg", crop=(55, 135))       # snow strip between tyre marks, 2.0 m tall
TEX_PPM = {"fresh": 128.0, "dirty": 96.0}


def sample_tile(tile, ppm, gx, gy, foot, angle, offset):
    """Perspective-correct, mip-mapped tile sampling in ground coordinates (BORDER_WRAP tiling)."""
    c, s = np.cos(angle), np.sin(angle)
    tx = (c * gx - s * gy + offset[0]) * ppm
    ty = (s * gx + c * gy + offset[1]) * ppm
    level = np.clip(np.log2(foot * ppm + 1e-6), 0, 6)
    pyr = [tile]
    for _ in range(6):
        pyr.append(cv2.pyrDown(pyr[-1]) if min(pyr[-1].shape[:2]) > 4 else pyr[-1])
    out = np.zeros(gx.shape + (3,), np.float32)
    for l, t in enumerate(pyr):
        w = np.clip(1 - np.abs(level - l), 0, 1)[..., None]
        if w.max() == 0:
            continue
        sc = t.shape[1] / tile.shape[1]
        smp = cv2.remap(t, (tx * sc).astype(np.float32), (ty * sc).astype(np.float32),
                        cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        out += w * smp
    return out


# ------------------------------------------------------------- render -------
def render(mat_bev, depth_bev, rng):
    img_lin = srgb_to_lin(IMG)
    surf = SURF

    # 1) clean ground -> illumination estimate (markings removed, low-passed)
    hsv = cv2.cvtColor(IMG, cv2.COLOR_BGR2HSV)
    paint = cv2.inRange(hsv, (15, 60, 90), (35, 255, 255)) | cv2.inRange(hsv, (0, 0, 175), (180, 50, 255))
    paint = cv2.dilate(paint, np.ones((5, 5), np.uint8))
    clean = cv2.inpaint(IMG, paint, 6, cv2.INPAINT_TELEA)
    # log-domain low-pass: keeps lamp pools / sky gradient, removes joints, cracks, stains
    lg = np.log(srgb_to_lin(clean) + 1e-3)
    illum = np.exp(cv2.GaussianBlur(lg, (0, 0), 12)) / GROUND_ALBEDO      # per-channel, BGR

    # 2) BEV -> image: materials (nearest) and soft coverage alpha (linear)
    mat = cv2.remap(mat_bev, BEV_MX, BEV_MY, cv2.INTER_NEAREST, borderValue=0)
    snowy_b = np.isin(mat_bev, (G.FRESH, G.PACKED, G.PILE))
    d_in = ndi.distance_transform_edt(snowy_b) / P                       # metres from snow edge
    feather = 0.03 + 0.25 * G.fbm(*mat_bev.shape, 3.0, 3, rng)           # thin powdery edges vary 3-28 cm
    grain = G.fbm(*mat_bev.shape, 0.15, 3, rng)                          # mottled partial cover in the feather zone
    ramp = np.clip(d_in / feather, 0, 1)
    snowy = np.clip(ramp * 1.6 - (1 - ramp) * (grain - 0.35) * 1.2, 0, 1).astype(np.float32) * snowy_b
    slushy = (mat_bev == G.SLUSH).astype(np.float32)
    soft = lambda a: cv2.GaussianBlur(a, (0, 0), 0.04 * P + 0.5)
    alpha = cv2.remap(soft(snowy), BEV_MX, BEV_MY, cv2.INTER_LINEAR, borderValue=0)
    a_sl = cv2.remap(soft(slushy), BEV_MX, BEV_MY, cv2.INTER_LINEAR, borderValue=0)
    # anti-alias far field: blur alpha by footprint (coarse two-level approach)
    far = np.clip((FOOT - 0.05) / 0.15, 0, 1)
    alpha = alpha * (1 - far) + cv2.GaussianBlur(alpha, (0, 0), 1.2) * far
    alpha *= surf
    a_sl *= surf

    # 3) snow height -> relief shading (dusk: mostly sky light from the far side, high ambient)
    hgt = (0.05 + 0.05 * depth_bev) * (mat_bev != G.BARE) * np.where(mat_bev == G.PACKED, 0.4, 1.0)
    hgt += 0.6 * ndi.gaussian_filter((mat_bev == G.PILE).astype(np.float32), 1.2 * P)
    hgt = ndi.gaussian_filter(hgt.astype(np.float32), 0.08 * P)
    gy_, gx_ = np.gradient(hgt * P)                     # slope (m/m)
    n = np.dstack([-gx_, -gy_, np.ones_like(hgt)])
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    L = np.array([0.15, -0.55, 0.82]); L /= np.linalg.norm(L)
    shade_bev = (0.72 + 0.28 * np.clip((n * L).sum(2), 0, 1)).astype(np.float32)
    shade = cv2.remap(shade_bev, BEV_MX, BEV_MY, cv2.INTER_LINEAR, borderValue=1)

    # 4) albedo: fresh texture everywhere, dirty texture weighted by tracks/piles/noise
    ang, off = rng.uniform(0, np.pi), rng.uniform(0, 2, 2)
    fresh = sample_tile(FRESH_T, TEX_PPM["fresh"], GX, GY, FOOT, ang, off)
    fresh2 = sample_tile(FRESH_T, TEX_PPM["fresh"] * 0.37, GX, GY, FOOT, ang + 1.1, off[::-1])
    fresh = 0.6 * fresh + 0.4 * fresh2                   # two scales -> hides tiling
    dirty = sample_tile(DIRTY_T, TEX_PPM["dirty"], GX, GY, FOOT, rng.uniform(-0.2, 0.2), off)
    dirt_bev = (mat_bev == G.PACKED).astype(np.float32) * 0.9
    dirt_bev = np.maximum(dirt_bev, 0.35 * (G.fbm(*mat_bev.shape, 6.0, 4, rng) > 0.62))
    dirt_bev = ndi.gaussian_filter(dirt_bev, 0.3 * P)
    dirt_w = cv2.remap(dirt_bev, BEV_MX, BEV_MY, cv2.INTER_LINEAR, borderValue=0)[..., None]
    packed = cv2.remap((mat_bev == G.PACKED).astype(np.float32), BEV_MX, BEV_MY, cv2.INTER_LINEAR)[..., None]
    fresh = 1 + 1.15 * (cv2.GaussianBlur(fresh, (0, 0), 0.7) - 1)                        # photo texture is low-contrast at this scale
    big = cv2.remap(G.fbm(*mat_bev.shape, 4.0, 4, rng).astype(np.float32), BEV_MX, BEV_MY, cv2.INTER_LINEAR)
    detail = (fresh * (1 - dirt_w) + dirty * dirt_w) * (0.9 + 0.2 * big)[..., None]
    albedo = SNOW_ALBEDO * detail * (1 - 0.35 * packed)  # compacted snow is greyer

    snow_lin = illum * albedo * shade[..., None]

    # 5) bare ground around snow: contact darkening + wet rim; slush = dark wet mix
    rim = cv2.GaussianBlur(alpha, (0, 0), 2.5)
    ground = img_lin * (1 - 0.18 * np.clip(rim - alpha, 0, 1))[..., None]
    wet = np.clip(a_sl + 0.6 * np.clip(rim - alpha, 0, 1) * (1 - alpha), 0, 1)[..., None]
    ground = ground * (1 - 0.35 * wet)
    slush_col = 0.5 * ground + 0.5 * illum * SNOW_ALBEDO * 0.35 * fresh
    ground = ground * (1 - a_sl[..., None]) + slush_col * a_sl[..., None]

    out = ground * (1 - alpha[..., None]) + snow_lin * alpha[..., None]

    # 6) camera matching: soften inserted layer, signal-dependent noise, JPEG
    out_s = lin_to_srgb(out).astype(np.float32)
    blurred = cv2.GaussianBlur(out_s, (0, 0), 0.6)
    out_s = out_s * (1 - alpha[..., None]) + blurred * alpha[..., None]
    hp = IMG.astype(np.float32) - cv2.GaussianBlur(IMG.astype(np.float32), (0, 0), 1.5)
    sigma0 = np.median(np.abs(hp[450:740, 100:900])) * 1.48           # robust noise estimate
    bright = out_s.mean(2, keepdims=True) / 128.0
    noise = rng.normal(0, 1, out_s.shape) * sigma0 * np.sqrt(np.clip(bright, 0.3, 3))
    out_s = out_s + noise * alpha[..., None]
    ok, enc = cv2.imencode(".jpg", np.clip(out_s, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 88])
    final = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    label = (alpha > 0.5) & surf
    return final, label, mat


if __name__ == "__main__":
    SURF = surface_mask()
    LAY = bev_layout(SURF)
    cases = [(0.08, "plowed"), (0.25, "plowed+melt"), (0.50, "partial"), (0.85, "uncleared")]
    tiles = []
    for i, (t, scn) in enumerate(cases):
        rng = np.random.default_rng(1000 + i)
        mat_bev, depth_bev, cov = make_material(LAY, t, scn, rng)
        img, label, _ = render(mat_bev, depth_bev, rng)
        cv2.imwrite(f"render_{i}_{scn}.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        ov = img.copy()
        ov[label] = (0.45 * ov[label] + 0.55 * np.array([255, 120, 0])).astype(np.uint8)
        cv2.imwrite(f"label_{i}_{scn}.png", (label * 255).astype(np.uint8))
        pix_cov = label[SURF].mean()
        zx0, zy0, zx1, zy1 = INSPECTION_ZONE
        inz = SURF & (GX > zx0) & (GX < zx1) & (GY > zy0) & (GY < zy1)
        area = FOOT ** 2                                 # m² per pixel
        lab_cov = (area * label)[inz].sum() / area[inz].sum()
        print(f"{scn:12s} target {t:.0%} | generator {cov:.1%} | final label, ground-area {lab_cov:.1%} | image-pixel {pix_cov:.1%}")
        tiles.append((img, ov, scn, cov))
    # contact sheet
    rows = []
    for img, ov, scn, cov in tiles:
        a, b = cv2.resize(img, (600, 450)), cv2.resize(ov, (600, 450))
        cv2.putText(a, f"{scn}: {cov:.0%} of inspection zone", (12, 30), 0, 0.8, (255, 255, 255), 2)
        cv2.putText(b, "label overlay", (12, 30), 0, 0.8, (255, 255, 255), 2)
        rows.append(np.hstack([a, b]))
    cv2.imwrite("contact_sheet.jpg", np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
