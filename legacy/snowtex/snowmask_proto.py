"""
Procedural snow-cover mask generator — prototype.

Generates physically-plausible snow masks for a parking lot in a rectified
top-down (bird's-eye, BEV) frame measured in metres, hits an exact target
coverage, and projects the result into a camera view with a homography.

Output per sample:
    material map  uint8  0=bare 1=fresh snow 2=packed/compacted 3=slush 4=pile
    snow mask     bool   (material in COUNTED) -> segmentation label
    coverage      float  snow pixels / surface pixels (exact, by construction)

Dependencies: numpy, scipy, opencv-python, matplotlib (all BSD/MIT/Apache).
"""
import numpy as np
import cv2
from scipy import ndimage as ndi

PX_PER_M = 10                       # BEV resolution: 10 px = 1 m
BARE, FRESH, PACKED, SLUSH, PILE = 0, 1, 2, 3, 4
COUNTED = (FRESH, PACKED, PILE)     # what counts as "snow" for coverage


def m(x):                           # metres -> pixels
    return int(round(x * PX_PER_M))


# ----------------------------------------------------------------- noise ----
def fbm(h, w, base_m, octaves, rng):
    """Fractal (multi-octave) value noise in [0,1]; base_m = largest feature size in metres."""
    out = np.zeros((h, w), np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        cell = max(base_m / (2 ** o), 0.05) * PX_PER_M
        gh, gw = max(2, int(h / cell) + 2), max(2, int(w / cell) + 2)
        grid = rng.random((gh, gw)).astype(np.float32)
        out += amp * cv2.resize(grid, (w, h), interpolation=cv2.INTER_CUBIC)
        total += amp
        amp *= 0.5
    out /= total
    return (out - out.min()) / (out.max() - out.min() + 1e-9)


def roughen(label, rng, amp_m=0.15, scale_m=0.6):
    """Domain-warp a label map so edges look ragged instead of CAD-clean."""
    h, w = label.shape
    dx = (fbm(h, w, scale_m, 3, rng) - 0.5) * 2 * amp_m * PX_PER_M
    dy = (fbm(h, w, scale_m, 3, rng) - 0.5) * 2 * amp_m * PX_PER_M
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(label, xs + dx, ys + dy, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)


# ---------------------------------------------------------------- layout ----
def parking_layout(W=50.0, H=34.0, stall_w=2.5, stall_d=5.0, aisle=7.0):
    """Toy lot: 4 stall rows + 2 aisles running along x. Real use: comes from
    the surface segmentation + detected markings / per-camera annotation."""
    h, w = m(H), m(W)
    surface = np.ones((h, w), bool)
    rows = [(0.0, stall_d), (stall_d + aisle, 2 * stall_d + aisle),
            (2 * stall_d + aisle, 3 * stall_d + aisle), (3 * stall_d + 2 * aisle, 4 * stall_d + 2 * aisle)]
    aisles = [(stall_d, stall_d + aisle), (3 * stall_d + aisle, 3 * stall_d + 2 * aisle)]
    stalls = [(x, y0, x + stall_w, y1) for (y0, y1) in rows
              for x in np.arange(0, W - stall_w + 1e-6, stall_w)]
    return dict(W=W, H=H, surface=surface, aisles=aisles, stalls=stalls, stall_w=stall_w)


# ------------------------------------------------------------- operations ----
def plow(snow, lay, rng, coverage_frac, blade_m):
    """Clear parallel blade-width strips along the aisles first, then stalls.
    Leaves thin windrow ridges at strip edges. Occupied stalls are skipped
    (a car was parked during plowing -> stall stays snowy afterwards)."""
    cleared = np.zeros_like(snow)
    ridges = np.zeros_like(snow)
    occupied = rng.random(len(lay['stalls'])) < rng.uniform(0.15, 0.6)
    targets = [(0, y0, lay['W'], y1) for (y0, y1) in lay['aisles']]
    targets += [s for s, occ in zip(lay['stalls'], occupied) if not occ]
    budget = coverage_frac * snow.sum()
    for i, (x0, y0, x1, y1) in enumerate(targets):
        if i >= len(lay['aisles']) and cleared.sum() > budget:  # aisles always attempted
            break
        y = y0
        while y < y1:
            yb = min(y + blade_m, y1)
            jitter = rng.uniform(-0.15, 0.15)          # imperfect driving line
            r0, r1 = m(y + jitter), m(yb + jitter)
            cleared[max(r0, 0):max(r1, 0), m(x0):m(x1)] = True
            if rng.random() < 0.5:                     # windrow left at the pass edge
                rr = m(yb + jitter)
                ridges[max(rr - m(0.25), 0):rr + m(0.1), m(x0):m(x1)] = True
            if yb >= y1:
                break
            y = yb - rng.uniform(0.0, 0.3)             # passes overlap a bit
    snow = snow & ~cleared
    return snow, ridges & cleared


def piles(lay, rng, n):
    """Plowed snow pushed to the perimeter: noisy blobs touching an edge."""
    h, w = lay['surface'].shape
    out = np.zeros((h, w), bool)
    yy, xx = np.mgrid[0:h, 0:w]
    for _ in range(n):
        r = rng.uniform(2.0, 4.5)
        side = rng.integers(4)
        cx = rng.uniform(0, lay['W']) if side < 2 else (0 if side == 2 else lay['W'])
        cy = (0 if side == 0 else lay['H']) if side < 2 else rng.uniform(0, lay['H'])
        d = np.hypot(xx - m(cx), yy - m(cy)) / PX_PER_M
        n_ = fbm(h, w, 2.0, 3, rng)
        out |= d < r * (0.75 + 0.5 * n_)
    return out


def tire_tracks(lay, rng, n):
    """Pairs of wheel ruts (gauge ~1.55 m, rut ~0.25 m) entering along an aisle
    and turning into a stall. Drawn as polylines in metric BEV space."""
    h, w = lay['surface'].shape
    out = np.zeros((h, w), np.uint8)
    for _ in range(n):
        a0, a1 = lay['aisles'][rng.integers(len(lay['aisles']))]
        ya = rng.uniform(a0 + 1.5, a1 - 1.5)
        x_turn = rng.uniform(5, lay['W'] - 5)
        into_low = rng.random() < 0.5                  # turn into the row below/above
        y_end = a0 - 4.0 if into_low else a1 + 4.0
        t = np.linspace(0, 1, 40)
        xs = np.where(t < 0.7, t / 0.7 * x_turn, x_turn + (t - 0.7) / 0.3 * rng.uniform(-2, 2))
        ys = np.where(t < 0.7, ya, ya + (t - 0.7) / 0.3 * (y_end - ya))
        dx, dy = np.gradient(xs), np.gradient(ys)
        nrm = np.stack([-dy, dx], 1) / (np.hypot(dx, dy)[:, None] + 1e-9)
        for side in (-0.775, 0.775):
            pts = np.stack([xs, ys], 1) + side * nrm
            cv2.polylines(out, [(pts * PX_PER_M).astype(np.int32)], False, 1, max(1, m(0.25)))
    return out.astype(bool)


def footpaths(lay, rng, n):
    h, w = lay['surface'].shape
    out = np.zeros((h, w), np.uint8)
    for _ in range(n):
        p = np.array([rng.uniform(0, lay['W']), rng.uniform(0, lay['H'])])
        heading = rng.uniform(0, 2 * np.pi)
        pts = [p.copy()]
        for _ in range(60):
            heading += rng.normal(0, 0.25)
            p = p + 0.5 * np.array([np.cos(heading), np.sin(heading)])
            pts.append(p.copy())
        cv2.polylines(out, [(np.array(pts) * PX_PER_M).astype(np.int32)], False, 1, m(0.5))
    return out.astype(bool)


# --------------------------------------------------------------- sampler ----
def sample(lay, target_cov, rng, scenario=None, tries=20):
    """Build one material map with coverage == target_cov (±0.5%)."""
    surf = lay['surface']
    h, w = surf.shape
    dist_edge = ndi.distance_transform_edt(np.pad(surf, 1))[1:-1, 1:-1] / PX_PER_M
    for _ in range(tries):
        scn = scenario or rng.choice(['uncleared', 'partial', 'plowed', 'plowed+melt'])
        depth = fbm(h, w, base_m=rng.uniform(4, 12), octaves=5, rng=rng)
        depth += 0.35 * np.exp(-dist_edge / rng.uniform(1.5, 4.0))   # drifts at edges
        snow = surf.copy()
        ridges = np.zeros_like(surf)
        if scn != 'uncleared':
            frac = {'partial': rng.uniform(0.2, 0.5)}.get(scn, rng.uniform(0.6, 0.95))
            snow, ridges = plow(snow, lay, rng, frac, rng.uniform(2.5, 3.5))
        pile = piles(lay, rng, rng.integers(0, 3)) if scn != 'uncleared' else np.zeros_like(surf)
        tracks = tire_tracks(lay, rng, rng.integers(2, 9))
        paths = footpaths(lay, rng, rng.integers(0, 3))

        # assemble materials (later layers win)
        mat = np.full((h, w), BARE, np.uint8)
        mat[snow] = FRESH
        mat[ridges] = FRESH
        mat[(tracks | paths) & snow] = PACKED
        mat[pile] = PILE
        mat[~surf] = BARE

        # coverage targeting: thin snow (low depth) melts first -> bisection on threshold
        def apply_melt(tm):
            mm = mat.copy()
            melt = (depth < tm) & np.isin(mm, (FRESH, PACKED))
            mm[melt] = BARE
            edge = ndi.binary_dilation(melt, iterations=m(0.35)) & np.isin(mm, (FRESH, PACKED))
            mm[edge & (rng.random(mm.shape) < 0.6)] = SLUSH          # wet slushy rim
            return mm
        cov = lambda mm: np.isin(mm, COUNTED)[surf].mean()
        if cov(mat) < target_cov:                     # not enough snow in this scenario
            continue
        lo, hi = 0.0, 1.0
        for _ in range(22):
            mid = (lo + hi) / 2
            if cov(apply_melt(mid)) > target_cov:
                lo = mid
            else:
                hi = mid
        mm = roughen(apply_melt((lo + hi) / 2), rng)
        mm[~surf] = BARE
        c = cov(mm)
        if abs(c - target_cov) < 0.03:
            return mm, c, scn
    raise RuntimeError("could not reach target coverage")


# ------------------------------------------------------- camera projection ----
def bev_to_camera_H(lay, img_w=1280, img_h=720):
    """Homography BEV(px) -> image(px). In production this comes from per-camera
    calibration (4 clicked ground points) or a fitted ground plane from depth."""
    w, h = m(lay['W']), m(lay['H'])
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[430, 170], [880, 170], [1250, 700], [30, 700]])
    return cv2.getPerspectiveTransform(src, dst), (img_w, img_h)


PALETTE = np.array([[70, 72, 76], [245, 247, 250], [160, 175, 200], [150, 120, 80], [120, 200, 235]], np.uint8)


def render(mat, lay):
    img = PALETTE[mat].copy()
    for (x0, y0, x1, y1) in lay['stalls']:            # stall markings, visible only on bare asphalt
        line = np.zeros(mat.shape, np.uint8)
        cv2.rectangle(line, (m(x0), m(y0)), (m(x1), m(y1)), 1, 1)
        img[(line > 0) & (mat == BARE)] = (230, 200, 60)
    return img


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    lay = parking_layout()
    H, size = bev_to_camera_H(lay)
    targets = [(0.05, 'plowed'), (0.15, 'plowed+melt'), (0.30, 'partial'), (0.65, 'uncleared')]
    fig, ax = plt.subplots(2, 4, figsize=(20, 8.2))
    for j, (t, scn) in enumerate(targets):
        rng = np.random.default_rng(100 + j)
        mat, c, s = sample(lay, t, rng, scenario=scn)
        ax[0, j].imshow(render(mat, lay))
        ax[0, j].set_title(f"BEV · {s} · target {t:.0%} → actual {c:.1%}", fontsize=11)
        cam = cv2.warpPerspective(render(mat, lay), H, size, flags=cv2.INTER_NEAREST, borderValue=(25, 28, 34))
        ax[1, j].imshow(cam)
        ax[1, j].set_title("same label, projected to a pole-camera view", fontsize=11)
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    fig.legend(handles=[Patch(color=PALETTE[i] / 255, label=n) for i, n in
                        enumerate(['bare asphalt', 'fresh snow', 'packed (tracks/paths)', 'slush (not counted)', 'pile'])],
               loc='lower center', ncol=5, fontsize=11, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig("snowmask_examples.png", dpi=90)
    print("saved snowmask_examples.png")
