"""
v3 near-field snow model: thin snow fills the pavement's own micro-relief.

Ideas (see REALISM-RESEARCH-2.md):
  - snow depth D (mm) vs pavement micro-height h (mm, ~1 mm mean profile depth):
    a pixel is covered where D > h; aggregate tips poke through thin snow as dark specks
  - h comes from the photo's own high-pass texture (registered to the real pavement) + sub-pixel noise
  - optical depth: thin snow is translucent, effective albedo -> snow albedo as 1 - exp(-D / d0)
  - tyre tracks: compaction (D x 0.25 in tread blocks), snow packed into grooves, loose ridges
    pushed to both sides; tread pattern stamped in ground coordinates (pitch ~3.5 cm)
  - footprints: compacted ellipses on random walks
  - coverage softness grows with pixel footprint (sub-pixel averaging in the far field)
Everything is evaluated per image pixel from ground coordinates (GX, GY metres), so the
near field gets full resolution instead of a 15 px/m raster.
"""
import numpy as np
import cv2
import render_snow_v2 as R

rng = np.random.default_rng(3)
IMG, GX, GY, FOOT = R.IMG, R.GX, R.GY, R.FOOT
H, W = IMG.shape[:2]
SURF = R.surface_mask()
img_lin = R.srgb_to_lin(IMG)


def gnoise(scale_m, octaves=4, seed=0):
    """Fractal value noise evaluated at ground coordinates (anti-aliased by footprint)."""
    r = np.random.default_rng(seed)
    out = np.zeros(GX.shape, np.float32); amp = 1.0; tot = 0
    for o in range(octaves):
        s = scale_m / 2 ** o
        n = 256
        grid = r.random((n, n)).astype(np.float32)
        u = (GX / s) % n; v = (GY / s) % n
        val = cv2.remap(grid, u.astype(np.float32), v.astype(np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
        fade = np.clip(1.5 - FOOT / s, 0, 1)          # drop octaves finer than a pixel
        out += amp * (val * fade + 0.5 * (1 - fade)); tot += amp; amp *= 0.55
    return out / tot


# ---------- 1. illumination and pavement albedo (incl. paint) ----------
hsv = cv2.cvtColor(IMG, cv2.COLOR_BGR2HSV)
paint = cv2.inRange(hsv, (15, 60, 90), (35, 255, 255)) | cv2.inRange(hsv, (0, 0, 175), (180, 50, 255))
clean = cv2.inpaint(IMG, cv2.dilate(paint, np.ones((5, 5), np.uint8)), 6, cv2.INPAINT_TELEA)
illum = np.exp(cv2.GaussianBlur(np.log(R.srgb_to_lin(clean) + 1e-3), (0, 0), 12)) / R.GROUND_ALBEDO
A_ground = np.clip(img_lin / illum, 0, 1)               # per-pixel pavement albedo (paint stays bright)

# ---------- 2. pavement micro-height h (mm) ----------
g = cv2.cvtColor(IMG, cv2.COLOR_BGR2GRAY).astype(np.float32)
hp = g - cv2.GaussianBlur(g, (0, 0), 2.0)               # resolvable aggregate texture from the photo
hp = (hp - hp.mean()) / (hp.std() + 1e-6)
sub = (gnoise(0.012, 3, seed=11) - 0.5) / 0.15          # sub-pixel aggregate (~1 cm stones)
h_mm = 1.0 + 0.7 * hp + 0.8 * sub                     # mean ~1 mm, +-1 mm (MPD ~1 mm)
h_mm += 0.0 * (paint > 0)                               # paint sits slightly proud, sheds snow less... (minor)

# ---------- 3. snow depth field D (mm) ----------
D0 = 3.2                                                # fresh snowfall depth, mm
D = D0 * (0.35 + 1.3 * gnoise(6.0, 4, seed=1))          # drifting / uneven deposition
D *= 0.35 + 1.3 * gnoise(0.8, 6, seed=2)                # metre-to-cm scale unevenness (wind, spray)
D *= 0.75 + 0.5 * gnoise(0.04, 3, seed=5)               # cm-scale flake clumping
# traffic wear: drive aisles have been churned by many cars -> thinner, mottled snow
traffic = np.zeros_like(D)
for yc_, hw in ((14.0, 3.0), (-4.0, 3.2), (-22.6, 3.2), (-41.3, 3.2)):
    traffic = np.maximum(traffic, np.clip(1 - (np.abs(GY - yc_) - hw) / 1.5, 0, 1))
D *= 1 - 0.55 * traffic * (0.6 + 0.8 * gnoise(1.5, 3, seed=9))

tread_mask = np.zeros_like(D); ridge = np.zeros_like(D); groove = np.zeros_like(D)
tracks = []
for k in range(60):                                     # vehicles that drove along aisles / into stalls
    aisle = rng.choice([14.0, -4.0, -22.6, -41.3])
    y0 = aisle + rng.uniform(-2.6, 2.6); A = rng.uniform(0.05, 0.6); L = rng.uniform(6, 25); ph = rng.uniform(0, 6.28)
    tracks.append((y0, A, L, ph))
for (y0, A, L, ph) in tracks:
    yc = y0 + A * np.sin(GX / L + ph)
    for side in (-0.78, 0.78):
        d = GY - (yc + side)                            # signed lateral distance to wheel centreline
        w = 0.115                                       # half tyre width
        inside = np.abs(d) < w
        s = GX + 0.0
        pitch = 0.034
        chevron = np.mod(s / pitch + 2.2 * np.abs(d) / w * (0.5 if side > 0 else -0.5), 1.0)
        blocks = (chevron < 0.62) & (np.abs(np.abs(d) - 0.035) > 0.008)   # tread blocks minus centre grooves
        aa = np.clip((pitch / FOOT - 1.5) / 1.5, 0, 1)                    # anti-alias tread where pitch < ~2 px
        tb = blocks * aa + 0.62 * (1 - aa)
        tread_mask = np.maximum(tread_mask, inside * tb)
        groove = np.maximum(groove, inside * (1 - tb))
        e = np.clip(1 - np.abs(np.abs(d) - w - 0.03) / 0.04, 0, 1)        # loose ridge pushed to both sides
        ridge = np.maximum(ridge, e)
D = D * (1 - 0.85 * tread_mask)                         # blocks compact snow to ~22 %
D = D * (1 + 0.6 * groove)                             # snow packed into grooves stays
D = D * (1 + 0.9 * ridge)                               # loose snow piled beside tyres

foot = np.zeros_like(D)                                 # footprints on a few random walks
for k in range(3):
    p = np.array([rng.uniform(-5, 12), rng.uniform(4, 16)]); hd = rng.uniform(0, 6.28)
    for i in range(40):
        hd += rng.normal(0, 0.15); p = p + 0.38 * np.array([np.cos(hd), np.sin(hd)])
        off = 0.09 * (1 if i % 2 else -1) * np.array([-np.sin(hd), np.cos(hd)])
        c = p + off
        dx = GX - c[0]; dy = GY - c[1]
        u = dx * np.cos(hd) + dy * np.sin(hd); v = -dx * np.sin(hd) + dy * np.cos(hd)
        foot = np.maximum(foot, ((u / 0.14) ** 2 + (v / 0.05) ** 2 < 1).astype(np.float32))
D = D * (1 - 0.45 * foot)

# ---------- 4. coverage from depth vs micro-height ----------
soft = 0.3 + 0.0 * FOOT  # mm; wider in the far field (sub-pixel averaging)
cov = 1 / (1 + np.exp(-(D - h_mm) / soft))
packed = np.clip(tread_mask + 0.8 * foot, 0, 1)

# ---------- 5. appearance ----------
a_snow = np.array([0.82, 0.82, 0.81]) * (1 - 0.35 * packed)[..., None]   # compacted snow greyer
d0 = np.where(packed > 0.5, 6.0, 2.5)[..., None]                          # packed snow more translucent
opt = 1 - np.exp(-np.clip(D, 0, None)[..., None] / d0)
grain = (0.92 + 0.16 * gnoise(0.015, 2, seed=21))[..., None]               # snow surface grain
A_snow_eff = A_ground + (a_snow * grain - A_ground) * opt
A_mix = A_ground * (1 - cov[..., None]) + A_snow_eff * cov[..., None]

# relief shading from the depth field (ground-space gradient), mostly diffuse sky light
Dm = cv2.GaussianBlur(D, (0, 0), 0.7) / 1000.0
dDdu = np.gradient(Dm, axis=1); dDdv = np.gradient(Dm, axis=0)
dXdu, dXdv = np.gradient(GX, axis=1), np.gradient(GX, axis=0)
dYdu, dYdv = np.gradient(GY, axis=1), np.gradient(GY, axis=0)
det = dXdu * dYdv - dXdv * dYdu + 1e-9
dDdX = (dDdu * dYdv - dDdv * dYdu) / det; dDdY = (dDdv * dXdu - dDdu * dXdv) / det
n = np.dstack([-dDdX, -dDdY, np.ones_like(D)]); n /= np.linalg.norm(n, axis=2, keepdims=True)
Lg = np.array([0.2, -0.5, 0.84]); Lg /= np.linalg.norm(Lg)
cavity = np.clip(cv2.GaussianBlur(D, (0, 0), 2.0) - D, 0, None) / (D0 + 1e-6)
shade = (0.78 + 0.22 * np.clip((n * Lg).sum(2), 0, 1)) * (1 - 0.35 * np.clip(cavity, 0, 1))
shade = np.where(SURF, shade, 1.0)

out = illum * np.where(SURF[..., None], A_mix * shade[..., None], A_ground)
out = np.where(SURF[..., None], out, img_lin)

# ---------- 6. camera ----------
o = R.lin_to_srgb(out).astype(np.float32)
o = cv2.GaussianBlur(o, (0, 0), 0.5)
hpI = IMG.astype(np.float32) - cv2.GaussianBlur(IMG.astype(np.float32), (0, 0), 1.5)
sigma0 = np.median(np.abs(hpI[450:740, 100:900])) * 1.48
o += rng.normal(0, 1, o.shape) * sigma0 * np.sqrt(np.clip(o.mean(2, keepdims=True) / 128, 0.3, 3))
ok, enc = cv2.imencode(".jpg", np.clip(o, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 88])
final = cv2.imdecode(enc, cv2.IMREAD_COLOR)
cv2.imwrite("v3_thin_snow.jpg", final, [cv2.IMWRITE_JPEG_QUALITY, 92])
label = (cov > 0.5) & SURF
cv2.imwrite("v3_thin_snow_label.png", (label * 255).astype(np.uint8))
print("pixel coverage in lot:", label[SURF].mean().round(3))
