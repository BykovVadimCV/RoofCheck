# Depth-first generator (v4): notes

Date: 2026-10-13. Script: `depth_v4.py`, which needs `render_snow_v2.py` and `snowmask_proto.py`. Each frame takes about 20 s on CPU.

## What changed from v3

Snow depth D (in mm) per pixel is now the one quantity that drives everything. A compaction map rides alongside it. The image, the labels, coverage at every threshold, and the thickness numbers are all derived from D.

**Macro stage**: a top-down metric grid at 15 px/m.
- Snowfall with drift against the lot edges.
- Plow strips:
  - The blade path wobbles, and the strip edge is a 12 cm ramp rather than a cut.
  - The blade leaves a residual film.
  - Snow spilling off the blade end forms a berm.
- Windrows built from the volume actually removed. They are chunky, wobble, and taper at the ends.
- Edge piles.
- Car holes: stalls that were occupied during snowfall and are now empty. Each hole is a rounded rectangle with a few clumps of snow that dropped off the car.

**Micro stage**: per image pixel, computed from ground coordinates.
- Unevenness at the cm and dm scales.
- Small wind relief on deeper snow.
- Traffic wear in the drive aisles.
- Tyre tracks:
  - Curved turns into stalls.
  - Front wheels off-track on the turns.
  - Five tread types.
  - Partial refill by newer snow.
  - Side ridges.
  - In deep snow a track becomes a rut.
- Exit ruts leading out of every car hole.
- Footprints.
- Plow-scrape striations. These are computed per pixel, because on the 15 px/m grid they aliased into moiré.

**Coverage targeting**: a bisection on the melt depth, run on the final per-pixel depth. The label at τ therefore matches the target exactly. In the first version it drifted, by up to 28 points.

**Appearance**:
- Coverage compares snow depth against the pavement micro-height. The micro-height now comes from the paint-free image, so paint lines no longer behave as tall ridges that stay bare.
- Paint lines count as ground and receive snow.
- Illumination is taken as a slow field. Asphalt patches stay in the albedo, so they no longer print as grey clouds on the snow.
- Shading is bounded, so steep slopes no longer render as black or glossy.
- Parallax uses integer shifts, so there is no resampling blur. A camera-facing face shows wherever the height steps down.

## Outputs per frame

- `v4_*.jpg`: the rendered image.
- `*_depth_mm.npy`: depth per pixel, as float16.
- `*_label_tau2mm.png`: the binary label.
- `v4_report.json` holds, for the inspection zone and weighted by ground area:
  - coverage at ≥1, 2, 5, 20 and 60 mm;
  - the melt applied;
  - mean depth and p95 depth in cm.

| Scenario | ≥1 mm | ≥2 mm (label) | ≥5 mm | ≥20 mm | ≥60 mm | Mean / p95 depth |
|---|---|---|---|---|---|---|
| Dusting, lot in use | 97% | 80% | 8% | 0% | 0% | 0.33 / 0.53 cm |
| 25 mm fresh, in use | 100% | 100% | 100% | 74% | 0% | 2.37 / 3.71 cm |
| Plowed, with film | 25% | 25% | 24% | 22% | 14% | 2.71 / 15.4 cm |
| Half plowed | 47% | 45% | 43% | 40% | 37% | 5.16 / 11.0 cm |

In the dusting scenario, the same frame scores 97%, 80% or 8% depending on the threshold. The "what counts as snow" threshold therefore has to be agreed with the client. The depth map lets us produce labels at any threshold from the same render. It also gives a thickness proxy in cm, which is the unit ГОСТ Р 50597 uses.

## Realism statistics

These use `img_stats.py`, re-implemented here. The absolute values are not comparable with the table in REALISM-RESEARCH-2.

| Crop | Mid-tone fraction | Spectral slope | Fine-detail energy |
|---|---|---|---|
| Real photo, near tracks | 0.60 | −1.74 | 21.3 |
| Real photo, far lot | 0.50 | −2.53 | 9.1 |
| v3, near field | 0.60 | −1.87 | 16.4 |
| v4 dusting, near field | 0.58 | −2.59 | 8.4 |
| v4 dusting, mid field | 0.44 | −2.53 | 2.0 |
| v4, 25 mm, near field | 0.55 | −2.31 | 6.1 |

v4 is better at layout and at deep snow: plow geometry, curved tracks, ruts, berms and holes. Its near-field fine detail is half of what v3 had.

The reason is that v3 packed 60 straight tracks into the near aisles. v4 spreads its tracks across the whole lot and partly refills them. The mid field is also too smooth.

## Known problems / next

1. **Fine detail.** Bring near-field fine detail back up to the real level. Options:
   - Track density per aisle, tied to how busy the lot is.
   - Stronger tread contrast.
   - Exemplar grain (step 2 of REALISM-RESEARCH-2).
2. **Mid-field texture.** It needs sub-pixel variance, meaning clump and wear patches at 5–30 cm.
3. **Edge piles.** Above about 25 cm the piles look glassy. They need exemplar texture and real occlusion. The parallax is capped at 25 cm for now.
4. **Ground truth.** The depth model is unvalidated against measured depths. Before anything is trusted:
   - Calibrate d0 (translucency) on real frames with known depth.
   - Calibrate the compaction ratios the same way.
   - Calibrate the coverage softness the same way.
5. **Speed.** Speed is fine for prototyping. Moving the tracks to the GPU, or rasterising them at an adaptive resolution, would make 10k frames practical.
