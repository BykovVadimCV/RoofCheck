# Research notes and decisions (October 2026)

Verified against primary sources unless marked otherwise.

## Snow optics (thin layers)
- Perovich 2007 (J. Glaciology): extinction at 500 nm 0.062–0.077 cm⁻¹ for dry snow (e-folding ≈ 13–16 cm), 0.219 cm⁻¹ for wet snow (≈ 4.5 cm).
  Over a black platform: visible at 0.5 cm, barely distinguishable at 3 cm, as white as deep snow at 8 cm; albedo 0.85–0.90 by ~4.5 cm.
- Wiscombe & Warren: pure snow is optically semi-infinite at ≥ ~10 cm.
- Decision: opacity α(D) = βD / (1 + βD), β = 0.2–0.5 mm⁻¹ (higher for wet snow). The old 1 − exp(−D/2–4 mm) saturated by ~1 cm.
- Labels: thresholds 1 / 2 / 5 / 20 / 300 mm, ±1 mm ignore band around each.

## Matching the host photo
- What survives the model's 518-px input: blur, haze/contrast, colour and clipping, coarse noise, sharpening halos ≥ ~1.5 px.
  CFA, PRNU and JPEG-grid traces do not; not worth emulating.
- Decision: per-photo blur (edge-spread fit), dark-channel haze (t, A), brightness-dependent noise σ²(I) = a·I + c,
  matched unsharp mask, white point from neutral pixels, clip to host 99.5 %, save at host JPEG quality.
- Not usable: TruFor/Noiseprint (nonprofit-only licence); `timothybrooks/unprocessing` has no licence file.

## Validation
- KID, not FID, at N ≤ 500 (Bińkowski et al. 2018). DINOv2 / CLIP features, not Inception (Stein et al. 2023).
- Two feature families: semantic (DINOv2) and low-level statistics; measure at native and model-input scale.
- All thresholds calibrated against a real-vs-real floor. Null-paste and blur/re-JPEG stress tests for shortcuts.
- Tools: `dgm-eval` (MIT), `prdc` (MIT). Best realism reference: the client's own ЕЦХД and report archives.

## Snow loads and rules
- Moscow: snow region III, Sg = 1.5 kPa (SP 20.13330.2016). μ = 1 up to 30°, 0 from 60°, linear between (scheme images not opened — flagged).
  Drift at height difference b = 2h ≤ 16 m. Ventilation-shaft μ table from one report: unverified, not used.
- Clearing trigger: 30 cm on roofs with external drainage (Госстрой № 170, п. 4.6.1.23; repeated by МО Госжилинспекция). Residue after clearing is not regulated.

## Depth / normal models (commercial use)
- MoGe-2 `-normal` (ViT-B 419 MB, ViT-L 1.32 GB): code MIT, weights MIT → primary.
- Depth Anything 3: Small / Base / Metric-Large / Mono-Large Apache-2.0; Large / Giant NC. No normals output.
- Depth Pro: Apple sample-code licence (permissive). Marigold-normals v1.1: OpenRAIL++-M. StableNormal: Apache-2.0 card, SD-based.
- Excluded: DSINE (Imperial College, non-commercial), Metric3D weights (no licence stated), DA-V2 Base/Large (CC BY-NC).

## Rework queue for the generator
Done (October 2026, `synth.py` v2, `hostmatch.py`, `tests/validate.py`):
- 2. Amount per SP 20 (μ by pitch; c_e, c_t; unbalanced slopes), ground snow drawn around the 30 cm trigger, fence and
  obstruction drifts (length 2h ≤ 5 m, μ ≤ 2.4) taken from the upwind part of the roof (mass ratio 1.0). States:
  not_cleared / partially_cleared / cleared / thaw; coverage balancing only in thaw. Geometry fixed per object.
- 3. Eave slab (height = eave depth × photo px per metre), rounded caps on outlined structures, cap on the top rail,
  cornice lip; seam print-through fades with depth.
- 4. Two-flux opacity α = βD/(1+βD), β 0.35–0.6 /mm + up to 0.4 for wet snow.
- 5. Host matching wired in: premultiplied blur with the measured PSF, haze on snow radiance, USM from measured
  overshoot, NLF noise, clip, JPEG at the host quality; overlays restored after processing.
- 6. Labels at 1/2/5/20/300 mm with a ±1 mm ignore band, ROI mask, overlays blanked in the network input.
- 7. Null-paste test with a positive control. Result on the 4 demo backgrounds: unpaired C2ST at chance (0.49) for
  both null and control; the paired sign test catches the control (bias 0.5) and finds a small systematic loss of
  chroma noise from the re-encode in the null (bias 0.2 at 518 px). Mitigation: same final re-encode for real images.

Fixes found on the way: structure detection by darkness flagged 38% of the slope (shadows, seams); now by colour,
with structures outlined once per vantage (`structure` role) since grey chimneys match metal in colour.

- 1. Normals for the whole frame (`roofcheck/normals.py`, MoGe-2 ViT-B normal, CPU ~40 s per photo, cached).
  Up vector fitted from facade normals (catches roll on the tilted photo). Roof pitch from normals: 26.0, 20.9,
  24.3, 31.2 deg on the 4 photos, median 25.1 used as the object's pitch. Snow on every surface tilted < 60 deg
  (depth = ground snow × μ(tilt)), mask snapped to SLIC superpixels so it follows real roof edges; thin ledges
  added per pixel; each surface cleared or left on its own; in a thaw the whole town melts. Haze by depth:
  t(d) = t_roof·exp(−b(d − d_roof)), b fitted per photo from the dark channel (0.0005–0.0036 /m).

Realism round 3 (critique: no aerial perspective, one texture at all distances, cut-out edges, October scene):
- Haze: snow at distance r (x target depth) drawn as V(r) + c(r)(C − V(1)); c(r) and V(r) measured on each photo
  (luminance range and level per distance bin, colour of the veil from the sky). Replaces the single fog fit,
  which gave near-zero fog on flat overcast frames. `tests/validate.py` distance check: before, snow kept full
  contrast at 6× distance (0.73–1.18 of near) while the scene fell to 0.53–0.79; now 0.43–0.82 (noisy, 4 photos).
- Texture in metres: grain ~17 cm and 4–8 m exemplar tiles scaled by px/m at each pixel's distance, amplitude
  ∝ r^-0.7, extra softening with distance. Snow detail now falls to 0.42–0.75 of near at the far band.
- Other roofs: outline wobble ~0.25 m, thinning over the last 0.15–0.5 m to a free edge, triangular drifts
  0.8–2.5 m against walls and parapets above, ~6 cm soft edge.
- Season: foliage mask = leaf colour + texture + rough normals (colour alone took ochre facades and window
  frames); whole crowns greyed with branch detail kept and leaf-sized gaps; snow along crown tops after heavy
  falls; AWB shift (cooler, scaled by snow share). Batch adds real autumn photos through the same pipeline,
  labelled clean, so leafy trees are not a cue for "synthetic".

Rendering round 4 (critique: snow on the target roof looks flat; depth model fine, rendering 2D):
- Snow is lit as a height field on the slope grid (2.5 cm px): surface = smoothed depth + seam ridges that
  widen and fade with depth (lambda 10-15 cm) + 0.5-2 m undulation; structures stand 1.5 m. Shading = CIE
  overcast sky term from the normal x horizon-based sky visibility (8 azimuths, 1.5 m reach) + weak
  directional term (10-30%). Close-up snow scans no longer painted on as albedo (they read as camouflage at
  roof scale); albedo only for old snow / soot spots, weak.
- Cover over a seam = D - 25 mm x (1 - exp(-D/120)): snow mounds over seams, so seams fade instead of
  showing as dark slashes through 6 cm of snow.
- Exposure: in 75% of samples the brightest snow sits at 0.88-0.97 of the host's white point, 25% allowed to
  clip. Before: 16-83% of snow pixels clipped in many samples.
- Railing in front of snow: soft matte keyed on bar darkness instead of a binary mask (no dashed bars).
- Bug found by the new shading: wind scour removed snow from a hard-edged rectangle (upwind half minus
  drift area), leaving a step in the depth map. Now a smooth weight.

Open:
- Small or partly hidden crowns keep their leaves; outlines on the nearest roofs look slightly torn.
8. Optional: diffusion winterisation of trees/ground (now: simple desaturation of foliage).
- Snow texture at 2× zoom reads as mottled; needs comparison with real winter roofs from this distance.
- Rust stains with strong colour still count as structures (bite out of the snow).
- Window sills get snow only where the normals are clean; many sills stay bare.
- Trees and foliage get no snow (red/dark foliage partly escapes the vegetation mask).
- Real-vs-synthetic comparison waits for real winter photos (`tests/validate.py --real`).
