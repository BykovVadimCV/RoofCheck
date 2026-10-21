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
1. Normals for the whole frame (MoGe-2) → snow on every upward-facing surface.
2. Target-roof snow per SP 20 (μ by pitch, triangular drifts, mass conserved); depths weighted around the 30 cm trigger.
3. Volume: eave slab, chimney mounds, caps on railings and dishes; seam stripes fade with depth.
4. Two-flux opacity curve.
5. Host-photo matching (above).
6. Multi-threshold labels + ignore band; per-roof geometry; overlays blanked in network input.
7. Validation harness (null-paste test runnable now).
8. Optional: diffusion winterisation of trees/ground.
