# Roof-snow dataset: Karlsruhe RGB to training candidates

This workflow uses original 4000×3000 RGB photographs and the existing experimental renderer. It generates candidates for **binary semantic segmentation** (snow vs clean roof), a reasonable starting point for DeepLabV3+ / SegFormer comparisons. It does not determine the best model; compare on an independently labelled real rooftop test set. Neither the source photos nor the code contain snow labels from real winter roofs.

## 1. Index the photographs

Install Python dependencies for the renderer (NumPy, OpenCV, SciPy, Pillow; see `requirements.txt`). From the repository root:

```bash
python3 tools/prepare_roof_dataset.py init \
  --root /Users/bykovvadim/Downloads --karlsruhe \
  --output out/karlsruhe_inventory
```

For a different archive, omit `--karlsruhe`; all JPG/PNG files under `--root` are indexed. `inventory.csv` refers to original files via absolute paths. It has one stable ID per relative source path, so repeated `DJI_*.jpg` names from different flights do not collide. Do not move or edit the source files afterwards. For a new project choose a new output directory; existing outputs are never overwritten.

For CVAT, first select the images in `inventory.csv` as described below. Then copy only those originals under unique names, without recompression:

```bash
python3 tools/prepare_roof_dataset.py stage-selected \
  --inventory out/karlsruhe_inventory/inventory.csv \
  --output out/karlsruhe_selected_images
```

Alternatively, add `--stage` to the initial indexing command to copy all originals. This takes additional disk space. The actual prebuilt inventory for this machine is `out/karlsruhe_inventory_v1/inventory.csv`.

Karlsruhe is licensed CC BY 4.0; retain attribution and verify all license obligations. The 2,848 RGB photographs contain strong overlap and repeatedly show the same buildings; they do not constitute 2,848 independent buildings. Determine the building/area grouping **before** reserving independent real validation and test photographs. One scene may show several buildings; group connected/overlapping views conservatively at the block level if necessary. The script cannot infer this or assess snow-free status.

## 2. Review and annotate selected training photographs

In `inventory.csv` choose photographs, set `selected=yes`, `split=train`, `snow_free_reviewed=yes`, and enter a manually assigned `group_id` using ASCII letters, numbers, `_`, `-`, `.`. Leave unselected photographs as `selected=no`. Only select snow-free photographs whose use and grouping have been checked. Reserve independent real validation/test buildings before selecting training data. Manual review is necessary: the archive is not a roof-geometry or snow-mask dataset.

Create a CVAT image task using selected staged originals; use `cvat_labels.json` as a label specification (or define the polygon labels below manually). Coordinate frames must match the EXIF-oriented original image; verify their displayed orientation. For **one target roof slope per photograph**, annotate:

* `roof`: one polygon enclosing the intended roof region.
* `slope`: exactly one convex four-point polygon, in order eave-left, eave-right, ridge-right, ridge-left. Corners must be within the roof polygon.
* `structure`, `exclude`, `overlay`: polygons for chimneys/roof equipment, occlusions, and markings as applicable. Use `exclude` when uncertain.

Export **CVAT for images 1.1** XML and import it:

```bash
python3 tools/prepare_roof_dataset.py import-cvat \
  --inventory out/karlsruhe_inventory/inventory.csv \
  --xml /path/to/annotations.xml --output out/karlsruhe_prepared
```

All selected rows must appear in the XML; output is written only after the input is validated. The importer checks sizes, polygon counts, bounds and geometry, but cannot verify actual geometry, corner semantics, snow absence, provenance, or overlap between different photographs. Different roofs in one photo currently need separate annotation tasks/inputs, not multiple slope polygons for one image. The current generator renders one slope at a time, not every roof in a scene.

Export genuine negative examples from the same manually checked clean photographs:

```bash
python3 tools/export_clean_roofs.py --input out/karlsruhe_prepared/input.csv \
  --output out/karlsruhe_real_clean --scale 1.0
```

The mask is 0 on the annotated snow-free slope and 255 outside it and around exclusions. Polygon edges are conservatively ignored. These are real clean-roof examples, not real snow labels. The source photo must actually be snow-free. Do not use the demonstration's rough polygon as training ground truth.

## 3. Generate candidates and diagnostic simulation depths

```bash
python3 tools/batch_existing.py \
  --input out/karlsruhe_prepared/input.csv --output out/karlsruhe_snow_v1 \
  --per-state 5 --ground-snow-levels 5 20 60 150 300 \
  --context local-volume --partial-style soft_front \
  --structure-mode manual --snow-variety mixed --save-depth --seed 42
```

The five numbers are **simulated ground snow depths**, not measured roof thicknesses. Each selected photograph gets five variants for each of four simulation states (`not_cleared`, `partially_cleared`, `cleared`, `thaw`): 20 candidates total. The actual synthetic roof field depends on state, weather, and geometry; variants are not guaranteed to form ordered physical thickness classes. The roof dimensions/pitch are uncalibrated priors unless measured geometry is supplied. `local-volume` makes an approximate projected 2.5D snow heightfield, not full physical 3D reconstruction; adjacent roofs and the surrounding scene can remain snow-free. Inspect seasonal consistency, seams, overhangs, shadows, structures and any summer context.

The output includes an RGB image, a binary mask, ROI, experimental float32 `.npy` depth array, metadata, checksums and a manifest. Mask values: `0` clean, `1` snow according to the renderer's **experimental 2 mm threshold**, `255` ignore. Outside the ROI is ignored, including visible projected overhangs; structure and blend boundary may also be ignored. Snow in the simulated depth field is not necessarily pixel-perfect visible snow; correct or reject inaccurate masks. Treat depth as a synthetic simulation parameter only, not ground-truth actual millimetres and not evidence that monocular physical thickness is recoverable. The 2 mm threshold is not validated as visible snow. The `qa_report.json` lists simulated depth percentiles and pixel counts, not a realism or model-quality score. Some outputs can contain no positive snow pixels; review balance and include genuine clean-roof negatives as well.

Output paths in the manifest are relative to the run directory. To restart a completed or interrupted run, rerun the identical command with `--resume`. A sample directory without `batch_record.json` indicates interruption and requires manual inspection before re-running. Source or code changes require a new run directory.

## 4. Check every image and mask; export reviewed candidates

```bash
python3 tools/review_batch.py --run out/karlsruhe_snow_v1 --template
# Inspect every image and mask. In review.csv fill decision=accept or reject,
# reviewer, and note for each row.
python3 tools/review_batch.py --run out/karlsruhe_snow_v1 --export
```

The contact sheet displays only the first 48 samples; inspect the rest individually. `approved_manifest.csv` is a training candidate list after manual review; the review script checks hashes and produces an audit, but cannot judge mask quality. Combine approved synthetic samples with the real samples:

```bash
python3 tools/assemble_roof_dataset.py \
  --manifest out/karlsruhe_real_clean/real_clean_manifest.csv \
  --manifest out/karlsruhe_snow_v1/approved_manifest.csv \
  --manifest /path/to/real_snow_manifest.csv \
  --output out/roof_dataset
```

Every input manifest uses paths relative to its own directory. The assembler copies the RGB images, masks and ROIs into a self-contained output with a `manifest.csv` compatible with `notebooks/03_Roof_finetune_real_vs_synthetic.ipynb` in the parent workspace. It checks dimensions, values and group/split consistency, but cannot detect overlapping buildings assigned different group IDs. Include at least one real train record for every synthetic train group, plus real snow and clean examples in each train/val/test split. It reports incomplete splits; the notebook will not train until they are populated. Copies omit the optional simulation depth arrays: they remain available in the original generation run. Keep all related synthetic variants in training only, and evaluate on real held-out buildings/scenes. Compare training with and without synthetic examples using snow IoU, snow precision/recall and failure cases on real roofs. High synthetic accuracy alone is not evidence of generalization.
