#!/usr/bin/env python3
"""Reproducible train-only batches using the existing planar renderer.

Inputs are reviewed, snow-free, registered frames with reference annotations. This is
an experimental data preparation tool, not an independent validation dataset.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import numpy as np
import cv2
import PIL
import scipy
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
STATES = ('not_cleared', 'partially_cleared', 'cleared', 'thaw')
FIELDS = ('id', 'group_id', 'image', 'reference', 'split', 'source', 'license', 'snow_free_reviewed')
SAFE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def stable_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def snow_appearance(seed, variety, brightness=1.0, detail=1.0):
    if variety == 'fixed':
        return brightness, detail
    if variety != 'mixed':
        raise ValueError('snow variety must be fixed or mixed')
    # Separate RNG prevents appearance sampling from changing snow geometry.
    rng = np.random.default_rng(int(stable_hash(['snow-appearance-v3', seed])[:16], 16))
    draw = rng.random()
    profile = 0 if draw < 0.15 else (1 if draw < 0.65 else 2)
    if profile == 0:
        return round(float(rng.uniform(0.97, 1.03)), 4), round(float(rng.uniform(0.7, 1.0)), 4)
    if profile == 1:
        return round(float(rng.uniform(1.04, 1.10)), 4), round(float(rng.uniform(0.12, 0.35)), 4)
    brightness = round(float(rng.uniform(1.04, 1.10)), 4)
    detail = 0.0 if rng.random() < 0.25 else round(float(rng.uniform(0, 0.03)), 4)
    return brightness, detail


def verify_code_hashes(expected):
    for name, sha in expected.items():
        if digest(ROOT / name) != sha:
            raise ValueError(f'Code changed during the run: {name}; use a new run folder')


def read_input(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not set(FIELDS).issubset(reader.fieldnames):
            raise ValueError(f'Input CSV must contain: {", ".join(FIELDS)}')
        rows = list(reader)
    if not rows:
        raise ValueError('Input CSV is empty')
    ids, groups, hashes = set(), {}, {}
    for row in rows:
        if any(not (row.get(k) or '').strip() for k in FIELDS):
            raise ValueError('All input fields are required')
        if not SAFE.fullmatch(row['id']) or not SAFE.fullmatch(row['group_id']):
            raise ValueError('id and group_id must use ASCII letters, numbers, _, . or -')
        if row['id'] in ids:
            raise ValueError(f"Duplicate id: {row['id']}")
        ids.add(row['id'])
        if row['split'] != 'train':
            raise ValueError('Only train backgrounds are permitted; validation/test must remain real')
        if row['snow_free_reviewed'].lower() != 'yes':
            raise ValueError(f"Snow-free status must be reviewed for {row['id']}")
        for field in ('image', 'reference'):
            p = Path(row[field]).expanduser()
            row[field] = str((path.parent / p).resolve() if not p.is_absolute() else p.resolve())
            if not Path(row[field]).is_file():
                raise FileNotFoundError(row[field])
        from roofcheck.schema import Reference
        ref = Reference.load(Path(row['reference']))
        if not ref.object_id:
            raise ValueError('Reference object_id must be set')
        # Different references for one object must be assigned the same building group.
        if ref.object_id in groups and groups[ref.object_id] != row['group_id']:
            raise ValueError(f'Object {ref.object_id} appears in multiple groups')
        groups[ref.object_id] = row['group_id']
        image_hash = digest(Path(row['image']))
        if image_hash in hashes and hashes[image_hash] != row['group_id']:
            raise ValueError('Identical images occur under different groups')
        hashes[image_hash] = row['group_id']
        row['image_sha256'] = image_hash
        row['reference_sha256'] = digest(Path(row['reference']))
    return rows


def read_textures(directory, manifest, train_groups):
    """Record every file the renderer may read; never infer licence or split."""
    from roofcheck import synth
    directory = directory.resolve()
    if not directory.is_dir() or not manifest.is_file():
        raise ValueError('Texture directory and provenance CSV must exist')
    found = {}
    for raw in synth.texture_paths([str(directory)]):
        file = Path(raw).resolve()
        try:
            key = file.relative_to(directory).as_posix()
        except ValueError as exc:
            raise ValueError('Texture must not resolve outside texture-dir') from exc
        if cv2.imread(str(file), cv2.IMREAD_GRAYSCALE) is None:
            raise ValueError(f'Unreadable texture: {file}')
        found[key] = file
    if not found:
        raise ValueError('No supported textures found')
    with manifest.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        fields = {'path', 'source', 'license', 'donor_group', 'donor_split'}
        if not reader.fieldnames or not fields.issubset(reader.fieldnames):
            raise ValueError('Texture manifest requires path,source,license,donor_group,donor_split')
        rows = list(reader)
    assets = {}
    for row in rows:
        if any(not (row.get(k) or '').strip() for k in fields):
            raise ValueError('Texture provenance fields must be nonempty')
        key = row['path']
        if key in assets or key not in found:
            raise ValueError(f'Duplicate or unused texture: {key}')
        if row['donor_split'] not in ('train', 'external'):
            raise ValueError('Texture donors must belong to train or a verified external source')
        if row['donor_split'] == 'train' and row['donor_group'] not in train_groups:
            raise ValueError('Train texture donor_group must be one of the input train groups')
        assets[key] = {**{k: row[k] for k in sorted(fields)}, 'sha256': digest(found[key])}
    if set(assets) != set(found):
        raise ValueError('Every texture the renderer can use must appear in the provenance CSV')
    return {'directory': str(directory), 'manifest_sha256': digest(manifest), 'assets': assets}


def atomic_text(path, content):
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as f:
        tmp = Path(f.name)
        try:
            f.write(content)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    os.replace(tmp, path)


def check_sample(path, source=None, scale=1.0):
    with Image.open(path / 'image.png') as im:
        image = np.asarray(im.convert('RGB'))
    with Image.open(path / 'snow_threshold_2mm_experimental.png') as im:
        mask = np.asarray(im)
    with Image.open(path / 'roi.png') as im:
        roi = np.asarray(im)
    if mask.ndim != 2 or roi.ndim != 2 or image.shape[:2] != mask.shape or mask.shape != roi.shape:
        raise ValueError(f'{path}: incompatible image/mask/ROI dimensions')
    if not set(np.unique(mask).tolist()) <= {0, 1, 255} or not set(np.unique(roi).tolist()) <= {0, 1}:
        raise ValueError(f'{path}: invalid mask values')
    if not np.any(roi == 1) or not np.any(mask != 255):
        raise ValueError(f'{path}: no labelled roof pixels')
    if np.any(mask[roi == 0] != 255):
        raise ValueError(f'{path}: labelled pixels outside ROI')
    qa = {'roi_pixels': int((roi == 1).sum()), 'snow_pixels': int((mask == 1).sum()),
          'clean_pixels': int((mask == 0).sum()), 'ignore_pixels': int((mask == 255).sum())}
    depth_path = path / 'depth_mm_experimental.npy'
    if depth_path.is_file():
        depth = np.load(depth_path, allow_pickle=False)
        if depth.shape != mask.shape or not np.isfinite(depth).all() or (depth < 0).any():
            raise ValueError(f'{path}: invalid experimental depth array')
        valid_depth = depth[mask != 255]
        qa['simulated_roof_depth_p50_mm'] = round(float(np.percentile(valid_depth, 50)), 3)
        qa['simulated_roof_depth_p95_mm'] = round(float(np.percentile(valid_depth, 95)), 3)
    if source is not None:
        with Image.open(source) as im:
            original = np.asarray(ImageOps.exif_transpose(im).convert('RGB'))
        if scale != 1.0:
            original = cv2.resize(original, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if original.shape != image.shape:
            raise ValueError(f'{path}: source and output dimensions differ')
        differences = np.abs(image.astype(np.int16) - original.astype(np.int16)).max(axis=2)
        if np.any(roi == 0):
            delta = np.abs(image.astype(np.int16) - original.astype(np.int16))[roi == 0]
            qa['outside_roi_mean_abs_change'] = round(float(delta.mean()), 4)
            qa['outside_roi_fraction_changed_gt8'] = round(float((delta.max(axis=1) > 8).mean()), 4)
        snow = mask == 1
        # This is an appearance warning, not a test of the label's correctness.
        qa['snow_fraction_rgb_change_le8'] = round(float((differences[snow] <= 8).mean()), 4) if snow.any() else None
    return qa


def contact_sheet(output, records):
    cards = []
    for record in records[:48]:
        p = output / record['path']
        with Image.open(p / 'image.png') as f:
            im = f.convert('RGB')
        with Image.open(p / 'snow_threshold_2mm_experimental.png') as f:
            mask = np.asarray(f)
        layer = np.zeros((*mask.shape, 4), dtype=np.uint8)
        layer[mask == 1] = (0, 220, 255, 105)
        # Grey is intentionally shown for ignored areas, including outside the ROI.
        layer[mask == 255] = (110, 110, 110, 90)
        over = Image.alpha_composite(im.convert('RGBA'), Image.fromarray(layer)).convert('RGB')
        im.thumbnail((320, 200)); over.thumbnail((320, 200))
        card = Image.new('RGB', (640, 224), 'white')
        card.paste(im, (0, 24)); card.paste(over, (320, 24))
        ImageDraw.Draw(card).text((4, 4), record['id'][:90], fill='black')
        cards.append(card)
    if cards:
        sheet = Image.new('RGB', (640, 224 * len(cards)), 'white')
        for i, card in enumerate(cards): sheet.paste(card, (0, i * 224))
        sheet.save(output / 'contact_sheet.jpg', quality=88)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path, help='CSV manifest; paths relative to CSV')
    parser.add_argument('--output', required=True, type=Path, help='dedicated run directory')
    parser.add_argument('--per-state', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--scale', type=float, default=1.0)
    parser.add_argument('--context', choices=('scene', 'local', 'local-volume'), default='scene')
    depths = parser.add_mutually_exclusive_group()
    depths.add_argument('--ground-snow-mm', type=float, help='optional simulated ground snow depth in mm')
    depths.add_argument('--ground-snow-levels', type=float, nargs='+', help='simulated ground depths in mm, cycled across variants')
    parser.add_argument('--save-depth', action='store_true', help='save experimental simulation depth arrays')
    parser.add_argument('--feather-px', type=int, default=4, help='local blending width in output pixels')
    parser.add_argument('--fence-mode', choices=('auto', 'none'), default='auto', help='none only for a verified fence-free roof')
    parser.add_argument('--structure-mode', choices=('auto', 'manual'), default='auto', help='manual uses annotated structures only')
    parser.add_argument('--partial-style', choices=('legacy', 'soft_front'), default='legacy')
    parser.add_argument('--visibility-threshold', type=int, help='optional low-change snow ignore threshold, local mode only')
    parser.add_argument('--texture-dir', type=Path)
    parser.add_argument('--texture-manifest', type=Path, help='required provenance CSV for every discovered texture')
    parser.add_argument('--texture-policy', choices=('legacy', 'isolated'), default='legacy')
    parser.add_argument('--texture-strength', type=float, default=0.02, help='isolated material contrast, from 0 to 0.25')
    parser.add_argument('--snow-brightness', type=float, default=1.0, help='snow brightness multiplier: 0.7..1.4')
    parser.add_argument('--snow-detail', type=float, default=1.0, help='snow high-frequency detail: 0..1')
    parser.add_argument('--snow-variety', choices=('fixed', 'mixed'), default='mixed', help='mixed chooses among three snow appearance levels deterministically (default)')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.per_state < 1 or args.per_state > 100 or not 0 < args.scale <= 1 or not 0 <= args.feather_px <= 100:
        parser.error('per-state must be 1..100; scale >0 and <=1; feather-px 0..100')
    if args.visibility_threshold is not None and (args.context not in ('local','local-volume') or not 0 <= args.visibility_threshold <= 255):
        parser.error('visibility-threshold requires local context and must be 0..255')
    if args.ground_snow_mm is not None and not 0 <= args.ground_snow_mm <= 1000:
        parser.error('ground-snow-mm must be 0..1000')
    if args.ground_snow_levels and (any(not 0 <= x <= 1000 for x in args.ground_snow_levels) or len(set(args.ground_snow_levels)) != len(args.ground_snow_levels)):
        parser.error('ground-snow-levels must be distinct numbers in 0..1000')
    if args.ground_snow_levels and args.per_state < len(args.ground_snow_levels):
        parser.error('per-state must be at least the number of ground-snow-levels')
    if not 0 <= args.texture_strength <= 0.25:
        parser.error('texture-strength must be 0..0.25')
    if not 0.7 <= args.snow_brightness <= 1.4 or not 0 <= args.snow_detail <= 1:
        parser.error('snow-brightness must be 0.7..1.4 and snow-detail 0..1')
    if args.snow_variety == 'mixed' and (args.snow_brightness != 1 or args.snow_detail != 1):
        parser.error('mixed snow-variety chooses its own brightness and detail')
    rows = read_input(args.input.resolve())
    if bool(args.texture_dir) != bool(args.texture_manifest):
        parser.error('provide both --texture-dir and --texture-manifest, or neither')
    textures = read_textures(args.texture_dir, args.texture_manifest.resolve(), {r['group_id'] for r in rows}) if args.texture_dir else None
    code = {str(p.relative_to(ROOT)): digest(p) for p in sorted((ROOT / 'roofcheck').rglob('*.py'))}
    code['tools/render_existing.py'] = digest(ROOT / 'tools/render_existing.py')
    code['tools/batch_existing.py'] = digest(Path(__file__).resolve())
    run = {'version': 1, 'inputs': rows, 'states': STATES, 'per_state': args.per_state,
           'seed': args.seed, 'scale': args.scale, 'context': args.context, 'ground_snow_mm': args.ground_snow_mm, 'ground_snow_levels': args.ground_snow_levels, 'save_depth': args.save_depth, 'feather_px': args.feather_px, 'fence_mode': args.fence_mode,
           'structure_mode': args.structure_mode, 'partial_style': args.partial_style,
           'visibility_threshold': args.visibility_threshold,
           'texture_policy': args.texture_policy, 'texture_strength': args.texture_strength,
           'snow_variety': args.snow_variety, 'snow_brightness': args.snow_brightness, 'snow_detail': args.snow_detail,
           'texture_mode': 'exemplar' if textures else 'procedural_only', 'textures': textures, 'code_sha256': code,
           'python': sys.version, 'numpy': np.__version__, 'opencv': cv2.__version__,
           'pillow': PIL.__version__, 'scipy': scipy.__version__}
    fingerprint = stable_hash(run)
    out = args.output.resolve()
    if out.exists():
        if not args.resume or not (out / 'run.json').is_file():
            raise FileExistsError(f'{out}: use a new folder or --resume with a matching run')
        if json.loads((out / 'run.json').read_text())['fingerprint'] != fingerprint:
            raise ValueError('Run inputs, configuration, code or environment changed; use a new folder')
    else:
        out.mkdir(parents=True)
        atomic_text(out / 'run.json', json.dumps({'fingerprint': fingerprint, 'config': run}, ensure_ascii=False, indent=2))
    (out / 'samples').mkdir(exist_ok=True)
    records, manifest = [], []
    for row in rows:
        for state in STATES:
            for index in range(args.per_state):
                sample_id = f"{row['id']}__{state}__{index:03d}"
                target = out / 'samples' / sample_id
                seed = int(stable_hash([args.seed, row['id'], state, index])[:16], 16)
                ground_depth = args.ground_snow_levels[index % len(args.ground_snow_levels)] if args.ground_snow_levels else args.ground_snow_mm
                brightness, detail = snow_appearance(seed, args.snow_variety, args.snow_brightness, args.snow_detail)
                expected = stable_hash([fingerprint, sample_id, seed])
                verify_code_hashes(code)
                if target.exists():
                    record_path = target / 'batch_record.json'
                    if not args.resume or not record_path.is_file():
                        raise FileExistsError(f'{target}: incomplete or unexpected sample; inspect it manually')
                    record = json.loads(record_path.read_text())
                    if record['sample_fingerprint'] != expected:
                        raise ValueError(f'{target}: sample fingerprint mismatch')
                    for name, sha in record['files_sha256'].items():
                        if digest(target / name) != sha: raise ValueError(f'{target / name}: checksum mismatch')
                    if check_sample(target, row['image'], args.scale) != record['qa']: raise ValueError(f'{target}: QA mismatch')
                else:
                    cmd = [sys.executable, str(ROOT / 'tools/render_existing.py'), '--image', row['image'],
                           '--reference', row['reference'], '--output', str(target), '--state', state,
                           '--seed', str(seed), '--scale', str(args.scale), '--context', args.context,
                           '--feather-px', str(args.feather_px), '--fence-mode', args.fence_mode,
                           '--structure-mode', args.structure_mode, '--partial-style', args.partial_style, '--texture-policy', args.texture_policy,
                           '--texture-strength', str(args.texture_strength), '--snow-brightness', str(brightness),
                           '--snow-detail', str(detail)]
                    if textures: cmd += ['--texture-dir', textures['directory']]
                    if ground_depth is not None: cmd += ['--ground-snow-mm', str(ground_depth)]
                    if args.save_depth: cmd += ['--save-depth']
                    if args.visibility_threshold is not None: cmd += ['--visibility-threshold', str(args.visibility_threshold)]
                    subprocess.run(cmd, cwd=ROOT, check=True)
                    verify_code_hashes(code)
                    qa = check_sample(target, row['image'], args.scale)
                    files = ('image.png', 'snow_threshold_2mm_experimental.png', 'roi.png', 'metadata.json')
                    if args.save_depth: files += ('depth_mm_experimental.npy',)
                    record = {'id': sample_id, 'path': f'samples/{sample_id}', 'group_id': row['group_id'],
                              'source_id': row['id'], 'source': row['source'], 'license': row['license'],
                              'state': state, 'seed': seed, 'ground_snow_mm': ground_depth, 'snow_brightness': brightness, 'snow_detail': detail,
                              'sample_fingerprint': expected, 'qa': qa,
                              'files_sha256': {name: digest(target / name) for name in files}}
                    atomic_text(target / 'batch_record.json', json.dumps(record, ensure_ascii=False, indent=2))
                records.append(record)
                base = record['path']
                manifest.append({'id': sample_id, 'image': f'{base}/image.png',
                                 'mask': f'{base}/snow_threshold_2mm_experimental.png',
                                 'group_id': row['group_id'], 'split': 'train', 'kind': 'synthetic',
                                 'roi': f'{base}/roi.png', 'depth_experimental': f'{base}/depth_mm_experimental.npy' if args.save_depth else '',
                                 'ground_snow_mm': '' if ground_depth is None else ground_depth, 'state': state})
    import io
    buf = io.StringIO(newline='')
    writer = csv.DictWriter(buf, fieldnames=['id', 'image', 'mask', 'group_id', 'split', 'kind', 'roi', 'depth_experimental', 'ground_snow_mm', 'state'])
    writer.writeheader(); writer.writerows(manifest)
    atomic_text(out / 'synthetic_manifest.csv', buf.getvalue())
    atomic_text(out / 'qa_report.json', json.dumps({'fingerprint': fingerprint, 'samples': records}, ensure_ascii=False, indent=2))
    contact_sheet(out, records)
    print(f'{len(records)} samples checked; see {out}')


if __name__ == '__main__':
    sys.path.insert(0, str(ROOT))
    main()
