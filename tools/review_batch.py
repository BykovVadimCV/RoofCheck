#!/usr/bin/env python3
"""Prepare an explicit visual review and export only accepted synthetic samples.

The reviewer must inspect the rendered image and its mask, not just automatic QA.
No program can determine from this CSV whether a review actually happened.
"""
import argparse
import csv
import io
import json
from pathlib import Path
import sys

from batch_existing import atomic_text, check_sample, digest

COLUMNS = ['id', 'decision', 'reviewer', 'note']


def load_run(root):
    run = json.loads((root / 'run.json').read_text())
    report = json.loads((root / 'qa_report.json').read_text())
    if run['fingerprint'] != report['fingerprint']:
        raise ValueError('Run and QA fingerprints differ')
    samples = report['samples']
    if not samples or len({r['id'] for r in samples}) != len(samples):
        raise ValueError('Empty QA report or duplicate IDs')
    return run, samples


def csv_text(fields, rows):
    buf = io.StringIO(newline='')
    w = csv.DictWriter(buf, fieldnames=fields); w.writeheader(); w.writerows(rows)
    return buf.getvalue()


def template(root, samples):
    path = root / 'review.csv'
    if path.exists():
        raise FileExistsError(f'{path} already exists; review is never overwritten')
    atomic_text(path, csv_text(COLUMNS, [{'id': x['id'], 'decision': 'pending', 'reviewer': '', 'note': ''} for x in samples]))
    print('Review template:', path)


def export(root, run, samples, destination):
    review_file = root / 'review.csv'
    with review_file.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not set(COLUMNS).issubset(reader.fieldnames):
            raise ValueError('review.csv requires id,decision,reviewer,note')
        reviews = list(reader)
    ids = [r['id'] for r in reviews]
    if len(ids) != len(set(ids)) or set(ids) != {s['id'] for s in samples}:
        raise ValueError('Review must contain exactly one row per QA sample')
    for r in reviews:
        if r['decision'] not in ('accept', 'reject'):
            raise ValueError(f"{r['id']}: decision must be accept or reject; pending cannot be exported")
        if not r['reviewer'].strip() or not r['note'].strip():
            raise ValueError(f"{r['id']}: reviewer and note are required")
    decisions = {r['id']: r for r in reviews}
    with (root / 'synthetic_manifest.csv').open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f); fields = reader.fieldnames; candidates = list(reader)
    if not fields or not {'id', 'image', 'mask', 'group_id', 'split', 'kind', 'roi'}.issubset(fields):
        raise ValueError('Candidate manifest has an invalid format')
    ids = [r['id'] for r in candidates]
    if len(ids) != len(set(ids)) or set(ids) != set(decisions):
        raise ValueError('Candidate manifest differs from QA samples')
    candidates_by_id = {r['id']: r for r in candidates}
    accepted = []
    for sample in samples:
        sid = sample['id']; rel = sample['path']
        if rel != f'samples/{sid}' or (root / rel).resolve().parent != (root / 'samples').resolve():
            raise ValueError(f'{sid}: invalid sample path')
        expected = {'image': f'{rel}/image.png', 'mask': f'{rel}/snow_threshold_2mm_experimental.png',
                    'roi': f'{rel}/roi.png', 'group_id': sample['group_id'], 'split': 'train', 'kind': 'synthetic'}
        if 'depth_experimental' in fields:
            expected['depth_experimental'] = f'{rel}/depth_mm_experimental.npy' if run['config'].get('save_depth') else ''
        if 'state' in fields:
            expected['state'] = sample['state']
        candidate = candidates_by_id[sid]
        if any(candidate.get(k) != v for k, v in expected.items()):
            raise ValueError(f'{sid}: candidate manifest does not match QA record')
        if decisions[sid]['decision'] != 'accept':
            continue
        folder = root / rel
        record = json.loads((folder / 'batch_record.json').read_text())
        if record != sample:
            raise ValueError(f'{sid}: batch record differs from QA report')
        expected_files = {'image.png', 'snow_threshold_2mm_experimental.png', 'roi.png', 'metadata.json'}
        if run['config'].get('save_depth'):
            expected_files.add('depth_mm_experimental.npy')
        if set(sample['files_sha256']) != expected_files:
            raise ValueError(f'{sid}: unexpected file list')
        for name, sha in sample['files_sha256'].items():
            if digest(folder / name) != sha:
                raise ValueError(f'{sid}: checksum mismatch for {name}')
        actual_qa = check_sample(folder)
        if any(actual_qa[k] != sample['qa'][k] for k in actual_qa):
            raise ValueError(f'{sid}: current mask/ROI does not pass QA')
        accepted.append(candidate)
    if not accepted:
        raise ValueError('No accepted samples to export')
    if destination.exists():
        raise FileExistsError(f'{destination} already exists; choose a new output file')
    if destination.parent.resolve() != root:
        raise ValueError('Output manifest must be in the run directory so relative paths stay valid')
    audit_path = destination.with_suffix(destination.suffix + '.review.json')
    if audit_path.exists():
        raise FileExistsError(f'{audit_path} already exists; choose a new output file')
    import hashlib
    content = csv_text(fields, accepted)
    audit = {'run_fingerprint': run['fingerprint'], 'qa_report_sha256': digest(root / 'qa_report.json'),
             'review_csv_sha256': digest(review_file), 'manifest_sha256': hashlib.sha256(content.encode()).hexdigest(),
             'accepted_ids': [r['id'] for r in accepted], 'decisions': reviews}
    atomic_text(audit_path, json.dumps(audit, ensure_ascii=False, indent=2))
    try:
        atomic_text(destination, content)
    except BaseException:
        audit_path.unlink(missing_ok=True)
        raise
    print(f'{len(accepted)} approved of {len(samples)}; manifest:', destination)
    print('Review decisions:', review_file, '| SHA256:', digest(review_file), '| run:', run['fingerprint'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    action = p.add_mutually_exclusive_group(required=True)
    action.add_argument('--template', action='store_true')
    action.add_argument('--export', action='store_true')
    p.add_argument('--output-name', default='approved_manifest.csv', help='filename inside the run directory')
    a = p.parse_args()
    root = a.run.resolve()
    if Path(a.output_name).name != a.output_name or a.output_name in {'.', '..'}:
        p.error('output-name must be a plain filename')
    run, samples = load_run(root)
    if a.template: template(root, samples)
    else: export(root, run, samples, root / a.output_name)


if __name__ == '__main__': main()
