#!/usr/bin/env python3
"""Evaluate roof instance and facet instance maps from machine vs expert.

CSV: predicted_roof_ids, gt_roof_ids, predicted_facet_ids, gt_facet_ids,
optional group_id. Single-channel 16-bit or 8-bit PNG, background ID 0.
Uses optimal one-to-one IoU matching, not raster semantic IoU.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment


def load_labels(path):
    with Image.open(path) as im:
        ids=np.asarray(im)
    if ids.ndim!=2 or not np.issubdtype(ids.dtype,np.integer):
        raise ValueError(f'Expected grayscale integer instance PNG: {path}')
    if np.any(ids<0):raise ValueError('Negative instance ID')
    return ids


def match_instance_maps(predicted, truth, threshold=.5):
    """Hungarian assignment by maximizing mask IoU, then threshold matches."""
    if predicted.shape!=truth.shape or predicted.ndim!=2:
        raise ValueError('Instance maps must be equally sized 2D arrays')
    if not 0<threshold<=1:
        raise ValueError('IoU match threshold must be (0,1]')
    pi=np.unique(predicted[predicted!=0]);ti=np.unique(truth[truth!=0])
    n,m=len(pi),len(ti)
    if n==0 or m==0:
        return dict(pred_instances=n,gt_instances=m,matched=0,
            recall=0.0 if m else 1.0,precision=0.0 if n else 1.0,
            mean_matched_iou=None,match_threshold=threshold)
    # Encode label IDs into contiguous indices without max(label) memory blowup.
    pv=np.searchsorted(pi,predicted.ravel())
    tv=np.searchsorted(ti,truth.ravel())
    valid=(predicted.ravel()!=0)&(truth.ravel()!=0)
    pairs=np.bincount(pv[valid]*m+tv[valid],minlength=n*m).reshape(n,m)
    pc=np.array([np.count_nonzero(predicted==x) for x in pi],np.int64)
    tc=np.array([np.count_nonzero(truth==x) for x in ti],np.int64)
    iou=pairs/(pc[:,None]+tc[None,:]-pairs+1e-12)
    # maximize IoU globally, not greedy by largest matched region.
    prow,tcol=linear_sum_assignment(1-iou)
    keep=iou[prow,tcol]>=threshold
    matched=iou[prow[keep],tcol[keep]]
    return dict(pred_instances=n,gt_instances=m,matched=int(keep.sum()),
        recall=float(keep.sum()/m),precision=float(keep.sum()/n),
        mean_matched_iou=float(matched.mean()) if len(matched) else None,
        match_threshold=threshold)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--iou-threshold',type=float,default=.5)
    a=p.parse_args()
    with a.manifest.open(newline='',encoding='utf-8-sig') as fp:
        rows=list(csv.DictReader(fp))
    required=['predicted_roof_ids','gt_roof_ids','predicted_facet_ids','gt_facet_ids']
    if not rows:raise ValueError('No evaluation records')
    results=[]
    for row in rows:
        def read(key):
            path=Path(row[key]);path=path if path.is_absolute() else a.manifest.parent/path
            return load_labels(path)
        roof=match_instance_maps(read(required[0]),read(required[1]),a.iou_threshold)
        facet=match_instance_maps(read(required[2]),read(required[3]),a.iou_threshold)
        results.append(dict(group_id=row.get('group_id',''),roof=roof,facet=facet))
    summary={}
    for kind in ('roof','facet'):
        gt=sum(r[kind]['gt_instances'] for r in results)
        pd=sum(r[kind]['pred_instances'] for r in results)
        found=sum(r[kind]['matched'] for r in results)
        ious=[r[kind]['mean_matched_iou'] for r in results if r[kind]['mean_matched_iou'] is not None]
        summary[kind]=dict(gt_instances=gt,pred_instances=pd,matched=found,
                           micro_recall=found/gt if gt else None,
                           micro_precision=found/pd if pd else None,
                           mean_image_matched_iou=float(np.mean(ious)) if ious else None)
    report=dict(n_images=len(results),threshold=a.iou_threshold,summary=summary,per_image=results,
        warning='Instance IDs are arbitrary per image; matching is spatial, not by equal ID. Review unlabeled/unknown GT regions separately.')
    if a.output.exists():raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
