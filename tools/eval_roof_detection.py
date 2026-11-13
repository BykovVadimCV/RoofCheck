#!/usr/bin/env python3
"""Evaluate pixel quality of roof detections against held-out expert masks.

CSV: predicted_roof, gt_roof, predicted_obstacle, gt_obstacle,
     [predicted_boundary,gt_boundary,group_id]. All paths relative to manifest.
Reported false_snowable measures the fraction of predicted allowed pixels
outside true allowed roof surface. This is more informative than roof IoU alone.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import cv2
from PIL import Image


def mask(path):
    with Image.open(path) as im:
        a=np.asarray(im)
    if a.ndim!=2 or not set(np.unique(a)) <= {0,1,255}:
        raise ValueError(f'Expected single channel binary mask: {path}')
    return a>0


def metrics(pred_roof,gt_roof,pred_occ,gt_occ,pred_boundary=None,gt_boundary=None,boundary_tol=3):
    a=pred_roof; b=gt_roof
    if not (a.shape==b.shape==pred_occ.shape==gt_occ.shape):
        raise ValueError('Mask dimensions differ')
    intersect=int((a&b).sum());union=int((a|b).sum())
    allowed=a&~pred_occ;truth=b&~gt_occ
    false=int((allowed&~truth).sum())
    eligible=int(allowed.sum())
    obs_tp=int((pred_occ & gt_occ & b).sum())
    obs_gt=int((gt_occ & b).sum())
    output=dict(roof_iou=intersect/union if union else 1.,
                false_snowable_rate=false/eligible if eligible else None,
                false_snowable_pixels=false,
                predicted_snowable_pixels=eligible,
                obstacle_recall=obs_tp/obs_gt if obs_gt else None)
    if pred_boundary is not None and gt_boundary is not None:
        if pred_boundary.shape!=b.shape or gt_boundary.shape!=b.shape:
            raise ValueError('Boundary dimension mismatch')
        k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*boundary_tol+1,)*2)
        near_gt=cv2.dilate(gt_boundary.astype(np.uint8),k)>0
        near_pred=cv2.dilate(pred_boundary.astype(np.uint8),k)>0
        precision=float((pred_boundary&near_gt).sum()/pred_boundary.sum()) if pred_boundary.any() else None
        recall=float((gt_boundary&near_pred).sum()/gt_boundary.sum()) if gt_boundary.any() else None
        output['boundary_f1']=(2*precision*recall/(precision+recall)
                    if precision is not None and recall is not None and precision+recall else None)
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--boundary-tolerance',type=int,default=3)
    args=p.parse_args()
    if args.boundary_tolerance < 0:p.error('boundary tolerance must be nonnegative')
    with args.manifest.open(newline='',encoding='utf-8-sig') as fp:rows=list(csv.DictReader(fp))
    if not rows:raise ValueError('No test rows')
    results=[]
    for row in rows:
        def read(col):
            path=Path(row[col]);path=path if path.is_absolute() else args.manifest.parent/path
            return mask(path)
        item=metrics(read('predicted_roof'),read('gt_roof'),read('predicted_obstacle'),read('gt_obstacle'),
            read('predicted_boundary') if row.get('predicted_boundary') else None,
            read('gt_boundary') if row.get('gt_boundary') else None,
            args.boundary_tolerance)
        item['group_id']=row.get('group_id','')
        results.append(item)
    averages={k:(float(np.mean(values)) if values else None) for k in ('roof_iou','false_snowable_rate','obstacle_recall','boundary_f1')
              if (values:=[r[k] for r in results if k in r and r[k] is not None]) is not None}
    output=dict(n=len(results),means=averages,per_image=results,
                caveat='Pixel metrics only; separate instance/facet matching and latency benchmark are also required.')
    if args.output.exists():raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2))
    print(json.dumps(output['means'],indent=2))


if __name__=='__main__':main()
