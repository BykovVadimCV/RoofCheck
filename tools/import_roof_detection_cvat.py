#!/usr/bin/env python3
"""Import manually reviewed multi-facet CVAT 1.1 polygons into RoofCheck.

Unlike legacy prepare_roof_dataset.py, this accepts multiple roof_facet
polygons per target roof and writes the *actual* roof_mask + surfaces contract.
No inference is performed here. Use this after reviewing detection proposals.
"""
import argparse
import json
import math
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET
import numpy as np
import cv2
from PIL import Image, ImageOps

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))


def _attrs(elem):
    return {a.get('name'): (a.text or '').strip() for a in elem.findall('attribute')}


def _shape(elem, size):
    raw=elem.get('points','')
    try: points=np.asarray([[float(v) for v in xy.split(',')] for xy in raw.split(';')],np.float32)
    except (TypeError,ValueError): raise ValueError(f'Invalid polygon points: {raw[:80]}')
    h,w=size
    if (points.ndim!=2 or points.shape[1]!=2 or len(points)<3 or
        not np.isfinite(points).all() or np.any(points[:,0]<0) or np.any(points[:,1]<0) or
        np.any(points[:,0]>=w) or np.any(points[:,1]>=h)):
        raise ValueError('Polygon outside oriented image or malformed')
    if abs(cv2.contourArea(points))<4:
        raise ValueError('Degenerate polygon')
    mask=np.zeros((h,w),np.uint8)
    cv2.fillPoly(mask,[np.round(points).astype(np.int32)],1)
    return mask.astype(bool), points.astype(float).tolist()


def import_review(xml, image, output, *, image_id='frame_001',group_id='building_001',
                  verified_clean=False, snow_mask=None):
    image=Path(image)
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    if verified_clean and snow_mask is not None:
        raise ValueError('verified_clean conflicts with snow_mask')
    with Image.open(image) as im:
        w,h=ImageOps.exif_transpose(im).size
    shape=(h,w)
    root=ET.parse(xml).getroot()
    items=list(root.iter('image'))
    if len(items)!=1:
        raise ValueError('Expected CVAT XML containing exactly one image')
    node=items[0]
    if (int(node.get('width','0')),int(node.get('height','0'))) != (w,h):
        raise ValueError('CVAT dimensions do not match EXIF-oriented image')
    polys=[]
    for elem in node.findall('polygon'):
        role=elem.get('label')
        if role not in {'roof','roof_facet','slope','structure','exclude'}:
            continue
        mask, points=_shape(elem,shape)
        polys.append(dict(role=role,mask=mask,points=points,attrs=_attrs(elem)))
    roofs=[x for x in polys if x['role']=='roof']
    selected=[x for x in roofs if x['attrs'].get('target')=='yes']
    if not selected and len(roofs)==1:selected=roofs
    if len(selected)!=1:
        raise ValueError('Review must mark exactly one target roof with attribute target=yes')
    roof=selected[0]['mask']
    roof_id=selected[0]['attrs'].get('roof_id') or 'roof_001'
    facets=[x for x in polys if x['role'] in {'roof_facet','slope'} and
            (not x['attrs'].get('roof_id') or x['attrs'].get('roof_id')==roof_id)]
    if not facets:
        raise ValueError('Target roof requires at least one reviewed roof_facet (or legacy slope)')
    for fac in facets:
        if fac['role']=='roof_facet' and fac['attrs'].get('reviewed')!='yes':
            raise ValueError('Every automatic roof_facet must have reviewed=yes before import')
    for region in polys:
        if region['role']=='structure' and region['attrs'].get('reviewed')=='no':
            raise ValueError('Automatic structure polygons must have reviewed=yes before import')
    labels=np.zeros(shape,np.uint16)
    surfaces=[dict(surface_id='roof_unassigned_base',mode='image',z_order=-100)]
    for i,fac in enumerate(facets,1):
        region=fac['mask'] & roof
        outside=np.count_nonzero(fac['mask'] & ~roof)
        if outside > max(4,int(.01*fac['mask'].sum())):
            raise ValueError('Facet extends significantly outside selected target roof')
        if np.count_nonzero(region & (labels!=0)) > max(4,int(.01*region.sum())):
            raise ValueError('Reviewed facet polygons overlap too much')
        labels[region]=i
        surfaces.append(dict(surface_id=fac['attrs'].get('facet_id') or f'{roof_id}_f{i:02}',
                             mode='approximate',polygon=fac['points'],grid_size=[256,256],z_order=0))
    uncovered=roof & (labels==0)
    if np.count_nonzero(uncovered)>max(4,int(.002*roof.sum())):
        raise ValueError(f'Facets leave {int(uncovered.sum())} roof pixels uncovered; correct CVAT polygons')
    # Cover small rasterization gaps with the first facet in the generator;
    # sample geometry treats this as approximate support rather than a real plane.
    if uncovered.any():
        # We must not silently fill gaps: store explicitly as image-mode region.
        outlines,_=cv2.findContours(uncovered.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for k,contour in enumerate(outlines):
            if len(contour)>=3 and cv2.contourArea(contour)>=1:
                surfaces.append(dict(surface_id=f'{roof_id}_gap{k:03}',mode='image',
                                     polygon=contour[:,0,:].astype(float).tolist()))

    obstacles=np.zeros(shape,bool)
    for p in polys:
        if p['role'] in {'structure','exclude'}:
            obstacles |= p['mask']
    obstacles &= roof
    boundary=np.zeros(shape,bool)
    for i in range(1,len(facets)+1):
        region=(labels==i).astype(np.uint8)
        outline=cv2.morphologyEx(region,cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))>0
        # Ignore exterior edge of roof; only shared interior facet boundaries.
        other=(labels!=0)&(labels!=i)
        adjacent=cv2.dilate(other.astype(np.uint8),np.ones((3,3),np.uint8))>0
        boundary |= outline & adjacent & roof
    # Training is for ALL visible roofs, not merely the customer's target.
    # Otherwise neighbouring real rooftops would become false negatives.
    all_roofs=np.zeros(shape,bool)
    for item in roofs:
        all_roofs |= item['mask']
    all_boundary=boundary.copy()
    for item in [x for x in polys if x['role'] in {'roof_facet','slope'}
                 and (not x['attrs'].get('roof_id') or x['attrs'].get('roof_id') != roof_id)]:
        inner=item['mask'] & all_roofs
        all_boundary |= cv2.morphologyEx(inner.astype(np.uint8), cv2.MORPH_GRADIENT,
                                        np.ones((3,3),np.uint8)).astype(bool) & all_roofs
    all_obstacles=np.zeros(shape,bool)
    for p in polys:
        if p['role'] in {'structure','exclude'}:
            all_obstacles |= p['mask']
    all_obstacles &= all_roofs
    sem=np.zeros(shape,np.uint8)
    sem[all_roofs]=1
    sem[all_boundary]=2
    sem[all_obstacles]=3
    # Write paths relative to output, matching the generator's input contract.
    output.mkdir(parents=True)
    source_name='source'+image.suffix.lower()
    shutil.copy2(image, output/source_name)
    Image.fromarray(roof.astype(np.uint8)*255).save(output/'roof_mask.png')
    Image.fromarray(obstacles.astype(np.uint8)*255).save(output/'occluder_mask.png')
    Image.fromarray(roof.astype(np.uint8)*255).save(output/'inspection_roi.png')
    Image.fromarray(sem).save(output/'semantic_training_mask.png')
    if len(roofs)>65535 or len([x for x in polys if x['role'] in {'roof_facet','slope'}])>65535:
        raise ValueError('Too many instances for uint16 mask')
    roof_instance_ids=np.zeros(shape,np.uint16)
    for rid,reg in enumerate(roofs,1):
        roof_instance_ids[reg['mask']]=rid
    facet_instance_ids=np.zeros(shape,np.uint16)
    for fid,reg in enumerate([x for x in polys if x['role'] in {'roof_facet','slope'}],1):
        facet_instance_ids[reg['mask'] & all_roofs]=fid
    Image.fromarray(roof_instance_ids).save(output/'roof_instance_ids.png')
    Image.fromarray(facet_instance_ids).save(output/'facet_instance_ids.png')
    sample=dict(image_id=image_id,image_path=source_name,group_id=group_id,
                split='train',roof_mask='roof_mask.png',inspection_roi='inspection_roi.png',
                occluder_mask='occluder_mask.png',annotation_mode=('verified_clean' if verified_clean else 'partial'),
                mask_coordinates='oriented',surfaces=surfaces,source='manually_reviewed_roof_geometry')
    if snow_mask is not None:
        with Image.open(snow_mask) as im:
            label=np.asarray(im)
        if label.shape != shape or label.ndim!=2 or not set(np.unique(label)) <= {0,1,255}:
            raise ValueError('Snow mask requires oriented 0/1/255 pixels')
        shutil.copy2(snow_mask,output/'snow_mask.png')
        sample['snow_mask']='snow_mask.png'
    (output/'sample.json').write_text(json.dumps({'sample':sample,'generation':{'scenario':'partial_cover','target_coverage':.45}},ensure_ascii=False,indent=2),encoding='utf-8')
    (output/'review.json').write_text(json.dumps({'reviewed':True,'roof_id':roof_id,
        'facets':len(facets),'roof_pixels':int(roof.sum()),'obstacle_pixels':int(obstacles.sum()),
        'boundary_pixels':int(boundary.sum()),'uncovered_pixels':int(uncovered.sum()),
        'warning':'Facet quadrilaterals remain perspective approximations, not metric 3D.'},indent=2),encoding='utf-8')
    return output/'sample.json'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--xml',type=Path,required=True)
    p.add_argument('--image',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--image-id',default='frame_001')
    p.add_argument('--group-id',default='building_001')
    p.add_argument('--verified-clean',action='store_true')
    p.add_argument('--snow-mask',type=Path)
    a=p.parse_args()
    print(import_review(a.xml,a.image,a.output,image_id=a.image_id,
          group_id=a.group_id,verified_clean=a.verified_clean,snow_mask=a.snow_mask))


if __name__=='__main__': main()
