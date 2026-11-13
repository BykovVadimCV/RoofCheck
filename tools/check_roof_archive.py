"""Run annotated real-photo probes; annotations remain private and explicitly provisional.

python tools/check_roof_archive.py --inventory out/archive_probe/inventory.json \
    --annotations out/archive_probe/annotations.json --output out/archive_roof_trial_v01
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import csv
import hashlib
import json
from dataclasses import asdict
import numpy as np
import cv2
from PIL import Image,ImageOps,ImageDraw
from roofcheck.roof import RoofSample,GenerationConfig,generate,save_result,load_sample
from roofcheck.roof.masks import save_mask
from roofcheck.roof.scenarios import SCENARIOS


def polygon_mask(shape,polygons):
    mask=np.zeros(shape,np.uint8)
    for polygon in polygons:
        cv2.fillPoly(mask,[np.round(polygon).astype(np.int32)],1)
    return mask


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory',type=Path,required=True)
    p.add_argument('--annotations',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    inventory={r['id']:r for r in json.loads(a.inventory.read_text())}
    annotations=json.loads(a.annotations.read_text())
    if annotations['status']!='assistant_draft_requires_expert_review':
        raise ValueError('Review status must be explicit')
    (a.output/'annotations.json').write_text(json.dumps(annotations,ensure_ascii=False,indent=2))
    (a.output/'inventory.json').write_text(a.inventory.read_text())
    results=[];panels=[];rows=[]
    for case in annotations['cases']:
        item=inventory[case['id']];raw=Path(item['path'])
        if hashlib.sha256(raw.read_bytes()).hexdigest()!=item['sha256']:raise ValueError('Source changed')
        image=np.array(ImageOps.exif_transpose(Image.open(raw)).convert('RGB'));shape=image.shape[:2]
        folder=a.output/'inputs'/case['id'];folder.mkdir(parents=True)
        Image.fromarray(image).save(folder/'image.png')
        polygons=case.get('roof_polygons',[])
        occ_polygons=case.get('occluder_polygons',[])
        if case.get('reference'):
            reference=json.loads(Path(case['reference']).read_text())
            if [reference['width'],reference['height']]!=list(image.shape[1::-1]):raise ValueError('Reference size mismatch')
            polygons=[p['points'] for p in reference['polygons'] if p['role']=='roof']
            occ_polygons += [p['points'] for p in reference['polygons'] if p['role']=='structure']
        # Small erosion makes the manual boundary conservative. This is an annotated subset, not a full-roof gold mask.
        roof=cv2.erode(polygon_mask(shape,polygons),np.ones((5,5),np.uint8))
        occluders=polygon_mask(shape,occ_polygons)
        for line in case.get('occluder_lines',[]):
            cv2.polylines(occluders,[np.int32(line['points'])],False,1,line['width'])
        occluders=cv2.dilate(occluders,np.ones((3,3),np.uint8))
        mask=np.full(shape,255,np.uint8)
        visible=(roof>0)&(occluders==0)
        if case['mode']=='clean_draft':mask[visible]=0
        else:
            snow=polygon_mask(shape,case['snow_polygons'])>0
            mask[visible&snow]=1
        for name,array in [('roof',roof),('occluders',occluders)]:Image.fromarray(array).save(folder/f'{name}.png')
        save_mask(folder/'snow.png',mask)
        spec=RoofSample(case['id'],folder/'image.png',case['group_id'],folder/'roof.png',
            snow_mask=folder/'snow.png',occluder_mask=folder/'occluders.png',annotation_mode='partial',
            source='customer Archive; assistant draft masks, NOT expert gold',
            surfaces=[dict(surface_id=f'{case["id"]}_slope_{i}',mode='approximate',polygon=polygon)
                      for i,polygon in enumerate(polygons)])
        loaded=load_sample(spec)
        plan=[('zero_addition',GenerationConfig(target_coverage=0))]
        if case['mode']=='clean_draft':plan += [(name,GenerationConfig(scenario=name)) for name in SCENARIOS if name!='panel_residue']
        examples={}
        for j,(name,cfg) in enumerate(plan):
            result=generate(spec,cfg,seed=42+j)
            output=a.output/'generated'/case['id']/name
            save_result(spec,result,output)
            for region in (loaded.occluders,~loaded.roof,loaded.snow==1):
                if not np.array_equal(result.image[region],loaded.image[region]):raise AssertionError('Protected RGB changed')
            if not np.array_equal(result.mask[~loaded.allowed],loaded.snow[~loaded.allowed]):raise AssertionError('Protected labels changed')
            results.append(dict(id=case['id']+'_'+name,source_id=case['id'],group_id=case['group_id'],scenario=name,
                annotation_status=annotations['status'],raw_sha256=item['sha256'],
                coverage=result.metadata['coverage'],checks=dict(protected_rgb=True,protected_labels=True,
                valid_classes=set(np.unique(result.mask))<={0,1,255})))
            rows.append(dict(id=case['id']+'_'+name,image=str((output/'image.png').relative_to(a.output)),
                mask=str((output/'mask.png').relative_to(a.output)),group_id=case['group_id'],split='train',kind='synthetic',roi='',
                annotation_status=annotations['status']))
            examples[name]=result
        # Include real sources in the probe manifest for parent-group provenance. No fake val/test split from four photos.
        rows.append(dict(id=case['id']+'_real',image=str((folder/'image.png').relative_to(a.output)),
            mask=str((folder/'snow.png').relative_to(a.output)),group_id=case['group_id'],split='train',kind='real',roi='',
            annotation_status=annotations['status']))
        ys,xs=np.nonzero(roof);x0=max(0,int(xs.min())-20);x1=min(shape[1],int(xs.max())+21)
        y0=max(0,int(ys.min())-20);y1=min(shape[0],int(ys.max())+21)
        box=(x0,y0,x1,y1)
        def thumb(rgb):
            im=Image.fromarray(rgb).crop(box)
            return ImageOps.pad(im,(420,270),method=Image.Resampling.LANCZOS,color='white')
        for name in ['partial_cover','near_full_cover'] if case['mode']=='clean_draft' else ['zero_addition']:
            result=examples[name];overlay=result.image.copy()
            overlay[result.mask==1]=(overlay[result.mask==1]*.5+np.array([0,210,255])*.5).astype(np.uint8)
            overlay[result.mask==255]=(overlay[result.mask==255]*.65+np.array([230,0,200])*.35).astype(np.uint8)
            panel=Image.new('RGB',(1260,305),'white');d=ImageDraw.Draw(panel)
            for i,rgb in enumerate((image,result.image,overlay)):panel.paste(thumb(rgb),(i*420,35))
            d.text((8,8),case['id']+' '+name+' | source / synthetic / labels; cyan=snow magenta=unknown',fill='black')
            panels.append(panel)
        # Per-case detailed sheet: all scenarios at useful roof crop size.
        sheet=Image.new('RGB',(1260,305*((len(examples)+2)//3)),'white');d=ImageDraw.Draw(sheet)
        for i,(name,result) in enumerate(examples.items()):
            x=(i%3)*420;y=(i//3)*305
            sheet.paste(thumb(result.image),(x,y+35));d.text((x+5,y+8),name,fill='black')
        sheet.save(a.output/(case['id']+'_scenarios.jpg'))
        roi_overlay=image.copy()
        roi_overlay[visible]=(roi_overlay[visible]*.6+np.array([60,220,60])*.4).astype(np.uint8)
        roi_overlay[(occluders>0)&(roof>0)]=(roi_overlay[(occluders>0)&(roof>0)]*.4+np.array([255,120,0])*.6).astype(np.uint8)
        Image.fromarray(roi_overlay).save(a.output/(case['id']+'_annotation.jpg'))
    sheet=Image.new('RGB',(1260,305*len(panels)),'white')
    for i,panel in enumerate(panels):sheet.paste(panel,(0,i*305))
    sheet.save(a.output/'comparison.jpg')
    with (a.output/'manifest_DRAFT.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    report=dict(inventory_images=len(inventory),selected_images=len(annotations['cases']),generated_images=len(results),
                annotation_status=annotations['status'],results=results,
                limitations=['Draft masks and provisional object grouping','No independent val/test created',
                'Not a final training dataset','Wide conservative exclusions around fences and structures'])
    (a.output/'checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({k:report[k] for k in ('inventory_images','selected_images','generated_images','annotation_status')},indent=2))
    print(a.output/'comparison.jpg')


if __name__=='__main__':main()
