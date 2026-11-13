"""Export CVAT proposals and exact EXIF-oriented masks to RoofCheck's existing contract."""
from __future__ import annotations
from pathlib import Path
import json
import re
import shutil
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from PIL import Image
from .types import Detection
from .masks import _polygons


def _write_mask(path, mask):
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)


def _draw_polygons(xmlimage, label, polys, attrs=None):
    for points in polys:
        if len(points) < 3:
            continue
        el=ET.SubElement(xmlimage, 'polygon', label=label, occluded='0',
                         source='auto', z_order='0',
                         points=';'.join(f'{x:.2f},{y:.2f}' for x,y in points))
        for name,value in (attrs or {}).items():
            ET.SubElement(el, 'attribute', name=name).text=str(value)


def _label_spec():
    # A new CVAT project needs this set; the legacy CVAT schema is untouched.
    return [dict(name=name, color=color, type='polygon', attributes=[
        dict(name=k, input_type='text', mutable=True, values=['']) for k in attrs])
        for name,color,attrs in (
            ('roof', '#e2603a', ('roof_id','target')),
            ('roof_facet', '#f2b134', ('facet_id','roof_id','reviewed')),
            ('structure', '#8c6d4f', ('type','reviewed')),
            ('exclude', '#5b5f7a', ('reviewed',)),
        )]


def _safe_id(value):
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]{0,79}',value):
        raise ValueError('ID must contain only alphanumeric, _, ., - and start alphanumeric')
    return value


def write_detection(image, image_path, det: Detection, output,
                    *, image_id='frame_001', group_id='building_001',
                    verified_clean=False, snow_mask=None, scenario='partial_cover',
                    target_coverage=.45):
    """Write candidates even if target is ambiguous. Only a selected target yields sample.json."""
    image_id, group_id = _safe_id(image_id), _safe_id(group_id)
    if verified_clean and snow_mask is not None:
        raise ValueError('Use either verified_clean or existing snow_mask, not both')
    if not 0 <= target_coverage <= 1:
        raise ValueError('target_coverage outside [0,1]')
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Output exists, refusing to overwrite: {output}')
    output.mkdir(parents=True)
    height, width=image.shape[:2]
    if det.shape != (height, width):
        raise ValueError('Detection and original oriented photo dimensions differ')
    selected = set(det.selection.roof_ids)
    selected_roofs = [r for r in det.roofs if r.roof_id in selected]
    roof_mask = np.zeros((height,width),bool)
    for roof in selected_roofs:
        roof_mask |= roof.mask
    obstacles=det.obstacle_mask & roof_mask
    _write_mask(output/'roof_mask.png',roof_mask)
    _write_mask(output/'occluder_mask.png',obstacles)
    _write_mask(output/'inspection_roi.png',roof_mask)
    _write_mask(output/'snowable_mask.png',roof_mask & ~obstacles)
    # Instance IDs are diagnostics/training evaluation, not generator masks.
    # Includes all found roofs even when only one is selected for snow.
    roof_instance_ids=np.zeros((height,width),np.uint16)
    facet_instance_ids=np.zeros((height,width),np.uint16)
    fid=0
    for rid,roof in enumerate(det.roofs,1):
        if rid>65535:raise ValueError('Too many roof instances for uint16 PNG')
        roof_instance_ids[roof.mask]=rid
        first_fac=None
        for fac in roof.facets:
            fid+=1
            if fid>65535:raise ValueError('Too many facets for uint16 PNG')
            first_fac=first_fac or fid
            raster=np.zeros((height,width),np.uint8)
            cv2.fillPoly(raster,[np.round(np.asarray(fac.polygon)).astype(np.int32)],1)
            facet_instance_ids[roof.mask & (raster>0)]=fid
        if first_fac is not None:
            # Keep a roof-contained fallback for disconnected or thin islands.
            facet_instance_ids[roof.mask & (facet_instance_ids==0)]=first_fac
    Image.fromarray(roof_instance_ids).save(output/'roof_instance_ids.png')
    Image.fromarray(facet_instance_ids).save(output/'facet_instance_ids.png')

    # Candidate preview: ALL roofs, not just the selected one.
    overlay=image.copy()
    colors=[(250,145,60), (40,190,240), (130,230,95), (230,65,200), (160,110,255)]
    for i,roof in enumerate(det.roofs):
        color=np.array(colors[i%len(colors)],np.float32)
        src=overlay[roof.mask].astype(np.float32)
        overlay[roof.mask]=np.clip(src*.60+color*.40,0,255).astype(np.uint8)
        for points in _polygons(roof.mask):
            cv2.polylines(overlay,[np.asarray(points,np.int32)],True,(255,255,255),2)
        x,y,_,_=roof.bbox
        cv2.putText(overlay,roof.roof_id+(' TARGET' if roof.roof_id in selected else ''),
                    (x,max(y-7,16)),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)
    overlay[obstacles] = (255, 65, 65)
    Image.fromarray(overlay).save(output/'overlay.jpg',quality=92)

    image_name=Path(image_path).name
    cvat=ET.Element('annotations')
    ET.SubElement(cvat,'version').text='1.1'
    xmlim=ET.SubElement(cvat,'image',id='0',name=image_name,width=str(width),height=str(height))
    roofs_data=[]
    for roof in det.roofs:
        _draw_polygons(xmlim,'roof',_polygons(roof.mask),
                       {'roof_id':roof.roof_id,'target':('yes' if roof.roof_id in selected else 'no')})
        facets=[]
        for fac in roof.facets:
            _draw_polygons(xmlim,'roof_facet',[fac.polygon],
                           {'facet_id':fac.facet_id, 'roof_id':roof.roof_id,'reviewed':'no'})
            facets.append(dict(facet_id=fac.facet_id,facet_type=fac.facet_type,
                               pixels=fac.pixels,confidence=fac.confidence,
                               polygon=fac.polygon,geometry={'mode':fac.mode,'metric_claim':False}))
        roofs_data.append(dict(roof_id=roof.roof_id,pixels=roof.pixels,
                               bbox_xyxy=roof.bbox,confidence=roof.confidence,
                               polygons=_polygons(roof.mask),facets=facets))
    _draw_polygons(xmlim,'structure',_polygons(obstacles),{'type':'unknown','reviewed':'no'})
    ET.ElementTree(cvat).write(output/'cvat_proposals.xml',encoding='utf-8',xml_declaration=True)
    (output/'cvat_labels.json').write_text(json.dumps(_label_spec(),indent=2),encoding='utf-8')
    info=dict(source=str(Path(image_path).resolve()), size=[width,height],
              status=det.selection.status, selection_reason=det.selection.reason,
              selected_roof_ids=list(det.selection.roof_ids), roofs=roofs_data,
              model_info=det.model_info,diagnostics=det.diagnostics,
              warning='Proposals are unreviewed; do not assume snowable pixels are correct.')
    (output/'detection.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')

    if det.selection.status != 'selected':
        return None
    if not selected_roofs:
        raise ValueError('Selection returned no roofs')
    # Existing RoofSample.surface_fields requires full coverage of roof_mask.
    # Geometric facets are polygon approximations, not exact raster coverage.
    # Image-mode base ensures any uncovered pixels remain inside the roof mask.
    # Base image-space surface guarantees full raster roof coverage even where
    # a contour contains holes or a very thin raster fragment. Later, known
    # facet polygons override it with their own canonical texture coordinates.
    surfaces=[dict(surface_id='roof_unassigned_base', mode='image', z_order=-100)]
    for roof in selected_roofs:
        for fac in roof.facets:
            surfaces.append(dict(fac.to_surface(), z_order=0))
    # Copy original bytes to preserve EXIF, then store oriented masks beside it.
    copied='source'+Path(image_path).suffix.lower()
    shutil.copy2(image_path,output/copied)
    sample=dict(image_id=image_id,image_path=copied,group_id=group_id,
                split='train',roof_mask='roof_mask.png',
                occluder_mask='occluder_mask.png',inspection_roi='inspection_roi.png',
                annotation_mode=('verified_clean' if verified_clean else 'partial'),
                mask_coordinates='oriented', source='roof_detection_unreviewed',
                surfaces=surfaces)
    if snow_mask is not None:
        with Image.open(snow_mask) as f:
            arr=np.asarray(f)
            if arr.shape!=(height,width) or arr.ndim!=2 or not set(np.unique(arr)) <= {0,1,255}:
                raise ValueError('snow_mask must be EXIF-oriented uint8, pixel labels 0/1/255')
        shutil.copy2(snow_mask,output/'existing_snow.png')
        sample['snow_mask']='existing_snow.png'
    payload={'sample':sample,'generation':{'scenario':scenario,'target_coverage':target_coverage}}
    (output/'sample.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    return output/'sample.json'
