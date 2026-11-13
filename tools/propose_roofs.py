#!/usr/bin/env python3
"""Propose building polygons from aerial RGB images for human roof annotation review.

A building mask is not a slope mask. Quad proposals are rough bounding rectangles;
no snow-free or grouping decision is made here.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageDraw
import torch
import segmentation_models_pytorch as smp


def load_model(path, device):
    model = smp.Unet('resnet34', encoder_weights=None, in_channels=3, classes=1)
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        checkpoint = checkpoint['model']
    elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        checkpoint = checkpoint['model_state_dict']
    model.load_state_dict(checkpoint, strict=True)
    return model.to(device).eval()


def predict(rgb, model, device, tile=512, stride=384):
    h, w = rgb.shape[:2]
    pad_h, pad_w = max(0, tile-h), max(0, tile-w)
    arr = cv2.copyMakeBorder(rgb, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
    h2,w2=arr.shape[:2]
    ys=list(range(0,max(1,h2-tile+1),stride)); xs=list(range(0,max(1,w2-tile+1),stride))
    if ys[-1]!=h2-tile: ys.append(h2-tile)
    if xs[-1]!=w2-tile: xs.append(w2-tile)
    probs=np.zeros((h2,w2),np.float32); counts=np.zeros_like(probs)
    mean=np.array([0.485,0.456,0.406],np.float32); std=np.array([0.229,0.224,0.225],np.float32)
    window=np.maximum(np.outer(np.hanning(tile),np.hanning(tile)).astype(np.float32),0.01)
    with torch.inference_mode():
        for y in ys:
            for x in xs:
                chunk=(arr[y:y+tile,x:x+tile].astype(np.float32)/255-mean)/std
                ten=torch.from_numpy(chunk.transpose(2,0,1).copy()).unsqueeze(0).to(device)
                prediction=torch.sigmoid(model(ten))[0,0].cpu().numpy()
                probs[y:y+tile,x:x+tile]+=prediction*window;counts[y:y+tile,x:x+tile]+=window
    return (probs/counts)[:h,:w]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory',required=True,type=Path)
    p.add_argument('--weights',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--limit',type=int,default=10,help='number of images; use 0 for all')
    p.add_argument('--max-side',type=int,default=1200)
    p.add_argument('--threshold',type=float,default=.5)
    p.add_argument('--min-area',type=int,default=1500)
    p.add_argument('--device',default='cpu')
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if args.limit<0 or args.max_side<512 or not 0<args.threshold<1 or args.min_area<1: p.error('invalid limit, max-side, threshold, or min-area')
    with args.inventory.open(newline='',encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    if args.limit:rows=rows[:args.limit]
    model=load_model(args.weights,args.device)
    args.output.mkdir(parents=True);(args.output/'previews').mkdir()
    xml=ET.Element('annotations');ET.SubElement(xml,'version').text='1.1'
    stats=[]
    for i,row in enumerate(rows):
        with Image.open(row['image']) as f: im=ImageOps.exif_transpose(f).convert('RGB')
        original=im.size;im.thumbnail((args.max_side,args.max_side),Image.Resampling.LANCZOS);rgb=np.array(im)
        prob=predict(rgb,model,args.device)
        mask=(prob>=args.threshold).astype('uint8')
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        overlay=rgb.copy();overlay[mask>0]=(0.55*overlay[mask>0]+0.45*np.array([0,220,255])).astype('uint8')
        xmlimage=ET.SubElement(xml,'image',id=str(i),name=row['cvat_image'],width=str(original[0]),height=str(original[1]))
        found=0
        for contour in contours:
            area=cv2.contourArea(contour)
            if area<args.min_area:continue
            epsilon=.008*cv2.arcLength(contour,True)
            polygon=cv2.approxPolyDP(contour,epsilon,True).reshape(-1,2)
            if len(polygon)<3:continue
            sx=original[0]/rgb.shape[1];sy=original[1]/rgb.shape[0]
            points=';'.join(f'{x*sx:.2f},{y*sy:.2f}' for x,y in polygon)
            ET.SubElement(xmlimage,'polygon',label='roof',points=points,occluded='0',source='auto',z_order='0')
            cv2.polylines(overlay,[polygon.astype('int32')],True,(255,120,0),2)
            found+=1
        Image.fromarray(overlay).save(args.output/'previews'/f"{row['id']}.jpg",quality=88)
        Image.fromarray(mask*255).save(args.output/'previews'/f"{row['id']}_mask.png")
        stats.append({'id':row['id'],'image':row['image'],'building_polygons':found,'mask_fraction':round(float(mask.mean()),5)})
        print(f"{i+1}/{len(rows)}: {row['id']} proposed {found} building polygons",flush=True)
    ET.ElementTree(xml).write(args.output/'proposals.xml',encoding='utf-8',xml_declaration=True)
    (args.output/'report.json').write_text(json.dumps({'model':'arthdubey/aerial-building-unet','weights_sha256':hashlib.sha256(args.weights.read_bytes()).hexdigest(),'max_side':args.max_side,'threshold':args.threshold,'min_area':args.min_area,'note':'UNREVIEWED building proposals; NOT roof slope annotations, snow masks, or train-ready references','images':stats},indent=2))
    print('Results:',args.output)

if __name__=='__main__':main()
