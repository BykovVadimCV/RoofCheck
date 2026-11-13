"""Offline diagnostic scene, not a real-photo realism benchmark."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from PIL import Image
from roofcheck.roof import RoofSample, GenerationConfig, generate, save_result
from roofcheck.roof.scenarios import SCENARIOS


def main():
    root=Path('out/roof_demo');root.mkdir(parents=True,exist_ok=True)
    yy,xx=np.mgrid[:240,:360]
    rgb=np.empty((240,360,3),np.uint8)
    rgb[:]=[145,158,165]
    roof=np.zeros((240,360),np.uint8)
    quad=np.array([[65,30],[275,30],[345,215],[15,215]],np.int32)
    cv2.fillPoly(roof,[quad],1)
    metal=np.clip(115+15*np.cos(xx/10)+yy*.15,0,255).astype(np.uint8)
    rgb[roof>0]=np.stack([metal,metal,metal+8],-1)[roof>0]
    occ=np.zeros_like(roof)
    cv2.line(occ,(20,60),(340,155),1,3)
    occ[75:130,175:195]=1
    rgb[occ>0]=[45,45,48]
    snow=np.full_like(roof,255);snow[(roof>0)&(occ==0)]=0
    snow[140:165,70:110]=1;rgb[140:165,70:110]=[225,233,239]
    # Unknown strip tests that uncertainty is not converted to background.
    snow[175:185,120:155]=255
    for name,a in [('photo',rgb),('roof',roof),('snow',snow),('occluders',occ)]:Image.fromarray(a).save(root/f'{name}.png')
    spec=RoofSample('diagnostic',root/'photo.png','diagnostic_building',root/'roof.png',
                    snow_mask=root/'snow.png',occluder_mask=root/'occluders.png',
                    surfaces=[dict(surface_id='slope',mode='approximate',quad=quad.astype(float).tolist(),
                                   seam_u=[.1,.25,.4,.55,.7,.85])])
    rows=[]
    for i,name in enumerate(SCENARIOS):
        result=generate(spec,GenerationConfig(scenario=name),42)
        save_result(spec,result,root/name)
        overlay=result.image.copy()
        overlay[result.mask==1]=(overlay[result.mask==1]*.4+np.array([0,230,255])*.6).astype(np.uint8)
        overlay[result.mask==255]=(overlay[result.mask==255]*.6+np.array([220,0,180])*.4).astype(np.uint8)
        row=np.concatenate([rgb,result.image,overlay],axis=1)
        row=cv2.copyMakeBorder(row,30,0,0,0,cv2.BORDER_CONSTANT,value=[255,255,255])
        cv2.putText(row,name,(10,21),cv2.FONT_HERSHEY_SIMPLEX,.6,[20,20,20],1,cv2.LINE_AA)
        rows.append(row)
    Image.fromarray(np.concatenate(rows,axis=0)).save(root/'contact_sheet.jpg')
    print(root/'contact_sheet.jpg')


if __name__=='__main__':main()
