import cv2, numpy as np, csv, re, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
rows=list(csv.DictReader(open(ROOT/"out/synth/index.csv")))
ratios=[]
for r in rows:
    im=cv2.imread(str(ROOT/f"out/synth/{r['id']}.jpg")); cls=cv2.imread(str(ROOT/f"out/synth/{r['id']}_cls.png"),0)
    d=cv2.imread(str(ROOT/f"out/synth/{r['id']}_depth.png"),-1)/10.0
    g=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY).astype(float)
    sn=(cls==1)&(d>=20); mt=(cls==2)
    if sn.sum()>500 and mt.sum()>500: ratios.append(g[sn].mean()/g[mt].mean())
print("snow/metal luminance ratio, synthetic: n=%d median %.2f p10 %.2f p90 %.2f"%(len(ratios),np.median(ratios),np.percentile(ratios,10),np.percentile(ratios,90)))
pick=[r for r in rows if int(re.search(r"_(\d{3})_",r["id"]).group(1)) in (0,3,4,5,7)]
pick=[r for r in pick if "135619" not in r["background"]][:10]
tiles=[]
for r in pick:
    im=cv2.imread(str(ROOT/f"out/synth/{r['id']}.jpg")); cls=cv2.imread(str(ROOT/f"out/synth/{r['id']}_cls.png"),0)
    ys,xs=np.nonzero((cls>0)&(cls<8))
    x0,x1=max(0,xs.min()-60),min(im.shape[1],xs.max()+60); y0,y1=max(0,ys.min()-50),min(im.shape[0],ys.max()+50)
    c=im[y0:y1,x0:x1].copy(); l=cls[y0:y1,x0:x1]
    ov=c.copy(); ov[l==1]=(0.45*ov[l==1]+[0,70,230]).clip(0,255); ov[l==7]=(0.45*ov[l==7]+[230,60,0]).clip(0,255); ov[l==3]=(0.5*ov[l==3]+[200,120,0]).clip(0,255)
    pane=np.hstack([cv2.resize(c,(760,240)),cv2.resize(ov,(760,240))])
    t=f"{r['background'][:24]} | {r['scenario']} | cov>=2mm {float(r['cov_2mm']):.2f}  >=20mm {float(r['cov_20mm']):.2f}"
    pane=np.vstack([np.zeros((30,1520,3),np.uint8),pane]); cv2.putText(pane,t,(8,21),cv2.FONT_HERSHEY_SIMPLEX,0.62,(255,255,255),1,cv2.LINE_AA)
    tiles.append(pane)
cv2.imwrite(str(ROOT/"out/stage2_synth_sheet.jpg"),np.vstack(tiles),[cv2.IMWRITE_JPEG_QUALITY,84])
json.dump({"snow_metal_ratio_median":float(np.median(ratios)),"p10":float(np.percentile(ratios,10)),"p90":float(np.percentile(ratios,90)),"n":len(ratios)},open(ROOT/"out/stage2_realism.json","w"))
