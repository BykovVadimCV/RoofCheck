"""Non-destructive comparison probe for the existing renderer; outputs are diagnostic only."""
from pathlib import Path
import sys,json,hashlib,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from PIL import Image,ImageOps
from roofcheck import synth,hostmatch,overlays
from roofcheck.schema import Reference


def main():
    inv=json.loads(Path('out/archive_probe/inventory.json').read_text())
    item=next(r for r in inv if r['id']=='archive_26')
    image=np.asarray(ImageOps.exif_transpose(Image.open(item['path'])).convert('RGB'))
    if hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()!=item['sha256']:raise ValueError('Source changed')
    factor=.5
    img=cv2.resize(image[:,:,::-1].copy(),None,fx=factor,fy=factor,interpolation=cv2.INTER_AREA)
    ref=Reference.load(Path('data/objects/demo_roof_01/vantages/v1/reference.json'))
    def polygons(role):return [np.asarray(p.points,np.float32)*factor for p in ref.by_role(role)]
    q=polygons('slope')[0];roof=polygons('roof')[0];structures=polygons('structure')
    bbox=(q[:,0].min(),q[:,1].min(),q[:,0].max(),q[:,1].max())
    host=hostmatch.estimate(img,bbox,ignore=overlays.detect(img).mask)
    geom=synth.roof_geometry(ref.object_id,ref.geometry or None)
    output=Path('out/existing_renderer_probe');output.mkdir(parents=True,exist_ok=False)
    Image.fromarray(img[:,:,::-1]).save(output/'source.png')
    rows=[]
    for index,state in enumerate(['partially_cleared','cleared','not_cleared']):
        t=time.perf_counter()
        sample=synth.make_sample(img,q,roof,np.random.default_rng(100+index),geom,host=host,
            other_roofs=[],structures=structures,state=state,scene=None)
        Image.fromarray(sample.image[:,:,::-1]).save(output/(state+'.png'))
        Image.fromarray(synth.labels_at(sample.depth_mm,sample.classes,2)).save(output/(state+'_threshold_2mm.png'))
        rows.append(dict(state=state,seed=100+index,seconds=round(time.perf_counter()-t,2),info=sample.info))
        print(state,rows[-1]['seconds'],sample.info.get('render'),flush=True)
    (output/'report.json').write_text(json.dumps(dict(source=item,scaling=factor,geometry=geom,scene=None,
        warning='Legacy planar fallback: visual depth priors, not calibrated physical measurement; labels are diagnostic',
        results=rows),indent=2,default=float))
    crop=(130,100,730,230)
    from PIL import ImageDraw
    sheet=Image.new('RGB',(1200,560),'white');d=ImageDraw.Draw(sheet)
    for i,name in enumerate(['source','partially_cleared','cleared','not_cleared']):
        im=Image.open(output/(name+'.png')).crop(crop)
        x=(i%2)*600;y=(i//2)*280;sheet.paste(im,(x,y+35));d.text((x+5,y+8),name,fill='black')
    sheet.save(output/'comparison.jpg',quality=95)
    print(output/'comparison.jpg',flush=True)

if __name__=='__main__':main()
