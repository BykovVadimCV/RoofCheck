"""Compare a saved flat run and volume run; probe height on the same input and seed."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,json,html
import numpy as np
from PIL import Image,ImageOps,ImageDraw
from roofcheck.roof import RoofSample,GenerationConfig,generate,save_result,load_sample


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--before',type=Path,required=True);p.add_argument('--after',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--before-label',default='old flat')
    p.add_argument('--after-label',default='new volume')
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    panels=[];stats=[];cases=['archive_06','archive_17','archive_26']
    for id in cases:
        for scenario in ['partial_cover','near_full_cover']:
            before=a.before/'generated'/id/scenario;after=a.after/'generated'/id/scenario
            m=json.loads((after/'metadata.json').read_text());spec=RoofSample(**m['sample']);loaded=load_sample(spec)
            ys,xs=np.nonzero(loaded.roof)
            box=(max(0,int(xs.min())-12),max(0,int(ys.min())-12),min(loaded.roof.shape[1],int(xs.max())+13),min(loaded.roof.shape[0],int(ys.max())+13))
            images=[loaded.image,np.array(Image.open(before/'image.png')),np.array(Image.open(after/'image.png'))]
            row=Image.new('RGB',(1500,340),'white');d=ImageDraw.Draw(row)
            for i,rgb in enumerate(images):
                thumb=ImageOps.pad(Image.fromarray(rgb).crop(box),(500,300),color='white')
                row.paste(thumb,(i*500,40))
            d.text((8,8),f'{id} {scenario} | source / {a.before_label} / {a.after_label}',fill='black')
            panels.append(row)
            h=np.load(after/'height_visual_px.npy');face=np.array(Image.open(after/'face_mask.png'))
            stats.append(dict(id=id,scenario=scenario,max_visual_height_px=float(h.max()),face_pixels=int((face>0).sum()),
                target=m['coverage']['target_coverage'],actual=m['coverage']['final_visible_coverage'],
                calibration_trials=m['coverage']['coverage_calibration_trials']))
    sheet=Image.new('RGB',(1500,340*len(panels)),'white')
    for i,row in enumerate(panels):sheet.paste(row,(0,340*i))
    sheet.save(a.output/'before_after.jpg',quality=95)
    panels[2].save(a.output/'preview.jpg',quality=95)
    # The same close-up roof and seed, independently recalibrated to the same final visible area.
    m=json.loads((a.after/'generated/archive_17/partial_cover/metadata.json').read_text())
    spec=RoofSample(**m['sample']);loaded=load_sample(spec)
    ys,xs=np.nonzero(loaded.roof);box=(int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1)
    sheet=Image.new('RGB',(1200,760),'white');d=ImageDraw.Draw(sheet)
    for i,height in enumerate([0.,3.,6.,12.]):
        cfg=GenerationConfig(**(m['config']|dict(height_px=height)))
        r=generate(spec,cfg,m['seed']);save_result(spec,r,a.output/f'height_{int(height):02d}')
        x=(i%2)*600;y=(i//2)*380
        sheet.paste(ImageOps.pad(Image.fromarray(r.image).crop(box),(600,340),color='white'),(x,y+40))
        d.text((x+8,y+8),f'height_px={height:g}; actual max={r.diagnostics["height_visual_px"].max():.1f}; target={m["coverage"]["target_coverage"]:.2f}',fill='black')
    sheet.save(a.output/'height_comparison.jpg',quality=95)
    (a.output/'comparison_metrics.json').write_text(json.dumps(stats,indent=2))
    body='''<!doctype html><html lang="ru"><meta charset="utf-8"><title>Снег: освещение и объём</title><style>body{font:17px system-ui;max-width:1500px;margin:32px auto;padding:0 20px;background:#f5f6f8;color:#20252c}img{max-width:100%}section{padding:20px;background:white;margin:24px 0}a{color:#1565c0}</style><h1>Сравнение старого и объёмного рендера</h1><p>Слева исходник, в центре старый плоский рендер, справа новая версия. Доля видимого снега одинаковая с допуском; после проекции контуры могут различаться.</p><p>Объём — приблизительное 2.5D поле высот в пикселях. Это не измерение сантиметров по фотографии. Маски входных поверхностей черновые.</p><section><h2>До / после</h2><a href="before_after.jpg"><img src="before_after.jpg"></a></section><section><h2>Управление высотой</h2><p>Один источник и seed, высота 0 / 3 / 6 / 12. Площадь повторно подбирается по финальной видимости.</p><a href="height_comparison.jpg"><img src="height_comparison.jpg"></a></section><p><a href="comparison_metrics.json">Параметры и проверки</a></p></html>'''
    body=body.replace('старый плоский рендер',html.escape(a.before_label)).replace('новая версия',html.escape(a.after_label))
    (a.output/'index.html').write_text(body)
    print(a.output/'index.html')


if __name__=='__main__':main()
