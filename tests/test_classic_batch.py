"""Batch CLI tests use synthetic RGB scenes; no ML weights or network access."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from tools import build_classic_roof_dataset as batch
from tools import approve_classic_roof_dataset as approve


def scene():
    im=np.full((200,340,3),(108,127,140),np.uint8)
    cv2.fillPoly(im,[np.array([[22,30],[157,35],[165,143],[20,148]])],(198,88,65))
    cv2.fillPoly(im,[np.array([[201,40],[307,24],[319,151],[191,155]])],(178,81,56))
    return im


class BatchTests(unittest.TestCase):
    def test_proposals_resumable_and_same_named_photos_not_colliding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'photos';out=root/'batch'
            for sub in ('a','b'):
                folder=source/sub;folder.mkdir(parents=True)
                Image.fromarray(scene()).save(folder/'same.jpg')
            args=['--input',str(source),'--output',str(out),
                  '--min-confidence','.35','--min-area','180','--max-roofs','2']
            self.assertEqual(batch.main(args),0)
            dirs=list((out/'samples').glob('img_*/records.json'))
            self.assertEqual(len(dirs),2)
            self.assertNotEqual(dirs[0].parent.name,dirs[1].parent.name)
            self.assertTrue(any((out/'samples'/p.parent.name/'proposals').glob('*/sample.json') for p in dirs))
            self.assertEqual(batch.main(args+['--resume']),0)
            summary=json.loads((out/'summary.json').read_text())
            self.assertEqual(summary['skipped_existing'],2)
            with self.assertRaises(SystemExit):
                batch.main(args+['--resume','--min-confidence','.6'])

    def test_resized_cropped_render_keeps_masks_aligned(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);src=root/'photos';src.mkdir();out=root/'fast'
            # Scaling the synthetic scene makes an HD-like source without
            # depending on a network, a pretrained net or private photos.
            large=cv2.resize(scene(),(1360,800),interpolation=cv2.INTER_NEAREST)
            Image.fromarray(large).save(src/'one.jpg')
            def fake_run(argv,**kwargs):
                dest=Path(argv[argv.index('--output')+1]);dest.mkdir()
                specfile=Path(argv[3])
                payload=json.loads(specfile.read_text())
                sample=payload['sample']
                with Image.open(specfile.parent/sample['image_path']) as image:
                    rgb=np.asarray(image.convert('RGB'))
                with Image.open(specfile.parent/sample['roof_mask']) as mask:
                    roof=np.asarray(mask)>0
                self.assertEqual(roof.shape,rgb.shape[:2])
                snow=np.full(roof.shape,255,np.uint8);snow[roof]=1
                Image.fromarray(rgb).save(dest/'image.png')
                Image.fromarray(snow).save(dest/'mask.png')
                return subprocess.CompletedProcess(argv,0,'','')
            cmd=['--input',str(src),'--output',str(out),'--render','--verified-clean',
                 '--max-roofs','2','--variants','1','--min-confidence','.35',
                 '--work-max-side','512','--roof-crop-pad','24',
                 '--include-clean-controls','--no-ridges']
            with patch.object(batch.subprocess,'run',side_effect=fake_run):
                self.assertEqual(batch.main(cmd),0)
            rows=batch.collect_rows(out/'samples')
            self.assertGreaterEqual(len(rows),2)
            for row in rows:
                with Image.open(out/row['image']) as im:
                    size=im.size
                with Image.open(out/row['mask']) as im:
                    self.assertEqual(im.size,size)
                    self.assertTrue(set(np.unique(im))<={0,1,255})
                with Image.open(out/row['roi']) as im:
                    self.assertEqual(im.size,size)
                self.assertEqual((int(row['original_width']),int(row['original_height'])),(1360,800))
                self.assertLessEqual(int(row['work_width']),512)
                self.assertLess(int(row['work_height']),512)
            # Cropped masks remain valid in their own image coordinate system.
            self.assertEqual(batch.main(cmd+['--resume']),0)
            with self.assertRaises(SystemExit):
                batch.main(cmd+['--resume','--roof-crop-pad','25'])

    def test_processing_uses_original_when_max_side_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);src=root/'scene.png'
            Image.fromarray(scene()).save(src)
            image,path,size=batch.prepare_work_image(src,root/'working',0,90)
            self.assertEqual(size,(340,200))
            self.assertEqual(path,src)
            self.assertEqual(image.shape,scene().shape)

    def test_refuse_to_render_unverified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);src=root/'photos';src.mkdir()
            Image.fromarray(scene()).save(src/'one.png')
            with self.assertRaises(SystemExit):
                batch.main(['--input',str(src),'--output',str(root/'out'),'--render'])

    def test_rendered_draft_can_be_reviewed_and_exported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);src=root/'photos';src.mkdir();out=root/'batch'
            Image.fromarray(scene()).save(src/'one.jpg')
            # Replaces only external renderer process. Detectors, roof masks,
            # export and dataset review are exercised with the actual code.
            def fake_run(argv,**kwargs):
                dest=Path(argv[argv.index('--output')+1]);dest.mkdir()
                payload=json.loads(Path(argv[3]).read_text())
                sample_path=Path(argv[3]).parent
                with Image.open(sample_path/payload['sample']['roof_mask']) as im:
                    roof=np.array(im)>0
                shape=roof.shape
                im=np.zeros((*shape,3),np.uint8);im[:]=[220,220,220]
                mask=np.full(shape,255,np.uint8)
                mask[roof]=0
                mask[roof & (np.indices(shape)[1]<shape[1]//2)]=1
                Image.fromarray(im).save(dest/'image.png')
                Image.fromarray(mask).save(dest/'mask.png')
                return subprocess.CompletedProcess(argv,0,'','')
            opts=['--input',str(src),'--output',str(out),'--render','--verified-clean',
                  '--min-confidence','.35','--min-area','180','--max-roofs','1',
                  '--variants','2','--include-clean-controls']
            with patch.object(batch.subprocess,'run',side_effect=fake_run):
                self.assertEqual(batch.main(opts),0)
            rows=batch.collect_rows(out/'samples')
            self.assertEqual(len(rows),3)
            self.assertEqual({r['split'] for r in rows},{'train_candidate'})
            self.assertEqual({r['review_status'] for r in rows},{'unreviewed'})
            with (out/'review.csv').open(newline='',encoding='utf-8') as f:reviews=list(csv.DictReader(f))
            reviews[0]['decision']='accept';reviews[0]['reviewer']='qa_tester'
            with (out/'review.csv').open('w',newline='',encoding='utf-8') as f:
                w=csv.DictWriter(f,fieldnames=reviews[0].keys());w.writeheader();w.writerows(reviews)
            approved=root/'approved'
            self.assertEqual(approve.main(['--run',str(out),'--output',str(approved)]),0)
            self.assertTrue((approved/'train/images'/f"{reviews[0]['sample_id']}.png").is_file())
            self.assertTrue((approved/'train/masks'/f"{reviews[0]['sample_id']}.png").is_file())
            self.assertEqual(len(batch.collect_rows(out/'samples')),3)
            self.assertEqual(batch.main(opts+['--resume']),0)
            self.assertEqual(len(batch.collect_rows(out/'samples')),3)


if __name__=='__main__':unittest.main()
