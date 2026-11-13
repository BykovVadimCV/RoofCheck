import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
from roofcheck.roof_detection import ProbabilityMaps, build_detection, select, write_detection


SCRIPT=Path(__file__).resolve().parents[1]/'tools'/'import_roof_detection_cvat.py'
S=importlib.util.spec_from_file_location('import_cvat_roofs',SCRIPT)
mod=importlib.util.module_from_spec(S)
S.loader.exec_module(mod)


class ReviewRoundTrip(unittest.TestCase):
    def test_export_import_two_facets_and_structure(self):
        h,w=150,240
        roof=np.zeros((h,w),np.float32);roof[20:120,20:200]=.99
        boundary=np.zeros_like(roof);boundary[20:120,109:112]=.99
        obstacle=np.zeros_like(roof);obstacle[50:75,40:65]=.99
        maps=ProbabilityMaps(roof,boundary,obstacle,model_info={'backend':'synthetic'})
        d=build_detection(maps,min_roof_area=50)
        self.assertEqual(len(d.roofs[0].facets),2)
        d.selection=select(d.roofs)
        with tempfile.TemporaryDirectory() as temp:
            temp=Path(temp)
            original=temp/'frame.png'
            Image.fromarray(np.zeros((h,w,3),np.uint8)).save(original)
            out=temp/'proposals';write_detection(np.zeros((h,w,3),np.uint8),original,d,out)
            reviewed=temp/'reviewed'
            # A reviewer must explicitly accept every machine-created facet/structure.
            import xml.etree.ElementTree as ET
            ann=ET.parse(out/'cvat_proposals.xml')
            for poly in ann.getroot().iter('polygon'):
                for att in poly.findall('attribute'):
                    if att.get('name')=='reviewed':att.text='yes'
            accepted=temp/'accepted.xml';ann.write(accepted)
            path=mod.import_review(accepted,original,reviewed,verified_clean=True)
            payload=json.loads(path.read_text())
            self.assertEqual(payload['sample']['annotation_mode'],'verified_clean')
            self.assertEqual(len(payload['sample']['surfaces']),3)
            self.assertEqual(int((np.asarray(Image.open(reviewed/'occluder_mask.png'))>0).sum()),25*25)
            sem=np.asarray(Image.open(reviewed/'semantic_training_mask.png'))
            self.assertTrue((sem==1).any());self.assertTrue((sem==2).any());self.assertTrue((sem==3).any())

    def test_unreviewed_machine_facets_cannot_enter_training(self):
        h,w=96,144
        roof=np.zeros((h,w),np.float32);roof[12:84,12:126]=.99
        det=build_detection(ProbabilityMaps(roof,np.zeros_like(roof),np.zeros_like(roof)),min_roof_area=50)
        det.selection=select(det.roofs)
        with tempfile.TemporaryDirectory() as temp:
            temp=Path(temp)
            photo=temp/'photo.png';Image.fromarray(np.zeros((h,w,3),np.uint8)).save(photo)
            proposals=temp/'p';write_detection(np.zeros((h,w,3),np.uint8),photo,det,proposals)
            with self.assertRaisesRegex(ValueError,'reviewed=yes'):
                mod.import_review(proposals/'cvat_proposals.xml',photo,temp/'imported')

    def test_semantic_training_labels_include_non_target_roofs(self):
        import xml.etree.ElementTree as ET
        h,w=96,192
        roof=np.zeros((h,w),np.float32)
        roof[10:75,10:80]=.99;roof[15:84,110:184]=.99
        det=build_detection(ProbabilityMaps(roof,np.zeros_like(roof),np.zeros_like(roof)),min_roof_area=50)
        det.selection=select(det.roofs,point=(20,20))
        with tempfile.TemporaryDirectory() as temp:
            temp=Path(temp)
            photo=temp/'photo.png';Image.fromarray(np.zeros((h,w,3),np.uint8)).save(photo)
            proposals=temp/'p';write_detection(np.zeros((h,w,3),np.uint8),photo,det,proposals)
            tree=ET.parse(proposals/'cvat_proposals.xml')
            for a in tree.getroot().iter('attribute'):
                if a.get('name')=='reviewed':a.text='yes'
            accepted=temp/'accepted.xml';tree.write(accepted)
            target=temp/'review';mod.import_review(accepted,photo,target)
            sem=np.asarray(Image.open(target/'semantic_training_mask.png'))
            mask=np.asarray(Image.open(target/'roof_mask.png'))
            self.assertEqual(sem[40,130],1)
            self.assertEqual(mask[40,130],0)
            self.assertEqual(mask[30,20],255)

    def test_multiple_roofs_without_target_rejected(self):
        import xml.etree.ElementTree as ET
        with tempfile.TemporaryDirectory() as temp:
            temp=Path(temp);original=temp/'im.png'
            Image.fromarray(np.zeros((100,100,3),np.uint8)).save(original)
            root=ET.Element('annotations');im=ET.SubElement(root,'image',width='100',height='100')
            for pts in ['1,1;40,1;40,40;1,40','50,50;90,50;90,90;50,90']:
                ET.SubElement(im,'polygon',label='roof',points=pts)
            xml=temp/'ann.xml';ET.ElementTree(root).write(xml)
            with self.assertRaisesRegex(ValueError,'target'):
                mod.import_review(xml,original,temp/'out')

if __name__=='__main__':unittest.main()
