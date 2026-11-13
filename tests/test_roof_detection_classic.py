"""Neural-free detector tests with exact synthetic roof/obstacle annotations."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np
from PIL import Image

from roofcheck.roof_detection.classical import ClassicalRoofBackend
from roofcheck.roof_detection import build_detection, select, write_detection


ROOT=Path(__file__).resolve().parents[1]


def scene():
    im=np.zeros((185,320,3),np.uint8)
    im[:]=[118,134,141]
    # Two disconnected warm tiled roofs with strict polygon boundaries.
    cv2.fillPoly(im,[np.array([[25,30],[145,33],[151,128],[27,132]])],(192,95,67))
    cv2.fillPoly(im,[np.array([[190,42],[294,28],[302,137],[183,140]])],(174,81,60))
    # Definite dark skylight fully enclosed in roof #1.
    cv2.rectangle(im,(74,72),(90,85),(55,73,98),-1)
    cv2.rectangle(im,(74,72),(90,85),(238,238,237),1)
    # Roof seam running from border to border. It may be proposed as ridge.
    cv2.line(im,(28,111),(148,108),(89,47,35),2)
    # Thin red marking on the ground should not become an entire roof.
    cv2.rectangle(im,(160,156),(313,160),(188,90,69),-1)
    return im


class ClassicRoofTests(unittest.TestCase):
    def test_classic_detects_both_roofs_without_weights(self):
        image=scene()
        maps=ClassicalRoofBackend(max_side=512,min_area=180).predict(image)
        self.assertEqual(maps.model_info['backend'],'classical-opencv')
        self.assertEqual(maps.model_info['weights'],'none')
        det=build_detection(maps,min_roof_area=180,instance_min_confidence=0.2)
        self.assertGreaterEqual(len(det.roofs),2)
        self.assertTrue(any(r.mask[50,45] for r in det.roofs))
        self.assertTrue(any(r.mask[60,250] for r in det.roofs))
        self.assertFalse(any(r.mask[158,240] for r in det.roofs))
        self.assertTrue(maps.obstacle[78,82]>.5)
        self.assertFalse(any(r.mask[78,82] and not maps.obstacle[78,82] for r in det.roofs))

    def test_invalid_settings(self):
        for bad in ({'max_side':100},{'min_area':10},{'density_window':8},
                    {'density_growth':.7,'density_seed':.4}):
            with self.assertRaises(ValueError):ClassicalRoofBackend(**bad)
        with self.assertRaises(ValueError):ClassicalRoofBackend().predict(np.zeros((12,20),np.uint8))

    def test_automated_multi_roof_export_without_neural_networks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            image=scene()
            file=root/'photo.png';Image.fromarray(image).save(file)
            result=root/'detections'
            command=[sys.executable,str(ROOT/'tools/detect_roof_classic.py'),
                     '--image',str(file),'--output',str(result),
                     '--min-area','180','--min-confidence','.4',
                     '--image-id','synthetic1','--group-id','scene01']
            run=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=90)
            self.assertEqual(run.returncode,0,run.stderr+'\n'+run.stdout)
            summary=json.loads((result/'summary.json').read_text())
            self.assertGreaterEqual(summary['exported'],2)
            for item in summary['samples']:
                folder=result/item['folder']
                payload=json.loads((folder/'sample.json').read_text())
                self.assertEqual(payload['sample']['annotation_mode'],'partial')
                self.assertEqual(payload['sample']['group_id'],'scene01')
                roof=np.asarray(Image.open(folder/'roof_mask.png'))>0
                occ=np.asarray(Image.open(folder/'occluder_mask.png'))>0
                snowable=np.asarray(Image.open(folder/'snowable_mask.png'))>0
                self.assertTrue(np.array_equal(snowable,roof&~occ))
                self.assertTrue((folder/'cvat_proposals.xml').is_file())
                self.assertTrue((folder/'source.png').is_file())
                self.assertGreater(int(roof.sum()),0)


if __name__=='__main__':unittest.main()
