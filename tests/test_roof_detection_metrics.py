import importlib.util
from pathlib import Path
import unittest
import numpy as np

SCRIPT=Path(__file__).resolve().parents[1]/'tools'/'eval_roof_detection.py'
SPEC=importlib.util.spec_from_file_location('eval_roof_detection',SCRIPT)
E=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(E)

class MetricContractTests(unittest.TestCase):
    def test_false_snowable_detects_facades_and_obstacles(self):
        roof=np.zeros((40,50),bool);roof[10:30,10:30]=True
        gt=roof.copy();pred=roof.copy();pred[5:10,10:30]=True  # facade false positive
        gtocc=np.zeros_like(roof);gtocc[15:20,15:20]=True
        predocc=np.zeros_like(roof)  # missed skylight
        score=E.metrics(pred,gt,predocc,gtocc)
        self.assertGreater(score['false_snowable_rate'],0)
        self.assertEqual(score['obstacle_recall'],0)
        self.assertLess(score['roof_iou'],1)

    def test_perfect_mask(self):
        roof=np.zeros((40,50),bool);roof[10:30,10:30]=True
        occ=np.zeros_like(roof);occ[12:15,12:15]=True
        score=E.metrics(roof,roof,occ,occ)
        self.assertEqual(score['roof_iou'],1)
        self.assertEqual(score['false_snowable_rate'],0)
        self.assertEqual(score['obstacle_recall'],1)

if __name__=='__main__':unittest.main()
