import importlib.util
from pathlib import Path
import unittest
import numpy as np

P=Path(__file__).resolve().parents[1]/'tools'/'eval_roof_instances.py'
S=importlib.util.spec_from_file_location('instance_metrics',P)
E=importlib.util.module_from_spec(S);S.loader.exec_module(E)


class InstanceMetrics(unittest.TestCase):
    def test_perfect_with_permuted_instance_ids(self):
        a=np.zeros((30,40),np.uint16);b=np.zeros_like(a)
        a[4:12,4:12]=10;a[15:25,20:30]=20
        b[4:12,4:12]=50;b[15:25,20:30]=30
        result=E.match_instance_maps(a,b)
        self.assertEqual(result['matched'],2)
        self.assertEqual(result['recall'],1.0)
        self.assertEqual(result['precision'],1.0)

    def test_missing_and_extra_instances(self):
        a=np.zeros((30,50),np.uint16);b=np.zeros_like(a)
        b[3:13,3:13]=1;b[4:15,20:30]=2
        a[3:13,3:13]=4;a[17:25,35:45]=5
        r=E.match_instance_maps(a,b)
        self.assertEqual(r['matched'],1)
        self.assertEqual(r['recall'],0.5)
        self.assertEqual(r['precision'],0.5)

    def test_unequal_shapes_rejected(self):
        with self.assertRaises(ValueError):
            E.match_instance_maps(np.zeros((10,10)),np.zeros((8,10)))

if __name__=='__main__':unittest.main()
