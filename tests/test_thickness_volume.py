"""The internal snow depth drives the planar height-field extrusion."""
import unittest
import numpy as np
from roofcheck import synth

class ThicknessVolumeTests(unittest.TestCase):
    def test_thick_depth_creates_visible_face(self):
        yy,xx=np.mgrid[:160,:240]
        image=np.stack([70+xx//4,80+yy//5,90+xx//5],axis=2).astype(np.uint8)
        quad=np.array([[20,140],[220,140],[200,30],[40,30]],np.float32)
        geom={'length_m':8.,'depth_m':5.,'pitch_deg':25.}
        def run(amount):
            return synth.make_sample(image,quad,quad,np.random.default_rng(22),geom,
                state='not_cleared',fence_mode='none',structure_mode='manual',
                texture_policy='isolated',ground_snow_mm=amount)
        thin,thick=run(20),run(450)
        self.assertGreater(np.median(thick.depth_mm[thick.roi]),np.median(thin.depth_mm[thin.roi])*5)
        self.assertGreater(thick.info['face_px'],thin.info['face_px'])
        self.assertTrue(set(np.unique(synth.labels_at(thick.depth_mm,thick.classes,2))) <= {0,1,255})

if __name__=='__main__':unittest.main()
