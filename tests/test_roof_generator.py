"""Contract/regression tests; no external photos, weights or network required."""
import json
import ast
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import numpy as np
import cv2
from PIL import Image

from roofcheck.roof import RoofSample, GenerationConfig, generate, load_sample, save_result
from roofcheck.roof.masks import validate_mask, save_mask, merge_snow, coverage, materials_to_binary
from roofcheck.roof.scenarios import SCENARIOS
from roofcheck.roof.renderer import render


class RoofTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        yy,xx = np.mgrid[:100,:160]
        self.rgb = np.stack([60+xx//3,65+yy//3,80+xx//4],axis=-1).astype(np.uint8)
        self.roof = np.zeros((100,160),np.uint8);self.roof[10:90,10:150] = 1
        self.occ = np.zeros_like(self.roof);self.occ[:,79:82] = 1;self.occ[35:50,50:65] = 1
        self.mask = np.full_like(self.roof,255);self.mask[self.roof==1] = 0
        self.mask[20:30,20:40] = 1;self.mask[60:70,30:50] = 255
        for name,array in [('photo',self.rgb),('roof',self.roof),('occ',self.occ),('snow',self.mask)]:
            Image.fromarray(array).save(self.root/f'{name}.png')
        self.spec = RoofSample('frame',self.root/'photo.png','building',self.root/'roof.png',
                               snow_mask=self.root/'snow.png',occluder_mask=self.root/'occ.png')

    def test_mask_contract(self):
        save_mask(self.root/'mask.png',self.mask)
        with self.assertRaises(ValueError):validate_mask(self.mask.astype(np.uint16))
        bad=self.mask.copy();bad[0,0]=3
        with self.assertRaises(ValueError):validate_mask(bad)
        self.assertEqual(materials_to_binary(np.array([[0,1,2,3,4]],np.uint8)).tolist(),[[0,1,1,255,1]])

    def test_merge(self):
        source=np.array([[0,1,255,255,0,0]],np.uint8)
        a=np.array([[1,0,0,1,.4,1]],np.float32)
        allowed=np.array([[1,1,1,1,1,0]],bool)
        np.testing.assert_array_equal(merge_snow(source,a,allowed),[[1,1,255,1,255,0]])
        self.assertEqual(coverage(source,np.ones_like(source,bool))['valid_pixels'],4)

    def test_all_scenarios_preserve_protected_pixels(self):
        hashes=[]
        for name in SCENARIOS:
            with self.subTest(name=name):
                spec=self.spec
                if name=='panel_residue':
                    spec=replace(spec,surfaces=[dict(surface_id='slope',mode='approximate',
                        quad=[[10,10],[149,10],[149,89],[10,89]],seam_u=[.1,.25,.4,.55,.7,.85])])
                result=generate(spec,GenerationConfig(scenario=name),42)
                loaded=load_sample(spec)
                np.testing.assert_array_equal(result.image[~loaded.allowed],self.rgb[~loaded.allowed])
                np.testing.assert_array_equal(result.mask[~loaded.allowed],self.mask[~loaded.allowed])
                self.assertTrue(set(np.unique(result.mask))<={0,1,255})
                # Projected walls alter the footprint; acceptance is on final visible coverage.
                self.assertLess(abs(result.metadata['coverage']['target_coverage']-
                                    result.metadata['coverage']['final_visible_coverage']),.001)
                hashes.append(result.mask.tobytes())
        self.assertEqual(len(set(hashes)),len(SCENARIOS))

    def test_reproducibility_and_changed_seed(self):
        config=GenerationConfig(target_coverage=.45)
        a=generate(self.spec,config,42);b=generate(self.spec,config,42);c=generate(self.spec,config,43)
        np.testing.assert_array_equal(a.image,b.image)
        np.testing.assert_array_equal(a.mask,b.mask)
        self.assertEqual(a.metadata,b.metadata)
        self.assertFalse(np.array_equal(a.mask,c.mask))

    def test_zero_and_full(self):
        a=generate(self.spec,GenerationConfig(target_coverage=0),10)
        np.testing.assert_array_equal(a.image,self.rgb)
        np.testing.assert_array_equal(a.mask,self.mask)
        b=generate(self.spec,GenerationConfig(target_coverage=1),10)
        loaded=load_sample(self.spec)
        self.assertTrue((b.mask[loaded.allowed]==1).all())

    def test_final_clip_after_visibility_transform(self):
        loaded=load_sample(self.spec)
        # Simulate a blur/parallax stage leaking alpha onto wires, existing snow and background.
        leaking=np.ones(loaded.roof.shape,np.float32)
        image,visible=render(loaded,leaking,np.zeros_like(leaking),GenerationConfig())
        self.assertFalse(visible[~loaded.allowed].any())
        np.testing.assert_array_equal(image[~loaded.allowed],loaded.image[~loaded.allowed])

    def test_no_unknown_to_background(self):
        spec=replace(self.spec,snow_mask=None)
        a=generate(spec,GenerationConfig(target_coverage=.1),9)
        self.assertFalse((a.mask==0).any())
        spec=replace(spec,annotation_mode='full')
        with self.assertRaises(ValueError):load_sample(spec)
        with self.assertRaises(ValueError):
            generate(replace(self.spec,snow_mask=None),GenerationConfig(target_coverage=0))

    def test_validation_and_atomic_save(self):
        with self.assertRaises(ValueError):generate(replace(self.spec,split='test'))
        with self.assertRaises(ValueError):generate(self.spec,GenerationConfig(target_coverage=1.1))
        with self.assertRaises(ValueError):generate(self.spec,GenerationConfig(scenario='parking'))
        with self.assertRaises(ValueError):load_sample(replace(self.spec,annotation_mode='full'))
        result=generate(self.spec,seed=2)
        out=self.root/'out'
        save_result(self.spec,result,out)
        np.testing.assert_array_equal(np.array(Image.open(out/'image.png')),result.image)
        np.testing.assert_array_equal(np.array(Image.open(out/'mask.png')),result.mask)
        self.assertEqual((out/'source.png').read_bytes(),(self.root/'photo.png').read_bytes())
        self.assertEqual(json.loads((out/'metadata.json').read_text())['seed'],2)
        with self.assertRaises(FileExistsError):save_result(self.spec,result,out)

    def test_exif_alignment(self):
        # Orientation 6: rotate 90 clockwise; masks in raw coordinates must rotate too.
        exif=Image.Exif();exif[274]=6
        Image.fromarray(self.rgb).save(self.root/'raw.jpg',exif=exif)
        spec=replace(self.spec,image_path=self.root/'raw.jpg',mask_coordinates='raw')
        loaded=load_sample(spec)
        self.assertEqual(loaded.image.shape[:2],(160,100))
        np.testing.assert_array_equal(loaded.snow,np.rot90(self.mask,-1))
        with self.assertRaises(ValueError):load_sample(replace(spec,mask_coordinates='oriented'))

    def test_multiple_surfaces_and_homography(self):
        # Two disjoint slopes covering the annotated roof, each with its own projection.
        surfaces=[]
        for name,x0,x1 in [('left',10,79),('right',80,149)]:
            q=[[x0,10],[x1,10],[x1,89],[x0,89]]
            surfaces.append(dict(surface_id=name,mode='approximate',polygon=q,quad=q,grid_size=[128,128]))
        result=generate(replace(self.spec,surfaces=surfaces),seed=1)
        self.assertEqual(len(result.metadata['geometry']),2)
        self.assertEqual(result.metadata['geometry'][0]['mode'],'approximate')
        # Explicit homography identity, full image grid.
        s=[dict(surface_id='all',mode='homography',grid_size=[160,100],grid_to_image=np.eye(3).tolist())]
        generate(replace(self.spec,surfaces=s),seed=1)
        inferred=generate(replace(self.spec,surfaces=[dict(surface_id='inferred',mode='approximate')]),seed=1)
        self.assertEqual(inferred.metadata['geometry'][0]['approximation'],'boundary_quad')
        with self.assertRaises(ValueError):generate(replace(self.spec,surfaces=surfaces[:1]))
        with self.assertRaises(ValueError):generate(replace(self.spec,surfaces=s+s))
        s[0]['grid_to_image']=np.zeros((3,3)).tolist()
        with self.assertRaises(ValueError):generate(replace(self.spec,surfaces=s))

    def test_existing_depth_labels_keep_clean_negatives(self):
        # Execute only this pure helper: the full legacy renderer needs optional MoGe dependencies.
        path=Path(__file__).resolve().parents[1]/'roofcheck/synth.py'
        node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='labels_at')
        namespace=dict(np=np,IGNORE_BAND_MM=1.,CLS={'ignore':8})
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
        labels=namespace['labels_at'](np.array([[0.,.5,1.,3.,0.]]),np.array([[0,0,0,0,8]]),1.)
        np.testing.assert_array_equal(labels,[[0,255,255,1,255]])

    def test_perspective_polygon_boundary_rounding(self):
        quad=np.float32([[30,10],[120,10],[149,89],[10,89]])
        roof=np.zeros_like(self.roof)
        cv2.fillPoly(roof,[quad.astype(np.int32)],1)
        Image.fromarray(roof).save(self.root/'trapezoid.png')
        spec=replace(self.spec,roof_mask=self.root/'trapezoid.png',
                     surfaces=[dict(surface_id='sloped',mode='approximate',quad=quad.tolist())])
        result=generate(spec,GenerationConfig(target_coverage=1),42)
        loaded=load_sample(spec)
        self.assertTrue((result.mask[loaded.allowed]==1).all())
        generate(replace(spec,surfaces=[dict(surface_id='inferred',mode='approximate')]),seed=42)

    def test_panel_seams_are_optional_validated_and_affect_placement(self):
        q=[[10,10],[149,10],[149,89],[10,89]]
        base=dict(surface_id='slope',mode='approximate',quad=q,grid_size=[128,128])
        config=GenerationConfig(scenario='panel_residue',target_coverage=.2)
        a=generate(replace(self.spec,surfaces=[base|dict(seam_u=[.1,.25,.4])]),config,15)
        b=generate(replace(self.spec,surfaces=[base|dict(seam_u=[.6,.75,.9])]),config,15)
        self.assertFalse(np.array_equal(a.mask,b.mask))
        self.assertEqual(a.metadata['geometry'][0]['seam_u'],[.1,.25,.4])
        with self.assertRaises(ValueError):
            generate(replace(self.spec,surfaces=[base|dict(seam_u=[1.3])]),config,15)
        with self.assertRaisesRegex(ValueError,'annotated seam_u'):
            generate(replace(self.spec,surfaces=[base]),config,15)


if __name__ == '__main__':unittest.main()
