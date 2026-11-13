"""Behavioural checks for projected volume, relighting and training-label consistency."""
import tempfile
import unittest
from pathlib import Path
from dataclasses import replace
import numpy as np
from PIL import Image
from roofcheck.roof import RoofSample,GenerationConfig,generate,load_sample,save_result
from roofcheck.roof.lighting import estimate_lighting
from roofcheck.roof.volume import build_volume,trace_heightfield


class VolumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        h,w=180,240
        rgb=np.full((h,w,3),205,np.uint8)
        rgb[15:165,15:225]=[65,72,80]
        roof=np.zeros((h,w),np.uint8);roof[15:165,15:225]=1
        occ=np.zeros_like(roof);occ[:,118:122]=1
        rgb[occ>0]=[30,32,34]
        mask=np.full_like(roof,255);mask[(roof>0)&(occ==0)]=0
        mask[35:50,35:55]=1;rgb[35:50,35:55]=[220,225,230]
        for name,a in [('image',rgb),('roof',roof),('occ',occ),('snow',mask)]:Image.fromarray(a).save(self.root/f'{name}.png')
        self.spec=RoofSample('volume',self.root/'image.png','one_roof',self.root/'roof.png',
                             snow_mask=self.root/'snow.png',occluder_mask=self.root/'occ.png')
        self.cfg=GenerationConfig(target_coverage=.35,height_px=12)

    def test_visible_height_faces_and_unit_normals(self):
        r=generate(self.spec,self.cfg,42);d=r.diagnostics
        self.assertGreater(float(d['height_visual_px'].max()),1)
        self.assertTrue(d['face_mask'].any())
        self.assertTrue(((d['face_mask']>0)&(r.mask==1)).any())
        np.testing.assert_allclose(np.linalg.norm(d['normals'],axis=2),1,atol=1e-5)
        self.assertTrue(np.isfinite(d['height_visual_px']).all())
        # Shading diagnostics are not metric ground-truth depth.
        self.assertIn('2.5D',r.metadata['rendering']['render_mode'])

    def test_height_changes_silhouette_and_preserves_protected_pixels(self):
        low=generate(self.spec,replace(self.cfg,height_px=0),42)
        high=generate(self.spec,self.cfg,42)
        self.assertFalse(np.array_equal(low.image,high.image))
        self.assertFalse(np.array_equal(low.alpha,high.alpha))
        self.assertFalse(low.diagnostics['face_mask'].any())
        loaded=load_sample(self.spec)
        np.testing.assert_array_equal(high.image[~loaded.allowed],loaded.image[~loaded.allowed])
        np.testing.assert_array_equal(high.mask[~loaded.allowed],loaded.snow[~loaded.allowed])
        self.assertLess(abs(high.metadata['coverage']['final_visible_coverage']-.35),.001)

    def test_light_changes_appearance_not_segmentation(self):
        a=generate(self.spec,self.cfg,3)
        b=generate(self.spec,replace(self.cfg,light_direction=(.5,.6,.8)),3)
        np.testing.assert_array_equal(a.mask,b.mask)
        np.testing.assert_array_equal(a.alpha,b.alpha)
        self.assertFalse(np.array_equal(a.image,b.image))
        self.assertFalse(np.array_equal(a.diagnostics['shadow'],b.diagnostics['shadow']))

    def test_shadows_are_not_snow(self):
        # A grazing source casts an actual ground shadow beyond the snow transition.
        r=generate(self.spec,replace(self.cfg,light_direction=(-1.,-.8,.25)),42)
        shadow=r.diagnostics['shadow']>.005
        self.assertTrue(shadow.any())
        # Pure cast shadows outside any snow transition retain source labels.
        bare=shadow&(r.alpha==0)
        self.assertTrue(bare.any())
        loaded=load_sample(self.spec)
        np.testing.assert_array_equal(r.mask[bare],loaded.snow[bare])

    def test_no_saturated_white_material(self):
        r=generate(self.spec,self.cfg,8)
        snow=(r.mask==1)&load_sample(self.spec).allowed
        self.assertTrue(snow.any())
        self.assertLess(float((r.image[snow].max(1)>=254).mean()),.01)

    def test_dark_material_not_interpreted_as_deep_shadow(self):
        loaded=load_sample(self.spec)
        owner=np.where(loaded.roof,0,-1).astype(np.int32)
        light,white,meta=estimate_lighting(loaded,owner)
        self.assertGreater(float(np.median(light[loaded.allowed])),.85)
        self.assertGreater(float(white.mean()),.3)
        self.assertTrue(np.isfinite(light).all())

    def test_tiny_blurred_component_is_not_confident_snow(self):
        loaded=load_sample(self.spec)
        a=np.zeros(loaded.roof.shape,np.float32);a[80,80]=1
        owner=np.where(loaded.roof,0,-1).astype(np.int32)
        cfg=replace(self.cfg,height_px=0,camera_blur=2)
        g=build_volume(loaded,a,np.zeros_like(a),cfg,owner,0)
        self.assertLess(float(g.visibility.max()),cfg.alpha_high)

    def test_diagnostics_roundtrip_and_flat_reference(self):
        r=generate(self.spec,self.cfg,2)
        save_result(self.spec,r,self.root/'out')
        np.testing.assert_array_equal(np.load(self.root/'out/height_visual_px.npy'),r.diagnostics['height_visual_px'])
        self.assertEqual(np.array(Image.open(self.root/'out/face_mask.png')).dtype,np.uint8)
        height=np.array(Image.open(self.root/'out/height_visual_px.png')).astype(float)/1000
        np.testing.assert_allclose(height,r.diagnostics['height_visual_px'],atol=.001)
        flat=generate(self.spec,replace(self.cfg,render_mode='flat'),2)
        self.assertFalse(flat.diagnostics)
        self.assertAlmostEqual(flat.metadata['coverage']['generated_coverage'],flat.metadata['coverage']['final_visible_coverage'])

    def test_invalid_volume_parameters(self):
        for change in [dict(height_px=-1),dict(height_px=float('nan')),dict(camera_blur=3),
                       dict(light_direction=(0,0,0)),dict(lift=(0,1)),dict(freshness='ice'),dict(shadow_strength=2)]:
            with self.subTest(change=change):
                with self.assertRaises(ValueError):replace(self.cfg,**change).validate()

    def test_continuous_ray_intersection_matches_analytic_ramp(self):
        y,x=np.mgrid[:60,:80].astype(np.float32)
        height=2+.07*y
        alpha=np.ones_like(height);region=np.ones_like(height,bool)
        _,_,_,u,v=trace_heightfield(height,alpha,region,(0,-1))
        # z = H(y+z) = 2 + .07*(y+z), solved analytically away from the image border.
        expected=(2+.07*y)/.93
        np.testing.assert_allclose((v-y)[8:40,8:70],expected[8:40,8:70],atol=.025)
        self.assertGreater(float(np.mean(np.abs((v-y)-np.round(v-y))[8:40,8:70])),.15)
        np.testing.assert_array_equal(u,x)

    def test_narrow_residue_does_not_form_a_thick_tube(self):
        loaded=load_sample(self.spec)
        owner=np.where(loaded.roof,0,-1).astype(np.int32)
        narrow=np.zeros_like(loaded.roof,np.float32);narrow[60:140,60:63]=1
        broad=np.zeros_like(narrow);broad[60:140,45:95]=1
        a=build_volume(loaded,narrow,np.zeros_like(narrow),self.cfg,owner,4)
        b=build_volume(loaded,broad,np.zeros_like(narrow),self.cfg,owner,4)
        self.assertLess(float(a.height.max()),float(b.height.max())*.5)

    def test_real_texture_changes_appearance_not_labels(self):
        rng=np.random.default_rng(77)
        tile=rng.integers(80,220,(64,64,3),dtype=np.uint8)
        path=self.root/'texture.png';Image.fromarray(tile).save(path)
        base=generate(self.spec,self.cfg,12)
        cfg=replace(self.cfg,texture_path=str(path),real_texture_mix=.8)
        textured=generate(self.spec,cfg,12)
        again=generate(self.spec,cfg,12)
        np.testing.assert_array_equal(base.mask,textured.mask)
        np.testing.assert_array_equal(base.alpha,textured.alpha)
        np.testing.assert_array_equal(textured.image,again.image)
        self.assertFalse(np.array_equal(base.image,textured.image))
        self.assertEqual(len(textured.metadata['rendering']['real_texture']['sha256']),64)
        with self.assertRaises(FileNotFoundError):
            generate(self.spec,replace(cfg,texture_path=str(self.root/'missing.jpg')),12)
        path.with_suffix('.json').write_text('{"split":"test"}')
        with self.assertRaises(ValueError):generate(self.spec,cfg,12)


if __name__=='__main__':unittest.main()
