"""Linear-light compositing; final visibility clipping occurs after all filtering."""
import cv2
import numpy as np


def srgb_to_lin(x):
    x = x.astype(np.float32)/255.
    return np.where(x <= .04045,x/12.92,((x+.055)/1.055)**2.4)


def lin_to_srgb(x):
    x = np.clip(x,0,1)
    return np.round(np.where(x <= .0031308,x*12.92,1.055*x**(1/2.4)-.055)*255).astype(np.uint8)


def _flat_render(loaded, alpha, texture, config):
    # Approximate low-frequency host illumination; no physical depth or centimetre claim.
    lum = loaded.image.astype(np.float32).mean(2)/255
    illum = cv2.GaussianBlur(lum,(0,0),max(2.,min(lum.shape)*.035))
    roof_lum = illum[loaded.roof & ~loaded.occluders]
    ref = float(np.median(roof_lum)) if roof_lum.size else .5
    light = np.clip(illum/max(ref,.05),.65,1.12)
    colour = np.array([.78,.81,.85],np.float32)[None,None,:]
    snow = colour*light[...,None]*(1+config.texture_strength*texture[...,None])
    # Defend against bleed after filtering/projection, not only before it.
    visible = alpha.copy()
    visible[~loaded.allowed] = 0
    visible[visible <= config.alpha_low] = 0
    image = lin_to_srgb(srgb_to_lin(loaded.image)*(1-visible[...,None])+snow*visible[...,None])
    untouched = (visible == 0) | loaded.occluders | (loaded.snow == 1) | ~loaded.roof
    image[untouched] = loaded.image[untouched]
    return image, visible


def render(loaded,alpha,texture,config,geometry=None,owner=None,seed=0,return_diagnostics=False,real_texture=None):
    if config.render_mode=='flat':
        image,visible=_flat_render(loaded,alpha,texture,config)
        return (image,visible,{}) if return_diagnostics else (image,visible)
    from .lighting import estimate_lighting
    from .volume import build_volume,shade_volume,project_colours,cast_shadow
    if owner is None:owner=np.where(loaded.roof,0,-1).astype(np.int32)
    if geometry is None:geometry=build_volume(loaded,alpha,texture,config,owner,seed)
    light,white,lighting_meta=estimate_lighting(loaded,owner)
    shade,normals=shade_volume(geometry,texture,config,loaded)
    # Two independent bands: soft body variation and small camera-resolved grain.
    body=cv2.GaussianBlur(texture,(0,0),max(1.,min(texture.shape)*.008))
    detail=.30*body+.70*texture
    if real_texture is not None:
        detail=(1-config.real_texture_mix)*detail+config.real_texture_mix*real_texture
    top=white[None,None,:]*light[...,None]*shade[...,None]*(1+config.texture_strength*detail[...,None])
    # Thin material may reveal subdued seams; deep material suppresses them exponentially.
    host=srgb_to_lin(loaded.image)
    low=cv2.GaussianBlur(host,(0,0),2.)
    seam=np.clip(host-low,-.12,.12)*np.exp(-geometry.height[...,None]/1.5)*.45
    top=np.clip(top+seam,0,1)
    wall=white[None,None,:]*light[...,None]*np.array([.77,.84,.94],np.float32)
    colour=project_colours(geometry,top,wall,config,loaded)
    colour=np.clip(colour,0,1)
    # Avoid featureless white clipping; respect the scene white rather than absolute white.
    ceiling=np.maximum(white*1.08,.25)
    colour=np.minimum(colour,ceiling[None,None,:])
    visible=geometry.visibility.copy();visible[~loaded.allowed]=0
    shadow=cast_shadow(geometry,config,loaded)
    ground=host*(1-shadow[...,None])
    image=lin_to_srgb(ground*(1-visible[...,None])+colour*visible[...,None])
    # The estimate is in encoded camera units: adding it in linear light washed
    # out fine texture. Match luminance-dominated camera grain after encoding.
    camera_rng=np.random.default_rng(seed+1729)
    grain=camera_rng.normal(0,1,visible.shape+(1,)).astype(np.float32)
    chroma=camera_rng.normal(0,.20,loaded.image.shape).astype(np.float32)
    sigma=lighting_meta['noise_sigma_srgb']*255
    image=np.clip(np.rint(image.astype(np.float32)+(grain+chroma)*sigma*visible[...,None]),0,255).astype(np.uint8)
    untouched=(~loaded.allowed)|((visible==0)&(shadow==0))
    image[untouched]=loaded.image[untouched]
    diagnostics=dict(height_visual_px=geometry.projected_height.astype(np.float32),
        height_base_px=geometry.height.astype(np.float32),face_mask=geometry.face.astype(np.uint8),
        normals=normals.astype(np.float32),shadow=shadow.astype(np.float32),lighting=lighting_meta,
        render_mode='2.5D continuous ray/heightfield intersection; not calibrated 3D',
        reused=['snow3d.solid_height','snow3d._horizon','continuous orthographic ray intersection'])
    return (image,visible,diagnostics) if return_diagnostics else (image,visible)
