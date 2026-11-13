"""Robust host exposure/colour matching. Illumination remains a single-image approximation."""
import cv2
import numpy as np


def srgb_to_lin(x):
    x = x.astype(np.float32)/255.
    return np.where(x <= .04045,x/12.92,((x+.055)/1.055)**2.4)


def lin_to_srgb(x):
    x = np.clip(x,0,1)
    return np.round(np.where(x <= .0031308,x*12.92,1.055*x**(1/2.4)-.055)*255).astype(np.uint8)


def estimate_lighting(loaded,owner):
    shape=loaded.roof.shape
    # Work at bounded resolution: broad illumination must not inherit every metal seam.
    ratio=min(1.,512/max(shape));size=(max(8,round(shape[1]*ratio)),max(8,round(shape[0]*ratio)))
    rgb=cv2.resize(loaded.image.astype(np.float32)/255,size,interpolation=cv2.INTER_AREA)
    labels=cv2.resize(owner.astype(np.float32),size,interpolation=cv2.INTER_NEAREST).astype(np.int32)
    usable=cv2.resize((loaded.roof & ~loaded.occluders).astype(np.uint8),size,interpolation=cv2.INTER_NEAREST)>0
    light=np.ones(rgb.shape[:2],np.float32)
    lum=rgb.mean(2)
    plane_stats=[]
    global_median=float(np.median(lum[usable])) if usable.any() else .5
    for i in np.unique(labels[usable]):
        if i<0:continue
        r=(labels==i)&usable
        values=lum[r]
        lo,hi=np.percentile(values,[10,85])
        med=max(float(np.median(values)),.03)
        # Reject highlights and black objects; cap residuals before smoothing.
        reliable=r&(lum>=lo)&(lum<=hi)
        if not reliable.any():reliable=r
        weight=reliable.astype(np.float32)
        sigma=max(2.,min(size)*.025)
        low=cv2.GaussianBlur(np.clip(lum/med,.55,1.3)*weight,(0,0),sigma)
        den=cv2.GaussianBlur(weight,(0,0),sigma)
        relative=np.where(den>1e-5,low/np.maximum(den,1e-5),1.)
        # Per-plane normalisation avoids interpreting dark roof paint as dark illumination.
        orientation=np.clip((med/max(global_median,.03))**.12,.88,1.08)
        light[r]=(1+.45*(np.clip(relative,.65,1.18)-1))[r]*orientation
        plane_stats.append(dict(owner=int(i),surface_median=med))
    light=cv2.resize(light,(shape[1],shape[0]),interpolation=cv2.INTER_LINEAR)
    # Estimate a scene white from neutral pixels, independently of roof albedo.
    bright=rgb.max(2);neutral=(rgb.max(2)-rgb.min(2)<.14)&(bright>.35)&(bright<.995)
    candidates=rgb[neutral]
    if len(candidates)<16:candidates=rgb.reshape(-1,3)
    white=np.percentile(candidates,97,axis=0)
    temperature=np.clip(white/max(float(white.mean()),.05),.90,1.10)
    white_level=float(np.clip(np.mean(white)*.96,.70,.96))
    white_lin=srgb_to_lin(np.array([[[white_level*255]*3]],np.float32))[0,0]
    # Conservative noise estimate: low-amplitude camera grain, never a pasted high-frequency roof texture.
    residual=lum-cv2.GaussianBlur(lum,(0,0),.7)
    noise=float(np.clip(np.median(np.abs(residual[usable])) if usable.any() else 0,0,.006))
    return light,white_lin*temperature.astype(np.float32),dict(
        white_level_srgb=white_level,colour_temperature_rgb=temperature.tolist(),noise_sigma_srgb=noise,
        planes=plane_stats,method='trimmed per-surface relative luminance; neutral-scene white',
        limitation='Single-image reflectance and illumination cannot be fully separated')
