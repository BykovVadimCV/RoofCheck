"""Image-space 2.5D snow solid. Reuses solid_height and horizon visibility from snow3d.

Visual height is in projected pixels, not measured metric snow depth. Continuous rays
intersect the snow surface; final visibility is clipped to the original surface.
"""
from dataclasses import dataclass
import cv2
import numpy as np
from ..snow3d import solid_height, _horizon


@dataclass
class VolumeGeometry:
    height: np.ndarray
    alpha: np.ndarray
    visibility: np.ndarray
    face: np.ndarray
    projected_height: np.ndarray
    owner: np.ndarray
    map_x: np.ndarray
    map_y: np.ndarray


def trace_heightfield(height,alpha,region,lift,step=.5):
    """Continuous orthographic ray/heightfield intersection, no rounded height layers.

    A ray through photo (x,y) samples base (x,y)-lift*z. March from the highest
    layer, then refine the first entry. Saved inverse maps drive appearance too.
    """
    hh,ww=height.shape
    yy,xx=np.mgrid[:hh,:ww].astype(np.float32)
    hit=np.zeros_like(region);z_hit=np.zeros_like(height)
    maximum=float(height.max())
    def inside(z):
        u=(xx-lift[0]*z).astype(np.float32);v=(yy-lift[1]*z).astype(np.float32)
        h=cv2.remap(height,u,v,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        a=cv2.remap(alpha,u,v,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        weight=cv2.remap(region.astype(np.float32),u,v,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        return (h/np.maximum(weight,1e-6)>=z)&(a>.001)&region
    if maximum>0:
        for z in np.arange(maximum+step,0,-step):
            new=inside(float(z))&~hit
            z_hit[new]=z;hit|=new
        # Keep a valid inside lower bound and outside upper bound for each first hit.
        lo=z_hit.copy();hi=z_hit+step
        for _ in range(5):
            middle=(lo+hi)*.5;enters=inside(middle)
            lo=np.where(hit&enters,middle,lo);hi=np.where(hit&~enters,middle,hi)
        z_hit=np.where(hit,lo,0).astype(np.float32)
    u=(xx-lift[0]*z_hit).astype(np.float32);v=(yy-lift[1]*z_hit).astype(np.float32)
    weight=cv2.remap(region.astype(np.float32),u,v,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    visible=cv2.remap(alpha,u,v,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)/np.maximum(weight,1e-6)*region
    visible=np.clip(visible,0,1)
    h=cv2.remap(height,u,v,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)/np.maximum(weight,1e-6)
    gy=np.gradient(height,axis=0) if hh>1 else np.zeros_like(height)
    gx=np.gradient(height,axis=1) if ww>1 else np.zeros_like(height)
    slope=cv2.remap(np.hypot(gx,gy),u,v,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    face=(slope>.7)&(visible>.5)&(h>.15)
    return visible,face,h,u,v


def _shift(a,dx,dy):
    return cv2.warpAffine(a,np.float32([[1,0,dx],[0,1,dy]]),(a.shape[1],a.shape[0]),flags=cv2.INTER_NEAREST)


def project_columns(height,alpha,region,lift,top=None,wall=None):
    """Near source rows win for the default up-image lift (painter order, as in paint_heightfield).

    Max-alpha avoids accumulating translucent fringe pixels into spurious opaque snow.
    Iterating column height draws the full wall, not a flat translated cutout.
    """
    maximum=int(np.ceil(height.max()))
    aout=np.zeros_like(alpha);face=np.zeros_like(region);hout=np.zeros_like(height)
    cout=np.zeros_like(top) if top is not None else None
    integer=np.rint(height).astype(np.int32)
    # Include a zero-height footprint. Extrusion therefore cannot remove its base footprint.
    for k in range(maximum+1):
        layer=(integer>=k)&(alpha>0)&region
        if not layer.any():continue
        dx,dy=int(round(lift[0]*k)),int(round(lift[1]*k))
        a=_shift(np.where(layer,alpha,0),dx,dy)
        front=(a>0)&(a>=aout)&region
        is_face=_shift((layer&(integer>k)).astype(np.uint8),dx,dy)>0
        aout[front]=a[front];face[front]=is_face[front]
        shifted_height=_shift(height,dx,dy);hout[front]=shifted_height[front]
        if top is not None:
            colour=np.where((integer>k)[...,None],wall,top)
            c=_shift(colour,dx,dy);cout[front]=c[front]
    # Quantized source heights can create isolated one-pixel "contour walls" inside
    # a smooth body. Keep only faces at least two pixels tall (legacy painter fix).
    adjacent=np.zeros_like(face)
    adjacent[1:]|=face[:-1];adjacent[:-1]|=face[1:]
    solid=face&adjacent
    sliver=face&~solid
    if cout is not None:
        above=np.concatenate([cout[:1],cout[:-1]],axis=0)
        above_a=np.concatenate([aout[:1],aout[:-1]],axis=0)
        replace=sliver&(above_a>0)
        cout[replace]=above[replace]
    return aout,solid,hout,cout


def _regions(owner,allowed):
    for i in np.unique(owner[allowed]):
        r=(owner==i)&allowed
        if not r.any():continue
        ys,xs=np.nonzero(r)
        # Crop per slope: horizon/projection cost is independent of the rest of a large city photo.
        sl=(slice(int(ys.min()),int(ys.max())+1),slice(int(xs.min()),int(xs.max())+1))
        yield int(i),sl,r[sl]


def smooth_visibility(alpha,region,sigma):
    if sigma<=0:return np.where(region,alpha,0).astype(np.float32)
    den=cv2.GaussianBlur(region.astype(np.float32),(0,0),sigma)
    num=cv2.GaussianBlur(alpha.astype(np.float32),(0,0),sigma)
    return np.where(region,num/np.maximum(den,1e-6),0).astype(np.float32)


def build_volume(loaded,alpha,texture,config,owner,seed):
    height=np.zeros_like(alpha);visible=np.zeros_like(alpha);face=np.zeros_like(loaded.allowed)
    projected=np.zeros_like(alpha)
    yy,xx=np.mgrid[:alpha.shape[0],:alpha.shape[1]].astype(np.float32)
    map_x=xx.copy();map_y=yy.copy()
    for i,sl,region in _regions(owner,loaded.allowed):
        core=(alpha[sl]>=config.alpha_high)&region
        a=np.where(region,alpha[sl],0).astype(np.float32)
        if core.any() and config.height_px>0:
            # Scale down for distant/thin photo surfaces; no inferred physical metres.
            scale=np.clip(min(region.shape)/350,.12,1.5)
            broad=cv2.GaussianBlur(texture[sl],(0,0),max(1.,min(region.shape)*.025))
            broad=(broad-float(np.mean(broad[region])))/max(float(np.std(broad[region])),.05)
            mid=cv2.GaussianBlur(texture[sl],(0,0),max(1.,min(region.shape)*.006))
            mid=(mid-float(np.mean(mid[region])))/max(float(np.std(mid[region])),.05)
            body=config.height_px*scale*(1+.16*np.tanh(broad)+.04*np.tanh(mid))
            D=np.where(core,body*1000,0).astype(np.float32)
            h,_=solid_height(D,None,None,config.freshness,np.random.default_rng(seed+i),P=1,
                             relief_mm=D)
            h[~core]=0
            # A melting/powder edge approaches the metal continuously instead of
            # ending in a constant-height cake wall. Narrow streaks stay thin.
            from scipy.ndimage import distance_transform_edt
            dist=distance_transform_edt(np.pad(core,1))[1:-1,1:-1].astype(np.float32)
            toe=np.clip(dist/(1.5+.6*body),0,1)
            toe=toe*toe*(3-2*toe)
            if config.scenario=='cleared_irregular':toe=.55+.45*toe
            h=np.minimum(h,dist*.70+.25)*toe
        else:h=np.zeros_like(a)
        av,f,hp,u,v=trace_heightfield(h,a,region,config.lift)
        av=smooth_visibility(av,region,config.camera_blur)
        av[av<=config.alpha_low]=0
        height[sl]=np.where(region,h,height[sl]);visible[sl]=np.where(region,av,visible[sl])
        face[sl]=np.where(region,f,face[sl]);projected[sl]=np.where(region,hp,projected[sl])
        map_x[sl]=np.where(region,u+sl[1].start,map_x[sl])
        map_y[sl]=np.where(region,v+sl[0].start,map_y[sl])
    return VolumeGeometry(height,alpha,visible,face,projected,owner,map_x,map_y)


def shade_volume(geometry,texture,config,loaded):
    shading=np.ones_like(geometry.height)
    normals=np.zeros(geometry.height.shape+(3,),np.float32);normals[...,2]=1
    for _,sl,region in _regions(geometry.owner,loaded.allowed):
        h=geometry.height[sl]
        # Subpixel material relief perturbs light, not the silhouette or snow labels.
        micro=.10*cv2.GaussianBlur(texture[sl],(0,0),.65)*np.clip(h/2,0,1)
        lit_height=h+micro
        gy=np.gradient(lit_height,axis=0) if h.shape[0]>1 else np.zeros_like(h)
        gx=np.gradient(lit_height,axis=1) if h.shape[1]>1 else np.zeros_like(h)
        n=np.stack([-gx,-gy,np.ones_like(h)],-1)
        n/=np.linalg.norm(n,axis=2,keepdims=True)
        light=np.array(config.light_direction,np.float32);light/=np.linalg.norm(light)
        direct=np.clip(n@light,0,1)
        flat=max(float(light[2]),.1)
        sky=.55+.45*n[...,2]
        vis=_horizon(h,1.,dirs=4,steps=(1,3,7,15))
        s=((1-config.directional_strength)*sky+config.directional_strength*direct/flat)
        s*=1-config.ambient_occlusion*(1-vis)
        shading[sl]=np.where(region,np.clip(s,.48,1.16),shading[sl])
        normals[sl]=np.where(region[...,None],n,normals[sl])
    return shading,normals


def project_colours(geometry,top,wall,config,loaded):
    colour=np.zeros_like(top)
    for _,sl,region in _regions(geometry.owner,loaded.allowed):
        u=(geometry.map_x[sl]-sl[1].start).astype(np.float32)
        v=(geometry.map_y[sl]-sl[0].start).astype(np.float32)
        h=geometry.height[sl]
        gy=np.gradient(h,axis=0) if h.shape[0]>1 else np.zeros_like(h)
        gx=np.gradient(h,axis=1) if h.shape[1]>1 else np.zeros_like(h)
        # A continuous face blend replaces binary top/wall bands at integer heights.
        nz=1/np.sqrt(1+gx*gx+gy*gy)
        blend=np.clip((1-nz-.08)/.72,0,.75)
        source=top[sl]*(1-blend[...,None])+wall[sl]*blend[...,None]
        weight=cv2.remap(region.astype(np.float32),u,v,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        raw=cv2.remap(geometry.alpha[sl],u,v,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)/np.maximum(weight,1e-6)*region
        c=cv2.remap(source*region[...,None],u,v,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)/np.maximum(weight[...,None],1e-6)
        # Premultiplied filtering prevents dark halos from empty texture pixels.
        if config.camera_blur>0:
            prem=cv2.GaussianBlur(c*raw[...,None],(0,0),config.camera_blur)
            den=cv2.GaussianBlur(raw,(0,0),config.camera_blur)
            c=prem/np.maximum(den[...,None],1e-6)
        colour[sl]=np.where(region[...,None],c,colour[sl])
    return colour


def cast_shadow(geometry,config,loaded):
    shadow=np.zeros_like(geometry.visibility)
    if config.shadow_strength==0 or not geometry.height.any():return shadow
    direction=np.array(config.light_direction[:2],np.float32)
    direction=-direction/max(float(config.light_direction[2]),.25)
    for _,sl,region in _regions(geometry.owner,loaded.allowed):
        h=geometry.height[sl];core=(h>0).astype(np.float32);s=np.zeros_like(core)
        # Height-dependent silhouettes, softened according to the area-light approximation.
        for k in range(1,int(np.ceil(h.max()))+1):
            s=np.maximum(s,_shift(core*(h>=k-.5),round(direction[0]*k),round(direction[1]*k)))
        s=cv2.GaussianBlur(s,(0,0),max(.7,config.camera_blur))
        s*=region*(1-geometry.visibility[sl])*config.shadow_strength
        shadow[sl]=np.where(region,s,shadow[sl])
    return shadow
