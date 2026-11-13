"""Roof-specific scalar fields; all randomness is consumed before area selection."""
import cv2
import numpy as np
from scipy import ndimage as ndi

SCENARIOS = {
    'sparse_patches': .06, 'scattered_patches': .18, 'partial_cover': .45,
    'near_full_cover': .92, 'edge_residue': .08, 'cleared_irregular': .35,
    'thin_streaks': .12, 'shadowed_snow': .4,
    'panel_residue': .18, 'wind_drift': .65,
}


def scenario_field(name, base, loaded, rng, owner=None, geometry=None):
    if name not in SCENARIOS:
        raise ValueError(f'Unknown scenario: {name}')
    h, w = base.shape
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    scale = max(1., np.sqrt(loaded.allowed.sum()))
    if name in {'panel_residue','wind_drift'}:
        if owner is None or not geometry:
            raise ValueError(f'{name} requires surface geometry')
        score=np.full((h,w),-1e6,np.float32)
        points=np.stack([xx,yy],-1).reshape(-1,1,2)
        for i,g in enumerate(geometry):
            region=(owner==i)&loaded.allowed
            if not region.any():continue
            if g['grid_to_image'] is not None:
                uv=cv2.perspectiveTransform(points,np.linalg.inv(np.asarray(g['grid_to_image'],np.float64))).reshape(h,w,2)
                u=uv[...,0]/max(g['grid_size'][0]-1,1);v=uv[...,1]/max(g['grid_size'][1]-1,1)
            else:
                ys,xs=np.nonzero(region)
                u=(xx-xs.min())/max(int(xs.max()-xs.min()),1)
                v=(yy-ys.min())/max(int(ys.max()-ys.min()),1)
            if name=='panel_residue':
                # User-provided seams are normalized in slope-grid coordinates.
                seams=g.get('seam_u')
                if seams is None:
                    raise ValueError('panel_residue requires annotated seam_u on every surface')
                distance=np.full_like(u,np.inf)
                for seam in seams:np.minimum(distance,np.abs(u-seam),out=distance)
                distance*=len(seams)
                near=np.exp(-distance*9)
                # Interruption along the fall line avoids continuous CAD-like stripes.
                breaks=np.sin(v*rng.uniform(10,24)+base*.9+rng.uniform(0,6.28))
                s=near*(.55+.3*breaks)+.28*base
            else:
                # A directional concentration prior, not a CFD prediction.
                wind=float(rng.choice([-1.,1.]))
                direction=wind*(u-.5)*.65+(v-.5)*.4
                s=direction+.55*base
            score[region]=s[region]
        return score
    if name in {'sparse_patches', 'scattered_patches'}:
        n = int(rng.integers(2, 5) if name == 'sparse_patches' else rng.integers(15, 31))
        ys, xs = np.nonzero(loaded.allowed)
        score = np.full((h,w), -1e6, np.float32)
        warp_x=xx+.025*scale*base
        warp_y=yy+.018*scale*cv2.GaussianBlur(base,(0,0),max(1.,scale*.008))
        for j in rng.integers(len(xs), size=n):
            radius = scale * (rng.uniform(.10,.23) if name == 'sparse_patches' else rng.uniform(.025,.07))
            stretch = rng.uniform(.6,1.7)
            angle=rng.uniform(0,np.pi)
            dx,dy=warp_x-xs[j],warp_y-ys[j]
            u=dx*np.cos(angle)+dy*np.sin(angle)
            v=-dx*np.sin(angle)+dy*np.cos(angle)
            score = np.maximum(score, -((u*stretch)**2 + (v/stretch)**2)/(radius**2))
        return score + .55*base
    if name == 'edge_residue':
        dist = ndi.distance_transform_edt(np.pad(loaded.roof,1))[1:-1,1:-1]
        return -dist/scale*20 + .25*base
    if name == 'thin_streaks':
        if owner is not None and geometry:
            score=np.zeros_like(base)
            points=np.stack([xx,yy],-1).reshape(-1,1,2)
            for i,g in enumerate(geometry):
                region=owner==i
                if g['grid_to_image'] is not None:
                    uv=cv2.perspectiveTransform(points,np.linalg.inv(np.asarray(g['grid_to_image'],np.float64))).reshape(h,w,2)
                    u,v=uv[...,0],uv[...,1]
                    period=max(3.,g['grid_size'][0]*.035)
                else:u,v,period=xx,yy,max(2.,scale*.022)
                angle=rng.uniform(-.08,.08)
                s=np.cos((u*np.cos(angle)+v*np.sin(angle))/period+.2*base)+.12*base
                score[region]=s[region]
            return score
        angle = rng.uniform(-.35,.35)
        transverse = xx*np.cos(angle) + yy*np.sin(angle)
        return np.cos(transverse / max(2.,scale*.022) + .18*base) + .08*base
    if name == 'cleared_irregular':
        a = rng.uniform(0,2*np.pi)
        return (xx*np.cos(a)+yy*np.sin(a))/scale + .4*base
    if name == 'shadowed_snow':
        gray = loaded.image.astype(np.float32).mean(2)/255
        shade = cv2.GaussianBlur(gray,(0,0),max(1.,scale*.04))
        return base*.25 - shade*4
    return base  # partial and near-full differ primarily by requested area
