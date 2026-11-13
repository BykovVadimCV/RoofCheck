"""Per-surface projection. Coordinates are pixel-relative, never inferred metres."""
import cv2
import numpy as np


def estimate_quad(region):
    """Approximate a slope from its annotated boundary, without a metric/camera claim."""
    contours, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(contour)
    method = 'boundary_quad'
    quad = None
    for epsilon in (.01,.02,.03,.05):
        approx = cv2.approxPolyDP(hull,epsilon*cv2.arcLength(hull,True),True)
        if len(approx) == 4:
            quad = approx[:,0,:].astype(np.float32)
            break
    if quad is None:
        quad = cv2.boxPoints(cv2.minAreaRect(hull))
        method = 'boundary_rectangle_fallback'
    centre = quad.mean(0)
    quad = quad[np.argsort(np.arctan2(quad[:,1]-centre[1],quad[:,0]-centre[0]))]
    quad = np.roll(quad,-int(np.argmin(quad.sum(1))),axis=0)
    return quad.astype(np.float32), method


def warp_texture(tile, matrix, shape):
    """Perspective sampling with Jacobian-based mip levels, adapted from sample_tile."""
    h, w = shape
    yy, xx = np.mgrid[:h,:w].astype(np.float32)
    points = np.stack([xx,yy],axis=-1).reshape(-1,1,2)
    uv = cv2.perspectiveTransform(points,np.linalg.inv(matrix)).reshape(h,w,2)
    u,v = uv[...,0],uv[...,1]
    ux,uy = np.gradient(u,axis=1),np.gradient(u,axis=0)
    vx,vy = np.gradient(v,axis=1),np.gradient(v,axis=0)
    footprint = np.maximum(np.hypot(ux,vx),np.hypot(uy,vy))
    maxlevel = min(6,max(0,int(np.floor(np.log2(min(tile.shape))))-1))
    level = np.clip(np.log2(np.maximum(footprint,1)),0,maxlevel)
    result = np.zeros(shape,np.float32)
    for k in range(maxlevel+1):
        weight = np.clip(1-np.abs(level-k),0,1)
        result += cv2.remap(tile,u/(2**k),v/(2**k),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)*weight
        if min(tile.shape)>2:
            tile = cv2.pyrDown(tile)
    return result


def projected_noise(shape, scale, rng, octaves=4):
    """Multi-octave value noise, derived from the existing fbm implementation."""
    h, w = shape
    out = np.zeros(shape, np.float32)
    for i in range(octaves):
        step = max(1., scale / 2**i)
        grid = rng.random((max(2, int(h / step) + 2), max(2, int(w / step) + 2))).astype(np.float32)
        out += .5**i * cv2.resize(grid, (w, h), interpolation=cv2.INTER_CUBIC)
    return (out - out.mean()) / (out.std() + 1e-6)


def material_texture(shape,scale,rng):
    h,w=shape
    texture=(.55*projected_noise(shape,scale,rng,3)+
             .30*projected_noise(shape,max(2.,scale/3),rng,2)+
             .15*projected_noise(shape,max(1.,scale/9),rng,1))
    matrix=cv2.getRotationMatrix2D((w/2,h/2),float(rng.uniform(0,360)),1.)
    texture=cv2.warpAffine(texture,matrix,(w,h),borderMode=cv2.BORDER_WRAP)
    return np.roll(texture,(int(rng.integers(h)),int(rng.integers(w))),(0,1))


def surface_fields(loaded, rng):
    """Return shape score/texture in photo space and explicit geometry provenance.

    image: image coordinates. approximate: annotated quad (TL,TR,BR,BL) to unit grid.
    homography: supplied grid->photo matrix + grid_size [width,height].
    Polygon coordinates always refer to the EXIF-oriented photo.
    Surfaces sorted by z_order; a later surface owns overlap pixels.
    """
    shape = loaded.roof.shape
    surfaces = loaded.spec.surfaces or [dict(surface_id='roof', mode='image')]
    ids = [s.get('surface_id') for s in surfaces]
    if any(not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('Each surface requires a unique surface_id')
    fields = np.zeros(shape, np.float32)
    texture = np.zeros(shape, np.float32)
    owner = np.full(shape, -1, np.int32)
    meta = []
    for i, s in enumerate(sorted(surfaces, key=lambda x: x.get('z_order', 0))):
        region = loaded.roof.copy()
        if 'polygon' in s:
            pts = np.asarray(s['polygon'], np.float32)
            if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3 or not np.isfinite(pts).all():
                raise ValueError('Invalid surface polygon')
            mask = np.zeros(shape, np.uint8)
            cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1)
            region &= mask > 0
        if not region.any():
            raise ValueError('Empty surface polygon')
        mode = s.get('mode', 'image')
        estimate_method = None
        scale = float(s.get('texture_scale', 24))
        seams=s.get('seam_u')
        if seams is not None and (not isinstance(seams,list) or not seams or
                not np.isfinite(seams).all() or any(not 0<=x<=1 for x in seams)):
            raise ValueError('seam_u must be a non-empty list of normalized coordinates')
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError('texture_scale must be positive')
        if mode == 'image':
            base = projected_noise(shape, scale * 2, rng)
            detail = material_texture(shape, scale, rng)
            matrix = None
        else:
            w, h = map(int, s.get('grid_size', [256, 256]))
            if not (8 <= w <= 4096 and 8 <= h <= 4096):
                raise ValueError('grid_size outside supported range 8..4096')
            if mode == 'approximate':
                if s.get('quad') is None:
                    q, estimate_method = estimate_quad(region)
                else:
                    q = np.asarray(s['quad'], np.float32)
                    estimate_method = 'annotated_quad'
                if q.shape != (4, 2) or not np.isfinite(q).all() or not cv2.isContourConvex(q):
                    raise ValueError('approximate requires convex quad TL,TR,BR,BL')
                matrix = cv2.getPerspectiveTransform(np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]]), q)
            elif mode == 'homography':
                matrix = np.asarray(s.get('grid_to_image'), np.float64)
            else:
                raise ValueError(f'Unknown geometry mode: {mode}')
            if matrix.shape != (3,3) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-10:
                raise ValueError('Invalid grid_to_image homography')
            corners = cv2.perspectiveTransform(np.float32([[[0,0],[w-1,0],[w-1,h-1],[0,h-1]]]),matrix)[0]
            if not np.isfinite(corners).all() or not cv2.isContourConvex(corners):
                raise ValueError('Projection must produce a finite convex surface')
            # Use the same polygon rasterization as annotations. Warping an all-ones bitmap
            # loses boundary pixels through nearest-sample rounding on trapezoids.
            valid = np.zeros(shape,np.uint8)
            cv2.fillPoly(valid,[np.round(corners).astype(np.int32)],1)
            # An inferred quad approximates the texture coordinates, not the semantic
            # boundary. Irregular/rasterized edges remain governed by the supplied mask.
            if not (mode == 'approximate' and s.get('quad') is None):
                region &= valid > 0
            base = cv2.warpPerspective(projected_noise((h,w), scale*2, rng), matrix, (shape[1],shape[0]),borderMode=cv2.BORDER_REFLECT_101)
            # Perspective warp naturally reduces texture scale with distance; antialias fine detail first.
            detail = warp_texture(material_texture((h,w),scale,rng),matrix,shape)
        fields[region] = base[region]
        texture[region] = detail[region]
        owner[region] = i
        meta.append(dict(surface_id=s['surface_id'],mode=mode,texture_scale=scale,
                         approximation=estimate_method,
                         seam_u=seams,
                         grid_size=[w,h] if matrix is not None else None,
                         grid_to_image=matrix.tolist() if matrix is not None else None,
                         scale_units='canonical pixels; not physical metres'))
    if np.any(loaded.roof & (owner < 0)):
        raise ValueError('Surface geometry does not cover the full annotated roof')
    return fields, texture, owner, meta
