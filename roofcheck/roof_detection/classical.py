"""Classical OpenCV roof proposals without neural nets or model weights.

This is an *unsupervised proposal generator*, not a semantic understanding
system. It detects strongly chromatic tiled roofs and (optionally) planar
neutral quadrilaterals; ridges and skylights are hypotheses. Confidence is a
heuristic ranking, NEVER a calibrated probability. A single oblique RGB image
cannot reliably distinguish every roof/facade, so downstream usage must carry
unreviewed status, allow rejection and validate masks on real examples.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np
from scipy import ndimage as ndi

from .types import ProbabilityMaps


def _components(binary: np.ndarray, min_area: int):
    n, ids, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
    for idx in range(1, n):
        x, y, w, h, area = map(int, stats[idx])
        if area >= min_area:
            yield (ids[y:y+h, x:x+w] == idx), (x, y, w, h), area


def _line_candidates(gray: np.ndarray):
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    found = lsd.detect(gray)[0]
    return found.reshape(-1, 4) if found is not None else np.empty((0,4), np.float32)


def _fill_small_holes(comp: np.ndarray, max_area: int, minimum: int):
    filled = ndi.binary_fill_holes(comp)
    holes = filled & ~comp
    accepted = np.zeros_like(comp)
    for part, (x, y, w, h), area in _components(holes, minimum):
        if area <= max_area:
            accepted[y:y+h, x:x+w] |= part
    return comp | accepted, accepted


def _extended_line(shape, segment):
    x1, y1, x2, y2 = map(float, segment)
    dx, dy = x2-x1, y2-y1
    length = math.hypot(dx,dy)
    if length <= 1e-6:
        return None
    ux, uy = dx/length, dy/length
    extent = math.hypot(*shape)*2
    a = (round(x1-ux*extent),round(y1-uy*extent))
    b = (round(x1+ux*extent),round(y1+uy*extent))
    return a,b


def _ridge_mask(rgb: np.ndarray, roof: np.ndarray, segments: np.ndarray,
                min_balance=.23, support_threshold=.55):
    """Find up to one strong long line which splits the roof into plausible facets.

    A seam may also be long and straight. The result is NOT confirmed geometry;
    facet proposals are marked unreviewed by the caller.
    """
    yx = np.argwhere(roof)
    if len(yx) < 300:
        return np.zeros(roof.shape, np.uint8), 0.0
    y0,x0 = yx.min(0);y1,x1 = yx.max(0)
    h,w = y1-y0+1,x1-x0+1
    edges = cv2.Canny(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),50,140)
    near = cv2.dilate(edges,np.ones((3,3),np.uint8))>0
    distance = cv2.distanceTransform(roof.astype(np.uint8),cv2.DIST_L2,3)
    cand=[]
    for line in segments:
        ax,ay,bx,by=map(float,line)
        length=math.hypot(ax-bx,ay-by)
        if length < max(15., .28*max(w,h)):
            continue
        cx,cy=(ax+bx)/2,(ay+by)/2
        ix,iy=int(round(cx)),int(round(cy))
        if not(0<=iy<roof.shape[0] and 0<=ix<roof.shape[1] and roof[iy,ix]):
            continue
        endpoints = [(int(round(ax)),int(round(ay))),(int(round(bx)),int(round(by)))]
        if any(not(0<=ex<roof.shape[1] and 0<=ey<roof.shape[0]) for ex,ey in endpoints):
            continue
        # Actual detected edge should span the roof, not just a local tile seam.
        if max(distance[ey,ex] for ex,ey in endpoints) > .14*min(w,h):
            continue
        pts=_extended_line(roof.shape,line)
        if pts is None:
            continue
        cut=np.zeros(roof.shape,np.uint8)
        cv2.line(cut,*pts,1,2)
        pieces=[]
        n,ids,stats,_=cv2.connectedComponentsWithStats((roof & (cut==0)).astype(np.uint8),8)
        for j in range(1,n):
            if stats[j,cv2.CC_STAT_AREA]>=max(40,int(roof.sum()*.06)):
                pieces.append(int(stats[j,cv2.CC_STAT_AREA]))
        if len(pieces)!=2 or min(pieces)/sum(pieces)<min_balance:
            continue
        num=max(20,int(length/2))
        xx=np.clip(np.rint(np.linspace(ax,bx,num)).astype(int),0,roof.shape[1]-1)
        yy=np.clip(np.rint(np.linspace(ay,by,num)).astype(int),0,roof.shape[0]-1)
        coverage=float(np.mean(near[yy,xx]))
        if coverage<support_threshold:
            continue
        score=.50*coverage+.30*min(1.,length/max(w,h))+.20*(min(pieces)/sum(pieces))*2
        cand.append((score,cut))
    if not cand:
        return np.zeros(roof.shape,np.uint8),0.
    score,cut=max(cand,key=lambda z:z[0])
    return cut.astype(np.uint8)*roof.astype(np.uint8),float(score)


def _neutral_quadrilaterals(image: np.ndarray, warm: np.ndarray, min_area: int):
    """Optional conservative closed-contour search for neutral planar roofs.

    The result can include courtyards/terraces; not enabled by default.
    """
    gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    hsv=cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    edges=cv2.Canny(cv2.GaussianBlur(gray,(5,5),1.2),65,170)
    edges=cv2.morphologyEx(edges,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    contours,_=cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    candidates=[]
    total=image.shape[0]*image.shape[1]
    for cnt in contours:
        area=cv2.contourArea(cnt)
        if area<min_area or area>total*.18:
            continue
        peri=cv2.arcLength(cnt,True)
        poly=cv2.approxPolyDP(cnt, .018*peri,True)
        if not(4<=len(poly)<=6 and cv2.isContourConvex(poly)):
            continue
        m=np.zeros(gray.shape,np.uint8)
        cv2.fillPoly(m,[poly],1)
        x,y,w,h=cv2.boundingRect(poly)
        if min(w,h)<15 or max(w,h)/max(min(w,h),1)>6:
            continue
        cover=int(m.sum())
        if np.count_nonzero(m.astype(bool)&warm)>cover*.12:
            continue
        inside=hsv[:,:,1][m>0]
        if len(inside)==0 or np.median(inside)>65:
            continue
        # Compare the immediate outside band against the enclosed surface.
        ring=(cv2.dilate(m,np.ones((13,13),np.uint8))>0)&(m==0)
        if not np.any(ring):
            continue
        contrast=abs(float(np.median(gray[m>0]))-float(np.median(gray[ring])))
        if contrast<17:
            continue
        # Strong long shape, distinct color, reasonably compact region.
        confidence=min(.55,.22+contrast/150)
        candidates.append((m.astype(bool),confidence))
    candidates.sort(key=lambda z:z[0].sum(),reverse=True)
    unique=[]
    for m,c in candidates:
        if all(np.count_nonzero(m&other)/max(1,np.count_nonzero(m|other))<.55 for other,_ in unique):
            unique.append((m,c))
        if len(unique)>=20:
            break
    return unique


@dataclass
class ClassicalRoofBackend:
    """Weight-free OpenCV backend focused on chromatic tiled roofs.

    Without pretrained semantics there is no general solution for arbitrary
    roof types. This backend emits explicit metadata for these limitations.
    """
    max_side: int = 1500
    min_area: int = 850
    warm_saturation: int = 32
    min_solidity: float = .29
    include_neutral: bool = False
    detect_ridges: bool = True
    detect_holes: bool = True
    min_confidence: float = .40
    density_window: int = 31
    density_seed: float = .43
    density_growth: float = .20

    def __post_init__(self):
        if self.max_side<256 or self.min_area<40 or not 0<self.min_solidity<=1:
            raise ValueError('Invalid classical detector dimensions/shape settings')
        if not 0<=self.min_confidence<=1 or not 0<self.density_growth<self.density_seed<1:
            raise ValueError('Invalid classical confidence/density thresholds')
        if self.density_window<3 or self.density_window%2!=1:
            raise ValueError('Density window must be odd and >=3')

    def predict(self, image: np.ndarray)->ProbabilityMaps:
        if image.ndim!=3 or image.shape[2]!=3 or image.dtype!=np.uint8:
            raise ValueError('Image must be HxWx3 RGB uint8')
        original_h,original_w=image.shape[:2]
        scale=min(1.,self.max_side/max(original_h,original_w))
        sw=max(32,int(round(original_w*scale)));sh=max(32,int(round(original_h*scale)))
        rgb=cv2.resize(image,(sw,sh),interpolation=cv2.INTER_AREA) if scale<1 else image.copy()
        hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
        h,s,v=cv2.split(hsv)
        red=rgb[:,:,0].astype(np.float32)
        green=rgb[:,:,1].astype(np.float32)
        blue=rgb[:,:,2].astype(np.float32)
        # Roof tiles: clay / red / orange / terracotta. Color alone also hits
        # brick facades; geometric filtering is used but cannot solve all ambiguity.
        warm=((h<=22)|(h>=175))&(s>=self.warm_saturation)&(v>=45)&(v<=245)&\
              (red>green*1.07)&(red>blue*1.055)
        warm=warm.astype(np.uint8)
        # Reject sparse orange clutter on asphalt/facades. Keep only dense
        # warm material, then reconstruct nearby roof-coloured edge pixels.
        density=cv2.boxFilter(warm.astype(np.float32),-1,
                             (self.density_window,self.density_window))
        mask=((warm>0)&(density>=self.density_seed)).astype(np.uint8)
        admissible=(warm>0)&(density>=self.density_growth)
        for _ in range(9):
            expanded=(cv2.dilate(mask,np.ones((3,3),np.uint8))>0)&admissible
            mask=expanded.astype(np.uint8)
        kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
        mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
        min_work=max(50,int(round(self.min_area*scale*scale)))
        instances=[]
        obstacle=np.zeros((sh,sw),np.float32)
        boundary=np.zeros((sh,sw),np.float32)
        roofprob=np.zeros((sh,sw),np.float32)
        segments=_line_candidates(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)) if self.detect_ridges else np.empty((0,4))
        rejection={'small_or_sparse':0,'shape':0,'confidence':0}
        proposal_info=[]
        for part,(x,y,w,hh),area in _components(mask,min_work):
            if min(w,hh)<14 or area/max(1,w*hh)<.15:
                rejection['small_or_sparse']+=1
                continue
            cnts,_=cv2.findContours(part.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cnt=max(cnts,key=cv2.contourArea)
            hull=cv2.convexHull(cnt)
            solidity=area/max(cv2.contourArea(hull),1.)
            if solidity<self.min_solidity:
                rejection['shape']+=1
                continue
            raw_count=int(warm[y:y+hh,x:x+w][part].sum())
            chromatic=raw_count/max(area,1)
            confidence=float(np.clip(.12+.42*chromatic+.26*min(1.,solidity)+.20*min(1.,area/(w*hh)),0,1))
            if confidence<self.min_confidence:
                rejection['confidence']+=1
                continue
            comp=np.zeros((sh,sw),bool)
            comp[y:y+hh,x:x+w]=part
            # Only fully enclosed holes are considered potential skylights,
            # vents, or obstructions. Gray speckles touching an edge remain unknown.
            if self.detect_holes:
                local,holes=_fill_small_holes(part,max(45,int(area*.04)),max(14,int(area*.0005)))
                comp[y:y+hh,x:x+w]=local
                ex=np.zeros((sh,sw),bool)
                ex[y:y+hh,x:x+w]=holes
                obstacle[ex]=1.
            if self.detect_ridges:
                r,score=_ridge_mask(rgb,comp,segments)
                if score>=.55:
                    boundary[r>0]=max(.75,float(score))
            roofprob[comp]=np.maximum(roofprob[comp],confidence)
            instances.append((comp,confidence))
            proposal_info.append(dict(area_work=int(comp.sum()),bbox_work=[x,y,w,hh],
                                      confidence=confidence,material='chromatic_tile',
                                      shape_solidity=float(solidity),chromatic_fraction=float(chromatic)))
        if self.include_neutral:
            for comp,confidence in _neutral_quadrilaterals(rgb,warm.astype(bool),min_work):
                if any(np.count_nonzero(comp&existing)> .25*comp.sum() for existing,_ in instances):
                    continue
                roofprob[comp]=np.maximum(roofprob[comp],confidence)
                instances.append((comp,confidence))
                proposal_info.append(dict(area_work=int(comp.sum()),confidence=confidence,
                                          material='neutral_planar_HYPOTHESIS'))
        # Scale masks with nearest-neighbour, preserve instance boundaries.
        def full(array):
            return cv2.resize(array,(original_w,original_h),interpolation=cv2.INTER_NEAREST) if scale<1 else array
        full_instances=[]
        taken=np.zeros((original_h,original_w),bool)
        for comp,confidence in sorted(instances,key=lambda t:t[1],reverse=True):
            region=full(comp.astype(np.uint8))>0
            region &= ~taken
            if np.count_nonzero(region)<self.min_area:
                continue
            full_instances.append((region,confidence))
            taken|=region
        roofprob=full(roofprob)
        return ProbabilityMaps(np.clip(roofprob,0,1).astype(np.float32),
            full(boundary).astype(np.float32),full(obstacle).astype(np.float32),
            instances=full_instances,
            model_info={'backend':'classical-opencv','weights':'none',
                        'max_side':self.max_side,'scale':scale,
                        'proposal_count':len(full_instances),
                        'candidate_details':proposal_info,
                        'rejected':rejection,
                        'confidence':'UNcalibrated geometric/color heuristic (not P(roof))',
                        'limitations':['chromatic roof tiles preferred',
                          'brick facades/roads may look like roofs',
                          'flat/grey roofs may be missed',
                          'ridge lines and obstacle holes are unreviewed hypotheses',
                          'do not treat proposal masks as pixel-accurate ground truth']}).validate((original_h,original_w))
