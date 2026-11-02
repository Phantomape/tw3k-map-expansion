"""Remove inherited canvas-edge jumps while keeping town plots and anchors.

Project narrow strips around the translated vanilla rectangle. Original slopes
remain allowed everywhere except the actual four artificial boundary edges.
"""
import json,struct,hashlib
from pathlib import Path
import numpy as np
from PIL import Image,ImageFilter
from analyze_reference import ROOT,digest
import campaign_world as _frame
from campaign_world import DX,DZ,W,H
from final_terrain_surface import LEVEL,NORM,GAIN
from fastops import cm_write   # vectorised native_campaign_climate.encode_map: byte-identical
from campaign_world import TERRAIN_VARIANTS

A=_frame.WORKSPACE;S=A/'work/release-input';OUT=A/'analysis/native-canvas-boundary-smoothing.json'

def envelope(values,cost_x,cost_z,max_sweeps=32):
    out=np.array(values,np.float32,copy=True)
    # The cumulative costs do not change between sweeps: build them once.
    coords=[]
    for axis,cost in ((1,cost_x),(0,cost_z)):
        shape=list(out.shape);shape[axis]=1
        coords.append((axis,np.concatenate((np.zeros(shape,np.float64),np.cumsum(cost,axis=axis,dtype=np.float64)),axis=axis)))
    # Each 1-D pass is a deterministic function of its line (row on axis 1, column on axis 0):
    # a line whose input equals its input at the previous pass on that axis gets that pass's
    # output back and only the other lines are recomputed. Every sweep's array, the
    # convergence test and the sweep cap are exactly those of the whole-array passes (the
    # anchored envelopes run to the cap on a few hundred ulp-level lines).
    seen={}
    for sweep in range(max_sweeps):
        old=out.copy()
        for axis,coord in coords:
            if axis in seen:
                prior_in,prior_out=seen[axis];lines=np.flatnonzero((out!=prior_in).any(axis=axis));new=prior_out.copy()
            else:lines=np.arange(out.shape[1-axis]);new=np.empty_like(out)
            if len(lines):
                sel=(lines,slice(None)) if axis==1 else (slice(None),lines);value=out[sel];c=coord[sel]
                forward=np.minimum.accumulate(value-c,axis=axis)+c
                backward=np.flip(np.minimum.accumulate(np.flip(value+c,axis=axis),axis=axis),axis=axis)-c
                new[sel]=np.minimum(forward,backward).astype(np.float32)
            seen[axis]=(out,new);out=new
        if np.array_equal(old,out) or (np.isfinite(old).all() and np.max(old-out)<=1e-7):break
    return out

def project(surface,fixed,cost_x,cost_z):
    # The four envelopes are independent; numpy releases the GIL in accumulate.
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(4) as pool:
        low,high,below,above=pool.map(lambda v:envelope(v,cost_x,cost_z),(np.where(fixed,-surface,np.inf),np.where(fixed,surface,np.inf),-surface,surface))
    lower=-low
    upper=high
    if np.any(lower>upper+.001):raise RuntimeError('Boundary anchors incompatible with a continuous slope')
    proposed=.5*(-below+above)
    result=np.clip(proposed,lower,upper);result[fixed]=surface[fixed]
    # Pairs of fixed pixels are kept verbatim; their slope may exceed the cost
    # by up to one u16 height step, which is not a convergence failure.
    free_x=~(fixed[:,1:]&fixed[:,:-1]);free_z=~(fixed[1:]&fixed[:-1])
    err=max(float(np.where(free_x,np.abs(np.diff(result,axis=1))-cost_x,0).max()),float(np.where(free_z,np.abs(np.diff(result,axis=0))-cost_z,0).max()))
    if err>1e-4:raise RuntimeError(f'Boundary projection did not converge: {err}')
    return result

def town_guard():
    data=(A/'work/final-map/map.hex').read_bytes()
    cells=np.frombuffer(data[-4-W*H*16:-4],np.uint8).reshape(H,W,16)
    plots=((cells[:,:,2]>>4)>0)|((cells[:,:,3]&1)>0)
    plots=np.asarray(Image.fromarray(np.uint8(plots)*255).filter(ImageFilter.MaxFilter(5)))>0
    yy,xx=np.mgrid[:H*2+1,:W*2];q,r=_frame.tile_hex(xx,yy)
    low=plots[r,q]
    return np.asarray(Image.fromarray(np.uint8(low)).resize((W*8,(H*2+1)*4),Image.Resampling.NEAREST))>0

def smooth_variant(variant):
    # The 536 MB height map is read once; both hashes come from the bytes in memory.
    path=S/f'terrain/campaigns/{variant}/lf_height_map.dds';data=path.read_bytes();source_hash=hashlib.sha256(data).hexdigest()
    prior=json.loads(OUT.read_text()) if OUT.exists() else {}
    context={'map':digest(A/'work/final-map/map.hex'),
             'design':digest(A/'work/current-region-design.json'),
             'implementation':digest(Path(__file__))}
    row=prior.get('variants',{}).get(variant,{})
    if prior.get('context')==context and row.get('output_sha256')==source_hash:
        print(variant,'boundary height correction already current',flush=True);return
    h,w=struct.unpack_from('<2I',data,12);assert (w,h)==(_frame.LF_W,_frame.LF_H)
    raw=np.frombuffer(data,'<u2',offset=128).reshape(h,w)[::-1].copy()
    surface=LEVEL+(raw.astype(np.float32)/65535-NORM)*GAIN
    original=surface.copy();guard=town_guard();radius=128;changes=[]
    # The known discontinuity is between column/row edge-1 and edge.
    def boundary(axis,edge):
        selection=(slice(None),slice(edge-radius,edge+radius)) if axis==1 else (slice(edge-radius,edge+radius),slice(None))
        block=surface[selection].copy();fixed=guard[selection].copy()
        fixed[:2]=True;fixed[-2:]=True;fixed[:,:2]=True;fixed[:,-2:]=True
        cx=np.maximum(DX/8,np.abs(np.diff(original[selection],axis=1)))
        cz=np.maximum(DZ/8,np.abs(np.diff(original[selection],axis=0)))
        if axis==1:cx[:,radius-1]=DX/8
        else:cz[radius-1,:]=DZ/8
        before=float(np.abs(np.diff(block,axis=axis)).take(radius-1,axis=axis).max())
        # polish_campaign_terrain.py now blends the native edge itself (carried
        # native height, meandering ring, near-border town datum). Where that
        # leaves anchors this strip projection cannot join, keep the polished
        # surface; audit_native_edges.py gates any wall left on the border.
        try:projected=project(block,fixed,cx,cz)
        except RuntimeError as error:
            return None,{'axis':axis,'pixel_edge':edge,'skipped':str(error),'maximum_jump_before':before}
        after=float(np.abs(np.diff(projected,axis=axis)).take(radius-1,axis=axis).max())
        return (selection,projected),{'axis':axis,'pixel_edge':edge,'maximum_jump_before':before,'maximum_jump_after':after}
    # The two column strips (then the two row strips) are disjoint and each reads only its
    # own pixels: project each pair concurrently, write them in the serial order.
    from concurrent.futures import ThreadPoolExecutor
    x0,x1,y0,y1=_frame.NATIVE_OFFSET_LF[0],_frame.NATIVE_OFFSET_LF[0]+_frame.NATIVE_LF_W,_frame.NATIVE_OFFSET_LF[1],_frame.NATIVE_OFFSET_LF[1]+_frame.NATIVE_LF_H
    assert x1-x0>=2*radius and y1-y0>=2*radius
    with ThreadPoolExecutor(2) as pool:
        for pair in (((1,x0),(1,x1)),((0,y0),(0,y1))):
            for done,change in list(pool.map(lambda job:boundary(*job),pair)):
                if done is not None:surface[done[0]]=done[1]
                changes.append(change)
    # The northern Yiwu farm approach exposes the last native LF rows beyond
    # their original coarse coverage. Their inherited pixel slopes must not
    # survive as a saw-toothed ridge against the new flat exterior. Limit the
    # junction in both axes; preserve plot guards and the surrounding artwork.
    design=json.loads((A/'work/current-region-design.json').read_text(encoding='utf-8'))
    farm=next(c for c in design['regions'] if c['region']=='tw3k_hist_yiwu_farm')
    q,r=farm['town'];north=int((_frame.PAD_BOTTOM+_frame.NATIVE_H+.5)*_frame.LF_PX_PER_HEX)  # native north edge, LF rows (was literal 5876)
    per=_frame.LF_PX_PER_HEX
    if abs(r-north/per)<16:
        left=(q-16)*per;right=(q+18)*per;bottom=north-128;top=north+128
        selection=(slice(bottom,top),slice(left,right));block=surface[selection].copy();fixed=guard[selection].copy()
        fixed[:2]=True;fixed[-2:]=True;fixed[:,:2]=True;fixed[:,-2:]=True
        cx=np.maximum(DX/8,np.abs(np.diff(block,axis=1)));cz=np.maximum(DZ/8,np.abs(np.diff(block,axis=0)))
        core=(slice(north-16-bottom,north+24-bottom),slice((q-3)*per-left,(q+7)*per-left))
        # The farm moved with the Western Regions re-layout: if its guards cannot be
        # joined at the gentlest grade, allow progressively steeper (still
        # continuous) slopes rather than leave the rim cliff.
        projected=None
        for grade in (1.,2.,4.,8.):
            gx,gz=cx.copy(),cz.copy();gx[core]=grade*DX/8;gz[core]=grade*DZ/8
            try:projected=project(block,fixed,gx,gz);break
            except RuntimeError:continue
        if projected is None:changes.append({'zone':'yiwu_farm_north_rim','skipped':'guards cannot be joined at 8x the base grade'})
        else:
         surface[selection]=projected
         changes.append({'zone':'yiwu_farm_north_rim','maximum_correction':float(np.max(np.abs(projected-block))),'town_guards_preserved':bool(np.array_equal(projected[fixed],block[fixed])),'grade':grade})
    encoded=np.rint(np.clip(NORM+(surface-LEVEL)/GAIN,0,1)*65535).astype('<u2')
    encoded[guard]=raw[guard];assert np.array_equal(encoded[guard],raw[guard])
    changed=encoded!=raw
    backup=A/'terrain-repair/boundary-height-before'/source_hash
    backup.mkdir(parents=True,exist_ok=True);p=backup/(variant+'.dds')
    if not p.exists():p.write_bytes(data)
    out=data[:128]+np.ascontiguousarray(encoded[::-1],dtype='<u2').tobytes();path.write_bytes(out);output_hash=hashlib.sha256(out).hexdigest();del out
    cm=path.with_suffix('.compressed_map');cm.write_bytes(cm_write(cm.read_bytes(),encoded[::-1]))
    low_changed=changed.reshape(_frame.TILE_H,4,_frame.TILE_W,4).any(axis=(1,3))
    alpha_path=A/f'analysis/polished-edit-alpha-{variant}.npy';alpha=np.load(alpha_path);alpha[low_changed]=np.maximum(alpha[low_changed],.03);np.save(alpha_path,alpha)
    row={'input_sha256':source_hash,'output_sha256':output_hash,'changed_pixels':int(changed.sum()),'town_plot_aprons_byte_preserved':True,'boundaries':changes}
    rows=prior.get('variants',{}) if prior.get('context')==context else {};rows[variant]=row
    OUT.write_text(json.dumps({'context':context,'variants':rows,'live_verified':False},indent=2),encoding='utf-8')
    print(variant,'native canvas boundary correction:',json.dumps(row),flush=True)

if __name__=='__main__':
    for variant in TERRAIN_VARIANTS:smooth_variant(variant)
