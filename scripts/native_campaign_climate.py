"""Inherit climate from the unmodified native map inside its translated bounds."""
import math
import numpy as np
from native_height_field import parse, decode, rebuild
import campaign_world as _frame
import biome_profiles as _bp
_BAND_WOBBLE = _bp.region('authored_bands')['wobble']   # amplitude, frequencies of the band edges' q/r wobble

def decode_map(data):
    """(h, w) uint16 grid of a TABLE_INDEXED map: the per-block decode loop, vectorised (fastops.cm_read)."""
    return _fastops.cm_read(data)

def inherit_climate(biome, native_cm, native_tiles, read_asset):
    native=decode_map(native_cm)[::-1]
    assert native.shape==(_frame.NATIVE_TILE_H,_frame.NATIVE_TILE_W) and np.all(native<=3)
    # Native CM values differ from tile-list climates at many cells. Use
    # actual placement climates wherever those placements cover a cell.
    import re
    from read_tile_list import parse as tile_list
    from native_tile_dimensions import decode_dimensions
    paths,climates,_,_,records,_=tile_list(native_tiles)
    assert climates==['sub_tropical','cold','temperate','arid']
    dims={};native=native.copy()
    for i,path in enumerate(paths):
        m=re.search(r'generic_(\d+)x(\d+)',path)
        if m: dims[i]=tuple(map(int,m.groups()));continue
        bits=path.strip(chr(92)).split(chr(92))
        key='terrain/tiles/campaign/_tile_database/tiles/'+bits[-2]+'_'+bits[-1]+'.bin'
        try: dims[i]=decode_dimensions(read_asset(key))
        # An unknown asset does not justify painting an arbitrary 8x8 block.
        # Its explicit origin climate is the only confirmed sample.
        except KeyError: dims[i]=(1,1)
    for rec in records:
        from native_tile_dimensions import placement_footprint
        w,h=placement_footprint(dims[rec[1]],rec[5]);x,y=rec[3:5]   # quarter turns swap w/h
        x0,y0,x1,y1=max(0,x),max(0,y),min(_frame.NATIVE_TILE_W,x+w),min(_frame.NATIVE_TILE_H,y+h)
        if x0<x1 and y0<y1:native[y0:y1,x0:x1]=rec[2]
    result=biome.copy();result[_frame.NATIVE_TILE_SLICE]=native
    return extend_native_climate(result)

def land_tile_coverage(tile_list_bytes, read_asset, shape):
    """Cells covered by any non-sea fine tile (rotated footprints) in a canvas-frame tile list."""
    import re
    from read_tile_list import parse as tile_list
    from native_tile_dimensions import decode_dimensions, placement_footprint
    paths,_,_,_,records,_=tile_list(tile_list_bytes);dims={};cover=np.zeros(shape,bool)
    for rec in records:
        path=paths[rec[1]]
        if 'generic_sea' in path:continue
        if rec[1] not in dims:
            m=re.search(r'generic_(\d+)x(\d+)',path)
            if m:dims[rec[1]]=tuple(map(int,m.groups()))
            else:
                bits=path.strip(chr(92)).split(chr(92))
                try:dims[rec[1]]=decode_dimensions(read_asset('terrain/tiles/campaign/_tile_database/tiles/'+bits[-2]+'_'+bits[-1]+'.bin'))
                except KeyError:dims[rec[1]]=(1,1)
        w,h=placement_footprint(dims[rec[1]],rec[5])
        cover[max(0,rec[4]):max(0,rec[4]+h),max(0,rec[3]):max(0,rec[3]+w)]=True
    return cover

import fastops as _fastops

def box_blur(a, radius):
    """Separable box blur of a 2-D float array (edges clamped).

    Runs in parallel strips (fastops.box_blur); identical to _box_blur_reference."""
    return _fastops.box_blur(a, radius)

def _box_blur_reference(a, radius):
    out=a.astype(np.float32)
    for axis in (0,1):
        pad=[(0,0),(0,0)];pad[axis]=(radius+1,radius);p=np.pad(out,pad,mode='edge');c=np.cumsum(p,axis=axis,dtype=np.float64)
        hi=np.take(c,np.arange(2*radius+1,c.shape[axis]),axis=axis);lo=np.take(c,np.arange(0,c.shape[axis]-2*radius-1),axis=axis)
        out=((hi-lo)/(2*radius+1)).astype(np.float32)
    return out

def band_climate(q, r, bands, origin):
    """Authored latitude/longitude climate bands for canvas land, with wavy edges.

    q, r are canvas hex coordinates (floats allowed); bands and origin come from
    canvas-authored-coordinates.json, so the bands migrate with the canvas.
    Returns the climate grid and the wavy r used, for forest rules keyed on latitude.
    """
    dq=np.asarray(q,np.float32)-origin[0];dr=np.asarray(r,np.float32)-origin[1]
    (a0,f0),(a1,f1,g1)=_BAND_WOBBLE['q'];(b0,h0),(b1,h1,k1)=_BAND_WOBBLE['r']
    qw=np.asarray(q,np.float32)+a0*np.sin(dr*f0)+a1*np.sin(dr*f1+dq*g1)
    rw=np.asarray(r,np.float32)+b0*np.sin(dq*h0)+b1*np.sin(dq*h1-dr*k1)
    out=np.full(np.shape(qw),2,np.uint16)
    out[(qw<bands['arid_max_q'])&(rw>bands['arid_min_r'])]=3
    out[rw<bands['tropical_max_r']]=0
    out[(rw>bands['cold_min_r'])&(qw>bands['cold_min_q'])]=1
    return out,qw,rw

# Canvas land next to the native map continues the native climate; the authored
# bands only take over beyond this distance (in tile cells), along a wavy front.
NATIVE_CLIMATE_REACH=48
NATIVE_CLIMATE_WAVE=45
NATIVE_CLIMATE_MIN_REACH=20   # cells: the front never comes back to the straight native border
NATIVE_CLIMATE_INSET=4
NATIVE_CLIMATE_SHIFT=24   # cells: noise displacement of the carried native climate

def propagate_nearest(values, seed, steps):
    """Carry seed values outward by 4-neighbour growth; returns (values, steps taken).

    Cells never reached keep distance int32 max. Frontier BFS (fastops); same
    growth order and direction priority as _propagate_nearest_reference.
    """
    return _fastops.propagate_nearest(values, seed, steps)

def _propagate_nearest_reference(values, seed, steps):
    th,tw=values.shape;big=np.iinfo(np.int32).max
    near=np.where(seed,values,0).astype(values.dtype);dist=np.where(seed,0,big).astype(np.int32)
    for step in range(1,steps+1):
        grown=dist<step;changed=False
        for dy,dx in ((0,1),(0,-1),(1,0),(-1,0)):
            src=np.zeros((th,tw),bool);val=np.zeros((th,tw),values.dtype)
            ys=slice(max(0,dy),th+min(0,dy));yd=slice(max(0,-dy),th+min(0,-dy))
            xs=slice(max(0,dx),tw+min(0,dx));xd=slice(max(0,-dx),tw+min(0,-dx))
            src[yd,xd]=grown[ys,xs];val[yd,xd]=near[ys,xs]
            new=src&(dist==big)
            if new.any():near[new]=val[new];dist[new]=step;changed=True
        if not changed:break
    return near,dist

def extend_native_climate(result):
    """Carry each native edge climate outward instead of meeting a band in a straight line."""
    th,tw=result.shape;inside=np.zeros((th,tw),bool);inside[_frame.NATIVE_TILE_SLICE]=True
    # The outermost native rows alternate climates cell by cell; seed from a few cells in.
    inset=NATIVE_CLIMATE_INSET;seed=np.zeros((th,tw),bool)
    seed[_frame.NATIVE_TILE_Y0+inset:_frame.NATIVE_TILE_Y1-inset,_frame.NATIVE_TILE_X0+inset:_frame.NATIVE_TILE_X1-inset]=True
    steps=NATIVE_CLIMATE_REACH+NATIVE_CLIMATE_WAVE+inset;near,dist=propagate_nearest(result,seed,steps)
    # Noise is measured from the native corner so a canvas migration moves it with the land.
    yy,xx=np.indices((th,tw),dtype=np.float32);x0,y0=_frame.NATIVE_OFFSET_TILE;xx-=x0;yy-=y0
    # Fractal noise, not two long sines: the sine front (wavelength ~200 cells)
    # read as a straight north-south sand/grass line once the canvas ground
    # followed the climate (live check 2026-10-20 e, west edge 714 and 834).
    from terrain_noise import fbm as _fbm
    wave=NATIVE_CLIMATE_WAVE*1.6*_fbm(xx/2,yy/2,(0,0),35,4,seed=85)
    # The native outer ring (inset cells) takes the inner climate too: it used to be the
    # hidden map edge and its alternating values now sit inside the canvas.
    # The minimum reach meanders too: a fixed one was a straight line wherever the
    # wave dipped below it (west edge columns 647-648, offline QA 2026-10-20).
    min_reach=NATIVE_CLIMATE_MIN_REACH*(.6+.8*np.clip(.5+.7*_fbm(xx/2,yy/2,(0,0),8,3,seed=86),0,1))
    take=(~seed)&(dist<=np.maximum(min_reach,NATIVE_CLIMATE_REACH+wave)+inset)
    # Nearest-value carrying extruded the blocky native edge climates into long
    # rectangles (straight lines and right angles south of Dunhuang, 2048 canvas):
    # read the carried field at a noise-displaced cell instead.
    from terrain_noise import fbm
    # No displacement at the border itself (both sides must agree there); it
    # grows to NATIVE_CLIMATE_SHIFT over the carried reach.
    shift=NATIVE_CLIMATE_SHIFT*np.clip(dist/NATIVE_CLIMATE_REACH,0,1)
    sx=np.clip(np.rint(xx+x0+shift*fbm(xx/2,yy/2,(0,0),16,3,seed=81)),0,tw-1).astype(np.int32)
    sy=np.clip(np.rint(yy+y0+shift*fbm(xx/2,yy/2,(0,0),16,3,seed=82)),0,th-1).astype(np.int32)
    # The displaced cell can lie beyond the BFS reach (near=0 there, read as sub_tropical):
    # keep the undisplaced carried value for those.
    out=result.copy();out[take]=np.where(dist[sy,sx]<=steps,near[sy,sx],near)[take]
    return out

def encode_map(data,grid):
    h,w=grid.shape; padded=np.pad(grid,((0,(-h)%16),(0,(-w)%16)),mode='edge');blocks=[]
    import struct
    for y in range(0,padded.shape[0],16):
        for x in range(0,padded.shape[1],16):
            v=np.ascontiguousarray(padded[y:y+16,x:x+16],dtype='<u2')
            blocks.append(b'\0'+struct.pack('<H',int(v[0,0])) if np.all(v==v[0,0]) else b'\x8f'+v.tobytes())
    head=bytearray(data);struct.pack_into('<2I',head,10,w,h)
    return rebuild(head,blocks)
