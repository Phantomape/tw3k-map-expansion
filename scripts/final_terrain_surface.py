"""Sample the final game DDS in vanilla's coarse-mesh frame: tile coordinate t is LF pixel 4t.

Vanilla's land-mesh vertices and trees sit exactly on LF pixel 8x/DX (rows south-up). The
former (t+.5)*4-.5 pixel-centre basis read 1.5 px (3/16 hex) north-east of them.
"""
from pathlib import Path
import struct
import numpy as np
import campaign_world as _frame
TILE_W,TILE_H=_frame.TILE_W,_frame.TILE_H
LEVEL,NORM,GAIN=_frame.LEVEL,_frame.NORM,_frame.GAIN

def load_final_surface(variant,stage):
 path=Path(stage)/'terrain/campaigns'/variant/'lf_height_map.dds'
 with path.open('rb') as f:header=f.read(128)
 assert header[:4]==b'DDS ';h,w=struct.unpack_from('<2I',header,12);assert (w,h)==(TILE_W*4,TILE_H*4);assert path.stat().st_size==128+w*h*2
 return np.memmap(path,mode='r',dtype='<u2',offset=128,shape=(h,w))[::-1]

def sample_final_surface(field,x,y):
 x,y=np.broadcast_arrays(np.asarray(x,dtype=np.float64),np.asarray(y,dtype=np.float64));assert np.isfinite(x).all() and np.isfinite(y).all()
 h,w=field.shape;u=np.clip(x*(w/TILE_W),0,w-1);v=np.clip(y*(h/TILE_H),0,h-1);u0,v0=u.astype(np.intp),v.astype(np.intp);u1,v1=np.minimum(u0+1,w-1),np.minimum(v0+1,h-1);fu,fv=u-u0,v-v0
 a=field[v0,u0].astype(np.float64);b=field[v0,u1].astype(np.float64);c=field[v1,u0].astype(np.float64);d=field[v1,u1].astype(np.float64);raw=(a*(1-fu)+b*fu)*(1-fv)+(c*(1-fu)+d*fu)*fv
 return LEVEL+(raw/65535-NORM)*GAIN

def point_resample_low_surface(field,chunk_rows=64):
 """The DDS pixels (4i, 4j): sample_final_surface at every integer tile coordinate."""
 out=np.empty((TILE_H,TILE_W),np.float32);x=np.arange(TILE_W,dtype=np.float64)[None,:]
 for start in range(0,TILE_H,chunk_rows):
  end=min(TILE_H,start+chunk_rows);out[start:end]=sample_final_surface(field,x,np.arange(start,end,dtype=np.float64)[:,None])
 return out

def low_edge_limits(original,dx,grade=1.,chunk_rows=64):
 """Triangle inequality bound through the original fine pixel path.

 Each low sample is the DDS pixel (4i, 4j) (point_resample_low_surface). Its
 adjacent sample is four fine edges away. Sum their original per-edge cost allowances.
 """
 result=[];unit=GAIN/65535
 for axis in (0,1):
  h,w=(TILE_H-1,TILE_W) if axis==0 else (TILE_H,TILE_W-1);out=np.zeros((h,w),np.float32);pitch=dx/8*(2/np.sqrt(3) if axis==0 else 1)
  xx=(np.arange(w)*4)[None,:]
  for start in range(0,h,chunk_rows):
   end=min(h,start+chunk_rows);yy=(np.arange(start,end)*4)[:,None];block=out[start:end]
   for k in range(4):
    y0,x0=yy+(k if axis==0 else 0),xx+(k if axis==1 else 0)
    delta=np.abs(original[y0+(axis==0),x0+(axis==1)].astype(np.int32)-original[y0,x0].astype(np.int32))*unit
    block+=np.maximum(grade*pitch,delta)
  result.append(out)
 return result

def display_tile(q,r):
 """Tile coordinate (tx, ty) of hex (q, r)'s display position (campaign_world.display_position),
 where the engine draws its town or character. It lies inside the hex's tile block, drifting
 from the anchor (2q, 2r+odd) at the west/south edge of the map to the far corner at the
 east/north edge. Raw (mesh) position: (tx, ty)*DX/2."""
 x,z=_frame.display_position(q,r);return x/_frame.DX*2,z/_frame.DZ*2

def surface_at_display(surface,q,r):
 """Bilinear height of a tile-resolution surface (analysis/polished-surface-*.npy,
 rows south-up) at hex (q, r)'s display position."""
 tx,ty=display_tile(q,r);h,w=surface.shape;x,y=min(max(tx,0),w-1),min(max(ty,0),h-1);x0,y0=int(x),int(y);x1,y1=min(x0+1,w-1),min(y0+1,h-1);fx,fy=x-x0,y-y0
 return (float(surface[y0,x0])*(1-fx)+float(surface[y0,x1])*fx)*(1-fy)+(float(surface[y1,x0])*(1-fx)+float(surface[y1,x1])*fx)*fy

def snap_grid_to_towns(gx,gz,towns,dx):
 """Move the nearest vertex of a regular (gx,gz) meshgrid onto each town at
 (q*dx, (r+odd/2)*dx). Pass dx=HEX_DX: that is the raw display position, where the
 engine draws the town model (q*DX is only the SW corner of the hex's tile block).

 Budget-coarsened grids (several hexes per step) otherwise put a town inside a
 triangle that tilts with the slopes around it (audit_city_ground_coverage).
 Heights are sampled afterwards at the moved positions, so every vertex stays
 on the final surface. towns: [(q,r)] hex coordinates.
 """
 if gx.shape[0]<2 or gx.shape[1]<2:return
 x0,z0=gx[0,0],gz[0,0];sx,sz=gx[0,1]-x0,gz[1,0]-z0
 for q,r in towns:
  tx,tz=q*dx,(r+q%2*.5)*dx;i,j=round((tz-z0)/sz),round((tx-x0)/sx)
  if 0<=i<gx.shape[0] and 0<=j<gx.shape[1]:gx[i,j],gz[i,j]=tx,tz
