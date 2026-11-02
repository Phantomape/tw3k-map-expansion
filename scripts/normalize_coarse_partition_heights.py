"""Seat edited cut vertices on the final DDS and preserve native coarse relief."""
import hashlib,json,struct
from pathlib import Path
import numpy as np
from analyze_reference import ROOT,digest
import campaign_world as _frame
from campaign_world import DX,W,H
from final_terrain_surface import load_final_surface,sample_final_surface
from repair_local_coarse_seams import native_reference,STEP
from repair_campaign_river_roots import Assets
from coarse_mesh_skirts import skirt_offsets
from native_edit_mask import at_world
from campaign_world import TERRAIN_VARIANTS

A=_frame.WORKSPACE;S=A/'work/release-input'
def sample(grid,x,z):
 x=np.clip(x,0,grid.shape[1]-1);z=np.clip(z,0,grid.shape[0]-1)
 x0,z0=x.astype(int),z.astype(int);x1=np.minimum(x0+1,grid.shape[1]-1);z1=np.minimum(z0+1,grid.shape[0]-1);fx=x-x0;fz=z-z0
 return grid[z0,x0]*(1-fx)*(1-fz)+grid[z0,x1]*fx*(1-fz)+grid[z1,x0]*(1-fx)*fz+grid[z1,x1]*fx*fz

# Points with both coordinates on the lattice: lattice_key is then (('L',kx),('L',kz))
# with the same indices as lattice_keys, so they are kept as sorted int64 codes
# and looked up in numpy. Any other point keeps its lattice_key in a dict.
def lattice_codes(xz):
 keys=_frame.lattice_keys(xz);both=(keys[:,0]==1)&(keys[:,2]==1)
 assert (np.abs(keys[both,3])<1<<30).all()
 return both,keys[:,1]*(1<<31)+keys[:,3]

for variant in TERRAIN_VARIANTS:
 native=Assets(original_only=True);prefix=f'terrain/campaigns/{variant}/global_meshes/land_mesh_';known={};native_points=[]
 for key in sorted(k for k in native.index if k.startswith(prefix) and k.endswith('.rigid_model_v2')):
  b=native.read(key);mp=struct.unpack_from('<I',b,152)[0];_,_,_,vo,vc,io,ic=struct.unpack_from('<HH5I',b,mp)
  v=np.frombuffer(b,'<f4',offset=mp+vo,count=vc*4).reshape(vc,4).copy();v[:,0]+=_frame.NATIVE_SHIFT_RAW[0];v[:,2]+=_frame.NATIVE_SHIFT_RAW[1]
  i=np.frombuffer(b,'<u2',offset=mp+io,count=ic).reshape(-1,3);t=v[i,:3];area=(t[:,1,0]-t[:,0,0])*(t[:,2,2]-t[:,0,2])-(t[:,2,0]-t[:,0,0])*(t[:,1,2]-t[:,0,2]);used=np.unique(i[(area>1e-8)&(t[:,:,1].min(1)>-5)])
  native_points.append(v[used])
 native_points=np.concatenate(native_points);both,codes=lattice_codes(native_points[:,[0,2]])
 for p in native_points[~both]:
  k=_frame.lattice_key(p[0],p[2]);known[k]=max(known.get(k,-float('inf')),float(p[1]))
 order=np.argsort(codes[both],kind='stable');sorted_codes=codes[both][order];sorted_y=native_points[both,1][order].astype(np.float64)
 starts=np.r_[0,1+np.flatnonzero(sorted_codes[1:]!=sorted_codes[:-1])];known_codes=sorted_codes[starts];known_y=np.maximum.reduceat(sorted_y,starts)
 ref=native_reference(variant,int(_frame.WORLD_WIDTH/STEP)+3,(H+1)*4+3);field=load_final_surface(variant,S);rows=[]
 for path in sorted((S/f'terrain/campaigns/{variant}/global_meshes').glob('land_mesh_*.rigid_model_v2')):
  source=path.read_bytes();b=bytearray(source);mp=struct.unpack_from('<I',b,152)[0];_,_,_,vo,vc,io,ic=struct.unpack_from('<HH5I',b,mp)
  v=np.frombuffer(b,'<f4',offset=mp+vo,count=vc*4).reshape(vc,4);used=np.zeros(vc,bool);used[np.frombuffer(b,'<u2',offset=mp+io,count=ic)]=True
  # One shared edit definition: seat edited vertices on the final surface and
  # give every other native vertex its vanilla height (no in-between band).
  edited=at_world(variant,v[:,0],v[:,2]*_frame.DZ/DX);target=v[:,1].copy();off=skirt_offsets(v,np.frombuffer(b,'<u2',offset=mp+io,count=ic));edit=used&edited&(v[:,1]>-40)
  target[edit]=sample_final_surface(field,v[edit,0]/DX*2,v[edit,2]/DX*2)+off[edit]
  native_at=np.flatnonzero(used&~edited&(v[:,1]>-5));both,codes=lattice_codes(v[native_at][:,[0,2]])
  slot=np.searchsorted(known_codes,codes).clip(0,len(known_codes)-1);hit=both&(known_codes[slot]==codes)
  # float32 + float32, as the scalar float + float32 below (NumPy 2 weak scalars).
  value=known_y[slot[hit]].astype(np.float32);ok=np.isfinite(value);target[native_at[hit][ok]]=value[ok]+off[native_at[hit][ok]]
  # The rest: an off-lattice point takes the vanilla height at its position key when there is
  # one; otherwise a point within .001 of a lattice node reads the native reference raster there.
  # float32 throughout, as the per-point form (float32 / float, round half-even = rint, weak scalars).
  rest=native_at[~hit];old=np.full(len(rest),np.nan,np.float32);looked=np.zeros(len(rest),bool)
  for n in np.flatnonzero(~both[~hit]):
   p=v[rest[n]];k=known.get(_frame.lattice_key(p[0],p[2]))
   if k is not None:old[n]=k;looked[n]=True
  free=np.flatnonzero(~looked);x,z=v[rest[free],0]/STEP,v[rest[free],2]/STEP;kx,kz=np.rint(x),np.rint(z)
  on=(np.abs(x-kx)<.001)&(np.abs(z-kz)<.001)&(0<=kx)&(kx<ref.shape[1])&(0<=kz)&(kz<ref.shape[0])
  old[free[on]]=ref[kz[on].astype(int),kx[on].astype(int)]
  # known/ref hold the surface (top) height; keep skirt vertices below it.
  ok=np.isfinite(old);target[rest[ok]]=old[ok]+off[rest[ok]]
  changed=used&(abs(target-v[:,1])>1e-5)
  if not changed.any():continue
  backup=A/'terrain-repair/coarse-height-before'/hashlib.sha256(source).hexdigest()/variant/path.name;backup.parent.mkdir(parents=True,exist_ok=True)
  if not backup.exists():backup.write_bytes(source)
  v[changed,1]=target[changed];bb=list(struct.unpack_from('<6f',b,mp+24));bb[1]=min(bb[1],float(v[:,1].min()));bb[4]=max(bb[4],float(v[:,1].max()));struct.pack_into('<6f',b,mp+24,*bb)
  tmp=path.with_suffix('.height-repair.tmp');tmp.write_bytes(b);tmp.replace(path)
  rows.append({'mesh':path.name,'changed_vertices':int(changed.sum()),'after':digest(path)})
 (A/f'analysis/coarse-partition-height-normalization-{variant}.json').write_text(json.dumps({'variant':variant,'meshes':rows,'lf_pixels_unchanged':True,'live_verified':False},indent=2),encoding='utf-8')
 print(variant,'normalized edited intersections and native relief:',len(rows),'meshes',flush=True)
