"""Fill coarse terrain cutouts using the synchronized local height fields.

Native mesh names/materials stay. Existing surface outside edited land remains.
"""
import json,struct,math,shutil,os,ctypes
from pathlib import Path
import numpy as np
from PIL import Image,ImageFilter,ImageDraw
from concurrent.futures import ProcessPoolExecutor
from analyze_reference import ROOT,read_hex
import campaign_world as _frame
from coarse_mesh_skirts import seat_on_surface,open_sea_faces
from native_edit_mask import edited_hexes
from final_terrain_surface import load_final_surface,sample_final_surface,snap_grid_to_towns
from native_height_field import parse,decode,rebuild
from terrain_checkpoint import StepSource,inputs
from analyze_reference import digest
from campaign_world import DX,W,H
from campaign_world import TERRAIN_VARIANTS
from campaign_world import LEVEL as _LEVEL,NORM as _NORM,GAIN as _GAIN
from fastops import cm_write
import mesh_lattice as ml
A=_frame.WORKSPACE;S=A/'work/release-input';BACK=A/'before-coarse-gap-fill-native-v2'
def edited(x,z):
 """Raw (mesh) points: the shared LF edit flag of their tile (native_edit_mask.at_world's rule on the
 hexes edited_hexes gives, computed once in the parent), or the dilated replacement mask."""
 tx=np.floor(_frame.tile_coord(x)+.5).astype(int).clip(0,replacement.shape[1]-1);ty=np.floor(_frame.tile_coord(z)+.5).astype(int).clip(0,replacement.shape[0]-1);out=replacement[ty,tx]
 rest=~out;q,r=_frame.world_hex(x[rest],z[rest]*_frame.DZ/DX);out[rest]=hexes[r,q]   # elementwise: looked up only where not replaced
 return out
from fastops import cm_read as cm_grid   # the per-block native_height_field decode loop, vectorised
def dilate(mask,size):
 """Image MaxFilter(size) of a boolean mask (square window clipped at the edges),
 as two 1-D window sums: same result, about 6x faster on the full tile grid."""
 k=size//2;out=np.asarray(mask,bool)
 for axis in (0,1):
  n=out.shape[axis];pad=[(0,0),(0,0)];pad[axis]=(k+1,k);cs=np.cumsum(np.pad(out,pad).astype(np.int32),axis=axis)
  out=(np.take(cs,np.arange(2*k+1,2*k+1+n),axis=axis)-np.take(cs,np.arange(n),axis=axis))>0
 return out
def sample(a,x,y):
 x=x.clip(0,a.shape[1]-1);y=y.clip(0,a.shape[0]-1);xi=x.astype(int);yi=y.astype(int);nx=np.minimum(xi+1,a.shape[1]-1);ny=np.minimum(yi+1,a.shape[0]-1);fx=x-xi;fy=y-yi
 return (a[yi,xi]*(1-fx)+a[yi,nx]*fx)*(1-fy)+(a[ny,xi]*(1-fx)+a[ny,nx]*fx)*fy
# Replacement lattices are counted in half-hex units (DX/2): a model on step k
# has nodes at multiples of k*DX/2, so equal steps give identical nodes.
def grid_rect(bb,k):
 st=k*DX/2
 return (k*math.floor(max(0,bb[0]-DX*2)/st),k*math.ceil(min(W*DX,bb[3]+DX*2)/st),k*math.floor(max(0,bb[2]-DX*2)/st),k*math.ceil(min(_frame.RAW_MESH_HEIGHT,bb[5]+DX*2)/st))
def _special(G):
 d=np.zeros(G.shape,bool)
 d[1:]|=G[1:]!=G[:-1];d[:-1]|=G[:-1]!=G[1:];d[:,1:]|=G[:,1:]!=G[:,:-1];d[:,:-1]|=G[:,:-1]!=G[:,1:]
 return d
def graded_grid(k,rect,G,towns):
 """Vertices (x, z) and faces of a k-step grid refined to G per cell.

 Plain cells keep the two-triangle split of the uniform grid. A cell next to a
 finer one gets that cell's edge nodes and is fanned from its centre, so the
 grid has no T-junctions. Node keys are quarter-hex integers (fan centres of
 odd steps sit on quarter hexes); node order is row-major as np.meshgrid.
 Also returns the first face of every square split in two, (A,C,D),(A,B,C)
 (see split_squares)."""
 ux0,ux1,uz0,uz1=rect;ncz,ncx=G.shape;special=(G<k)|_special(G)
 cz,cx=np.nonzero(~special);X=2*(ux0+cx*k);Z=2*(uz0+cz*k);d=2*k
 A,B,C,D=np.stack((X,Z),1),np.stack((X+d,Z),1),np.stack((X+d,Z+d),1),np.stack((X,Z+d),1)
 tris=[np.stack((A,C,D),1),np.stack((A,B,C),1)];tris=[np.stack(tris,1).reshape(-1,3,2)]
 extra=[];quads=[2*t for t in range(len(cz))]
 for j,i in zip(*np.nonzero(special)):
  r=int(G[j,i]);n=k//r;x=ux0+i*k;z=uz0+j*k
  side={'s':math.gcd(r,int(G[j-1,i])) if j>0 else r,'n':math.gcd(r,int(G[j+1,i])) if j<ncz-1 else r,'w':math.gcd(r,int(G[j,i-1])) if i>0 else r,'e':math.gcd(r,int(G[j,i+1])) if i<ncx-1 else r}
  for bz in range(n):
   for ax in range(n):
    sx,sz=x+ax*r,z+bz*r;es=side['s'] if bz==0 else r;ee=side['e'] if ax==n-1 else r;en=side['n'] if bz==n-1 else r;ew=side['w'] if ax==0 else r
    if es==ee==en==ew==r:
     a_,b_,c_,d_=(2*sx,2*sz),(2*sx+2*r,2*sz),(2*sx+2*r,2*sz+2*r),(2*sx,2*sz+2*r);quads.append(2*len(cz)+len(extra));extra+=[(a_,c_,d_),(a_,b_,c_)];continue
    ring=[(2*(sx+t),2*sz) for t in range(0,r,es)]+[(2*(sx+r),2*(sz+t)) for t in range(0,r,ee)]+[(2*(sx+r-t),2*(sz+r)) for t in range(0,r,en)]+[(2*sx,2*(sz+r-t)) for t in range(0,r,ew)]
    o=(2*sx+r,2*sz+r);extra+=[(o,ring[t],ring[(t+1)%len(ring)]) for t in range(len(ring))]
 if extra:tris.append(np.asarray(extra,np.int64).reshape(-1,3,2))
 tris=np.concatenate(tris).astype(np.int64);BIG=1<<32
 keys=tris[:,:,1]*BIG+tris[:,:,0];nodes,inv=np.unique(keys,return_inverse=True)
 qx=(nodes%BIG).astype(np.float64);qz=(nodes//BIG).astype(np.float64);gx=qx*DX/4;gz=qz*DX/4
 # Budget-coarsened lattices (step above one hex): the nearest node of the
 # lattice at the town takes the town's raw display position (HEX_DX, where the
 # engine draws the model; q*DX is the block corner), as snap_grid_to_towns.
 # Models sharing an overlap lattice move the same node.
 for q,rr in towns:
  tx,tz=q*_frame.HEX_DX,(rr+q%2*.5)*_frame.HEX_DX;ux,uz=tx/(DX/2),tz/(DX/2)
  i,j=math.floor((ux-ux0)/k),math.floor((uz-uz0)/k)
  if not(0<=i<ncx and 0<=j<ncz):
   i,j=round((ux-ux0)/k),round((uz-uz0)/k)
   if not(0<=i<=ncx and 0<=j<=ncz):continue
   lat=k
  else:lat=int(G[j,i]) if special[j,i] else k
  if lat*DX/2<=DX:continue
  key=2*(uz0+round((uz-uz0)/lat)*lat)*BIG+2*(ux0+round((ux-ux0)/lat)*lat);at=np.searchsorted(nodes,key)
  if at<len(nodes) and nodes[at]==key:gx[at],gz[at]=tx,tz
 return gx,gz,inv.reshape(-1,3),np.asarray(quads,np.int64)
def split_squares(faces,gy,quads):
 """Split each square on the diagonal whose ends differ least in height.

 One fixed diagonal cut lake floors into triangles with parallel straight
 deep/shallow edges (live 2048j, Yangtong lake). The choice reads only the
 square's own four node heights, sampled from the shared final surface, so
 models sharing an overlap lattice still triangulate it identically."""
 a,c,d,b=faces[quads,0],faces[quads,1],faces[quads,2],faces[quads+1,1]
 alt=np.abs(gy[b]-gy[d])<np.abs(gy[a]-gy[c]);q=quads[alt]
 faces[q]=np.stack((a,b,d),1)[alt];faces[q+1]=np.stack((b,c,d),1)[alt]
 return faces
def workers(per_worker_gb=1.,cap=6):
 """Pool size: at most `cap`, one per CPU, and one per per_worker_gb of free commit above a 10 GB reserve
 (the post-sync graph runs other nodes beside this one). Workers peak at 0.64-0.91 GB (all-directions-30)."""
 n=min(cap,os.cpu_count() or 1)
 try:
  class Status(ctypes.Structure):_fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(f,ctypes.c_ulonglong) for f in ('phys','phys_free','commit','commit_free','virtual','virtual_free','extended')]
  s=Status();s.length=ctypes.sizeof(Status)
  if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s)):n=min(n,int((s.commit_free/2**30-10)/per_worker_gb))
 except (AttributeError,OSError):pass
 return max(1,n)
def _worker(*state):
 """Pool initializer: the per-variant inputs edited(), prepare() and refill() read as globals."""
 global variant,hexes,replacement,alpha,prefab_cutouts,playable,actual_surface
 variant,hexes,replacement,alpha,prefab_cutouts,playable=state;actual_surface=load_final_surface(variant,S)
def edit_edge_cells(rect,k,n=9):
 """K-cells of rect holding both replaced and kept points (edited() or a removed-prefab cutout, n x n samples per
 cell): global masks only, so every model refines the same cells."""
 ux0,ux1,uz0,uz1=rect;ncx,ncz=(ux1-ux0)//k,(uz1-uz0)//k;o=(np.arange(n)+.5)/n*k
 X=((ux0+np.arange(ncx)[:,None]*k+o[None,:]).ravel())*DX/2;Z=((uz0+np.arange(ncz)[:,None]*k+o[None,:]).ravel())*DX/2
 xx,zz=np.meshgrid(X,Z);e=edited(xx.ravel(),zz.ravel())
 e|=prefab_cutouts[np.rint(zz.ravel()/DX*2).astype(int).clip(0,prefab_cutouts.shape[0]-1),np.rint(xx.ravel()/DX*2).astype(int).clip(0,prefab_cutouts.shape[1]-1)]
 e=e.reshape(ncz,n,ncx,n)
 return e.any(axis=(1,3))&~e.all(axis=(1,3))
def playable_cells(rect,k,n=5):
 """K-cells of rect (lattice units) touching a playable land region (n x n samples per cell, map.hex hexes):
 they take mesh_lattice.TOL_PLAYABLE, the rest TOL_NONPLAYABLE (mesh_lattice.cell_tolerance)."""
 ux0,ux1,uz0,uz1=rect;ncx,ncz=(ux1-ux0)//k,(uz1-uz0)//k;o=(np.arange(n)+.5)/n*k
 X=((ux0+np.arange(ncx)[:,None]*k+o[None,:]).ravel())*DX/2;Z=((uz0+np.arange(ncz)[:,None]*k+o[None,:]).ravel())*DX/2
 xx,zz=np.meshgrid(X,Z);q,r=_frame.world_hex(xx,zz*_frame.DZ/DX)
 p=playable[np.clip(np.asarray(r),0,playable.shape[0]-1),np.clip(np.asarray(q),0,playable.shape[1]-1)]
 return p.reshape(ncz,n,ncx,n).any(axis=(1,3))
def prepare(b):
 """Pass 1 for one mesh (its bytes): the retained native part and the budget lattice step."""
 mp=struct.unpack_from('<I',b,152)[0];typ,flags,size,vo,vc,io,ic=struct.unpack_from('<HH5I',b,mp);bb=struct.unpack_from('<6f',b,mp+24);v=np.frombuffer(b,dtype='<f4',offset=mp+vo,count=vc*4).reshape(vc,4).copy();indices=np.frombuffer(b,dtype='<u2',offset=mp+io,count=ic).reshape(-1,3).copy();center=v[indices,:3].mean(axis=1)
 vertex_edit=edited(v[:,0],v[:,2])|(sample(alpha,_frame.tile_coord(v[:,0]),_frame.tile_coord(v[:,2]))>0)
 replace=edited(center[:,0],center[:,2])|vertex_edit[indices].any(axis=1)
 # Refilling only edited centroids misses the outer parts of deleted faces.
 # Carry their complete projected footprint into the replacement grid.
 # Only this mesh's footprint is read below (face centres within its bounds +2 hexes), so the
 # mask is built in that window (plus a margin wider than the 3x3 halo) instead of the canvas.
 x0=max(0,int(math.floor((bb[0]-3*DX)/DX*2))-2);x1=min(W*2,int(math.ceil((bb[3]+3*DX)/DX*2))+3)
 z0=max(0,int(math.floor((bb[2]-3*DX)/DX*2))-2);z1=min(H*2+1,int(math.ceil((bb[5]+3*DX)/DX*2))+3)
 # ImageDraw.polygon(xy,fill=1) is draw_polygon(xy,draw_ink(1),1); called directly (millions of faces).
 removed_mask=Image.new('1',(max(1,x1-x0),max(1,z1-z0)));removed_draw=ImageDraw.Draw(removed_mask).draw;ink=removed_draw.draw_ink(1)
 for triangle in (v[indices[replace]][:,:,[0,2]]/DX*2-np.array([x0,z0],np.float32)).reshape(-1,6).tolist():
  removed_draw.draw_polygon(triangle,ink,1)
 # Removed mountain props expose native cutouts even beyond the edit mask.
 window=np.asarray(removed_mask)[:z1-z0,:x1-x0]|prefab_cutouts[z0:z1,x0:x1]
 # One half-cell halo covers raster rounding without recursive expansion.
 window=dilate(window,3)
 # Consume this generation of polish geometry, preserving current retained vertices.
 changed_vertices=vertex_edit;seat_on_surface(v,sample_final_surface(actual_surface,v[:,0]/DX*2,v[:,2]/DX*2),changed_vertices,faces=indices)
 # Retain native unedited triangles. Reconstruct the edited footprint at a
 # regular grid, including the empty band inside the old model's padded bounds.
 keep=indices[~replace]
 # Drop vertices belonging only to replaced faces. Retaining the old 65K
 # perimeter vertices left almost no room and forced 15-hex triangles.
 used=np.unique(keep);v=v[used];keep=np.searchsorted(used,keep).astype('<u2');vc=len(v)
 # Native model bounds can leave narrow gaps between neighbouring meshes,
 # and stop short of the expanded theatre rim. Give the replacement surface
 # a two-hex overlap, seated on the same final height field in every model.
 # Every model uses the same absolute lattice of K-cells, each at its error-bounded step (mesh_lattice, mesh bake
 # 3.2.1); the vertex budget no longer widens the step (that left canvas faces 2.7 hexes apart, |LF - mesh| p95 .49):
 # a model over 65k vertices continues in overflow files (refill, OVERFLOW).
 k=ml.K
 return dict(mp=mp,typ=typ,flags=flags,vo=vo,bb=bb,v=v,vc=vc,keep=keep,k=k,window=(x0,z0,window),changed=bool(changed_vertices.any()))
def refill(m,others,cm,towns):
 """Pass 3 for one mesh: (rigid_model_v2 bytes, compressed_map bytes, new faces, overflow [(rigid_model_v2 bytes,
 compressed_map bytes)]), or None when it stays as it is."""
 path,head,mp,typ,flags,vo,bb,v,vc,keep,k,rect=(m[key] for key in ('path','head','mp','typ','flags','vo','bb','v','vc','keep','k','rect'))
 # Error-bounded steps per K-cell (a function of the absolute cell and the LF only, so overlapping models and the
 # overflow files of one model triangulate alike), on the rect plus a one-cell halo: the edge cells stitch to the
 # neighbour's step as inside, and only faces inside rect are kept (inner below).
 ux0,ux1,uz0,uz1=rect;hrect=(max(0,ux0-k),ux1+k,max(0,uz0-k),uz1+k)
 G=ml.cell_steps(actual_surface,hrect,_LEVEL,_NORM,_GAIN,k,tol=ml.cell_tolerance(playable_cells(hrect,k)))
 # Cells on the edge of the edited land take the finest step: faces are kept by their centre (use below), and a coarse
 # face across that edge left the retained native faces a hole or overlapped them (seam scan pass 0 on run
 # 20261030-104249: 1181 hole / 391 stack hexes, main 383 / 682; the seam loop's local repair took 11 problem hexes,
 # main 4). edited() is the shared edit flag, so the refinement is the same in every model.
 G[edit_edge_cells(hrect,k)]=1
 x0,z0,window=m['window'];removed=np.zeros((H*2+1,W*2),bool);removed[z0:z0+window.shape[0],x0:x0+window.shape[1]]=window
 a=cm_grid(cm);cb=struct.unpack_from('<6f',cm,26)
 # Native-style headers store zero x/z extents, meaning "the mesh bounds".
 if cb[3]<=cb[0] or cb[5]<=cb[2]:cb=(bb[0],cb[1],bb[2],bb[3],cb[4],bb[5])
 # Budget-coarsened grids: one vertex sits on each town centre.
 gx,gz,faces,quads=graded_grid(k,hrect,G,towns)
 gy=sample_final_surface(actual_surface,gx/DX*2,gz/DX*2);faces=split_squares(faces,gy,quads);new=np.column_stack((gx,gy,gz,np.ones(gx.size))).astype('<f4');faces=faces+vc;nv=np.vstack((v,new));centers=nv[faces,:3].mean(axis=1)
 rx=np.rint(centers[:,0]/DX*2).astype(int).clip(0,removed.shape[1]-1);rz=np.rint(centers[:,2]/DX*2).astype(int).clip(0,removed.shape[0]-1)
 inner=(centers[:,0]>=ux0*DX/2)&(centers[:,0]<ux1*DX/2)&(centers[:,2]>=uz0*DX/2)&(centers[:,2]<uz1*DX/2)
 use=(edited(centers[:,0],centers[:,2])|removed[rz,rx])&inner;faces=faces[use]
 # Open sea outside the native map shows the ocean background at the native floor instead
 # (coarse_mesh_skirts.open_sea_faces); lattice nodes used by those faces alone are dropped.
 # A model left with no triangle at all keeps them (an empty land model is never written).
 _sea=open_sea_faces(nv,faces)
 if len(keep) or not _sea.all():faces=faces[~_sea]
 if not use.any() and not m['changed']:return None
 # Faces past the 16-bit vertex budget continue in overflow files, whole K-cell rows at a time (a shared row edge
 # repeats its nodes in both files at the same position).
 chunks=chunk_faces(faces,nv,vc)
 faces=chunks[0];_new=np.unique(faces[faces>=vc]);_at=np.arange(len(nv));_at[_new]=vc+np.arange(len(_new));main=np.vstack((v,nv[_new]));faces=_at[faces]
 overflow=[]
 for ch in chunks[1:]:
  _u=np.unique(ch);_at=np.zeros(len(nv),np.int64);_at[_u]=np.arange(len(_u));ov=nv[_u]
  rb,box_=model_bytes(head,mp,typ,flags,vo,ov,_at[ch]);overflow.append((rb,overflow_map(cm,box_)))
 nv=main
 assert len(nv)<65000,(path,len(nv))
 ids_=np.vstack((keep,faces)).astype('<u2');vb=nv.tobytes();ib=ids_.tobytes();head=bytearray(head);struct.pack_into('<II',head,144,len(vb),len(ib));struct.pack_into('<HH5I',head,mp,typ,flags,vo+len(vb)+len(ib),vo,len(nv),vo+len(vb),ids_.size);box=(float(nv[:,0].min()),float(nv[:,1].min()),float(nv[:,2].min()),float(nv[:,0].max()),float(nv[:,1].max()),float(nv[:,2].max()));struct.pack_into('<6f',head,mp+24,*box)
 # Every sampler here is elementwise: each grid point is sampled only where its value is used
 # (the old map off the patch, the final surface on it), the alpha only off the edit flag; and
 # in row bands of ~1M points, so the largest maps (13M points) need no full-grid temporaries.
 xs,zs=np.linspace(box[0],box[3],a.shape[1]),np.linspace(box[2],box[5],a.shape[0]);rows=max(1,(1<<20)//max(1,a.shape[1]));height=None
 for r0 in range(0,a.shape[0],rows):
  cx,cz=np.meshgrid(xs,zs[r0:r0+rows]);patch=edited(cx,cz);off=~patch;patch[off]=sample(alpha,cx[off]/DX*2,cz[off]/DX*2)>0;off=~patch
  kept=cb[1]+sample(a,(cx[off]-cb[0])/(cb[3]-cb[0])*(a.shape[1]-1),(cz[off]-cb[2])/(cb[5]-cb[2])*(a.shape[0]-1))/65535*(cb[4]-cb[1])
  if height is None:height=np.empty(a.shape,kept.dtype)
  band=height[r0:r0+rows];band[off]=kept;band[patch]=sample_final_surface(actual_surface,cx[patch]/DX*2,cz[patch]/DX*2)
 lo=min(float(height.min()),box[1]);hi=max(float(height.max()),box[4],lo+.001);values=np.rint((height-lo)/(hi-lo)*65535).clip(0,65535).astype('<u2');padded=np.pad(values,((0,(-values.shape[0])%16),(0,(-values.shape[1])%16)),mode='edge');blocks=[]
 for cy in range(0,padded.shape[0],16):
  for cx_ in range(0,padded.shape[1],16):
   block=np.ascontiguousarray(padded[cy:cy+16,cx_:cx_+16],dtype='<u2');blocks.append(b'\0'+struct.pack('<H',int(block[0,0])) if np.all(block==block[0,0]) else b'\x8f'+block.tobytes())
 cmhead=bytearray(cm);struct.pack_into('<6f',cmhead,26,box[0],lo,box[2],box[3],hi,box[5]);return head+vb+ib,rebuild(cmhead,blocks),len(faces)+sum(len(c) for c in chunks[1:]),overflow
BUDGET=65000   # vertices per model (16-bit indices), as the assert above
def OVERFLOW(variant):
 """Manifest of the overflow files (parent model -> file names), per variant."""
 return A/f'analysis/coarse-mesh-overflow-{variant}.json'
def mesh_index(path):
 return int(Path(path).name.split('.')[0].rsplit('_',1)[1])
def chunk_faces(faces,nv,vc):
 """Split faces (indices into nv; nv[:vc] the retained native vertices, which the first chunk carries) into chunks of
 whole K-cell rows (by face centre) so each chunk's vertices stay under BUDGET. One chunk when they fit."""
 if not len(faces):return [faces]
 cz=np.floor(nv[faces,2].mean(axis=1)/(ml.K*DX/2)).astype(np.int64);order=np.argsort(cz,kind='stable');faces=faces[order];cz=cz[order]
 rows,starts=np.unique(cz,return_index=True);ends=np.append(starts[1:],len(faces))
 seen=np.zeros(len(nv),bool);count=0;cap=BUDGET-1-vc;chunks=[];start=0
 for a,b in zip(starts,ends):
  nodes=np.unique(faces[a:b]);nodes_new=nodes[nodes>=vc] if not chunks else nodes;add=int((~seen[nodes_new]).sum())
  if count+add>cap and (a>start or not chunks):
   # close the chunk (the first may stay empty when the retained native part leaves no room for a row)
   chunks.append(faces[start:a]);start=a;seen[:]=False;count=0;cap=BUDGET-1;nodes_new=nodes;add=len(nodes)
  if add>cap:raise RuntimeError(f'one K-cell row needs {add} vertices, over the budget {cap}')
  seen[nodes_new]=True;count+=add
 chunks.append(faces[start:])
 return chunks
def model_bytes(head,mp,typ,flags,vo,nv,ids):
 """rigid_model_v2 bytes of an overflow model: the parent's header (type, flags, materials) with these vertices and
 faces; and its box."""
 ids_=np.asarray(ids).astype('<u2');vb=np.ascontiguousarray(nv,'<f4').tobytes();ib=ids_.tobytes();hd=bytearray(head);struct.pack_into('<II',hd,144,len(vb),len(ib));struct.pack_into('<HH5I',hd,mp,typ,flags,vo+len(vb)+len(ib),vo,len(nv),vo+len(vb),ids_.size);box=(float(nv[:,0].min()),float(nv[:,1].min()),float(nv[:,2].min()),float(nv[:,0].max()),float(nv[:,1].max()),float(nv[:,2].max()));struct.pack_into('<6f',hd,mp+24,*box)
 return bytes(hd)+vb+ib,box
def overflow_map(cm,box):
 """compressed_map of an overflow model: the final surface over its box at quarter-hex spacing (as polish sizes the
 extended maps), in the parent's header."""
 cw_=max(16,math.ceil((box[3]-box[0])/(DX/4)/16)*16);ch=max(16,math.ceil((box[5]-box[2])/(DX/4)/16)*16)
 xx,zz=np.meshgrid(np.linspace(box[0],box[3],cw_),np.linspace(box[2],box[5],ch));height=sample_final_surface(actual_surface,xx/DX*2,zz/DX*2)
 lo=min(float(height.min()),box[1]);hi=max(float(height.max()),box[4],lo+.001)
 return bytes(cm_write(cm,np.rint((height-lo)/(hi-lo)*65535).clip(0,65535).astype('<u2'),(box[0],lo,box[2],box[3],hi,box[5])))
if __name__=='__main__':
 m=read_hex(A/'work/final-map/map.hex');b=(A/'work/final-map/map.hex').read_bytes();d=np.frombuffer(b[-4-W*H*16:-4],np.uint8).reshape(H,W,16);ids=(d[:,:,1].astype(int)<<5)+(d[:,:,0].astype(int)>>3)-1;names=m['land_regions']+m['sea_regions'];p=json.loads((A/'work/current-region-design.json').read_text(encoding='utf-8'));edit=_frame.load_canvas_array(A/'analysis/continuous-land-mask.npy')|np.isin(ids,[names.index(c['region']) for c in p['regions']]);yy,xx=np.indices((H,W));edit|=(xx<_frame.PAD_LEFT)|(xx>=_frame.PAD_LEFT+_frame.NATIVE_W)|(yy<_frame.PAD_BOTTOM)|(yy>=_frame.PAD_BOTTOM+_frame.NATIVE_H)
 if (A/'layout-v2/display-edit-hex-mask.npy').exists():edit|=np.load(A/'layout-v2/display-edit-hex-mask.npy')
 # playable land hexes (the mesh tolerance per cell, mesh_lattice.cell_tolerance)
 _pl=np.array(['non_playable' not in n for n in m['land_regions']]+[False]*len(m['sea_regions'])+[False]);playable=_pl[np.where((ids>=0)&(ids<len(names)),ids,len(names))]
 report=[]
 source_step=StepSource("coarse-fill",{"implementation_sha256":digest(Path(__file__)),"edit_alpha":digest(A/"analysis/polished-edit-alpha.npy"),"prefab_reconciliation":digest(A/"analysis/terrain-prefab-reconciliation.json"),**inputs(),**{v+"_coverage":digest(A/f"analysis/cleared-terrain-replacement-mask-{v}.npy") for v in TERRAIN_VARIANTS},**{v:digest(S/f"terrain/campaigns/{v}/lf_height_map.dds") for v in TERRAIN_VARIANTS}})
 alpha=np.load(A/'analysis/polished-edit-alpha.npy')
 for variant in TERRAIN_VARIANTS:
  # Native triangles are regenerated only where the LF really changed (shared
  # definition, native_edit_mask.edited_hexes in edited()). The former union of every
  # design-region hex, the continuous land mask and the display-edit mask covered
  # ~46% of native land and replaced vanilla coarse relief, skirts and prefab
  # cutouts far from any edit.
  replacement=np.load(A/f'analysis/cleared-terrain-replacement-mask-{variant}.npy')
  replacement=dilate(replacement,17)
  alpha_path=A/f'analysis/polished-edit-alpha-{variant}.npy';alpha=np.load(alpha_path) if alpha_path.exists() else np.load(A/'analysis/polished-edit-alpha.npy')
  surface=np.load(A/f'analysis/polished-surface-{variant}.npy')
  changed=0;added_faces=0
  prefab_cutouts=np.zeros((H*2+1,W*2),bool)
  prefab_report=json.loads((A/'analysis/terrain-prefab-reconciliation.json').read_text(encoding='utf-8'))
  for group in prefab_report['variants']:
   if group['variant']!=variant:continue
   for instance in group['removed_instances']:
    for q,r in instance['edited_neighbourhoods']:
     prefab_cutouts[max(0,r*2-8):min(H*2+1,r*2+9),max(0,q*2-8):min(W*2,q*2+9)]=True
  # Meshes are independent in passes 1 and 3: their math runs on a process pool, while
  # the step-source reads, backups and writes stay here, in mesh order.
  pool=ProcessPoolExecutor(workers(),initializer=_worker,initargs=(variant,edited_hexes(variant),replacement,alpha,prefab_cutouts,playable))
  try:
   # Pass 1: per mesh, the retained native part and the budget lattice step.
   prepared=[]
   # Overflow files of the previous run (OVERFLOW manifest) are this step's own output, never a model input.
   gm=S/f'terrain/campaigns/{variant}/global_meshes';man_path=OVERFLOW(variant)
   manifest=json.loads(man_path.read_text(encoding='utf-8')) if man_path.exists() else {'files':{}}
   own={n for names in manifest['files'].values() for n in names}
   for path in sorted(gm.glob('land_mesh_*.rigid_model_v2'),key=mesh_index):
    if path.name in own:continue
    # every model is a game-pack or canvas mesh of the base snapshot; anything else is a stray overflow file (polish drops them)
    if not (A/'terrain-before-polish'/path.relative_to(S)).exists():raise RuntimeError(f'land mesh outside the base snapshot: {path.name} (an overflow file of an earlier run?)')
    backup=BACK/path.relative_to(S)
    if not backup.exists():backup.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,backup)
    b=source_step.read(path)
    if b is None:continue
    prepared.append(dict(path=path,b=b,job=pool.submit(prepare,b)))
   for m in prepared:m.update(m.pop('job').result())
   # Pass 2 (before 3.2.1: per-model budget lattices, overlaps shared gcd(k_a,k_b)) is gone: every model uses the
   # one absolute K-cell lattice at the error-bounded steps, so overlaps triangulate alike by construction.
   rects=[grid_rect(m['bb'],m['k']) for m in prepared]
   maps=[]
   for m in prepared:
    cp=m['path'].with_suffix('.compressed_map');cm=source_step.read(cp)
    maps.append((cp,cp.read_bytes() if cm is None else cm))
   # Largest compressed maps first (one takes ~5x the median): results are still written in mesh order.
   jobs={}
   for i in sorted(range(len(prepared)),key=lambda i:-len(maps[i][1])):
    m=prepared[i];task={key:m[key] for key in ('path','mp','typ','flags','vo','bb','v','vc','keep','k','window','changed')};task.update(head=m['b'][:m['mp']+m['vo']],rect=rects[i])
    jobs[i]=pool.submit(refill,task,[(n['k'],rects[j]) for j,n in enumerate(prepared) if j!=i],maps[i][1],[c['town'] for c in p['regions']])
   new_overflow={}
   for i,m in enumerate(prepared):
    path,cp=m['path'],maps[i][0];out=jobs[i].result()
    if out is None:continue
    path.write_bytes(out[0]);cp.write_bytes(out[1]);changed+=1;added_faces+=out[2]
    if out[3]:new_overflow[path.name]=out[3]
   # Overflow files: the processed models' new chunks, plus the kept chunks of models this run skipped (verified
   # output), numbered contiguously after the last model (the game's land meshes run 0..N without gaps).
   done={m['path'].name for m in prepared}
   # kept: a skipped model's chunks, while that model and all its files are still in the stage (polish drops
   # overflow files outside the base snapshot, and a manifest entry of a model that is gone is stale)
   kept={parent:[((gm/n).read_bytes(),(gm/n).with_suffix('.compressed_map').read_bytes()) for n in names]
         for parent,names in manifest['files'].items() if parent not in done and parent not in own and (gm/parent).exists()
         and all((gm/n).exists() and (gm/n).with_suffix('.compressed_map').exists() for n in names)}
   for n in own:
    for f in (gm/n,(gm/n).with_suffix('.compressed_map')):f.unlink(missing_ok=True)
   first=1+max(mesh_index(f) for f in gm.glob('land_mesh_*.rigid_model_v2'));files={};at=first
   for parent,parts in sorted({**kept,**new_overflow}.items(),key=lambda kv:mesh_index(gm/kv[0])):
    files[parent]=[]
    for rb,cb in parts:
     n=f'land_mesh_{at}.rigid_model_v2';(gm/n).write_bytes(rb);(gm/n).with_suffix('.compressed_map').write_bytes(cb);files[parent].append(n);at+=1
   man_path.write_text(json.dumps({'note':'land_mesh overflow files of fill_polished_mesh_gaps (mesh bake 3.2.1): chunks of a model past the 65k vertex budget','first':first,'files':files},indent=1),encoding='utf-8')
  finally:pool.shutdown(cancel_futures=True)
  report.append({'variant':variant,'meshes_rebuilt':changed,'regular_triangles_in_edited_land':added_faces,'overflow_files':sum(len(v) for v in files.values()),'lattice':{'K':ml.K,'TOL_PLAYABLE':ml.TOL_PLAYABLE,'TOL_NONPLAYABLE':ml.TOL_NONPLAYABLE}});print(variant,changed,'meshes; regular triangles',added_faces,'; overflow files',sum(len(v) for v in files.values()),flush=True)
 source_step.finish()
 (A/'analysis/coarse-terrain-gap-fill.json').write_text(json.dumps({'variants':report,'live_verified':False},indent=2),encoding='utf-8')
