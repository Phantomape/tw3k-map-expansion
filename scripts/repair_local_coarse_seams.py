"""Partition confirmed coarse defects into a single owner on a shared lattice.

Clip existing triangles at repair-cell boundaries; add one final-DDS surface
per cell. The existing surface outside these cells is interpolated unchanged.
No LF pixels, tile caches, logical regions or props change.
"""
import argparse,hashlib,json,math,os,struct
from pathlib import Path
import numpy as np
from analyze_reference import ROOT,digest
import campaign_world as _frame
from native_edit_mask import at_world
from campaign_world import DX,W,H
from final_terrain_surface import load_final_surface,sample_final_surface
from native_campaign_climate import decode_map
from fastops import cm_write   # vectorised encode_map: byte-identical TABLE_INDEXED blocks
from repair_campaign_river_roots import Assets
from audit_ground_seams import mesh_triangles,raster
from sync_native_cache_height_ranges import cache as read_cache

A=_frame.WORKSPACE;S=A/'work/release-input';STEP=DX/4
CLASSES=('coarse_hole_native_removed','coarse_hole_prefab_removed','coarse_stack','coarse_step','coarse_hole_new_land')

def native_reference(variant,nx,nz):
    native=Assets(original_only=True);pre=f'terrain/campaigns/{variant}/global_meshes/land_mesh_'
    # The raster is in canvas coordinates, so the native shift is part of its identity.
    sources=[];identity=hashlib.sha256(repr((_frame.NATIVE_SHIFT_RAW,nx,nz)).encode())
    for key in sorted(k for k in native.index if k.startswith(pre) and k.endswith('.rigid_model_v2')):
        data=native.read(key);identity.update((key+':'+hashlib.sha256(data).hexdigest()+'\n').encode());sources.append(data)
    cache=A/'terrain-repair'/('native-coarse-reference-'+variant+'-'+identity.hexdigest()+'.npz')
    # Replaced atomically below; an unreadable file (an interrupted older run) is recomputed.
    if cache.exists():
        try:
            with np.load(cache) as z:field=z['height']
            assert field.shape==(nz,nx);return field
        except Exception as error:print(variant,'native coarse reference cache unreadable, recomputing:',error,flush=True)
    triangles=[]
    for data in sources:
        tr=mesh_triangles(data,(_frame.NATIVE_SHIFT_RAW[0],0,_frame.NATIVE_SHIFT_RAW[1]));area=(tr[:,1,0]-tr[:,0,0])*(tr[:,2,2]-tr[:,0,2])-(tr[:,2,0]-tr[:,0,0])*(tr[:,1,2]-tr[:,0,2])
        triangles.append(tr[(area>1e-8)&(tr[:,:,1].min(1)>-5)])
    _,_,field=raster(np.concatenate(triangles),nx,nz,STEP);partial=cache.with_name(cache.stem+f'.{os.getpid()}.partial.npz')
    cache.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(partial,height=field);partial.replace(cache)
    print(variant,'native coarse reference raster cached',flush=True);return field

# In the tile-aligned frame native vertices lie within ~5e-4 of repair-cell
# boundaries (float32 and the z lattice). Treat such a vertex as on the
# boundary and snap it there: no sliver pieces, no near-zero denominators, and
# clipped native pieces share the exact boundary coordinate with the cell's new
# surface. A skirt lying along a boundary is kept whole.
CLIP_EPS=1e-3

# Plain float lists: the same float64 operations in the same order as numpy on
# 4-vectors (bit-identical), without numpy's per-call cost on ~1.5M tiny clips.
def clip(poly,axis,bound,lower):
    if not poly:return poly
    snapped=[]
    for p in poly:
        if abs(p[axis]-bound)<=CLIP_EPS:p=list(p);p[axis]=bound
        snapped.append(p)
    poly=snapped
    result=[];previous=poly[-1];inside_previous=(previous[axis]>=bound) if lower else (previous[axis]<=bound)
    for current in poly:
        inside=(current[axis]>=bound) if lower else (current[axis]<=bound)
        if inside!=inside_previous:
            t=(bound-previous[axis])/(current[axis]-previous[axis]);point=[p+t*(c-p) for p,c in zip(previous,current)];point[axis]=bound;result.append(point)
        if inside:result.append(current)
        previous=current;inside_previous=inside
    return result

def rectangle(poly,x,z):
    """Clip a polygon (float64 rows as lists, e.g. tri.tolist()) to repair cell (x, z); a list
    of float rows (the same values as the array form; callers take len() and the rows)."""
    for axis,bound,lower in ((0,x*STEP,True),(0,(x+1)*STEP,False),(2,z*STEP,True),(2,(z+1)*STEP,False)):
        poly=clip(poly,axis,bound,lower)
        if len(poly)<3:break
    return poly

class Mesh:
    def __init__(self,path):
        self.path=path;self.data=path.read_bytes();self.before=hashlib.sha256(self.data).hexdigest()
        self.mp=struct.unpack_from('<I',self.data,152)[0]
        self.typ,self.flags,_,self.vo,vc,io,ic=struct.unpack_from('<HH5I',self.data,self.mp)
        assert self.typ==101
        self.box=struct.unpack_from('<6f',self.data,self.mp+24)
        self.v=np.frombuffer(self.data,'<f4',offset=self.mp+self.vo,count=vc*4).reshape(vc,4).copy()
        self.indices=np.frombuffer(self.data,'<u2',offset=self.mp+io,count=ic).reshape(-1,3).copy()
        self.points=None;self.faces=None;self.lookup=None;self.changed=False;self.pending=[]
    def editable(self,indices=None):
        if self.points is not None:return
        ids=self.indices if indices is None else indices;used=np.unique(ids)
        self.points=[v.copy() for v in self.v[used]];self.faces=np.searchsorted(used,ids).tolist()
        self.lookup={v.tobytes():i for i,v in enumerate(self.points)}
    def point(self,v):
        v=np.asarray(v,dtype='<f4');key=v.tobytes();i=self.lookup.get(key)
        if i is None:i=len(self.points);self.lookup[key]=i;self.points.append(v.copy())
        return i
    def polygon(self,poly):
        ids=[self.point(v) for v in poly]
        # Fan faces are candidates; bytes() drops the degenerate ones in one batch.
        for i in range(1,len(ids)-1):
            self.pending.append(len(self.faces));self.faces.append([ids[0],ids[i],ids[i+1]])
    def split(self,pos,corners,inserted):
        """Replace face pos (its source corners, in winding order) by a fan through the points
        inserted inside its edges ((edge index, point) rows, the point at the edge's height):
        from the opposite corner when they lie on one edge, else from the centroid."""
        ring=[];used=set()
        for k in range(3):
            a,b=corners[k],corners[(k+1)%3];dx,dz=b[0]-a[0],b[2]-a[2]
            on=sorted((p for e,p in inserted if e==k),key=lambda p:(p[0]-a[0])*dx+(p[2]-a[2])*dz)
            if on:used.add(k)
            ring+=[a]+on
        if len(used)==1:
            k=ring.index(corners[(next(iter(used))+2)%3]);ring=ring[k:]+ring[:k];ids=[self.point(v) for v in ring];faces=[[ids[0],ids[i],ids[i+1]] for i in range(1,len(ids)-1)]
        else:
            g=self.point([sum(c[i] for c in corners)/3 for i in range(3)]+[1.]);ids=[self.point(v) for v in ring];faces=[[g,ids[i],ids[(i+1)%len(ids)]] for i in range(len(ids))]
        self.faces[pos]=faces[0];self.pending.append(pos)
        for f in faces[1:]:self.pending.append(len(self.faces));self.faces.append(f)
    def drop_degenerate(self):
        # Retain valid 3-D skirts as well as top surfaces. Same float32 cross product
        # per face as one call per face; a length near the threshold is decided by
        # the original per-face norm.
        if not self.pending:return
        rows=np.asarray(self.pending);p=np.asarray(self.points)[np.asarray(self.faces)[rows]][:,:,:3]
        cr=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]);length=np.sqrt((cr.astype(np.float64)**2).sum(1))
        keep=length>1e-9;near=np.flatnonzero((length>.5e-9)&(length<2e-9))
        for k in near:keep[k]=np.linalg.norm(cr[k])>1e-9
        drop=set(rows[~keep].tolist());self.faces=[f for n,f in enumerate(self.faces) if n not in drop];self.pending=[]
    def bytes(self):
        self.drop_degenerate()
        v=np.asarray(self.points,dtype='<f4');i=np.asarray(self.faces,dtype='<u2')
        assert len(v)<65536,(self.path,len(v))
        head=bytearray(self.data[:self.mp+self.vo]);vb=v.tobytes();ib=i.tobytes()
        struct.pack_into('<II',head,144,len(vb),len(ib))
        struct.pack_into('<HH5I',head,self.mp,self.typ,self.flags,self.vo+len(vb)+len(ib),self.vo,len(v),self.vo+len(vb),i.size)
        box=list(self.box)
        for axis in range(3):box[axis]=min(box[axis],float(v[:,axis].min()));box[axis+3]=max(box[axis+3],float(v[:,axis].max()))
        struct.pack_into('<6f',head,self.mp+24,*box)
        return bytes(head)+vb+ib,box

LARGE_MESH=40000   # vertices; such meshes take whole-triangle removal, not clipping

# normalize_coarse_partition_heights seats every edited cut vertex on the final surface. A
# clipped triangle far from it then bends at the cut points on its outer edges, and the uncut
# neighbour across such an edge stays straight: a T-junction step. The next pass flags and
# clips that neighbour, so a field of big triangles (land_mesh_154's 8-hex faces at a canvas
# corner, all 0.6-1.8 off the LF) walks the step outward instead of closing it. Growing the
# repair over every far triangle would take that whole field (its neighbours are just as far),
# so the neighbour gets those edge points as vertices instead: normalize seats them with the
# clipped side and the neighbour bends continuously. STITCH_TOL is half the audit's coarse_step
# floor (0.4 between adjacent quarter-cell samples); a smaller seat cannot open a step on its
# own, and leaving it out keeps every repair without far cuts exactly as before.
STITCH_TOL=.2

def edges(corners):
    """(start, xz direction, height change, xz length) of a surface triangle's three edges."""
    out=[]
    for k in range(3):
        a,b=corners[k],corners[(k+1)%3];dx,dz=b[0]-a[0],b[2]-a[2];out.append((a,dx,b[1]-a[1],dz,math.hypot(dx,dz)))
    return out

def on_edge(p,sides):
    """(edge index, edge height) of the triangle edge (sides=edges(corners)) that has xz point p
    strictly inside it (off its corners by more than CLIP_EPS), else None."""
    for k,(a,dx,dy,dz,n) in enumerate(sides):
        ex,ez=p[0]-a[0],p[2]-a[2];d=ex*dx+ez*dz
        if abs(dx*ez-dz*ex)<=CLIP_EPS*n and CLIP_EPS*n<d<n*(n-CLIP_EPS):return k,a[1]+d/(n*n)*dy
    return None

def edge_points(poly,sides):
    """Vertices of a clipped piece inside an edge of its source triangle, not at a corner."""
    return [p for p in poly if on_edge(p,sides) is not None]

def inside_edge(xz,a,b):
    """on_edge's rule for one edge a-b over xz rows: (inside mask, parameter along a-b)."""
    dx,dz=b[0]-a[0],b[2]-a[2];n=math.hypot(dx,dz);ex=xz[:,0]-a[0];ez=xz[:,1]-a[2];d=ex*dx+ez*dz
    return (np.abs(dx*ez-dz*ex)<=CLIP_EPS*n)&(CLIP_EPS*n<d)&(d<n*(n-CLIP_EPS)),d/(n*n)

def t_junctions(variant,meshes,clipped_by,points,field):
    """Edges of uncut surface triangles that an edited repair vertex (points: rows with x, z first
    and third) lies strictly inside, where the edge is more than STITCH_TOL off the final surface:
    {mesh index: {triangle: [(edge index, [x, edge height, z, 1])]}}. Such an edge takes every
    vertex inside it, edited or not, the repair's and any mesh's surface vertices: once seated
    piecewise, a vertex left out would see the crack between its seated neighbours instead."""
    if not points:return {}
    every=np.asarray([(q[0],q[2]) for q in points]);p=every[at_world(variant,every[:,0],every[:,1]*_frame.DZ/DX)]
    if not len(p):return {}
    # One point per float32 position (the vertex the repair side gets).
    _,first=np.unique(p.astype('<f4'),axis=0,return_index=True);p=p[np.sort(first)]
    kx=np.floor(p[:,0]/STEP).astype(int);kz=np.floor(p[:,1]/STEP).astype(int);where={}
    for n,key in enumerate(zip(kz.tolist(),kx.tolist())):where.setdefault(key,[]).append(n)
    mask=np.zeros((kz.max()+2,kx.max()+2),bool);mask[kz,kx]=True;near=box_counts(mask);hits=[]
    for m,mesh in enumerate(meshes):
        tri=mesh.v[mesh.indices].astype(np.float64)
        # One cell of slack: an edge on a lattice line may sit a float32 step on either side of it.
        x0=np.floor(tri[:,:,0].min(1)/STEP).astype(int)-1;x1=np.floor(tri[:,:,0].max(1)/STEP).astype(int)+1
        z0=np.floor(tri[:,:,2].min(1)/STEP).astype(int)-1;z1=np.floor(tri[:,:,2].max(1)/STEP).astype(int)+1
        area=(tri[:,1,0]-tri[:,0,0])*(tri[:,2,2]-tri[:,0,2])-(tri[:,2,0]-tri[:,0,0])*(tri[:,1,2]-tri[:,0,2])
        ok=(np.abs(area)>1e-6)&(tri[:,:,1].min(1)>-40)&(x1>=0)&(z1>=0)
        if m in clipped_by:ok&=~clipped_by[m]
        ix=np.flatnonzero(ok);ix=ix[near(z0[ix],z1[ix],x0[ix],x1[ix])>0]
        for t in ix:
            sides=None
            for n in (n for z in range(max(0,z0[t]),z1[t]+1) for x in range(max(0,x0[t]),x1[t]+1) for n in where.get((z,x),())):
                sides=sides or edges(tri[t].tolist());hit=on_edge((p[n,0],0.,p[n,1]),sides)
                if hit is not None:hits.append((m,int(t),hit[0],p[n,0],hit[1],p[n,1]))
    if not hits:return {}
    h=np.asarray([row[3:] for row in hits]);far=(np.abs(h[:,1]-sample_final_surface(field,h[:,0]/DX*2,h[:,2]/DX*2))>STITCH_TOL)&(h[:,1]>-40)
    far_edges=sorted({row[:3] for row,bad in zip(hits,far) if bad})
    if not far_edges:return {}
    pool=[every]
    for mesh in meshes:
        tri=mesh.v[mesh.indices];top=np.abs((tri[:,1,0]-tri[:,0,0])*(tri[:,2,2]-tri[:,0,2])-(tri[:,2,0]-tri[:,0,0])*(tri[:,1,2]-tri[:,0,2]))>1e-6
        v=mesh.v[np.unique(mesh.indices[top])];pool.append(v[v[:,1]>-40][:,[0,2]].astype(np.float64))
    pool=np.concatenate(pool);_,first=np.unique(pool.astype('<f4'),axis=0,return_index=True);pool=pool[np.sort(first)]
    found={};skipped=[]
    for m,t,k in far_edges:
        mesh=meshes[m];c=mesh.v[mesh.indices[t]].astype(np.float64);a,b=c[k],c[(k+1)%3]
        if skirted(mesh,a,b):skipped.append((mesh.path.name,t,k));continue
        box=(pool[:,0]>=min(a[0],b[0])-CLIP_EPS)&(pool[:,0]<=max(a[0],b[0])+CLIP_EPS)&(pool[:,1]>=min(a[2],b[2])-CLIP_EPS)&(pool[:,1]<=max(a[2],b[2])+CLIP_EPS)
        q=pool[box];inside,s=inside_edge(q,a,b)
        found.setdefault(m,{}).setdefault(t,[]).extend((k,[x,a[1]+u*(b[1]-a[1]),z,1.]) for (x,z),u in zip(q[inside].tolist(),s[inside].tolist()))
    # A skirted edge is a mesh border: bending its top would leave the skirt standing above or
    # hanging below it, so it keeps the old straight T-junction (and the seam walk check names it).
    for name,t,k in skipped:print(variant,f'far T-junction on a skirted border edge left as is: {name} triangle {t} edge {k}',flush=True)
    return found

def skirted(mesh,a,b):
    """Whether mesh has a vertical (skirt) face lying along the xz segment a-b."""
    if not hasattr(mesh,'skirts'):
        tri=mesh.v[mesh.indices].astype(np.float64)
        area=np.abs((tri[:,1,0]-tri[:,0,0])*(tri[:,2,2]-tri[:,0,2])-(tri[:,2,0]-tri[:,0,0])*(tri[:,1,2]-tri[:,0,2]))
        mesh.skirts=tri[(area<=1e-6)&(tri[:,:,1].max(1)-tri[:,:,1].min(1)>.3)&(tri[:,:,1].max(1)>-40)]
    if not len(mesh.skirts):return False
    dx,dz=b[0]-a[0],b[2]-a[2];n=math.hypot(dx,dz);ex=mesh.skirts[:,:,0]-a[0];ez=mesh.skirts[:,:,2]-a[2];d=(ex*dx+ez*dz)/(n*n)
    return bool(((np.abs(dx*ez-dz*ex)<=CLIP_EPS*n).all(1)&(d.min(1)>-1e-6)&(d.max(1)<1+1e-6)&(d.max(1)-d.min(1)>1e-6)).any())
def box_counts(mask):
    """Set cells of mask per inclusive rectangle [z0,z1] x [x0,x1] (arrays): an integral image over
    the mask's bounding box only, the same counts as one over the whole lattice at a fraction of it."""
    rz=np.flatnonzero(mask.any(1));rx=np.flatnonzero(mask.any(0))
    if not len(rz):return lambda z0,z1,x0,x1:np.zeros(len(z0),np.int64)
    zb,ze,xb,xe=rz[0],rz[-1],rx[0],rx[-1];summed=np.pad(mask[zb:ze+1,xb:xe+1].astype(np.int32).cumsum(0).cumsum(1),((1,0),(1,0)))
    def count(z0,z1,x0,x1):
        a0=np.maximum(z0,zb)-zb;a1=np.minimum(z1,ze)-zb;b0=np.maximum(x0,xb)-xb;b1=np.minimum(x1,xe)-xb;ok=(a0<=a1)&(b0<=b1)
        a0,a1,b0,b1=(np.where(ok,v,0) for v in (a0,a1,b0,b1))
        return np.where(ok,summed[a1+1,b1+1]-summed[a0,b1+1]-summed[a1+1,b0]+summed[a0,b0],0)
    return count

def repair(variant,scan_dir):
    report=json.loads((scan_dir/'ground-seam-scan.json').read_text(encoding='utf-8'))
    assert report['variant']==variant and report['stage'] and Path(report['stage']).resolve()==S.resolve()
    assert Path(report['map']).read_bytes()==(A/'work/final-map/map.hex').read_bytes()
    # Decompress each mask once: indexing the lazy NpzFile inside the per-hex loop
    # below re-read the archive member every time (~4000 times, most of the step).
    with np.load(scan_dir/'ground-seam-masks.npz') as _z:masks={k:_z[k] for k in _z.files}
    target=np.zeros((H,W),bool)
    for key in CLASSES:target|=masks[key]>0
    if not target.any():print(variant,'no coarse defects to repair',flush=True);return
    # Lattice over the whole tile-aligned world (W*DX wide), as audit_ground_seams.
    nx=int(_frame.WORLD_WIDTH/STEP)+3;nz=(H+1)*4+3;cells=np.zeros((nz-1,nx-1),bool);removed_prefab_cells=np.zeros_like(cells)
    # Repair box centred on the hex's tile block (cw.tile_hex ownership, the rule
    # audit_ground_seams pools defects by): [q-.5, q+1.5)*DX, half a hex beyond the block.
    for r,q in np.argwhere(target):
        cx=q*4+2;cz=r*4+(q%2)*2+2
        cells[max(0,cz-4):min(nz-1,cz+4),max(0,cx-4):min(nx-1,cx+4)]=True
        if masks['coarse_hole_prefab_removed'][r,q]>0:
            removed_prefab_cells[max(0,cz-4):min(nz-1,cz+4),max(0,cx-4):min(nx-1,cx+4)]=True
    prefix=S/f'terrain/campaigns/{variant}/global_meshes';meshes=[Mesh(p) for p in sorted(prefix.glob('land_mesh_*.rigid_model_v2'))]
    field=load_final_surface(variant,S)
    # Large canvas meshes (one covers a whole corner of the 2048 canvas) cannot
    # absorb clipped pieces under the u16 vertex limit: their pieces go to the
    # nearby mesh with the most room instead (the same choice the partition makes).
    in_cells=box_counts(cells)
    overflow=[]
    clipped=0;clipped_by={};cut=[]
    for m,mesh in enumerate(meshes):
        tri=mesh.v[mesh.indices].astype(np.float64)
        x0=np.maximum(0,np.floor(tri[:,:,0].min(1)/STEP).astype(int));x1=np.minimum(nx-2,np.floor(tri[:,:,0].max(1)/STEP).astype(int))
        z0=np.maximum(0,np.floor(tri[:,:,2].min(1)/STEP).astype(int));z1=np.minimum(nz-2,np.floor(tri[:,:,2].max(1)/STEP).astype(int))
        affected=np.zeros(len(tri),bool)
        # An integral image makes triangle-box intersection constant time.
        valid=(x0<=x1)&(z0<=z1)
        ix=np.flatnonzero(valid);affected[ix]=in_cells(z0[ix],z1[ix],x0[ix],x1[ix])>0
        if not affected.any():continue
        mesh.editable(mesh.indices[~affected]);mesh.changed=True;clipped+=int(affected.sum());clipped_by[m]=affected
        # Surface faces only collect T-junction points (skirts are vertical; normalize keeps them under the top).
        surface=np.abs((tri[:,1,0]-tri[:,0,0])*(tri[:,2,2]-tri[:,0,2])-(tri[:,2,0]-tri[:,0,0])*(tri[:,1,2]-tri[:,0,2]))>1e-6
        for t in np.flatnonzero(affected):
            corners=tri[t].tolist();sides=edges(corners) if surface[t] else None
            for z in range(z0[t],z1[t]+1):
                for x in range(x0[t],x1[t]+1):
                    if cells[z,x]:continue
                    poly=rectangle(corners,x,z)
                    if len(poly)>=3:
                        if sides:cut+=edge_points(poly,sides)
                        if len(mesh.v)>LARGE_MESH:overflow.append(poly)
                        else:mesh.polygon(poly)
        if len(mesh.points)>65000:raise RuntimeError(f'Clipped mesh exceeds native vertex capacity: {mesh.path}')
    # Close the far T-junctions on the uncut side (see STITCH_TOL). Faces of a clipped mesh
    # keep the order of its retained triangles, so a triangle's face is its rank among them.
    # The repair cells' outline nodes too: an uncut edge along a lattice line meets their corners.
    touch=np.zeros((nz,nx),np.int8)
    for dz in (0,1):
        for dx in (0,1):touch[dz:dz+nz-1,dx:dx+nx-1]+=cells
    kz,kx=np.nonzero((touch>0)&(touch<4));cut+=[(x*STEP,0.,z*STEP) for z,x in zip(kz.tolist(),kx.tolist())]
    stitched=t_junctions(variant,meshes,clipped_by,cut,field);inserted=0
    for m,rows in stitched.items():
        mesh=meshes[m];mesh.editable(mesh.indices[~clipped_by[m]] if m in clipped_by else None);mesh.changed=True
        rank=np.cumsum(~clipped_by[m])-1 if m in clipped_by else np.arange(len(mesh.indices))
        for t,points in rows.items():mesh.split(int(rank[t]),mesh.v[mesh.indices[t]].astype(np.float64).tolist(),points);inserted+=len(points)
        if len(mesh.points)>65000:raise RuntimeError(f'Stitched mesh exceeds native vertex capacity: {mesh.path}')
    if stitched:print(variant,f'far cut T-junctions closed: {sum(map(len,stitched.values()))} uncut triangles, {inserted} edge points',flush=True)
    reference=native_reference(variant,nx,nz)
    alpha=np.load(A/f'analysis/polished-edit-alpha-{variant}.npy')
    native_assets=Assets(original_only=True)
    old_paths,old_layers=read_cache(native_assets.read('terrain/campaigns/3k_main_map/tile_map.tiles'),_frame.NATIVE_TILE_W,_frame.NATIVE_TILE_H)
    new_paths,new_layers=read_cache((S/f'terrain/campaigns/{variant}/tile_map.tiles').read_bytes(),_frame.TILE_W,_frame.TILE_H)
    assert old_paths==new_paths
    zz,xx=np.nonzero(cells);tx=(xx/2).astype(int);tz=(zz/2).astype(int)
    inside=(tx>=_frame.NATIVE_TILE_X0)&(tx<_frame.NATIVE_TILE_X1)&(tz>=_frame.NATIVE_TILE_Y0)&(tz<_frame.NATIVE_TILE_Y1);restore=np.zeros(cells.shape,bool)
    for z,x,cy,cx in zip(zz[inside],xx[inside],tz[inside],tx[inside]):
        # Matching fine cache records do not preserve a separate mountain
        # prefab. Its intentional coarse cutout needs ground after removal.
        if removed_prefab_cells[z,x]:continue
        if alpha[cy,cx]>1e-7:continue
        before=old_layers[0][cy-_frame.NATIVE_TILE_Y0,cx-_frame.NATIVE_TILE_X0];now=new_layers[0][cy,cx]
        if int(before['flags'])==16:continue
        old_parent=int(before['path'])
        if old_parent<0:
            parent=-old_parent-1;old_parent=_frame.native_cache_owner(parent)
        same=(int(now['path'])==old_parent and int(now['flags'])==int(before['flags']) and int(now['x'])==int(before['x'])+_frame.NATIVE_TILE_X0 and int(now['y'])==int(before['y'])+_frame.NATIVE_TILE_Y0 and int(now['climate'])==int(before['climate']) and int(now['extra'])==int(before['extra']))
        if same and max(abs(float(now[k])-float(before[k])) for k in ('lo','hi'))<5e-5:restore[z,x]=True
    # If the original fine terrain is still identical, restore its original
    # coarse triangles and intentional cutouts, rather than filling under it.
    native_polygons={};in_restore=box_counts(restore)
    native_pre=f'terrain/campaigns/{variant}/global_meshes/land_mesh_'
    for key in sorted(k for k in native_assets.index if k.startswith(native_pre) and k.endswith('.rigid_model_v2')):
        data=native_assets.read(key);mp=struct.unpack_from('<I',data,152)[0];_,_,_,vo,vc,io,ic=struct.unpack_from('<HH5I',data,mp)
        v=np.frombuffer(data,'<f4',offset=mp+vo,count=vc*4).reshape(vc,4).astype(np.float64);v[:,0]+=_frame.NATIVE_SHIFT_RAW[0];v[:,2]+=_frame.NATIVE_SHIFT_RAW[1]
        ids=np.frombuffer(data,'<u2',offset=mp+io,count=ic).reshape(-1,3);tr=v[ids]
        x0=np.maximum(0,np.floor(tr[:,:,0].min(1)/STEP).astype(int));x1=np.minimum(nx-2,np.floor(tr[:,:,0].max(1)/STEP).astype(int))
        z0=np.maximum(0,np.floor(tr[:,:,2].min(1)/STEP).astype(int));z1=np.minimum(nz-2,np.floor(tr[:,:,2].max(1)/STEP).astype(int));valid=(x0<=x1)&(z0<=z1);ix=np.flatnonzero(valid)
        count=in_restore(z0[ix],z1[ix],x0[ix],x1[ix])
        for t in ix[count>0]:
            corners=tr[t].tolist()
            for z in range(z0[t],z1[t]+1):
                for x in range(x0[t],x1[t]+1):
                    if not restore[z,x]:continue
                    p=rectangle(corners,x,z)
                    if len(p)>=3:native_polygons.setdefault((z,x),[]).append(p)
    zz,xx=np.nonzero(cells);new_faces=0
    boxes=np.asarray([m.box for m in meshes])
    if overflow:
        # The pick below reads every mesh's vertex count; editable() is idempotent and
        # the meshes the first loop clipped are already editable with their subset.
        for m in meshes:m.editable()
    counts=np.array([len(m.points) if m.points is not None else 0 for m in meshes]);small=np.array([len(m.v)<=LARGE_MESH for m in meshes])
    def roomiest(wx,wz,need):
        near=np.argsort(((boxes[:,0]+boxes[:,3])/2-wx)**2+((boxes[:,2]+boxes[:,5])/2-wz)**2)
        inside=near[((boxes[:,0]<=wx)&(wx<=boxes[:,3])&(boxes[:,2]<=wz)&(wz<=boxes[:,5]))[near]]
        # Outside every small box, only the 8 nearest compete (as for repair cells),
        # then the nearest with room: a far-away owner would grow its bounds, and the
        # compressed_map resampled over them, across the canvas.
        for group in (inside,near[:8]):
            # Fewest vertices first, then lowest index (as min over (count, index)).
            room=group[small[group]&(counts[group]+need<65000)]
            if len(room):return int(room[np.lexsort((room,counts[room]))[0]])
        room=near[small[near]&(counts[near]+need<65000)]
        if len(room):return int(room[0])
        raise RuntimeError(f'No vertex capacity for clipped piece at {(wx,wz)}')
    for poly in overflow:
        c=np.asarray(poly)[:,[0,2]].mean(0)
        j=roomiest(float(c[0]),float(c[1]),len(poly));m=meshes[j];m.polygon(poly);m.changed=True;counts[j]=len(m.points)
    # Corner heights and edit flags for every repair cell at once (elementwise, as per cell).
    corner_x=np.stack((xx,xx+1,xx+1,xx),1).astype(np.float64)*STEP;corner_z=np.stack((zz,zz,zz+1,zz+1),1).astype(np.float64)*STEP
    corner_heights=sample_final_surface(field,corner_x/DX*2,corner_z/DX*2)
    corner_edited=at_world(variant,corner_x,corner_z*_frame.DZ/DX)
    for n,(z,x) in enumerate(zip(zz,xx)):
        wx=(x+.5)*STEP;wz=(z+.5)*STEP
        candidates=np.flatnonzero((boxes[:,0]<=wx)&(boxes[:,3]>=wx)&(boxes[:,2]<=wz)&(boxes[:,5]>=wz))
        if not len(candidates):candidates=np.argsort(((boxes[:,0]+boxes[:,3])/2-wx)**2+((boxes[:,2]+boxes[:,5])/2-wz)**2)[:8]
        choices=[]
        for j in candidates:
            m=meshes[j]
            if m.points is None:m.editable()
            if len(m.points)+4<65000:choices.append((len(m.points),int(j)))
        if not choices:
            # Every mesh around the cell is near the 16-bit index limit: use the
            # nearest mesh that still has room (bytes() grows its bounds).
            for j in np.argsort(((boxes[:,0]+boxes[:,3])/2-wx)**2+((boxes[:,2]+boxes[:,5])/2-wz)**2):
                m=meshes[j];m.editable()
                if len(m.points)+4<65000:choices.append((len(m.points),int(j)));break
        if not choices:raise RuntimeError(f'No vertex capacity for repair cell {(x,z)}')
        _,owner=min(choices);mesh=meshes[owner];mesh.changed=True
        if restore[z,x]:
            for poly in native_polygons.get((z,x),[]):mesh.polygon(poly)
            continue
        coords=np.array([[x,z],[x+1,z],[x+1,z+1],[x,z+1]],np.float64)*STEP
        heights=corner_heights[n].copy()
        # Untouched native relief has authored coarse simplification, which
        # need not equal the LF samples on mountain peaks. Preserve it.
        qi=np.array([x,x+1,x+1,x]);ri=np.array([z,z,z+1,z+1])
        native_y=reference[ri,qi]
        ax=np.rint(qi/2).astype(int).clip(0,alpha.shape[1]-1);az=np.rint(ri/2).astype(int).clip(0,alpha.shape[0]-1)
        # Same edit definition as fill/normalize, so neighbouring native triangles
        # and these corners agree on which ground is vanilla (no T-junction cracks).
        unchanged=~corner_edited[n]&np.isfinite(native_y)
        heights[unchanged]=native_y[unchanged]
        ids=[mesh.point([p[0],y,p[1],1.]) for p,y in zip(coords,heights)]
        mesh.faces.extend(([ids[0],ids[1],ids[2]],[ids[0],ids[2],ids[3]]));new_faces+=2
    plan=[];combined=hashlib.sha256()
    for m in meshes:
        if not m.changed:continue
        blob,box=m.bytes();combined.update((m.path.name+':'+m.before+'\n').encode());plan.append((m,blob,box))
    backup=A/'terrain-repair/local-coarse-seams-before'/combined.hexdigest()/variant;rows=[]
    for mesh,blob,box in plan:
        if digest(mesh.path)!=mesh.before:raise RuntimeError('Coarse source changed during repair')
        saved=backup/mesh.path.name;saved.parent.mkdir(parents=True,exist_ok=True)
        if not saved.exists():saved.write_bytes(mesh.data)
        cm=mesh.path.with_suffix('.compressed_map');data=cm.read_bytes();saved_cm=backup/cm.name
        if not saved_cm.exists():saved_cm.write_bytes(data)
        values=decode_map(data);cb=struct.unpack_from('<6f',data,26)
        gx,gz=np.meshgrid(np.linspace(box[0],box[3],values.shape[1]),np.linspace(box[2],box[5],values.shape[0]))
        ox=np.rint(np.clip((gx-cb[0])/max(cb[3]-cb[0],1e-9)*(values.shape[1]-1),0,values.shape[1]-1)).astype(int)
        oz=np.rint(np.clip((gz-cb[2])/max(cb[5]-cb[2],1e-9)*(values.shape[0]-1),0,values.shape[0]-1)).astype(int)
        old_y=cb[1]+values[oz,ox]/65535*(cb[4]-cb[1])
        cx=np.floor(gx/STEP).astype(int).clip(0,cells.shape[1]-1);cz=np.floor(gz/STEP).astype(int).clip(0,cells.shape[0]-1);patch=cells[cz,cx]
        old_y[patch]=sample_final_surface(field,gx[patch]/DX*2,gz[patch]/DX*2)
        lo=min(cb[1],float(old_y.min()));hi=max(cb[4],float(old_y.max()),lo+.001)
        encoded=np.rint((old_y-lo)/(hi-lo)*65535).clip(0,65535).astype('<u2')
        head=bytearray(data);struct.pack_into('<6f',head,26,box[0],lo,box[2],box[3],hi,box[5])
        cm_blob=cm_write(head,encoded)
        temporary=mesh.path.with_suffix('.seam-repair.tmp');temporary.write_bytes(blob);temporary.replace(mesh.path)
        temporary=cm.with_suffix('.seam-repair.tmp');temporary.write_bytes(cm_blob);temporary.replace(cm)
        rows.append({'mesh':mesh.path.name,'before':mesh.before,'after':digest(mesh.path),'vertices':len(mesh.points)})
    result={'variant':variant,'input_scan':str(scan_dir),'backup':str(backup),'problem_hexes':int(target.sum()),'problem_hex_list':[[int(q),int(r)] for r,q in np.argwhere(target)],'quarter_cells':int(cells.sum()),'triangles_clipped':clipped,'new_faces':new_faces,'meshes':rows,'lf_and_props_unchanged':True,'live_verified':False}
    # Only when it acted, so a repair without far triangles writes the same report as before.
    if stitched:result['far_t_junctions']={'tolerance':STITCH_TOL,'edge_points':inserted,'triangles':[{'mesh':meshes[m].path.name,'triangle':t,'points':len(p)} for m,rows in stitched.items() for t,p in rows.items()]}
    (A/f'analysis/local-coarse-seam-repair-{variant}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(variant,'exact coarse partition:',len(rows),'meshes;',new_faces,'new faces',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--variant',default='3k_dlc07_main_map');parser.add_argument('--scan',type=Path,required=True);args=parser.parse_args();repair(args.variant,args.scan)
