"""Supply actual coarse land for authored islands absent from vanilla land meshes."""
import json,struct,math,shutil,sys
from pathlib import Path
import numpy as np
from analyze_reference import ROOT,read_hex
import campaign_world as _frame
from final_terrain_surface import load_final_surface,sample_final_surface
from coarse_mesh_skirts import open_sea_faces
from native_height_field import parse,decode,rebuild
from campaign_world import TERRAIN_VARIANTS
A=_frame.WORKSPACE;S=A/'work/release-input';BACK=A/'before-island-ground-native-v2';_frame.check_snapshot(BACK);DX=_frame.DX;plan=json.loads((A/'work/current-region-design.json').read_text(encoding='utf-8'));m=read_hex(A/'work/final-map/map.hex');W,H=m['width'],m['height'];names=m['land_regions']+m['sea_regions'];b=(A/'work/final-map/map.hex').read_bytes();d=np.frombuffer(b[-4-W*H*16:-4],np.uint8).reshape(H,W,16);ids=(d[:,:,1].astype(int)<<5)+(d[:,:,0].astype(int)>>3)-1
islands=[g for g in plan['provinces'] if g.get('reachability_mode')=='naval' and any(k in g['province']['key'] for k in ('zhuyai','daner','zhouhu','wa_nu','yamatai','kunu','touma'))]
from fastops import cm_read as grid   # the per-block native_height_field decode loop, vectorised
def sample(a,x,y):
 x=x.clip(0,a.shape[1]-1);y=y.clip(0,a.shape[0]-1);ix=x.astype(int);iy=y.astype(int);nx=np.minimum(ix+1,a.shape[1]-1);ny=np.minimum(iy+1,a.shape[0]-1);fx=x-ix;fy=y-iy
 return (a[iy,ix]*(1-fx)+a[iy,nx]*fx)*(1-fy)+(a[ny,ix]*(1-fx)+a[ny,nx]*fx)*fy
def cm_write(src,a,box):
 h,w=a.shape;p=np.pad(a,((0,(-h)%16),(0,(-w)%16)),mode='edge');blocks=[]
 for y in range(0,p.shape[0],16):
  for x in range(0,p.shape[1],16):
   v=np.ascontiguousarray(p[y:y+16,x:x+16],dtype='<u2');blocks.append(b'\0'+struct.pack('<H',int(v[0,0])) if np.all(v==v[0,0]) else b'\x8f'+v.tobytes())
 head=bytearray(src);struct.pack_into('<2I',head,10,w,h);struct.pack_into('<6f',head,26,*box);return rebuild(head,blocks)
reports=[]
for variant in TERRAIN_VARIANTS:
 directory=S/f'terrain/campaigns/{variant}/global_meshes';surface=np.load(A/f'analysis/polished-surface-{variant}.npy');actual_surface=load_final_surface(variant,S);meshes=[]
 for path in directory.glob('land_mesh_*.rigid_model_v2'):
  b=path.read_bytes();mp=struct.unpack_from('<I',b,152)[0];box=struct.unpack_from('<6f',b,mp+24);vc=struct.unpack_from('<I',b,mp+12)[0]
  if vc<45000:meshes.append((path,box))
 additions={}
 # One rectangle per island: provinces whose padded boxes touch are merged (Kyushu's
 # wa_nu/yamatai/kunu/touma), so overlapping patch grids never stack. MARGIN reaches the
 # LF coastal slope down to the sea floor; 3 hexes left hex-stepped repair cells at the
 # shore (live 2048k: Kyushu west bay, south and east coasts).
 MARGIN=6;rects=[]
 for group in islands:
  region_ids=[names.index(c['region']) for c in group['regions']];rr,qq=np.nonzero(np.isin(ids,region_ids))
  rects.append([max(0,int(qq.min())-MARGIN),min(W,int(qq.max())+MARGIN),max(0,int(rr.min())-MARGIN),min(H,int(rr.max())+MARGIN),[group['province']['key']]])
 merged=True
 while merged:
  merged=False
  for i in range(len(rects)):
   for j in range(i+1,len(rects)):
    a,b=rects[i],rects[j]
    if a[0]<=b[1] and b[0]<=a[1] and a[2]<=b[3] and b[2]<=a[3]:
     rects[i]=[min(a[0],b[0]),max(a[1],b[1]),min(a[2],b[2]),max(a[3],b[3]),a[4]+b[4]];del rects[j];merged=True;break
   if merged:break
 for q0,q1,r0,r1,keys in rects:
  x0,x1,z0,z1=q0*DX,q1*DX,r0*DX,r1*DX;cx,cz=(x0+x1)/2,(z0+z1)/2;group={'province':{'key':'+'.join(keys)}}
  path,box=min(meshes,key=lambda v:(max(v[1][0]-cx,0,cx-v[1][3]))**2+(max(v[1][2]-cz,0,cz-v[1][5]))**2);additions.setdefault(path,[]).append((x0,x1,z0,z1,group['province']['key']))
 for path,patches in additions.items():
  old=BACK/path.relative_to(S);cp=path.with_suffix('.compressed_map');oldcm=BACK/cp.relative_to(S)
  if not old.exists() or '--refresh-base' in sys.argv:old.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,old);shutil.copy2(cp,oldcm)
  src=old.read_bytes();mp=struct.unpack_from('<I',src,152)[0];typ,flags,size,vo,vc,io,ic=struct.unpack_from('<HH5I',src,mp);verts=np.frombuffer(src,dtype='<f4',offset=mp+vo,count=vc*4).reshape(vc,4).copy();indices=np.frombuffer(src,dtype='<u2',offset=mp+io,count=ic);used=np.unique(indices);verts=verts[used];indices=np.searchsorted(used,indices).tolist()
  for x0,x1,z0,z1,key in patches:
   step=DX/2
   for attempt in range(60):
    nx=math.ceil((x1-x0)/step);nz=math.ceil((z1-z0)/step)
    if len(verts)+(nx+1)*(nz+1)<65000:break
    step*=1.2
   else:raise RuntimeError('Island patch exceeded the mesh vertex budget')
   xx,zz=np.meshgrid(np.linspace(x0,x1,nx+1),np.linspace(z0,z1,nz+1));yy=sample_final_surface(actual_surface,xx/DX*2,zz/DX*2);n=len(verts);extra=np.column_stack((xx.ravel(),yy.ravel(),zz.ravel(),np.ones(xx.size))).astype('<f4');verts=np.vstack((verts,extra));r,c=np.mgrid[:nz,:nx];aa=n+r*(nx+1)+c;bb=aa+nx+1;_f=np.stack((aa,bb+1,bb,aa,aa+1,bb+1),axis=2).reshape(-1,3);indices.extend(_f[~open_sea_faces(verts,_f)].ravel().tolist())
  # Open sea outside the native map (coarse_mesh_skirts.open_sea_faces) shows the ocean background
  # at the native floor; patch vertices left without a face are dropped.
  _used=np.unique(indices);verts=verts[_used];indices=np.searchsorted(_used,indices).tolist()
  vb=verts.tobytes();ib=np.array(indices,dtype='<u2').tobytes();head=bytearray(src[:mp+vo]);box=(float(verts[:,0].min()),float(verts[:,1].min()),float(verts[:,2].min()),float(verts[:,0].max()),float(verts[:,1].max()),float(verts[:,2].max()));struct.pack_into('<II',head,144,len(vb),len(ib));struct.pack_into('<HH5I',head,mp,typ,flags,vo+len(vb)+len(ib),vo,len(verts),vo+len(vb),len(indices));struct.pack_into('<6f',head,mp+24,*box);path.write_bytes(head+vb+ib)
  cm=oldcm.read_bytes();oldbox=struct.unpack_from('<6f',cm,26);a=grid(cm);cw=max(a.shape[1],math.ceil((box[3]-box[0])/(DX/3)/16)*16);ch=max(a.shape[0],math.ceil((box[5]-box[2])/(DX/3)/16)*16);xx,zz=np.meshgrid(np.linspace(box[0],box[3],cw),np.linspace(box[2],box[5],ch));height=oldbox[1]+sample(a,(xx-oldbox[0])/(oldbox[3]-oldbox[0])*(a.shape[1]-1),(zz-oldbox[2])/(oldbox[5]-oldbox[2])*(a.shape[0]-1))/65535*(oldbox[4]-oldbox[1])
  for x0,x1,z0,z1,key in patches:
   mask=(xx>=x0)&(xx<=x1)&(zz>=z0)&(zz<=z1);value=sample_final_surface(actual_surface,xx/DX*2,zz/DX*2);height[mask]=value[mask]
  lo=min(float(height.min()),box[1]);hi=max(float(height.max()),box[4],lo+.001);values=np.rint((height-lo)/(hi-lo)*65535).clip(0,65535).astype('<u2');cp.write_bytes(cm_write(cm,values,(box[0],lo,box[2],box[3],hi,box[5])));reports.append({'variant':variant,'mesh':path.name,'groups':[p[-1] for p in patches],'vertices':len(verts)});print(variant,path.name,[p[-1] for p in patches],flush=True)
(A/'analysis/island-ground-patches.json').write_text(json.dumps({'patches':reports,'live_verified':False},indent=2),encoding='utf-8')
