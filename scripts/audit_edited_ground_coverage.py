"""Raster-check actual coarse top triangles across edited and cleared land."""
import argparse,hashlib,json,struct
import numpy as np
from PIL import Image,ImageDraw
from analyze_reference import ROOT,read_hex,digest
from campaign_world import DX
import campaign_world as _frame
from campaign_world import TERRAIN_VARIANTS
parser=argparse.ArgumentParser();parser.add_argument('--strict',action='store_true');args=parser.parse_args()
A=_frame.WORKSPACE;S=A/'work/release-input';map_=read_hex(A/'work/final-map/map.hex');W,H=map_['width'],map_['height'];TW,TH=W*2,H*2+1
raw=(A/'work/final-map/map.hex').read_bytes();cells=np.frombuffer(raw[-4-W*H*16:-4],np.uint8).reshape(H,W,16);yy,xx=np.indices((TH,TW));q,r=_frame.tile_hex(xx,yy);land=(cells[r,q,0]&3)!=1
clears=json.loads((A/'analysis/obsolete-terrain-overlays.json').read_text())['variants'];reports=[]
for variant in TERRAIN_VARIANTS:
 alpha=np.load(A/f'analysis/polished-edit-alpha-{variant}.npy');wanted=alpha>.02
 cache_coverage=A/'analysis/cache-replacement-mask.npy'
 if cache_coverage.exists():wanted|=np.load(cache_coverage)
 for group in clears:
  if group['variant']!=variant or group['list']!='tile_list.bin':continue
  for row in group['removed']:
   x,y=row['xy'];w,h=row['size']
   for sx,sy in {(w,h),(h,w)}:
    wanted[max(0,y-4):min(TH,y+2*sy+4),max(0,x-4):min(TW,x+2*sx+4)]=True
 wanted&=land
 # ImageDraw.polygon(xy,fill=1) is draw_polygon(xy,draw_ink(1),1) (no outline): call it on .tolist() rows directly.
 mask=Image.new('1',(TW,TH));draw=ImageDraw.Draw(mask);ink=draw.draw.draw_ink(1);count=0;mesh_hash=hashlib.sha256()
 for path in sorted((S/f'terrain/campaigns/{variant}/global_meshes').glob('land_mesh_*.rigid_model_v2')):
  mesh_hash.update((path.name+':'+digest(path)+'\n').encode())
  b=path.read_bytes();mp=struct.unpack_from('<I',b,152)[0];_,_,_,vo,vc,io,ic=struct.unpack_from('<HH5I',b,mp);verts=np.frombuffer(b,dtype='<f4',offset=mp+vo,count=vc*4).reshape(vc,4);idx=np.frombuffer(b,dtype='<u2',offset=mp+io,count=ic).reshape(-1,3);tri=verts[idx,:3]
  # Native land faces use positive projected X/Z winding. Back faces and
  # vertical skirts cannot supply a visible ground surface from above.
  area=(tri[:,1,0]-tri[:,0,0])*(tri[:,2,2]-tri[:,0,2])-(tri[:,2,0]-tri[:,0,0])*(tri[:,1,2]-tri[:,0,2]);top=(tri[:,:,1].min(axis=1)>-5)&(area>1e-6);xz=tri[top][:,:,[0,2]]/DX*2
  for polygon in xz.tolist():draw.draw.draw_polygon(polygon,ink,1)
  count+=int(top.sum())
 covered=np.asarray(mask);missing=wanted&~covered;points=np.column_stack(np.nonzero(missing));known=[]
 for name,(cx,cy) in ((k,_frame.load_authored_coordinates()['check_points'][k]) for k in ('cliff_a','cliff_b','overlap')):
  local=missing[max(0,int(cy*2)-20):int(cy*2)+21,max(0,int(cx*2)-20):int(cx*2)+21];known.append({'name':name,'missing_half_hex_samples':int(local.sum())})
 reports.append({'variant':variant,'top_triangles':count,'wanted_samples':int(wanted.sum()),'missing_samples':int(missing.sum()),'known_sites':known,'first_missing_tile_yx':points[:40].tolist(),'coarse_mesh_set_sha256':mesh_hash.hexdigest(),'alpha_sha256':digest(A/f'analysis/polished-edit-alpha-{variant}.npy')});np.save(A/f'analysis/audited-ground-wanted-mask-{variant}.npy',wanted);Image.fromarray(np.uint8(missing)*255).save(A/f'terrain-repair/ground-coverage-missing-{variant}.png');print(json.dumps(reports[-1]),flush=True)
(A/'analysis/edited-ground-coverage.json').write_text(json.dumps({'variants':reports,'raster_pitch':'half hex; native front-facing top faces only; no vertical skirts','map_sha256':digest(A/'work/final-map/map.hex'),'clear_report_sha256':digest(A/'analysis/obsolete-terrain-overlays.json'),'live_verified':False},indent=2))
if args.strict and any(r['missing_samples'] for r in reports):raise RuntimeError('Edited or cleared land lacks coarse ground; see edited-ground-coverage.json')
