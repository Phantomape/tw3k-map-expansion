"""Ray-test authored towns against actual coarse triangles, not bounding boxes."""
import json,struct,math
from pathlib import Path
import numpy as np
from analyze_reference import ROOT
import campaign_world as _frame
from final_terrain_surface import display_tile,surface_at_display
A=_frame.WORKSPACE;S=A/'work/release-input';DX=_frame.DX;plan=json.loads((A/'work/current-region-design.json').read_text(encoding='utf-8'));mesh=[]
for path in (S/'terrain/campaigns/3k_dlc07_main_map/global_meshes').glob('land_mesh_*.rigid_model_v2'):
 b=path.read_bytes();mp=struct.unpack_from('<I',b,152)[0];box=struct.unpack_from('<6f',b,mp+24);typ,flags,size,vo,vc,io,ic=struct.unpack_from('<HH5I',b,mp);v=np.frombuffer(b,dtype='<f4',offset=mp+vo,count=vc*4).reshape(vc,4);idx=np.frombuffer(b,dtype='<u2',offset=mp+io,count=ic).reshape(-1,3);a=v[idx[:,0],:3].copy();d=v[idx[:,1],:3]-a;e=v[idx[:,2],:3]-a;det=d[:,0]*e[:,2]-e[:,0]*d[:,2];mesh.append((box,a,d,e,det))
surface=np.load(A/'analysis/polished-surface-3k_dlc07_main_map.npy')
result=[]
for c in plan['regions']:
 # Probe where the engine draws the town (display position, raw frame), not the
 # SW corner q*DX of its tile block.
 q,r=c['town'];x,z=(t*DX/2 for t in display_tile(q,r));heights=[]
 for box,a,d,e,det in mesh:
  if not(box[0]-1e-3<=x<=box[3]+1e-3 and box[2]-1e-3<=z<=box[5]+1e-3):continue
  # Coarsened grids put a vertex on each town (snap_grid_to_towns): tolerances above
  # float32 noise, and vertical skirts (near-zero plan area) never count.
  ok=abs(det)>1e-6;u=np.zeros(len(det));v=np.zeros(len(det));u[ok]=((x-a[ok,0])*e[ok,2]-(z-a[ok,2])*e[ok,0])/det[ok];v[ok]=(d[ok,0]*(z-a[ok,2])-d[ok,2]*(x-a[ok,0]))/det[ok];hit=ok&(u>=-1e-3)&(v>=-1e-3)&(u+v<=1.001);ys=a[:,1]+u*d[:,1]+v*e[:,1];heights.extend(float(y) for y in ys[hit])
 result.append({'region':c['region'],'town':c['town'],'expected_surface_height':surface_at_display(surface,q,r),'triangle_ground_heights':heights})
missing=[c for c in result if not c['triangle_ground_heights']];wrong=[c for c in result if c['triangle_ground_heights'] and abs(max(c['triangle_ground_heights'])-c['expected_surface_height'])>.04];report={'towns':len(result),'towns_with_triangles':len(result)-len(missing),'towns_with_matched_terraces':len(result)-len(wrong)-len(missing),'terrace_tolerance':.04,'missing':missing,'wrong_top_surface':wrong,'results':result,'live_verified':False};(A/'analysis/city-ground-triangle-coverage.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in report.items() if k!='results'}));assert not missing,'Some authored towns still lack coarse ground';assert not wrong,'Some authored towns do not match the synchronized height field'
