"""Fill the expanded ocean background with a native coarse sea material.

One existing loaded sea mesh carries a low plane beneath existing water and
land. It also extends past the theatre rim, avoiding a view into empty space.
"""
import struct,shutil,math,json
from pathlib import Path
from analyze_reference import ROOT,digest
import campaign_world as _frame
from terrain_checkpoint import StepSource,inputs
from campaign_world import WORLD_WIDTH,RAW_MESH_HEIGHT
from campaign_world import TERRAIN_VARIANTS
A=_frame.WORKSPACE;S=A/'work/release-input';BACK=A/'before-ocean-background'
width=WORLD_WIDTH;height=RAW_MESH_HEIGHT;report=[]
# Background at the native sea meshes' open-ocean floor (-2.029), not -1.049: the water shader
# shades by the depth to the floor it sees, so -1.049 here against -2.445 coarse sea floor and
# -2.029 native sea meshes drew straight-edged light/dark zones (live 2048k).
# L2048 (sea fork): the native sea meshes' open-sea floor is -1.029016 (7.95M of 8.0M vertices;
# measured from the 2048l stage); -2.029 here drew one straight deep/shallow edge along the whole
# native rim (west of Kyushu, the south rim). One floor level everywhere: match them.
OCEAN_Y=-1.029016;OLD_Y=(-1.049016,-2.029016,-1.029016)
water_source=json.loads((A/'analysis/native-water-restoration.json').read_text(encoding='utf-8'))
step=StepSource("ocean-background",{**inputs(),'implementation_sha256':digest(Path(__file__)),'native_water_geometry_sha256':water_source['native_water_geometry_sha256']})
for variant in TERRAIN_VARIANTS:
 p=S/f'terrain/campaigns/{variant}/global_meshes/sea_mesh_0.rigid_model_v2';old=BACK/p.relative_to(S)
 if not old.exists():old.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,old)
 src=step.read(p)
 if src is None:continue
 mp=struct.unpack_from('<I',src,152)[0];typ,flags,size,vo,vc,io,ic=struct.unpack_from('<HH5I',src,mp);assert typ==101 and vc+4<65536
 vertex_end=mp+io;index_end=len(src)
 if vc>=4 and ic>=6:
  last=[struct.unpack_from('<4f',src,mp+vo+(vc-4+i)*16) for i in range(4)]
  quad=struct.unpack_from('<6H',src,len(src)-12)
  expected=((vc-4),(vc-1),(vc-2),(vc-4),(vc-3),(vc-1))
  if quad==expected and all(min(abs(v[1]-y) for y in OLD_Y)<1e-5 and abs(v[3]-1)<1e-5 for v in last) and abs(last[0][0]+32)<1e-5 and abs(last[0][2]+32)<1e-5:
   vc-=4;ic-=6;vertex_end-=64;index_end-=12
 vertices=src[mp+vo:vertex_end]+b''.join(struct.pack('<4f',x,OCEAN_Y,z,1.) for x,z in ((-32,-32),(width+32,-32),(-32,height+32),(width+32,height+32)))
 indices=src[mp+io:index_end]+struct.pack('<6H',vc,vc+3,vc+2,vc,vc+1,vc+3)
 head=bytearray(src[:mp+vo]);struct.pack_into('<II',head,144,len(vertices),len(indices));struct.pack_into('<HH5I',head,mp,typ,flags,vo+len(vertices)+len(indices),vo,vc+4,vo+len(vertices),ic+6)
 box=list(struct.unpack_from('<6f',src,mp+24));box[0]=min(box[0],-32);box[2]=min(box[2],-32);box[3]=max(box[3],width+32);box[5]=max(box[5],height+32);struct.pack_into('<6f',head,mp+24,*box)
 p.write_bytes(head+vertices+indices);report.append({'variant':variant,'mesh':p.name,'background_y':OCEAN_Y,'extra_vertices':4,'extra_triangles':2,'native_material_preserved':True})
step.finish()
(A/'analysis/ocean-background.json').write_text(json.dumps({'variants':report,'live_verified':False},indent=2));print('Native ocean background extended beneath the theatre rim')
