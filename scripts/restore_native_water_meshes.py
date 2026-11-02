"""Restore native river/ocean triangles in the expanded raw mesh frame."""
import tw3k_paths
import argparse,json,struct,hashlib
from pathlib import Path
import numpy as np
from analyze_reference import ROOT,inventory,digest
import campaign_world as _frame
from campaign_world import DX,WORLD_WIDTH,RAW_MESH_HEIGHT
from campaign_world import TERRAIN_VARIANTS

parser=argparse.ArgumentParser();parser.add_argument('--audit',action='store_true');args=parser.parse_args()
A=_frame.WORKSPACE;S=A/'work/release-input'
G=tw3k_paths.GAME_DATA;index={}
for name in ('terrain.pack','fast.pack','data.pack'):
 for e in inventory(G/name,False)['entries']:
  k=e['path']
  if any(k.startswith(f'terrain/campaigns/{v}/global_meshes/sea_mesh_') for v in TERRAIN_VARIANTS) and k.endswith('.rigid_model_v2'):
   index.setdefault(k,(G/name,e))
assert len(index)==125*len(TERRAIN_VARIANTS)   # 125 vanilla sea meshes per variant
records=[]
for key,(pack,e) in sorted(index.items()):
 with pack.open('rb') as f:f.seek(e['offset']);source=bytearray(f.read(e['size']))
 mp=struct.unpack_from('<I',source,152)[0];typ,flags,size,vo,vc,io,ic=struct.unpack_from('<HH5I',source,mp)
 assert typ==101 and io-vo==vc*16
 vertices=np.frombuffer(source,dtype='<f4',offset=mp+vo,count=vc*4).reshape(vc,4)
 vertices[:,0]+=_frame.NATIVE_SHIFT_RAW[0];vertices[:,2]+=_frame.NATIVE_SHIFT_RAW[1]
 box=list(struct.unpack_from('<6f',source,mp+24))
 for i in (0,3):box[i]+=_frame.NATIVE_SHIFT_RAW[0]
 for i in (2,5):box[i]+=_frame.NATIVE_SHIFT_RAW[1]
 struct.pack_into('<6f',source,mp+24,*box)
 target=S/key;before=digest(target)
 if not args.audit:target.write_bytes(source)
 else:
  current=target.read_bytes();cm=struct.unpack_from('<I',current,152)[0];ct,cf,cs,cv,cn,ci,cc=struct.unpack_from('<HH5I',current,cm)
  extra=4 if key.endswith('/sea_mesh_0.rigid_model_v2') else 0
  assert (ct,cf,cv)==(typ,flags,vo),key
  assert (cn,cc)==(vc+extra,ic+(6 if extra else 0)),key
  assert current[cm+cv:cm+cv+vc*16]==source[mp+vo:mp+io],key+' native water vertices changed'
  assert current[cm+ci:cm+ci+ic*2]==source[mp+io:mp+io+ic*2],key+' native water triangles changed'
  if extra:
   actual=np.frombuffer(current,dtype='<f4',offset=cm+cv+vc*16,count=16).reshape(4,4)
   Y=-1.029016   # extend_campaign_ocean_background.OCEAN_Y (fbd099fd moved it to the native open-sea floor)
   expected=np.array([[-32,Y,-32,1],[WORLD_WIDTH+32,Y,-32,1],[-32,Y,RAW_MESH_HEIGHT+32,1],[WORLD_WIDTH+32,Y,RAW_MESH_HEIGHT+32,1]],dtype='<f4')
   assert np.array_equal(actual,expected),key+' unexpected ocean background'
   assert struct.unpack_from('<6H',current,cm+ci+ic*2)==(vc,vc+3,vc+2,vc,vc+1,vc+3),key
  else:assert len(current)==len(source) and current==source,key+' native water header changed'
 records.append({'path':key,'native_vertices':vc,'native_indices':ic,'input_sha256':before,'output_sha256':digest(target),'native_translated_sha256':hashlib.sha256(source).hexdigest()})
report={'native_meshes':len(records),'native_vertices_and_triangles_preserved':True,'raw_mesh_translation':list(_frame.NATIVE_SHIFT_RAW),'mode':'audit' if args.audit else 'restore','meshes':records}
report['native_water_geometry_sha256']=hashlib.sha256(''.join(r['path']+':'+r['native_translated_sha256']+'\n' for r in records).encode()).hexdigest()
(A/'analysis'/('native-water-preservation.json' if args.audit else 'native-water-restoration.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
print('Native water '+report['mode']+': '+str(len(records))+' meshes verified',flush=True)
