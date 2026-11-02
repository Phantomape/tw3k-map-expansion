"""Structural acceptance for the current terrain pass, separate from live QA."""
import tw3k_paths
import struct,json,math
from pathlib import Path
import numpy as np
from analyze_reference import ROOT,inventory,read_hex,digest
import campaign_world as _frame
from native_height_field import parse,decode
from campaign_world import TERRAIN_VARIANTS
A=_frame.WORKSPACE;S=A/'work/release-input';W,H=_frame.W,_frame.H;TW,TH=_frame.TILE_W,_frame.TILE_H;n=TW*TH
DT=np.dtype([('path','<i4'),('flags','<u2'),('x','<u2'),('y','<u2'),('climate','u1'),('extra','u1'),('lo','<f4'),('hi','<f4')]);report={'live_acceptance':False,'variants':[]}
# The original game contains one opaque non-asset marker (path == path count).
# Permit only that exact native record at its translated native cell.
donor=tw3k_paths.GAME_DATA / 'terrain.pack'
entry=next(e for e in inventory(donor,False)['entries'] if e['path']=='terrain/campaigns/3k_main_map/tile_map.tiles')
with donor.open('rb') as f:f.seek(entry['offset']);original_cache=f.read(entry['size'])
original_n=_frame.NATIVE_TILE_W*_frame.NATIVE_TILE_H;original_path_count=struct.unpack_from('<I',original_cache,original_n*60)[0];native_markers={}
for layer in range(3):
 native=np.frombuffer(original_cache,dtype=DT,offset=layer*original_n*20,count=original_n)
 for at in np.flatnonzero((native['path']>=original_path_count)&(native['flags']!=16)):
  row=native[at].copy();row['x']+=_frame.NATIVE_TILE_X0;row['y']+=_frame.NATIVE_TILE_Y0;translated=-_frame.native_cache_owner(int(at))-1
  native_markers[layer,translated]=row.tobytes()
fine_generated=np.load(A/'analysis/cache-fine-coverage-mask.npy').ravel()
for variant in TERRAIN_VARIANTS:
 pre=S/f'terrain/campaigns/{variant}';meshes=0;triangles=0
 for p in (pre/'global_meshes').glob('land_mesh_*.rigid_model_v2'):
  b=p.read_bytes();mp=struct.unpack_from('<I',b,152)[0];t,flags,size,vo,vc,io,ic=struct.unpack_from('<HH5I',b,mp);assert t==101 and io-vo==vc*16 and len(b)==mp+size and ic%3==0
  verts=np.frombuffer(b,dtype='<f4',offset=mp+vo,count=vc*4).reshape(vc,4);indices=np.frombuffer(b,dtype='<u2',offset=mp+io,count=ic);assert np.isfinite(verts).all() and indices.max()<vc and vc<65536
  bb=np.array(struct.unpack_from('<6f',b,mp+24));assert np.all(verts[:,:3].min(axis=0)>=bb[:3]-.001) and np.all(verts[:,:3].max(axis=0)<=bb[3:]+.001)
  cm=p.with_suffix('.compressed_map').read_bytes();cw,ch,bs=parse(cm);cb=np.array(struct.unpack_from('<6f',cm,26));assert np.all(abs(cb[[0,2,3,5]]-bb[[0,2,3,5]])<.001)
  meshes+=1;triangles+=ic//3
 dds=(pre/'lf_height_map.dds').read_bytes();dh,dw=struct.unpack_from('<2I',dds,12);assert(dw,dh)==(_frame.LF_W,_frame.LF_H);pixels=np.frombuffer(dds,dtype='<u2',offset=128).reshape(dh,dw)
 cw,ch,blocks=parse((pre/'lf_height_map.compressed_map').read_bytes());assert(cw,ch)==(dw,dh);cols=math.ceil(cw/16);sampled=0
 # Hundreds of independent blocks include every part of the theatre and seams.
 for i in range(0,len(blocks),487):
  x=i%cols*16;y=i//cols*16;bw=min(16,dw-x);bh=min(16,dh-y);decoded=np.array(decode(blocks[i]),dtype='<u2').reshape(16,16);assert np.array_equal(decoded[:bh,:bw],pixels[y:y+bh,x:x+bw]);sampled+=1
 cache=(pre/'tile_map.tiles').read_bytes();assert len(cache)>n*60;tail=cache[n*60:];pos=4;paths=[]
 for i in range(struct.unpack_from('<I',tail)[0]):z=struct.unpack_from('<H',tail,pos)[0];pos+=2;paths.append(tail[pos:pos+z].decode());pos+=z
 counts={}
 for layer in range(3):
  a=np.frombuffer(cache,dtype=DT,offset=layer*n*20,count=n);child=a['path']<0;parents=-a['path'][child]-1;assert np.all((parents>=0)&(parents<n));assert np.all(a['path'][parents]>=0) and np.all(a['flags'][parents]!=16)
  opaque=(a['path']>=len(paths))&(a['flags']!=16)
  for at in np.flatnonzero(opaque):assert native_markers.get((layer,int(at)))==a[at].tobytes(),'Unexpected non-asset cache marker'
  root=(a['path']>=0)&(a['path']<len(paths))&(a['flags']!=16);assert np.all(np.isfinite(a['lo'][root])) and np.all(a['hi'][root]>=a['lo'][root])
  owner=np.where(child,-a['path']-1,np.arange(n));population=np.bincount(owner[a['flags']!=16],minlength=n)
  for name,size in (('generic_8x8',16),('generic_1x1',2)):
   ident=paths.index('terrain\\tiles\\campaign\\generic\\'+name+'\\');selected=root&(a['path']==ident);at=np.flatnonzero(selected&fine_generated);expected=np.minimum(size,TW-a['x'][at].astype(int))*np.minimum(size,TH-a['y'][at].astype(int));assert np.all(population[at]==expected),(variant,layer,name,int(np.sum(population[at]!=expected)))
   counts[str(layer)+':'+name]=len(at)
 report['variants'].append({'variant':variant,'meshes':meshes,'triangles':triangles,'sampled_exact_height_blocks':sampled,'native_land_cache_footprints':counts,'cache_parents_valid':True})
map_=read_hex(A/'work/final-map/map.hex');report['sdk_total_regions']=len(map_['land_regions'])+len(map_['sea_regions']);import map_hex as _mh;_v=_mh.read(_frame.VANILLA_MAP);_d=json.loads(_frame.DESIGN_PATH.read_text(encoding='utf-8'))
_expected_land=len(_v['land_regions'])+len(_d['regions'])+len(_d.get('sdk_internal_land_regions',[]));_expected_sea=len([k for k in _v['sea_regions'] if k not in _d.get('retired_regions',{})])+len({(x['region'] if isinstance(x,dict) else x) for x in _d.get('imported_sea_regions',[])+_d.get('new_sea_regions',[])})
assert (len(map_['land_regions']),len(map_['sea_regions']))==(_expected_land,_expected_sea),((len(map_['land_regions']),len(map_['sea_regions'])),(_expected_land,_expected_sea));report['map_sha256']=map_['sha256'];report['status']='structural checks passed; live terrain acceptance pending'
(A/'analysis/polish-structural-acceptance.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)
