"""Remove historical authoring terraces from the baseline's vanilla footprint.

Only the translated native rectangle is restored. Expanded exterior input and
all historical baseline evidence are retained. Town terraces are authored later.
"""
import tw3k_paths
import io,json,struct,shutil
from pathlib import Path
import numpy as np
from PIL import Image,PngImagePlugin
from analyze_reference import ROOT,inventory,digest
import campaign_world as _frame
from native_height_field import parse,rebuild
from campaign_world import TERRAIN_VARIANTS
A=_frame.WORKSPACE;B=A/'terrain-before-polish';BACK=A/'terrain-repair/baseline-before-native-clean';G=tw3k_paths.GAME_DATA
S=A/'work/release-input'
_frame.check_snapshot(B);_frame.check_snapshot(BACK)
# First builds have no pre-polish snapshot yet. Seed it from the untouched
# authoring stage before cleaning height fields; never replace an existing base.
for variant in TERRAIN_VARIANTS:
 for rel in (f'terrain/campaigns/{variant}/global_props.bin',f'terrain/campaigns/{variant}/lf_height_map.dds',f'terrain/campaigns/{variant}/lf_height_map.compressed_map',f'campaign_maps/{variant}/camera_heightmap.png'):
  p=B/rel
  if not p.exists() and _frame.seed_from_stage_allowed(B):p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(S/rel,p)
REPORT=A/'analysis/native-height-baseline-restoration.json'
prior=json.loads(REPORT.read_text()) if REPORT.exists() else {}
anchor_inputs={};anchor_cameras={}
index={}
for name in ('terrain.pack','fast.pack','data.pack'):
 p=G/name
 for e in inventory(p,False)['entries']:index.setdefault(e['path'],(p,e))
def asset(key):
 p,e=index[key]
 if e['compressed']:raise RuntimeError('Native height input is compressed')
 with p.open('rb') as f:f.seek(e['offset']);return f.read(e['size'])
def retain(p):
 q=BACK/p.relative_to(B);q.parent.mkdir(parents=True,exist_ok=True)
 if not q.exists():shutil.copy2(p,q)
def put(p,body):
 retain(p);temp=p.with_suffix(p.suffix+'.native-clean.tmp');temp.write_bytes(body);temp.replace(p)
reports=[]
for variant in TERRAIN_VARIANTS:
 props_hash=digest(B/f'terrain/campaigns/{variant}/global_props.bin');anchor_inputs[variant]=props_hash
 anchor=A/f'terrain-repair/resource-scene-anchor-inputs/{variant}/{props_hash}/camera_heightmap.png'
 if not anchor.exists():
  original_camera=B/f'campaign_maps/{variant}/camera_heightmap.png'
  if prior.get('resource_scene_anchor_inputs',{}).get(variant)==props_hash:
   original_camera=Path(prior['historical_backup'])/f'campaign_maps/{variant}/camera_heightmap.png'
  anchor.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(original_camera,anchor)
 anchor_cameras[variant]={'path':str(anchor),'sha256':digest(anchor)}
 pre=f'terrain/campaigns/{variant}/';key=pre+'lf_height_map.dds';p=B/key;raw=p.read_bytes();h,w=struct.unpack_from('<2I',raw,12);values=np.frombuffer(raw,dtype='<u2',offset=128).reshape(h,w).copy();native=asset(key);nh,nw=struct.unpack_from('<2I',native,12);original=np.frombuffer(native,dtype='<u2',offset=128).reshape(nh,nw)
 assert (w,h)==(_frame.LF_W,_frame.LF_H) and nw==_frame.NATIVE_LF_W and nh==_frame.NATIVE_LF_H
 rect=values[_frame.NATIVE_OFFSET_LF_FILE[1]:_frame.NATIVE_OFFSET_LF_FILE[1]+nh,_frame.NATIVE_OFFSET_LF_FILE[0]:_frame.NATIVE_OFFSET_LF_FILE[0]+nw];changed=int(np.count_nonzero(rect!=original));maximum=int(np.max(np.abs(rect.astype(np.int32)-original.astype(np.int32))));rect[:]=original
 put(p,raw[:128]+values.tobytes())
 cp=B/(pre+'lf_height_map.compressed_map');src=cp.read_bytes();cw,ch,_=parse(src);assert (cw,ch)==(w,h);pad=np.pad(values,((0,(-h)%16),(0,(-w)%16)),mode='edge')
 # 16x16 blocks row by row as one (n,256) array: a uniform block is its first sample, any other its 512 bytes.
 tiles=np.ascontiguousarray(pad.astype('<u2',copy=False).reshape(pad.shape[0]//16,16,pad.shape[1]//16,16).swapaxes(1,2)).reshape(-1,256);flat=tiles.tobytes()
 blocks=[b'\0'+struct.pack('<H',v) if u else b'\x8f'+flat[i*512:i*512+512] for i,(u,v) in enumerate(zip((tiles==tiles[:,:1]).all(1).tolist(),tiles[:,0].tolist()))]
 put(cp,rebuild(src,blocks))
 camera=f'campaign_maps/{variant}/camera_heightmap.png';cp=B/camera;im=Image.open(cp);arr=np.asarray(im).copy();ni=Image.open(io.BytesIO(asset(camera)));assert ni.size==(_frame.NATIVE_TILE_W,_frame.NATIVE_TILE_H) and arr.shape==(_frame.TILE_H,_frame.TILE_W);arr[_frame.NATIVE_CAMERA_FILE_SLICE]=np.asarray(ni)
 png=Image.fromarray(arr);meta=PngImagePlugin.PngInfo()
 for key,value in im.info.items():
  if isinstance(value,str):meta.add_text(key,value)
 stream=io.BytesIO();png.save(stream,format='PNG',pnginfo=meta);put(cp,stream.getvalue())
 reports.append({'variant':variant,'old_artificial_samples_replaced':changed,'maximum_old_error_uint16':maximum,'native_rectangle_byte_exact':bool(np.array_equal(rect,original)),'native_source_sha256':__import__('hashlib').sha256(native).hexdigest(),'clean_baseline_sha256':digest(p)})
 print(variant,'restored',changed,'artificial native-footprint samples',flush=True)
REPORT.write_text(json.dumps({'variants':reports,'first_restoration':prior.get('first_restoration',prior.get('variants',reports)),'historical_backup':str(BACK),'resource_scene_anchor_inputs':anchor_inputs,'resource_scene_anchor_cameras':anchor_cameras,'native_offset_pixels_north_left':[512,1536],'source':'unmodified installed vanilla terrain/fast/data packs','live_verified':False},indent=2))
