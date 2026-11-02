"""Keep the GPU sea height texture identical to the expanded native CPU cache.

The stage cache is rebuilt from the base (terrain-before-polish) on every run, then the frontier rivers'
water level (polish_campaign_terrain.py RIVER CHANNELS, analysis/river-water-level.npy) is written over
their channels, as vanilla raises its sea height map along the Yellow River and Yangtze.
"""
import tw3k_paths
import struct,json,hashlib
import numpy as np
from pathlib import Path
from analyze_reference import inventory,digest
import campaign_world as _frame
from campaign_world import TERRAIN_VARIANTS
import fastops
# The game's absolute path, as audit_ground_seams GAME and expand_campaign_water_colour GAME (was the repo's
# parent directory + Steam, which points into .claude/worktrees for a worktree there)
GAME=tw3k_paths.GAME_DATA
A=_frame.WORKSPACE;S=A/'work/release-input';game=GAME/'terrain.pack'
RIVER_LEVEL=A/'analysis/river-water-level.npy'
if RIVER_LEVEL.exists():
 # Polish stamps the level with the map.hex and canvas it was drawn on (river-water-level.json): a level
 # left from another map would raise the water over hexes that are no longer river (the shape check
 # below catches only another canvas size).
 _stamp=RIVER_LEVEL.with_suffix('.json');_want={'map_sha256':digest(A/'work/final-map/map.hex'),'canvas':_frame.canvas_stamp()}
 _found=json.loads(_stamp.read_text(encoding='utf-8')) if _stamp.exists() else None
 if _found!=_want:raise SystemExit(f'{RIVER_LEVEL} was drawn on {_found or "an unstamped map"}, current is {_want}: rerun polish_campaign_terrain.py')
index={e['path']:e for e in inventory(game,False)['entries']};reports=[]
for variant in TERRAIN_VARIANTS:
 key=f'terrain/campaigns/{variant}/lf_sea_height_map.dds';cmkey=key.replace('.dds','.compressed_map')
 # Always from the base: a stage copy left from an earlier canvas of the SAME size but other padding
 # passed the old size-only check (2048m: the stage cache dated from a 658/230 padding, so vanilla's
 # raised river levels sat 230 hexes west and 145 south of their rivers, and half the Yellow River was dry).
 # fastops.cm_read: the per-block native_height_field.decode loop, vectorised (262k blocks on the 2048 canvas)
 data=(A/'terrain-before-polish'/cmkey).read_bytes();a=fastops.cm_read(data);h,w=a.shape
 assert (w,h)==(_frame.W*_frame.SEA_PX_PER_HEX,(2*_frame.H+1)*_frame.SEA_PX_PER_HEX//2),(w,h)
 river_px=0
 if RIVER_LEVEL.exists():
  lvl=np.load(RIVER_LEVEL)
  assert lvl.shape==(_frame.TILE_H,_frame.TILE_W),(lvl.shape,'river level from another canvas: rerun polish_campaign_terrain.py')
  k=_frame.SEA_PX_PER_HEX//_frame.TILE_PER_HEX   # sea pixels per tile cell
  lvl=np.repeat(np.repeat(lvl,k,0),k,1)[:h,:w][::-1]   # tile rows run south-up, the cache north-first
  raw=np.rint(np.clip(_frame.NORM+(np.nan_to_num(lvl,nan=0.)-_frame.LEVEL)/_frame.GAIN,0,1)*65535).astype('<u2')
  wet=np.isfinite(lvl);river_px=int(wet.sum())
  # River channels lie outside the native map (apply_rivers NATIVE_GUARD): native sea heights stay byte-identical.
  native=np.zeros((h,w),bool);_nx,_ny=_frame.NATIVE_OFFSET_SEA;_nh=(2*_frame.NATIVE_H+1)*_frame.SEA_PX_PER_HEX//2
  native[h-_ny-_nh:h-_ny,_nx:_nx+_frame.NATIVE_W*_frame.SEA_PX_PER_HEX]=True
  assert not (wet&native).any(),'river water level inside the native map'
  a=np.where(wet,raw,a).astype('<u2')
  data=bytes(fastops.cm_write(data,a))
 (S/cmkey).write_bytes(data)
 e=index[key]
 with game.open('rb') as f:f.seek(e['offset']);header=bytearray(f.read(128))
 oldh,oldw=struct.unpack_from('<2I',header,12);struct.pack_into('<2I',header,12,h,w);struct.pack_into('<I',header,20,w*2)
 path=S/key;backup=A/'layout-v2/sea-dds-before'/key
 if path.exists() and not backup.exists():backup.parent.mkdir(parents=True,exist_ok=True);backup.write_bytes(path.read_bytes())
 path.write_bytes(header+a.tobytes());assert path.stat().st_size==128+w*h*2
 # Every pixel of the written cache decodes to the DDS payload (was the first pixel of 512 sampled blocks).
 assert np.array_equal(fastops.cm_read(data),a)
 reports.append({'variant':variant,'native_grid':[oldw,oldh],'expanded_grid':[w,h],'river_water_pixels':river_px,'dds_sha256':digest(path),'cm_sha256':digest(S/cmkey),'dds_payload_equals_decoded_cm':True})
(A/'layout-v2/sea-height-texture-sync.json').write_text(json.dumps({'reports':reports,'visual_acceptance_pending':True},indent=2)+'\n');print(json.dumps(reports))
