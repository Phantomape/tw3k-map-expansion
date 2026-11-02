"""Tile bake (pipeline-bake-design.md 3.3, step 1): the fine-tile list records of the edited ground.

polish_campaign_terrain.py's cache segment calls bake() right after it writes tile_map.tiles and the two
tile_list.bin files, on the final LF (sync_final_terrain_surface smooths the native/canvas boundary before
it runs polish --cache-only). The two passes run in this order, unchanged from the separate pipeline steps
they were (repair_native_placements.py, clear_obsolete_terrain_overlays.py):

  place_floors()    rebuilds both tile lists from the verified canonical pair when explicitly configured,
                    otherwise terrain-before-polish: native placements outside the edited
                    land kept, vanilla floors and special pieces the SDK export lost restored, the edited cells floored as polish
                    laid them in the cache (land generic, generic_sea), salt/road/mountain/coast records added;
  clear_overlays()  culls native special tiles whose footprint (plus two cells) meets changed ground and refills
                    their uncovered land cells with generic_1x1; writes cleared-terrain-replacement-mask-*.npy.

Height ranges (repair_calibrated_native_tile_ranges, sync_*_cache_*ranges) stay separate steps for now.
"""
import tw3k_paths
import argparse,struct,json,re,shutil,math,functools
from pathlib import Path
import numpy as np
from analyze_reference import ROOT,inventory,read_hex
import campaign_world as _frame
from native_tile_dimensions import placement_footprint,FLOOR_TILE,floor_relay,decode_dimensions
from read_tile_list import parse
from campaign_world import TERRAIN_VARIANTS
from native_edit_mask import edited_tiles,TOLERANCE
import sea_floor_tiles as _sf
import native_tile_grid as _ntg
from sea_floor_tiles import FLOOR_FAMILY

_GAME=tw3k_paths.GAME_DATA


@functools.lru_cache(maxsize=None)
def _pack_index():
 idx={}
 for pn in ('terrain.pack','fast.pack','data.pack'):
  p=_GAME/pn
  for e in inventory(p,False)['entries']:idx.setdefault(e['path'],(p,e))
 return idx


def _pack_data(k):
 p,e=_pack_index()[k]
 with p.open('rb') as f:f.seek(e['offset']);return f.read(e['size'])


@functools.lru_cache(maxsize=None)
def tile_dims(path):
 """Tile path -> (w, h) cells unrotated, from the vanilla tile database (read-only, one record per path)."""
 parts=path.strip(chr(92)).split(chr(92));k='terrain/tiles/campaign/_tile_database/tiles/'+parts[-2]+'_'+parts[-1]+'.bin'
 if k not in _pack_index():return (8,8)
 return decode_dimensions(_pack_data(k))


def record_footprint(path,flags):
 """Rotated cell footprint (w, h) of a tile record: generic floors by name, everything else by the database."""
 match=re.search(r'generic_(\d+)x(\d+)',path) or re.search(r'generic_sea.(\d+)x(\d+)',path)   # the database gives generic_sea 1x1
 return placement_footprint(tuple(map(int,match.groups())) if match else tile_dims(path),flags)


CULL_APRON=2   # cells: clear_overlays culls a native special whose footprint plus this apron meets the edit


def cull_window(x,y,w,h,shape):
 """(x0, x1, y0, y1) of a record footprint plus CULL_APRON, clipped to the tile grid: clear_overlays' cull test."""
 return max(0,x-CULL_APRON),min(shape[1],x+w+CULL_APRON),max(0,y-CULL_APRON),min(shape[0],y+h+CULL_APRON)


def cull_reach(edit):
 """Cells whose record would meet the edit within cull_window (the edit grown by CULL_APRON, square): the native
 specials a footprint there loses to clear_overlays where the ground changed (stage_invariants.native_vanilla_records)."""
 from scipy import ndimage
 return ndimage.binary_dilation(edit,np.ones((2*CULL_APRON+1,)*2,bool))


def land_cover(paths,records):
 """Cells covered by the records that are not seabed (generic_sea)."""
 TH,TW=_frame.TILE_H,_frame.TILE_W;cover=np.zeros((TH,TW),bool)
 for r in records:
  path=paths[r[1]]
  if 'generic_sea' in path:continue
  w,h=record_footprint(path,r[5]);cover[max(0,r[4]):max(0,r[4]+h),max(0,r[3]):max(0,r[3]+w)]=True
 return cover


def vanilla_specials(native_rows,native_paths,paths,have,blocked,biome):
 """Vanilla special records (not generic floors, not seabed) the base lacks, to restore on unedited native ground:
 wholly inside the native rectangle, no cell blocked, the tile climate rule equal to the record's climate. Returns
 (records on the canvas grid, skip counts by reason)."""
 NX0,NY0,NX1,NY1=_frame.NATIVE_TILE_X0,_frame.NATIVE_TILE_Y0,_frame.NATIVE_TILE_X1,_frame.NATIVE_TILE_Y1
 out=[];skip={'path_not_listed':0,'outside_native':0,'blocked':0,'climate':0}
 for v in native_rows:
  name=native_paths[v[1]]
  if re.search(r'generic_(\d+)x(\d+)',name) or FLOOR_FAMILY.search(name.replace(chr(92),'/')):continue
  x,y=_ntg.list_xy(v[3],v[4])
  if (name,v[2],x,y,v[5]) in have:continue
  if name not in paths:skip['path_not_listed']+=1;continue
  w,h=record_footprint(name,v[5]);x0,x1,y0,y1=x,x+w,y,y+h
  if not(NX0<=x0 and x1<=NX1 and NY0<=y0 and y1<=NY1):skip['outside_native']+=1;continue
  if np.any(blocked[y0:y1,x0:x1]):skip['blocked']+=1;continue
  if np.any(biome[y0:y1,x0:x1]!=v[2]):skip['climate']+=1;continue
  out.append((v[0],paths.index(name),v[2],x,y,v[5],v[6],v[7]))
 return out,skip


def seabed_replaced(edited,replaced,land):
 """A vanilla generic_sea record (footprint arrays: edit request, polish's cache replacement, map.hex land) is
 replaced when it meets the edit and every cell is replaced or land: polish floored the replaced ones, the land
 ones take land floors. A record reaching unreplaced sea stays as vanilla."""
 return bool(np.any(edited|replaced)) and bool((replaced|land).all())


def place_floors():
 """repair_native_placements.py: runtime floor placements, rebuilt from terrain-before-polish."""
 from repair_campaign_river_roots import Assets
 A=_frame.WORKSPACE;S=A/'work/release-input';BASE=A/'terrain-before-polish';BACK=A/'before-placement-probe';W,H=_frame.W,_frame.H;TW,TH=_frame.TILE_W,_frame.TILE_H
 import tile_source
 tile_inputs=tile_source.load(A,TERRAIN_VARIANTS,_frame.canvas_stamp(),(_frame.TILE_W,_frame.TILE_H))
 native_assets=Assets(original_only=True)
 m=read_hex(A/'work/final-map/map.hex');names=m['land_regions']+m['sea_regions'];b=(A/'work/final-map/map.hex').read_bytes();d=np.frombuffer(b[-4-W*H*16:-4],np.uint8).reshape(H,W,16);ids=(d[:,:,1].astype(int)<<5)+(d[:,:,0].astype(int)>>3)-1;p=json.loads((A/'work/current-region-design.json').read_text(encoding='utf-8'));authored=np.isin(ids,[names.index(c['region']) for c in p['regions'] if c.get('introduced_in_bulk')]);continuous=_frame.load_canvas_array(A/'analysis/continuous-land-mask.npy');yy,xx=np.mgrid[:TH,:TW];q=(xx//2).clip(0,W-1);r=(yy//2).clip(0,H-1);mask=authored[r,q]|continuous[r,q]|(xx<_frame.NATIVE_TILE_X0)|(xx>=_frame.NATIVE_TILE_X1)|(yy<_frame.NATIVE_TILE_Y0)|(yy>=_frame.NATIVE_TILE_Y1);report=[]
 # map.hex land per tile cell, vanilla playable hexes left out: their records stay as vanilla lists them
 # (vanilla playable ground unchanged), so a seabed reaching them is replaced only wholly inside the replacement
 _vm=read_hex(_frame.VANILLA_MAP);_vn=_vm['land_regions']+_vm['sea_regions'];_vb=_frame.VANILLA_MAP.read_bytes();_vd=np.frombuffer(_vb[-4-_vm['width']*_vm['height']*16:-4],np.uint8).reshape(_vm['height'],_vm['width'],16)
 _vid=(_vd[:,:,1].astype(int)<<5)+(_vd[:,:,0].astype(int)>>3)-1;_vplay=np.array([not('non_playable' in n or 'sea' in n) for n in _vn]+[False])[np.where((_vid>=0)&(_vid<len(_vn)),_vid,len(_vn))]
 vplay_hex=np.zeros((H,W),bool);vplay_hex[_frame.NATIVE_HEX_SLICE]=_vplay
 _tq,_tr=_frame.tile_hex(np.arange(TW,dtype=np.int32)[None,:],np.arange(TH,dtype=np.int32)[:,None]);land_cells=(((d[:,:,0]&3)!=1)&~vplay_hex)[_tr,_tq]
 # Restore the valid cache after diagnostic omission.
 for variant in TERRAIN_VARIANTS:
  # The renderer cache is regenerated from vanilla by the terrain authoring pass.
  # Edited = polish alpha on ground whose LF really changed (shared definition,
  # no halo: a floor beside changed ground keeps its tile and only its range is
  # recalibrated). Alpha alone also covers untouched native rings.
  mask=(np.load(A/'analysis/polished-edit-alpha.npy')>.02)&edited_tiles(variant,halo=0)
  # Road pieces laid by polish_campaign_terrain.py (road_tiles.py): no generic floor under their
  # cells, also on unedited native ground where a new road joins (those floors are re-laid around them).
  _rp=A/'analysis/road-tile-records.json';road_recs=json.loads(_rp.read_text(encoding='utf-8')) if _rp.exists() else []
  road_cover=np.load(A/'analysis/road-tile-cover.npy') if road_recs else np.zeros((TH,TW),bool)
  mask|=road_cover
  # Coast pieces (coast_tiles.py, polish): no vanilla floor is kept or restored under them, also on edited
  # native shores where the alpha/LF rule alone would leave native floors listed.
  _cc=A/'analysis/coast-tile-cover.npy'
  if _cc.exists():mask|=np.load(_cc)
  # [fix-batch floor-split] cells polish_campaign_terrain.py re-stamped as 1-cell generic_1x1 cache owners
  # (split 2x2 floor groups, analysis/floor-split-cells.npy): every one gets its list record.
  _fsp=A/'analysis/floor-split-cells.npy'
  if _fsp.exists():mask|=np.load(_fsp)
  # The tile climate polish_campaign_terrain.py laid its floors and display
  # cache with (authored bands, inherited native climate, town overrides). A
  # separate per-ground-type climate here put temperate (green) floors under
  # desert towns on 'plains' hexes while the cache said arid (Yutian, Qule,
  # live check 2026-10-20 d).
  biome=np.load(A/'analysis/polished-tile-climate.npy').astype(np.uint8);assert biome.shape==(TH,TW)
  # Open-sea cells polish floored with generic_sea (sea_floor_tiles.py); None before that change: land floors only.
  _sfw=A/_sf.WATER_FILE;sea_water=np.load(_sfw) if _sfw.exists() else None;assert sea_water is None or sea_water.shape==(TH,TW)
  surface=np.load(A/f'analysis/polished-surface-{variant}.npy');norm=np.clip(_frame.NORM+(surface-_frame.LEVEL)/_frame.GAIN,0,1)
  # Native ranges include two placement cells around each nominal footprint.
  # Calibration against 161 vanilla 1x1 records matches within 0.000047.
  dds=(S/f'terrain/campaigns/{variant}/lf_height_map.dds').read_bytes();dh,dw=struct.unpack_from('<2I',dds,12)
  lf=np.frombuffer(dds,dtype='<u2',offset=128).reshape(dh,dw)[::-1]
  # Floor footprints (FLOOR_TILE sizes, ~4M records) read their range from sliding min/max windows over
  # per-tile LF extremes: tiles [y-2,y+size+2) x [x-2,x+size+2) clipped to the map, the same texels as the slice.
  tile_ext=(lambda b:(b.min(axis=(1,3)),b.max(axis=(1,3))))(lf.reshape(dh//4,4,dw//4,4)) if dh%4==0 and dw%4==0 else None;windows={}
  def window(size):
   if size not in windows:
    out=[]
    for a,f,fill in zip(tile_ext,(np.minimum,np.maximum),(65535,0)):
     p_=np.pad(a,((2,size+1),(2,size+1)),constant_values=fill);rows_=p_[:a.shape[0]].copy()
     for k in range(1,size+4):f(rows_,p_[k:k+a.shape[0]],out=rows_)
     win=rows_[:,:a.shape[1]].copy()
     for k in range(1,size+4):f(win,rows_[:,k:k+a.shape[1]],out=win)
     out.append(win)
    windows[size]=out
   return windows[size]
  def height_range(x,y,size):
   if tile_ext is not None and size in FLOOR_TILE and 0<=y<dh//4 and 0<=x<dw//4:
    low,high=window(size);return float(low[y,x])/65535,float(high[y,x])/65535
   values=lf[max(0,(y-2)*4):min(dh,(y+size+2)*4),max(0,(x-2)*4):min(dw,(x+size+2)*4)]
   return float(values.min())/65535,float(values.max())/65535
  for suffix in ('tile_list.bin','global_map/tile_list.bin'):
   rel=Path(f'terrain/campaigns/{variant}/{suffix}');source_bytes=tile_inputs.list_bytes(variant,suffix);src=source_bytes
   # Salt pans laid by polish_campaign_terrain.py (cache layer 0 + these list records).
   _sp=A/'analysis/salt-pan-records.json';salt_recs=json.loads(_sp.read_text(encoding='utf-8')) if _sp.exists() else []
   _need=sorted({s[0] for s in salt_recs}-set(parse(src)[0]))
   if _need:src=__import__('salt_pools').add_list_paths(src,_need)
   _need=sorted({s[0] for s in road_recs}-set(parse(src)[0]))
   if _need:src=__import__('salt_pools').add_list_paths(src,_need)
   # SEA FLOORS: the generic_sea floor tiles (vanilla lists carry them; added if a base list lacks one)
   _need=sorted(set(_sf.SEA_TILE.values())-set(parse(src)[0])) if sea_water is not None else []
   if _need:src=__import__('salt_pools').add_list_paths(src,_need)
   paths,cl,cp,start,rs,info=parse(src);original=parse(source_bytes)[4];gi={sz:paths.index(FLOOR_TILE[sz]) for sz in FLOOR_TILE};donors={sz:next(rec for rec in original if rec[1]==gi[sz] and rec[2]==0) for sz in FLOOR_TILE}
   np_,_,_,_,native_rows,_=parse(native_assets.read(rel.as_posix()))
   native_ids={(np_[v[1]],v[2],*_ntg.list_xy(v[3],v[4]),v[5]) for v in native_rows}   # the vanilla list base on the canvas grid
   # Keep established native placements outside the edited land. Grow only the
   # actual runtime footprint where a native generic floor is replaced.
   salt_cover=np.load(A/'analysis/salt-pan-cover.npy') if salt_recs else np.zeros((TH,TW),bool)
   # Mountain tiles (mountain_tiles.py) ride in the salt records; their cells are covered the same way.
   if (A/'analysis/mountain-tile-cover.npy').exists():salt_cover=salt_cover|np.load(A/'analysis/mountain-tile-cover.npy')
   # So do the coast tiles (coast_tiles.py: sea_coast pieces on layer 0, their generic_sea underlays on layer 2).
   if (A/'analysis/coast-tile-cover.npy').exists():salt_cover=salt_cover|np.load(A/'analysis/coast-tile-cover.npy')
   edit=mask.copy();kept=[];dropped=0;sea_replaced=0
   _crm=A/'analysis/cache-replacement-mask.npy';sea_replaced_cells=np.load(_crm) if _crm.exists() else None
   NX0,NY0,NX1,NY1=_frame.NATIVE_TILE_X0,_frame.NATIVE_TILE_Y0,_frame.NATIVE_TILE_X1,_frame.NATIVE_TILE_Y1
   for rec in rs:
    x,y=rec[3:5];name=paths[rec[1]];match=__import__('re').search(r'generic_(\d+)x(\d+)',name)
    if match:
     sx,sy=placement_footprint(tuple(map(int,match.groups())),rec[5]);x0,x1=max(0,x),min(TW,x+sx);y0,y1=max(0,y),min(TH,y+sy)
     # Select from the immutable edit request; growing coverage must not make
     # later native records newly eligible merely because of record ordering.
     native_identity=(name,rec[2],x,y,rec[5]) in native_ids
     climate_mismatch=x0<x1 and y0<y1 and np.any(biome[y0:y1,x0:x1]!=rec[2])
     in_native=NX0<=x0 and x1<=NX1 and NY0<=y0 and y1<=NY1
     if in_native and x0<x1 and y0<y1 and not climate_mismatch and not np.any(mask[y0:y1,x0:x1]):
      # Unedited native ground takes its generic floors from vanilla (added
      # below); the SDK export drops or re-ranges some of them.
      if not native_identity:dropped+=1;continue
     elif x0<x1 and y0<y1 and (not native_identity or climate_mismatch or np.any(mask[y0:y1,x0:x1])):edit[y0:y1,x0:x1]=True;continue
    # A native generic_sea floor (vanilla seabed, square N x N) that meets the edited ground is replaced
    # like a native land floor: polish removed its cache owner whole and floored every cell. Kept until
    # clear_obsolete_terrain_overlays culled it, the part of its footprint outside the edit had no fine
    # tile at all: 38,458 seabed holes inside the native map in v16, the long pale strip along the drowned
    # vanilla shore off Lin Yi (vanilla 3k_dlc07_main_map: 134 uncovered sea cells).
    sea_match=_sf.SEA_RECORD.search(name)
    if sea_match:
     n_=int(sea_match.group(1));x0,x1=max(0,x),min(TW,x+n_);y0,y1=max(0,y),min(TH,y+n_)
     # Replaced only where it lies wholly inside polish's cache replacement (cache-replacement-mask): there
     # polish floored every cell. The cache donor (3k_main_map) lays its seabed owners on another grid than
     # the dlc07 list, so a record reaching past the replacement stays as vanilla, as its unedited cells do;
     # the relaid floors leave its cells alone (kept_sea below).
     # Its land cells (map.hex land off vanilla playable ground: a vanilla seabed where the canvas coast runs
     # wider than vanilla's) count as replaced too: a kept record leaves them with no land floor (kept_sea
     # below), 132 bare land cells on the rebuilt Korean coast, run 20261101-134607 (tile_land_coverage).
     _rm=sea_replaced_cells if sea_replaced_cells is not None else mask
     if x0<x1 and y0<y1 and seabed_replaced(mask[y0:y1,x0:x1],_rm[y0:y1,x0:x1],land_cells[y0:y1,x0:x1]):edit[y0:y1,x0:x1]=True;sea_replaced+=1;continue
    # Special tiles are culled by their decoded footprint in the next step,
    # never by generic-floor growth or record ordering.
    kept.append(rec)
   # Vanilla generic floors on unedited native ground that the export lost.
   have={(paths[r[1]],r[2],r[3],r[4],r[5]) for r in kept};restored=0
   for v in native_rows:
    name=np_[v[1]];match=__import__('re').search(r'generic_(\d+)x(\d+)',name)
    if not match or name not in paths:continue
    x,y=_ntg.list_xy(v[3],v[4])
    if (name,v[2],x,y,v[5]) in have:continue
    sx,sy=placement_footprint(tuple(map(int,match.groups())),v[5]);x0,x1=max(0,x),min(TW,x+sx);y0,y1=max(0,y),min(TH,y+sy)
    if not(NX0<=x0 and x1<=NX1 and NY0<=y0 and y1<=NY1) or x0>=x1 or y0>=y1:continue
    if np.any(mask[y0:y1,x0:x1]) or np.any(edit[y0:y1,x0:x1]) or np.any(biome[y0:y1,x0:x1]!=v[2]):continue
    kept.append((v[0],paths.index(name),v[2],x,y,v[5],v[6],v[7]));restored+=1
   # Vanilla special pieces on unedited native ground that the export lost (mountains_*, cliffs, coast, river
   # mouths): the base fills their cells with its own generic floors, which the loop above drops as not vanilla,
   # and the generic restore cannot refill them: 6,996 bare land cells in Jiangyang / Yunnan, run 20261030-123911,
   # 312 vanilla mountain pieces missing from terrain-before-polish (stage_invariants.tile_land_coverage).
   # Blocked by edits, polish's special pieces and the kept records that are not vanilla (on edited ground); kept
   # vanilla records do not block: vanilla's own pieces overlap each other and are restored as vanilla lists them.
   restored_special,special_skipped=vanilla_specials(native_rows,np_,paths,have,mask|edit|salt_cover|land_cover(paths,[r for r in kept if (paths[r[1]],r[2],r[3],r[4],r[5]) not in native_ids]),biome)
   kept+=restored_special
   edit&=~salt_cover   # pan cells carry the donor pan's own salt and floor records
   edit&=~road_cover   # road piece cells carry the road records
   # Kept native generic_sea floors keep their cells: no relaid floor under them (two seabed floors on one
   # cell). Coast pieces need no rule here: polish's own never meet a floor on an owned cell, and the
   # unowned cells of a vanilla 4x3 piece footprint carry floors in vanilla too.
   kept_sea_coast=np.zeros((TH,TW),bool)
   for r in kept:
    _m=_sf.SEA_RECORD.search(paths[r[1]])
    if _m:_n=int(_m.group(1));kept_sea_coast[max(0,r[4]):max(0,r[4]+_n),max(0,r[3]):max(0,r[3]+_n)]=True
   # Land cells off vanilla playable ground keep their floors under a kept seabed: a vanilla seabed that also
   # reaches vanilla playable land stays as vanilla lists it, and its other land cells are relaid (Xuantu / Liaodong).
   sea_kept_cells=int((edit&kept_sea_coast).sum());edit&=~(kept_sea_coast&~land_cells)
   # [m2048 salt-shards] The global_map list stores generic, generic_sea, sea_coast and mountains_* records
   # with rotation-only flags (vanilla 3k_main_map/dlc07: 0x10/0x20/0x40/0x80, never 0x7xx); only roads,
   # rivers, cliffs, canals and salt pools keep the full 0x7xx flags there. The salt windows copy the donor
   # pan's generic_1x1/2x2/4x4 floors (and mountain_tiles.py rides in the same records) with the local-list
   # flags: in the 2048m pack exactly those 104 generic_2x2 + 13 generic_4x4 + 112 generic_1x1 global records
   # drew as the pure-black right-angle blocks beside the pans (Shanshan, Xihai, Piaoyue, empty-1).
   _bare=__import__('re').compile(r'\\(generic|generic_sea|sea_coast|mountains_[^\\]+)\\')
   def list_flags(nm,fl):return fl&0xF0 if suffix.startswith('global_map/') and _bare.search(nm) else fl
   added=[(1,paths.index(nm),cl_,x_,y_,list_flags(nm,fl),*height_range(cols[0],rows[0],max(rows[1]-rows[0],cols[1]-cols[0])+1)) for nm,x_,y_,fl,cl_,rows,cols in salt_recs]
   added+=[(1,paths.index(nm),cl_,x_,y_,fl,*(lambda v:(float(v.min())/65535,float(v.max())/65535))(lf[max(0,(rows[0]-2)*4):(rows[1]+3)*4,max(0,(cols[0]-2)*4):(cols[1]+3)*4])) for nm,x_,y_,fl,cl_,rows,cols in road_recs]
   # Floors named and sized as polish lays them in the cache (native_tile_dimensions.FLOOR_TILE;
   # cache values are one-based): generic_16x16 on whole uniform 16-aligned blocks, generic_2x2
   # on whole 2-aligned groups, generic_1x1 on the remaining (split) cells. The old 8x8/1x1 relay
   # named the tiles one cache entry off, over half / a quarter of the cache footprints.
   covered=np.zeros((TH,TW),bool);relay_sizes={sz:0 for sz in FLOOR_TILE}
   relay=floor_relay(edit,biome)
   # --- SEA FLOORS (sea_floor_tiles.py) --- the floors polish laid as generic_sea on open sea: the same
   # split of the same relay (sea 8x8 / 2x2 / 1x1, owner turn and flags as vanilla, ranges raised to the
   # sea plane).
   sea=None
   if sea_water is not None and relay:
    (_ls,_lx,_ly),sea=_sf.split(*(np.array(c_,np.int64) for c_ in zip(*[(s_,x_,y_) for s_,x_,y_,_ in relay])),sea_water)
    relay=list(zip(_ls.tolist(),_lx.tolist(),_ly.tolist(),_sf.climate_at(_ls,_lx,_ly,biome).tolist()))
   for sz,x,y,climate in relay:
    low,high=height_range(x,y,sz);added.append((1,gi[sz],climate,x,y,donors[sz][5],low,high));covered[y:y+sz,x:x+sz]=True;relay_sizes[sz]+=1
   sea_sizes={}
   if sea is not None and len(sea[0]):
    _ss,_sx,_sy=sea;_sc=_sf.climate_at(_ss,_sx,_sy,biome);_st=_sf.turn(_sx,_sy)
    if tile_ext is not None:_slo,_shi=_sf.ranges(tile_ext[0],tile_ext[1],_ss,_sx,_sy)
    else:_r=np.array([height_range(x_,y_,s_) for s_,x_,y_ in zip(_ss.tolist(),_sx.tolist(),_sy.tolist())]).reshape(-1,2);_slo,_shi=np.maximum(_r[:,0],_sf.SEA_RANGE_FLOOR),np.maximum(_r[:,1],_sf.SEA_RANGE_FLOOR)
    _sp={s_:paths.index(_sf.SEA_TILE[s_]) for s_ in _sf.SEA_TILE};sea_cover=np.zeros((TH,TW),bool)
    for s_,x_,y_,c_,t_,lo_,hi_ in zip(_ss.tolist(),_sx.tolist(),_sy.tolist(),_sc.tolist(),_st.tolist(),_slo.tolist(),_shi.tolist()):
     added.append((1,_sp[s_],c_,x_,y_,list_flags(_sf.SEA_TILE[s_],0x700|t_),lo_,hi_));sea_cover[y_:y_+s_,x_:x_+s_]=True;sea_sizes[s_]=sea_sizes.get(s_,0)+1
    covered|=sea_cover
    _la=_sf.anchors((TH,TW),_ss,_sx,_sy)   # each sea floor's size at its anchor, against the cache's
    _ca=A/_sf.CACHE_ANCHORS_FILE
    if _ca.exists():_ca=np.load(_ca);sea_sizes['anchors_not_in_cache']=int(((_la>0)&(_la!=_ca)).sum());sea_sizes['cache_anchors_not_listed']=int(((_ca>0)&(_la!=_ca)).sum())
    del _ss,_sx,_sy,_sc,_st,_slo,_shi,_la
   # --- end sea floors
   assert np.all(covered[edit])
   if suffix.startswith('global_map/'):
    _bad=[r for r in kept+added if r[5]&0x700 and _bare.search(paths[r[1]])]
    assert not _bad,f'{len(_bad)} global generic/sea/coast/mountain records with local 0x7xx flags, e.g. {_bad[:3]}'
   head=bytearray(src[:start]);struct.pack_into('<I',head,cp,len(kept)+len(added));(S/rel).write_bytes(head+b''.join(struct.pack('<HIBhhHff',*r) for r in kept+added));report.append({'variant':variant,'list':suffix,'native_generic_flags':{str(k):donors[k][5] for k in donors},'runtime_generic_sizes':{f'{sz}x{sz}':[sz,sz] for sz in FLOOR_TILE},'floor_records_by_size':{str(sz):n for sz,n in relay_sizes.items()},'sea_floor_records':{str(k_):v_ for k_,v_ in sea_sizes.items()},'native_sea_floors_replaced':sea_replaced,'edit_cells_left_to_kept_sea_floors':sea_kept_cells,'kept':len(kept),'added':len(added),'export_floors_dropped_on_unedited_native':dropped,'vanilla_floors_restored':restored,'vanilla_specials_restored':len(restored_special),'vanilla_specials_not_restored':special_skipped,'edited_cells_covered':int(covered.sum())});print(variant,suffix,len(added),'runtime floor placements;',dropped,'export floors dropped,',restored,'vanilla floors restored,',len(restored_special),'vanilla specials restored',flush=True)
 payload={'variants':report,'live_verified':False}
 if tile_inputs.explicit:payload['tile_source']=tile_inputs.provenance()
 (A/'analysis/runtime-placement-coverage.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
 return report


def clear_overlays(dry_run=False,source=None):
 """clear_obsolete_terrain_overlays.py: cull obsolete native special tiles, refill their land cells."""
 A=_frame.WORKSPACE;S=A/'work/release-input';BACK=A/'before-overlay-clear-native-v2'
 dims=tile_dims   # one tile database record per path (module level, shared with place_floors)

 GENERIC_1X1='terrain\\tiles\\campaign\\generic\\generic_1x1\\'
 # Floors are not overlays: the land generic tiles and the generic_sea seabed (FLOOR_FAMILY). repair_native_placements.py
 # replaces every native floor that meets the edited ground over its whole footprint; culling a native
 # generic_sea floor here (as a special tile within two cells of changed ground) left the rest of its
 # footprint without any fine tile (the seabed holes along the drowned vanilla shores, Lin Yi strip v14-v16).
 def footprint(path,rec):return record_footprint(path,rec[5])
 def refill_removed_footprints(variant,paths,keep,removed_records):
  if not removed_records:return []
  one=paths.index(GENERIC_1X1)
  # The most common native generic 1x1 flags (material mask and rotation bits).
  flags=max(set(r[5] for r in keep if r[1]==one),key=[r[5] for r in keep if r[1]==one].count)
  TH,TW=_frame.TILE_H,_frame.TILE_W;covered=np.zeros((TH,TW),bool)
  for r in keep:
   path=paths[r[1]]
   # A kept generic_sea floor covers its cells too: generic_sea is no longer culled here (FLOOR_FAMILY), and a
   # refill generic_1x1 on a land cell under a vanilla seabed record listed two floors where vanilla has one
   # (75 cells on the 2048 stage copy, 72 of them unedited vanilla coast cells).
   fw,fh=footprint(path,r);covered[max(0,r[4]):max(0,r[4]+fh),max(0,r[3]):max(0,r[3]+fw)]=True
  dds=(S/f'terrain/campaigns/{variant}/lf_height_map.dds').read_bytes();lh,lw=struct.unpack_from('<2I',dds,12)
  lf=np.frombuffer(dds,dtype='<u2',offset=128).reshape(lh,lw)[::-1]
  added=[]
  for r in removed_records:
   fw,fh=footprint(paths[r[1]],r)
   for y in range(max(0,r[4]),min(TH,r[4]+fh)):
    for x in range(max(0,r[3]),min(TW,r[3]+fw)):
     q=min(w-1,x//2);rr=min(h-1,max(0,(y-(q%2))//2))   # tile x = 2q, tile y = 2r + odd(q)
     if covered[y,x] or (d[rr,q,0]&3)==1:continue
     values=lf[max(0,(y-2)*4):min(lh,(y+3)*4),max(0,(x-2)*4):min(lw,(x+3)*4)]
     added.append((1,one,r[2],x,y,flags,float(values.min())/65535,float(values.max())/65535));covered[y,x]=True
  return added
 _lf_cache={}
 def lf_pair(variant):
  if variant not in _lf_cache:
   def south_up(p):
    b=p.read_bytes();h,w=struct.unpack_from('<2I',b,12);return np.frombuffer(b,dtype='<u2',offset=128).reshape(h,w)[::-1]
   rel=f'terrain/campaigns/{variant}/lf_height_map.dds'
   # "Unchanged" means unchanged from vanilla inside the native map (the pre-polish
   # base may already differ there); canvas land compares with the pre-polish base.
   before=south_up(A/'terrain-before-polish'/rel).copy()
   from repair_campaign_river_roots import Assets
   vb=Assets(original_only=True).read(rel);vh,vw=struct.unpack_from('<2I',vb,12)
   ox,oy=_frame.NATIVE_OFFSET_LF;before[oy:oy+vh,ox:ox+vw]=np.frombuffer(vb,dtype='<u2',offset=128).reshape(vh,vw)[::-1]
   _lf_cache[variant]=(before,south_up(S/rel))
  return _lf_cache[variant]
 def ground_changed(variant,x0,y0,x1,y1):
  before,after=lf_pair(variant)
  a=before[y0*4:y1*4,x0*4:x1*4];b=after[y0*4:y1*4,x0*4:x1*4]
  return a.shape!=b.shape or not a.size or int(np.abs(a.astype(np.int32)-b).max())>LF_UNCHANGED
 from native_edit_mask import edited_tiles,TOLERANCE
 LF_UNCHANGED=TOLERANCE  # LF units (of 65535) treated as unchanged ground (shared edit definition)
 m=read_hex(A/'work/final-map/map.hex');w,h=m['width'],m['height'];plan=json.loads((A/'work/current-region-design.json').read_text(encoding='utf-8'));b=(A/'work/final-map/map.hex').read_bytes();d=np.frombuffer(b[-4-w*h*16:-4],np.uint8).reshape(h,w,16);ids=(d[:,:,1].astype(int)<<5)+(d[:,:,0].astype(int)>>3)-1;names=m['land_regions']+m['sea_regions'];edited=_frame.load_canvas_array(A/'analysis/continuous-land-mask.npy')|np.isin(ids,[names.index(c['region']) for c in plan['regions'] if c.get('introduced_in_bulk')]);mask=np.repeat(np.repeat(edited,2,axis=0),2,axis=1);mask=np.pad(mask,((0,1),(0,0)));reports=[]
 # OCCUPANCY ORDER (pipeline-bake-design 3.3.1 item 2; invariant stage_invariants.tile_placed_first): the special
 # pieces polish lays take their cells first (salt pools, mountain and coast tiles in salt-pan-records.json, road
 # pieces in road-tile-records.json); place_floors keeps floors off their cells, and this cull, which removes
 # native special tiles near changed ground, never removes them. Vanilla special tiles get no such priority.
 # Salt pans laid on edited ground by polish_campaign_terrain.py are new, not obsolete.
 _sp=A/'analysis/salt-pan-records.json'
 salt_pans={(s_[0],s_[1],s_[2]) for s_ in json.loads(_sp.read_text(encoding='utf-8'))} if _sp.exists() else set()
 # So are the road pieces road_tiles.py lays on the new roads (analysis/road-tile-records.json).
 _rp=A/'analysis/road-tile-records.json'
 salt_pans|={(s_[0],s_[1],s_[2]) for s_ in json.loads(_rp.read_text(encoding='utf-8'))} if _rp.exists() else set()
 for variant in TERRAIN_VARIANTS:
  # Polish alpha only where the LF really changed (shared definition, no halo);
  # alpha alone also covers untouched native rings.
  mask=(np.load(A/'analysis/polished-edit-alpha.npy')>.02)&edited_tiles(variant,halo=0)
  coverage=mask.copy()
  cache_coverage=A/'analysis/cache-replacement-mask.npy'
  if cache_coverage.exists():
   cache_mask=np.load(cache_coverage);assert cache_mask.shape==coverage.shape
   coverage|=cache_mask
  for suffix in ('tile_list.bin','global_map/tile_list.bin'):
   rel=Path(f'terrain/campaigns/{variant}/{suffix}');p=S/rel;bp=BACK/rel
   if not dry_run and not bp.exists():bp.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,bp)
   b=((source or S)/rel).read_bytes();paths,cl,cp,start,rs,info=parse(b);sizes={i:dims(path) for i,path in enumerate(paths) if not FLOOR_FAMILY.search(path.replace(chr(92),'/'))};keep=[];removed=[];removed_records=[]
   for rec in rs:
    if rec[1] not in sizes or (paths[rec[1]],rec[3],rec[4]) in salt_pans:keep.append(rec);continue
    x,y=rec[3:5];sx,sy=sizes[rec[1]]
    # rec[2] is climate; the quarter-turn rotation is in the flags.
    wx,hy=placement_footprint((sx,sy),rec[5])
    x0,x1,y0,y1=cull_window(x,y,wx,hy,mask.shape)
    # Keep the vanilla detail when the final height map under its sampled footprint
    # did not actually change; the edit mask also covers untouched apron/ring ground.
    if x0<x1 and y0<y1 and np.any(mask[y0:y1,x0:x1]) and ground_changed(variant,x0,y0,x1,y1):
     # Removal follows the rotated footprint; the coarse refill keeps covering
     # both orientations, which later gap filling and coverage audits rely on.
     for cw_,ch_ in set(((sx,sy),(sy,sx))):coverage[max(0,y-4):min(mask.shape[0],y+2*ch_+4),max(0,x-4):min(mask.shape[1],x+2*cw_+4)]=True
     removed.append({'path':paths[rec[1]],'xy':[x,y],'size':[wx,hy],'footprint_policy':'rotated footprint from flags; climate untouched'});removed_records.append(rec)
    else:keep.append(rec)
   # A removed special tile leaves its footprint without any fine tile wherever the
   # authored floor does not reach; the renderer then shows bare coarse ground in
   # tile-sized squares. Floor every such land cell with a generic 1x1 placement.
   added=refill_removed_footprints(variant,paths,keep,removed_records)
   head=bytearray(b[:start]);struct.pack_into('<I',head,cp,len(keep)+len(added))
   if not dry_run:p.write_bytes(head+b''.join(struct.pack('<HIBhhHff',*r) for r in keep+added))
   reports.append({'variant':variant,'list':suffix,'removed':removed,'retained':len(keep),'refilled_generic_1x1':len(added)});print(variant,suffix,'obsolete overlays',len(removed),'refilled cells',len(added),flush=True)
  if not dry_run:np.save(A/f'analysis/cleared-terrain-replacement-mask-{variant}.npy',coverage)
 if dry_run:print('Dry run: tile lists and report unchanged');return reports
 (A/'analysis/obsolete-terrain-overlays.json').write_text(json.dumps({'variants':reports,'live_verified':False},indent=2),encoding='utf-8')
 return reports


def bake():
 place_floors()
 clear_overlays()
