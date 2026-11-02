"""Verify all untouched native cache ownership, paths and flags survive."""
import tw3k_paths
import hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageFilter
from analyze_reference import ROOT,inventory,digest
import campaign_world as _frame
from sync_native_cache_height_ranges import cache

A=_frame.WORKSPACE;S=A/'work/release-input'
G=tw3k_paths.GAME_DATA
key='terrain/campaigns/3k_main_map/tile_map.tiles'
source=G/'terrain.pack'
entry=next(e for e in inventory(source,False)['entries'] if e['path']==key)
with source.open('rb') as f:f.seek(entry['offset']);raw=f.read(entry['size'])
paths,layers=cache(raw,_frame.NATIVE_TILE_W,_frame.NATIVE_TILE_H)
intended=np.load(A/'analysis/cache-intended-edit-mask.npy')
# Margin protects the native height-range sampling apron as well as owners.
near=np.asarray(Image.fromarray(np.uint8(intended)*255).filter(ImageFilter.MaxFilter(5)))>0
# Owners whose ground really changed (shared edit definition) are recalibrated by
# sync_native_cache_height_ranges even when outside the intended cache edit.
from native_edit_mask import edited_tiles
from campaign_world import TERRAIN_VARIANTS
lf_edited=np.logical_or.reduce([edited_tiles(v) for v in TERRAIN_VARIANTS])
near=near|lf_edited[:near.shape[0],:near.shape[1]]
native_near=near[_frame.NATIVE_TILE_SLICE];reports=[]
# --- HEHUANG PATCH: owners released only by the patch's cache cut (intended cells with
# patch weight > 0) are exempt_hehuang; one changed outside the patch mask (+2 hexes) is a leak.
from native_edit_mask import hehuang_tiles
import hehuang_terrain
_hhp=hehuang_terrain.load();_hht=hehuang_tiles()
if _hhp is not None:
 _base=np.asarray(Image.fromarray(np.uint8(intended&~(_hhp.w>0))*255).filter(ImageFilter.MaxFilter(5)))>0
 _base=(_base|(lf_edited[:near.shape[0],:near.shape[1]]&~_hht))[_frame.NATIVE_TILE_SLICE];_hh_native=_hht[_frame.NATIVE_TILE_SLICE]
for variant in TERRAIN_VARIANTS:
 p=S/f'terrain/campaigns/{variant}/tile_map.tiles';nowpaths,now=cache(p.read_bytes(),_frame.TILE_W,_frame.TILE_H)
 assert paths==nowpaths,'Native cache path dictionary changed'
 protected=missing=0;range_drifts=0;examples=[];hh_exempt=hh_leak_cells=0
 from sync_native_cache_height_ranges import pixels,owner_bounds
 from repair_campaign_river_roots import Assets
 van_px=pixels(Assets(original_only=True).read(f'terrain/campaigns/{variant}/lf_height_map.dds'));stage_px=pixels((S/f'terrain/campaigns/{variant}/lf_height_map.dds').read_bytes())
 native_bounds=[owner_bounds(l) for l in layers]
 for layer_number,(layer,current) in enumerate(zip(layers,now)):
  active=(layer['path']!=0)|(layer['flags']!=16)
  owners=np.where(layer['path']<0,-layer['path']-1,np.arange(layer.size).reshape(layer.shape))
  affected=np.unique(owners[native_near&active]);keep=active&~np.isin(owners,affected)
  if _hhp is not None:   # HEHUANG PATCH
   _ab=np.unique(owners[_base&active]);_only=active&np.isin(owners,affected)&~np.isin(owners,_ab)
   # An owner released only by the patch must have cells inside the patch mask (+2 hexes).
   _inm=np.unique(owners[_only&_hh_native]);hh_exempt+=int(_only.sum());hh_leak_cells+=int((_only&~np.isin(owners,_inm)).sum())
  expected=layer.copy();child=expected['path']<0;parents=-expected['path'][child]-1
  expected['path'][child]=_frame.native_cache_owner(parents)
  expected['x'][active]+=_frame.NATIVE_TILE_X0;expected['y'][active]+=_frame.NATIVE_TILE_Y0
  actual=current[_frame.NATIVE_TILE_SLICE];bad=np.zeros(keep.shape,bool)
  for name in ('path','flags','x','y','climate','extra'):bad|=actual[name]!=expected[name]
  missing+=int((keep&bad).sum());protected+=int(keep.sum())
  drift=keep&((abs(actual['lo']-expected['lo'])>5e-5)|(abs(actual['hi']-expected['hi'])>5e-5))
  # Same rule as sync_native_cache_height_ranges: an owner whose sampling window
  # (owner bounds + 2 cells) sees ground that differs from vanilla is recalibrated.
  x0,y0,x1,y1=native_bounds[layer_number]
  for yy,xx in np.argwhere(drift):
   o=int(owners[yy,xx]);ox0,oy0,ox1,oy1=int(x0[o]),int(y0[o]),int(x1[o]),int(y1[o])
   vw=van_px[max(0,(oy0-2)*4):(oy1+3)*4,max(0,(ox0-2)*4):(ox1+3)*4]
   sw=stage_px[max(0,(oy0-2+_frame.NATIVE_TILE_Y0)*4):(oy1+3+_frame.NATIVE_TILE_Y0)*4,max(0,(ox0-2+_frame.NATIVE_TILE_X0)*4):(ox1+3+_frame.NATIVE_TILE_X0)*4]
   if vw.shape!=sw.shape or not np.array_equal(vw,sw):drift[yy,xx]=False
  range_drifts+=int(drift.sum())
  for y,x in np.argwhere(drift)[:3]:examples.append({'layer':layer_number,'native_cell':[int(x),int(y)],'path_id':int(layer[y,x]['path']),'origin':[int(layer[y,x]['x']),int(layer[y,x]['y'])],'before':[float(layer[y,x]['lo']),float(layer[y,x]['hi'])],'after':[float(actual[y,x]['lo']),float(actual[y,x]['hi'])]})
 if _hhp is not None:hh_report={'exempt_hehuang_owner_cells':hh_exempt,'leak_cells_outside_mask':hh_leak_cells};print('Hehuang',hh_report,flush=True)
 reports.append({'hehuang':hh_report if _hhp is not None else None,'variant':variant,'protected_owner_cells':protected,'changed_identity_cells':missing,'height_range_drifts':range_drifts,'range_drift_examples':examples,'cache_sha256':digest(p)})
 print(json.dumps(reports[-1]),flush=True)
result={'passed':all(r['changed_identity_cells']==0 and r['height_range_drifts']==0 and not (r['hehuang'] or {}).get('leak_cells_outside_mask') for r in reports),'variants':reports,'intended_mask_sha256':digest(A/'analysis/cache-intended-edit-mask.npy'),'native_source_sha256':hashlib.sha256(raw).hexdigest(),'live_verified':False}
(A/'analysis/native-cache-preservation.json').write_text(json.dumps(result,indent=2))
if not result['passed']:raise RuntimeError('Untouched native display-cache owners were replaced')
