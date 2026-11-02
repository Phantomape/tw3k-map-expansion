"""Require every baseline special tile outside current edits to survive."""
import tw3k_paths
import json,re,struct
from pathlib import Path
import numpy as np
from analyze_reference import ROOT,inventory,digest
import campaign_world as _frame
from read_tile_list import parse,identity_sha256
from native_tile_dimensions import decode_dimensions
from campaign_world import TERRAIN_VARIANTS
A=_frame.WORKSPACE;S=A/'work/release-input';B=A/'terrain-before-polish';G=tw3k_paths.GAME_DATA
index={}
for name in ('terrain.pack','fast.pack','data.pack'):
 p=G/name
 for e in inventory(p,False)['entries']:index.setdefault(e['path'],(p,e))
sizes={}
def dimensions(path):
 if path in sizes:return sizes[path]
 parts=path.strip('\\').split('\\');key='terrain/tiles/campaign/_tile_database/tiles/'+parts[-2]+'_'+parts[-1]+'.bin'
 p,e=index[key]
 with p.open('rb') as f:f.seek(e['offset']);value=decode_dimensions(f.read(e['size']))
 sizes[path]=value;return value
mask=np.load(A/'analysis/polished-edit-alpha.npy')>.02;reports=[]
# --- SEA FLOORS (sea_floor_tiles.py) --- polish's cache replacement and the cells floor records cover now
import sea_floor_tiles as _sf
from native_tile_dimensions import placement_footprint as _pf
_crm=A/'analysis/cache-replacement-mask.npy';replacement=np.load(_crm) if _crm.exists() else None
def _floor_cells(paths_,recs,shape):
 cover=np.zeros(shape,bool)
 for r_ in recs:
  p_=paths_[r_[1]]
  if not _sf.FLOOR_FAMILY.search(p_.replace(chr(92),'/')):continue
  m_=re.search(r'(\d+)x(\d+)',p_.rstrip(chr(92)).split(chr(92))[-1]);w_,h_=_pf((int(m_.group(1)),int(m_.group(2))),r_[5])
  cover[max(0,r_[4]):max(0,r_[4]+h_),max(0,r_[3]):max(0,r_[3]+w_)]=True
 return cover
# --- HEHUANG PATCH: special tiles waived only by patch cells (alpha raised to its weight)
# are exempt_hehuang; a lost one anchored outside the patch mask (+2 hexes) is a leak.
from native_edit_mask import hehuang_tiles
import hehuang_terrain
_hhp=hehuang_terrain.load();hh_tiles=hehuang_tiles();base_mask=mask if _hhp is None else mask&~(_hhp.w>0)
for variant in TERRAIN_VARIANTS:
 rel=f'terrain/campaigns/{variant}/tile_list.bin';paths,_,_,_,old,_=parse((B/rel).read_bytes());nowpaths,_,_,_,now,_=parse((S/rel).read_bytes())
 floor_now=_floor_cells(nowpaths,now,mask.shape);refloored=0
 identities={(nowpaths[r[1]],r[2],r[3],r[4],r[5]) for r in now};protected=0;local=0;missing=[];hh_exempt=0;hh_lost=0;hh_leaks=[]
 for r in old:
  path=paths[r[1]]
  if '/generic/' in path.replace('\\','/'):continue
  w,h=dimensions(path);x,y=r[3:5];overlap=False
  # --- SEA FLOORS (sea_floor_tiles.py) --- a vanilla generic_sea floor (FLOOR_FAMILY) lying wholly inside
  # polish's cache replacement was re-floored there with the rest of its removed cache owner (repair_native_
  # placements replaces it like a native land floor); it is a floor, not a lost overlay, when every cell of its
  # footprint carries a floor record now. Outside the replacement, or left uncovered, it stays protected.
  _sm=_sf.SEA_RECORD.search(path)
  if _sm and replacement is not None and _sf.FLOOR_FAMILY.search(path.replace(chr(92),'/')):
   _n=int(_sm.group(1));_f=(slice(y,y+_n),slice(x,x+_n))
   if replacement[_f].shape==(_n,_n) and replacement[_f].all() and floor_now[_f].all():refloored+=1;continue
  for wx,hy in {(w,h),(h,w)}:
   x0,x1=max(0,x-2),min(mask.shape[1],x+wx+2);y0,y1=max(0,y-2),min(mask.shape[0],y+hy+2)
   overlap|=x0<x1 and y0<y1 and bool(np.any(mask[y0:y1,x0:x1]))
  if overlap and hh_tiles is not None:   # HEHUANG PATCH
   only_patch=True;in_mask=False
   for wx,hy in {(w,h),(h,w)}:
    x0,x1=max(0,x-2),min(mask.shape[1],x+wx+2);y0,y1=max(0,y-2),min(mask.shape[0],y+hy+2)
    only_patch&=not(x0<x1 and y0<y1 and bool(np.any(base_mask[y0:y1,x0:x1])))
    in_mask|=x0<x1 and y0<y1 and bool(np.any(hh_tiles[y0:y1,x0:x1]))
   if only_patch:
    # Waived only by patch cells: its footprint (x, y = south-west corner, +2 apron) must
    # reach into the patch mask; one waived by patch alpha outside the mask is a leak.
    hh_exempt+=1;lost=(path,r[2],x,y,r[5]) not in identities;hh_lost+=lost
    if lost and not in_mask:hh_leaks.append({'path':path,'xy':[x,y]})
  if overlap:continue
  protected+=1;near=abs(x-1420)<=120 and abs(y-1278)<=120;local+=near
  if (path,r[2],x,y,r[5]) not in identities:missing.append({'path':path,'xy':[x,y],'youzhou':near})
 report={'variant':variant,'generic_sea_refloored_in_replacement':refloored,'protected_special_tiles':protected,'youzhou_protected_special_tiles':local,'missing':missing,'tile_list_sha256':digest(S/rel),'tile_list_identity_sha256':identity_sha256((S/rel).read_bytes())};reports.append(report)
 if hh_tiles is not None:report['hehuang']={'exempt_hehuang':hh_exempt,'exempt_lost':hh_lost,'leaks':hh_leaks};print(variant,'Hehuang exempt',hh_exempt,'lost',hh_lost,'leaks',len(hh_leaks),flush=True)
 print(variant,'generic_sea floors re-floored inside the cache replacement',refloored,flush=True)
 print(variant,'protected',protected,'Youzhou',local,'missing',len(missing),flush=True)
result={'passed':all(not r['missing'] and not r.get('hehuang',{}).get('leaks') for r in reports),'source_map_sha256':digest(A/'work/final-map/map.hex'),'source_design_sha256':digest(A/'work/current-region-design.json'),'variants':reports,'live_verified':False}
(A/'analysis/native-overlay-preservation.json').write_text(json.dumps(result,indent=2))
if not result['passed']:raise RuntimeError('Protected original special terrain tiles are missing')
