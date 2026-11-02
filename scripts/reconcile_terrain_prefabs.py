"""Remove obsolete mountain overlays from land whose underlying relief changed.

Root instances only are rebuilt; native child assets and all settlements stay.
"""
import json,re,struct,math,sys
from pathlib import Path
import numpy as np
from analyze_reference import ROOT,read_hex
import campaign_world as _frame
from parse_global_props import parse
from read_root_prefabs import decode
from translate_campaign_lake_instances import records as leaf_records
from repair_campaign_river_roots import Assets,embedded
from final_terrain_surface import LEVEL,NORM,GAIN
from campaign_world import TERRAIN_VARIANTS
import prefab_source
# Decorative mountain/rock models of the unplayable area keep their vanilla base
# heights; on reshaped canvas ground they float or sink. Remove those that do.
FLOAT_LIMIT,GROUND_MOVED,TOWN_CLEAR=.8,.8,6
# Vanilla places its large decorative mountain clusters (karst, snow peaks; one
# spans 6-11 hexes) on non-playable land only. Where authored regions made that
# land playable they hid towns and stood across march routes (Rinan, Guanghan:
# live checks 2026-10-18 and canvas 1400), so any kept non-playable mountain
# whose footprint covers authored land is disabled. FOOTPRINT_CORE: the share
# of the bounding radius the peaks themselves fill.
FOOTPRINT_CORE=.8
_extent={}
def model_radius(model):
 """Horizontal radius of a model's LOD0 bounding box in model units (0 if unknown)."""
 key=model.lower()
 if key not in _extent:
  import xml.etree.ElementTree as E
  r=0.
  try:
   a=_vanilla();geo=key if key.endswith('.rigid_model_v2') else E.fromstring(a.read(key)).findtext('geometry').lower();g=a.read(geo)
   if g[:4]!=b'RMV2':raise KeyError(geo)
   lods=struct.unpack_from('<I',g,8)[0];groups=struct.unpack_from('<I',g,140)[0];p=140+lods*28
   for _ in range(groups):
    lo=struct.unpack_from('<3f',g,p+24);hi=struct.unpack_from('<3f',g,p+36)
    r=max(r,abs(lo[0]),abs(lo[2]),abs(hi[0]),abs(hi[2]));p+=struct.unpack_from('<I',g,p+4)[0]
  except (KeyError,AttributeError,struct.error,E.ParseError):r=0.
  _extent[key]=r
 return _extent[key]
_assets=None
def _vanilla():
 global _assets
 if _assets is None:
  from repair_campaign_river_roots import G
  from analyze_reference import inventory
  _assets=Assets(original_only=True)   # mountain geometry is also in models.pack
  for e in inventory(G/'models.pack',False)['entries']:_assets.index.setdefault(e['path'],(G/'models.pack',e))
 return _assets
def ground_heights(variant):
 d=(S/f'terrain/campaigns/{variant}/lf_height_map.dds').read_bytes();hh,ww=struct.unpack_from('<2I',d,12)
 return LEVEL+(np.frombuffer(d,'<u2',offset=128).reshape(hh,ww)[::-1].astype(np.float32)/65535-NORM)*GAIN
def hex_grounds(variant,ground):
 # Hex-centre ground of the stage and of vanilla placed at the native offset: the LF
 # texel at the block centre (8q+4, 8r+4*odd+4), not the block corner (8q, 8r+4*odd).
 qq,rr=np.meshgrid(np.arange(_frame.W),np.arange(_frame.H))
 yy=np.minimum(ground.shape[0]-1,np.rint((rr+(qq%2)*.5)*8).astype(int)+4);xx=np.minimum(ground.shape[1]-1,qq*8+4)
 stage=ground[yy,xx];native=np.full_like(stage,np.nan)
 d=Assets(original_only=True).read(f'terrain/campaigns/{variant}/lf_height_map.dds');hh,ww=struct.unpack_from('<2I',d,12)
 van=LEVEL+(np.frombuffer(d,'<u2',offset=128).reshape(hh,ww)[::-1].astype(np.float32)/65535-NORM)*GAIN
 nq,nr=np.meshgrid(np.arange(_frame.NATIVE_W),np.arange(_frame.NATIVE_H))
 vy=np.minimum(hh-1,np.rint((nr+(nq%2)*.5)*8).astype(int)+4);vx=np.minimum(ww-1,nq*8+4)
 native[_frame.NATIVE_HEX_SLICE]=van[vy,vx]
 return stage,np.where(np.isnan(native),stage,native)
def rechunk(root,r,retired):
 """A root record's bytes, its region string renamed when the region was retired."""
 chunk=root[r['start']:r['end']]
 if r['region'] not in retired:return chunk
 at=r['region_pos']-r['start'];old=r['region'].encode();new=retired[r['region']].encode()
 assert struct.unpack_from('<H',chunk,at)[0]==len(old) and chunk[at+2:at+2+len(old)]==old
 return chunk[:at]+struct.pack('<H',len(new))+new+chunk[at+2+len(old):]
def hex_of(X,Z):
 """Hex (q, r) owning a world point: campaign_world.world_hex (the tile_hex rule), or
 (-1, -1) off the map, where world_hex would clip it onto an edge hex."""
 q,rr=(int(v) for v in _frame.world_hex(X,Z));x0,z0=_frame.world_position(q,rr);e=1e-3*DX
 return (q,rr) if x0-e<=X<x0+DX+e and z0-e<=Z<z0+DZ+e else (-1,-1)
A=_frame.WORKSPACE;S=A/'work/release-input';BASE=A/'terrain-before-polish';DX=_frame.DX;DZ=_frame.DZ
_prefab_source=prefab_source.load(A,TERRAIN_VARIANTS,_frame.canvas_stamp(),(_frame.TILE_W,_frame.TILE_H))
m=read_hex(A/'work/final-map/map.hex');w,h=m['width'],m['height'];plan=json.loads((A/'work/current-region-design.json').read_text(encoding='utf-8'));b=(A/'work/final-map/map.hex').read_bytes();cells=np.frombuffer(b[-4-w*h*16:-4],np.uint8).reshape(h,w,16);ids=(cells[:,:,1].astype(int)<<5)+(cells[:,:,0].astype(int)>>3)-1;names=m['land_regions']+m['sea_regions'];mask=_frame.load_canvas_array(A/'analysis/continuous-land-mask.npy')|np.isin(ids,[names.index(c['region']) for c in plan['regions'] if c.get('introduced_in_bulk')]);reports=[]
for variant in TERRAIN_VARIANTS:
 ground=ground_heights(variant);hex_ground,native_hex_ground=hex_grounds(variant,ground);pre=f'terrain/campaigns/{variant}/';data=_prefab_source.props(variant);entries,base=parse(data);offsets=sorted(set(o for p,o in entries))+[len(data)-base];bounds={o:offsets[i+1] for i,o in enumerate(offsets[:-1])};lookup={p:data[base+o:base+bounds[o]] for p,o in entries};rootkey=pre+'bmd_objects.bin';root=lookup[rootkey];cp,records=decode(root);removed=[];kept=[]
 native_props=embedded(Assets(original_only=True).read(pre+'global_props.bin'))[0]
 # Scenes owned by regions that no longer exist (deleted in a redesign, e.g. old
 # resource-scene clones in the pre-polish baseline) would still render there.
 # Retired regions (seas merged into a neighbour, SDK filler) keep their props under the surviving
 # region: rechunk() rewrites the root's region string on output.
 known_regions=set(m['land_regions'])|set(m['sea_regions']);orphans=[];retired=plan.get('retired_regions',{})
 for rec in records:
  if rec['region'] not in known_regions and retired.get(rec['region']) not in known_regions:orphans.append({'path':rec['path'],'region':rec['region']});continue
  if rec['region']!='3k_main_reg_non_playable':kept.append(rec);continue
  stem=rec['path'][:-4];leaves=[p for p in lookup if p.startswith(stem+'.') and p.endswith('.31.bin')];models=[q.group().lower() for p in leaves for q in re.finditer(rb'(?i)rigidmodels/[a-zA-Z0-9_./]+',lookup[p])]
  if not any(b'/campaign/mountains/' in x or b'/campaign/rocks/' in x for x in models) or any(b'/campaign/settlements/' in x for x in models):kept.append(rec);continue
  # Use the parsed model instances. A raw float scan of the leaf bytes also
  # matched non-position triples (zeros + root translation -> hex [192,32] on the
  # edited native rim) and deleted intact native mountains far from any edit.
  T=rec['matrix'][12:15];placed=[];parsed=True
  for leaf in leaves:
   try:instances=leaf_records(lookup[leaf])
   except Exception:parsed=False;break
   for inst in instances:
    if '/mountains/' not in inst['model'].lower() and '/rocks/' not in inst['model'].lower():continue
    X,Y,Z=inst['matrix'][9]+T[0],inst['matrix'][10]+T[1],inst['matrix'][11]+T[2];q,rr=hex_of(X,Z)
    if 0<=q<w and 0<=rr<h:placed.append((q,rr,Y))
  if not parsed:
   placed=[]
   for leaf in leaves:
    body=lookup[leaf]
    for pos in range(36,len(body)-12):
     x,y,z=struct.unpack_from('<3f',body,pos)
     if not(0<x<596 and 0<z<543 and -10<y<100):continue
     rot=struct.unpack_from('<9f',body,pos-36)
     if not all(math.isfinite(v) and abs(v)<100 for v in rot) or sum(abs(v) for v in rot)<.1:continue
     q,rr=hex_of(x+T[0],z+T[2])
     if 0<=q<w and 0<=rr<h:placed.append((q,rr,None))
  # Native mountain models sit 5-10 units into the ground on purpose; only the
  # ground having moved under them since vanilla makes one float or sink.
  mismatched=[]
  for q,rr,Y in placed:
   if Y is not None and Y<-100:continue          # disabled instance
   nq,nr=q-_frame.PAD_LEFT,rr-_frame.PAD_BOTTOM
   if 0<=nq<_frame.NATIVE_W and 0<=nr<_frame.NATIVE_H:
    win=(slice(max(0,rr-2),rr+3),slice(max(0,q-2),q+3));moved=float(np.abs(hex_ground[win]-native_hex_ground[win]).max())
    if moved>GROUND_MOVED:mismatched.append([q,rr,round(moved,2)])
   elif Y is not None:
    gy=float(ground[min(ground.shape[0]-1,int(round((rr+q%2*.5)*8))+4),min(ground.shape[1]-1,q*8+4)])   # block centre, as hex_grounds
    if Y-gy>FLOAT_LIMIT:mismatched.append([q,rr,round(Y-gy,2)])
  hit=[m[:2] for m in mismatched]
  if hit:removed.append({'path':rec['path'],'region':rec['region'],'edited_neighbourhoods':hit,'height_mismatch':mismatched,'models':sorted(set(x.decode(errors='replace') for x in models))})
  else:kept.append(rec)
 # Vanilla mountain/rock roots dropped by earlier stages (their leaves are still
 # in the baseline) come back under the same rule: the ground within 2 hexes
 # has not moved more than GROUND_MOVED and no new town is within TOWN_CLEAR.
 restored=[];skipped_restore=[]
 vroot=native_props[rootkey];_,vrecords=decode(vroot);have={r['path'] for r in records}
 towns=np.array([c['town'] for c in plan['regions'] if 'town' in c],float)
 for rec in vrecords:
  if rec['path'] in have:continue
  # Judge and restore from the vanilla leaves: the baseline copies of dropped
  # roots were often disabled (y=-200) by earlier stages.
  stem=rec['path'][:-4];leaves=[p for p in native_props if p.startswith(stem+'.') and p.endswith('.bin') and p!=rootkey]
  models=[q.group().lower() for p in leaves for q in re.finditer(rb'(?i)rigidmodels/[a-zA-Z0-9_./]+',native_props[p])]
  if not leaves or not any(b'/campaign/mountains/' in x or b'/campaign/rocks/' in x for x in models) or any(b'/campaign/settlements/' in x for x in models):continue
  placed=[]
  for leaf in leaves:
   if not leaf.endswith('.31.bin'):continue          # model-instance leaves only
   try:instances=leaf_records(native_props[leaf])
   except Exception:continue
   for inst in instances:
    if inst['matrix'][10]<-100:continue
    X,Z=inst['matrix'][9]+_frame.NATIVE_SHIFT_WORLD[0],inst['matrix'][11]+_frame.NATIVE_SHIFT_WORLD[1];q,rr=hex_of(X,Z)
    if 0<=q<w and 0<=rr<h:placed.append((q,rr))
  why=None
  if not placed:why='no parsable instances'
  else:
   for q,rr in placed:
    win=(slice(max(0,rr-2),rr+3),slice(max(0,q-2),q+3))
    if float(np.abs(hex_ground[win]-native_hex_ground[win]).max())>GROUND_MOVED:why=f'ground moved at {[q,rr]}';break
    if len(towns) and float(np.min(np.hypot(towns[:,0]-q,towns[:,1]-rr)))<TOWN_CLEAR:why=f'new town within {TOWN_CLEAR} hexes of {[q,rr]}';break
  if why:skipped_restore.append({'path':rec['path'],'reason':why});continue
  chunk=bytearray(vroot[rec['start']:rec['end']]);off=rec['matrix_pos']-rec['start']
  struct.pack_into('<f',chunk,off+48,rec['matrix'][12]+_frame.NATIVE_SHIFT_WORLD[0]);struct.pack_into('<f',chunk,off+56,rec['matrix'][14]+_frame.NATIVE_SHIFT_WORLD[1])
  restored.append((rec,bytes(chunk)))
 print(variant,'vanilla mountain roots restored',len(restored),'skipped',len(skipped_restore),flush=True)
 # Vanilla instances disabled (y=-200) by earlier stages come back under the
 # same rule, in every vanilla root present after this pass (resource scenes of
 # authored regions are re-seated by complete_campaign_resource_scenes instead).
 out=bytearray(data);reenabled=[];reenable_skipped=[]
 present={r['path']:r for r in kept};present.update({r['path']:r for r,_ in restored})
 vroots={r['path']:r for r in vrecords};counties={c['region'] for c in plan['regions']}
 offset_of={p:o for p,o in entries}
 def blocked(q,rr):
  win=(slice(max(0,rr-2),rr+3),slice(max(0,q-2),q+3))
  if float(np.abs(hex_ground[win]-native_hex_ground[win]).max())>GROUND_MOVED:return f'ground moved at {[q,rr]}'
  if len(towns) and float(np.min(np.hypot(towns[:,0]-q,towns[:,1]-rr)))<TOWN_CLEAR:return f'new town within {TOWN_CLEAR} hexes of {[q,rr]}'
 for path,rec in present.items():
  if path not in vroots or rec['region'] in counties:continue
  vt=vroots[path]['matrix'][12:15];stem=path[:-4]
  for leaf in [p for p in native_props if p.startswith(stem+'.') and p.endswith('.bin') and p!=rootkey]:
   cur,van=lookup.get(leaf),native_props[leaf]
   if cur is None or cur==van or len(cur)!=len(van):continue
   try:ci,vi=leaf_records(cur),leaf_records(van)
   except Exception:continue
   if len(ci)!=len(vi) or any(a['model']!=b['model'] or a['position_offset']!=b['position_offset'] for a,b in zip(ci,vi)):continue
   for a,b in zip(ci,vi):
    if not(a['matrix'][10]<-100 and b['matrix'][10]>=-100):continue
    if '/lakes/' in b['model'].lower() or '/river' in b['model'].lower():continue
    X=b['matrix'][9]+vt[0]+_frame.NATIVE_SHIFT_WORLD[0];Z=b['matrix'][11]+vt[2]+_frame.NATIVE_SHIFT_WORLD[1];q,rr=hex_of(X,Z)
    why=blocked(q,rr) if (0<=q<w and 0<=rr<h) else 'outside the map'
    if why:reenable_skipped.append({'leaf':leaf,'model':b['model'],'reason':why});continue
    m0=b['position_offset']-36;at=base+offset_of[leaf]+m0;out[at:at+48]=van[m0:m0+48]
    reenabled.append({'leaf':leaf,'model':b['model'],'hex':[q,rr]})
 # Kept non-playable mountain/rock instances whose footprint covers authored
 # land are disabled (y=-200, as earlier stages did), not their whole root.
 cleared=[]
 authored_land=np.isin(ids,[names.index(c['region']) for c in plan['regions'] if c['region'] in names])&((cells[:,:,0]&3)!=1)
 for rec in present.values():          # kept and restored roots
  if rec['region']!='3k_main_reg_non_playable':continue
  T=rec['matrix'][12:15];stem=rec['path'][:-4]
  for leaf in [p for p in lookup if p.startswith(stem+'.') and p.endswith('.31.bin')]:
   o=offset_of[leaf]
   try:instances=leaf_records(bytes(out[base+o:base+bounds[o]]))
   except Exception:continue
   for inst in instances:
    mdl=inst['model'].lower()
    if '/mountains/' not in mdl and '/rocks/' not in mdl:continue
    M=inst['matrix'];Y=M[10]+T[1]
    if Y<-100:continue
    scale=float(np.mean([math.sqrt(sum(v*v for v in M[i:i+3])) for i in (0,3,6)]))
    q,rr=_frame.world_hex(M[9]+T[0],M[11]+T[2]);q,rr=int(q),int(rr);reach=max(1.,scale*model_radius(inst['model'])/_frame.HEX_DX*FOOTPRINT_CORE)
    R=int(reach)+1;y0,y1,x0,x1=max(0,rr-R),min(h,rr+R+1),max(0,q-R),min(w,q+R+1)
    yy,xx=np.mgrid[y0:y1,x0:x1];covered=int((authored_land[y0:y1,x0:x1]&(np.hypot(xx-q,yy-rr)<reach)).sum())
    if covered:
     at=base+o+inst['position_offset']+4;out[at:at+4]=struct.pack('<f',-200.)
     cleared.append({'leaf':leaf,'model':inst['model'],'hex':[q,rr],'reach':round(reach,1),'authored_hexes_covered':covered})
 print(variant,'kept non-playable mountain instances cleared from authored land',len(cleared),flush=True)
 # Loose copies of re-enabled or cleared leaves (packs ship both) follow the bundle.
 for leaf in sorted({r['leaf'] for r in reenabled}|{r['leaf'] for r in cleared}):
  loose=S/leaf
  if loose.exists() and '--apply' in sys.argv:o=offset_of[leaf];loose.write_bytes(bytes(out[base+o:base+bounds[o]]))
 data=bytes(out)
 print(variant,'disabled vanilla instances re-enabled',len(reenabled),'kept disabled',len(reenable_skipped),flush=True)
 print(variant,'roots of regions no longer on the map removed',len(orphans),flush=True)
 if '--apply' in sys.argv:
  head=bytearray(root[:records[0]['start']]);struct.pack_into('<I',head,cp,len(kept)+len(restored));updated=bytes(head)+b''.join(rechunk(root,r,retired) for r in kept)+b''.join(c for _,c in restored)+root[records[-1]['end']:];_,check=decode(updated);assert [r['path'] for r in check]==[r['path'] for r in kept]+[r['path'] for r,_ in restored];off=next(o for p,o in entries if p==rootkey);assert off==max(o for p,o in entries)
  (S/(pre+'global_props.bin')).write_bytes(data[:base+off]+updated);(S/rootkey).write_bytes(updated)
 reports.append({'variant':variant,'restored_vanilla_roots':[r['path'] for r,_ in restored],'restore_skipped':skipped_restore,'orphan_region_roots_removed':orphans,'reenabled_vanilla_instances':reenabled,'reenable_skipped':reenable_skipped,'authored_land_cleared_instances':cleared,'removed_instances':removed,'retained_instances':len(kept),'settlement_instances_preserved':True,'child_assets_preserved':True})
 if _prefab_source.explicit:reports[-1]['prefab_source']=_prefab_source.provenance()
 print(variant,'obsolete mountain overlays',len(removed),'kept',len(kept),flush=True)
(A/'analysis/terrain-prefab-reconciliation.json').write_text(json.dumps({'applied':'--apply' in sys.argv,'variants':reports,'live_verified':False},ensure_ascii=False,indent=2),encoding='utf-8')
