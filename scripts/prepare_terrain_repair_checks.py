"""Prepare isolated terrain-repair evidence and dynamically loaded game checks."""
import argparse,json,math,shutil
from pathlib import Path
from analyze_reference import ROOT,digest
import campaign_world as _frame
from final_terrain_surface import surface_at_display
import numpy as np
A=_frame.WORKSPACE;O=A/'terrain-repair';DX=_frame.DX;DZ=_frame.DZ
parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',action='store_true');args=parser.parse_args()
if args.checkpoint:
 for rel in ('analysis/polished-edit-alpha.npy','analysis/polished-surface-3k_dlc07_main_map.npy','analysis/geographic-terrain-polish.json','analysis/compiler-manifest.json','layout-v5/native-generated-state.json','events.log'):
  p=A/rel;q=O/'before'/rel
  if p.exists() and not q.exists():q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
 candidate=ROOT/'output/tw3k_regions_and_battles_layout_v5_candidate.pack'
 state={'baseline_candidate':str(candidate),'baseline_sha256':digest(candidate),'baseline_snapshot':str(O/'before'),'game_source_map_sha256':digest(A/'work/final-map/map.hex'),'source_design_sha256':digest(A/'work/current-region-design.json')}
 O.mkdir(exist_ok=True);(O/'baseline.json').write_text(json.dumps(state,indent=2),encoding='utf-8');print('Terrain source checkpoint retained')
else:
 checks=A/'checks';plan=json.loads((A/'work/current-region-design.json').read_text(encoding='utf-8'));surface=np.load(A/'analysis/polished-surface-3k_dlc07_main_map.npy')
 specs=[]
 # Expected settlement height: the surface where the engine draws the town (display
 # position), not the SW corner texel of its tile block.
 for c in plan['regions']:
  q,r=c['town'];height=surface_at_display(surface,q,r);specs.append('{key='+json.dumps(c['region'])+',height='+format(height,'.9g')+'}')
 lua='local specs={'+','.join(specs)+'}\nlocal out={};local n=0\nfor _,s in ipairs(specs) do\n local path=\'CampaignRoot.SettlementList.FirstContext(RegionRecordKey == "\'..s.key..\'").Position.\'\n local y=tonumber(effect.get_context_value(path.."y"))\n assert(y and math.abs(y-s.height)<.15,"Height mismatch "..s.key.." actual="..tostring(y).." expected="..s.height)\n n=n+1;out[#out+1]=s.key.."="..y\nend\nreturn "Canonical city heights verified="..n.."; "..table.concat(out,"; ")\n'
 (checks/'terrain_city_heights.lua').write_text(lua,encoding='utf-8')
 authored_checks=_frame.load_authored_coordinates()['check_points']
 # The region a check selects: the playable region owning its hex on the final map, else the design region whose town
 # is nearest (was a key list: two of its keys had left the design, single-source audit 12, 2026-11-01).
 import map_hex
 _m=map_hex.read(A/'work/final-map/map.hex');_names=_m['land_regions']+_m['sea_regions'];_ids=map_hex.region_ids(_m['cells'])
 def owner(q,r):
  k=_names[_ids[int(round(r)),int(round(q))]]
  if k in _m['land_regions'] and 'non_playable' not in k:return k
  return min(plan['regions'],key=lambda c:(c['town'][0]-q)**2+(c['town'][1]-r)**2)['region']
 points=[('terrain_'+name,*authored_checks[name],owner(*authored_checks[name])) for name in ('cliff_a','cliff_b','overlap','west','youzhou','korea')]
 for name,q,r,key in points:
  x,z=_frame.world_position(q,r)   # the engine's odd-column rule (int(q) % 2), as everywhere else
  text='tw3k_test_region='+json.dumps(key)+'\ncm:set_camera_position('+format(x,'.12g')+','+format(z,'.12g')+',25,0,12)\nreturn '+json.dumps(name+' q='+str(q)+' r='+str(r))+'\n'
  (checks/(name+'.lua')).write_text(text,encoding='utf-8')
 O.mkdir(exist_ok=True);(O/'live-check-points.json').write_text(json.dumps([{'check':n,'q':q,'r':r,'region':k} for n,q,r,k in points],indent=2),encoding='utf-8');print('Prepared canonical height and six camera checks')
