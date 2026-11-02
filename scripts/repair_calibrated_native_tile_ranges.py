"""Re-derive retained native detailed-tile height ranges from the final height map.

Vanilla stores each placement's range as the LF extrema over its rotated
footprint plus two placement cells, optionally widened by a padding (63% exact,
37% widened on 3k_main_map land; none narrower). The footprint orientation is
decoded from the flags (native_tile_dimensions.placement_footprint). A retained
placement gets the vanilla range back where its ground is unchanged, otherwise
the new extrema widened by that placement's own vanilla padding.
"""
import tw3k_paths
import argparse,json,struct,re,shutil
from pathlib import Path
import numpy as np
from analyze_reference import ROOT,inventory,digest
import campaign_world as _frame
from read_tile_list import parse
from native_tile_dimensions import decode_dimensions,placement_footprint
from campaign_world import TERRAIN_VARIANTS
A=_frame.WORKSPACE;S=A/'work/release-input';GAME=tw3k_paths.GAME_DATA
index={}
for p in (GAME/'terrain.pack',GAME/'fast.pack',GAME/'data.pack'):
 for e in inventory(p,False)['entries']:index.setdefault(e['path'],(p,e))
def asset(k):
 p,e=index[k]
 with p.open('rb') as f:f.seek(e['offset']);return f.read(e['size'])
def pixels(b):
 h,w=struct.unpack_from('<2I',b,12);return np.frombuffer(b,dtype='<u2',offset=128).reshape(h,w)[::-1]
def bounds(field,x,y,w,h):
 a=field[max(0,(y-2)*4):min(field.shape[0],(y+h+2)*4),max(0,(x-2)*4):min(field.shape[1],(x+w+2)*4)]
 if not a.size:return None
 return [float(a.min())/65535,float(a.max())/65535]
def dimensions(path):
 match=re.search(r'generic_(\d+)x(\d+)',path)
 if match:return tuple(map(int,match.groups()))
 parts=path.strip(chr(92)).split(chr(92));key='terrain/tiles/campaign/_tile_database/tiles/'+parts[-2]+'_'+parts[-1]+'.bin'
 if key not in index:return None
 return decode_dimensions(asset(key))

arg=argparse.ArgumentParser();arg.add_argument('--apply',action='store_true');args=arg.parse_args();results=[];tol=.00005;native_tol=.0002
for variant in TERRAIN_VARIANTS:
 pre=f'terrain/campaigns/{variant}/';key=pre+'lf_height_map.dds'
 if key not in index:continue
 original=pixels(asset(key));current=pixels((S/key).read_bytes())
 for suffix in ('tile_list.bin','global_map/tile_list.bin'):
  key=pre+suffix;native=asset(key);pp,_,_,_,rr,_=parse(native);src=(S/key).read_bytes();paths,_,_,start,records,_=parse(src)
  # Height ranges are excluded from identity so rebuilding or a prior repair
  # cannot hide a newly stale field on the same native placement.
  native_rows={(pp[r[1]],r[2],r[3]+_frame.NATIVE_TILE_X0,r[4]+_frame.NATIVE_TILE_Y0,r[5]):r for r in rr}
  sizes={path:dimensions(path) for path in paths};out=bytearray(src);edits=[];unverified=[];calibrated=0;padded=0;counts={}
  for i,r in enumerate(records):
   path=paths[r[1]];donor=native_rows.get((path,r[2],r[3],r[4],r[5]));size=sizes[path]
   if donor is None or size is None:continue
   w,h=placement_footprint(size,r[5]);old=bounds(original,donor[3],donor[4],w,h);new=bounds(current,r[3],r[4],w,h)
   if old is None or new is None:unverified.append({'record':i,'path':path,'reason':'footprint outside the height map'});continue
   pad=(donor[6]-old[0],donor[7]-old[1])
   extrema_changed=max(abs(new[j]-old[j]) for j in (0,1))>tol
   # Unchanged ground keeps the exact vanilla fields, whatever they are (sea
   # tiles store a water range below the land samples; the SDK export rewrites
   # it from the land). Changed ground keeps this placement's vanilla padding
   # around the new extrema, which needs a range that contains its own sample.
   if extrema_changed and (pad[0]>native_tol or pad[1]<-native_tol):
    unverified.append({'record':i,'path':path,'reason':'vanilla range narrower than its own sample','padding':list(pad)});continue
   calibrated+=1;padded+=max(-pad[0],pad[1])>native_tol
   expected=[max(0.,new[0]+min(0.,pad[0])),min(1.,new[1]+max(0.,pad[1]))] if extrema_changed else list(donor[6:])
   if max(abs(expected[j]-r[6+j]) for j in (0,1))<=tol:continue
   struct.pack_into('<2f',out,start+i*21+13,*expected);counts[path]=counts.get(path,0)+1
   edits.append({'record':i,'path':path,'xy':list(r[3:5]),'sample_extent':[w,h],'ground_changed':extrema_changed,'before':list(r[6:]),'after':expected})
  checked=parse(out)[4];changed={e['record']:e for e in edits};assert out[:start]==src[:start]
  for i,(old,new) in enumerate(zip(records,checked)):
   if i not in changed:assert old==new
   else:assert old[:6]==new[:6] and max(abs(changed[i]['after'][j]-new[6+j]) for j in (0,1))<1e-7
  before=digest(S/key)
  if args.apply:
   backup=A/'layout-v2/calibrated-range-before'/key;backup.parent.mkdir(parents=True,exist_ok=True)
   if not backup.exists():backup.write_bytes(src)
   (S/key).write_bytes(out)
  results.append({'variant':variant,'list':suffix,'native_calibrated_records':calibrated,'padded_native_records':padded,'repaired_records':len(edits),
   'counts_by_tile':counts,'unverified_records':unverified,'changes':edits,'input_sha256':before,'output_sha256':digest(S/key),
   'unselected_records_and_header_unchanged':True})
  print(variant,suffix,'calibrated',calibrated,'padded',padded,'repaired',len(edits),'unverified',len(unverified),flush=True)
(A/'analysis/calibrated-native-tile-range-repair.json').write_text(json.dumps({'applied':args.apply,'tolerance':tol,'native_calibration_tolerance':native_tol,'lists':results,
 'live_steps_cause_proven':False,'new_input_reviewed':True},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
