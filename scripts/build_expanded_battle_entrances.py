"""Align battle entrance coordinates and the expanded spatial grid."""
import json,struct
from pathlib import Path
import numpy as np
from analyze_reference import ROOT,digest
import campaign_world as _frame
from campaign_world import W,H,assert_grid
from fastbin import parse,enc
A=_frame.WORKSPACE;STAGE=A/'work/release-input';plan=json.loads((A/'work/current-region-design.json').read_text(encoding='utf-8'))
key='terrain/battles/3k_main_map/battle_locations_map.bin';source=(ROOT/'reference/vanilla-terrain'/key).read_bytes();header,groups,tail=parse(source)
w,h,count=struct.unpack_from('<3I',tail);assert (w,h,count)==(_frame.NATIVE_W,_frame.NATIVE_H+1,_frame.NATIVE_W*(_frame.NATIVE_H+1));old=np.frombuffer(tail,dtype='<u4',offset=12).reshape(h,w);new=np.zeros((H+1,W),dtype='<u4');new[_frame.PAD_TOP:_frame.PAD_TOP+_frame.NATIVE_H+1,_frame.PAD_LEFT:_frame.PAD_LEFT+_frame.NATIVE_W]=old
# Grid value 1 marks battle-capable land (vanilla: ~90% of passable land in
# playable regions). The expanded area started as all 0 and authored regions
# inside the native block kept vanilla's non-playable 0, so every new region read
# as non-playable to the battle-location lookup. Mark their passable land.
from all_directions_build import _hex_cells
fnames,fcells=_hex_cells(A/'work/final-map/map.hex');ids=(fcells[:,:,1].astype(int)<<5)+(fcells[:,:,0].astype(int)>>3)-1
authored={c['region'] for c in plan['regions']}
index={n:i for i,n in enumerate(fnames)}
authored_ids=np.array([index[n] for n in authored if n in index])
passable=((fcells[:,:,0]&3)!=1)&((fcells[:,:,2]&8)==0)
battle_land=passable&np.isin(ids,authored_ids)
rr,qq=np.nonzero(battle_land)
new[H-rr,qq]=1   # grid rows are north-first: hex row r -> grid row H-r (as the entrances)
records=[];added=0
for name,rows in groups:
 own=[]
 for row in rows:
  b=bytearray(row['raw']);coords=list(row['coordinates'])
  for i in (0,2,4):coords[i]+=_frame.PAD_LEFT
  for i in (1,3,5):coords[i]+=_frame.PAD_TOP
  struct.pack_into('<6i',b,2,*coords);own.append(bytes(b))
 if name in ('settlement_standard','settlement_unfortified'):
  for county in plan['regions']:
   x,y=county['town'];y=H-y;coords=(x-7,y-7,x+7,y+7,x,y);prefix=bytearray(rows[0]['raw'][:42]);struct.pack_into('<6i',prefix,2,*coords)
   parts=[county.get('battle_layout','resource_han_lumber_a_small'),'',county['region']+('_std' if name=='settlement_standard' else '_unf'),'']
   own.append(bytes(prefix)+b''.join(enc(v) for v in parts)+rows[0]['raw'][-6:]);added+=1
 records.append(enc(name)+struct.pack('<I',len(own))+b''.join(own))
result=header+b''.join(records)+struct.pack('<3I',W,H+1,W*(H+1))+new.tobytes();_,check,ct=parse(result);assert len(ct)==12+W*(H+1)*4
expected={c['region']+suffix:(c['town'][0],H-c['town'][1]) for c in plan['regions'] for suffix in ('_std','_unf')}
actual={row['strings'][2]:tuple(row['coordinates'][4:6]) for _,rows in check for row in rows if len(row.get('strings',[]))>2 and row['strings'][2] in expected}
assert actual==expected, 'Battle entrances differ from current design'
p=STAGE/key;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(result)
# Preservation: the vanilla block at the frame offsets line 11 wrote it to, less the authored battle land marked above.
nat=(slice(_frame.PAD_TOP,_frame.PAD_TOP+_frame.NATIVE_H+1),slice(_frame.PAD_LEFT,_frame.PAD_LEFT+_frame.NATIVE_W));marked=np.zeros(new.shape,bool);marked[H-rr,qq]=True;keep=~marked[nat]
(A/'analysis/combined-battle-entrances.json').write_text(json.dumps({'source_design_sha256':digest(A/'work/current-region-design.json'),'source_map_sha256':digest(A/'work/final-map/map.hex'),'all_current_entrances_verified':len(actual),'added_settlement_entrances':added,'vanilla_entrances_translated':sum(len(r) for _,r in groups),'old_grid':[_frame.NATIVE_W,_frame.NATIVE_H+1],'new_grid':[W,H+1],'grid_preserved_under_translation':bool(np.array_equal(new[nat][keep],old[keep]))},indent=2))
print('Battle entrance/grid alignment PASS',added)
