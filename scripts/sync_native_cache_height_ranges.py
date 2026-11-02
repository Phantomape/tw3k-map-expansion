"""Calibrate retained cache roots from exact native owner footprints."""
import tw3k_paths
import argparse,hashlib,itertools,json,struct
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image,ImageFilter
from analyze_reference import ROOT,inventory
import campaign_world as _frame
from campaign_world import TERRAIN_VARIANTS
DT=np.dtype([('path','<i4'),('flags','<u2'),('x','<u2'),('y','<u2'),('climate','u1'),('extra','u1'),('lo','<f4'),('hi','<f4')]);TOL=.00005;SHIFT=_frame.NATIVE_OFFSET_TILE
def digest(b):return hashlib.sha256(b).hexdigest()
def file_digest(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def pixels(b):
 h,w=struct.unpack_from('<2I',b,12);assert len(b)==128+h*w*2
 return np.frombuffer(b,'<u2',offset=128).reshape(h,w)[::-1]
def cache(b,w,h):
 n=w*h;pos=n*60;count=struct.unpack_from('<I',b,pos)[0];pos+=4;paths=[]
 for _ in range(count):
  length=struct.unpack_from('<H',b,pos)[0];pos+=2;paths.append(b[pos:pos+length].decode());pos+=length
 return paths,[np.frombuffer(b,DT,count=n,offset=i*n*20).reshape(h,w) for i in range(3)]
def roots(paths,layers):
 for layer,data in enumerate(layers):
  flat=data.ravel();valid=(flat['path']>=0)&(flat['path']<len(paths))&(flat['flags']!=16)
  for at in np.flatnonzero(valid):
   row=flat[at];yield layer,int(at),row,paths[int(row['path'])]
def identity(layer,at,row,path,width,shift=(0,0)):
 return (layer,at%width+shift[0],at//width+shift[1],path,int(row['x'])+shift[0],int(row['y'])+shift[1],int(row['flags']),int(row['climate']),int(row['extra']))
def root_table(paths,layers,width,shift=(0,0),bounds=None):
 """roots() as columns, in roots() order: per root its identity() key, layer, cell, path, lo, hi and
 (given owner_bounds per layer) its owner bounds, as Python scalars: no per-row numpy access."""
 for layer,data in enumerate(layers):
  flat=data.ravel();at=np.flatnonzero((flat['path']>=0)&(flat['path']<len(paths))&(flat['flags']!=16));row=flat[at];name=[paths[i] for i in row['path'].tolist()]
  key=zip(itertools.repeat(layer),(at%width+shift[0]).tolist(),(at//width+shift[1]).tolist(),name,(row['x'].astype(np.int64)+shift[0]).tolist(),(row['y'].astype(np.int64)+shift[1]).tolist(),row['flags'].tolist(),row['climate'].tolist(),row['extra'].tolist())
  yield from zip(key,itertools.repeat(layer),at.tolist(),name,row['lo'].tolist(),row['hi'].tolist(),zip(*(a[at].tolist() for a in bounds[layer])) if bounds else itertools.repeat(None))
def ownership(layer,at,x,y,w,h):
 block=layer[y:y+h,x:x+w]
 if block.shape!=(h,w):return None
 ax,ay=at%layer.shape[1]-x,at//layer.shape[1]-y
 if not(0<=ax<w and 0<=ay<h):return None
 owned=(block['path']==-at-1).copy();owned[ay,ax]=True;return owned

def owner_bounds(layer):
 h,w=layer.shape;flat=layer.ravel();n=len(flat);cells=np.arange(n,dtype=np.int32)
 owners=np.where(flat['path']<0,-flat['path']-1,cells)
 valid=(((flat['path']!=0)|(flat['flags']!=16))&(owners>=0)&(owners<n))
 owners,cells=owners[valid],cells[valid]
 x0=np.full(n,w,np.int32);y0=np.full(n,h,np.int32);x1=np.full(n,-1,np.int32);y1=np.full(n,-1,np.int32)
 if not len(owners):return x0,y0,x1,y1
 # Per-owner extremes by one stable sort + reduceat (np.minimum.at/maximum.at: ~7x slower).
 # Each owner's cells stay ascending, so its first/last cell give the row range.
 order=np.argsort(owners,kind='stable');owners,cells=owners[order],cells[order]
 start=np.flatnonzero(np.r_[True,owners[1:]!=owners[:-1]]);u=owners[start];cx=cells%w
 x0[u]=np.minimum.reduceat(cx,start);x1[u]=np.maximum.reduceat(cx,start);y0[u]=cells[start]//w;y1[u]=cells[np.r_[start[1:],len(cells)]-1]//w
 return x0,y0,x1,y1

def sample_range(field,x,y,w,h):
 block=field[max(0,(y-2)*4):min(field.shape[0],(y+h+2)*4),max(0,(x-2)*4):min(field.shape[1],(x+w+2)*4)]
 return [float(block.min())/65535,float(block.max())/65535] if block.size else None

def affected_native_owner_ids(layers,intended):
 near=np.asarray(Image.fromarray(np.uint8(intended)*255).filter(ImageFilter.MaxFilter(5)))>0
 native_near=near[_frame.NATIVE_TILE_SLICE];affected=[]
 for layer in layers:
  active=(layer['path']!=0)|(layer['flags']!=16)
  owners=np.where(layer['path']<0,-layer['path']-1,np.arange(layer.size).reshape(layer.shape))
  affected.append(set(map(int,np.unique(owners[native_near&active]))))
 return affected

def audit(stage,game):
 index={}
 for name in ('terrain.pack','fast.pack','data.pack'):
  source=game/name
  for entry in inventory(source,False)['entries']:index.setdefault(entry['path'],(source,entry))
 def asset(key):
  source,entry=index[key];assert not entry['compressed']
  with source.open('rb') as f:f.seek(entry['offset']);return f.read(entry['size'])
 prefix='terrain/campaigns/3k_main_map/';nb=asset(prefix+'tile_map.tiles');nd=asset(prefix+'lf_height_map.dds');paths0,layers0=cache(nb,_frame.NATIVE_TILE_W,_frame.NATIVE_TILE_H);field0=pixels(nd)
 # sha256 releases the GIL: the report's digests are hashed on a thread while the audit runs.
 hashing=ThreadPoolExecutor(2);native_hashes=hashing.submit(digest,nb),hashing.submit(digest,nd)
 # A placement's nominal origin is not its cache ownership footprint. Keep
 # native ranges for owners outside the intended edit plus sampling halo,
 # even if a placement with the same origin was recalibrated independently.
 intended=np.load(_frame.WORKSPACE/'analysis/cache-intended-edit-mask.npy')
 affected_owners=affected_native_owner_ids(layers0,intended)
 native_bounds=[owner_bounds(layer) for layer in layers0]
 donors={key:(layer,at,lo,hi,bounds) for key,layer,at,_,lo,hi,bounds in root_table(paths0,layers0,_frame.NATIVE_TILE_W,SHIFT,native_bounds)}
 report={'applied':False,'native_cache_donor':'3k_main_map','native_cache_sha256':None,'native_dds_sha256':None,'tolerance':TOL,'sampling':'exact native owner bounding rectangle, margin2, four DDS pixels/cell','variants':[]}
 for variant in TERRAIN_VARIANTS:
  folder=stage/'terrain/campaigns'/variant;cp=folder/'tile_map.tiles';dp=folder/'lf_height_map.dds';cb=cp.read_bytes();db=dp.read_bytes();hashes=hashing.submit(digest,cb),hashing.submit(digest,db);paths,layers=cache(cb,_frame.TILE_W,_frame.TILE_H);field=pixels(db);variant_original=pixels(asset('terrain/campaigns/'+variant+'/lf_height_map.dds'));current_bounds=[owner_bounds(layer) for layer in layers];counts=Counter();changes=[]
  for key,layer,at,path,lo,hi,bounds in root_table(paths,layers,_frame.TILE_W,bounds=current_bounds):
   counts['active_roots']+=1;donor=donors.get(key)
   if donor is None:counts['not_exact_native_root']+=1;continue
   counts['exact_native_roots']+=1;dl,da,dlo,dhi,nbounds=donor
   if bounds!=(nbounds[0]+SHIFT[0],nbounds[1]+SHIFT[1],nbounds[2]+SHIFT[0],nbounds[3]+SHIFT[1]):counts['owner_bounds_changed']+=1;continue
   nx,ny,nx1,ny1=nbounds;x,y,x1,y1=bounds;w,h=x1-x+1,y1-y+1
   if w<=0 or h<=0:counts['empty_owner_bounds']+=1;continue
   old_owned=ownership(layers0[dl],da,nx,ny,w,h);new_owned=ownership(layers[layer],at,x,y,w,h)
   if old_owned is None or new_owned is None or not np.array_equal(old_owned,new_owned):counts['ownership_changed']+=1;continue
   original=[dlo,dhi]
   old_pixels=variant_original[max(0,(ny-2)*4):min(variant_original.shape[0],(ny+h+2)*4),max(0,(nx-2)*4):min(variant_original.shape[1],(nx+w+2)*4)]
   new_pixels=field[max(0,(y-2)*4):min(field.shape[0],(y+h+2)*4),max(0,(x-2)*4):min(field.shape[1],(x+w+2)*4)]
   # "Protected" (outside the intended edit) only keeps vanilla when the ground
   # under its sampling window really is vanilla; otherwise recalibrate below.
   if da not in affected_owners[dl] and old_pixels.shape==new_pixels.shape and np.array_equal(old_pixels,new_pixels):
    counts['protected_native_owner_ranges_preserved']+=1
    before=[lo,hi]
    if before!=original:
     changes.append({'offset':layer*_frame.TILE_W*_frame.TILE_H*20+at*20+12,'layer':layer,'root_cell':[at%_frame.TILE_W,at//_frame.TILE_W],'origin':[x,y],'path':path,'extent':[w,h],'before':before,'expected':original,'reason':'native ownership footprint and its sampling halo are outside intended terrain edits'})
     counts['protected_native_owner_ranges_restored']+=1
    continue
   if da not in affected_owners[dl]:counts['protected_owner_ground_changed']+=1
   same_pixels=old_pixels.shape==new_pixels.shape and np.array_equal(old_pixels,new_pixels)
   same_extrema=old_pixels.size and new_pixels.size and old_pixels.min()==new_pixels.min() and old_pixels.max()==new_pixels.max()
   if same_pixels or same_extrema:
    # DLC07 uses the native shared main-map cache even where its own DDS is
    # slightly different. Recalibrating unchanged DLC pixels against main-map
    # ranges corrupts a working native tile. Preserve its exact donor values.
    counts['unchanged_variant_pixels_preserved' if same_pixels else 'unchanged_variant_extrema_preserved']+=1
    before=[lo,hi]
    if before!=original:
     counts['unchanged_native_ranges_restored']+=1
     changes.append({'offset':layer*_frame.TILE_W*_frame.TILE_H*20+at*20+12,'layer':layer,'root_cell':[at%_frame.TILE_W,at//_frame.TILE_W],'origin':[x,y],'path':path,'extent':[w,h],'before':before,'expected':original,'reason':'variant DDS pixels or bounding extrema unchanged from native'})
    continue
   old_range=sample_range(field0,nx,ny,w,h)
   if old_range is None:counts['native_calibration_failed']+=1;continue
   wanted=sample_range(field,x,y,w,h)
   if wanted is None:counts['current_sampling_failed']+=1;continue
   if max(abs(old_range[j]-original[j]) for j in (0,1))>TOL:
    # Vanilla authored a padded range that the samples do not reproduce. Keep
    # its widening-only padding around the new extrema; leaving the stale range
    # sinks or lifts the tile against its neighbours (stepped "walls").
    pad=(original[0]-old_range[0],original[1]-old_range[1])
    wanted=[max(0.,wanted[0]+min(0.,pad[0])),min(1.,wanted[1]+max(0.,pad[1]))]
    counts['padded_ranges_recalibrated']+=1
   else:
    counts['calibrated_roots']+=1;counts['calibrated_generic' if '/generic/generic_' in path.replace(chr(92),'/') else 'calibrated_special_or_water']+=1
   before=[lo,hi];drift=max(abs(wanted[j]-before[j]) for j in (0,1))
   if drift<=TOL:continue
   counts['range_mismatches']+=1;changes.append({'offset':layer*_frame.TILE_W*_frame.TILE_H*20+at*20+12,'layer':layer,'root_cell':[at%_frame.TILE_W,at//_frame.TILE_W],'origin':[x,y],'path':path,'extent':[w,h],'before':before,'expected':wanted,'drift':drift})
  report['variants'].append({'variant':variant,'cache_path':str(cp),'dds_path':str(dp),'cache_sha256':hashes[0].result(),'dds_sha256':hashes[1].result(),'counts':dict(counts),'changes':changes})
 report['native_cache_sha256'],report['native_dds_sha256']=(h.result() for h in native_hashes);hashing.shutdown()
 return report

def apply(report):
 with ThreadPoolExecutor(2) as hashing:
  for v in report['variants']:
   cache_now,dds_now=hashing.map(file_digest,(Path(v['cache_path']),Path(v['dds_path'])));assert cache_now==v['cache_sha256'];assert dds_now==v['dds_sha256']
 if not any(v['changes'] for v in report['variants']):return
 plan_hash=digest(json.dumps([v['changes'] for v in report['variants']],sort_keys=True).encode())[:12];tag='-'.join(v['cache_sha256'][:12] for v in report['variants'])+'-'+plan_hash;journal=_frame.WORKSPACE/'analysis'/('native-cache-range-journal-'+tag+'.json');journal.parent.mkdir(parents=True,exist_ok=True)
 if journal.exists():assert json.loads(journal.read_text(encoding='utf-8'))==report,'Existing journal does not match deterministic input: '+str(journal)
 else:journal.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
 for v in report['variants']:
  path=Path(v['cache_path'])
  with path.open('r+b') as f:
   for c in v['changes']:
    f.seek(c['offset']);assert f.read(8)==struct.pack('<2f',*c['before']);f.seek(c['offset']);f.write(struct.pack('<2f',*c['expected']))
   f.flush()
  v['output_sha256']=file_digest(path)
 report['applied']=True;report['rollback_journal']=str(journal)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',type=Path,default=_frame.WORKSPACE/'work/release-input');p.add_argument('--game',type=Path,default=tw3k_paths.GAME_DATA);p.add_argument('--apply',action='store_true');p.add_argument('--output',type=Path);args=p.parse_args();r=audit(args.stage,args.game)
 if args.apply:apply(r)
 if args.output:
  target=args.output.resolve();assert (target.is_relative_to(ROOT.resolve()) or target.is_relative_to(_frame.WORKSPACE.resolve())) and not target.is_relative_to(args.stage.resolve());   # the workspace may lie outside this checkout (a worktree's scripts on the main workspace)
  target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8')
 summary={k:v for k,v in r.items() if k!='variants'};summary['variants']=[{k:v for k,v in i.items() if k!='changes'} for i in r['variants']];print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
