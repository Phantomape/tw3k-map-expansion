"""Read-only terrain continuity preflight; --output saves the evidence."""
from __future__ import annotations
import argparse,hashlib,json,os,struct
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image
from final_terrain_surface import load_final_surface,sample_final_surface,point_resample_low_surface,low_edge_limits
from analyze_reference import ROOT,read_hex
import campaign_world as _frame
from native_edit_mask import at_world   # same edit definition as the height passes
from coarse_mesh_skirts import skirt_offsets   # skirt bottoms sit 1.0 below the surface by design
from campaign_world import TERRAIN_VARIANTS
A=_frame.WORKSPACE;DX=_frame.DX

def sample(a,x,y):
 x,y=np.clip(x,0,a.shape[1]-1),np.clip(y,0,a.shape[0]-1);ix,iy=x.astype(int),y.astype(int);jx,jy=np.minimum(ix+1,a.shape[1]-1),np.minimum(iy+1,a.shape[0]-1);fx,fy=x-ix,y-iy
 return (a[iy,ix]*(1-fx)+a[iy,jx]*fx)*(1-fy)+(a[jy,ix]*(1-fx)+a[jy,jx]*fx)*fy

def stats(a):
 if not len(a):return {'count':0}
 return {'count':int(len(a)),'max':float(np.max(a)),'p99_9':float(np.percentile(a,99.9)),'median':float(np.median(a))}

def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

SUBMERGED_EPS=1e-3   # world units: a pixel lies under the frontier river water this far below its stamped level

def river_level(analysis_dir,map_path):
 """Polish's stamped frontier river level (TH, TW, NaN off the band) and a note; None when the file is
 missing or was stamped for another map.hex. Pairs with both samples under it are river bed under the
 water (river_channels RIVER BEDS), not visible ground: counted apart, not as hard failures."""
 lp=analysis_dir/'river-water-level.npy';sp=lp.with_suffix('.json')
 if not(lp.exists() and sp.exists()):return None,'no river level file: nothing counted as submerged'
 if json.loads(sp.read_text(encoding='utf-8')).get('map_sha256')!=sha(map_path):return None,'river level stamped for another map.hex: nothing counted as submerged'
 return np.load(lp),f'both samples under the stamped river level by {SUBMERGED_EPS}'

def land_at(cells,x,z):
 q,r=_frame.raw_hex(x,z)
 return (cells[r,q,0]&3)!=1

def original_surface(path,shape):
 return point_resample_low_surface(load_final_surface(path.parent.name,path.parents[3]))

def slopes(s,alpha,cells,original,limits,grade=1.0,level=None):
 yy,xx=np.mgrid[:s.shape[0],:s.shape[1]];land=land_at(cells,xx*DX/2,yy*DX/2);result=[]
 under=np.zeros(s.shape,bool)
 if level is not None:L=level[:s.shape[0],:s.shape[1]];under[:L.shape[0],:L.shape[1]]=np.isfinite(L)&(s[:L.shape[0],:L.shape[1]]<np.nan_to_num(L,nan=-1e9)-SUBMERGED_EPS)
 for axis in (0,1):
  left=(slice(None,-1),slice(None)) if axis==0 else (slice(None),slice(None,-1));right=(slice(1,None),slice(None)) if axis==0 else (slice(None),slice(1,None))
  d=np.abs(s[right]-s[left]);aj=np.abs(alpha[right]-alpha[left]);pair=land[left]&land[right];native=pair&(alpha[left]==0)&(alpha[right]==0);edited=pair&(np.maximum(alpha[left],alpha[right])>.02);transition=edited&(np.minimum(alpha[left],alpha[right])<.98)
  lx,ly=xx[left],yy[left];(kx0,ky0),(kx1,ky1)=_frame.from_authored_tile(720,955),_frame.from_authored_tile(775,1005);known_window=(lx>=kx0)&(lx<=kx1)&(ly>=ky0)&(ly<=ky1)
  original_delta=np.abs(original[right]-original[left]);grade_step=grade*DX/2*(2/np.sqrt(3) if axis==0 else 1);allowed=limits[axis]
  changed=(np.abs(s[left]-original[left])>.0001)|(np.abs(s[right]-original[right])>.0001);excess=np.maximum(d-allowed,0);hard_cliffs=pair&changed&(excess>.002);submerged=hard_cliffs&under[left]&under[right];hard_cliffs&=~submerged
  baseline=stats(d[native]);threshold=max(.04,baseline.get('p99_9',.04)*1.5);rows=[];candidates=np.flatnonzero(transition&(d>threshold));candidates=candidates[np.argsort(d.ravel()[candidates])[-12:][::-1]]
  for at in candidates:
   y,x=np.unravel_index(at,d.shape);rows.append({'tile_xy':[int(x),int(y)],'world_xz':[x*DX/2,y*DX/2],'height_delta':float(d[y,x]),'alpha_jump':float(aj[y,x])})
  result.append({'axis':'z' if axis==0 else 'x','all_height_delta':stats(d.ravel()),'native_land_delta':baseline,'edited_land_delta':stats(d[edited]),'transition_delta':stats(d[transition]),'transition_outliers':int((transition&(d>threshold)).sum()),'alpha_jumps_over_quarter':int((transition&(aj>.25)).sum()),'diagnostic_threshold':threshold,'worst_transition_locations':rows,'natural_slope_is_not_hard_failure':True,'hard_edited_cliff_pairs':int(hard_cliffs.sum()),'submerged_cliff_pairs':int(submerged.sum()),'known_problem_window_max_edited_delta':float(d[edited&known_window].max(initial=0)),'newly_introduced_step_excess':stats(excess[pair&changed]),'allowed_grade_step':float(grade_step),'gradient_gate_margin':.002})
 return result

def audit_dds_gradients(current_path,original_path,cells,grade=1.,allowance=.002,chunk_rows=64,level=None):
 """Read actual game texture in strips and reject introduced pixel steps."""
 def open_height(path):
  with path.open('rb') as f:header=f.read(128)
  assert header[:4]==b'DDS ';h,w=struct.unpack_from('<2I',header,12);assert path.stat().st_size==128+h*w*2
  return np.memmap(path,mode='r',dtype='<u2',offset=128,shape=(h,w))
 current,original=open_height(current_path),open_height(original_path);assert current.shape==original.shape;h,w=current.shape
 th,tw=cells.shape[0]*2+1,cells.shape[1]*2;unit=14.3/65535;zs=2/np.sqrt(3);pitch_x,pitch_z=DX/2*tw/w,DX*zs/2*th/h
 stamps=[(p.stat().st_size,p.stat().st_mtime_ns) for p in (current_path,original_path)]
 tx=np.arange(w)*tw/w;col=np.floor(tx).astype(int)   # LF pixel u lies in tile floor(u/4), owned by tile_hex
 results=[{'axis':axis,'physical_pixel_pitch':pitch,'edited_land_pairs':0,'hard_failure_pairs':0,'submerged_pairs':0,'maximum_land_delta':0.,'maximum_introduced_excess':0.,'worst_examples':[]} for axis,pitch in [('x',pitch_x),('z',pitch_z)]];changed_pixels=0
 # Strips run on threads (numpy releases the GIL) and the two sha256s in the background;
 # each strip returns its own counts, merged below in strip order (same sums, maxima, example order).
 def strip(start):
  end=min(h,start+chunk_rows);stop=min(h,end+1);cur=np.asarray(current[start:stop],dtype=np.int32);old=np.asarray(original[start:stop],dtype=np.int32);part=[{'edited_land_pairs':0,'hard_failure_pairs':0,'submerged_pairs':0,'maximum_land_delta':0.,'maximum_introduced_excess':0.,'worst_examples':[]} for _ in results];count=int((cur[:end-start]!=old[:end-start]).sum())
  raw_y=np.arange(start,stop);ty=(h-1-raw_y)*th/h;q,r=_frame.tile_hex(col[None,:],np.floor(ty).astype(int)[:,None]);land=(cells[r,q,0]&3)!=1
  under=np.zeros(cur.shape,bool)
  if level is not None:
   L=level[np.clip(np.floor(ty).astype(int),0,level.shape[0]-1)][:,np.clip(col,0,level.shape[1]-1)]
   under=np.isfinite(L)&(_frame.LEVEL+(cur/65535-_frame.NORM)*_frame.GAIN<np.nan_to_num(L,nan=-1e9)-SUBMERGED_EPS)
  for axis,index in [(1,0),(0,1)]:
   if axis==1:c,o=cur[:end-start],old[:end-start];pair_land=land[:end-start,:-1]&land[:end-start,1:];changed=(c[:,:-1]!=o[:,:-1])|(c[:,1:]!=o[:,1:]);both_under=under[:end-start,:-1]&under[:end-start,1:]
   else:c,o=cur,old;pair_land=land[:-1]&land[1:];changed=(c[:-1]!=o[:-1])|(c[1:]!=o[1:]);both_under=under[:-1]&under[1:]
   delta=np.abs(np.diff(c,axis=axis)).astype(np.float32)*unit;original_delta=np.abs(np.diff(o,axis=axis)).astype(np.float32)*unit;out=part[index];allowed=np.maximum(original_delta,grade*results[index]['physical_pixel_pitch']);excess=np.maximum(delta-allowed,0);edited_land=pair_land&changed;bad=edited_land&(excess>allowance);sub=bad&both_under;bad&=~sub;out['submerged_pairs']+=int(sub.sum())
   out['edited_land_pairs']+=int(edited_land.sum());out['hard_failure_pairs']+=int(bad.sum());out['maximum_land_delta']=max(out['maximum_land_delta'],float(delta[pair_land].max(initial=0)));out['maximum_introduced_excess']=max(out['maximum_introduced_excess'],float(excess[edited_land].max(initial=0)))
   if bad.any():
    pos=np.unravel_index(np.argmax(np.where(bad,excess,-1)),excess.shape);py,px=start+pos[0],pos[1];jy,jx=py+(axis==0),px+(axis==1);tile_y=(h-1-py)*th/h
    def height(pixel):return _frame.LEVEL+(float(pixel)/65535-_frame.NORM)*_frame.GAIN
    out['worst_examples'].append({'dds_xy':[int(px),int(py)],'tile_xy':[float(tx[px]),float(tile_y)],'world_xz_approx':[float(tx[px]*DX/2),float(tile_y*DX*zs/2)],'current_height_pair':[height(current[py,px]),height(current[jy,jx])],'original_height_pair':[height(original[py,px]),height(original[jy,jx])],'height_delta':float(delta[pos]),'original_delta':float(original_delta[pos]),'allowed_delta':float(allowed[pos]),'introduced_excess':float(excess[pos])})
  return count,part
 with ThreadPoolExecutor(2) as hashing,ThreadPoolExecutor(min(8,os.cpu_count() or 1)) as pool:
  digests=[hashing.submit(sha,p) for p in (current_path,original_path)]
  for count,part in pool.map(strip,range(0,h,chunk_rows)):
   changed_pixels+=count
   for out,got in zip(results,part):
    out['edited_land_pairs']+=got['edited_land_pairs'];out['hard_failure_pairs']+=got['hard_failure_pairs'];out['submerged_pairs']+=got['submerged_pairs'];out['maximum_land_delta']=max(out['maximum_land_delta'],got['maximum_land_delta']);out['maximum_introduced_excess']=max(out['maximum_introduced_excess'],got['maximum_introduced_excess']);out['worst_examples']+=got['worst_examples']
  current_sha,original_sha=(d.result() for d in digests)
 assert stamps==[(p.stat().st_size,p.stat().st_mtime_ns) for p in (current_path,original_path)],'DDS changed during audit'
 for out in results:out['worst_examples']=sorted(out['worst_examples'],key=lambda x:x['introduced_excess'],reverse=True)[:12]
 return {'current_source':str(current_path),'original_source':str(original_path),'current_sha256':current_sha,'original_sha256':original_sha,'dimensions':[w,h],'changed_pixels':changed_pixels,'height_unit_per_uint16':unit,'grade':grade,'quantization_allowance':allowance,'basis':'DDS north-to-south; LF pixel u at tile u/4; physical Z uses renderer ZS approximation','hard_failure_pairs':sum(x['hard_failure_pairs'] for x in results),'submerged_pairs':sum(x['submerged_pairs'] for x in results),'axis_results':results}

def meshes(pre,surface,alpha,cells,tolerance):
 actual_surface=load_final_surface(pre.name,pre.parents[2])
 parts,results,errors=[],[],[];paths=sorted((pre/'global_meshes').glob('land_mesh_*.rigid_model_v2'))
 assert paths, f'No staged land meshes: {pre}'
 for number,path in enumerate(paths):
  b=path.read_bytes();mp=struct.unpack_from('<I',b,152)[0];typ,flags,size,vo,vc,io,ic=struct.unpack_from('<HH5I',b,mp);assert typ==101 and io-vo==vc*16 and len(b)==mp+size
  v=np.frombuffer(b,'<f4',vc*4,mp+vo).reshape(vc,4);assert np.isfinite(v).all();used=np.zeros(vc,bool);indices=np.frombuffer(b,'<u2',ic,mp+io);assert indices.max(initial=0)<vc;used[indices]=True
  x,y=_frame.tile_coord(v[:,0]),_frame.tile_coord(v[:,2]);aw,target=sample(alpha,x,y),sample_final_surface(actual_surface,x,y);edited=used&land_at(cells,v[:,0],v[:,2])&(aw>.02)&at_world(pre.name,v[:,0],v[:,2]*_frame.DZ/DX);skirt=skirt_offsets(v,indices)<0;negative=edited&((v[:,1]<-40)|skirt);top=edited&~negative;err=np.abs(v[:,1]-target);errors.append(err[top]);bad=np.flatnonzero(top&(err>tolerance));worst=bad[np.argsort(err[bad])[-8:][::-1]]
  results.append({'mesh':path.name,'edited_top_vertices':int(top.sum()),'mismatches':int(len(bad)),'negative_skirt_vertices_diagnostic':int(negative.sum()),'max_error':float(err[top].max(initial=0)),'worst':[{'vertex':int(i),'world_xz':v[i,[0,2]].astype(float).tolist(),'actual':float(v[i,1]),'canonical':float(target[i]),'alpha':float(aw[i]),'error':float(err[i])} for i in worst]})
  ids=np.flatnonzero(top)
  if len(ids):parts.append(np.column_stack((v[ids,:3],np.full(len(ids),number))))
 points=np.concatenate(parts) if parts else np.empty((0,4));repeated=crossbad=0;maxgap=0.;examples=[]
 if len(points):
  keys=_frame.lattice_keys(points[:,[0,2]]);order=np.lexsort((keys[:,3],keys[:,2],keys[:,1],keys[:,0]));points,keys=points[order],keys[order];starts=np.r_[0,1+np.flatnonzero(np.any(keys[1:]!=keys[:-1],axis=1))];counts=np.diff(np.r_[starts,len(points)]);low=np.minimum.reduceat(points[:,1],starts);high=np.maximum.reduceat(points[:,1],starts);cross=np.minimum.reduceat(points[:,3],starts)!=np.maximum.reduceat(points[:,3],starts);gap=high-low;repeated=int((counts>1).sum());bad=cross&(gap>tolerance);crossbad=int(bad.sum());maxgap=float(gap[cross].max(initial=0));at=np.flatnonzero(bad)
  for i in at[np.argsort(gap[at])[-12:][::-1]]:examples.append({'world_xz':points[starts[i],[0,2]].tolist(),'height_gap':float(gap[i]),'min_max_height':[float(low[i]),float(high[i])]})
 combined=np.concatenate(errors) if errors else np.empty(0)
 return {'edited_top_error':stats(combined),'mismatched_vertices':int((combined>tolerance).sum()),'samepoint_crossmesh_mismatches':crossbad,'samepoint_max_crossmesh_gap':maxgap,'samepoint_repeated_groups':repeated,'samepoint_examples':examples,'models':results}

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',type=Path,default=A/'work/release-input');p.add_argument('--analysis-dir',type=Path,default=A/'analysis');p.add_argument('--map',type=Path,default=A/'work/final-map/map.hex');p.add_argument('--output',type=Path);p.add_argument('--transition-max-step',type=float,default=1.0);p.add_argument('--tolerance',type=float,default=.04);args=p.parse_args();meta=read_hex(args.map);w,h=meta['width'],meta['height'];b=args.map.read_bytes();cells=np.frombuffer(b[-4-w*h*16:-4],np.uint8).reshape(h,w,16)
 level,level_note=river_level(args.analysis_dir,args.map)
 report={'stage':str(args.stage),'map_sha256':sha(args.map),'submerged_rule':level_note,'tolerance':args.tolerance,'live_verified':False,'visual_acceptance_complete':False,'variants':[],'hard_failures':[],'coverage_gaps':['Differently tessellated edges need interpolated edge/ray checks.','Detailed tile/rivers/water/LOD are not reconstructed.','Startpos world frame and effective pack load order need separate verification.']}
 for variant in TERRAIN_VARIANTS:
  ap=args.analysis_dir/f'polished-edit-alpha-{variant}.npy';ap=ap if ap.exists() else args.analysis_dir/'polished-edit-alpha.npy';alpha=np.load(ap,mmap_mode='r');s=np.load(args.analysis_dir/f'polished-surface-{variant}.npy',mmap_mode='r');assert s.shape==alpha.shape and np.isfinite(s).all() and np.isfinite(alpha).all() and np.all((alpha>=0)&(alpha<=1));m=meshes(args.stage/'terrain/campaigns'/variant,s,alpha,cells,args.tolerance);original_path=A/'terrain-before-polish'/f'terrain/campaigns/{variant}/lf_height_map.dds';original=original_surface(original_path,s.shape);row={'variant':variant,'alpha_source':str(ap),'original_height_source':str(original_path),'original_height_sha256':sha(original_path),'surface_neighbours':slopes(s,alpha,cells,original,low_edge_limits(load_final_surface(variant,A/'terrain-before-polish'),DX,args.transition_max_step),args.transition_max_step,level),'meshes':m};report['variants'].append(row)
  rel=Path(f'terrain/campaigns/{variant}/lf_height_map.dds');dds=audit_dds_gradients(args.stage/rel,A/'terrain-before-polish'/rel,cells,grade=args.transition_max_step,level=level);row['actual_dds_continuity']=dds
  if dds['hard_failure_pairs']:report['hard_failures'].append({'variant':variant,'actual_dds_introduced_gradient_pairs':dds['hard_failure_pairs']})
  cliffs=sum(x['hard_edited_cliff_pairs'] for x in row['surface_neighbours'])
  row['submerged_pairs']={'dds_gradient':dds['submerged_pairs'],'surface_cliff':sum(x['submerged_cliff_pairs'] for x in row['surface_neighbours'])}
  if cliffs:report['hard_failures'].append({'variant':variant,'edited_transition_cliff_pairs':cliffs})
  if m['mismatched_vertices'] or m['samepoint_crossmesh_mismatches']:report['hard_failures'].append({'variant':variant,'canonical_vertex_mismatches':m['mismatched_vertices'],'crossmesh_samepoint_mismatches':m['samepoint_crossmesh_mismatches']})
 report['known_geometry_checks_passed']=not report['hard_failures']
 if args.output:args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({'known_geometry_checks_passed':report['known_geometry_checks_passed'],'hard_failures':report['hard_failures'],'variants':[{'variant':r['variant'],'max_canonical_error':r['meshes']['edited_top_error'].get('max'),'max_crossmesh_gap':r['meshes']['samepoint_max_crossmesh_gap'],'maximum_neighbour_delta':max(a['all_height_delta']['max'] for a in r['surface_neighbours'])} for r in report['variants']]},ensure_ascii=False),flush=True)
 return int(bool(report['hard_failures']))
if __name__=='__main__':raise SystemExit(main())
