"""Replant native foliage and rocks on the synchronized expanded surface."""
import json,struct,math
from pathlib import Path
import numpy as np
from terrain_noise import fbm
import foliage_biomes
from analyze_reference import ROOT,read_hex
import campaign_world as _frame
import tree_bake as tb   # --- PROPS BAKE (bake design step 3) --- stateless draws, accept(tx, ty, species)
AUTHORED=_frame.load_authored_coordinates();BANDS=AUTHORED['biome_bands'];NQ,NR=AUTHORED['noise_origin']
from final_terrain_surface import load_final_surface,sample_final_surface
from campaign_world import TERRAIN_VARIANTS
A=_frame.WORKSPACE;S=A/'work/release-input';BASE=A/'terrain-before-polish';DX=_frame.DX;DZ=_frame.DZ
m=read_hex(A/'work/final-map/map.hex');w,h=m['width'],m['height'];b=(A/'work/final-map/map.hex').read_bytes();d=np.frombuffer(b[-4-w*h*16:-4],np.uint8).reshape(h,w,16);ids=(d[:,:,1].astype(int)<<5)+(d[:,:,0].astype(int)>>3)-1;names=m['land_regions']+m['sea_regions'];plan=json.loads((A/'work/current-region-design.json').read_text(encoding='utf-8'));bulk=np.isin(ids,[names.index(c['region']) for c in plan['regions'] if c.get('introduced_in_bulk')]);mask=bulk|_frame.load_canvas_array(A/'analysis/continuous-land-mask.npy');land=(d[:,:,0]&3)!=1
def sample(field,x,y):
 if field is surface:return float(sample_final_surface(actual_surface,x,y))
 x=max(0.,min(float(x),field.shape[1]-1));y=max(0.,min(float(y),field.shape[0]-1));ix,iy=int(x),int(y);nx,ny=min(ix+1,field.shape[1]-1),min(iy+1,field.shape[0]-1);fx,fy=x-ix,y-iy
 return float((field[iy,ix]*(1-fx)+field[iy,nx]*fx)*(1-fy)+(field[ny,ix]*(1-fx)+field[ny,nx]*fx)*fy)
# Inside the native map vanilla foliage is kept and follows the ground: a tree
# moves by the ground change under it. Only where the ground moved more than
# MOVED_LIMIT (plateaus, rim cuts) or right at a new town is it removed, and only
# there is the procedural replanting allowed. Deleting every vanilla tree with
# edit alpha > 1e-7 thinned native forests and left bare rings.
# Authored towns built inside vanilla forest (Rinan, Guanghan dependency) were
# hidden under the canopy with a 1.5-hex clearing (live checks 2026-10-19):
# clear about what vanilla leaves around its own towns.
# Props bake (bake design step 3, tree_bake): every procedural draw is a stateless hash keyed by position and
# purpose (no shared random stream: one tree more or less upstream moves nothing else), and every planter asks
# acc.accept(tx, ty, species) before it writes a row (dry footprint, treeline, flat-arid rocks, autumn shrubs).
# The four post-plant filters still run at the end and must count 0 (logged: post_filter_removals) before they
# are deleted. Vanilla native rows are not touched by either.
MOVED_LIMIT,TOWN_CLEAR=.6,7   # 4 still left the canopy in front of Rinan (live check 2026-10-19 b)
def vanilla_ground(variant):
 from repair_campaign_river_roots import Assets
 d=Assets(original_only=True).read(f'terrain/campaigns/{variant}/lf_height_map.dds');hh,ww=struct.unpack_from('<2I',d,12)
 return _frame.LEVEL+(np.frombuffer(d,'<u2',offset=128).reshape(hh,ww)[::-1].astype(np.float32)/65535-_frame.NORM)*_frame.GAIN
def native_xz(x,z):
 q=x/DX;return _frame.PAD_LEFT<=q<_frame.PAD_LEFT+_frame.NATIVE_W and _frame.PAD_BOTTOM<=z/DZ<_frame.PAD_BOTTOM+_frame.NATIVE_H
def ground_moved(x,z):
 """Final minus vanilla ground at a native world point, both sampled exactly as
 sample_final_surface does (tile t -> LF pixel 4t)."""
 tx,tz=x/DX*2,z/DZ*2
 px=tx*4-_frame.NATIVE_OFFSET_LF[0];pz=tz*4-_frame.NATIVE_OFFSET_LF[1]
 return sample(surface,tx,tz)-sample(van_ground,px,pz)
# Array forms of sample/native_xz/ground_moved for the deterministic per-tree and
# per-hex tests (over a million calls). Same arithmetic per element: in sample()
# the Python-float weights take the field's dtype (NumPy 2 weak scalars), so here
# they are cast to it explicitly.
def sample_many(field,x,y):
 if field is surface:return sample_final_surface(actual_surface,x,y)
 x=np.clip(np.asarray(x,np.float64),0.,field.shape[1]-1);y=np.clip(np.asarray(y,np.float64),0.,field.shape[0]-1)
 ix,iy=x.astype(np.int64),y.astype(np.int64);nx,ny=np.minimum(ix+1,field.shape[1]-1),np.minimum(iy+1,field.shape[0]-1);fx,fy=x-ix,y-iy
 t=field.dtype;ux,wx,uy,wy=(1-fx).astype(t),fx.astype(t),(1-fy).astype(t),fy.astype(t)
 return ((field[iy,ix]*ux+field[iy,nx]*wx)*uy+(field[ny,ix]*ux+field[ny,nx]*wx)*wy).astype(np.float64)
native_many=tb.native_many
def ground_moved_many(x,z):
 tx,tz=x/DX*2,z/DZ*2
 px=tx*4-_frame.NATIVE_OFFSET_LF[0];pz=tz*4-_frame.NATIVE_OFFSET_LF[1]
 return sample_many(surface,tx,tz)-sample_many(van_ground,px,pz)
def town_distance(q,r,chunk=4096,centres=None):
 """float(np.min(np.hypot(centres[:,0]-q,centres[:,1]-r))) per point (centres: towns)."""
 c=towns if centres is None else centres;out=np.empty(len(q))
 for s in range(0,len(q),chunk):out[s:s+chunk]=np.hypot(c[None,:,0]-q[s:s+chunk,None],c[None,:,1]-r[s:s+chunk,None]).min(1)
 return out
def positions(rows):
 return np.array([struct.unpack_from('<3f',row,0) for row in rows],np.float64).reshape(-1,3)
reports=[]
# --- HEHUANG PATCH (hehuang_terrain.py) --- vanilla trees dropped by weight, replanted by ground class
import hehuang_terrain;HH=hehuang_terrain.load()
HH_GF=HH.ground_field() if HH is not None else None   # the polish ground field (same contour as the steppe ground)
def hehuang_w_many(x,z):
 tx=np.clip((np.asarray(x)/DX*2).astype(np.int64),0,_frame.TILE_W-1);tz=np.clip((np.asarray(z)/DZ*2).astype(np.int64),0,_frame.TILE_H-1);return HH_GF[tz,tx]
HH_SPECIES={0:(('temperate_tree_katsura_small_1','temperate_tree_katsura_medium_1'),.008,.008),   # steppe/meadow
            1:(('general_rock_medium_1',),.06,.06),2:(('general_rock_medium_1',),.06,.06),           # alpine rock, snow/scree
            3:(('cold_tree_fir_small_1','cold_tree_fir_medium_1'),.36,.10),                          # conifer (patch > -.1 / else)
            4:(('arid_tree_poplar_small_1','arid_tree_poplar_medium_1'),.004,.004)}                  # Chaka desert
if HH is not None:HH_SPECIES={**HH_SPECIES,**HH.species}   # other patches (korea) bring their classes' species in their npz
for variant in TERRAIN_VARIANTS:
 van_ground=vanilla_ground(variant)
 surface=np.load(A/f'analysis/polished-surface-{variant}.npy');actual_surface=load_final_surface(variant,S);alpha=np.load(A/f'analysis/polished-edit-alpha-{variant}.npy')
 towns=np.array([c['town'] for c in plan['regions']],float)
 # --- PROPS BAKE --- the accept rules (and the same masks for the post-filters at the end): biome_profiles
 # predicates (alpine_treeline_tile, flat_arid_hex, species_bans) on today's inputs (strategic_ink hex biome,
 # classify rerun on raw RESOLVE in the treeline box), through foliage_biomes; not the landform field yet.
 _rlp=A/'analysis/river-water-level.npy'
 river_level=tb.load_river_level(_rlp,alpha.shape)
 if river_level is None:print(variant,'WARNING: analysis/river-water-level.npy missing (polish RIVER CHANNELS writes it): river-bank trees are tested against the water plane only',flush=True)
 _tlr=A/'analysis/west-real-terrain.npz'
 _tlm=foliage_biomes.treeline_hexes(_tlr,(NQ,NR)) if _tlr.exists() else None
 _flat=foliage_biomes.flat_rock_hexes()
 _cold=((d[:,:,7].astype(int)>>1)-1)==m['climates'].index('cold')
 _bans=foliage_biomes.species_ban_masks(_cold)
 # [to read the river spec] the local water level: max of the river level, the lake level on lake tiles and the
 # sea plane on sea tiles (without lakes and sea a new row of trees would stand on low beaches and lake shores)
 water,water_kind=tb.local_water(d,names,river_level,alpha.shape)
 acc=tb.TreeAccept(surface,sample_many,water,water_kind,treeline=_tlm,flat=_flat,bans=_bans)
 # The shared forest density per hex (landform-tiles forest_density): the band chance and the frontier fill read
 # it, as the woodland ground does (one field for both; user decision 2026-10-26)
 _forest=tb.forest_field_hex(A,(h,w));fdens=_forest['density'];fshape=_forest['shape'];fon=_forest['fill']>=tb.SHAPE_FILL_MIN;_oasf=_forest['oasis'];_eco,_econ=_forest['eco'],_forest['names']
 # Towns as the engine draws them (display_position), in x/DX, z/DZ units (HEX_DZ/DZ=HEX_DX/DX).
 shown=np.column_stack((towns[:,0],towns[:,1]+towns[:,0]%2*.5))*(_frame.HEX_DX/DX)
 source_path=BASE/f'campaign_maps/{variant}/display/trees/trees.campaign_tree_list';assert source_path.exists();source=source_path.read_bytes()
 # Canvas trees come from the pre-polish baseline; native trees come straight
 # from vanilla (shifted), so foliage deleted by any earlier stage returns.
 from repair_campaign_river_roots import Assets
 vanilla_trees=Assets(original_only=True).read(f'campaign_maps/{variant}/display/trees/trees.campaign_tree_list')
 def tree_groups(blob):
  pos=24;out=[]
  for g in range(struct.unpack_from('<I',blob,20)[0]):
   start=pos;n=struct.unpack_from('<H',blob,pos)[0];pos+=2;name=blob[pos:pos+n].decode();pos+=n;header=blob[start:pos];count=struct.unpack_from('<I',blob,pos)[0];pos+=4;rows=[]
   for i in range(count):
    start=pos;seasons=struct.unpack_from('<I',blob,pos+14)[0];pos+=18+seasons*4;rows.append(bytearray(blob[start:pos]))
   out.append((name,header,rows))
  assert pos==len(blob)
  return out
 base_groups=tree_groups(source);native_groups=tree_groups(vanilla_trees);hh_dropped=0
 assert [g[0] for g in base_groups]==[g[0] for g in native_groups],'tree group tables differ from vanilla'
 SX,SZ=_frame.NATIVE_SHIFT_WORLD;groups=[];base_rejected=0
 for (name,header,base_rows),(_,_,vanilla_rows) in zip(base_groups,native_groups):
  rows=[]
  p=positions(base_rows);x,z=p[:,0],p[:,2];ax,az=x/DX*2,z/DZ*2
  drop=native_many(x,z);test=np.flatnonzero(~drop&(0<=ax)&(ax<alpha.shape[1])&(0<=az)&(az<alpha.shape[0]))
  drop[test]=sample_many(alpha,ax[test],az[test])>1e-7
  # --- PROPS BAKE --- kept pre-polish canvas rows answer accept as well (the post-filters dropped them after)
  _kb=np.flatnonzero(~drop)
  if len(_kb):_ka=acc.accept(ax[_kb],az[_kb],name);drop[_kb[~_ka]]=True;base_rejected+=int((~_ka).sum())
  rows.extend(bytes(row) for row,d in zip(base_rows,drop) if not d)
  p=positions(vanilla_rows);x,y,z=p[:,0]+SX,p[:,1],p[:,2]+SZ;at=np.flatnonzero(native_many(x,z))
  # Town clearance from the tree to the displayed town model (the block corner q*DX
  # left the clearing up to a hex SW of the model).
  moved=ground_moved_many(x[at],z[at])
  keep=~(np.abs(moved)>MOVED_LIMIT);keep[keep]=~(town_distance(x[at][keep]/DX,z[at][keep]/DZ,centres=shown)<TOWN_CLEAR)
  if HH is not None:   # --- HEHUANG PATCH (hehuang_terrain.py) --- drop w > .3, and with probability w over .05-.3
   # This stream thins VANILLA rows (fixed input, fixed order): it stays sequential so the kept vanilla rows
   # stay byte-identical through the props bake.
   _hw=hehuang_w_many(x[at],z[at]);_u=np.random.default_rng(9050+len(groups)).random(len(at))
   # Thinned with the ground field: none kept past the steppe contour, a fraying density before it.
   _drop=_u<np.clip((_hw-.35)/.4,0,1)**1.5;keep&=~_drop;hh_dropped=hh_dropped+int(_drop.sum())
  # Vanilla y == 0 rows go on the final ground (the LF procedural rows use), the rest by the ground change
  # (tree_bake.native_row_heights; invariant native_y0_trees_grounded).
  _ki=at[keep];_g0=np.zeros(len(_ki));_z0=y[_ki]==0
  if _z0.any():_g0[_z0]=sample_many(surface,x[_ki][_z0]/DX*2,z[_ki][_z0]/DZ*2)
  for i,y_ in zip(_ki,tb.native_row_heights(y[_ki],moved[keep],_g0)):
   row=vanilla_rows[i];struct.pack_into('<3f',row,0,x[i],y_,z[i]);rows.append(bytes(row))
  template=bytes(base_rows[-1]) if base_rows else (bytes(vanilla_rows[-1]) if vanilla_rows else None)
  groups.append({'name':name,'header':header,'rows':rows,'template':template,'added':0})
 pos=len(source)
 assert pos==len(source);byname={g['name']:g for g in groups}
 logical=(d[:,:,5].astype(int)>>4)+((d[:,:,6].astype(int)&7)<<4)-1
 # Patch-based density: valley forests, arid oasis belts and rocky high slopes.
 # The first three tests draw no random numbers: evaluate them for every hex at
 # once, then walk the survivors in the original row-major order.
 rr,qq=np.meshgrid(np.arange(3+NR%2,h-3,2),np.arange(3+NQ%2,w-3,2),indexing='ij');rr,qq=rr.ravel(),qq.ravel()
 ty=(rr+qq%2*.5)*2;ok=land[rr,qq];ok[ok]=sample_many(alpha,qq[ok]*2,ty[ok])>.02
 wz=(rr+qq%2*.5)*DZ;check=ok&native_many(qq*DX,wz);check[check]=np.abs(ground_moved_many(qq[check]*DX,wz[check]))<=MOVED_LIMIT;ok&=~check
 if HH is not None:   # --- HEHUANG PATCH (hehuang_terrain.py) --- patch hexes (w > .05) replant in their own pass below
  _hhex=HH_GF[np.minimum(ty.astype(np.int64)+1,_frame.TILE_H-1),np.minimum(qq*2+1,_frame.TILE_W-1)]>.5;hh_ok=land[rr,qq]&_hhex;ok&=~_hhex
 distances=np.full(len(rr),-1.);distances[ok]=town_distance(qq[ok].astype(np.float64),rr[ok].astype(np.float64));ok&=~(distances<8)
 grounds=np.zeros(len(rr));grounds[ok]=sample_many(surface,qq[ok]*2,ty[ok])
 # --- PROPS BAKE --- per hex one species draw and one chance draw, keyed by (q, r): the hex's outcome no longer
 # depends on how many draws the hexes before it took
 _bq,_br=qq[ok],rr[ok];_usp=tb.u01('band.species',_bq,_br);_uch=tb.u01('band.chance',_bq,_br);_hit=[]
 for k_,(r,q,distance,ground) in enumerate(zip(_br.tolist(),_bq.tolist(),distances[ok].tolist(),grounds[ok].tolist())):
   patch=(math.sin((q-NQ)*.071)+math.cos((r-NR)*.083)+math.sin((q-NQ+r-NR)*.137))/3
   ground_name=m['land_ground_types'][int(logical[r,q])]
   arid=ground_name in ('desert','hilly_plains') and q<BANDS['foliage_arid_max_q'] and r>BANDS['arid_min_r']
   cold=r>BANDS['cold_min_r'] and q>BANDS['cold_min_q'];tropical=r<BANDS['tropical_max_r']
   # (map forest / grassland ground: foliage_biomes.map_ground_chance below)
   u_=float(_usp[k_])
   _b=_econ[int(_eco[r,q])] if int(_eco[r,q])<len(_econ) else None
   shp=float(fshape[r,q]) if fon[r,q] else foliage_biomes.patch_shape(_b,patch)   # field shape, else the pre-bake patch
   picked=foliage_biomes.choose(q,r,ground,shp,distance,tb.Draw(u_),_b,float(_oasf[r,q]))   # eco_bio class + shared forest density (shape: mean 1 per class, level BAND_MEAN)
   if picked is not None:
    species,chance=picked
   elif ground>4.8:
    species='general_rock_medium_1';chance=.06
   elif arid:
    species='arid_tree_poplar_small_1' if u_<.5 else 'arid_tree_poplar_medium_1';chance=.22 if distance<12 else .007
   elif cold:
    species='cold_tree_fir_small_1' if u_<.5 else 'cold_tree_fir_medium_1';chance=.28 if patch>.0 else .03
   elif tropical:
    species='bamboo_small_1' if u_<.18 else 'subtropical_castanopsis_medium_1';chance=.40 if patch>-.1 else .05
   else:
    species='temperate_tree_katsura_small_1' if u_<.5 else 'temperate_tree_katsura_medium_1';chance=.19 if patch>.05 else .015
   # map ground: forest hexes keep the table's floor (FOREST_DENSITY map_forest_floor), grassland the steppe cap
   chance=foliage_biomes.map_ground_chance(species,chance,ground_name,shp,_b)
   if _uch[k_]>chance:continue
   _hit.append((q,r,species))
 # Trees of the hit hexes (1 rock or 3 trees each), in hex then tree order. Jitter (+-.7 hex) around the hex's
 # block centre, tile (2q+1, 2r+odd+1); around the block corner q*DX it carried coastal trees into the sea
 # hexes west/south. Keyed by (q, r, j).
 def plant_hits(hits,purpose='band'):
  """Plant 1 rock or 3 trees per (q, r, species) hit; returns the rows kept per hit (dry footprint, land, accept)."""
  if not hits:return np.zeros(0,np.int64)
  _hq=np.array([h_[0] for h_ in hits],np.int64);_hr=np.array([h_[1] for h_ in hits],np.int64);_hs=np.array([h_[2] for h_ in hits],object)
  _n=np.array([1 if s_.startswith('general_rock') else 3 for s_ in _hs]);_i=np.repeat(np.arange(len(hits)),_n);_j=np.arange(len(_i))-np.repeat(np.cumsum(_n)-_n,_n)
  Q_,R_,SP_=_hq[_i],_hr[_i],_hs[_i]
  tx_=2*Q_+1+2*tb.uniform(-.7,.7,purpose+'.jx',Q_,R_,_j);ty_=2*R_+Q_%2+1+2*tb.uniform(-.7,.7,purpose+'.jy',Q_,R_,_j)
  keep_=sample_many(alpha,tx_,ty_)>1e-7
  # The tree's own tile's hex must be land too (as for the seam trees below).
  hq2,hr2=_frame.tile_hex(tx_.astype(np.int64),ty_.astype(np.int64));keep_&=land[hr2,hq2]
  if keep_.any():keep_[keep_]=acc.accept(tx_[keep_],ty_[keep_],SP_[keep_])
  _y=np.zeros(len(tx_));_y[keep_]=sample_many(surface,tx_[keep_],ty_[keep_])
  for i_ in np.flatnonzero(keep_):
   g=byname[SP_[i_]];g['rows'].append(tb.pack_row(g['template'],float(tx_[i_]),float(ty_[i_]),float(_y[i_])));g['added']+=1
  return np.bincount(_i[keep_],minlength=len(hits))
 _kept=plant_hits(_hit)
 # --- TOWN OASIS FLOOR (foliage_biomes.town_oasis_rule: biome_profiles CONSTS['props']['TOWN_OASIS']) --- a town's
 # oasis hexes are the walk hexes within floor_ring_hex of it (nearest town), eco_bio class in classes or
 # oasis_classes, landform-tiles 'oasis' >= oasis_min; it keeps at least ceil(min_hit_share x their count) of them
 # planted with poplars: when the band's draws left fewer, the next free oasis hexes in a stateless order keyed by the
 # town and the hex (tb.u01 'oasis.floor') are planted too, until the floor holds or the hexes run out.
 _tor=foliage_biomes.town_oasis_rule();oasis_report={'rule':{k:(list(v) if isinstance(v,tuple) else v) for k,v in _tor.items()},'towns':{}}
 _cls=np.array([n_ in _tor['classes'] or n_ in _tor['oasis_classes'] for n_ in _econ]+[False])
 _rq,_rr,_rd=qq[ok],rr[ok],distances[ok]
 _in=(_rd>=_tor['floor_ring_hex'][0])&(_rd<_tor['floor_ring_hex'][1])&_cls[np.minimum(_eco[_rr,_rq].astype(np.int64),len(_econ))]&(_oasf[_rr,_rq]>=_tor['oasis_min'])
 _rq,_rr=_rq[_in],_rr[_in]
 _own=np.empty(len(_rq),np.int64)
 for s_ in range(0,len(_rq),4096):_own[s_:s_+4096]=np.hypot(towns[None,:,0]-_rq[s_:s_+4096,None],towns[None,:,1]-_rr[s_:s_+4096,None]).argmin(1)
 _hitkey={(h_[0],h_[1]) for h_,k_ in zip(_hit,_kept) if k_>0 and h_[2] in foliage_biomes.SPECIES_OASIS}
 _floor=foliage_biomes.oasis_floor_hits(np.bincount(_own,minlength=len(towns)),_tor)
 for t_ in np.flatnonzero(_floor):
  sel=np.flatnonzero(_own==t_);cq,cr=_rq[sel],_rr[sel]
  have=sum((int(a_),int(b_)) in _hitkey for a_,b_ in zip(cq,cr));need=int(_floor[t_]);added_=0;have0=have
  if have<need:
   free=np.array([(int(a_),int(b_)) not in _hitkey for a_,b_ in zip(cq,cr)])
   order=np.argsort(tb.u01('oasis.floor',np.full(len(cq),int(towns[t_,0])),np.full(len(cq),int(towns[t_,1])),cq,cr)[free],kind='stable')
   fq,fr=cq[free][order],cr[free][order];us_=tb.u01('oasis.species',fq,fr);pos=0
   while have<need and pos<len(fq):
    take=min(need-have,len(fq)-pos)
    batch=[(int(fq[pos+j_]),int(fr[pos+j_]),foliage_biomes.SPECIES_OASIS[0] if us_[pos+j_]<.5 else foliage_biomes.SPECIES_OASIS[1]) for j_ in range(take)]
    kept_=plant_hits(batch,'oasis');have+=int((kept_>0).sum());added_+=int((kept_>0).sum());pos+=take
  oasis_report['towns'][plan['regions'][t_].get('region',str(t_))]={'oasis_hexes':int(len(sel)),'floor':need,'band_hits':int(have0),'planted_hexes':int(have),'floor_added':added_}
 oasis_report['oasis_towns']=len(oasis_report['towns']);oasis_report['below_floor']=sum(1 for v_ in oasis_report['towns'].values() if v_['planted_hexes']<v_['floor'])
 oasis_report['floor_added_hexes']=sum(v_['floor_added'] for v_ in oasis_report['towns'].values())
 print(variant,'town oasis floor:',json.dumps({k_:oasis_report[k_] for k_ in ('oasis_towns','below_floor','floor_added_hexes')}),flush=True)
 if HH is not None:   # --- HEHUANG PATCH (hehuang_terrain.py) --- replant the patch by ground class (own random stream)
  hh_added=0;_g=HH.ground
  # Draws keyed by the patch name and the hex (props bake): each patch replants as when alone, whichever
  # patches are present (the old per-patch streams 9051 + PATCHES index did the same by seed).
  for _pp in HH.parts:
   _P='hh.'+_pp.name;_pin=hh_ok&_pp.in_window(np.minimum(qq*2+1,_frame.TILE_W-1),np.minimum(2*rr+qq%2+1,_frame.TILE_H-1))
   _pr,_pq=rr[_pin],qq[_pin];_td=town_distance(_pq.astype(np.float64),_pr.astype(np.float64));_uh=tb.u01(_P+'.hit',_pq,_pr);_trees=[]
   for k_,(r,q) in enumerate(zip(_pr.tolist(),_pq.tolist())):
    if _td[k_]<8:continue
    cls=int(_g[min(2*r+q%2+1,_frame.TILE_H-1),min(2*q+1,_frame.TILE_W-1)])
    if cls not in HH_SPECIES:continue
    kinds,dense,sparse=HH_SPECIES[cls][:3];patch=(math.sin((q-NQ)*.071)+math.cos((r-NR)*.083)+math.sin((q-NQ+r-NR)*.137))/3
    if _uh[k_]>(dense if patch>-.1 else sparse):continue
    rock=kinds[0].startswith('general_rock')
    _n,_jit=HH_SPECIES[cls][3:5] if len(HH_SPECIES[cls])>3 else ((1 if rock else 3),.7)   # optional per-class trees per hit and jitter (hexes)
    _trees+=[(q,r,j,kinds,_jit) for j in range(_n)]
   if not _trees:continue
   Q_=np.array([t_[0] for t_ in _trees],np.int64);R_=np.array([t_[1] for t_ in _trees],np.int64);J_=np.array([t_[2] for t_ in _trees],np.int64);JT_=np.array([t_[4] for t_ in _trees],np.float64)
   _ks=tb.u01(_P+'.species',Q_,R_,J_);SP_=np.array([t_[3][min(int(u_*len(t_[3])),len(t_[3])-1)] for t_,u_ in zip(_trees,_ks)],object)
   tx_=2*Q_+1+2*(-JT_+2*JT_*tb.u01(_P+'.jx',Q_,R_,J_));ty_=2*R_+Q_%2+1+2*(-JT_+2*JT_*tb.u01(_P+'.jy',Q_,R_,J_))
   keep_=sample_many(alpha,tx_,ty_)>1e-7
   hq2,hr2=_frame.tile_hex(tx_.astype(np.int64),ty_.astype(np.int64));keep_&=land[hr2,hq2]&(_g[np.clip(ty_.astype(np.int64),0,_g.shape[0]-1),np.clip(tx_.astype(np.int64),0,_g.shape[1]-1)]!=5)
   if keep_.any():keep_[keep_]=acc.accept(tx_[keep_],ty_[keep_],SP_[keep_])
   _y=np.zeros(len(tx_));_y[keep_]=sample_many(surface,tx_[keep_],ty_[keep_])
   for i_ in np.flatnonzero(keep_):
    g=byname[SP_[i_]];g['rows'].append(tb.pack_row(g['template'],float(tx_[i_]),float(ty_[i_]),float(_y[i_])));g['added']+=1;hh_added+=1
  print(variant,'Hehuang patch: dropped',hh_dropped,'vanilla trees, replanted',hh_added,flush=True)
 # --- FRONTIER FOREST (frontier_forest.py) --- one tree per empty hex of real forest at vanilla
 # density (Siberia, Indochina; not the Western Regions band), rock clusters and shrubs on frontier ground
 import frontier_forest
 def _ff_ok(q_,r_):
  ty_=(r_+q_%2*.5)*2;ok_=land[r_,q_].copy();ok_[ok_]=sample_many(alpha,q_[ok_]*2,ty_[ok_])>.02
  wz_=(r_+q_%2*.5)*DZ;ck_=ok_&native_many(q_*DX,wz_);ck_[ck_]=np.abs(ground_moved_many(q_[ck_]*DX,wz_[ck_]))<=MOVED_LIMIT;ok_&=~ck_
  if HH is not None:ok_&=~(HH_GF[np.minimum(ty_.astype(np.int64)+1,_frame.TILE_H-1),np.minimum(q_*2+1,_frame.TILE_W-1)]>.5)
  return ok_
 _ffc=A/'analysis/road-tile-cover.npy'
 _ffr=frontier_forest.plant(byname,{n_:rows_ for n_,_,rows_ in native_groups},dict(sample=sample_many,surface=surface,alpha=alpha,land=land,cells=d,towns_shown=shown,replant_ok=_ff_ok,town_clear=TOWN_CLEAR,road_cover=np.load(_ffc) if _ffc.exists() else None,names=names,accept=acc.accept,forest=_forest))
 print(variant,'frontier forest:',json.dumps(_ffr),flush=True)
 # --- end FRONTIER FOREST
 # Climate seams: a tile climate cannot blend, so where neighbouring tile cells
 # differ the floor textures meet in half-hex steps. Dress those seams in edited
 # land with the trees and rocks of the cell's own climate (it hides the steps
 # the way vanilla dresses its own terrain seams).
 import fastops as _fo   # cm_read: the per-block native_height_field decode loop, vectorised
 tile_climate=_fo.cm_read((S/f'terrain/campaigns/{variant}/climate_map.cm').read_bytes())[::-1]
 seam=np.zeros(tile_climate.shape,bool)
 seam[:,1:]|=tile_climate[:,1:]!=tile_climate[:,:-1];seam[:,:-1]|=tile_climate[:,1:]!=tile_climate[:,:-1]
 seam[1:]|=tile_climate[1:]!=tile_climate[:-1];seam[:-1]|=tile_climate[1:]!=tile_climate[:-1]
 seam_species={0:('subtropical_castanopsis_medium_1','bamboo_small_1'),1:('cold_tree_fir_small_1','cold_tree_fir_medium_1'),
               2:('temperate_tree_katsura_small_1','temperate_tree_katsura_medium_1'),3:('arid_tree_poplar_small_1','arid_tree_poplar_medium_1')}
 dressed=0
 # Deterministic cell tests and the grove noise for all seam cells at once (fbm
 # is elementwise on 1-D input).
 sy,sx=np.nonzero(seam);cell=(sy<alpha.shape[0])&(sx<alpha.shape[1]);cell[cell]=~(alpha[sy[cell],sx[cell]]<=.02)
 # Odd-q owner rule (tile_hex): odd columns own rows 2r+1 and 2r+2, not 2r and 2r+1.
 sq,sr=_frame.tile_hex(sx,sy);cell&=(sq<w)&(sr<h);cell[cell]=land[sr[cell],sq[cell]]
 groves=.5+.6*fbm(sq[cell].astype(np.float32),sr[cell].astype(np.float32),(NQ,NR),6,3,seed=91).astype(np.float64)
 # --- PROPS BAKE --- every draw keyed by the seam tile (x, y) and the tree index k (it was the band loop's
 # stream, drawn only for the cells that passed the town and grove tests)
 cy_,cx_,cq_,cr_=sy[cell],sx[cell],sq[cell],sr[cell]
 # Patchy groves, not a line along every seam cell (rows read as planted).
 go_=~(town_distance(cq_.astype(np.float64),cr_.astype(np.float64))<6)&~(groves<.55)&~(tb.u01('seam.hit',cx_,cy_)>.3)
 cy_,cx_=cy_[go_],cx_[go_]
 # Trees only (1 or 2 per cell): rocks set on seam cells still traced the seams as lines.
 _n=1+tb.integers(2,'seam.n',cx_,cy_);_i=np.repeat(np.arange(len(cx_)),_n);K_=np.arange(len(_i))-np.repeat(np.cumsum(_n)-_n,_n)
 TY_,TX_=cy_[_i],cx_[_i]
 _kinds=[seam_species.get(int(c_),seam_species[2]) for c_ in tile_climate[TY_,TX_]]
 _ks=tb.u01('seam.species',TX_,TY_,K_);SP_=np.array([k_[min(int(u_*len(k_)),len(k_)-1)] for k_,u_ in zip(_kinds,_ks)],object)
 tx2=TX_+.5+1.6*tb.normal('seam.dx',TX_,TY_,K_);ty2=TY_+.5+1.6*tb.normal('seam.dy',TX_,TY_,K_)
 keep_=np.array([s_ in byname and byname[s_]['template'] is not None for s_ in SP_],bool)
 keep_&=(0<=tx2)&(tx2<alpha.shape[1]-1)&(0<=ty2)&(ty2<alpha.shape[0]-1)
 if keep_.any():keep_[keep_]=sample_many(alpha,tx2[keep_],ty2[keep_])>.02
 # The jitter can carry a coastal seam tree off the land: its own tile's hex must be land too.
 hq2,hr2=_frame.tile_hex(np.clip(tx2,0,None).astype(np.int64),np.clip(ty2,0,None).astype(np.int64));keep_&=land[hr2,hq2]
 if keep_.any():keep_[keep_]=acc.accept(tx2[keep_],ty2[keep_],SP_[keep_])
 _y=np.zeros(len(tx2));_y[keep_]=sample_many(surface,tx2[keep_],ty2[keep_])
 for i_ in np.flatnonzero(keep_):
  g=byname[SP_[i_]];g['rows'].append(tb.pack_row(g['template'],float(tx2[i_]),float(ty2[i_]),float(_y[i_])));g['added']+=1;dressed+=1
 print(variant,'climate seam dressing:',dressed,'trees and rocks on',int(seam.sum()),'seam cells',flush=True)
 # --- NATIVE EDGE TREES (native_edge_trees.py) --- canvas tree density feathered into vanilla's
 # within FEATHER hexes of the native rectangle where vanilla forest meets the edge (Ailao q=888:
 # ~15 canvas trees per hex column against ~33-40 vanilla). Hashed draws, added rows ask accept; native rows untouched.
 import native_edge_trees
 _etr=native_edge_trees.feather(groups,land,lambda tx_,ty_:sample_many(surface,tx_,ty_),_ff_ok,shown,TOWN_CLEAR,np.load(_ffc) if _ffc.exists() else None,accept=acc.accept)
 print(variant,'native edge trees:',json.dumps(_etr),flush=True)
 # --- end NATIVE EDGE TREES
 # --- RIPARIAN BELTS (riparian_belts.py) --- poplar and reed belts along the frontier rivers in real desert
 # (after every other planting; rows in the water are refused by accept, WET TREES below must find none)
 import riparian_belts
 _rib=riparian_belts.plant(byname,{n_:rows_ for n_,_,rows_ in native_groups},dict(sample=sample_many,surface=surface,land=land,cells=d,towns_shown=shown,road_cover=np.load(_ffc) if _ffc.exists() else None,cold=_cold,channels=A/'work/final-map/river-channels.json',accept=acc.accept))
 print(variant,'riparian belts:',json.dumps(_rib),flush=True)
 # --- end RIPARIAN BELTS
 # ===== POST-PLANT FILTERS (props bake: kept running behind accept until they count 0; design 3.4) =====
 # --- L2048 WET TREES --- canvas trees whose footprint ground lies under the campaign water
 # plane are dropped. The lake bowls round the shore past the hex outline, so land-hex tiles
 # sink below -0.015 and the hex land test kept 158 trees in Baikal's shallows (live 2048l).
 # Native (vanilla) rows are left as they are.
 # River water: polish's RIVER CHANNELS block raises the water along the frontier rivers to a
 # level that follows the banks (analysis/river-water-level.npy, NaN off the band), well above
 # the plane, so the plane test kept the first ring of bank trees standing in the river.
 # The local level is the highest river level under the footprint, with the same margin (river_level: the
 # per-cell local level loaded above for accept; vanilla-style rivers reach ~9.3 here, far above WET_LEVEL).
 wet_dropped,river_dropped=tb.drop_wet_trees(groups,surface,river_level,sample_many)
 print(variant,'wet trees dropped:',wet_dropped,'below the plane,',river_dropped,'below the local river level',flush=True)
 # --- end L2048 WET TREES
 # --- TREELINE ALPINE (foliage_biomes.drop_treeline_trees) --- no canvas trees above the real treeline on the
 # classify alpine meadow / steppe (after every placement: no random stream moves, every other row stays);
 # native rows are vanilla's and stay. Dropped rows per map region: analysis/treeline-alpine-trees.json.
 if _tlm is not None:
  def _tl_hex(x_,z_):
   q_,r_=_frame.world_hex(x_,z_);return np.asarray(q_,np.int64),np.asarray(r_,np.int64),~native_many(x_,z_)
  treeline_dropped,_tlw=foliage_biomes.drop_treeline_trees(groups,_tl_hex,_tlm)
  _tlg={}
  for q_,r_ in _tlw:_n_=names[ids[r_,q_]] if 0<=ids[r_,q_]<len(names) else '(none)';_tlg[_n_]=_tlg.get(_n_,0)+1
  (A/'analysis/treeline-alpine-trees.json').write_text(json.dumps({'variant':variant,'mask_hexes':int(_tlm.sum()),'dropped_by_species':treeline_dropped,'dropped_by_region':dict(sorted(_tlg.items())),'live_verified':False},indent=1),encoding='utf-8')
  print(variant,'treeline alpine trees dropped:',json.dumps(treeline_dropped),'mask hexes',int(_tlm.sum()),flush=True);del _tlw,_tlg
 else:treeline_dropped={};print(variant,'WARNING: no analysis/west-real-terrain.npz: TREELINE ALPINE skipped',flush=True)
 # --- end TREELINE ALPINE
 # --- ROCK RELIEF (foliage_biomes.drop_flat_rocks) --- canvas rocks off flat arid ground (after every draw:
 # no random stream changes); native rows are vanilla's and stay
 def _canvas_hex(x_,z_):
  q_,r_=_frame.world_hex(x_,z_);return np.asarray(q_,np.int64),np.asarray(r_,np.int64),~native_many(x_,z_)
 flat_rocks=foliage_biomes.drop_flat_rocks(groups,_canvas_hex,_flat)
 print(variant,'flat-ground rocks dropped:',json.dumps(flat_rocks),flush=True)
 # --- end ROCK RELIEF
 # --- AUTUMN SHRUBS (foliage_biomes.drop_autumn_shrubs) --- no autumn-orange shrubs / katsura on open desert
 # or cold land (map.hex climate byte 7: (index + 1) << 1)
 autumn=foliage_biomes.drop_autumn_shrubs(groups,_canvas_hex,_cold,_bans)
 print(variant,'autumn shrubs dropped:',json.dumps(autumn),flush=True)
 # --- end AUTUMN SHRUBS
 # Every planted row asked accept with the same rules and masks, so the filters above must have removed nothing.
 post_removed=wet_dropped+river_dropped+sum(treeline_dropped.values())+sum(flat_rocks.values())+sum(autumn.values())
 print(variant,'props bake: accept rejected',json.dumps(acc.rejected),'dry by water',json.dumps(acc.dry_by_water),'(kept pre-polish canvas rows',base_rejected,'of them); post-filter removals',post_removed,flush=True)
 if post_removed:print(variant,'WARNING: post-plant filters removed',post_removed,'rows that accept had passed: a planter bypasses accept or the rules diverged',flush=True)
 # ===== end POST-PLANT FILTERS
 del _tlm
 out=bytearray(source[:24])
 for g in groups:out.extend(g['header']+struct.pack('<I',len(g['rows']))+b''.join(g['rows']))
 p=S/f'campaign_maps/{variant}/display/trees/trees.campaign_tree_list';p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(out)
 report={'variant':variant,'native_species':True,'baseline_source':str(source_path),'height_sampling':'bilinear','native_instances_preserved_outside_edit':sum(len(g['rows'])-g['added'] for g in groups),'replanted_trees_and_rocks':sum(g['added'] for g in groups),'total_instances':sum(len(g['rows']) for g in groups),'town_clearance_hexes':8,'by_species':{g['name']:g['added'] for g in groups if g['added']},'native_edge_trees':_etr,'wet_trees_dropped':wet_dropped,'river_bank_trees_dropped':river_dropped,'flat_rocks_dropped':flat_rocks,'autumn_shrubs_dropped':autumn,'treeline_trees_dropped':treeline_dropped,'riparian_belts':_rib,'town_oasis':oasis_report,'props_bake':{'rng':'tree_bake splitmix64 (purpose, position)','seed':tb.SEED,'accept_rejected':acc.rejected,'dry_rejected_by_water':acc.dry_by_water,'baseline_rows_rejected':base_rejected,'post_filter_removals':post_removed},'live_verified':False};reports.append(report);print(json.dumps(report),flush=True)
report=dict(reports[0]);report['variants']=reports;(A/'analysis/terrain-native-foliage.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
