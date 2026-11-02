"""Geographic relief and native tile coverage for the expanded campaign.

Idempotent: reads an immutable snapshot of the pre-polish stage. Logical regions,
settlements, passability and SDK exports are not altered by this renderer pass.
"""
import tw3k_paths
import os,json,struct,math,shutil,io,re,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageFilter,ImageDraw,ImageFont,PngImagePlugin
from analyze_reference import ROOT,inventory
from fast_hex import read_hex   # analyze_reference.read_hex, vectorised (same dict)
import campaign_world as _frame
from coarse_mesh_skirts import seat_on_surface
from final_terrain_surface import snap_grid_to_towns,point_resample_low_surface
from native_height_field import parse,decode,rebuild
from native_cache_replacement import remove_edited_owners
from native_campaign_climate import inherit_climate,band_climate,land_tile_coverage,box_blur,propagate_nearest
import time as _time
import fastops   # output-identical fast helpers (BFS, box blur, cm_write, dilation, pack index)
_T0=_time.time();_TL=[_T0]
# TOWN SPEC (town ground mix, coordinator option A 2026-10-31): the town floor and cluster levels round a town
# are an inverse-distance-squared mix of the towns (distance from the town's plot edge, TOWN_MIX_CORE tile cells
# for a capital / other town), not a Gaussian: at a plot's edge the mix is that town's own level, so the slope
# projection no longer rims each plateau with a grade-cap step where the mix differed from it (v21 art review:
# dark creases between the Wusun plateaus, a fine rim round every plateau in the 2048 mesh). The weight fades to 0
# at TOWN_MIX_REACH cells; the old Gaussian (sigma 10) still decides where the mix applies at all (gate), so the
# mixed area is unchanged. Invariant: town_rim_step.
TOWN_MIX_CORE=(10,8)   # tile cells: plot radius of a capital / other town (as LAKE_TOWN_PLOT)
TOWN_MIX_REACH=40
def _town_mix_weight(dx,dy,core):
 d=np.hypot(dx,dy)
 return (np.clip(1-d/TOWN_MIX_REACH,0,1)**2/np.maximum(d-core,1.)**2).astype(np.float32)

def _mark(label):
 print(f'[polish-timing] {label}: {_time.time()-_TL[0]:.1f}s (total {_time.time()-_T0:.0f}s)',flush=True);_TL[0]=_time.time()

def _select_blend_sources(flag,take,bridge,slots):
 """Keep provenance only for the candidate actually selected. Low two bits name its source; the
 selected material (0-31) fits in the upper bits of the existing uint8 flag, with no second full-map array."""
 if flag is None:return
 flag&=np.where(take[...,0],2,0).astype(np.uint8)
 if bridge is not None:flag[bridge]=1   # a bridge replaces the donor candidate, including woodland
 active=(flag&3)!=0
 flag[active]|=slots[active].astype(np.uint8)<<2

def _discard_blend_sources(cur,flag,rows=256):
 """Drop selected candidates subsequently replaced by an existing writer, in bounded row blocks.
 Report them separately so they cannot dilute the banned/re-sampled ratio at the material pass."""
 out={'tone_bridge':0,'donor_woodland':0}
 if flag is None:return out
 for row in range(0,len(flag),rows):
  f=flag[row:row+rows];changed=((f&3)!=0)&((f>>2)!=cur[row:row+rows,:,0])
  for bit,name in ((1,'tone_bridge'),(2,'donor_woodland')):out[name]+=int((changed&((f&bit)!=0)).sum())
  f[changed]=0
 return out

def _blend_source_ban_report(cur,flag,fields,gm,rows=1024):
 """Actual selected candidates still present at the material pass and banned there (one denominator per source)."""
 oh,ow=cur.shape[:2];out={}
 for bit,name in ((1,'tone_bridge'),(2,'donor_woodland')):
  n=b=0
  for row in range(0,oh,rows):
   end=min(oh,row+rows);f=(flag[row:end]&bit)>0
   if not f.any():continue
   lb,al=gm._ban_rows_fields(fields,(np.arange(ow)+.5)/4-.5,(oh-1-np.arange(row,end)+.5)/4-.5)
   n+=int(f.sum());b+=int((f&gm._banned(lb,al,cur[row:end,:,0])).sum())
  out[name]={'pixels':n,'banned_resampled':b,'share':round(b/max(n,1),5)}
 return out

def _town_ground_selection(current,template,take_disc,protected,normal_eligible,town_clear=None,dry_clear=False,banned=None):
 """One selection in the existing town writer: its normal disc, plus dry town-clearance pixels.
 The extra selection only replaces wetland 31 inside the town spec's clearance; protected pixels never write.
 A wetland source template for a selected clearance pixel is a rule-data error, never an accepted replacement.
 banned (town spec): template pixels whose slot is banned for the town's own tiles; they keep the ground-materials
 slot. Returns (selected, dry, (banned pixels skipped, of them dry-clear pixels))."""
 normal=(take_disc&~protected) if normal_eligible else np.zeros(current.shape,bool)
 dry=np.zeros(current.shape,bool)
 if dry_clear:
  if town_clear is None:raise ValueError('Town dry-clear selection requires the town-spec clearance mask')
  dry=town_clear&(current==31)&~protected
 selected=normal|dry
 skipped=(0,0)
 if banned is not None:
  skipped=(int((selected&banned).sum()),int((dry&banned).sum()))
  selected=selected&~banned;dry=dry&~banned
 if dry_clear:
  invalid=selected&town_clear&(template==31)
  if invalid.any():
   first=tuple(int(v) for v in np.argwhere(invalid)[0])
   raise ValueError(f'Town ground template contains wetland 31 on {int(invalid.sum())} selected clearance pixels; first local offset {first}: fix TOWN_GROUND rule data')
 return selected,dry,skipped

from terrain_noise import fbm,ridged
# Tile-grid fbm/ridged of the setup and variant relief, and the material blend's fbm, on the GPU
# (gpu_noise.fbm/fbm_at/ridged: same bytes as terrain_noise; CPU fallback without CuPy/CUDA or on
# any GPU error; TW3K_GPU=0 forces the CPU).
import gpu_noise
# Landform/biome tuning constants: biome_profiles (CONSTS['polish']); the names below read them back.
import biome_profiles as _bp;_BP=_bp.consts('polish')
# Native rim ramp width replaced by carried native height, cells kept at that height
# outside the native map, and the fade into authored relief (tile cells).
# TW3K_POLISH_VARIANTS selects the variant(s) one --loop-only process polishes;
# by default every variant in campaign_world.TERRAIN_VARIANTS (dlc07 only).
POLISH_VARIANTS=tuple(os.environ.get('TW3K_POLISH_VARIANTS',','.join(_frame.TERRAIN_VARIANTS)).split(','))
# Parallel build: one process per variant runs with --loop-only (per-variant height,
# blend, mesh and camera work), then one --merge run combines their alpha/report
# pieces and builds the shared tile cache exactly as a serial run would.
LOOP_ONLY='--loop-only' in sys.argv;MERGE='--merge' in sys.argv;ALL_VARIANTS=_frame.TERRAIN_VARIANTS
# --cache-only (sync_final_terrain_surface) reads only the climate, protection and
# canvas masks of the global setup; relief, edit alpha and town plots are skipped.
CACHE_ONLY='--cache-only' in sys.argv and not MERGE
NATIVE_RIM,RELIEF_HOLD,RELIEF_FADE=20,4,45
RIM_BLEND=16
# Mountain-front width (tile cells) and ridge height share where the native edge is a high plateau.
FRONT_FADE,FRONT_RIDGE,FRONT_MEANDER=80,.35,60
# Box-blur radius (tile cells) of the native height carried across the border.
CARRY_SMOOTH=12
# Noise displacement (tile cells) of the carried native height.
CARRY_SHIFT=24
# Native edge ring blended into the transition (tile cells) and its meander.
EDGE_RING,EDGE_RING_MEANDER=12,24
# Ring (tile cells) inside the native border where non-playable vanilla tiles are replaced.
TILE_RING_MIN,TILE_RING_MEANDER=8,32
# Settlement ground copied from a same-climate vanilla town (native hex), by
# tile climate 0 subtropical, 1 cold, 2 temperate, 3 arid; radii in hexes.
TOWN_GROUND=_BP['TOWN_GROUND']   # Jiaozhi, Youbeiping, Luoyang, Wuwei
TOWN_GROUND_R=_BP['TOWN_GROUND_R']
# Blend materials of vanilla woodland floors (never under vanilla towns).
WOODLAND_MATERIALS=_BP['WOODLAND_MATERIALS']
# Forest towns: where at least FOREST_TOWN[2] of the ground in the ring FOREST_TOWN[0]
# to FOREST_TOWN[1] hexes around a town is woodland material (the blend before town
# ground), the town keeps that forest floor, as vanilla forest towns do (Liaodong
# resource_1, Yunnan): a copied light disc in the forest read as a ring (Lin Yi, live
# 2048j O21), and widened it cut the native border (daner_farm). Only the trees thin.
FOREST_TOWN=_BP['FOREST_TOWN']
# Authored towns on non-playable vanilla land whose vanilla climate contradicts
# their geography (tile climate as above). Climate, floor tiles and blend are
# replaced within a noisy disc of TOWN_CLIMATE_R hexes (inner, outer).
TOWN_CLIMATE=_bp.region('town_climate')['towns']   # e.g. {'region key': 3}; Longle was moved to the Gobi instead (move_town.py)
TOWN_CLIMATE_R=_bp.region('town_climate')['radius_hex']
# Towns this many tile cells outside the native map take the native plateau datum.
NEAR_BORDER_TOWN=24
# Lowest plateau height for a town plot on land (vanilla coastal towns sit near 1.0-1.3).
MIN_TOWN_HEIGHT=1.0
# Box-blur radius (LF pixels, 4 per tile cell) used twice to round town aprons.
APRON_SMOOTH=4
# Box-blur radius (LF pixels) used twice to round flat town plot outlines.
PLOT_ROUND=6
# Steepest authored grade. It bounds every edited edge, and lowering it squeezed the
# remaining relief onto the old native border (whose base slope allowance is large).
TOWN_GRADE=1.0
# Native blend cells next to the old map edge that are replaced by the mirrored interior.
BLEND_RING=12
# Meander (tile cells) of the mirrored-native to donor blend transition.
BLEND_MEANDER=70
MIRROR_WARP=48   # pixels: warp of the mirrored native blend source
# Patterned field materials the mirrored band must not copy: a reflected paddy block
# (subtropical_4 stripes) reads as a symmetric twin across the native border (Rinan,
# live check 2048j). Those mirror pixels take the canvas donor instead.
MIRROR_SKIP=np.array(_BP['MIRROR_SKIP'],np.uint8)
# [EGOBI] mirrored vanilla woodland on real open ground takes the canvas donor (biome_profiles MIRROR_OPEN_RAMP)
MIRROR_WOOD=np.array(sorted(WOODLAND_MATERIALS),np.uint8);MIRROR_OPEN_RAMP=_BP['MIRROR_OPEN_RAMP']
REAL_DESERT_FADE=_BP['REAL_DESERT_FADE']
# EGOBI switch (user stop-loss 2026-10-26): 20b8b948 off for the video package - mirror_open_w 0, real_dgw 0 (real
# desert/steppe take the climate only from REAL_GROUND_FADE), desert pass weight back to real_gw x donor_share. The
# soft mirror band at the native west edge (Qilian / Hexi, q 884) had turned into a straight sand/green line.
# TW3K_EGOBI=1 turns it on (redo with the NONPLAY filler repaint).
EGOBI=os.environ.get('TW3K_EGOBI','0')=='1'
DONOR_WOODLAND=os.environ.get('TW3K_DONOR_WOODLAND','0')=='1'   # ground contrast fix §3: donor woodland floor by the shared forest density
TONE_BRIDGE=os.environ.get('TW3K_TONE_BRIDGE','0')=='1'   # ground contrast fix §1: bridge slots on high-contrast mirror/donor dither edges (needs the measured tone table)
MIRROR_EDGE=os.environ.get('TW3K_MIRROR_EDGE','0')=='1'   # ground contrast fix §2: the mirror mismatch ramp on the meandering native edge (0: the old straight ramp)
NONPLAY_RAMP=_BP['NONPLAY_RAMP']   # --- NONPLAY GROUND --- hexes (box radius) from vanilla playable land
# NONPLAY_GROUND off: user stop-loss 2026-10-26: filler repaint removed from this package; redo with trees + edge
# transition after. Off, polish treats the vanilla non-playable filler as before terrain-monotony: no repaint weight
# (nonplay_hex/nonplay_w 0, so the edge fade, the mirror ramp and the guard do nothing), real_on = real_w > 0, and
# no filler mask for farm fields, the wetland blend and the coast tiles. TW3K_NONPLAY_GROUND=1 turns it on.
NONPLAY_GROUND=os.environ.get('TW3K_NONPLAY_GROUND','0')=='1'
SINGLE_GROUND=os.environ.get('TW3K_SINGLE_GROUND','0')=='1'
# LAKE_BOWL off: user stop-loss 2026-10-31 (decisions, lake stop-loss): the v21b-v21d lake basin / shore heights (lake
# floor L, smooth waterline, lake bed to LAKE_FLOOR and ceilings U, flat-core bowl with the fine relief) failed the
# v21c / v21d art reviews (walls, dark ledges, pale water; the root cause is the one global water plane). Off, the
# lake bowl and the lake shore are the v20 ones byte for byte (60-cell bowl, no shore shaping, no floors / ceilings
# in the projection) and the lake shore invariants skip. The lake cap chain and the town ground mix (TOWN SPEC) stay.
# TW3K_LAKE_BOWL=1 turns the v21d lake basin on. The fix goes on as step 2 (a water level per lake, Bosten test pack).
LAKE_BOWL_V21=os.environ.get('TW3K_LAKE_BOWL','0')=='1'
# LAKE LEVEL (lake_level.py, step 2 of the lake stop-loss; height-bake-design 2.1): TW3K_LAKE_LEVEL=1 solves a water
# level per lake of data/lake-level-spec.json (Bosten) from its shore, writes analysis/lake-water-levels.json, keeps
# those lakes out of the bowl and carves them from their own level. Off (the default): every output as before.
import lake_level as _lake_level
LAKE_LEVEL=_lake_level.enabled()
if LAKE_LEVEL and LAKE_BOWL_V21:raise RuntimeError('TW3K_LAKE_LEVEL=1 is built on the v20 lakes: leave TW3K_LAKE_BOWL off')
if SINGLE_GROUND and os.environ.get('TW3K_VISUAL_DOMAINS','0')!='1':
 # The single ground bake reads the shared visual domains (visual_domains.build: native_keep, material and
 # vegetation takeover); there is no single-ground path without them. Both off (the defaults): nothing new runs.
 raise RuntimeError('TW3K_SINGLE_GROUND=1 requires TW3K_VISUAL_DOMAINS=1 (the single ground bake reads the shared '
                    'visual domains); set both to 1, or leave both unset/0 for the legacy ground path')
DONOR_WARP=_BP['DONOR_WARP']   # pixels: warp of the donor patch tiling (was 260 over 60 hexes, three octaves:
# its shear stretched the donor ground into parallel streaks once the patches held
# more than one material, Funan, offline view frontier-ground-1)
DONOR_SCALE=_BP['DONOR_SCALE']   # hexes: largest feature of the noise choosing among a climate's donor patches
DONOR_RELIEF=_BP['DONOR_RELIEF']   # weight of local relief (vs noise) in that choice
# Patch transitions: each tile holds the share of every patch within a band of
# DONOR_BAND cells (the patch choice thresholded and box-blurred twice: about 20
# hexes wide), and each pixel draws its patch from those shares with smooth noise
# of DONOR_GRAIN hexes. A near-hard threshold (.12 fray) set a dark forest patch
# straight against a light one with no intermediate tone (Funan, Siberia offline
# views 2048j; vanilla shades over tens of hexes).
DONOR_BAND=_BP['DONOR_BAND']
DONOR_GRAIN=_BP['DONOR_GRAIN']
# --- USR3 ECOTONE PATCH --- steppe/forest ecotone of the real biomes: half-width (tile cells,
# box-blurred twice) and noise grain (hexes); see the real-terrain block.
ECOTONE=_BP['ECOTONE']
ECOTONE_GRAIN=_BP['ECOTONE_GRAIN']
# --- MATERIALS ECOTONE --- share of the fine climate share (box radius ECO_DETAIL_R tile cells) put
# back into the donor climate score inside redrawn ecotones (see eco_zone): 1 = the score follows the
# radius-2 share there (bilinear per pixel: no tile staircases), 0 = the old 6-cell level set.
ECO_DETAIL,ECO_DETAIL_R=_BP['ECO_DETAIL'],_BP['ECO_DETAIL_R']
# --- USR3 BOSTEN PATCH --- lake rings: riparian (reed) ground only near river mouths
# (river share box-blurred twice over LAKE_MOUTH[0] cells, times LAKE_MOUTH[1]); the rest
# of a low-valley ring takes the steppe donor (desert steppe). Oasis belts draw no paddy
# or cold material: OASIS_REMAP maps the temperate donor's subtropical 23-26 and cold 4-7
# to temperate 27-30 (the patches hold 31-51% material 23-26, mostly paddy-striped 26).
LAKE_MOUTH=_BP['LAKE_MOUTH']
OASIS_REMAP=np.arange(256,dtype=np.uint8)
for (_a,_b),(_c,_d) in _BP['OASIS_REMAP']:OASIS_REMAP[_a:_b]=np.arange(_c,_d)
# --- FIX-BATCH TONAL BUFFER --- forest -> open-ground transitions in the vanilla manner.
# Vanilla shades from dense forest (cold_1/2) through mid-dark buffer ground (steppe_3/4,
# temperate_3/4, subtropical_4, cold_4) to bright grass over ~20 hexes; the canvas went
# from cold_3/4 straight to bright steppe/sand donor ground (luminance 6-12 / 12-20 hexes
# from dense forest: canvas north 135/152, vanilla 124/128 by the same measure). After the
# blend rows, donor ground near dense forest (>= BUFFER_DENSE[1] material 4/5 in a box of
# radius BUFFER_DENSE[0] tile cells) has its bright materials swapped for a buffer
# material of the same donor climate band (G channel) with probability
# interp(ring distance in tile cells, *TONAL_BUFFER), dithered by fbm of BUFFER_GRAIN hexes.
# Weight is low next to the forest (the cold donor already carries cold_3/4 there), full
# 16-24 hexes out, and gone by ~36 hexes (open steppe/desert keeps its own tones).
# Real-terrain canvas ground only (times donor_share and real_gw): native ground is untouched.
# Buffer materials per band are the ones vanilla's own transition bands use (G0 cold:
# temperate_4/3, cold_4; G1 arid/steppe: steppe_3/4; G2 temperate: temperate_3/4,
# subtropical_4; G3 subtropical: subtropical_4). Tropical grass takes the steppe donor (G1).
TONAL_BUFFER=_BP['TONAL_BUFFER']
BUFFER_DENSE=_BP['BUFFER_DENSE']
# --- FIX2 BUFFER SHAPE --- dither grain/octaves (2 hexes x 3 octaves of value noise read as a
# checker lattice), distance meander, and forest mass (box radius in tile cells, blurred twice;
# mass at the nearest dense cell mapped (m - lo) / span to the 0..1 weight factor).
BUFFER_GRAIN,BUFFER_OCTAVES=_BP['BUFFER_GRAIN'],_BP['BUFFER_OCTAVES']
BUFFER_MEANDER=_BP['BUFFER_MEANDER']
BUFFER_MASS=_BP['BUFFER_MASS']
# --- end FIX2 BUFFER SHAPE
BUFFER_REMAP=_BP['BUFFER_REMAP']   # subtropical (3): only subtropical_1; Funan's band is already darker than vanilla's G3 (107/112/116 vs 115/124/124 at 3-20 hexes)
BUFFER_TABLE=np.tile(np.arange(256,dtype=np.uint8),(256,1))
for _g,_mp in BUFFER_REMAP.items():
 for _a,_b in _mp.items():BUFFER_TABLE[_g,_a]=_b
# --- end FIX-BATCH TONAL BUFFER
# Share of each patch of a climate (default equal). The flattest subtropical and
# cold patches are 90-99% cold_1 (luminance ~74): at a full share they laid one
# dark sheet over every plain, town plots included (Funan, Lin Yi).
DONOR_SHARES=_BP['DONOR_SHARES']
DONOR_PHASE=_BP['DONOR_PHASE']   # pixels: tiling phase step from one patch of a climate to the next
DONOR_JITTER=_BP['DONOR_JITTER']   # pixels: per-pixel jitter of the donor lookup; frays vanilla's hex-stepped paint edges (Siberia/Yilou, live check 2048j)
# Reached over 24 tiles from the native border: offline audit simulation
# (2026-10-20) showed any ramp steeper than ~2 px per tile cuts the band at
# the border (cut up to 15 at east 1550,642), so the warp stays this gentle.
A=_frame.WORKSPACE;S=A/'work/release-input';BASE=A/'terrain-before-polish'
_frame.check_snapshot(BASE)
# --- HEHUANG PATCH (hehuang_terrain.py) --- load analysis/hehuang-patch.npz (None without it)
import hehuang_terrain;HH=hehuang_terrain.load()
AUTHORED=_frame.load_authored_coordinates();BANDS=AUTHORED['biome_bands'];NQ,NR=AUTHORED['noise_origin']
GAME=tw3k_paths.GAME_DATA
W,H=_frame.W,_frame.H;TW,TH=_frame.TILE_W,_frame.TILE_H;DX=_frame.DX;ZS=_frame.ZS;SX,SZ=_frame.NATIVE_SHIFT_RAW
LEVEL,NORM,GAIN=_frame.LEVEL,_frame.NORM,_frame.GAIN
SEA_FLOOR=LEVEL+(3084/65535-NORM)*GAIN   # vanilla dlc07 open-ocean LF (raw 3084), about -2.445
idx={}
for pack in (GAME/'terrain.pack',GAME/'fast.pack',GAME/'data.pack'):
 for e in fastops.pack_entries(pack):idx.setdefault(e['path'],(pack,e))   # same entries as inventory(pack,False)['entries']
def asset(k):
 p,e=idx[k];assert not e['compressed']
 with p.open('rb') as f:f.seek(e['offset']);return f.read(e['size'])
def base(k):
 p=BASE/k
 if not p.exists() and _frame.seed_from_stage_allowed(BASE):
  p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(S/k,p)
 return p.read_bytes()
def put(k,b):
 p=S/k;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
def put_parts(k,parts):
 # put(k,b''.join(parts)) without building the joined copy (same file bytes).
 p=S/k;p.parent.mkdir(parents=True,exist_ok=True)
 with open(p,'wb') as f:
  for part in parts:f.write(part)
# Vectorised TABLE_INDEXED reader: the per-block native_height_field decode loop, identical grid.
cm_grid=fastops.cm_read
# Vectorised TABLE_INDEXED writer, byte-identical to _cm_write_reference.
cm_write=fastops.cm_write
def _cm_write_reference(b,a,bounds=None):
 h,w=a.shape;padded=np.pad(a,((0,(-h)%16),(0,(-w)%16)),mode='edge');bs=[]
 for y in range(0,padded.shape[0],16):
  for x in range(0,padded.shape[1],16):
   v=np.ascontiguousarray(padded[y:y+16,x:x+16],dtype='<u2')
   bs.append(b'\0'+struct.pack('<H',int(v[0,0])) if np.all(v==v[0,0]) else b'\x8f'+v.tobytes())
 hd=bytearray(b);struct.pack_into('<2I',hd,10,w,h)
 if bounds:struct.pack_into('<6f',hd,26,*bounds)
 return rebuild(hd,bs)
def sample(a,x,y):
 x=np.clip(x,0,a.shape[1]-1);y=np.clip(y,0,a.shape[0]-1);xi=x.astype(np.int32);yi=y.astype(np.int32);xx=np.minimum(xi+1,a.shape[1]-1);yy=np.minimum(yi+1,a.shape[0]-1);fx=x-xi;fy=y-yi
 return (a[yi,xi]*(1-fx)+a[yi,xx]*fx)*(1-fy)+(a[yy,xi]*(1-fx)+a[yy,xx]*fx)*fy
def resize(a,size):return np.asarray(Image.fromarray(a.astype(np.float32)).resize(size,Image.Resampling.BILINEAR))
# Integer upscales (tile grid -> full-resolution DDS) in column bands on the thread pool (Pillow
# releases the GIL): a band whose edges are multiples of the scale has box offsets and scale exact
# in double, so it reads the same source taps with the same weights as the whole resize (the
# coefficients are clipped to the full source). Column bands only: row bands do not match. A
# small upscale per filter/mode/scale is compared with the whole resize at first use; on any
# difference (another Pillow) the serial resize is kept. post maps each band (e.g. a dtype cast).
_BAND_OK={}
def _band_ok(resample,mode,f):
 key=(resample,mode,f);band=6*f   # 256 columns: ten bands and a short last one
 if key not in _BAND_OK:
  rng=np.random.default_rng(5);a=rng.standard_normal((63,64)).astype(np.float32)*100   # odd height, as the 4095-row tile grid
  im=Image.fromarray({'F':a,'I':a.astype(np.int32),'L':(a>0).astype(np.uint8)}[mode]);W_,H_=64*f,63*f;whole=np.asarray(im.resize((W_,H_),resample))
  bands=[np.asarray(im.resize((min(c+band,W_)-c,H_),resample,box=(c/f,0,min(c+band,W_)/f,63))) for c in range(0,W_,band)]
  _BAND_OK[key]=whole.dtype==bands[0].dtype and np.array_equal(whole.view(np.uint8),np.ascontiguousarray(np.concatenate(bands,axis=1)).view(np.uint8))
  if not _BAND_OK[key]:print('[polish] column-band resize differs from the whole resize here; serial resize kept',key,flush=True)
 return _BAND_OK[key]
def _resize_bands(a,size,resample,c0=0,c1=None,post=None,band=512):
 """post(np.asarray(Image.fromarray(a).resize(size,resample)))[:,c0:c1] for an integer upscale."""
 h,w=a.shape;W_,H_=size;f=W_//w;c1=W_ if c1 is None else c1;post=post or (lambda b:b);im=Image.fromarray(a)
 if not (im.mode in ('F','I','L') and f>1 and W_==f*w and H_==f*h and c0%f==0 and (c1%f==0 or c1==W_) and band%f==0 and _band_ok(resample,im.mode,f)):
  return post(np.asarray(im.resize(size,resample)))[:,c0:c1]
 def piece(b0):b1=min(b0+band,c1);return post(np.asarray(im.resize((b1-b0,H_),resample,box=(b0/f,0,b1/f,h))))
 first=piece(c0);res=np.empty((H_,c1-c0),first.dtype);res[:,:first.shape[1]]=first;del first
 def job(b0):res[:,b0-c0:min(b0+band,c1)-c0]=piece(b0)
 list(_POOL.map(job,range(c0+band,c1,band)))
 return res
def resize_high(a,size):return _resize_bands(a.astype(np.float32),size,Image.Resampling.BILINEAR)   # = resize(a,size)
def _rows(fn,n,step=256):
 """fn(row slice) for blocks of rows on the thread pool: elementwise work, same bytes as one call."""
 list(_POOL.map(lambda r0:fn(slice(r0,min(r0+step,n))),range(0,n,step)))
def blur(a,radius):return np.asarray(Image.fromarray(np.clip(a*255,0,255).astype('uint8')).filter(ImageFilter.GaussianBlur(radius)),dtype=np.float32)/255
# Box-blur radius (LF pixels / BASE_FIELD_STEP) of the continuous base outside the native map.
BASE_FIELD_STEP,BASE_FIELD_SMOOTH=4,16
def continuous_canvas_base(old):
 """Pre-polish LF (north-first rows) with a smooth field outside the native map.

 The base outside the native map holds only the old canvases' edge-mode fills
 (19798 land, 16298 sea), whose hex-stepped boundaries run across today's land.
 The polish adds its edit as a delta on this base, and the resample of that delta
 keeps the sub-tile step: thin straight walls (x=14235 Outer Manchuria, x=10043
 Indochina) and hex-stepped terraces on Kyushu. The exterior becomes the native
 map extended edge-wise and blurred; native pixels are returned unchanged.
 """
 h,w=old.shape;ox,oy=_frame.NATIVE_OFFSET_LF_FILE;nw,nh=_frame.NATIVE_LF_W,_frame.NATIVE_LF_H;f=BASE_FIELD_STEP
 if (h,w)==(nh,nw):return old
 assert h%f==0 and w%f==0 and ox%f==0 and oy%f==0 and nw%f==0 and nh%f==0
 native=old[oy:oy+nh,ox:ox+nw].astype(np.float32).reshape(nh//f,f,nw//f,f).mean(axis=(1,3))
 iy=np.clip(np.arange(h//f)-oy//f,0,nh//f-1);ix=np.clip(np.arange(w//f)-ox//f,0,nw//f-1)
 field=box_blur(box_blur(np.ascontiguousarray(native[iy][:,ix]),BASE_FIELD_SMOOTH),BASE_FIELD_SMOOTH)
 out=np.rint(np.clip(resize(field,(w,h)),0,65535)).astype('<u2');out[oy:oy+nh,ox:ox+nw]=old[oy:oy+nh,ox:ox+nw]
 return out

def min_plus_envelope(values,step_x,step_z):
 """Separable minimum seed height plus weighted grid distance."""
 out=np.asarray(values,dtype=np.float32).copy()
 for axis,step in ((1,step_x),(0,step_z)):
  shape=[1,1];shape[axis]=out.shape[axis]
  coordinate=np.arange(out.shape[axis],dtype=np.float32).reshape(shape)*step
  forward=np.minimum.accumulate(out-coordinate,axis=axis)+coordinate
  backward=np.flip(np.minimum.accumulate(np.flip(out+coordinate,axis=axis),axis=axis),axis=axis)-coordinate
  out=np.minimum(forward,backward)
 return out

# Strips are independent within an axis pass; numpy releases the GIL in these
# loops, so a thread pool runs them in parallel (identical results). The
# cumulative edge-cost coordinates depend only on the original heights and are
# cached across sweeps and envelope calls (memory traded for speed).
from concurrent.futures import ThreadPoolExecutor
_POOL=ThreadPoolExecutor(max_workers=max(1,(os.cpu_count() or 4)))
# Commit of one material-blend row-block thread's temporaries (256 rows x 16384 pixels), measured;
# the row loop runs on memory_budget.pool_size threads.
import memory_budget
BLEND_THREAD_GB=.55
_COORDINATES={}

def _strips(shape,axis,strip):
 count=shape[0] if axis==1 else shape[1]
 return [(slice(start,min(start+strip,count)),slice(None)) if axis==1 else (slice(None),slice(start,min(start+strip,count))) for start in range(0,count,strip)]

def _edge_coordinates(oldheight,step_x,step_z,strip):
 key=(id(oldheight),oldheight.shape,float(step_x),float(step_z),strip)
 if key not in _COORDINATES:
  _COORDINATES.clear()
  def build(axis,step,selection):
   costs=np.maximum(step,np.abs(np.diff(oldheight[selection],axis=axis)));zero_shape=list(costs.shape);zero_shape[axis]=1
   return np.concatenate((np.zeros(zero_shape,np.float64),np.cumsum(costs,dtype=np.float64,axis=axis)),axis=axis)
  _COORDINATES[key]={axis:list(_POOL.map(lambda sel,axis=axis,step=step:build(axis,step,sel),_strips(oldheight.shape,axis,strip))) for axis,step in ((1,step_x),(0,step_z))}
 return _COORDINATES[key]

def edge_cost_envelope(values,oldheight,step_x,step_z,max_sweeps=12,strip=64):
 """Graph envelope; cumulative coordinates cached per strip, strips in parallel.

 A strip whose values did not change since its last pass on an axis is
 skipped (the 1-D envelope is idempotent): later sweeps touch only the
 strips that the other axis changed. Results are identical."""
 out=np.array(values,dtype=np.float32,copy=True)
 coordinates=_edge_coordinates(oldheight,step_x,step_z,strip)
 strips={axis:_strips(out.shape,axis,strip) for axis in (1,0)}
 dirty={axis:np.ones(len(strips[axis]),bool) for axis in (1,0)}
 def update(axis,i):
  selection=strips[axis][i];coordinate=coordinates[axis][i]
  target=out[selection];value=np.asarray(target,dtype=np.float64)
  forward=np.minimum.accumulate(value-coordinate,axis=axis)+coordinate;backward=np.flip(np.minimum.accumulate(np.flip(value+coordinate,axis=axis),axis=axis),axis=axis)-coordinate;updated=np.minimum(forward,backward).astype(np.float32)
  if np.any(~np.isfinite(target)&np.isfinite(updated)):improvement=float('inf')
  elif np.isfinite(target).all():improvement=float(np.max(target-updated))
  else:improvement=0.
  changed=np.flatnonzero((updated!=target).any(axis=1-axis))   # indices along the other axis
  target[:]=updated
  return improvement,changed
 for sweep in range(max_sweeps):
  improvement=0.
  for axis in (1,0):
   todo=np.flatnonzero(dirty[axis]);dirty[axis][:]=False
   for imp,changed in _POOL.map(lambda i,axis=axis:update(axis,i),todo):
    improvement=max(improvement,imp)
    dirty[1-axis][np.unique(changed//strip)]=True
  if improvement<=1e-7:break
 return out,sweep+1

def maximum_edge_constraint_error(surface,oldheight,step_x,step_z,strip=64):
 result={}
 for axis,step,key in ((1,step_x,'x'),(0,step_z,'z')):
  def excess(selection,axis=axis,step=step):
   allowed=np.maximum(step,np.abs(np.diff(oldheight[selection],axis=axis)));actual=np.abs(np.diff(surface[selection],axis=axis));return float(np.max(actual-allowed))
  result[key]=max(_POOL.map(excess,_strips(surface.shape,axis,strip)))
 return result

def constrained_town_surface(oldheight,proposed,owners,native_plots,requested_alpha,grade=1.0,pixel_pitch=None,plateau_targets=None,smooth=0,group_names=None,env_future=None,floors=None,ceils=None):
 """Run the projection with a shared-memory worker pool (envelope_mp.py) for its envelopes."""
 import envelope_mp
 _tpi=_time.time()
 pitch_x,pitch_z=pixel_pitch or (DX/2,DX*ZS/2)
 # The pool and edge-cost coordinates may already be under construction (env_future,
 # started as soon as the original heights exist; same arguments).
 env=env_future.result() if env_future is not None else envelope_mp.Envelope(oldheight,grade*pitch_x,grade*pitch_z)
 print(f'[projection-timing] envelope pool + coordinates: {_time.time()-_tpi:.1f}s',flush=True)
 try:return _constrained_town_surface(oldheight,proposed,owners,native_plots,requested_alpha,grade,pixel_pitch,plateau_targets,smooth,group_names,env,floors,ceils)
 finally:env.close()

def _constrained_town_surface(oldheight,proposed,owners,native_plots,requested_alpha,grade=1.0,pixel_pitch=None,plateau_targets=None,smooth=0,group_names=None,env=None,floors=None,ceils=None):
 group_names=group_names or {};_t_in=_time.time()
 # The near-town weight depends only on the plot owners. It is built after the first projection:
 # beside the bound or projection envelopes its 1 GB result and blur temporaries set the solver's
 # commit peak (mem-budget; it took ~4 s of waiting there anyway).
 _near_weight=lambda:np.clip(box_blur(box_blur((owners>=0).astype(np.float32),smooth*6),smooth*6)*4,0,1)
 """Preserve original samples outside edits; project new slopes locally."""
 pitch_x,pitch_z=pixel_pitch or (DX/2,DX*ZS/2);step_x,step_z=grade*pitch_x,grade*pitch_z
 # requested_alpha None: native_plots is already the fixed mask. proposed may come in a one-element list,
 # taken out here so that the projection below can release it.
 if isinstance(proposed,list):proposed=proposed.pop()
 fixed=native_plots if requested_alpha is None else native_plots|(requested_alpha<=1e-7);flat=owners>=0;flat_indices=np.flatnonzero(flat.ravel());flat_groups=owners.ravel()[flat_indices];groups=np.unique(flat_groups);group_indices={int(g):flat_indices[flat_groups==g] for g in groups};sweeps=[]
 _tp=[_time.time()];print(f'[projection-timing] masks + group indices: {_tp[0]-_t_in:.1f}s',flush=True)
 # Beyond this many pixels from its finite seeds an envelope value exceeds the
 # whole height range in use, so it can never bind: those cells are skipped
 # (envelope_mp.Envelope.run reach), every binding value stays identical.
 span=float(max(np.nanmax(oldheight),np.nanmax(proposed))-min(np.nanmin(oldheight),np.nanmin(proposed)))+1
 reach=int(np.ceil(span/min(step_x,step_z)))+2
 def envelope(values):
  t0=_time.time()
  result,count=env.run(values,reach=reach) if env is not None else edge_cost_envelope(values,oldheight,step_x,step_z);sweeps.append(count)
  print(f'[projection-timing] envelope {len(sweeps)}: {_time.time()-t0:.1f}s, {count} sweeps',flush=True);return result
 # Memory (mem-budget): the full-resolution bounds below are built and combined in place,
 # with the same float32 operations in the same operand order (identical bytes); each
 # temporary is one 1 GB array fewer at the projection's commit peak.
 def neg_envelope(values):   # -envelope(values), negated in place
  result=envelope(values);np.negative(result,out=result);return result
 def seeded(values,mask,negate=False):   # np.where(mask,-values if negate else values,np.inf), float32
  out=np.negative(values) if negate else np.array(values,dtype=np.float32,copy=True);np.copyto(out,np.float32(np.inf),where=~mask);return out
 def projection(values):   # .5*(-envelope(-values)+envelope(values))
  out=neg_envelope(-values);np.add(out,envelope(values),out=out);np.multiply(.5,out,out=out);return out
 lower_fixed=neg_envelope(seeded(oldheight,fixed,True));upper_fixed=envelope(seeded(oldheight,fixed))
 # HEIGHT BAKE lower bound L (floors: a value per pixel, -inf where none; in a one-element list, released here): the
 # lower envelope of the floors joins the fixed lower bound (height-bake-design 2)
 if floors is not None:
  _fl=floors.pop() if isinstance(floors,list) else floors
  if np.isfinite(_fl).any():
   _lfl=neg_envelope(seeded(_fl,np.isfinite(_fl),True));np.maximum(lower_fixed,_lfl,out=lower_fixed);del _lfl
  del _fl
 if np.any(lower_fixed>upper_fixed+.001):raise RuntimeError('Original fixed terrain anchors are incompatible')
 intervals={};levels={};infeasible=[]
 for group,indices in group_indices.items():
  lo=float(lower_fixed.ravel()[indices].max());hi=float(upper_fixed.ravel()[indices].min())
  if lo>hi+.001:infeasible.append((group,lo,hi));continue
  intervals[group]=(lo,hi);levels[group]=float(np.clip((plateau_targets or {}).get(group,LEVEL),lo,hi))
 if infeasible:
  # Report every plot that cannot be levelled between the fixed (vanilla) slopes at once.
  raise RuntimeError('Authored plot groups cannot fit fixed terrain: '+'; '.join(f'group {g} [{", ".join(group_names.get(g,[]))}] needs >= {lo:.3f} and <= {hi:.3f}' for g,lo,hi in infeasible))
 conflicts={}
 for attempt in range(len(groups)+1):
  pins=lower=None   # the previous attempt's arrays, released before the new ones exist
  pins=np.full(oldheight.shape,np.inf,np.float32)
  for group,height in levels.items():pins.ravel()[group_indices[group]]=-height
  lower=neg_envelope(pins);changed=False
  for group,height in tuple(levels.items()):
   required=float(lower.ravel()[group_indices[group]].max());hi=intervals[group][1]
   if required>hi+.001:conflicts[group]=(required,hi)
   if required>height+.0001 and levels[group]<hi-1e-6:levels[group]=min(required,hi);changed=True
  if not changed:break
 else:raise RuntimeError('Authored plateau constraints did not converge')
 if conflicts:
  # A neighbouring plateau pulls this plot above what its fixed (vanilla) slopes allow.
  raise RuntimeError('Authored plot groups have incompatible fixed/town anchors: '+'; '.join(f'group {g} [{", ".join(group_names.get(g,[]))}] pulled to {r:.3f} > {h:.3f}' for g,(r,h) in conflicts.items()))
 pins=lower=None
 pins=np.full(oldheight.shape,np.inf,np.float32)
 for group,height in levels.items():pins.ravel()[group_indices[group]]=height
 # lower=np.maximum(lower_fixed,-envelope(np.where(flat,-pins,np.inf))), upper=np.minimum(upper_fixed,envelope(pins)), in place
 # (upper first: pins then becomes the lower seed in place, -pins where flat, else inf)
 upper=np.minimum(upper_fixed,envelope(pins),out=upper_fixed);np.negative(pins,out=pins);np.copyto(pins,np.float32(np.inf),where=~flat)
 lower=np.maximum(lower_fixed,neg_envelope(pins),out=lower_fixed);del pins,flat,lower_fixed,upper_fixed
 if np.any(lower>upper+.001):raise RuntimeError('Combined fixed terrain and plateau bounds are incompatible')
 # HEIGHT BAKE upper bound U (ceils: a value per pixel, +inf where none; one-element list, released here): the upper
 # envelope of the ceilings joins the upper bound; where the fixed or plateau lower bound is higher, that wins (the
 # ceiling gives way, never an infeasible plot): height-bake-design 2
 if ceils is not None:
  _cl=ceils.pop() if isinstance(ceils,list) else ceils
  if np.isfinite(_cl).any():
   _ucl=envelope(seeded(_cl,np.isfinite(_cl)));np.minimum(upper,_ucl,out=upper);del _ucl;np.maximum(upper,lower,out=upper)
  del _cl
 projected=projection(proposed);del proposed
 _near_future=_POOL.submit(_near_weight) if smooth else None
 # Smoothing a slope-bounded target keeps it slope-bounded; the clamp below then
 # rounds the square four-neighbour aprons that read as boxes around town plots.
 if smooth:
  # The bound is per edge (original slopes add allowances), so re-project after
  # blurring; the projection keeps most of the rounding.
  # Only around town plots: smoothing everywhere let the old border's large slope
  # allowances re-form steps along the native edge.
  _tn=_time.time();near_town=_near_future.result()
  print(f'[projection-timing] near-town weight: {_time.time()-_tn:.1f}s',flush=True)
  # projected=projected*(1-near_town)+box_blur(box_blur(projected,smooth),smooth)*near_town, in place
  _blurred=box_blur(box_blur(projected,smooth),smooth);np.multiply(_blurred,near_town,out=_blurred)
  _kept=np.subtract(1,near_town);del near_town;_near_future=None;np.multiply(projected,_kept,out=_kept);np.add(_kept,_blurred,out=_kept);del _blurred
  projected=None;projected=projection(_kept);del _kept
 # surface=np.minimum(np.maximum(projected,lower),upper), in place
 surface=np.minimum(np.maximum(projected,lower,out=projected),upper,out=projected);del projected,lower,upper;surface[fixed]=oldheight[fixed]
 for group,height in levels.items():surface.ravel()[group_indices[group]]=height
 t0=_time.time();errors=maximum_edge_constraint_error(surface,oldheight,step_x,step_z);excess_x,excess_z=errors['x'],errors['z'];print(f'[projection-timing] edge constraint check: {_time.time()-t0:.1f}s',flush=True)
 if max(excess_x,excess_z)>1e-4:raise RuntimeError(f'Terrain slope projection did not converge: X excess={excess_x}, Z excess={excess_z}')
 if not np.array_equal(surface[fixed],oldheight[fixed]):raise RuntimeError('Terrain projection changed original fixed samples')
 for group,height in levels.items():
  if float(np.max(np.abs(surface.ravel()[group_indices[group]]-height)))>1e-5:raise RuntimeError(f'Terrain projection changed flat plateau {group}')
 return surface,levels,{'approach_grade':grade,'distance_metric':'four-neighbour terrain graph with original edge slope allowances','overlapping_apron_groups':len(groups),'constraint_iterations':attempt+1,'maximum_envelope_sweeps':max(sweeps),'maximum_edge_constraint_error':{'x':excess_x,'z':excess_z},'original_fixed_samples':int(fixed.sum()),'original_fixed_samples_preserved':True,'adapted_plateau_groups':sum(abs(height-LEVEL)>.001 for height in levels.values())}
m=read_hex(A/'work/final-map/map.hex');b=(A/'work/final-map/map.hex').read_bytes();cells=np.frombuffer(b[-4-W*H*16:-4],np.uint8).reshape(H,W,16);ids=(cells[:,:,1].astype(int)<<5)+(cells[:,:,0].astype(int)>>3)-1
plan=json.loads((A/'work/current-region-design.json').read_text(encoding='utf-8'));names=m['land_regions']+m['sea_regions'];newkeys={r['region'] for r in plan['regions'] if r.get('introduced_in_bulk')};new=np.isin(ids,[names.index(k) for k in newkeys]);land=(cells[:,:,0]&3)!=1
# Tile coordinates run south-to-north. Use the same continuous surface everywhere.
y,x=np.mgrid[:TH,:TW].astype(np.float32);q=x/2;r=y/2
hx,hy=_frame.tile_hex(x.astype(int),y.astype(int))
continuous=_frame.load_canvas_array(A/'analysis/continuous-land-mask.npy') if (A/'analysis/continuous-land-mask.npy').exists() else np.zeros((H,W),bool)
physical=land[hy,hx];bulk=new[hy,hx]|continuous[hy,hx];outer=(x<_frame.NATIVE_TILE_X0)|(x>=_frame.NATIVE_TILE_X1)|(y<_frame.NATIVE_TILE_Y0)|(y>=_frame.NATIVE_TILE_Y1)
# Tile cells whose vanilla native terrain the dry-land floor and the town footprint leave unchanged.
# One place for that gate: the whole native rectangle for now (revealed filler hexes may later count as canvas).
native_protected=~outer
if not CACHE_ONLY:
 marine_keys={c['region'] for g in plan['provinces'] if g.get('reachability_mode')=='naval' for c in g['regions']}
 marine_hex=np.isin(ids,[names.index(k) for k in marine_keys]);marine=marine_hex[hy,hx]
# Native continental relief is already authored. Keep it, and terrace only
# new town approaches; outer canvas receives the new geographical relief.
def _town_centre(tq,tr):
 # Town masks are measured from where the engine draws the town (display_position), in the
 # q=x/2, r=y/2 units below; q*DX (the tile block's SW corner) is up to a hex off it.
 x,z=_frame.display_position(tq,tr);return x/DX,z/_frame.DZ
def _town_window(tq,tr,reach):
 # Tile-grid window holding every pixel within `reach` hexes of a town (q=x/2, r=y/2); the
 # town masks below are exactly zero beyond their reach, so windowing changes nothing.
 cx,cy=_town_centre(tq,tr);cx,cy=2*cx,2*cy;R=2*reach+2
 return slice(max(0,int(cy-R)),min(TH,int(cy+R)+1)),slice(max(0,int(cx-R)),min(TW,int(cx+R)+1))
if not CACHE_ONLY:
 clearance=np.zeros((TH,TW),np.float32)
 for town in plan['regions']:
  tq,tr=town['town']
  inner,outer_town=(8,20) if town['is_capital'] else (5,13)
  win=_town_window(tq,tr,outer_town);sq,sr=_town_centre(tq,tr);d=np.hypot(q[win]-sq,r[win]-sr)
  t=np.clip((d-inner)/(outer_town-inner),0,1)
  clearance[win]=np.maximum(clearance[win],1-t*t*(3-2*t))
_mark('global: town/marine edit masks')
if not CACHE_ONLY:
 alpha=np.maximum(clearance,blur(marine.astype(float),4));alpha[marine|outer]=1
# LAYOUT CLEANUP (layout_cleanup.py): the land of the pre-layout-v2 map and the coast-revision-4 change mask, as
# tracked data (data/map-layout/layout-cleanup.npz, extract_layout_cleanup.py). The public source snapshot leaves it
# out (old project layouts, source compliance): then the cleanup is all 0 with a WARN; present, the mask is unchanged.
import layout_cleanup as _lc
layout_cleanup=_lc.mask((TH,TW),land,hy,hx,plan,q,r,blur,_town_window,_town_centre)
# History cleanup (older layouts' towns/coasts) belongs to the canvas only. Inside
# the native map the baseline is vanilla: keep it only where land/sea now
# differs from vanilla (new islands); otherwise old-layout differences would
# repaint restored vanilla ground.
import map_hex as _mh
_vm=_mh.read(_frame.VANILLA_MAP);_vland=np.zeros((H,W),bool);_vland[_frame.NATIVE_HEX_SLICE]=(_vm['cells'][:,:,0]&3)!=1
_native_hex=np.zeros((H,W),bool);_native_hex[_frame.NATIVE_HEX_SLICE]=True
_native_differs=_native_hex&(land!=_vland)
_keep=outer|(blur(_native_differs[hy,hx].astype(float),3)>.02)
layout_cleanup=np.where(_keep,layout_cleanup,0.).astype(np.float32)
# Former town terraces and retired island tips receive geographic relief,
# rather than keeping the old flat city pads above the new water surface.
if not CACHE_ONLY:alpha=np.maximum(alpha,layout_cleanup)
# Blend three hexes into the original perimeter instead of a vertical seam.
edge=np.minimum.reduce((x-_frame.NATIVE_TILE_X0,_frame.NATIVE_TILE_X1-1-x,y-_frame.NATIVE_TILE_Y0,_frame.NATIVE_TILE_Y1-1-y))
# The outer native cells were the vanilla world edge (rim slope, truncated ridges).
# Blend a meandering ring of them into the canvas transition instead of a straight 8-cell strip.
if not CACHE_ONLY:
 ring_meander=EDGE_RING_MEANDER*np.clip(.5+.6*gpu_noise.fbm(q,r,(NQ,NR),25,3,seed=5),0,1)
 ring_t=np.clip(1-(edge-ring_meander)/EDGE_RING,0,1);alpha=np.maximum(alpha,ring_t*ring_t*(3-2*ring_t))
authored_ids={names.index(c['region']) for c in plan['regions']};native_town_hex=(((cells[:,:,2]>>4)>0)|((cells[:,:,3]&1)>0))&~np.isin(ids,list(authored_ids))
# Sea-region ownership does not identify a vanilla port: imported ports also
# have sprawl in sea regions. Build the guard from the actual vanilla map.
original_map=_frame.VANILLA_MAP;original_meta=read_hex(original_map)   # pure vanilla (original-map carries an early mod region)
assert (original_meta['width'],original_meta['height'])==(_frame.NATIVE_W,_frame.NATIVE_H)
original_bytes=original_map.read_bytes()
original_cells=np.frombuffer(original_bytes[-4-_frame.NATIVE_W*_frame.NATIVE_H*16:-4],np.uint8).reshape(_frame.NATIVE_H,_frame.NATIVE_W,16)
native_land_hex=np.zeros((H,W),bool)
native_land_hex[_frame.NATIVE_HEX_SLICE]=(original_cells[:,:,0]&3)!=1
# Frontier river hexes inside the native rectangle (apply_rivers lays them on the revealed filler): vanilla land
# whose river water stops at the native border (river level, beds and water meshes are canvas only). Their
# relief stays land relief (coast, carried native heights): as open water they took the coastal falloff and the
# projection cut a dry gorge round them into the native plateau, a wall against the canvas side (integ 458ea872).
_rvn=np.zeros((H,W),bool);_rvn[_frame.NATIVE_HEX_SLICE]=True
native_river=(_rvn&native_land_hex&~land&np.isin(ids,[i for i,k in enumerate(names) if k.startswith('tw3k_riv_sea_')]))[hy,hx];del _rvn
native_town_hex=np.zeros((H,W),bool)
native_town_hex[_frame.NATIVE_HEX_SLICE]=((original_cells[:,:,2]>>4)>0)|((original_cells[:,:,3]&1)>0)
if not CACHE_ONLY:
 native_guard=fastops.max_filter_mask(native_town_hex,17)
# Vanilla playable regions keep their vanilla heights: a hex still owned by the
# same vanilla playable region (land or sea) is never re-shaped, except in the
# native edge ring that must blend into the canvas. New-town clearance, ramps,
# aprons and marine blurs otherwise reached 13-20 hexes into neighbouring
# vanilla regions.
_vnames=original_meta['land_regions']+original_meta['sea_regions']
_vids=(original_cells[:,:,1].astype(int)<<5)+(original_cells[:,:,0].astype(int)>>3)-1
# Retired regions (seas merged into a neighbour) count as their surviving region, so a vanilla sea
# cell keeps its protection after the merge; with nothing retired this is the plain name match.
_retired=plan.get('retired_regions',{});_fidx={n:i for i,n in enumerate(names)}
_vanilla_to_final=np.array([_fidx.get(_retired.get(n,n),-2) for n in _vnames]+[-3])
_fids=ids[_frame.NATIVE_HEX_SLICE];_same=(_vanilla_to_final[np.where(_vids>=0,_vids,len(_vnames))]==_fids)&(_vids>=0)
_same&=np.array([n!='3k_main_reg_non_playable' for n in _vnames]+[False])[np.where(_vids>=0,_vids,len(_vnames))]
_nq,_nr=np.meshgrid(np.arange(_frame.NATIVE_W),np.arange(_frame.NATIVE_H))
# Vanilla playable land is exempt from protection only within the original
# 10-hex ring; the wider ring meander (a straight smoothed band read as a cliff
# line on the 2048 canvas) reaches further into non-playable land only.
NATIVE_RIM_HEX=10
# The protected line meanders between 6 and 10 hexes in (never further in than
# the original ring): a straight line at exactly 10 hexes left a north-south
# height crease at hex column 668 (offline render, 2026-10-20).
_rim=6+(NATIVE_RIM_HEX-6)*np.clip(.5+.7*gpu_noise.fbm(_nq+_frame.PAD_LEFT,_nr+_frame.PAD_BOTTOM,(NQ,NR),9,3,seed=89),0,1)
_same&=np.minimum.reduce([_nq,_nr,_frame.NATIVE_W-1-_nq,_frame.NATIVE_H-1-_nr])>_rim
# --- L2048 MOD-COAST SEA (sea fork) --- vanilla's decorative non-playable coasts carry land LF under
# sea hexes; where our coast moved (drowned vanilla land, authored land/seas within 8 hexes) a protected
# or half-weighted vanilla sea kept that land: Lin Yi's detached sand bar, the Kimhae wedge (live 2048l).
_vland_full=np.zeros((H,W),bool);_vland_full[_frame.NATIVE_HEX_SLICE]=(original_cells[:,:,0]&3)!=1
_vnp=np.zeros((H,W),bool);_vnp[_frame.NATIVE_HEX_SLICE]=np.isin(_vids,[i for i,n in enumerate(_vnames) if n in('3k_main_reg_non_playable','3k_main_sea_non_playable')])
# --- NONPLAY GROUND --- vanilla's non-playable filler under shroud (cold_1 sheets, round brush stamps) that is now
# playable mod land takes canvas ground (real-terrain climate and materials, the donor; not the relief). Weight
# 0 on and next to vanilla playable land, full NONPLAY_RAMP hexes away (user approval 2026-10-25).
import revealed_filler as _rf;nonplay_hex=_rf.filler_weight(cells,names,NONPLAY_RAMP) if NONPLAY_GROUND else np.zeros((H,W),np.float32)   # all vanilla non-playable filler now land (scripts/revealed_filler.py; user 2026-10-25)
mod_coast_sea_hex=_vnp&~land&fastops.max_filter_mask(np.isin(ids,list(authored_ids))|(_vland_full&~land),17)
_same&=~mod_coast_sea_hex[_frame.NATIVE_HEX_SLICE]
# --- end L2048 MOD-COAST SEA ---
vanilla_protected_hex=np.zeros((H,W),bool);vanilla_protected_hex[_frame.NATIVE_HEX_SLICE]=_same
if not CACHE_ONLY:
 guard=blur(native_guard[hy,hx].astype(float),3);alpha*=np.clip((1-guard-.1)/.8,0,1)
 # Flat town aprons have adaptive elevations; native building plots stay fixed.
 authored_plot=(((cells[:,:,2]>>4)>0)|((cells[:,:,3]&1)>0))&np.isin(ids,list(authored_ids))
 native_plot_apron=fastops.max_filter_mask(native_town_hex,5)
_mark('global: layout cleanup + native guards')
if not CACHE_ONLY:
 parent=list(range(len(plan['regions'])))
 def find_group(i):
  while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
  return i
 flat_owner_hex=np.full((H,W),-1,np.int32)
 # Authored plot hexes grouped by region id, row-major within a region (np.nonzero order of its
 # plot mask); each town's plot is rebuilt in its window instead of a full-map ids== per town.
 _pc=np.flatnonzero(authored_plot);_pid=ids.ravel()[_pc];_po=np.argsort(_pid,kind='stable');_pc,_pid=_pc[_po],_pid[_po];del _po
 # Land hexes next to a frontier river hex are not flat apron (plot hexes always are). apply_rivers routes
 # the rivers round the towns, so a river can skirt the 2-hex apron; the plateau (often raised to
 # MIN_TOWN_HEIGHT over a delta) then bounded the channel from below by plateau - TOWN_GRADE*distance
 # after RIVER CHANNELS had carved it: 27 river hex centres beside an apron dry (15% vs 1% elsewhere,
 # Ganges 691,546 at 0.98 = plateau 1.31 - 4 px * DX/8, level 0.32; v16 regression run 2026-10-24).
 from hex_components import hex_dilate as _hex_dilate
 _river_rim=_hex_dilate(np.isin(ids,[i for i,k in enumerate(names) if k.startswith('tw3k_riv_sea_')])&~land)&~authored_plot
 for i,town in enumerate(plan['regions']):
  _k=names.index(town['region']);rr,qq=np.divmod(_pc[np.searchsorted(_pid,_k):np.searchsorted(_pid,_k,'right')],W)
  if not len(rr):raise RuntimeError('Authored town has no building plot: '+town['region'])
  x0,x1=max(0,int(qq.min())-2),min(W,int(qq.max())+3);y0,y1=max(0,int(rr.min())-2),min(H,int(rr.max())+3)
  region_plot=np.zeros((y1-y0,x1-x0),bool);region_plot[rr-y0,qq-x0]=True   # = (authored_plot&(ids==_k))[y0:y1,x0:x1]
  apron=fastops.max_filter_mask(region_plot,5)
  apron&=land[y0:y1,x0:x1]&~native_plot_apron[y0:y1,x0:x1]&~vanilla_protected_hex[y0:y1,x0:x1]&~_river_rim[y0:y1,x0:x1]
  owner=flat_owner_hex[y0:y1,x0:x1]
  for other in np.unique(owner[apron&(owner>=0)]):parent[find_group(int(other))]=find_group(i)
  owner[apron]=i
 town_groups={town['region']:find_group(i) for i,town in enumerate(plan['regions'])}
 # Each owner index to its group root (roots map to themselves, so one table lookup = the per-index loop).
 _root=np.array([find_group(i) for i in range(len(parent))],np.int32);_m=flat_owner_hex>=0;flat_owner_hex[_m]=_root[flat_owner_hex[_m]];del _root,_m,_pc,_pid,_river_rim
 flat_owner=flat_owner_hex[hy,hx];full_town_edit=flat_owner>=0;native_plots=native_town_hex[hy,hx]
 town_distance=min_plus_envelope(np.where(full_town_edit,0,np.inf),DX/2,DX*ZS/2)
 t=np.clip(town_distance/(10*DX),0,1);town_edit_weight=1-t*t*(3-2*t)
_mark('global: town plots')
if not CACHE_ONLY:
 alpha=np.maximum(alpha,town_edit_weight);alpha[native_plots]=0
# Fade edits out over ~1.5 hexes (3 tile cells) before protected vanilla ground, then zero it.
# Edits fade to exactly zero at protected vanilla land, linearly over
# PROTECT_FADE tile cells of distance. A blurred-mask fade left 37% of the edit
# at the boundary, and the slope-limited projection turned the rest into a
# straight crease along vanilla's playable edge (hex column 668, offline render).
PROTECT_FADE=16
_protected=vanilla_protected_hex[hy,hx]
if not CACHE_ONLY:
 alpha=np.maximum(alpha,mod_coast_sea_hex[hy,hx].astype(np.float32))   # L2048 MOD-COAST SEA: full edit (coast falloff to SEA_FLOOR)
 # Ring index of 4-neighbour growth from protected ground, capped at PROTECT_FADE
 # (frontier BFS; same values as growing the mask PROTECT_FADE-1 times).
 _pd=fastops.grow_steps(_protected,PROTECT_FADE-1);_pd=np.where(_pd<PROTECT_FADE,_pd,PROTECT_FADE).astype(np.float32)
 _ramp=np.clip(_pd/PROTECT_FADE,0,1);alpha*=_ramp*_ramp*(3-2*_ramp);alpha[_protected]=0;del _pd,_ramp
 assert not np.any(full_town_edit&native_plots)
 # --- HEHUANG PATCH (hehuang_terrain.py) --- the patch is an edit: alpha=max(alpha,w) (w is 0 on protected land and plots)
 if HH is not None:alpha=HH.raise_alpha(alpha,_protected)
# Town climate overrides (TOWN_CLIMATE): mask, value and a soft weight per tile.
climate_override=np.zeros((TH,TW),bool);climate_override_value=np.zeros((TH,TW),np.uint16);climate_override_weight=np.zeros((TH,TW),np.float32)
for _town in plan['regions']:
 if _town['region'] not in TOWN_CLIMATE:continue
 _sq,_sr=_town_centre(*_town['town']);_d=np.hypot(q-_sq,r-_sr)
 _R=TOWN_CLIMATE_R[0]+(TOWN_CLIMATE_R[1]-TOWN_CLIMATE_R[0])*np.clip(.5+.7*gpu_noise.fbm(q,r,(NQ,NR),5,3,seed=87),0,1)
 _m=(_d<_R)&physical&~_protected;climate_override|=_m;climate_override_value[_m]=TOWN_CLIMATE[_town['region']]
 climate_override_weight=np.maximum(climate_override_weight,np.where(_m,np.clip((_R-_d)/2,0,1),0).astype(np.float32))
def apply_climate_override(b):
 b=b.copy();b[climate_override]=climate_override_value[climate_override];return b
# --- LAKE LEVEL --- the lakes with a water level of their own: name -> (lake-mask component, spec entry); empty off
lake_level_lakes={};lake_level_mask=None
if LAKE_LEVEL and not CACHE_ONLY and '3k_main_sea_lake' in names:
 _ll_spec,_ll_spec_sha=_lake_level.load_spec()
 _ll_tiles=(~physical)&outer&(ids[hy,hx]==names.index('3k_main_sea_lake'))
 for _ll_name,_ll_e in sorted(_ll_spec['lakes'].items()):
  _ll_seed=(int(_ll_e['seed_world_xz'][1]/(_frame.DZ/2)),int(_ll_e['seed_world_xz'][0]/(DX/2)))
  lake_level_lakes[_ll_name]=(_lake_level.find_component(_ll_tiles,_ll_seed,_ll_e['seed_search_cells'])[0],_ll_e)
 lake_level_mask=np.zeros((TH,TW),bool)
 for _ll_m,_ in lake_level_lakes.values():lake_level_mask|=_ll_m
 del _ll_tiles
if not CACHE_ONLY:
 coast=blur((physical|native_river).astype(float),3)
 # LAKE LEVEL: the coastal falloff does not pull the ground round these lakes (or their own cells) toward the sea
 # floor: they are carved from their own level in the lake bowl section, their shores keep the original relief
 if lake_level_lakes:coast=blur((physical|native_river|lake_level_mask).astype(float),3)
# Climate (climate_map.cm, fine floor tiles, blend donor materials) follows
# the authored bands alone. Per-plot climates (oasis, pasture, retired sites)
# drew hex-staircase blocks in deserts: a tile climate cannot blend, and the
# donor material switched per tile cell (live checks 2026-10-18 and canvas
# 1400: yutian, jushi, shanshan). Band edges are dithered per pixel below.
biome=band_climate(q,r,BANDS,(NQ,NR))[0]
# Real geography for the Western Regions (build_west_real_terrain.py, run with
# Anaconda: ETOPO relief, Natural Earth water, RESOLVE biomes in the towns'
# projection). real_w is its blend weight on canvas land (0 outside its box).
REAL_TERRAIN=A/'analysis/west-real-terrain.npz'
REAL_GROUND_FADE=(24,72)   # tile cells from the native map: real ground from 0 to full weight
NONPLAY_EDGE_FADE=24   # --- NONPLAY GROUND --- tile cells inside the native map: repainted filler from 0 to full weight
eco_zone=None   # --- MATERIALS ECOTONE --- tile weight where the ecotone redraw interleaved classes (set below)
sub_split=None   # --- OASIS GROUND --- (paddy, forest) shares of the subtropical donor split (set below)
real_w=np.zeros((TH,TW),np.float32);real_rel=np.full((TH,TW),LEVEL,np.float32);nonplay_w=np.zeros((TH,TW),np.float32)
real_river=np.zeros((TH,TW),np.float32);real_bio=np.full((TH,TW),255,np.uint8);real_bio_eco=real_bio   # replaced by the edited classes below when real terrain exists
real_on=np.zeros((TH,TW),bool)   # --- NONPLAY GROUND --- real-terrain gate (set below; real_w > 0 off the filler)
real_elev=np.zeros((TH,TW),np.float32)   # ETOPO metres (0 outside the real terrain)
# --- SETUP MEMO --- of this block --cache-only (sync_final_terrain_surface) reads only `biome`, and the
# block was most of that run's setup. Every run that computes it (not --loop-only: no concurrent
# writers) saves it to SETUP_MEMO under a key of everything it depends on: the bytes of every
# scripts/*.py and of REAL_TERRAIN, the authored bands and noise origin, the canvas frame,
# TW3K_MATERIALS_ECOTONE, TW3K_ARCTIC_TREELINE, TW3K_EGOBI (real_dgw writes the setup biome), TW3K_TUNDRA_BAN, TW3K_TUNDRA_SCREE, TW3K_GROUND_RECIPE (biome_profiles.ground_recipe), TW3K_TAIGA_DONOR, TW3K_TREELINE_BLEND, TW3K_TUNDRA_OASIS_OFF (its gate runs only without the memo) and the NumPy/Python versions. --cache-only takes it only on an equal key
# (else it computes the block as before). TW3K_POLISH_SETUP_MEMO overrides the path, '0' turns it off.
SETUP_MEMO=os.environ.get('TW3K_POLISH_SETUP_MEMO',str(A/'analysis/polish-setup-memo.npz'));SETUP_MEMO=None if SETUP_MEMO=='0' else Path(SETUP_MEMO)
def _setup_memo_key():
 import hashlib;h=hashlib.sha256()
 for _p in sorted(Path(__file__).resolve().parent.glob('*.py')):h.update(_p.name.encode()+b'\0'+hashlib.sha256(_p.read_bytes()).digest())
 if REAL_TERRAIN.exists():
  with open(REAL_TERRAIN,'rb') as _f:h.update(hashlib.file_digest(_f,'sha256').digest())
 # the block reads the map (filler and non-playable hexes) and the design: an edit of either must not reuse a memo
 # (single-source audit item 30, 2026-11-01)
 for _p in (A/'work/final-map/map.hex',A/'work/current-region-design.json'):h.update(hashlib.sha256(_p.read_bytes()).digest())
 h.update(repr((REAL_TERRAIN.exists(),_bp.ground_recipe(),BANDS,NQ,NR,_frame.canvas_stamp(),W,H,TW,TH,_frame.NATIVE_TILE_X0,_frame.NATIVE_TILE_X1,_frame.NATIVE_TILE_Y0,_frame.NATIVE_TILE_Y1,os.environ.get('TW3K_MATERIALS_ECOTONE','1'),_frame.arctic_treeline_on(),EGOBI,_bp.tundra_ban_on(),_bp.tundra_scree_on(),_bp.taiga_donor_on(),_bp.treeline_blend_on(),_bp.tundra_oasis_off_on(),np.__version__,sys.version)).encode())
 return h.hexdigest()
_memo_key=_setup_memo_key() if SETUP_MEMO is not None and (CACHE_ONLY or not LOOP_ONLY) else None;_memo=None
if CACHE_ONLY and _memo_key is not None and SETUP_MEMO.exists():
 try:
  with np.load(SETUP_MEMO) as _f:
   if str(_f['key'])==_memo_key and _f['biome'].shape==(TH,TW):_memo=_f['biome']
 except Exception as _e:print('Setup memo unreadable, computing the real-terrain climate:',_e,flush=True)
if _memo is not None:
 # Only biome is read below in this run; the others are not used by --cache-only.
 biome=_memo;real_gw=real_w;real_steppe=np.zeros((TH,TW),bool);real_wet=np.zeros((TH,TW),bool)
 print('Setup memo: real-terrain climate from',SETUP_MEMO,flush=True)
elif REAL_TERRAIN.exists():
 _z=np.load(REAL_TERRAIN);_x0,_x1,_y0,_y1=(int(v) for v in _z['box'])
 real_rel[_y0:_y1,_x0:_x1]=_z['relief'];real_w[_y0:_y1,_x0:_x1]=_z['weight'];real_river[_y0:_y1,_x0:_x1]=_z['river']
 real_bio[_y0:_y1,_x0:_x1]=_z['biome'];_bnames=list(_z['biome_names'])
 if 'elevation' in _z.files:real_elev[_y0:_y1,_x0:_x1]=_z['elevation']
 del _z
 nonplay_w=(nonplay_hex[hy,hx]*(real_bio!=255)*physical).astype(np.float32)   # --- NONPLAY GROUND --- tile weight (the real-terrain weight is 0 inside the native map; its biomes are not)
 real_w*=outer
 real_on=np.where(_rf.filler_hex(cells,names)[hy,hx],real_bio!=255,real_w>0) if NONPLAY_GROUND else real_w>0   # --- NONPLAY GROUND --- 'real terrain here' gate: real_w > 0 off the filler (as before), the real biome on it
 # desert/steppe/grass and barren high ground (alpine/tundra/snow) arid, conifer/taiga
 # cold, forest/temperate temperate. Cold for all high ground painted the whole
 # Tian Shan and Kunlun one dark green (offline render 2048bc).
 # biome_profiles.BIOMES climate ('other' has none: it keeps the band climate, OTHER_DEFAULT).
 _clim=np.full(256,255,np.uint16)
 for _n,_c in _bp.biome_map('polish_climate','climate').items():   # 0 sub_tropical (tropics, whole-canvas real terrain)
  if _n in _bnames:_clim[_bnames.index(_n)]=_c
 # Ground (biome, steppe, wetland) fades in away from the native map: at full
 # weight right outside it the real biomes cut the native ground exactly on the
 # border (audit_native_edges 'cut', west edge). Meandered so the fade has no line.
 _dn=np.hypot(np.maximum(np.maximum(_frame.NATIVE_TILE_X0-x,x-_frame.NATIVE_TILE_X1+1),0),np.maximum(np.maximum(_frame.NATIVE_TILE_Y0-y,y-_frame.NATIVE_TILE_Y1+1),0))
 _mz=1+.45*gpu_noise.fbm(q,r,(NQ,NR),18,3,seed=93);_dn=_dn*_mz
 # --- NONPLAY GROUND --- the repainted filler fades in from the native border (NONPLAY_EDGE_FADE tile cells, the
 # same meander) as the canvas ground fades in outside it: at full weight on the border line it cut the canvas
 # side's carried ground there (audit_native_edges 'cut': 46 segments on all four edges, integ 458ea872).
 nonplay_w=(nonplay_w*np.clip(np.maximum(edge,0)*_mz/NONPLAY_EDGE_FADE,0,1)).astype(np.float32);del _mz
 real_gw=np.maximum(real_w*np.clip((_dn-REAL_GROUND_FADE[0])/(REAL_GROUND_FADE[1]-REAL_GROUND_FADE[0]),0,1),nonplay_w).astype(np.float32)   # --- NONPLAY GROUND ---
 # [EGOBI] real desert takes its climate from the shorter REAL_DESERT_FADE (the carried cold climate drew the cold donor there)
 real_dgw=(real_w*np.clip((_dn-REAL_DESERT_FADE[0])/(REAL_DESERT_FADE[1]-REAL_DESERT_FADE[0]),0,1)*np.isin(real_bio,[_bnames.index(n) for n in _bp.biome_set('real_desert') if n in _bnames])).astype(np.float32) if EGOBI else np.zeros_like(real_gw);del _dn
 # --- USR3 ECOTONE PATCH --- RESOLVE ecoregion polygons cut steppe against taiga along
 # coarse vector outlines whose finger lobes run parallel to the ranges: pale/dark bands
 # with hard edges south of Baikal (Dingling, live 2048k). Within ECOTONE cells of a
 # steppe/forest boundary the class is redrawn from the blurred steppe share, noise and
 # relief: forest on ridges and north-facing slopes, steppe in basins and on sunny slopes.
 # [fix-batch] generalised to every open-ground/forest pair, tropical too (Funan grass against
 # tropical forest, Lin Yi, Shan): a flipped cell takes the dominant class of the other side
 # within 2*ECOTONE cells (the usr3 version turned open ground into taiga, cold, in the tropics).
 _eb=real_bio.copy();_ix=lambda ns:[_bnames.index(n) for n in ns if n in _bnames]
 _eopen=_bp.biome_set('ecotone_open');_efor=_bp.biome_set('ecotone_forest')   # (steppe,grass,savanna) / (taiga,conifer,forest,temperate,tropical,tropical_dry); tundra<->forest is ecotone_bands.tundra_forest (§9.3; the ridge rule is backwards at a cold treeline)
 if not _frame.arctic_treeline_on():_eopen=_eopen+('tundra',)   # switch off (TW3K_ARCTIC_TREELINE): tundra on the open side as before (same order, same bytes)
 _est=np.isin(real_bio,_ix(_eopen));_efo=np.isin(real_bio,_ix(_efor))
 if _est.any() and _efo.any():
  # Blurs and noise on gpu_noise (same bytes as box_blur / terrain_noise; CPU fallback).
  _gb=gpu_noise.box_blur
  _esh=_gb(_gb(_est.astype(np.float32),ECOTONE),ECOTONE);_ee=_gb(real_elev,2);_egy=np.gradient(_ee,axis=0)
  _et=_esh+.45*gpu_noise.fbm(q,r,(NQ,NR),ECOTONE_GRAIN,3,seed=121)-.35*np.clip((_ee-_gb(_gb(real_elev,16),16))/250,-1,1)-.30*np.clip(-_egy/120,-1,1)
  _eco=(_est|_efo)&(_esh>.02)&(_esh<.98)
  def _edom(ns):
   best=np.full(real_bio.shape,-1.,np.float32);cls=np.zeros(real_bio.shape,np.uint8)
   for i in _ix(ns):
    sh=_gb((real_bio==i).astype(np.float32),2*ECOTONE);up=sh>best;best[up]=sh[up];cls[up]=i
   return cls
  _fl=_eco&_efo&(_et>.5);_eb[_fl]=_edom(_eopen)[_fl]
  _fl=_eco&_est&(_et<=.5);_eb[_fl]=_edom(_efor)[_fl]
  del _esh,_ee,_egy,_et,_eco,_fl,_gb
 del _est,_efo
 # --- MATERIALS ECOTONE (ecotone_bands.py) --- pairs the block above leaves hard: alpine<->forest by a
 # latitude treeline on real elevation (alpine above, forest below; the ridge rule is backwards there),
 # then alpine<->steppe/snow/desert/grass by blurred share + fbm, then tundra<->forest (the straight RESOLVE arctic
 # treeline: warped outline + mosaic, river tongues; §9.3). TW3K_MATERIALS_ECOTONE=0 skips.
 if os.environ.get('TW3K_MATERIALS_ECOTONE','1')!='0':
  import ecotone_bands as _etb
  with np.load(REAL_TERRAIN) as _gz:
   _elat=np.zeros((TH,TW),np.float32);_x0,_x1,_y0,_y1=(int(v) for v in _gz['box']);_elat[_y0:_y1,_x0:_x1]=_gz['lat']
   _eriv=np.zeros((TH,TW),bool);_eriv[_y0:_y1,_x0:_x1]=_gz['river']
  _eb,_etn=_etb.treeline(_eb,_bnames,real_elev,_elat,q,r,(NQ,NR),ECOTONE,ECOTONE_GRAIN)
  _eb,_edn=_etb.alpine_dither(_eb,_bnames,q,r,(NQ,NR),ECOTONE,ECOTONE_GRAIN)
  _eb,_etf=_etb.tundra_forest(_eb,_bnames,real_elev,_eriv,q,r,(NQ,NR)) if _frame.arctic_treeline_on() else (_eb,0)
  print('Materials ecotone: treeline cells',_etn,'alpine dither cells',_edn,'tundra/forest cells',_etf,flush=True);del _elat,_eriv
 # --- OASIS GROUND (oasis_ground.conifer_floor) --- lower forest line of the dry ranges: RESOLVE conifer on the
 # Bogda north piedmont (600-1300 m) drew one dark cold-donor sheet (live 2048p); below CONIFER_FLOOR on dry
 # ground forest takes the dominant open class around it (humid forests keep theirs: dryness 0).
 import oasis_ground as _og;_eb,_cfn=_og.conifer_floor(_eb,_bnames,real_bio,real_elev,q,r,(NQ,NR));print('Conifer floor cells',_cfn,flush=True)
 real_bio_eco=_eb   # edited classes for GROUND MATERIALS
 # The donor climate score (blend rows) box-blurs the tile climate over 6 cells: an interleaved
 # ecotone came out as one smooth level-set line between the cold forest and the arid donors
 # (taiga/tundra at hex 1650-1690,1740-1760, review 2026-10-23). Where the classes were redrawn
 # (USR3 and the blocks above), ECO_DETAIL of the fine (radius ECO_DETAIL_R) share goes back into the score.
 if os.environ.get('TW3K_MATERIALS_ECOTONE','1')!='0':eco_zone=(np.clip(box_blur(((_eb!=real_bio)&(real_gw>.5)).astype(np.float32),4)*4,0,1)*real_gw).astype(np.float32)
 # --- end MATERIALS ECOTONE
 _rc=_clim[_eb];biome=np.where(((real_gw>.5)|(real_dgw>.5))&(_rc<4),_rc,biome).astype(biome.dtype)   # [EGOBI] real_dgw
 real_steppe=np.isin(_eb,[_bnames.index(n) for n in _bp.biome_set('polish_steppe_donor') if n in _bnames])&((real_gw>.5)|(real_dgw>.5))   # [EGOBI] real_dgw: steppe near the native border (East Gobi 1327,1092 took the carried cold donor)
 import oasis_ground as _og;sub_split=_og.subtropical_split(_eb,_bnames,real_gw)   # --- OASIS GROUND --- subtropical donor split
 real_wet=np.isin(real_bio,[_bnames.index(n) for n in _bp.biome_set('polish_wet') if n in _bnames])&(real_gw>.5)
else:
 real_gw=real_w;real_steppe=np.zeros((TH,TW),bool);real_wet=np.zeros((TH,TW),bool)
if _memo is None and _memo_key is not None:   # --- SETUP MEMO --- temp + replace; only an optimisation, never fatal
 _tmp=SETUP_MEMO.with_name(f'{SETUP_MEMO.name}.{os.getpid()}.tmp')
 try:
  with open(_tmp,'wb') as _f:np.savez(_f,biome=biome,key=np.array(_memo_key))
  os.replace(_tmp,SETUP_MEMO)
 except OSError as _e:
  print('Setup memo not saved:',_e,flush=True)
  try:_tmp.unlink(missing_ok=True)
  except OSError:pass
_memo_hit=_memo is not None;del _memo   # the memo state for later guards (_memo itself is gone)
# --- HEHUANG PATCH (hehuang_terrain.py) --- load analysis/hehuang-patch.npz; steppe donor on its steppe class
if HH is not None:real_steppe=real_steppe|HH.steppe(q,r);print('Hehuang patch',HH.report(),flush=True)
# Fractal value noise, not sine sums: periodic terms drew evenly spaced parallel stripes.
if not CACHE_ONLY:
 relief=LEVEL+.16*gpu_noise.fbm(q,r,(NQ,NR),18,4,seed=11)
# Relief lines, valleys and biome bands live in canvas-authored-coordinates.json
features=_frame.relief_features(AUTHORED,plan.get('coast_coverage_revision'))
import relief_model
def line_distance(points,q=q,r=r):return relief_model.line_distance(points,q,r)
if not CACHE_ONLY:
 # Massifs (relief_model.py; tune with preview_relief.py): towns sit in broad valleys.
 relief_model.add_massifs(relief,q,r,(NQ,NR),features,relief_model.town_valleys(q,r,plan['regions']))
 # Over water the real relief fades in away from the native map like the real ground (real_gw):
 # at full weight right outside it, the DEM's coastal shelf met vanilla's deep sea floor in a step
 # on the native border (seam audit lf_cliff off Kyushu, whole-canvas real terrain).
 _real_rw=np.where(physical,real_w,real_gw)
 relief=relief*(1-_real_rw)+real_rel*_real_rw
 # Town floors in the real terrain: each town's ground levels to ONE height (the
 # smoothed real height at the town), not to a smoothed field that still tilts on
 # mountain feet (town terraces failed audit_city_ground_coverage at Wensu, Yiwu...).
 # Gaussian-weighted mix of the town heights (no step where two towns' floors meet).
 _sm=box_blur(box_blur(real_rel,6),6);_num=np.zeros((TH,TW),np.float32);_den=np.zeros((TH,TW),np.float32);_gden=np.zeros((TH,TW),np.float32)
 for _t in plan['regions']:
  _tq,_tr=_t['town'];_cx,_cy=int(_tq*2),int(round((_tr+_tq%2*.5)*2))
  if not(0<=_cx<TW and 0<=_cy<TH) or real_w[_cy,_cx]<=0:continue
  _ys,_xs=slice(max(0,_cy-40),min(TH,_cy+41)),slice(max(0,_cx-40),min(TW,_cx+41))
  _g=np.exp(-((x[_ys,_xs]-_cx)**2+(y[_ys,_xs]-_cy)**2)/(2*10.**2)).astype(np.float32)
  _iw=_town_mix_weight(x[_ys,_xs]-_cx,y[_ys,_xs]-_cy,TOWN_MIX_CORE[0] if _t['is_capital'] else TOWN_MIX_CORE[1])   # TOWN SPEC (town ground mix)
  _num[_ys,_xs]+=_iw*_sm[_cy,_cx];_den[_ys,_xs]+=_iw;_gden[_ys,_xs]+=_g
 _floor=np.where(_gden>1e-3,(_num+_sm*1e-6)/(_den+1e-6),_sm);del _gden
 town_floor=LEVEL*(1-real_w)+_floor*real_w;del _num,_den,_floor
 # Plain smoothstep apron in the real terrain as well: the sqrt-strengthened apron
 # ended in a ring cliff around every town (offline render 2048bc, Yanqi, Qiuci).
 # Coarse grids put a vertex on each town instead (snap_grid_to_towns).
 # --- TOWN FOOTPRINT --- real-terrain canvas towns: the floor ramp (clearance inner..outer) narrows
 # where the ground around falls away from the town floor. On a piedmont the 8-20 hex capital ring
 # held the floor out over a 3.1-unit fall to the basin and read as a disc (r20, north of Danhuan,
 # q=650: 5.69 floor over a 2.55 basin). The ramp width (outer-inner) scales by 1 down to
 # TOWN_RAMP_MIN as the smoothed real ground there lies TOWN_DROP[0]..TOWN_DROP[1] under the floor;
 # the flat inner disc stays. Per pixel along each ray, so level or rising sides keep the full ring.
 # Vanilla native towns and towns off the real terrain keep the clearance unchanged.
 TOWN_DROP=(.5,2.)       # world units under the town floor: ramp starts / stops narrowing
 TOWN_RAMP_MIN=5/12      # narrowest ramp share (a capital's 12-hex ramp -> 5 hexes, outer 13)
 _surround=LEVEL*(1-real_w.astype(np.float64))+_sm.astype(np.float64)*real_w
 clearance_floor=np.zeros((TH,TW),np.float32)
 for town in plan['regions']:
  tq,tr=town['town']
  inner,outer_town=(8,20) if town['is_capital'] else (5,13)
  win=_town_window(tq,tr,outer_town);sq,sr=_town_centre(tq,tr);d=np.hypot(q[win]-sq,r[win]-sr)
  _cx,_cy=int(tq*2),int(round((tr+tq%2*.5)*2))
  if 0<=_cx<TW and 0<=_cy<TH and real_w[_cy,_cx]>0 and not native_protected[_cy,_cx]:
   _drop=np.maximum(town_floor[win].astype(np.float64)-_surround[win],0.)
   _s=np.clip(1-(_drop-TOWN_DROP[0])/(TOWN_DROP[1]-TOWN_DROP[0])*(1-TOWN_RAMP_MIN),TOWN_RAMP_MIN,1.)
   t=np.clip((d.astype(np.float64)-inner)/((outer_town-inner)*_s),0,1);_ck=(1-t*t*(3-2*t)).astype(np.float32);del _drop,_s
  else:
   t=np.clip((d-inner)/(outer_town-inner),0,1);_ck=1-t*t*(3-2*t)   # as the clearance above
  clearance_floor[win]=np.maximum(clearance_floor[win],_ck)
 # Native-map pixels keep the clearance exactly (the edit alpha above still uses the clearance).
 clearance_floor=np.where(native_protected,clearance,np.minimum(clearance_floor,clearance)).astype(np.float32);del _surround,_sm
 # Real rivers cut shallow valleys.
 relief-=.35*box_blur(real_river,2)*real_w
 # Preserve wide playable valley approaches and native town height anchors.
 # The town floor is blended into the relief ONCE, on land after the lake bowls and the town
 # cluster (TOWN BLEND below): town_level carries the floor through the coastal falloff, the
 # lake bowl and the cluster levels instead. Blending relief*(1-c)+town_floor*c here, again
 # after the coastal falloff and a third time in TOWN_CLUSTER gave weight 1-(1-c)^3: a flat
 # plateau ~13 hexes wide round real-terrain piedmont capitals, then a rim falling ~3 units
 # over 5 hexes (Danhuan 5.69, Beilu 5.64, Jushi rear 4.47 over a 2.5-2.8 basin; the stage LF
 # fits the three-pass model). The sea keeps its blend here (before the coastal falloff, as before).
 relief=np.where(physical,relief,relief*(1-clearance_floor)+town_floor*clearance_floor)
 town_level=town_floor.copy()
 # Coast rolls down below the native sea plane, including small islands.
 shore=np.clip((coast-.25)/.5,0,1);shore=shore*shore*(3-2*shore)
 # Open sea floor = vanilla open-ocean floor (raw 3084 = -2.445). A flat -2 met
 # vanilla's deeper floor in a straight step on the native border, read live as
 # rectangular deep/shallow water (2048j Kyushu, Kimhae, Rinan, Lin Yi).
 # v21d: the lake bowl keeps the ground's fine relief (high-pass of the terrain before this falloff: the lake cells
 # still hold the flat DEM lake surface there, so no pit bleeds into it); see LAKE_BOWL_CORE below
 _bowl_detail=relief-box_blur(box_blur(relief,6),6) if LAKE_BOWL_V21 else None
 # LAKE LEVEL: the original relief (before this falloff and the lake bowl) that the lake levels are solved from
 relief_orig=relief.copy() if lake_level_lakes else None
 relief=SEA_FLOOR+(relief-SEA_FLOOR)*shore
 # Town plots on land are levelled after the coastal falloff (TOWN BLEND below). Applying
 # the falloff last previously submerged one fishing town's ground apron.
# Only visual river valleys here. Native river meshes are retained when valid;
# New river water still requires separate connected water geometry and live QA.
valleys=[(v['name'],[tuple(p) for p in v['points']],v['width']) for v in AUTHORED['valleys']]
if not CACHE_ONLY:
 for name,points,width in valleys:
  # exp(-(d/width)**2) is exactly 0 in float32 beyond ~10.2 widths, so the distance is
  # only evaluated within 11 widths of each segment (relief_model.line_distance_near).
  v=np.exp(-(relief_model.line_distance_near(points,q,r,11*width+2)/width)**2)*.55*(1-clearance)*(1-real_w)
  relief=relief*(1-v)+LEVEL*v
 # Oasis belts: desert river valleys take green riparian ground (temperate donor)
 # in a belt of varying width, dithered at its edge (Tarim etc. read as bare sand).
 # Only low valleys (Tarim, Hexi, Ili, Transoxiana: full below OASIS_ELEV[0] metres,
 # none above OASIS_ELEV[1]): on the Tibetan plateau the belts drew bright green
 # bands along every valley (Yangtong, offline frontier-combined). High valleys
 # take narrower alpine-meadow belts of steppe donor ground (alpine) instead.
 OASIS_VALLEYS=_bp.region('oasis_valleys')['weights']   # dead path (only without REAL_TERRAIN): marked for removal, kept (biome_profiles.REGIONS)
 OASIS_ELEV=_BP['OASIS_ELEV']
 oasis=np.zeros((TH,TW),np.float32)
 low_valley=np.clip((OASIS_ELEV[1]-box_blur(real_elev,3))/(OASIS_ELEV[1]-OASIS_ELEV[0]),0,1).astype(np.float32)
 _rv=box_blur(real_river,3);_bw=(.6+.6*np.clip(.5+.7*gpu_noise.fbm(q,r,(NQ,NR),7,3,seed=92),0,1))
 oasis=np.maximum(oasis,(np.clip(_rv*5-.15,0,1)*np.isin(biome,[3])+real_wet*.9).astype(np.float32)*real_gw*_bw*low_valley)
 oasis=np.clip(oasis,0,1).astype(np.float32)
 # Alpine belts: half the oasis width (blur 2, steeper ramp), wetlands at .6.
 _rv=box_blur(real_river,2)
 alpine=np.clip((np.clip(_rv*4-.45,0,1)*np.isin(biome,[3])+real_wet*.6)*real_gw*_bw*(1-low_valley),0,1).astype(np.float32);del _rv,_bw
 for name,points,width in valleys:
  if name not in OASIS_VALLEYS or REAL_TERRAIN.exists():continue
  _w=width*OASIS_VALLEYS[name]*(.45+.9*np.clip(.5+.7*gpu_noise.fbm(q,r,(NQ,NR),9,3,seed=91),0,1))
  oasis=np.maximum(oasis,np.clip(1.3-line_distance(points)/_w,0,1).astype(np.float32))
 # Inland lakes on the canvas (3k_main_sea_lake, add_inland_lake.py): the water
 # plane sits at about -1, so a lake in a plain at LEVEL read as a pit with a
 # wall. Land around them sinks to the shore in a broad bowl of LAKE_BOWL cells,
 # and a ring of riparian ground follows the shore.
 # Real lakes sit far above the water plane (Bosten ~1050 m): the bowl reaches
 # 30 hexes so lake-shore towns stand near the shore, not on 4-unit plateaus whose
 # grade limit held the lake floor above the water (Bosten invisible, 2048bc).
 # The campaign water plane is ~0, not ~-1.05: lf_sea_height_map holds 14219 (-0.015 in LF
 # units) over the canvas lakes, and the offline render at sea level 0 reproduces the live
 # water outline at Yangtong (live 2048k). A shore target of -0.7 put the land of the bowl's
 # first ~5 hexes under water: the visible lake ~5x its 3k_main_sea_lake hexes, read as 平原.
 LAKE_WATER=_lake_level.LAKE_WATER   # campaign water plane (world height; lake_level.LAKE_WATER, one constant shared with river_water_mesh)
 LAKE_FREEBOARD=.12     # land hexes never sink below LAKE_WATER+LAKE_FREEBOARD
 LAKE_BOWL,LAKE_SHORE_HEIGHT,LAKE_OASIS=80,LAKE_WATER+.25,_BP['LAKE_OASIS']
 # v21d (lake water-level diagnosis, decisions 2026-10-31): the water plane sits ~2-5 under the real ground around
 # every canvas lake (Bosten 4.97, Lop Nur 4.20, Balkhash 2.55), so the bowl is a ~4 deep pit. A 60-cell smoothstep
 # put the lake-shore towns (4-9 hexes out) on its .07-.1 per cell flank: a plot spanned 1.5-3 units and stood as a
 # wall and a pale platform (Shanshan lumber .475 under a 4.3 plain, Lop Nur west; Yanqi / Weili / Weixu, Bosten).
 # The bowl now has a flat core of LAKE_BOWL_CORE cells (~12 hexes: the shore towns stand on level ground at the
 # shore height) and rises over LAKE_BOWL cells beyond it. The root fix (a water level per lake from its shore
 # ground) is an engine-layer test of its own (decisions 2026-10-31, step 2).
 LAKE_BOWL_CORE=24      # tile cells: the bowl's flat core around each lake (weight 1)
 LAKE_BOWL_DETAIL=.4    # world units: the fine relief the bowl keeps (clipped; mountain flanks would stand as spikes)
 if not LAKE_BOWL_V21:LAKE_BOWL,LAKE_BOWL_CORE=60,0   # lake stop-loss (LAKE_BOWL_V21 off): the v20 bowl
 LAKE_FLOOR=-1.6        # lake hexes never rise above this (well under the water plane)
 # r15 Bosten review: .3 dropped the floor to LAKE_FLOOR within 5 cells (2.5 hexes) of the shore, a
 # pit wall under the water; .12 reaches it after ~12 cells, so a narrow lake (Bosten ~9 rows) has a
 # sloping floor from shore to middle, still under the water plane everywhere (-.135 at the shore cell).
 LAKE_SHELF=.12         # world units per tile cell the lake floor deepens from the shore
 TOWN_CLUSTER=8         # hexes: Gaussian radius over which real-terrain towns share a level
 LAKE_TOWN_RISE=.03     # town level cap above LAKE_SHORE_HEIGHT per tile cell from the lake
 LAKE_TOWN_SLOPE=.1     # TOWN SPEC: a capped town's level carries to its neighbours at this rise per tile cell ...
 LAKE_TOWN_COUPLE=32    # ... within this many tile cells (2 x 2 TOWN_CLUSTER: the cluster mix's reach)
 LAKE_TOWN_TOUCH=2      # tile cells: plots this much closer than LAKE_TOWN_PLOT touch once rounded (Wusun capital / pasture: .25)
 LAKE_TOWN_PLOT=(10,8)  # tile cells: plot radius of a capital / other town (equivalent radius of the plots, Wusun 9.7 / 7.7); the rise counts from plot edge to plot edge
 MIN_LAKE_TOWN_HEIGHT=.4   # town datum floor for lake-shore towns (MIN_TOWN_HEIGHT elsewhere)
 LAKE_TOWN_REACH=24     # tile cells: towns this close to a lake outline count as lake-shore towns
 _lake=(~physical)&outer&(ids[hy,hx]==names.index('3k_main_sea_lake')) if '3k_main_sea_lake' in names else np.zeros_like(physical)
 lake_dist=np.full((TH,TW),np.float32(LAKE_BOWL))   # tile cells to the rounded lake outline (capped)
 # LAKE LEVEL: the solved levels (name -> product entry; empty off)
 lake_level_solved={}
 lake_level_sdf={}   # LAKE LEVEL: name -> (signed distance on the LF grid of its window, tile window, tile samples)
 lake_level_shape_tiles={}   # LAKE LEVEL (v4): name -> window tile relief (before, after) the tile shape, outer (LAKE LEVEL SHAPE)
 # LAKE LEVEL (v3): name -> {region: nearest plot pixel's signed distance} of the towns with a plot in the lake's
 # window (the town level solve reads its envelope cap from it); the tile cells nearer a level lake than any other
 # lake (lake-shore towns there take the envelope cap); the tile cells whose cluster-mix input the switch changes
 # (the Bosten-grown v20 bowl support and the lake windows: the reach seed)
 lake_level_town_s={};lake_level_nearest=None;lake_level_support=None
 if _lake.any():
  # Distances from a rounded outline, 8-neighbour steps: the hex-cell outline with
  # 4-neighbour steps drew half-hex teeth as north-south streaks on the shores.
  _round=box_blur(box_blur(_lake.astype(np.float32),3),3)>=.5
  # The noise below scales the distance by .75-1.25, so the bowl reaches LAKE_BOWL/.75 cells.
  # The distance grew only to LAKE_BOWL (capped there), and .75*LAKE_BOWL left a bowl weight
  # of up to .16 on about half the canvas far from any lake (stage surface: up to ~1 unit of
  # ground pulled toward the shore height in noise patches). It now grows to LAKE_REACH, where
  # the weight is exactly 0; below LAKE_BOWL cells every value is as before.
  LAKE_REACH=int(np.ceil((LAKE_BOWL_CORE+LAKE_BOWL)/.75))
  def _lake_grow(_seed):
   _ld=np.full(_lake.shape,np.float32(LAKE_REACH));_front=_g=None
   if _seed.any():
    # The growth reaches at most LAKE_REACH-1 cells (one per step, Chebyshev) from the
    # outline, so it runs in that window around the lakes only (identical values).
    _rr=np.flatnonzero(_seed.any(axis=1));_cc=np.flatnonzero(_seed.any(axis=0))
    _lwin=(slice(max(0,int(_rr[0])-LAKE_REACH),min(_lake.shape[0],int(_rr[-1])+LAKE_REACH+1)),slice(max(0,int(_cc[0])-LAKE_REACH),min(_lake.shape[1],int(_cc[-1])+LAKE_REACH+1)))
    _ldw=_ld[_lwin];_ldw[_seed[_lwin]]=0;_front=_seed[_lwin].copy()
    for _i in range(1,LAKE_REACH):
     _g=_front.copy();_g[1:]|=_front[:-1];_g[:-1]|=_front[1:];_g[:,1:]|=_front[:,:-1];_g[:,:-1]|=_front[:,1:]
     if _i%3:_g[1:,1:]|=_front[:-1,:-1];_g[:-1,:-1]|=_front[1:,1:];_g[1:,:-1]|=_front[:-1,1:];_g[:-1,1:]|=_front[1:,:-1]
     _ldw[_g&~_front]=_i;_front=_g
    del _ldw
   return _ld
  _front=_g=None
  if not lake_level_lakes:_ld=_lake_grow(_round);_ld_bowl=_ld
  else:
   # LAKE LEVEL: the bowl grows from the other lakes' outline only (_ld_bowl); the lake distance (lake-shore towns,
   # the riparian ring) is the minimum of both growths, equal to the growth from the whole outline (the dilation
   # steps distribute over the union of the seeds). Outline cells within 8 cells of a level lake belong to it.
   _ll_round=_round&fastops.max_filter_mask(lake_level_mask,17)
   _ld_bowl=_lake_grow(_round&~_ll_round);_ld_ll=_lake_grow(_ll_round);_ld=np.minimum(_ld_bowl,_ld_ll)
  _lnz=(.75+.5*np.clip(.5+.7*gpu_noise.fbm(q,r,(NQ,NR),6,3,seed=94),0,1))
  _lw=_ld*_lnz;_lw_bowl=_lw if _ld_bowl is _ld else _ld_bowl*_lnz;del _lnz
  _b=np.clip(1-(_lw_bowl-LAKE_BOWL_CORE)/LAKE_BOWL,0,1);_b=_b*_b*(3-2*_b)
  # Near authored towns the bowl stops at the town datum (MIN_TOWN_HEIGHT) and
  # reaches the shore height only ~10 hexes out: lake-shore towns otherwise stood
  # as walled plateaus in the water (Bosten, Issyk-Kul, offline render).
  _tw=np.clip((town_distance-4*DX)/(10*DX),0,1)
  # r15 Bosten review: near the lake the town term is the lake-shore town floor (MIN_LAKE_TOWN_HEIGHT,
  # the datum those towns get below), rising smoothly to MIN_TOWN_HEIGHT by 1.5 LAKE_TOWN_REACH from the
  # outline. MIN_TOWN_HEIGHT right at the water stood a 1.0 bank beside a lake port (焉耆·渔港) and beside
  # lake-shore towns whose datum is already .4.
  _tl=np.clip(_ld_bowl/(1.5*LAKE_TOWN_REACH),0,1);_tl=_tl*_tl*(3-2*_tl)
  _town_h=MIN_LAKE_TOWN_HEIGHT+(MIN_TOWN_HEIGHT-MIN_LAKE_TOWN_HEIGHT)*_tl;del _tl
  # v21d: the bowl keeps the fine relief (_bowl_detail, +-LAKE_BOWL_DETAIL): a flat core alone read as a slab
  if LAKE_BOWL_V21:relief=np.where(physical,relief*(1-_b)+(LAKE_SHORE_HEIGHT*_tw+_town_h*(1-_tw)+np.clip(_bowl_detail,-LAKE_BOWL_DETAIL,LAKE_BOWL_DETAIL))*_b,relief)
  else:relief=np.where(physical,relief*(1-_b)+(LAKE_SHORE_HEIGHT*_tw+_town_h*(1-_tw))*_b,relief)
  town_level=np.where(physical,town_level*(1-_b)+(LAKE_SHORE_HEIGHT*_tw+_town_h*(1-_tw))*_b,town_level);del _town_h
  # Plateau lakes (Yangtong) take a narrower alpine ring.
  # --- USR3 BOSTEN PATCH --- reed ring at river mouths only, desert-steppe ring elsewhere;
  # lake hexes are included (those left above the water read as bare sand blocks).
  _ring=np.clip(1.2-_lw/LAKE_OASIS,0,1).astype(np.float32)*(physical|_lake)
  _mouth=np.clip(box_blur(box_blur(real_river,LAKE_MOUTH[0]),LAKE_MOUTH[0])*LAKE_MOUTH[1],0,1).astype(np.float32)
  oasis=np.maximum(oasis,_ring*low_valley*_mouth);alpine=np.maximum(alpine,_ring*low_valley*(1-_mouth));del _ring,_mouth
  alpine=np.maximum(alpine,np.clip(1.2-_lw/(LAKE_OASIS/2),0,1).astype(np.float32)*physical*(1-low_valley))
  # Lake floors stay under the water plane right to the hex edge: the coastal
  # falloff left the outer cells of lake hexes near the shore height, and
  # where they rose above the water they showed as pointed islands (Lop Nur,
  # Balkhash, offline render 2048bg).
  # A shelf from LAKE_WATER-LAKE_FREEBOARD at the shore cell down to LAKE_FLOOR, under the
  # projection's TOWN_GRADE (1/3 world per tile cell): a -1.6 floor right against the dry shore
  # was ramped by the projection and pulled the shore under the water again.
  _shelf=fastops.grow_steps(fastops.max_filter_mask(_lake,3)&~_lake,int(np.ceil((LAKE_WATER-LAKE_FREEBOARD-LAKE_FLOOR)/LAKE_SHELF))+1).astype(np.float32)
  if not lake_level_lakes:relief=np.where(_lake,np.minimum(relief,np.maximum(LAKE_FLOOR,LAKE_WATER-LAKE_FREEBOARD-LAKE_SHELF*(_shelf-1))),relief)
  else:
   # LAKE LEVEL (v2): each level lake is solved once here (lake_level.solve_level: shore-ring original relief, before
   # the coastal falloff and this bowl) and its waterline drawn once: the signed distance to the smoothed, grown lake
   # outline on the LF grid of a window round it (lake_level.outline_sdf; never the hex edge). Its cells keep the
   # relief here; the ground is shaped from that distance once, after the TOWN BLEND (LAKE LEVEL SHAPE below), and
   # the projection is bounded by it. The other lakes as before.
   relief=np.where(_lake&~lake_level_mask,np.minimum(relief,np.maximum(LAKE_FLOOR,LAKE_WATER-LAKE_FREEBOARD-LAKE_SHELF*(_shelf-1))),relief)
   _ll_plot=authored_plot[hy,hx]
   # v3: each design town's plot hexes labelled (town index + 1) for its nearest plot pixel to the waterline
   _ll_lut=np.zeros(len(names)+1,np.int32)
   for _i,_t in enumerate(plan['regions']):_ll_lut[names.index(_t['region'])]=_i+1
   _ll_plab=np.where(authored_plot,_ll_lut[ids],0).astype(np.int32);del _ll_lut
   lake_level_nearest=_ld_ll<_ld_bowl;lake_level_support=_ld_ll<LAKE_REACH
   for _ll_name,(_ll_m,_ll_e) in sorted(lake_level_lakes.items()):
    _ll=_lake_level.solve_level(relief_orig,_ll_m,physical&outer,_ll_e);_lv=_ll['level']
    _wy0,_wy1,_wx0,_wx1=_lake_level.window(_ll_m,_ll_e['window_cells'],(TH,TW))
    _ll_s,_ll_info=_lake_level.outline_sdf(_ll_m[_wy0:_wy1,_wx0:_wx1],_ll_plot[_wy0:_wy1,_wx0:_wx1],_ll_e,_lake_level.lf_pitch())
    lake_level_sdf[_ll_name]=(_ll_s,(_wy0,_wy1,_wx0,_wx1),_lake_level.tile_samples(_ll_s))
    _F=_lake_level.FINE;_pl=np.repeat(np.repeat(_ll_plab[hy[_wy0:_wy1,_wx0:_wx1],hx[_wy0:_wy1,_wx0:_wx1]],_F,0),_F,1)
    lake_level_town_s[_ll_name]={plan['regions'][_k-1]['region']:_v for _k,_v in sorted(_lake_level.plot_distances(_ll_s,_pl).items())};del _pl
    lake_level_support[_wy0:_wy1,_wx0:_wx1]=True
    _ys,_xs=np.nonzero(_ll_m)
    lake_level_solved[_ll_name]={**_ll,'cells':int(_ll_m.sum()),'centroid_tile_yx':[round(float(_ys.mean()),2),round(float(_xs.mean()),2)],
                                 'bbox_tile_yx':[int(_ys.min()),int(_ys.max()),int(_xs.min()),int(_xs.max())],'window_tile':[_wy0,_wy1,_wx0,_wx1],
                                 'outline':_ll_info,'spec':_ll_e,'cells_yx':np.stack([_ys,_xs],1).tolist()}
    print('Lake level',_ll_name,{k_:v_ for k_,v_ in lake_level_solved[_ll_name].items() if k_!='cells_yx'},flush=True)
   del _ll_m,_ys,_xs,_ll_s,_ll_plot,_ll_plab
  del _shelf
  # Water only over lake hexes: land in the bowl (incl. cells the coastal falloff pulled toward
  # SEA_FLOOR, and land cells inside the rounded outline) keeps a dry shore.
  relief=np.where(physical&(_b>0),np.maximum(relief,LAKE_WATER+LAKE_FREEBOARD),relief)
  town_level=np.where(physical&(_b>0),np.maximum(town_level,LAKE_WATER+LAKE_FREEBOARD),town_level)
  lake_dist=np.minimum(_ld,LAKE_BOWL).astype(np.float32)   # capped at LAKE_BOWL as before (v21d: 80, was 60; read below LAKE_TOWN_REACH*1.5 only)
  del _ld,_front,_g,_lw,_b,_round,_ld_bowl,_lw_bowl
  if lake_level_lakes:del _ld_ll,_ll_round
 del _bowl_detail
 # --- DRY-LAND FLOOR --- canvas land keeps the dry-shore floor beyond the lake bowls as well.
 # The far-field bowl (weight up to .16 everywhere, removed with LAKE_REACH) carried the floor
 # max(relief,LAKE_WATER+LAKE_FREEBOARD) over about half the canvas; it kept thin non-playable
 # coastal land slivers dry. Without it 13 land hexes sank to the sea floor (LF -1.2..-2.44: 940,
 # 163-166; 36,1045; 79,1063-64; 2007,1323; 1450-1453,1843-44, r20), fell under OCEAN_DROP_Y, lost
 # their coarse faces, and patch_coverage_holes refilled ~1184 faces. Only the floor comes back,
 # not the bowl. Inside the bowls the clamp above already holds it (no change there); cells within
 # DRY_FLOOR_RIVER_GAP of a frontier river hex keep their height (river banks and the river level
 # read from them stay as they are). The native map is untouched.
 DRY_FLOOR_RIVER_GAP=24   # tile cells (12 hexes)
 _rvc=(np.isin(ids,[i for i,k in enumerate(names) if k.startswith('tw3k_riv_sea_')])&~land)[hy,hx]
 _dry=physical&~native_protected&~fastops.max_filter_mask(_rvc,2*DRY_FLOOR_RIVER_GAP+1);del _rvc
 relief=np.where(_dry,np.maximum(relief,LAKE_WATER+LAKE_FREEBOARD),relief)
 town_level=np.where(_dry,np.maximum(town_level,LAKE_WATER+LAKE_FREEBOARD),town_level)
 # HEIGHT BAKE lower bound L, first piece (height-bake-design 2): the lake-shore land keeps its dry floor through the
 # slope projection too. The projection pulled shore land under the water plane at the grade from the lake floor
 # (LAKE_FLOOR -1.6) once nothing held it up (v21: 14 land tiles at the Wusun shore after the lake cap fix; the
 # floor above holds the relief only). Lake-bowl land (LAKE_FLOOR_REACH tile cells from a lake, outside the
 # native map, off the frontier rivers' DRY_FLOOR_RIVER_GAP) is a lower bound of the projection at the floor.
 # Invariant: lake_shore_dry.
 LAKE_FLOOR_REACH=40      # tile cells (20 hexes) from the lake outline
 # v21b art review: a floor at the full height right to the lake hexes cut the canvas lakes back to their hex
 # outline (straight cuts, Bosten west; a strip parting a pool off the Aral) and stood as a flat .105 terrace with a
 # step at both edges (bowl ground .25 -> .105 -> the water). The shore is shaped in the relief instead, along a
 # smooth waterline: the signed distance to the lake tiles (+ inland, tile cells), Gaussian LAKE_SHORE_SMOOTH, held
 # >= LAKE_CENTRE_MARGIN at every land hex centre of the zone and <= -LAKE_CENTRE_MARGIN at every lake hex centre.
 # Inland of the line the ground rises from the water plane at LAKE_SHORE_SLOPE per cell and joins the relief over
 # LAKE_SHORE_BAND cells; on the water side (lake hex corners and land hex edges alike) it is cut under the plane
 # along the lake shelf. The projection floor (shore_sd_tiles, below) only keeps the inland side above the plane.
 LAKE_SHORE_SMOOTH=2.     # tile cells: Gaussian of the signed distance (rounds the hex outline of the waterline)
 LAKE_SHORE_SLOPE=.02     # world units per tile cell: the projection floor rises at half this from the water plane
 LAKE_SHORE_BAND=6.       # tile cells: the relief eases down to the water plane over this from the line (smoothstep)
 LAKE_CENTRE_MARGIN=.8    # tile cells: land hex centres this far inland of the line, lake hex centres this far out
 LAKE_CEIL_DEPTH=.03      # world units: lake tiles stay at least this far under the plane through the projection
 LAKE_BED_REACH=float(np.ceil((LAKE_WATER-LAKE_FLOOR)/LAKE_SHELF))+2   # tile cells: the shelf reaches LAKE_FLOOR (v21d)
 LAKE_CEIL_SLACK=.15      # world units: the projection may lift the lake bed at most this far above its shelf (v21d)
 # Euclidean distance to the lake tiles themselves (not lake_dist: its rounded outline drops lakes of a few hexes,
 # whose shores then sank: 113 tiles in the first copy run); stage_invariants.lake_shore_dry measures the same zone.
 from scipy import ndimage as _nd_floor
 if LAKE_BOWL_V21 and _lake.any():   # lake stop-loss: off, no shore shaping and no floors / ceilings (v20)
  _sdl=(_nd_floor.distance_transform_edt(~_lake)-_nd_floor.distance_transform_edt(_lake)).astype(np.float32)
  shore_zone_tiles=_dry&(_sdl<LAKE_FLOOR_REACH)
  _sds=_nd_floor.gaussian_filter(np.clip(_sdl,-LAKE_BED_REACH,LAKE_FLOOR_REACH),LAKE_SHORE_SMOOTH).astype(np.float32);del _sdl   # v21d: clipped at -8 the bed stopped at -.975 (pale water)
  # hex centre tiles (2q+1, 2r+q%2+1): land centres of the zone stay inland of the line, lake centres stay out
  _qq,_rr=np.meshgrid(np.arange(W),np.arange(H));_cyy=np.minimum(2*_rr+_qq%2+1,TH-1);_cxx=np.minimum(2*_qq+1,TW-1)
  _lc=np.zeros((TH,TW),bool);_kc=np.zeros((TH,TW),bool)
  _lc[_cyy[shore_zone_tiles[_cyy,_cxx]],_cxx[shore_zone_tiles[_cyy,_cxx]]]=True
  _kc[_cyy[_lake[_cyy,_cxx]],_cxx[_lake[_cyy,_cxx]]]=True;del _qq,_rr,_cyy,_cxx
  if _lc.any():_sds=np.maximum(_sds,LAKE_CENTRE_MARGIN-_nd_floor.distance_transform_edt(~_lc).astype(np.float32))
  if _kc.any():_sds=np.minimum(_sds,_nd_floor.distance_transform_edt(~_kc).astype(np.float32)-LAKE_CENTRE_MARGIN)
  del _lc,_kc
  _sh=shore_zone_tiles|_lake
  # inland: the relief itself eases to the water plane (water + (relief - water) x smoothstep(distance / BAND)), so
  # the shore has no terrace and no kink where it meets the relief (a separate shore ramp joining the relief drew a
  # line one hex out, first copy run)
  _t=np.clip(_sds/LAKE_SHORE_BAND,0,1);_t=_t*_t*(3-2*_t)
  _land=_sh&(_sds>0)&~_lake   # lake tiles are always the water side (a smooth line across a narrow arm dried 84 lake hex centres)
  relief=np.where(_land,LAKE_WATER+(np.maximum(relief,LAKE_WATER)-LAKE_WATER)*_t,relief).astype(np.float32)
  _wet=np.maximum(LAKE_FLOOR,LAKE_WATER-LAKE_SHELF*np.where(_lake,np.maximum(-_sds,.5),np.maximum(-_sds,0)))   # lake tiles at least half a cell's shelf under   # from the plane itself on the line (a freeboard offset there stood as a wall under the waterline)
  relief=np.where(_sh&~_land,_wet,relief).astype(np.float32);lake_bed_tiles=np.where(_lake,_wet,np.nan).astype(np.float32);del _t,_wet   # set, not min: the coastal falloff had pulled the lake cells to the sea floor, so the shelf never bound and the bed fell at the grade cap under the waterline
  shore_zone_tiles=shore_zone_tiles&~_lake;shore_sd_tiles=_sds;del _land,_sh   # the projection floor: land only
 else:shore_zone_tiles=np.zeros_like(_dry);shore_sd_tiles=None;lake_bed_tiles=None
 del _dry
 # Neighbouring towns in the real terrain share one level: each town's ground
 # height is averaged with the towns around it (Gaussian over TOWN_CLUSTER
 # hexes) and the aprons are re-levelled to that mix. Six Yanqi-basin towns
 # 13-35 hexes apart otherwise each kept its own level, and their plot edges read
 # as a field of terraces (offline render 2048bg).
 lake_chain_towns=set()   # TOWN SPEC: towns the lake cap carried to (filled below)
 if real_w.any():
  _tt=[(int(t['town'][0]*2),int(round((t['town'][1]+t['town'][0]%2*.5)*2)),bool(t['is_capital']),t['region']) for t in plan['regions']]
  _tt=[(cx,cy,cap,rg) for cx,cy,cap,rg in _tt if 0<=cx<TW and 0<=cy<TH and real_w[cy,cx]>0]
  _tcap=np.array([cap for _,_,cap,_ in _tt],bool);_treg=[rg for _,_,_,rg in _tt];_tt=[(cx,cy) for cx,cy,_,_ in _tt]
  if _tt:
   # town_level at a town centre = the relief there before (clearance 1: floor, falloff, bowl).
   _p=np.array(_tt,np.float32);_h=np.array([town_level[cy,cx] for cx,cy in _tt],np.float32)
   _w=np.exp(-((_p[:,None,0]-_p[None,:,0])**2+(_p[:,None,1]-_p[None,:,1])**2)/(2*(2*TOWN_CLUSTER)**2))
   _h=(_w@_h)/_w.sum(1)
   # Lake-shore towns stand a little above their lake: a capital flattens ~8 hexes around it,
   # so at the cluster's 1.4-1.8 the flat ground ran to within a hex of the water and dropped
   # ~1.4 there (Bosten north shore). Cap by distance to the lake outline.
   _ldt=np.array([lake_dist[cy,cx] for cx,cy in _tt],np.float32)
   _capv=np.where(_ldt<LAKE_TOWN_REACH,LAKE_SHORE_HEIGHT+LAKE_TOWN_RISE*_ldt,np.inf)
   # LAKE LEVEL (v3): a town nearer a level lake than any other lake is capped by the lake's bank envelope at its
   # nearest plot pixel (lake_level.town_cap: level + E(s); +inf without a plot in the lake window), not by the one
   # water plane's shore height: the town level solve reads the lake product (v2 left Weixu's plateau .55 over the
   # level .46 from the water, a terrace)
   if lake_level_solved:
    _llc=np.array([min(_lake_level.town_cap(lake_level_solved[n_]['level'],lake_level_town_s[n_].get(rg_),lake_level_lakes[n_][1]) for n_ in sorted(lake_level_solved)) for rg_ in _treg])
    _capv=np.where(np.array([bool(lake_level_nearest[cy,cx]) for cx,cy in _tt]),_llc,_capv);del _llc
   _h=np.minimum(_h,_capv).astype(np.float32);del _capv
   # TOWN SPEC (lake cap continuity): the cap stopped dead at LAKE_TOWN_REACH, so a capped lake-shore town sat ~4
   # units under its uncapped neighbours a few hexes inland; the projection's plot coupling lifted its plateau
   # back (Wusun capital: 1.45 -> 5.22, 3.77 over its own ground) and left grade-cap walls and V slots between
   # the plateaus (v20 art review, town-wusun). The cap now carries on at LAKE_TOWN_SLOPE per tile cell to the
   # towns within LAKE_TOWN_COUPLE cells, and on from each town it lowers (a chain from the capped towns only;
   # elsewhere the levels are as before). Invariant: town_plateau_lift.
   _src=_ldt<LAKE_TOWN_REACH
   if _src.any():
    _dd=np.hypot(_p[:,None,0]-_p[None,:,0],_p[:,None,1]-_p[None,:,1]).astype(np.float32)
    _near=_dd<=LAKE_TOWN_COUPLE
    _rp=np.where(_tcap,*LAKE_TOWN_PLOT).astype(np.float32)
    _dd=np.maximum(_dd-_rp[:,None]-_rp[None,:]-LAKE_TOWN_TOUCH,0)   # plot edge to plot edge (less the rounding of the plots: plots this close touch and share the level)
    for _it in range(len(_h)):
     _lim=np.where(_near&_src[:,None],_h[:,None]+LAKE_TOWN_SLOPE*_dd,np.inf).min(0)
     _low=_lim<_h-1e-6
     if not _low.any():break
     _h=np.minimum(_h,_lim).astype(np.float32);_src=_src|_low
    del _dd,_near
   lake_chain_towns={rg for rg,s_ in zip(_treg,_src) if s_}   # TOWN SPEC: these take the lake datum floor (plateau targets)
   if lake_level_solved:
    # LAKE LEVEL (v3) reach towns: the towns whose cluster-mix level the switch can change, from the mix's own input:
    # every town whose input (town level at its centre) lies in the seed (lake_level_support), every town with a
    # non-zero float32 cluster weight to one of those, and the lake chain's closure from them (a chain source carries
    # its level LAKE_TOWN_COUPLE cells; the towns that can be chain sources: the closure of the lake-shore towns)
    _ti=np.array([bool(lake_level_support[cy,cx]) or any(np.isfinite(lake_level_town_s[n_].get(rg_,np.inf)) for n_ in lake_level_town_s) for (cx,cy),rg_ in zip(_tt,_treg)])
    _ti|=(_w[:,_ti]>0).any(1)
    _dc=np.hypot(_p[:,None,0]-_p[None,:,0],_p[:,None,1]-_p[None,:,1])<=LAKE_TOWN_COUPLE
    _cs=_ldt<LAKE_TOWN_REACH
    while True:
     _n2=_cs|_dc[_cs].any(0)
     if (_n2==_cs).all():break
     _cs=_n2
    while True:
     _n2=_ti|_dc[_ti&_cs].any(0)
     if (_n2==_ti).all():break
     _ti=_n2
    lake_level_reach_towns=[rg_ for rg_,k_ in zip(_treg,_ti) if k_];del _ti,_dc,_cs,_n2
   _num=np.zeros((TH,TW),np.float32);_den=np.zeros((TH,TW),np.float32);_gden=np.zeros((TH,TW),np.float32)
   for (_cx,_cy),_v,_cap in zip(_tt,_h,_tcap):
    _ys,_xs=slice(max(0,_cy-40),min(TH,_cy+41)),slice(max(0,_cx-40),min(TW,_cx+41))
    _g=np.exp(-((x[_ys,_xs]-_cx)**2+(y[_ys,_xs]-_cy)**2)/(2*10.**2)).astype(np.float32)
    _iw=_town_mix_weight(x[_ys,_xs]-_cx,y[_ys,_xs]-_cy,TOWN_MIX_CORE[0] if _cap else TOWN_MIX_CORE[1])   # TOWN SPEC (town ground mix)
    _num[_ys,_xs]+=_iw*_v;_den[_ys,_xs]+=_iw;_gden[_ys,_xs]+=_g
   # The cluster level enters the town level by real_w (it was relief*(1-cf)+level*cf, cf =
   # clearance*real_w, a third blend). After TOWN BLEND this is the floor blend plus the delta
   # (cluster level - town level)*clearance*real_w; at clearance 1 the cluster level as before.
   _cw=real_w*(_gden>1e-3)   # the Gaussian gate as before (TOWN SPEC town ground mix)
   town_level=np.where(physical,town_level*(1-_cw)+np.where(_gden>1e-3,_num/np.maximum(_den,1e-12),town_level)*_cw,town_level);del _gden
   del _p,_h,_w,_num,_den,_cw
 # --- TOWN BLEND --- the one blend of the town level into the relief on land (see the floor above).
 relief=np.where(physical,relief*(1-clearance_floor)+town_level*clearance_floor,relief);del town_level
 if lake_level_solved:
  # LAKE LEVEL product (v3: written after the town level solve's cluster mix, which supplies the reach): per lake the
  # level, the waterline field, the window towns' nearest plot distances and envelope caps, and the reach: the seed
  # (lake_level_support), the +-40-cell mix window of every reach town and of every town sharing a plot group with
  # one, grown by the slope projection's carry (a change of at most level - SEA_FLOOR travels that far at TOWN_GRADE)
  _rt=set(globals().get('lake_level_reach_towns',[]));_rg={town_groups[k_] for k_ in _rt}
  _rt|={t_['region'] for t_ in plan['regions'] if town_groups[t_['region']] in _rg}
  _reach=lake_level_support.copy()
  for t_ in plan['regions']:
   if t_['region'] in _rt:
    _cx,_cy=int(t_['town'][0]*2),int(round((t_['town'][1]+t_['town'][0]%2*.5)*2))
    _reach[max(0,_cy-40):min(TH,_cy+41),max(0,_cx-40):min(TW,_cx+41)]=True
  _ll_reach={}
  for _ll_name in sorted(lake_level_solved):
   _e=lake_level_solved[_ll_name];_lv=_e['level'];_ll_e=lake_level_lakes[_ll_name][1]
   _mg=int(np.ceil((_lv-SEA_FLOOR)/(TOWN_GRADE*DX/2)))
   _r=fastops.max_filter_mask(_reach,2*_mg+1);_ry0,_ry1,_rx0,_rx1=_lake_level.reach_window(_r)
   _ll_reach[_ll_name]=_r[_ry0:_ry1,_rx0:_rx1];del _r
   _e.update({'reach_towns':sorted(_rt),'reach_window_tile':[_ry0,_ry1,_rx0,_rx1],'reach_cells':int(_ll_reach[_ll_name].sum()),'reach_margin_cells':_mg,
              'town_plots':{k_:{'nearest_plot_s':round(v_,4),'cap_over_level':round(_lake_level.town_cap(_lv,v_,_ll_e)-_lv,4)} for k_,v_ in sorted(lake_level_town_s[_ll_name].items())}})
  del _reach,_rt,_rg
  import hashlib as _hl_ll
  _lake_level.write_product(A,lake_level_solved,{'map_sha256':_hl_ll.sha256(b).hexdigest(),'spec_sha256':_ll_spec_sha,'canvas':_frame.canvas_stamp(),
                            'water_plane':LAKE_WATER,'writer':'polish_campaign_terrain.py (LAKE LEVEL)'},{k_:v_[0] for k_,v_ in lake_level_sdf.items()},_ll_reach);del _ll_reach
  for _ll_name in sorted(lake_level_solved):print('Lake level reach',_ll_name,{k_:lake_level_solved[_ll_name][k_] for k_ in ('reach_window_tile','reach_cells','reach_margin_cells')},len(lake_level_solved[_ll_name]['reach_towns']),'towns',flush=True)
 # --- LAKE LEVEL SHAPE --- (v2) the one writer of the ground round a level lake: bank and bed from the signed distance
 # to its waterline (lake_level.shaped_ground: the ground crosses the level on the waterline, low banks within
 # ring_width, a bed of the spec depth inside), on land and lake cells alike; beyond ring_width + blend_width the
 # relief is untouched. After the town blend, so no later tile-level writer but the river carve (whose level eases to
 # the lake's level at the mouth) touches it; the projection keeps it through its floors and ceilings.
 # v4: this tile-grid evaluation is what the tile-grid readers (river carve, variant relief) see; the LF gets the same
 # shape evaluated per LF pixel (LAKE LEVEL LF SHAPE in the variant loop replaces the resize of this one), so the
 # window's tile relief before and after it is kept (lake_level_shape_tiles: name -> (pre, shaped, outer)).
 for _ll_name in sorted(lake_level_sdf):
  _,(_wy0,_wy1,_wx0,_wx1),_ll_st=lake_level_sdf[_ll_name]
  _ll_pre=relief[_wy0:_wy1,_wx0:_wx1].copy()
  _ll_h,_ll_w=_lake_level.shaped_ground(_ll_pre,_ll_st,lake_level_solved[_ll_name]['level'],lake_level_lakes[_ll_name][1])
  _ll_win=relief[_wy0:_wy1,_wx0:_wx1];np.copyto(_ll_win,_ll_h.astype(np.float32),where=(_ll_w>0)&outer[_wy0:_wy1,_wx0:_wx1])
  lake_level_shape_tiles[_ll_name]=(_ll_pre,_ll_win.copy(),outer[_wy0:_wy1,_wx0:_wx1].copy());del _ll_h,_ll_w,_ll_win,_ll_st,_ll_pre
 # --- RIVER CHANNELS --- frontier rivers (apply_rivers.py: tw3k_riv_sea_* sea hexes) the way vanilla
 # draws the Yellow River and Yangtze: a channel carved under a river water level that follows the
 # banks, and that level written into lf_sea_height_map (sync_campaign_sea_height_texture.py reads
 # RIVER_LEVEL_FILE). Vanilla's sea height map is raised along its rivers (Yellow River up to ~10.7,
 # Yangtze ~1.6) over channels ~1.0 under the banks; ours stayed at -0.015 over a 1-hex gap the
 # coastal falloff barely lowers (coast ~.74 there): 96% of the river hexes stayed above the water,
 # only wide spots (bends, junctions, filled slivers) dipped under it and read as isolated flat
 # pits, squares/diamonds on the ~1-hex coarse grid (live 2048m: Amur, Ganges, Irtysh, Mekong).
 # r15: the water edge is a smooth curve (river_channels.py: distance to the smoothed centre line of the
 # river hex centres, half-width growing downstream and varying with noise), not the step distance from
 # the river's 2x2-cell hex bricks, which drew a regular half-hex sawtooth and a constant width (offline
 # r14 Amur, Mekong, Funan). apply_rivers routes the rivers round the towns as vanilla does (no town cuts
 # any more), so the water runs on past a town to its edge; links only join pieces of one river cut
 # elsewhere and run pieces that end on land near the coast on to the water (visual links over land hexes;
 # map.hex unchanged).
 # The carve itself is river_channels.carve (constants RIVER_SLOPE .. RIVER_MOUTH there); it also returns the
 # bed distance and level that RIVER BEDS (river_channels.carve_beds, after the slope projection below) cuts
 # the bed under the water with: the projection's grade held the channel too shallow (live 2048p).
 import river_channels
 RIVER_LEVEL_FILE='analysis/river-water-level.npy'
 relief_pre_carve=relief.copy()   # --- RIVER RIM --- the variant loop carves the carried rim of this again
 # LAKE LEVEL: near a lake with a level of its own the river level eases to that level (not LAKE_WATER) at the
 # mouth, back to LAKE_WATER over RIVER_MOUTH (river_channels.carve lake_levels; None off: as before)
 river_lake_levels=None
 if lake_level_solved:
  _rlf=np.full((TH,TW),LAKE_WATER,np.float64);_rlz=np.zeros((TH,TW),bool)
  for _ll_name in sorted(lake_level_solved):
   _ll_m=lake_level_lakes[_ll_name][0]
   _ll_t=river_channels.mouth_ease(_ll_m,fastops.max_filter_mask(_ll_m,4*river_channels.RIVER_MOUTH+1),river_channels.RIVER_MOUTH)
   _ll_z=_ll_t<1;_rlf[_ll_z]=np.maximum(_rlf[_ll_z],LAKE_WATER+(lake_level_solved[_ll_name]['level']-LAKE_WATER)*(1-_ll_t[_ll_z].astype(np.float64)));_rlz|=_ll_z
  river_lake_levels=(_rlf,_rlz);del _rlf,_rlz,_ll_m,_ll_t,_ll_z
 relief,river_level,river_bed,report_river=river_channels.carve(relief,names,ids,cells,land,physical,outer,hy,hx,q,r,(NQ,NR),A/'work/final-map/river-channels.json',LAKE_WATER,fastops.grow_steps,lake_levels=river_lake_levels)
 if report_river is not None:print('River channels',json.dumps(report_river),flush=True)
 def write_river_level(level,primary=False):
  # Temp + replace (same .npy bytes), with a stamp of the map.hex (b, read above) and canvas it was drawn
  # on: sync_campaign_sea_height_texture refuses a level left from another map or canvas.
  if LOOP_ONLY and not primary:return   # a --loop-only run writes it from its primary variant's rim carve
  import hashlib;_rl=A/RIVER_LEVEL_FILE;_rs=_rl.with_suffix('.json')
  with open(_rl.with_name(_rl.name+'.tmp'),'wb') as _f:np.save(_f,level)
  _rs.with_name(_rs.name+'.tmp').write_text(json.dumps({'map_sha256':hashlib.sha256(b).hexdigest(),'canvas':_frame.canvas_stamp()}),encoding='utf-8')
  os.replace(_rl.with_name(_rl.name+'.tmp'),_rl);os.replace(_rs.with_name(_rs.name+'.tmp'),_rs)
 write_river_level(river_level)
 oasis*=physical|_lake;alpine*=physical|_lake;del low_valley   # --- USR3 BOSTEN PATCH --- (was physical)
 if _bp.tundra_oasis_off_on() and REAL_TERRAIN.exists() and not _memo_hit:   # [TUNDRA OASIS OFF] no desert oasis / alpine belts on tundra, snow, arctic grass
  with np.load(REAL_TERRAIN) as _gz:
   _olat=np.zeros((TH,TW),np.float32);_ox0,_ox1,_oy0,_oy1=(int(v) for v in _gz['box']);_olat[_oy0:_oy1,_ox0:_ox1]=_gz['lat']
  _noa,_noal=_bp.no_oasis_masks(real_bio_eco,_bnames,_olat,q,r,(NQ,NR));del _olat
  print('Tundra oasis off: oasis tiles cleared',int(((oasis>0)&_noa).sum()),'alpine belt tiles cleared',int(((alpine>0)&_noal).sum()),flush=True)
  oasis=np.where(_noa,0,oasis).astype(np.float32);alpine=np.where(_noal,0,alpine).astype(np.float32);del _noa,_noal
 # --- HEHUANG PATCH (hehuang_terrain.py) --- lake-shore meadow ring
 if HH is not None:oasis=np.maximum(oasis,HH.oasis()*physical)
 # --- OASIS GROUND (oasis_ground.oasis_seeds) --- real oases (Turpan basin, Hami, Dunhuang, Guazhou,
 # the Tarim rim) seed the oasis belts: temperate donor + OASIS_REMAP, no GROUND MATERIALS, oasis fields.
 # The belts above follow rivers only, and these oases stood as bare sand (Jushi diagnosis 2026-10-24).
 if REAL_TERRAIN.exists():
  import oasis_ground as _og
  with np.load(REAL_TERRAIN) as _gz:
   _slon=np.zeros((TH,TW),np.float32);_slat=np.zeros((TH,TW),np.float32);_x0,_x1,_y0,_y1=(int(v) for v in _gz['box'])
   _slon[_y0:_y1,_x0:_x1]=_gz['lon'];_slat[_y0:_y1,_x0:_x1]=_gz['lat']
  oasis=np.maximum(oasis,_og.oasis_seeds(real_elev,_slon,_slat,real_river,q,r,(NQ,NR))*real_gw*physical).astype(np.float32)
  oasis,_isl=_og.clean_islands(oasis,_slon,_slat);print('Oasis islands removed (Turpan basin)',_isl,flush=True);del _slon,_slat   # [TURPAN3] no stray specks
 # Salt-pan windows (salt_pools.py) are planned here, before the height projection, so the
 # ground under them is levelled in the variant loop; the cache stage stamps exactly these.
 # A window needs edit alpha >= EDIT_ALPHA (inside the native map only Xihai's town edit) and
 # stays off town plots and aprons (their plateau pins would tilt the pan).
 import salt_pools,strategic_ink
 _salt_shape=(salt_pools.CLUSTER[1]-salt_pools.CLUSTER[0],salt_pools.CLUSTER[3]-salt_pools.CLUSTER[2])
 _salt_sites=[(2*r_['town'][0],2*r_['town'][1]+r_['town'][0]%2) for r_ in plan['regions'] if str(r_.get('battle_layout','')).startswith('resource_han_salt') and r_.get('introduced_in_bulk')]
 # --- SALT CRUST (salt_pools.crust_mask) --- the classify salt_crust range (with the Natural Earth salt raster of
 # REAL_TERRAIN clipped to it) joins the playa hexes; playa/crust windows stay clear of water, roads, towns, real rivers/lakes
 # SALT_CRUST off: user package 2026-10-26: the crust windows need irregular placement and edge blending (they
 # showed as brown-framed squares on a regular grid at Qarhan and south-east of Lop Nur). TW3K_SALT_CRUST=1 turns
 # them on; off, the salt pans are planned as before SALT CRUST (playa and gameplay site windows only).
 SALT_CRUST=os.environ.get('TW3K_SALT_CRUST','0')=='1'
 _salt_flat=strategic_ink.load()[2][hy,hx];_salt_clear=None;_salt_crust_rep=None;_scr=None
 if REAL_TERRAIN.exists() and SALT_CRUST:
  with np.load(REAL_TERRAIN) as _sz:
   _x0,_x1,_y0,_y1=(int(v) for v in _sz['box']);_sw_=(slice(_y0,_y1),slice(_x0,_x1))
   _slon=np.zeros((TH,TW),np.float32);_slat=np.zeros((TH,TW),np.float32);_slon[_sw_]=_sz['lon'];_slat[_sw_]=_sz['lat']
   _sne=np.zeros((TH,TW),bool);_slk=np.zeros((TH,TW),bool)
   if 'salt' in _sz.files:_sne[_sw_]=_sz['salt']
   if 'lake' in _sz.files:_slk[_sw_]=_sz['lake']
  _scr=salt_pools.crust_mask(_slon,_slat,real_elev)
  _sne&=_scr   # the Natural Earth salt raster only inside the crust range (polygons below max_elev), never map-wide
  _salt_crust_rep={'crust_cells':int(_scr.sum()),'ne_salt_cells':int(_sne.sum()),'added_cells':int((((_scr|_sne)&~_salt_flat)).sum())}
  _salt_flat=_salt_flat|_scr|_sne;_salt_clear=salt_pools.clear_mask(cells,hy,hx,real_river,_slk)
  _salt_polys=salt_pools.crust_by_polygon(_slon,_slat);del _slon,_slat,_sne,_slk
 # --- end SALT CRUST
 salt_windows,_n_sites,_n_skipped=salt_pools.plan(_salt_flat,(alpha>=salt_pools.EDIT_ALPHA)&~full_town_edit,physical,_salt_shape,_salt_sites,relief,_salt_clear,_scr);del _scr
 if _salt_crust_rep is not None:   # --- SALT CRUST --- playa/crust windows (not the site windows, last) per crust polygon (centre cell)
  _salt_crust_rep['windows_by_polygon']={n_:sum(1 for R_,C_ in salt_windows[:len(salt_windows)-_n_sites] if m_[R_+_salt_shape[0]//2,C_+_salt_shape[1]//2]) for n_,m_ in _salt_polys.items()};del _salt_polys
  print('Salt crust:',json.dumps(_salt_crust_rep),flush=True)
 salt_scale=np.clip(alpha/salt_pools.EDIT_ALPHA,0,1).astype(np.float32)
 if not LOOP_ONLY:(A/salt_pools.WINDOWS_FILE).write_text(json.dumps({'shape':list(_salt_shape),'windows':[list(map(int,w_)) for w_ in salt_windows],'site_windows':_n_sites,'playa_windows_skipped':_n_skipped,'salt_crust':_salt_crust_rep}),encoding='utf-8')
 print('Salt-pan windows',len(salt_windows),'site windows',_n_sites,'of',len(_salt_sites),'playa windows skipped',_n_skipped,flush=True)
 if _n_sites<len(_salt_sites):print('WARNING: salt sites without a pan window',len(_salt_sites)-_n_sites,flush=True)
report={'kind':'geographic-terrain-polish','logical_grid':[W,H],'tile_grid':[TW,TH],'sources':[{'title':'NASA Tarim basin terrain','url':'https://science.nasa.gov/earth/earth-observatory/taklimakan-desert-dust-storm-45714/'},{'title':'NASA Cambodia and Vietnam','url':'https://modis.gsfc.nasa.gov/gallery/individual.php?db_date=2016-01-12'}],'geography':'Stylized mountain belts and valleys fitted to the existing game atlas, not a georeferenced DEM.','features':[{'name':n,'hex_polyline':p,'height':h,'width':w} for n,p,h,w in features],'valleys':[{'name':n,'hex_polyline':p,'width':w,'water_verified':False} for n,p,w in valleys],'variants':[],'live_verified':False}
if not CACHE_ONLY and lake_level_solved:report['lake_levels']={k_:{kk:vv for kk,vv in v_.items() if kk!='cells_yx'} for k_,v_ in lake_level_solved.items()}   # LAKE LEVEL (product: analysis/lake-water-levels.json)
if not CACHE_ONLY:
 requested_alpha=alpha.copy();coverage_alpha=np.zeros_like(alpha)
# Overflow land meshes (fill_polished_mesh_gaps, mesh bake 3.2.1: a model's chunks past 65k vertices) are graph output,
# never polish input nor part of the snapshot below: the model set is the game packs' land meshes plus the snapshot's
# canvas meshes, and any other land_mesh file in the stage (the previous run's overflow) is dropped first. Seeded into
# the snapshot, the 10-30 overflow files came back as models on the next run (land_mesh_budget 15.5M triangles).
for p in S.glob('terrain/campaigns/*/global_meshes/land_mesh_*'):
 rel=p.relative_to(S).as_posix()
 if rel not in idx and not (BASE/rel).exists():p.unlink()
# Immutable snapshot before any mutation, also making repeated tuning reproducible.
for p in S.rglob('*'):
 if p.is_file() and p.relative_to(S).as_posix().startswith(('terrain/campaigns/','campaign_maps/')):
  target=BASE/p.relative_to(S)
  if not target.exists() and _frame.seed_from_stage_allowed(BASE):target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
print('Baseline retained',BASE,flush=True)
_mark('global: all pre-variant setup')
if '--cache-only' not in sys.argv and not MERGE:
 for variant in POLISH_VARIANTS:
  alpha=requested_alpha.copy();alpha[alpha<=1e-7]=0
  pre=f'terrain/campaigns/{variant}/';print('Polishing',variant,flush=True)
  src=base(pre+'lf_height_map.dds');oh,ow=struct.unpack_from('<2I',src,12);old=continuous_canvas_base(np.frombuffer(src,dtype='<u2',offset=128).reshape(oh,ow));old_south=old[::-1];_lf_head=src[:128];del src   # only the header is written back (the pixels are 0.5 GB)
  low=resize(old_south,(TW,TH))/65535;oldheight=LEVEL+(low-NORM)*GAIN
  # Continental town clearings retain the local elevation. LEVEL is only the
  # reference for newly authored exterior terrain, never a plateau datum.
  native_rectangle=(x>=_frame.NATIVE_TILE_X0)&(x<_frame.NATIVE_TILE_X1)&(y>=_frame.NATIVE_TILE_Y0)&(y<_frame.NATIVE_TILE_Y1)
  # Canvas relief beside the native map starts from the native height inside the
  # vanilla rim ramp (the old map edge sloped down) and fades into authored
  # relief; the 8-cell alpha ring then blends native land into that, not into a
  # low exterior plateau. Meeting LEVEL directly left an 8-cell wall on the north edge.
  native_land_cells=native_rectangle&(physical|native_river)&native_land_hex[hy,hx]
  # Carry a smoothed native height: carrying raw cells extruded every ridge and dune
  # that ends at the old border into regular streaks perpendicular to it.
  carry_seed=native_land_cells&(edge>=NATIVE_RIM)
  weight=box_blur(carry_seed.astype(np.float32),CARRY_SMOOTH);smooth_old=box_blur(np.where(carry_seed,oldheight,0).astype(np.float32),CARRY_SMOOTH)/np.maximum(weight,1e-3)
  carried,carried_steps=propagate_nearest(smooth_old.astype(np.float32),carry_seed,NATIVE_RIM+RELIEF_HOLD+max(RELIEF_FADE,FRONT_FADE)+FRONT_MEANDER)
  # Far from the border the carried height comes from a much wider blur: a
  # nearest-value carry over the wide plateau front drew every edge ridge out
  # as a straight streak (Dunhuang, 2048 canvas).
  weight_far=box_blur(carry_seed.astype(np.float32),CARRY_SMOOTH*4);smooth_far=box_blur(np.where(carry_seed,oldheight,0).astype(np.float32),CARRY_SMOOTH*4)/np.maximum(weight_far,1e-3)
  carried_far,_=propagate_nearest(smooth_far.astype(np.float32),carry_seed,NATIVE_RIM+RELIEF_HOLD+max(RELIEF_FADE,FRONT_FADE)+FRONT_MEANDER)
  _mark('variant:   carry propagate')
  far=np.clip((carried_steps.astype(np.float32)-NATIVE_RIM)/(FRONT_FADE*.5),0,1);far=far*far*(3-2*far)
  carried=carried*(1-far)+carried_far*far
  # Read the carried height at a noise-displaced cell: a nearest-value carry
  # extruded the straight native edge into a straight plateau rim (Dunhuang,
  # 2048 canvas), as it did the edge climates (native_campaign_climate).
  # Zero at the native border (both sides agree), full over the fade band.
  _shift=CARRY_SHIFT*np.clip((carried_steps.astype(np.float32)-NATIVE_RIM)/RELIEF_FADE,0,1)
  _sx=np.clip(np.rint(x+_shift*gpu_noise.fbm(q,r,(NQ,NR),16,3,seed=83)),0,TW-1).astype(np.int32)
  _sy=np.clip(np.rint(y+_shift*gpu_noise.fbm(q,r,(NQ,NR),16,3,seed=84)),0,TH-1).astype(np.int32)
  carried=carried[_sy,_sx];del _sx,_sy
  # Where the native edge stands far above the authored canvas (the Tibetan plateau
  # over the Tarim basin), the transition is a narrow ridged mountain front, like
  # the Kunlun north face, so basin towns keep a level floor instead of sitting on
  # a long smooth incline. Small height gaps keep the long gentle fade.
  carried_gap=np.clip(carried-relief,0,None);steep_front=np.clip((carried_gap-1)/2,0,1)
  # Every transition meanders (0..FRONT_MEANDER cells) so no slope runs parallel to the old border.
  # Headlands and bays about 15 hexes apart, kept away from authored town plots.
  meander=FRONT_MEANDER*np.clip(.5+.6*gpu_noise.fbm(q,r,(NQ,NR),22,3,seed=7),0,1)
  meander*=np.clip((town_distance-8*DX)/(12*DX),0,1)
  fade_cells=RELIEF_FADE*(1-steep_front)+FRONT_FADE*steep_front;hold_cells=RELIEF_HOLD*(1-steep_front)+meander
  carried_fade=np.clip((carried_steps.astype(np.float32)-NATIVE_RIM-hold_cells)/fade_cells,0,1);carried_fade=carried_fade*carried_fade*(3-2*carried_fade)
  ridges=gpu_noise.ridged(q,r,(NQ,NR),10,4,seed=37)
  front=4*carried_fade*(1-carried_fade)*steep_front*carried_gap*FRONT_RIDGE*ridges
  relief_v=np.where(physical|native_river,(SEA_FLOOR+(carried-SEA_FLOOR)*shore)*(1-carried_fade)+relief*carried_fade+front*shore,relief)
  # --- RIVER RIM --- RIVER CHANNELS again on the carried rim. By the native rim the carried native height
  # (carried_fade < 1) replaces the authored relief that the setup carve took its bank heights and river level
  # from: the channel sat under a raised plateau, and its bed and centre pull cut walls into the banks (or,
  # tapered, left the centres dry) where the frontier rivers run on across the west border (integ 458ea872).
  # The rim relief of the pre-carve relief is carved here: away from the rim it is the setup carve's input, so
  # the result is the same there. The river level file and the beds come from this carve.
  if river_bed is not None:
   _rim=np.where(physical|native_river,(SEA_FLOOR+(carried-SEA_FLOOR)*shore)*(1-carried_fade)+relief_pre_carve*carried_fade+front*shore,relief_pre_carve).astype(np.float32)
   relief_v,river_level,river_bed,report_river=river_channels.carve(_rim,names,ids,cells,land,physical,outer,hy,hx,q,r,(NQ,NR),A/'work/final-map/river-channels.json',LAKE_WATER,fastops.grow_steps,lake_levels=river_lake_levels);del _rim
   print('River channels (carried rim)',json.dumps(report_river),flush=True)
   write_river_level(river_level,primary=variant==ALL_VARIANTS[0])
  # Native land takes its own heights beyond the rim: blended over RIM_BLEND cells
  # from a meandering start, not switched at exactly NATIVE_RIM cells (that hard
  # switch was a straight crease at hex column 668, offline render 2026-10-20).
  _rim_start=NATIVE_RIM*(.5+.8*np.clip(.5+.7*gpu_noise.fbm(q,r,(NQ,NR),11,3,seed=88),0,1))-RIM_BLEND/2
  _rim_t=np.clip((edge-_rim_start)/RIM_BLEND,0,1);_rim_t=_rim_t*_rim_t*(3-2*_rim_t)
  local_relief=np.where(native_land_cells,relief_v*(1-_rim_t)+oldheight*_rim_t,relief_v)
  # --- HEHUANG PATCH (hehuang_terrain.py) --- real relief (DEM + vanilla datum offset, lake bowl, town floors) by weight
  if HH is not None:local_relief=HH.blend(local_relief)
  plateau_targets={};town_reference={}
  for town in plan['regions']:
   tq,tr=town['town'];cx=int(tq*2);cy=int((tr+tq%2*.5)*2)
   # Towns just outside the old border sit on the carried native ground; levelling
   # them to the exterior relief cut a terrace wall into the plateau edge.
   # Only towns OUTSIDE the native map (edge<0) near its border take the native
   # datum; a former-sea island inside the native rectangle must not, or it sinks
   # to the seabed median (Zhouhu, Wa Nu).
   town_edge=edge[min(cy,TH-1),min(cx,TW-1)]
   inherited=bool(native_land_hex[tr,tq]) or (-NEAR_BORDER_TOWN<=town_edge<0)
   window=(slice(max(_frame.NATIVE_TILE_Y0,cy-28),min(_frame.NATIVE_TILE_Y1,cy+29)),slice(max(_frame.NATIVE_TILE_X0,cx-28),min(_frame.NATIVE_TILE_X1,cx+29)))
   # Median over native LAND only: seabed cells pulled coastal plateaus under water.
   neighborhood=oldheight[window][native_land_cells[window]] if inherited else None
   target=float(np.median(neighborhood)) if inherited and neighborhood.size else float(relief_v[cy,cx])
   # A town plot on land never sits below the coastal datum of vanilla shore towns.
   # Lake-shore towns (within LAKE_TOWN_REACH) may stand lower: the lake water plane is ~0.
   # TOWN SPEC (lake cap continuity): a town the lake cap carried to (lake_chain_towns) takes the lake datum floor too;
   # MIN_TOWN_HEIGHT put the Wusun pasture back at 1.0 over the chain's .78 and lifted the touching capital plot .23.
   if land[tr,tq]:target=max(target,MIN_LAKE_TOWN_HEIGHT if lake_dist[min(cy,TH-1),min(cx,TW-1)]<LAKE_TOWN_REACH or town['region'] in lake_chain_towns else MIN_TOWN_HEIGHT)
   # --- HEHUANG PATCH (hehuang_terrain.py) --- patch towns stand on the levelled real relief
   if HH is not None:target=HH.town_target(town['region'],target)
   group=town_groups[town['region']];town_reference[town['region']]=target
   plateau_targets.setdefault(group,[]).append(target)
  plateau_targets={group:float(np.median(values)) for group,values in plateau_targets.items()}
  # LAKE LEVEL (v3): a plot group with a plot in a level lake's window stands at most at that lake's bank envelope over
  # its nearest plot pixel (lake_level.town_cap, the cluster mix's cap): the plateau reads the lake product
  if lake_level_solved:
   for t_ in plan['regions']:
    _g=town_groups[t_['region']]
    for n_ in sorted(lake_level_solved):
     _c=_lake_level.town_cap(lake_level_solved[n_]['level'],lake_level_town_s[n_].get(t_['region']),lake_level_lakes[n_][1])
     if _c<plateau_targets[_g]:plateau_targets[_g]=_c
  _mark('variant:   relief/front/town targets')
  # Protected vanilla ground is fixed; a plot that cannot be levelled between
  # its slopes is reported (all at once) instead of re-shaping vanilla land.
  proposed=oldheight*(1-alpha)+local_relief*alpha
  # The full-resolution arrays below: resizes in column bands, elementwise work in row blocks
  # (resize_high, _resize_bands, _rows; same bytes as the whole-array operations).
  high_old=np.empty(old_south.shape,np.float32);_rows(lambda s:np.copyto(high_old[s],LEVEL+(old_south[s].astype(np.float32)/65535-NORM)*GAIN),oh)
  # Build the projection's worker pool and edge-cost coordinates (they depend only on
  # high_old) in the background while the single-threaded resizes below run.
  # The base's step on the native border (vanilla sea floor against the flat canvas
  # plateau) is artificial: across it, between free water pixels, the projection
  # gets the plain grade, not that step as a slope allowance. With it, the coastal
  # ramp of new land near the border (Kyushu, whole-canvas real terrain) ended in a
  # ~0.8 cliff against the native sea floor exactly on the border (seam audit lf_cliff).
  high_alpha=resize_high(alpha,(ow,oh))
  high_native=_resize_bands(native_plots.astype(np.uint8),(ow,oh),Image.Resampling.NEAREST,post=lambda b:b>0)
  _hphys=_resize_bands(physical.astype(np.uint8),(ow,oh),Image.Resampling.NEAREST,post=lambda b:np.asarray(b,bool));_hfree=np.empty((oh,ow),bool)
  _rows(lambda s:np.copyto(_hfree[s],(~_hphys[s])&(high_alpha[s]>1e-7)&~high_native[s]),oh);del _hphys
  _plain_x=np.zeros((oh,ow-1),bool);_plain_z=np.zeros((oh-1,ow),bool)
  for _e in (_frame.NATIVE_TILE_X0*ow//TW,_frame.NATIVE_TILE_X1*ow//TW):_plain_x[:,_e-1]=_hfree[:,_e-1]&_hfree[:,_e]
  for _e in (_frame.NATIVE_TILE_Y0*oh//TH,_frame.NATIVE_TILE_Y1*oh//TH):_plain_z[_e-1]=_hfree[_e-1]&_hfree[_e]
  del _hfree
  import envelope_mp;_env_future=_POOL.submit(envelope_mp.Envelope,high_old,TOWN_GRADE*(DX/2*TW/ow),TOWN_GRADE*(DX*ZS/2*TH/oh),plain_x=_plain_x,plain_z=_plain_z)
  _d=resize_high(proposed-oldheight,(ow,oh));high_proposed=np.empty((oh,ow),np.float32);_rows(lambda s:np.add(high_old[s],_d[s],out=high_proposed[s]),oh);del _d
  # --- LAKE LEVEL LF SHAPE --- (v4) the bank and bed of a level lake on the LF grid: the resize above carries the tile
  # shape (LAKE LEVEL SHAPE) as a bilinear patchwork of tile values, axis-biased at the level crossing (v3b: share .73
  # against the designed outline's .52); lake_level.lf_shape_delta replaces it in the window by the same shaped_ground
  # evaluated per LF pixel from the product's LF signed distance (bilinear is linear: every other writer's delta stays).
  # Scaled by the relief's weight in proposed (alpha x carried fade x native rim x Hehuang, 1 at Bosten; printed).
  for _ll_name in sorted(globals().get('lake_level_shape_tiles',{})):
   _ll_s,(_wy0,_wy1,_wx0,_wx1),_=lake_level_sdf[_ll_name];_ll_pre,_ll_tsh,_ll_keep=lake_level_shape_tiles[_ll_name]
   _ll_sz=(_ll_s.shape[1],_ll_s.shape[0]);_ws=(slice(_wy0,_wy1),slice(_wx0,_wx1))
   _ll_d=_lake_level.lf_shape_delta(_ll_pre,_ll_tsh,_ll_keep,_ll_s,lake_level_solved[_ll_name]['level'],lake_level_lakes[_ll_name][1],lambda a_:resize_high(a_,_ll_sz))
   _ll_k=alpha[_ws]*np.where((physical|native_river)[_ws],carried_fade[_ws],1)*np.where(native_land_cells[_ws],1-_rim_t[_ws],1)
   if HH is not None:_ll_k=_ll_k*(1-np.asarray(HH.hw)[_ws])   # HEHUANG PATCH blend: local_relief (1 - hw) + DEM hw
   _ll_d*=resize_high(_ll_k.astype(np.float32),_ll_sz)
   _py0,_px0=_wy0*_lake_level.FINE,_wx0*_lake_level.FINE;_ps=(slice(_py0,_py0+_ll_s.shape[0]),slice(_px0,_px0+_ll_s.shape[1]))
   high_proposed[_ps]+=_ll_d
   print('Lake level LF shape',_ll_name,{'relief_weight_min':round(float(_ll_k.min()),6),'delta_px':int((_ll_d!=0).sum()),'delta_abs_max':round(float(np.abs(_ll_d).max()),4)},flush=True)
   del _ll_s,_ll_pre,_ll_tsh,_ll_keep,_ll_d,_ll_k
  # At the old canvas rim, blend absolute heights. Adding a resized delta
  # preserves discontinuities in the full-resolution native edge pixels.
  # (The absolute blend high_old*(1-high_alpha)+resized local_relief*high_alpha is evaluated on the seam band only.)
  high_edge=resize_high(edge,(ow,oh));seam_band=np.empty((oh,ow),bool);_rows(lambda s:np.copyto(seam_band[s],(high_edge[s]>=-8)&(high_edge[s]<=8)),oh)
  _ix=np.flatnonzero(seam_band);_ha=high_alpha.ravel()[_ix]
  high_proposed.ravel()[_ix]=high_old.ravel()[_ix]*(1-_ha)+resize_high(local_relief,(ow,oh)).ravel()[_ix]*_ha
  del high_edge,seam_band,_ix,_ha
  # Level the ground under the salt pans (salt_pools.level_field): the rigid flat pan meshes
  # cut through sloped ground as black sheets and torn rims (live 2048k Yangtong, Xihai).
  # The projection below keeps the flat (its aprons stay under TOWN_GRADE) and the fixed pixels.
  _sw,_st,salt_levels=salt_pools.level_field(salt_windows,_salt_shape,proposed,salt_scale)
  # high_proposed+=(resize(_st)-high_proposed)*resize(_sw), the resizes only in the columns whose bilinear
  # taps (source columns within 2 of a column/4) reach a nonzero weight or target; elsewhere both
  # resize to +0.0, so the same update runs with zeros.
  _scn=np.flatnonzero(_sw.any(axis=0)|_st.any(axis=0));_sspans=[];_sf=ow//TW
  for _scol in (np.split(_scn,np.flatnonzero(np.diff(_scn)>8)+1) if _scn.size and _sf*TW==ow else [[0,TW]] if _scn.size else []):
   _sa,_sb=(max(0,_sf*(int(_scol[0])-3)),min(ow,_sf*(int(_scol[-1])+4))) if _sf*TW==ow else (0,ow)
   if _sspans and _sa<=_sspans[-1][1]:_sspans[-1][1]=max(_sspans[-1][1],_sb)
   else:_sspans.append([_sa,_sb])
  _sz=np.float32(0);_sc0=0
  for _sa,_sb in _sspans+[[ow,ow]]:
   if _sa>_sc0:_shp=high_proposed[:,_sc0:_sa];_rows(lambda s:np.add(_shp[s],(_sz-_shp[s])*_sz,out=_shp[s]),oh)
   if _sb>_sa:
    _sww=_resize_bands(_sw,(ow,oh),Image.Resampling.BILINEAR,_sa,_sb);_stt=_resize_bands(_st,(ow,oh),Image.Resampling.BILINEAR,_sa,_sb);_shp=high_proposed[:,_sa:_sb]
    _rows(lambda s:np.add(_shp[s],(_stt[s]-_shp[s])*_sww[s],out=_shp[s]),oh);del _sww,_stt
   _sc0=_sb
  del _sw,_st,_scn,_sspans,_shp,_sz,_sc0,_sf
  high_owner=_resize_bands(flat_owner.astype(np.int32),(ow,oh),Image.Resampling.NEAREST,post=lambda b:b.astype(np.int16))
  _mark('variant:   high-res resizes')
  # Round the plot outlines: nearest-resized hex cells give staircase edges with a
  # bevel all round. Threshold a blurred plot mask; added pixels join the nearest plot.
  # Rounding adds fringe and trims apron pixels, never pixels of an owned authored
  # building-plot hex: on a coastal tip (land-only apron, sea on two or three sides)
  # the plot hex is a convex corner, and its dropped corner pixels, the town anchor
  # among them, sloped to the sea (Hanseong_resource_1 town at .58 against 1.0).
  plot_mask=high_owner>=0;high_plot=_resize_bands(authored_plot[hy,hx].astype(np.uint8),(ow,oh),Image.Resampling.NEAREST,post=lambda b:b>0)
  rounded_mask=(box_blur(box_blur(plot_mask.astype(np.float32),PLOT_ROUND),PLOT_ROUND)>=.5)|(plot_mask&high_plot)
  near_owner,_=propagate_nearest(high_owner,plot_mask,PLOT_ROUND*2);del _   # its full-resolution int32 distances (1 GB) lived on to the cache stage
  high_owner=np.where(rounded_mask&~high_native,near_owner,-1).astype(np.int16);del plot_mask,high_plot,rounded_mask,near_owner
  pitch_x,pitch_z=DX/2*TW/ow,DX*ZS/2*TH/oh
  _mark('variant: surface relief/carry/towns')
  print('Projecting actual DDS',ow,oh,variant,flush=True)
  _t_cts=_time.time();print('[projection-timing] solver start',flush=True)
  # The solver needs the alpha and native plots only as the fixed mask; it takes the proposed heights out of the list
  # and drops them once projected (mem-budget: 2 GB fewer at its commit peak; same arrays, same bytes).
  fixed_high=high_native|(high_alpha<=1e-7);del high_alpha,high_native;_proposed_box=[high_proposed];del high_proposed
  # HEIGHT BAKE lower bound L (lake shore, see shore_zone_tiles / shore_sd_tiles): the inland side of the smooth
  # waterline stays above the plane through the projection (half the shore slope: the relief's own profile sits
  # above it, so the floor binds only where the projection would pull the shore under); -inf elsewhere
  _floors=None
  if shore_sd_tiles is not None and shore_zone_tiles.any():
   _fl=resize_high(shore_sd_tiles,(ow,oh))   # smooth signed distance (tile cells, + inland), bilinear
   _zh=_resize_bands(shore_zone_tiles.astype(np.uint8),(ow,oh),Image.Resampling.NEAREST,post=lambda b:b>0)
   np.copyto(_zh,False,where=_fl<=0)
   np.minimum(_fl,np.float32(LAKE_SHORE_BAND),out=_fl);_fl*=np.float32(.5*LAKE_SHORE_SLOPE);_fl+=np.float32(LAKE_WATER+.005)
   np.copyto(_fl,np.float32(-np.inf),where=~_zh);del _zh
   _floors=[_fl];del _fl
  # HEIGHT BAKE upper bound U (lake tiles): every lake pixel just under the plane, whatever the shore beside it (a
  # shallow shelf next to a steep basin rim rose above the water through the projection's midpoint: Lop Nur west)
  _ceils=None
  if shore_sd_tiles is not None and _lake.any():
   # v21d: the ceiling follows the shelf (LAKE_CEIL_SLACK over it): at the plane alone the projection lifted the Lop Nur
   # bed from -.48 to -.12 (median) between steep shores, and the water read pale and uniform
   _bed=resize_high(np.nan_to_num(lake_bed_tiles,nan=LAKE_WATER),(ow,oh))
   _ceils=[np.where(_resize_bands(_lake.astype(np.uint8),(ow,oh),Image.Resampling.NEAREST,post=lambda b:b>0),np.minimum(np.float32(LAKE_WATER-LAKE_CEIL_DEPTH),_bed+np.float32(LAKE_CEIL_SLACK)),np.float32(np.inf)).astype(np.float32)];del _bed
  # LAKE LEVEL bounds (same interface as the v21 lake shore, which is off with it): from the waterline's signed
  # distance on the LF grid (lake_level.projection_bounds): the bed stays under the level inside it, the bank over
  # level + bank_floor and under level + the bank envelope outside it. v3: the town ground on the water side follows
  # the envelope too: an owner pixel (plot apron) under the water or where level + E(s) lies under its group's plateau
  # target is freed from the plateau and takes the envelope ceiling (the plot cells themselves keep their plateau: the
  # target is capped at E over the nearest plot pixel, so they never qualify).
  if lake_level_solved:
   _fl=np.full((oh,ow),-np.inf,np.float32);_cl=np.full((oh,ow),np.inf,np.float32);_lsb=np.float32(GAIN/65535)
   _gl=np.full(max(int(high_owner.max())+1,1),np.inf)
   for _g,_v in plateau_targets.items():_gl[_g]=_v
   for _ll_name in sorted(lake_level_solved):
    _ll_s,(_wy0,_wy1,_wx0,_wx1),_=lake_level_sdf[_ll_name];_ll_e=lake_level_lakes[_ll_name][1];_lv=lake_level_solved[_ll_name]['level']
    _py0,_px0=_wy0*_lake_level.FINE,_wx0*_lake_level.FINE;_ps=(slice(_py0,_py0+_ll_s.shape[0]),slice(_px0,_px0+_ll_s.shape[1]))
    _pp=np.repeat(np.repeat(authored_plot[hy[_wy0:_wy1,_wx0:_wx1],hx[_wy0:_wy1,_wx0:_wx1]],_lake_level.FINE,0),_lake_level.FINE,1)
    _ow=high_owner[_ps];_own=(_ow>=0)&~_pp
    _free=_own&((_ll_s<0)|(_lv+_lake_level.town_envelope(np.maximum(_ll_s,0),_ll_e)<_gl[np.maximum(_ow,0)]))
    print('Lake level freed owner pixels',_ll_name,int(_free.sum()),flush=True)
    _ow[_free]=-1
    _f,_c=_lake_level.projection_bounds(_ll_s,_lv,_ll_e,_lsb,town=_free)
    np.maximum(_fl[_ps],_f,out=_fl[_ps]);np.minimum(_cl[_ps],_c,out=_cl[_ps]);del _f,_c,_ow,_ll_s,_pp,_own,_free
   _floors=[_fl];_ceils=[_cl];del _fl,_cl,_ll_e,_gl
  high_surface,plateau_levels,approach_report=constrained_town_surface(high_old,_proposed_box,high_owner,fixed_high,None,floors=_floors,ceils=_ceils,grade=TOWN_GRADE,pixel_pitch=(pitch_x,pitch_z),plateau_targets=plateau_targets,smooth=APRON_SMOOTH,env_future=_env_future,group_names={g:[t['region'] for t in plan['regions'] if town_groups[t['region']]==g] for g in set(town_groups.values())})
  print(f'[projection-timing] solver total: {_time.time()-_t_cts:.1f}s',flush=True)
  approach_report['plateau_datum']='median immutable native land within 14 hexes; exterior and new islands on former sea use authored relief'
  encoded=np.rint(np.clip(NORM+(high_surface-LEVEL)/GAIN,0,1)*65535).astype('<u2');encoded[fixed_high]=old_south[fixed_high];assert np.array_equal(encoded[fixed_high],old_south[fixed_high])
  quantized=LEVEL+(encoded.astype(np.float32)/65535-NORM)*GAIN;errors=maximum_edge_constraint_error(quantized,high_old,pitch_x,pitch_z);quantization_tolerance=2*GAIN/65535+1e-4
  if max(errors.values())>quantization_tolerance:raise RuntimeError(f'Quantized DDS introduced a height step: {errors}')
  approach_report['quantized_dds_edge_constraint_error']=errors;approach_report['dds_fixed_pixels_preserved']=int(fixed_high.sum());approach_report['authority']='final quantized full-resolution DDS'
  # --- LAKE LEVEL BED (lake_level.cut_bed) --- (v5, user 2026-10-31 option a) the bed of a level lake (s < 0 from the
  # product's LF signed distance) cut after the slope projection to the designed bed profile, as RIVER BEDS below cuts
  # the river beds: the projection's grade (TOWN_GRADE on per-axis edges) held the whole bed within .5 of the waterline
  # ~.2 over the design, so the crossing gradients were axis-aligned whatever bed the shape proposed (lakelevel-runs/v4
  # A5b). Under the water only: s >= 0 (waterline, bank, towns) keeps the projection's ground and its grade (checked
  # below: unchanged pixels, so the edge check above still holds there), never fixed (native) or plot pixels (the
  # outline never enters a plot hex; v3 freed every owner pixel under the water from its plateau, so RIVER BEDS' two-
  # pixel plot margin has nothing to protect here). The grade check above is for the visible ground.
  if lake_level_solved:
   _llb={}
   for _ll_name in sorted(lake_level_solved):
    _ll_s,(_wy0,_wy1,_wx0,_wx1),_=lake_level_sdf[_ll_name]
    _py0,_px0=_wy0*_lake_level.FINE,_wx0*_lake_level.FINE;_ps=(slice(_py0,_py0+_ll_s.shape[0]),slice(_px0,_px0+_ll_s.shape[1]))
    _pp=np.repeat(np.repeat(authored_plot[hy[_wy0:_wy1,_wx0:_wx1],hx[_wy0:_wy1,_wx0:_wx1]],_lake_level.FINE,0),_lake_level.FINE,1)[:_ll_s.shape[0],:_ll_s.shape[1]]
    _ew=encoded[_ps];_land0=_ew[_ll_s>=0].copy()
    _llb[_ll_name]=_lake_level.cut_bed(_ew,_ll_s,lake_level_solved[_ll_name]['level'],lake_level_lakes[_ll_name][1],~fixed_high[_ps]&~_pp,LEVEL,NORM,GAIN)
    if not np.array_equal(_ew[_ll_s>=0],_land0):raise RuntimeError(f'Lake level bed cut changed land pixels ({_ll_name})')
    _hw=LEVEL+(_ew.astype(np.float64)/65535-NORM)*GAIN;_lb=0.
    for _ax,_pt in ((0,pitch_z),(1,pitch_x)):
     _dl=np.diff((_ll_s>=0).astype(np.int8),axis=_ax)!=0
     if _dl.any():_lb=max(_lb,float((np.abs(np.diff(_hw,axis=_ax))[_dl]/_pt).max()))
    _llb[_ll_name].update({'fixed_px_under_water_kept':int((fixed_high[_ps]&(_ll_s<0)).sum()),'plot_px_under_water_kept':int((_pp&(_ll_s<0)).sum()),
                           'land_px_unchanged':int(_land0.size),'waterline_edge_slope_max':round(_lb,4)})
    print('Lake level bed',_ll_name,_llb[_ll_name],flush=True);del _ll_s,_pp,_ew,_land0,_hw
   approach_report['lake_level_beds']=_llb;del _llb
   assert np.array_equal(encoded[fixed_high],old_south[fixed_high])
   quantized=LEVEL+(encoded.astype(np.float32)/65535-NORM)*GAIN
  # --- RIVER BEDS (river_channels.carve_beds) --- the bed under the frontier river water, cut after the slope
  # projection, whose grade held RIVER CHANNELS' channel to ~.5 under the level (live 2048p: pale bed triangles
  # and land bars through the shallow water). Under the water only (the waterline, banks and river level stay),
  # never on fixed (native) pixels; the grade check above is for the visible ground.
  # Town plots (and two pixels round them: the display position samples the LF bilinearly) keep their
  # plateau: a river port on its channel (Tamralipti on ganges_1) was cut .14 below its plot level.
  # RIVER_BEDS off: deep beds need the water level from the height solution (pipeline-bake-design §3.5/§4);
  # cut under the pre-projection level they stood as steep pits and cliffs under high banks or left centres dry
  # (audit_terrain_continuity, audit_ground_seams steep generic, integ 3ea79353). 2026-10-26 user: off for this
  # package. TW3K_RIVER_BEDS=1 turns the cut on; off, the LF and the level file are as without RIVER BEDS.
  RIVER_BEDS=os.environ.get('TW3K_RIVER_BEDS','0')=='1'
  approach_report['river_beds_enabled']=RIVER_BEDS
  if river_bed is not None and RIVER_BEDS:
   from scipy import ndimage as _nd;_bed_ok=~fixed_high&~_nd.binary_dilation(high_owner>=0,iterations=2)
   approach_report['river_beds']=river_channels.carve_beds(encoded,resize_high(river_bed[0],(ow,oh)),resize_high(river_bed[1],(ow,oh)),_bed_ok,LEVEL,NORM,GAIN,centre=resize_high(river_bed[2],(ow,oh)));del _bed_ok
   assert np.array_equal(encoded[fixed_high],old_south[fixed_high])
   quantized=LEVEL+(encoded.astype(np.float32)/65535-NORM)*GAIN
   print('River beds',json.dumps(approach_report['river_beds']),flush=True)
  # Tile surface = the DDS pixels (4i,4j), as sync_final_terrain_surface resamples it: meshes,
  # towns and the camera read tile t at LF pixel 4t (vanilla's frame), not the antialiased 4t+1.5.
  surface=point_resample_low_surface(encoded);norm=np.clip(NORM+(surface-LEVEL)/GAIN,0,1);lf=np.ascontiguousarray(encoded[::-1],dtype='<u2')
  _spans=salt_pools.footprint_spans(salt_windows,_salt_shape,encoded.astype(np.float32)/65535)
  approach_report['salt_pan_levelling']={'groups':salt_levels,'footprint_lf_span':_spans,'max_footprint_lf_span':max(_spans,default=0.),'vanilla_max_span':.035}
  print('Salt-pan footprint LF spans',_spans,flush=True)
  put_parts(pre+'lf_height_map.dds',(_lf_head,lf.reshape(-1).view(np.uint8)));put(pre+'lf_height_map.compressed_map',cm_write(base(pre+'lf_height_map.compressed_map'),lf))
  # Preserve native special tiles where the final DDS did not actually change.
  # Include the native two-cell sampling halo, without growing ownership
  # recursively through already-selected neighbours.
  changed_pixels=(encoded!=old_south).reshape(TH,4,TW,4).any(axis=(1,3))
  changed_halo=fastops.max_filter_mask(changed_pixels,5)
  native_rectangle=(x>=_frame.NATIVE_TILE_X0)&(x<_frame.NATIVE_TILE_X1)&(y>=_frame.NATIVE_TILE_Y0)&(y<_frame.NATIVE_TILE_Y1)
  alpha[native_rectangle&~changed_halo]=0
  # Same antialiased reduction as oldheight: the changed-ground test stays like for like.
  corrected=np.abs(resize(quantized,(TW,TH))-oldheight)>1e-5;alpha[corrected]=np.maximum(alpha[corrected],.03);coverage_alpha=np.maximum(coverage_alpha,alpha);np.save(A/f'analysis/polished-edit-alpha-{variant}.npy',alpha)
  del high_old,_proposed_box,high_owner,high_surface,fixed_high,encoded,quantized,lf,_plain_x,_plain_z
  print('Actual DDS projection and quantization passed',variant,errors,flush=True)
  _mark('variant: DDS projection/quantization')
  if SINGLE_GROUND:
   # Finalize the existing boundary-height operation before any material or
   # forest fields read the surface. sync_final later verifies the output hash
   # and skips the same operation; water support therefore sees the final LF.
   from smooth_native_canvas_boundary import smooth_variant as _finalize_boundary
   _finalize_boundary(variant)
   _final_lf=np.memmap(S/(pre+'lf_height_map.dds'),mode='r',dtype='<u2',offset=128,shape=(oh,ow))[::-1]
   surface=point_resample_low_surface(_final_lf);del _final_lf
   norm=np.clip(NORM+(surface-LEVEL)/GAIN,0,1)
   alpha=np.load(A/f'analysis/polished-edit-alpha-{variant}.npy')
   coverage_alpha=np.maximum(coverage_alpha,alpha)
   _boundary_report=json.loads((A/'analysis/native-canvas-boundary-smoothing.json').read_text(encoding='utf-8'))
   approach_report['height_final_before_materials']=_boundary_report['variants'][variant]['output_sha256']
   _mark('variant: final boundary height before material fields')
  biome=apply_climate_override(inherit_climate(biome,asset(pre+'climate_map.cm'),asset(pre+'tile_list.bin'),asset))
  if HH is not None:biome=HH.climate(biome,q,r)   # --- HEHUANG PATCH (hehuang_terrain.py) ---
  # The two native blend channels pick the ground material (grass/sand/rock).
  # Native land keeps its own blend. Canvas land continues the native blend by
  # mirroring it across the native border, then fades into vanilla donor patches
  # per climate; native water turned into land takes the donor patches. All
  # weights are soft: a hard alpha>.5 paste drew tile-sized sand staircases.
  # The channels are material indices (0-31 and 0-3), never weights: averaging two
  # indices names an unrelated material (the blue-green bands and town rings). So
  # every pixel takes one source; fractal-noise dithering makes transitions organic.
  _mark('variant: climate inheritance')
  k=pre+'global_map/global_blend.dds';src=base(k);cur=np.frombuffer(src,dtype=np.uint8,offset=128).reshape(oh,ow,2).copy();orig=asset(k);hh,ww=struct.unpack_from('<2I',orig,12);donor=np.frombuffer(orig,dtype=np.uint8,offset=128).reshape(hh,ww,2)
  # Donor patches (native hex centres, 48x48 hexes) per climate, picked on
  # vanilla land without water, with low relief and no thin linear features
  # (rivers, shores, roads): the old patches held a river, a lake shore and
  # banded slopes that tiled into parallel pale stripes on the canvas (west
  # edge 510, live check 2026-10-20 f) -- and mostly of their own climate's material family
  # (texture_arrays.xml: arid 0-3, cold 4-7, subtropical 23-26, temperate 27-30):
  # a steppe-heavy arid patch put yellow-green blotches on every Western desert
  # town (live check 2026-10-20 h).
  # Several patches per climate: one patch each read as one flat material over
  # whole frontier zones (the cold and subtropical patches were 95-99% material 4,
  # so the Siberian taiga and Indochina were one dark olive sheet, offline views
  # 2048bi). Each list runs from the flattest to the most rugged vanilla ground
  # (90th percentile slope in brackets); see donor_choice below.
  # 0: subtropical: dense material-4 forest (.17), 4/26/30/28 mixes (.37), with subtropical_2 (.47), (.52); a steppe-bearing patch cut the south coast, these hold <=5% steppe and no river bands
  # 1: cold (vanilla north-west): material 4 (.09), cold_1-4 mixes (.37, .39, .44); (100,412) held a temperate 27-29 blob on its fold row 383, mirrored into pale stepped patches over Siberia (live check 2048j)
  # 2: temperate (.09, .33, .39)
  # 3: arid: plain sand (.01), arid_1-4 dunes (.27); no steppe materials
  # 9: steppe (77% steppe materials), real-geography steppe only
  # --- OASIS GROUND (oasis_ground.subtropical_split) --- subtropical real ground by biome. 10: paddy
  # and mangrove: vanilla G3 paddy lowland (survey 2026-10-24, 48-hex windows on all-land G3 ground
  # with < 1 % sand, < 2 % mud and wetland, no road, < 12 % steppe: 308,188 subtropical_4 paddy 25 %,
  # subtropical_2 21 %, cold_1 17 %; 540,304 paddy 33 %, cold_1 25 %, subtropical_2 18 %; the flat
  # patch 64,202 is 99 % cold_1; 532,312 was rejected: its sand corner, wetland band and mud blob
  # tiled into a mirrored cross over the Bengal plain). 11: open subtropical (savanna, grass,
  # wetland, other): the mixed patches without the dense-forest one. Forest biomes keep climate 0's list.
  patches=_BP['DONOR_PATCHES']   # 0 subtropical, 1 cold, 2 temperate, 3 arid, 9 steppe, 10 paddy lowland, 11 open subtropical (OASIS GROUND)
  native_land=native_rectangle&land_tile_coverage(base(pre+'tile_list.bin'),asset,(TH,TW))
  ty_,tx_=np.mgrid[:TH,:TW];border=np.maximum.reduce([_frame.NATIVE_TILE_X0-tx_,tx_-(_frame.NATIVE_TILE_X1-1),_frame.NATIVE_TILE_Y0-ty_,ty_-(_frame.NATIVE_TILE_Y1-1),np.zeros_like(tx_)])
  # --- NONPLAY GROUND --- donor weight of a canvas/ring pixel whose mirror source lies on the repainted filler
  # (the mirror carried its vanilla filler): 0 on the border, full NONPLAY_EDGE_FADE cells out, as the filler's own
  # fade inside. A full donor there put the filler's and the canvas's donors side by side on the border line (cut).
  nonplay_mirror_w=np.clip(border/NONPLAY_EDGE_FADE,0,1).astype(np.float32)
  # Short mirrored band: a long mirror shows as a symmetric copy of the native texture.
  # The fade measured straight from the native border put the mirror/donor
  # transition on a line parallel to it (right-angled edges south of Dunhuang,
  # live check 2026-10-19 c): it now meanders and varies in width.
  fade=np.clip((border-BLEND_MEANDER*(.5+.6*gpu_noise.fbm(q,r,(NQ,NR),7,3,seed=97)))/(24+BLEND_MEANDER*np.clip(.5+.6*gpu_noise.fbm(q,r,(NQ,NR),20,3,seed=98),0,1)),0,1)
  fade=fade*fade*(3-2*fade)
  edit_weight=np.clip(alpha*2,0,1);edit_weight=edit_weight*edit_weight*(3-2*edit_weight)
  # Donor share per tile: canvas land fades in with distance; native water-now-land follows the edit weight.
  # The outer native rows carry no land tiles (they were the world edge), so they
  # must not count as former water: the BLEND_RING band takes the pure mirror.
  native_core=(x>=_frame.NATIVE_TILE_X0+BLEND_RING)&(x<_frame.NATIVE_TILE_X1-BLEND_RING)&(y>=_frame.NATIVE_TILE_Y0+BLEND_RING)&(y<_frame.NATIVE_TILE_Y1-BLEND_RING)
  donor_share=np.where(native_core,np.where(physical&~native_land,edit_weight,0),np.where(native_rectangle,0,fade)).astype(np.float32)
  # Land on vanilla sea (new islands, filled coast) takes the donor outright, in
  # the core and the outer ring alike: sea-bed material on land read as pale sand
  # with straight edges, and a partial share left a cut on the native border
  # where the canvas side was already all donor (Wa Nu, canvas 1400/2048).
  import map_hex as _mh
  vanilla_sea_hex=np.zeros((H,W),bool);vanilla_sea_hex[_frame.NATIVE_HEX_SLICE]=(_mh.read(_frame.VANILLA_MAP)['cells'][:,:,0]&3)==1
  vanilla_sea=vanilla_sea_hex[hy,hx]
  donor_share[physical&vanilla_sea]=1.
  # The mirrored native band carries the native ground: where the native climate
  # at the mirror source differs from the canvas climate (cold alpine grass
  # mirrored into the Shanshan desert, live check 2026-10-20 d) the canvas takes
  # its own climate's donor. Zero at the border, where the carried climate agrees.
  _X0t,_X1t=_frame.NATIVE_TILE_X0+BLEND_RING,_frame.NATIVE_TILE_X1-1-BLEND_RING;_Y0t,_Y1t=_frame.NATIVE_TILE_Y0+BLEND_RING,_frame.NATIVE_TILE_Y1-1-BLEND_RING
  def _rt(v,lo,hi):return np.clip(np.where(v<lo,2*lo-v,np.where(v>hi,2*hi-v,v)),lo,hi)
  mirror_mismatch=(border>0)&physical&(biome!=biome[_rt(ty_,_Y0t,_Y1t),_rt(tx_,_X0t,_X1t)])
  if MIRROR_EDGE:
   # --- MIRROR EDGE (ground contrast fix design §2) --- the mismatch hands the band to the donor over a smooth
   # ramp out to a meandering radius on the one native edge geometry (biome_profiles.native_edge), not the straight
   # clip(border/8) ramp (the straight edge + staircase at ring-west, v19 live check). Within the tile climate ring.
   _me=_bp.CONSTS['ground_materials']['MIRROR_EDGE'];_nr=_bp.CONSTS['ground_materials']['NATIVE_RING']
   assert _me['reach']<=_nr['reach'] and _me['meander']<=_nr['meander'],'MIRROR_EDGE must lie inside the tile climate ring (NATIVE_RING)'
   _dn,_m=_bp.native_edge((TH,TW),q,r,(NQ,NR));_ramp=np.clip(_dn/(_me['reach']+_me['meander']*_m),0,1);_ramp=_ramp*_ramp*(3-2*_ramp)
   donor_share=np.maximum(donor_share,box_blur(mirror_mismatch.astype(np.float32),4)*_ramp).astype(np.float32);del _dn,_m,_ramp
  else:
   donor_share=np.maximum(donor_share,np.clip(box_blur(mirror_mismatch.astype(np.float32),4)*1.5,0,1)*np.clip(border/8,0,1)).astype(np.float32)
  # [EGOBI] the mirror band copied vanilla's north-rim woodland (cold_1/2) onto the real East Gobi: a mirrored
  # woodland pixel takes the canvas donor with this weight (real open ground, ramped in from the native border)
  # MIRROR_OPEN switch (user stop-loss 2026-10-26): with the filler repaint off, the band's donor sand stood as a
  # straight ~11-hex sand strip beside the dark vanilla filler at q=888 (offline run2fix2 edge_w800); off for this
  # package, TW3K_EGOBI=1 turns it on with the rest of EGOBI (redo with the NONPLAY filler repaint).
  MIRROR_OPEN=EGOBI   # the mirror_open part of the EGOBI switch (TW3K_EGOBI=1)
  mirror_open_w=(np.clip(real_w,0,1)*np.isin(real_bio_eco,[_bnames.index(n) for n in _bp.biome_set('mirror_open') if n in _bnames])
                 *np.clip(border/MIRROR_OPEN_RAMP,0,1)*physical).astype(np.float32) if REAL_TERRAIN.exists() and MIRROR_OPEN else np.zeros((TH,TW),np.float32)
  # Town climate overrides take their climate's donor ground too.
  donor_share=np.maximum(donor_share,climate_override_weight)
  # --- HEHUANG PATCH (hehuang_terrain.py) --- the patch takes donor ground of its classes
  if HH is not None:donor_share=np.maximum(donor_share,HH.donor_share()*physical).astype(np.float32)
  donor_share=np.maximum(donor_share,nonplay_w).astype(np.float32)   # --- NONPLAY GROUND --- repurposed filler takes the donor
  _visual_domains=None;_selection_biome=biome;_material_real_gw=real_gw
  _material_steppe=real_steppe;_material_eco_zone=eco_zone;_material_climate_shares=None
  if SINGLE_GROUND:
   import visual_domains as _vd
   _visual_domains=_vd.build(cells,names,(TH,TW),edit_exceptions=native_rectangle&~_protected,protect=_protected)
   _material_real_gw=(_visual_domains['material_takeover']*(real_bio_eco!=255)).astype(np.float32)
   # Geographical source support has one cross-boundary domain; neither the
   # rectangular asset border nor height alpha creates a material dead band.
   donor_share=np.maximum(donor_share,_material_real_gw).astype(np.float32)
   donor_share[_visual_domains['native_keep']]=0
   _geocl=np.full(256,255,np.uint8)
   for _name,_climate in _bp.biome_map('polish_climate','climate').items():
    if _name in _bnames:_geocl[_bnames.index(_name)]=_climate
   _geo=_geocl[real_bio_eco]
   _geo=apply_climate_override(_geo)
   if HH is not None:_geo=HH.climate(_geo,q,r)
   _selection_biome=np.where((_material_real_gw>.5)&(_geo<4),_geo,biome).astype(biome.dtype)
   _material_steppe,_material_eco_zone=_vd.material_ecology(real_bio,real_bio_eco,_bnames,
     _material_real_gw,real_steppe,box_blur,
     authored_steppe=None if HH is None else HH.steppe(q,r),
     ecotone_enabled=os.environ.get('TW3K_MATERIALS_ECOTONE','1')!='0')
   _material_climate_shares=_vd.material_climate_shares(biome,_geo,_material_real_gw)
   import oasis_ground as _og
   sub_split=_og.subtropical_split(real_bio_eco,_bnames,_material_real_gw)
  # Per-tile climate score, sampled bilinearly per pixel with pixel-scale noise
  # below: picking one donor climate per tile cell drew half-hex staircases.
  if _material_climate_shares is None:_material_climate_shares=[(biome==c).astype(np.float32) for c in range(4)]
  climate_score=np.stack([box_blur(_material_climate_shares[c],6)+.35*gpu_noise.fbm(q,r,(NQ,NR),4,3,seed=50+c) for c in patches if c<4])
  if _material_eco_zone is not None and _material_eco_zone.any():   # --- MATERIALS ECOTONE --- ecotone detail back into the donor score
   climate_score+=np.stack([ECO_DETAIL*_material_eco_zone*(box_blur(_material_climate_shares[c],ECO_DETAIL_R)-box_blur(_material_climate_shares[c],6)) for c in patches if c<4])
  del _material_climate_shares
  _tl=None   # [section 9.3 treeline donor blend] TW3K_TREELINE_BLEND: per-pixel donor climate dither on the tundra side of the line
  if _bp.treeline_blend_on() and REAL_TERRAIN.exists():
   import ground_donor as _gdn;_tlr=_BP['TREELINE_DONOR'];_tl=_gdn.treeline_weight(real_bio_eco,_bnames,_tlr)
   if _tl is not None:
    _tgq=fbm(*(np.mgrid[0:1024,0:1024][::-1]/4.),(NQ,NR),_tlr['grain'],3,seed=_tlr['seed']);_tl=(_tl[0],_tl[1],np.quantile(_tgq,np.linspace(0,1,257)));del _tgq
  def _treeline(picked,xx,yy,p,zone):
   if not zone.any():return picked
   u=np.interp(gpu_noise.fbm(np.broadcast_to(xx/8,picked.shape),np.broadcast_to((oh-1-yy)/8,picked.shape),(NQ,NR),_tlr['grain'],3,seed=_tlr['seed']),_tl[2],np.linspace(0,1,257)).astype(np.float32)
   return _gdn.treeline_pick(picked,p,u,zone,_tlr['climates'])
  climate_keys=np.asarray([c for c in patches if c<4],np.int8)
  # Donor patch choice per tile in [0,1]: rank of large-scale noise blended with
  # the rank of local relief (box-blurred slope of the polished surface), ranked
  # again over canvas land. Every patch covers an equal share, rugged ground
  # leans to the rugged patches and plains to the lowland ones; a tile takes
  # patch int(choice*len(list)) of its climate. Off land the noise rank alone
  # (only the bilinear fringe of the coast reads it).
  def _rank(a,m):
   o=np.zeros(a.shape,np.float32);v=a[m];o[m]=np.searchsorted(np.sort(v),v)/max(len(v)-1,1);return o
  _gz,_gx=np.gradient(surface.astype(np.float32));_rel=box_blur(np.hypot(_gx,_gz).astype(np.float32),6);del _gz,_gx
  _all=np.ones((TH,TW),bool);_nr=_rank(gpu_noise.fbm(q,r,(NQ,NR),DONOR_SCALE,4,seed=104),_all)
  donor_choice=np.where(physical,_rank((1-DONOR_RELIEF)*_rank(_nr,physical)+DONOR_RELIEF*_rank(_rel,physical),physical),_nr).astype(np.float32);del _rel,_nr,_all
  if _bp.taiga_donor_on() and REAL_TERRAIN.exists():   # [section 9.3 taiga flecks] TW3K_TAIGA_DONOR: flat patches only on DONOR_COMPRESS biomes
   import ground_donor as _gdn;donor_choice=_gdn.compress_choice(donor_choice,real_bio_eco,_bnames,DONOR_BAND,_BP['DONOR_COMPRESS'],box_blur)
  # Soft patch shares per tile: P(patch<=k) is the indicator choice<threshold
  # box-blurred twice over DONOR_BAND cells (a linear ramp ~2*2*DONOR_BAND cells
  # wide), stored per distinct threshold; patch_cum[climate] lists them in order.
  _thr={};patch_cum={}
  for _cl,_pl in patches.items():
   _sh=np.asarray(DONOR_SHARES.get(_cl,[1/len(_pl)]*len(_pl)),np.float64);assert len(_sh)==len(_pl)
   patch_cum[_cl]=[]
   for _t in np.cumsum(_sh/_sh.sum())[:-1]:
    _k=round(float(_t),6)
    if _k not in _thr:_thr[_k]=box_blur(box_blur((donor_choice<_k).astype(np.float32),DONOR_BAND),DONOR_BAND).astype(np.float16)
    patch_cum[_cl].append(_thr[_k])
  del _thr
  # Quantiles of the grain noise, to draw patches with a uniform variable.
  _gq=fbm(*(np.mgrid[0:1024,0:1024][::-1]/4.),(NQ,NR),DONOR_GRAIN,3,seed=107);grain_q=np.quantile(_gq,np.linspace(0,1,257));del _gq
  # The outer native blend rows were the old map edge and hold an edge material;
  # mirroring them drew stripes parallel to the border. Mirror across a line
  # BLEND_RING cells inside the native map and replace that ring too.
  ring=BLEND_RING*4
  X0p,X1p=_frame.NATIVE_TILE_X0*4+ring,_frame.NATIVE_TILE_X1*4-1-ring;Y0p,Y1p=oh-_frame.NATIVE_TILE_Y1*4+ring,oh-_frame.NATIVE_TILE_Y0*4-1-ring
  def reflect(v,lo,hi):return np.clip(np.where(v<lo,2*lo-v,np.where(v>hi,2*hi-v,v)),lo,hi)
  source_blend=np.frombuffer(src,dtype=np.uint8,offset=128).reshape(oh,ow,2)
  xx=np.arange(ow,dtype=np.int32)[None,:];mx=reflect(xx,X0p,X1p)
  print(f'[blend-timing] setup before rows: {_time.time()-_TL[0]:.1f}s',flush=True);_tb=_time.time()
  # Row blocks are independent (each reads and writes its own rows): a thread
  # pool runs them in parallel (numpy releases the GIL), identical result.
  # Their GPU noise calls take turns on one lock in gpu_noise; the CPU work overlaps.
  # Mirror warp (pixels) per tile: grows with distance from the native border.
  _wg=np.clip(border/24,0,1).astype(np.float32)*MIRROR_WARP
  mirror_jx=np.rint(_wg*gpu_noise.fbm(q,r,(NQ,NR),6,3,seed=99)).astype(np.int32);mirror_jy=np.rint(_wg*gpu_noise.fbm(q,r,(NQ,NR),6,3,seed=100)).astype(np.int32);del _wg
  # Donor tiling warp (pixels) per tile: the mirrored 768-pixel tiling repeated
  # the patch in a visible kaleidoscope; a smooth warp and the per-patch phases
  # break the period. Float per tile, interpolated per pixel: an integer offset
  # per tile sheared the ground into tile-sized staircases.
  donor_wx=(DONOR_WARP*gpu_noise.fbm(q,r,(NQ,NR),90,2,seed=101)).astype(np.float32);donor_wy=(DONOR_WARP*gpu_noise.fbm(q,r,(NQ,NR),90,2,seed=102)).astype(np.float32)
  # Same per-pixel arithmetic as _blend_rows_reference (identical bytes), but
  # tile-grid lookups are gathered per row and per column, bilinear weights are
  # separable (rows of horizontally interpolated tiles), the donor patches are
  # read only where their climate is picked, and the materials stay uint8.
  _bxv=(np.arange(ow,dtype=np.int32)//4).clip(0,TW-1)
  _fx1=((xx+.5)/4-.5).clip(0,TW-1);_x01=np.minimum(_fx1.astype(np.int32),TW-2)[0];_wx1_32=(_fx1-_x01).astype(np.float32);_wx1_64=_fx1-_x01
  def _rowcols(T,byv):
   rows=T[byv]
   return np.repeat(rows,4,axis=1) if ow==4*TW and T.shape[1]==TW else rows[:,_bxv]
  def _bilinear_rows(c,y0,wy,wx,dtype_weights):
   # (c[y0,x0]*(1-wx)+c[y0,x0+1]*wx)*(1-wy)+(c[y0+1,x0]*(1-wx)+c[y0+1,x0+1]*wx)*wy, with each
   # horizontal term computed once per distinct tile row.
   u=np.unique(np.concatenate((y0[:,0],y0[:,0]+1)))
   cu=c[u];hrows=cu[:,_x01]*(1-wx)+cu[:,_x01+1]*wx
   i0=np.searchsorted(u,y0[:,0]);i1=np.searchsorted(u,y0[:,0]+1)
   return hrows[i0]*(1-wy)+hrows[i1]*wy
  # --- FIX2 KOREA GROUND --- inside the real-geography patches (ground weight > 0: no protected land,
  # no plots) the donor share is read bilinearly: a per-tile share
  # drew every forest/open edge of the Korea classes as half-hex stairs (offline 2048bq).
  _hh_win=(HH.w>0) if HH is not None else None
  def _build_ground_fields():
   # One classification for source selection and the later material pass; no blend bytes are read or changed.
   import ground_materials as gm
   with np.load(REAL_TERRAIN) as gz:
    lon=np.zeros((TH,TW),np.float32);lat=np.zeros((TH,TW),np.float32);x0,x1,y0,y1=(int(v) for v in gz['box'])
    lon[y0:y1,x0:x1]=gz['lon'];lat[y0:y1,x0:x1]=gz['lat']
   gw=(_material_real_gw*physical if SINGLE_GROUND else real_gw*donor_share*physical).astype(np.float32)
   # Shared ownership already provides the protected-land transition. The
   # legacy desert/native fade is not a second material authority in this mode.
   dw=gw if SINGLE_GROUND else (np.maximum(np.clip(real_w,0,1)*physical*np.clip(border/gm.DESERT_RAMP,0,1),nonplay_w).astype(np.float32) if EGOBI else (nonplay_w if NONPLAY_GROUND else None))
   fields=gm.tile_fields(real_elev,real_bio_eco,_bnames,lon,lat,gw,q,r,(NQ,NR),oasis=oasis,desert_w=dw,river=real_river,keep=alpine)
   return fields,lon,lat,gw,dw
  # --- TONE BRIDGE (ground contrast fix §1) --- where the dither picks between a mirror and a donor material whose
  # tones differ more than the output G band's own vanilla neighbour contrast (CONSTS CONTRAST by_band p99), the pixels
  # nearest the dither threshold (|u - share| < BRIDGE_W x 2 share (1 - share)) take the bridge slot whose tone lies
  # between the two (minimax CIE76 over the natural slots, ground_tone). Chosen in the same single dither, never
  # redrawn after. Off without the measured tone table or with TW3K_TONE_BRIDGE=0.
  _tb_tab=_tb_rank=_tb_dist=None
  if TONE_BRIDGE:
   import ground_tone as _gt
   _tones,_mde=_gt.tone_table(),_gt.max_dE()
   if _tones is not None and _mde is not None:
    _D=_gt.slot_distance(_tones);_nat=np.array(_bp.CONSTS['ground_materials']['BRIDGE_SLOTS'],np.int64)
    _mm=np.maximum(_D[:,None,_nat],_D[None,:,_nat])                     # (a, b, candidate): worse of the two distances
    _bi=np.argmin(_mm,axis=2);_bv=np.take_along_axis(_mm,_bi[...,None],2)[...,0]
    _mde_band=np.asarray([_gt.max_dE(b) for b in range(4)],np.float32)
    _br=np.where((_D[None]>_mde_band[:,None,None])&(_bv[None]<_D[None]),_nat[_bi][None],255).astype(np.uint8)    # G band x a x b; 255: no closer bridge needed
    _tb_rank=_nat[np.argsort(_mm,axis=2,kind='stable')].astype(np.uint8);_tb_dist=_D
    _tb_tab=(_br,float(_bp.CONSTS['ground_materials']['BRIDGE_W']));del _D,_mm,_bi,_bv,_mde_band
  # First actual run re-sampled 8.86% of surviving bridges as banned (> the reviewed 1% limit).
  # Move the one tile_fields computation before the source choice and reuse it unchanged in GROUND MATERIALS.
  # Sources off / GM off keep their former calculation path. The thread budget observes these live arrays.
  _early_ground=None
  if os.environ.get('TW3K_GROUND_MATERIALS','1')!='0' and REAL_TERRAIN.exists() and (_tb_tab is not None or DONOR_WOODLAND or SINGLE_GROUND):
   import ground_materials as _gm_source
   _early_ground=_build_ground_fields()
  def _source_bans(xx,yy):
   if _early_ground is None:return None
   lb,al=_gm_source._ban_rows_fields(_early_ground[0],(xx[0]+.5)/4-.5,(oh-1-yy[:,0]+.5)/4-.5)
   return _gm_source,lb,al
  # --- DONOR WOODLAND (ground contrast fix §3) --- the donor's woodland floor (CONSTS DONOR_WOODLAND slots, cold_1)
  # follows the shared forest density (biome_profiles.forest_density, the props bake's field): a donor pixel is
  # woodland where its noise < density x k[climate], else the patch's own open material (or the climate's open list
  # when the patch gave woodland). Chosen with the donor material, no redraw after. TW3K_DONOR_WOODLAND=0: off.
  _src_flag=np.zeros((oh,ow),np.uint8) if (TONE_BRIDGE or DONOR_WOODLAND) and not SINGLE_GROUND else None   # low bits 1/2: selected tone bridge / donor woodland; high bits: selected slot
  _fdt=None;_ffl=_bp.forest_fill((TH,TW),cells,names,domains=_visual_domains,real_bio=real_bio_eco,bnames=_bnames)   # --- FOREST DENSITY --- the fill fade, once
  if (DONOR_WOODLAND or SINGLE_GROUND) and REAL_TERRAIN.exists():
   # The shared forest density: the donor woodland floor's field, and the single ground bake's own input
   # (computed here for SINGLE_GROUND whether or not TW3K_DONOR_WOODLAND is on; same function and arguments).
   _fdt=_bp.forest_density(real_bio_eco,_bnames,q,r,(NQ,NR),fill=_ffl)
   if SINGLE_GROUND and HH is not None:
    _fdt=np.maximum(_fdt,HH._canvas('forest_floor',0,np.float32)).astype(np.float32)
    _fdt[_visual_domains['native_keep']]=0
  if DONOR_WOODLAND and REAL_TERRAIN.exists():
   _dw=_bp.CONSTS['ground_materials']['DONOR_WOODLAND']
   _dw_reach=np.clip(real_gw*physical,0,1).astype(np.float32)   # the same geographic takeover as the material bake; native carry stays intact
   _dw_slots=np.zeros(256,bool);_dw_slots[list(_dw['slots'])]=True
   _dw_k=np.zeros(16,np.float32);_dw_open_n=np.zeros(16,np.int32)
   assert _dw['open'] and all(len(v)>0 for v in _dw['open'].values()),'Empty donor open-material recipe'
   _dw_open=np.zeros((16,max(len(v) for v in _dw['open'].values())),np.uint8)
   for _c,_v in _dw['k'].items():_dw_k[int(_c)]=_v
   for _c,_v in _dw['open'].items():_dw_open_n[int(_c)]=len(_v);_dw_open[int(_c),:len(_v)]=_v
  def _woodland(mix,picked,xx,yy,noise,flag=None,bans=None,rejected=None):
   if _fdt is None:return mix
   fx=np.broadcast_to(((xx+.5)/4-.5).clip(0,TW-1),picked.shape);fy=np.broadcast_to(((oh-1-yy+.5)/4-.5).clip(0,TH-1),picked.shape)
   x0=np.minimum(fx.astype(np.int32),TW-2);y0=np.minimum(fy.astype(np.int32),TH-2);wx=(fx-x0).astype(np.float32);wy=(fy-y0).astype(np.float32)
   fd=(_fdt[y0,x0]*(1-wx)+_fdt[y0,x0+1]*wx)*(1-wy)+(_fdt[y0+1,x0]*(1-wx)+_fdt[y0+1,x0+1]*wx)*wy
   gx=np.broadcast_to(xx/8,picked.shape);gy=np.broadcast_to((oh-1-yy)/8,picked.shape)
   u=.5+.5*noise(gx,gy,(NQ,NR),_dw['grain'],3,seed=_dw['seed'])
   wood=_dw_slots[mix[...,0].astype(np.int64)]
   rw=(_dw_reach[y0,x0]*(1-wx)+_dw_reach[y0,x0+1]*wx)*(1-wy)+(_dw_reach[y0+1,x0]*(1-wx)+_dw_reach[y0+1,x0+1]*wx)*wy
   target=(1-rw)*wood+rw*fd*_dw_k[picked]
   want=np.where(rw<=0,wood,np.where(rw>=1,u<fd*_dw_k[picked],u<target))   # preserve both reach endpoints, including u == 1
   out=mix.copy()
   add=want&~wood;out[...,0]=np.where(add,_dw['slots'][0],out[...,0])
   rem=~want&wood
   if rem.any():
    _uopen=.5+.5*noise(gx,gy,(NQ,NR),2.,2,seed=_dw['seed']+1)
    # Retain the old float arithmetic and clipping exactly for three candidates.
    j=(np.floor(_uopen*3).astype(np.int64)).clip(0,2)
    _n=_dw_open_n[picked];assert (_n>0).all(),'Missing donor open-material recipe'
    _var=_n!=3
    if _var.any():
     _j=np.floor(_uopen[_var]*_n[_var]).astype(np.int64)
     j[_var]=np.minimum(np.maximum(_j,0),_n[_var]-1)
    out[...,0]=np.where(rem,_dw_open[picked,j],out[...,0])
   if bans is not None:
    gm,lb,al=bans;bad=(add|rem)&gm._banned(lb,al,out[...,0].astype(np.uint8,copy=False));initial=int(bad.sum())
    # Only forbidden candidates change: preserve all original noise/three-choice boundary arithmetic otherwise.
    for rank in range(max(len(_dw['slots']),_dw_open.shape[1])):
     si,sj=np.nonzero(bad)
     if not len(si):break
     climates=picked[si,sj];slot=np.where(add[si,sj],_dw['slots'][min(rank,len(_dw['slots'])-1)],_dw_open[climates,np.minimum(rank,_dw_open_n[climates]-1)])
     legal=~gm._banned(lb[si,sj],None if al is None else al[si,sj],slot)
     out[si[legal],sj[legal],0]=slot[legal];bad[si[legal],sj[legal]]=False
    if rejected is not None:
     rejected['donor_woodland_preferred_banned']=initial;rejected['donor_woodland_rejected']=int(bad.sum())
    out[...,0]=np.where(bad,mix[...,0],out[...,0])   # no legal family member: retain the original donor candidate
   if flag is not None:flag|=((out[...,0]!=mix[...,0]).astype(np.uint8)<<1)
   return out
  def _tone_bridge(take,mix,base_val,share,dither,flag=None,bans=None,rejected=None):
   # the material and band of each pixel after the bridge: the dither's choice, or the bridge slot (band of the side taken)
   out=np.where(take,mix,base_val);sel=None
   if _tb_tab is not None:
    _br,_w=_tb_tab;b=_br[out[...,1].astype(np.int64),mix[...,0].astype(np.int64),base_val[...,0].astype(np.int64)]
    sel=(b!=255)&(np.abs(dither-share)<_w*2*share*(1-share))
    if bans is not None and sel.any():
     gm,lb,al=bans;bad=np.zeros(sel.shape,bool)
     si,sj=np.nonzero(sel);bad[si,sj]=gm._banned(lb[si,sj],None if al is None else al[si,sj],b[si,sj]);initial=int(bad.sum())
     for rank in range(_tb_rank.shape[2]):
      si,sj=np.nonzero(bad)
      if not len(si):break
      a=mix[si,sj,0].astype(np.int64);c=base_val[si,sj,0].astype(np.int64);slot=_tb_rank[a,c,rank]
      legal=(np.maximum(_tb_dist[a,slot],_tb_dist[c,slot])<_tb_dist[a,c])&~gm._banned(lb[si,sj],None if al is None else al[si,sj],slot)
      b[si[legal],sj[legal]]=slot[legal];bad[si[legal],sj[legal]]=False
     if rejected is not None:
      rejected['tone_bridge_preferred_banned']=initial;rejected['tone_bridge_rejected']=int(bad.sum())
     sel&=~bad   # no legal closer bridge: the original mirror/donor dither choice stands
    if sel.any():out[...,0]=np.where(sel,b,out[...,0])
   _select_blend_sources(flag,take,sel,out[...,0])
   return out
  # --- end FIX2 KOREA GROUND
  _single_bake=None;_owner_map=None;_written_rows=None
  if SINGLE_GROUND:
   if _early_ground is None or _fdt is None:raise RuntimeError('Single ground bake requires the geographical fields (TW3K_GROUND_MATERIALS on, west-real-terrain.npz present) and the shared forest density')
   import ground_bake as _gb
   _gf,_glon,_glat,_ggw,_gdw=_early_ground
   _water_field=None
   if os.environ.get('TW3K_CLIMATE_WETLANDS','1')!='0':
    import ground_water_support as _gws
    _water_keys=set(idx)
    _water_keys.update(p.relative_to(BASE).as_posix() for p in BASE.rglob('*') if p.is_file())
    def _water_read(key):return (BASE/key).read_bytes() if (BASE/key).is_file() else asset(key)
    def _water_list(prefix,suffix):return [k for k in _water_keys if k.startswith(prefix) and k.endswith(suffix)]
    _water_lf=np.memmap(S/(pre+'lf_height_map.dds'),mode='r',dtype='<u2',offset=128,shape=(oh,ow))[::-1]
    _native_water=_gws.NativeWater(_water_read,_water_list,(oh,ow),variant,
      frame_for_key=lambda key:'canvas' if (BASE/key).is_file() else 'vanilla',
      planned_lakes=[] if HH is None or HH.prop is None else [HH.prop])
    _water_reader=_gws.prepare(_water_lf,river_level,cells,names,native_water=_native_water,
      background_water=_gws.ProductionBackground((oh,ow)))
    _water_field={'shape':_water_reader.shape,'mask':_water_reader.read,'source':_water_reader.provenance}
   _single_bake=_gb.prepare(fields=_gf,lon=_glon,lat=_glat,domains=_visual_domains,forest=_fdt,origin=(NQ,NR),
     cells=cells,names=names,plan=plan,biome=_selection_biome,surface=surface,alpha=alpha,physical=physical,
     protected=_visual_domains['native_keep'],plots=authored_plot[hy,hx],oasis=oasis,real_bio=real_bio_eco,
     bnames=_bnames,real_elev=real_elev,real_river=real_river,q=q,r=r,inherited=source_blend,donor=donor,
     water_field=_water_field,settings=_BP,
     authored_forest=None if HH is None else HH._canvas('forest_floor',0,np.float32))
   if variant=='3k_dlc07_main_map':
    import landform_tiles as _lft
    print('Landform tiles:',_lft.save(A,_gf,real_bio_eco,{'ground_w':_ggw,'desert_w':_gdw,'oasis':oasis,'keep':alpine,
      'river':real_river,'forest_density':_fdt,'forest_fill':_ffl,'biome_names':_bnames,'visual_domains':_visual_domains}),flush=True)
   _owner_map=np.lib.format.open_memmap(A/f'analysis/blend-owners-{variant}.npy',mode='w+',dtype=np.uint8,shape=(oh,ow))
   _written_rows=np.zeros(oh,bool)
   gm_cold=_gf['cold'];_gban=None
   (A/f'analysis/farm-fields-{variant}.json').write_text(json.dumps(_single_bake.farm_report,ensure_ascii=False,indent=1),encoding='utf-8')
  def _blend_rows(row):
   _core_row=row;_core_end=min(row+(64 if SINGLE_GROUND else 256),oh)
   _end=min(_core_end+2,oh) if SINGLE_GROUND else _core_end
   if SINGLE_GROUND:row=max(0,row-2)
   yy=np.arange(row,_end,dtype=np.int32)[:,None];by=((oh-1-yy)//4).clip(0,TH-1);byv=by[:,0];n=len(yy)
   share=_rowcols(donor_share,byv);outside=(xx<X0p)|(xx>X1p)|(yy<Y0p)|(yy>Y1p)
   oi,oj=np.nonzero(outside)
   base_val=(source_blend if SINGLE_GROUND else cur)[row:row+n].copy()
   if len(oi):
    _jx=mirror_jx[byv[oi],_bxv[oj]];_jy=mirror_jy[byv[oi],_bxv[oj]]
    wmx=reflect(np.clip(xx[0,oj]+_jx,0,ow-1).astype(np.int32),X0p,X1p);wmy=reflect(np.clip(yy[oi,0]+_jy,0,oh-1).astype(np.int32),Y0p,Y1p)
    base_val[oi,oj]=source_blend[wmy,wmx]
    _vs=vanilla_sea[((oh-1-wmy)//4).clip(0,TH-1),(wmx//4).clip(0,TW-1)]|np.isin(source_blend[wmy,wmx,0],MIRROR_SKIP)
    _sea=np.zeros(outside.shape,bool);_sea[oi[_vs],oj[_vs]]=True
    share=np.where(_sea,1.,share)
    # --- NONPLAY GROUND --- a mirror source on the repainted filler takes the donor, ramped in from the border
    _npv=(nonplay_w[((oh-1-wmy)//4).clip(0,TH-1),(wmx//4).clip(0,TW-1)]>.5)&~_vs
    if _npv.any():
     _nw=np.zeros(outside.shape,np.float32);_nw[oi[_npv],oj[_npv]]=_rowcols(nonplay_mirror_w,byv)[oi[_npv],oj[_npv]]
     share=np.maximum(share,_nw);del _nw
    del _npv
    # [EGOBI] mirrored woodland on real open ground: at least the open-ground weight of the canvas tile
    _wd=np.isin(source_blend[wmy,wmx,0],MIRROR_WOOD)
    if _wd.any():
     _ow=np.zeros(outside.shape,np.float32);_ow[oi[_wd],oj[_wd]]=_rowcols(mirror_open_w,byv)[oi[_wd],oj[_wd]]
     share=np.maximum(share,_ow);del _ow
    del _jx,_jy,wmx,wmy,_vs,_sea,_wd
   fy=((oh-1-yy+.5)/4-.5).clip(0,TH-1);y0=np.minimum(fy.astype(np.int32),TH-2);wy=(fy-y0).astype(np.float32)
   if _hh_win is not None:   # --- FIX2 KOREA GROUND ---
    _hs=_rowcols(_hh_win,byv)   # patch weight > 0: never protected land or plots (run 1 also required share > 0, which kept the zero-share tile edges as stairs)
    if _hs.any():share=np.where(_hs,_bilinear_rows(donor_share,y0,wy,_wx1_32,np.float32),share)
    del _hs
   score=np.stack([_bilinear_rows(c,y0,wy,_wx1_32,np.float32) for c in climate_score])
   # The noise (|.12 fbm| <= .12 per climate) can only change the winner where the
   # two best scores are within .24: evaluate it there only (identical result).
   # Two largest of the four scores without a sort (exact: comparisons only).
   _a,_b,_c,_d=score;_h1=np.maximum(_a,_b);_l1=np.minimum(_a,_b);_h2=np.maximum(_c,_d);_l2=np.minimum(_c,_d)
   near=(np.maximum(_h1,_h2)-np.maximum(np.minimum(_h1,_h2),np.maximum(_l1,_l2)))<.24;del _a,_b,_c,_d,_h1,_l1,_h2,_l2
   if near.any():
    if gpu_noise.active():
     # GPU: the block-grid noise at the near pixels only, all climates in one call (the bytes of both CPU branches).
     _nz=gpu_noise.fbm_at(_fx1[0]/2,fy[:,0]/2,near,(NQ,NR),1.5,2,seeds=range(90,90+len(climate_score)))
     for i in range(len(climate_score)):score[i][near]+=.12*_nz[i]
     del _nz
    elif near.mean()>.08:
     # fbm on the whole block grid then selected: bit-identical to the per-pixel path.
     gfx=np.broadcast_to(_fx1,near.shape);gfy=np.broadcast_to(fy,near.shape)
     for i in range(len(climate_score)):score[i][near]+=.12*fbm(gfx/2,gfy/2,(NQ,NR),1.5,2,seed=90+i)[near]
    else:
     nfx=np.broadcast_to(_fx1,near.shape)[near];nfy=np.broadcast_to(fy,near.shape)[near]
     for i in range(len(climate_score)):score[i][near]+=.12*fbm(nfx/2,nfy/2,(NQ,NR),1.5,2,seed=90+i)
   mix=np.zeros_like(base_val);picked=climate_keys[np.argmax(score,axis=0)];del score,near
   if _tl is not None:   # [section 9.3 treeline donor blend]
    picked=_treeline(picked,xx,yy,_bilinear_rows(_tl[0],y0,wy,_wx1_32,np.float32),_tl[1][np.rint(fy).astype(np.int64).clip(0,TH-1)[:,:1],np.rint(np.broadcast_to((xx+.5)/4-.5,picked.shape)).astype(np.int64).clip(0,TW-1)])
   # Bilinear per pixel: a per-tile lookup drew tile-sized steps on the belt edge.
   _wy64=fy-y0
   _oa=_bilinear_rows(oasis,y0,_wy64,_wx1_64,np.float64);_al=_bilinear_rows(alpine,y0,_wy64,_wx1_64,np.float64)
   if SINGLE_GROUND:
    _st_weight=_bilinear_rows(_material_steppe,y0,wy,_wx1_32,np.float32)
    # Purpose 101 is separate from all final material/source draws. Only source
    # family support is resolved here, as with the existing donor patch choice.
    _source_draw=_single_bake.draw.rows(row,oh)
    picked=_vd.steppe_donor(picked,_st_weight,
      lambda mask:_source_draw(101,np.zeros_like(picked,dtype=np.uint8),mask))
    del _st_weight,_source_draw
   else:
    _st=_rowcols(_material_steppe,byv)&(picked==3)
    if _st.any():picked=np.where(_st,9,picked)
   if sub_split is not None and (picked==0).any():   # --- OASIS GROUND --- subtropical donor by biome
    _su=.5+.5*gpu_noise.fbm(np.broadcast_to(xx/8,picked.shape),np.broadcast_to((oh-1-yy)/8,picked.shape),(NQ,NR),3,3,seed=108)
    picked=_og.subtropical_pick(picked,_bilinear_rows(sub_split[0],y0,wy,_wx1_32,np.float32),_bilinear_rows(sub_split[1],y0,wy,_wx1_32,np.float32),_su);del _su
   if _oa.any() or _al.any():
    _od=.5+.5*gpu_noise.fbm(np.broadcast_to(xx/6,_oa.shape),np.broadcast_to((oh-1-yy)/6,_oa.shape),(NQ,NR),2.5,4,seed=73)
    _sel=_od<_oa;picked=np.where(_sel,2,picked);share=np.where(_sel,1.,share);_osel=_sel   # --- USR3 BOSTEN PATCH ---
    # Alpine belts take the steppe donor (steppe_1-4: meadow tones, not oasis green).
    _sel=(_od<_al)&~_sel;picked=np.where(_sel,9,picked);share=np.where(_sel,1.,share)
   else:_osel=None   # --- USR3 BOSTEN PATCH ---
   # Dither, never average: a pixel takes the donor material where noise < share.
   dither=.5+.5*gpu_noise.fbm(np.broadcast_to(xx/8,share.shape),np.broadcast_to((oh-1-yy)/8,share.shape),(NQ,NR),3,4,seed=71)
   # Uniform per-pixel variable drawing the patch from the tile shares (patch_cum).
   pick_u=np.interp(gpu_noise.fbm(np.broadcast_to(xx/8,share.shape),np.broadcast_to((oh-1-yy)/8,share.shape),(NQ,NR),DONOR_GRAIN,3,seed=107),grain_q,np.linspace(0,1,257)).astype(np.float32)
   _jfx=np.broadcast_to(xx/8,share.shape);_jfy=np.broadcast_to((oh-1-yy)/8,share.shape)
   warp_x=np.rint(_bilinear_rows(donor_wx,y0,wy,_wx1_32,np.float32)+DONOR_JITTER*gpu_noise.fbm(_jfx,_jfy,(NQ,NR),1,2,seed=105)).astype(np.int32);warp_y=np.rint(_bilinear_rows(donor_wy,y0,wy,_wx1_32,np.float32)+DONOR_JITTER*gpu_noise.fbm(_jfx,_jfy,(NQ,NR),1,2,seed=106)).astype(np.int32);del _jfx,_jfy
   for climate,plist in patches.items():
    si,sj=np.nonzero(picked==climate)
    if not len(si):continue
    # Patch k+1 or later where the variable exceeds P(patch<=k), bilinear per pixel.
    _u=pick_u[si,sj];_y,_fy=y0[si,0],wy[si,0];_x,_fx=_x01[sj],_wx1_32[0,sj];_j=np.zeros(len(si),np.int32)
    for _cm in patch_cum[climate]:
     _j+=_u>(_cm[_y,_x]*(1-_fx)+_cm[_y,_x+1]*_fx)*(1-_fy)+(_cm[_y+1,_x]*(1-_fx)+_cm[_y+1,_x+1]*_fx)*_fy
    _c=np.asarray(plist,np.int32)[_j]
    # Each patch tiles with its own phase, so neighbouring patches never share mirror axes.
    px=(xx[0,sj]+warp_x[si,sj]+_j*DONOR_PHASE[0])%768;px=np.where(px<384,px,767-px)+_c[:,0]*8-192;py=(yy[si,0]+warp_y[si,sj]+_j*DONOR_PHASE[1])%768;py=np.where(py<384,py,767-py)+(hh-1-_c[:,1]*8)-192;px=px.clip(0,ww-1);py=py.clip(0,hh-1)
    mix[si,sj]=donor[py,px]
   if SINGLE_GROUND:
    if _osel is not None and _osel.any():mix[...,0]=np.where(_osel,OASIS_REMAP[mix[...,0].astype(np.uint8)],mix[...,0])
    _result,_owners,_stats=_single_bake.rows(row,row+n,base_val,mix,share,picked,source_blend[row:row+n],
                          base_band=np.where(dither<share,mix[...,1],base_val[...,1]))
    _sl=slice(_core_row-row,_core_end-row)
    if _written_rows[_core_row:_core_end].any():raise AssertionError('A final blend core row was written twice')
    cur[_core_row:_core_end]=_result[_sl]  # sole final RG write, disjoint core rows
    _owner_map[_core_row:_core_end]=_owners[_sl]
    _written_rows[_core_row:_core_end]=True
    return {f'owner_{name}':int((_owners[_sl]==i).sum()) for i,name in _gb.SOURCES.items()}
   _bans=_source_bans(xx,yy);_rejected={}
   mix=_woodland(mix,picked,xx,yy,gpu_noise.fbm,None if _src_flag is None else _src_flag[row:row+n],_bans,_rejected)   # --- DONOR WOODLAND ---
   if _osel is not None and _osel.any():mix[...,0]=np.where(_osel,OASIS_REMAP[mix[...,0].astype(np.uint8)],mix[...,0])   # --- USR3 BOSTEN PATCH ---
   take=(dither<share)[...,None];cur[row:row+n]=_tone_bridge(take,mix,base_val,share,dither,None if _src_flag is None else _src_flag[row:row+n],_bans,_rejected)
   return _rejected
  def _blend_rows_reference(row):
   yy=np.arange(row,min(row+256,oh),dtype=np.int32)[:,None];bx=(xx//4).clip(0,TW-1);by=((oh-1-yy)//4).clip(0,TH-1)
   share=donor_share[by,bx];outside=(xx<X0p)|(xx>X1p)|(yy<Y0p)|(yy>Y1p)
   # Warp the mirror source: a wide mirrored band showed its symmetry.
   wmx=reflect(np.clip(xx+mirror_jx[by,bx],0,ow-1).astype(np.int32),X0p,X1p);wmy=reflect(np.clip(yy+mirror_jy[by,bx],0,oh-1).astype(np.int32),Y0p,Y1p)
   base_val=cur[row:row+len(yy)].astype(np.float32)
   mirrored=source_blend[wmy,wmx].astype(np.float32);base_val[outside]=mirrored[outside]
   # A mirror source on vanilla sea carries sea-bed material: canvas land there
   # takes the donor as well.
   share=np.where(outside&(vanilla_sea[((oh-1-wmy)//4).clip(0,TH-1),(wmx//4).clip(0,TW-1)]|np.isin(mirrored[...,0].astype(np.uint8),MIRROR_SKIP)),1.,share)
   # --- NONPLAY GROUND --- (as _blend_rows) a mirror source on the repainted filler: the donor ramped in from the border
   share=np.where(outside&(nonplay_w[((oh-1-wmy)//4).clip(0,TH-1),(wmx//4).clip(0,TW-1)]>.5)&~(vanilla_sea[((oh-1-wmy)//4).clip(0,TH-1),(wmx//4).clip(0,TW-1)]|np.isin(mirrored[...,0].astype(np.uint8),MIRROR_SKIP)),np.maximum(share,nonplay_mirror_w[by,bx]),share)
   # [EGOBI] mirrored woodland on real open ground (as _blend_rows)
   share=np.where(outside&np.isin(mirrored[...,0].astype(np.uint8),MIRROR_WOOD),np.maximum(share,mirror_open_w[by,bx]),share)
   fx=np.broadcast_to((xx+.5)/4-.5,share.shape).clip(0,TW-1);fy=np.broadcast_to((oh-1-yy+.5)/4-.5,share.shape).clip(0,TH-1)
   x0=np.minimum(fx.astype(np.int32),TW-2);y0=np.minimum(fy.astype(np.int32),TH-2);wx=(fx-x0).astype(np.float32);wy=(fy-y0).astype(np.float32)
   score=np.stack([(c[y0,x0]*(1-wx)+c[y0,x0+1]*wx)*(1-wy)+(c[y0+1,x0]*(1-wx)+c[y0+1,x0+1]*wx)*wy for c in climate_score])
   # The noise (|.12 fbm| <= .12 per climate) can only change the winner where the
   # two best scores are within .24: evaluate it there only (identical result).
   top2=np.sort(score,axis=0)[-2:];near=(top2[1]-top2[0])<.24
   if near.any():
    nfx,nfy=fx[near],fy[near]
    for i in range(len(climate_score)):score[i][near]+=.12*fbm(nfx/2,nfy/2,(NQ,NR),1.5,2,seed=90+i)
   mix=np.zeros_like(base_val);picked=climate_keys[np.argmax(score,axis=0)]
   if _tl is not None:   # [section 9.3 treeline donor blend] (as _blend_rows)
    _tp=(_tl[0][y0,x0]*(1-wx)+_tl[0][y0,x0+1]*wx)*(1-wy)+(_tl[0][y0+1,x0]*(1-wx)+_tl[0][y0+1,x0+1]*wx)*wy
    picked=_treeline(picked,xx,yy,_tp,_tl[1][np.rint(fy).astype(np.int64),np.rint(fx).astype(np.int64)]);del _tp
   del score,top2,near,fx,fy,x0,y0,wx,wy
   # Bilinear per pixel: a per-tile lookup drew tile-sized steps on the belt edge.
   _fx=np.broadcast_to((xx+.5)/4-.5,share.shape).clip(0,TW-1);_fy=np.broadcast_to((oh-1-yy+.5)/4-.5,share.shape).clip(0,TH-1)
   _x0=np.minimum(_fx.astype(np.int32),TW-2);_y0=np.minimum(_fy.astype(np.int32),TH-2);_wx=_fx-_x0;_wy=_fy-_y0
   _oa=(oasis[_y0,_x0]*(1-_wx)+oasis[_y0,_x0+1]*_wx)*(1-_wy)+(oasis[_y0+1,_x0]*(1-_wx)+oasis[_y0+1,_x0+1]*_wx)*_wy
   _al=(alpine[_y0,_x0]*(1-_wx)+alpine[_y0,_x0+1]*_wx)*(1-_wy)+(alpine[_y0+1,_x0]*(1-_wx)+alpine[_y0+1,_x0+1]*_wx)*_wy
   del _fx,_fy,_x0,_y0,_wx,_wy
   _st=real_steppe[by,bx]&(picked==3)
   if _st.any():picked=np.where(_st,9,picked)
   if sub_split is not None and (picked==0).any():   # --- OASIS GROUND --- (same as _blend_rows)
    _fx=np.broadcast_to((xx+.5)/4-.5,share.shape).clip(0,TW-1);_fy=np.broadcast_to((oh-1-yy+.5)/4-.5,share.shape).clip(0,TH-1)
    _x0=np.minimum(_fx.astype(np.int32),TW-2);_y0=np.minimum(_fy.astype(np.int32),TH-2);_wx=(_fx-_x0).astype(np.float32);_wy=(_fy-_y0).astype(np.float32)
    _sp=[(c[_y0,_x0]*(1-_wx)+c[_y0,_x0+1]*_wx)*(1-_wy)+(c[_y0+1,_x0]*(1-_wx)+c[_y0+1,_x0+1]*_wx)*_wy for c in sub_split]
    picked=_og.subtropical_pick(picked,_sp[0],_sp[1],.5+.5*fbm(np.broadcast_to(xx/8,share.shape),np.broadcast_to((oh-1-yy)/8,share.shape),(NQ,NR),3,3,seed=108));del _fx,_fy,_x0,_y0,_wx,_wy,_sp
   _osel=None   # --- USR3 BOSTEN PATCH ---
   if _oa.any() or _al.any():
    _od=.5+.5*fbm(np.broadcast_to(xx/6,_oa.shape),np.broadcast_to((oh-1-yy)/6,_oa.shape),(NQ,NR),2.5,4,seed=73)
    _sel=_od<_oa;picked=np.where(_sel,2,picked);share=np.where(_sel,1.,share);_osel=_sel   # --- USR3 BOSTEN PATCH ---
    # Alpine belts take the steppe donor (steppe_1-4: meadow tones, not oasis green).
    _sel=(_od<_al)&~_sel;picked=np.where(_sel,9,picked);share=np.where(_sel,1.,share)
   # Dither, never average: a pixel takes the donor material where noise < share.
   dither=.5+.5*fbm(np.broadcast_to(xx/8,share.shape),np.broadcast_to((oh-1-yy)/8,share.shape),(NQ,NR),3,4,seed=71)
   _fx=np.broadcast_to((xx+.5)/4-.5,share.shape).clip(0,TW-1);_fy=np.broadcast_to((oh-1-yy+.5)/4-.5,share.shape).clip(0,TH-1)
   _x0=np.minimum(_fx.astype(np.int32),TW-2);_y0=np.minimum(_fy.astype(np.int32),TH-2);_wx=(_fx-_x0).astype(np.float32);_wy=(_fy-_y0).astype(np.float32)
   pick_u=np.interp(fbm(np.broadcast_to(xx/8,share.shape),np.broadcast_to((oh-1-yy)/8,share.shape),(NQ,NR),DONOR_GRAIN,3,seed=107),grain_q,np.linspace(0,1,257)).astype(np.float32)
   pick_j={cl:sum((pick_u>(c[_y0,_x0]*(1-_wx)+c[_y0,_x0+1]*_wx)*(1-_wy)+(c[_y0+1,_x0]*(1-_wx)+c[_y0+1,_x0+1]*_wx)*_wy).astype(np.int32) for c in cums) if cums else np.zeros(share.shape,np.int32) for cl,cums in patch_cum.items()}
   _jfx=np.broadcast_to(xx/8,share.shape);_jfy=np.broadcast_to((oh-1-yy)/8,share.shape)
   warp_x,warp_y=[np.rint((c[_y0,_x0]*(1-_wx)+c[_y0,_x0+1]*_wx)*(1-_wy)+(c[_y0+1,_x0]*(1-_wx)+c[_y0+1,_x0+1]*_wx)*_wy+DONOR_JITTER*fbm(_jfx,_jfy,(NQ,NR),1,2,seed=sd)).astype(np.int32) for c,sd in ((donor_wx,105),(donor_wy,106))]
   del _fx,_fy,_x0,_y0,_wx,_wy
   for climate,plist in patches.items():
    _j=pick_j[climate];_c=np.asarray(plist,np.int32)[_j]
    px=(xx+warp_x+_j*DONOR_PHASE[0])%768;px=np.where(px<384,px,767-px)+_c[...,0]*8-192;py=(yy+warp_y+_j*DONOR_PHASE[1])%768;py=np.where(py<384,py,767-py)+(hh-1-_c[...,1]*8)-192;px=px.clip(0,ww-1);py=py.clip(0,hh-1)
    sel=picked==climate;mix[sel]=donor[py,px][sel]
   _bans=_source_bans(xx,yy);_rejected={}
   mix=_woodland(mix,picked,xx,yy,fbm,None if _src_flag is None else _src_flag[row:row+len(yy)],_bans,_rejected)   # --- DONOR WOODLAND --- (as _blend_rows)
   if _osel is not None and _osel.any():mix[...,0]=np.where(_osel,OASIS_REMAP[mix[...,0].astype(np.uint8)],mix[...,0])   # --- USR3 BOSTEN PATCH --- (same remap as _blend_rows)
   take=(dither<share)[...,None];cur[row:row+len(yy)]=_tone_bridge(take,mix.astype(np.uint8),base_val.astype(np.uint8),share,dither,None if _src_flag is None else _src_flag[row:row+len(yy)],_bans,_rejected).astype(np.uint8)
   return _rejected
  # Threads for the row blocks from free commit (memory_budget.pool_size; each holds ~BLEND_THREAD_GB
  # of block temporaries): each block writes only its own rows, so any count gives the same bytes.
  _blend_workers=_gb.single_ground_workers() if SINGLE_GROUND else memory_budget.pool_size(BLEND_THREAD_GB,os.cpu_count() or 4,reserve_gb=memory_budget.STEP_RESERVE_GB,env='TW3K_BLEND_THREADS')
  if SINGLE_GROUND:print(f'[single-ground] workers {_blend_workers} (default1, cap2, 3GiB/worker + 3GiB reserve)',flush=True)
  _blend_pool=ThreadPoolExecutor(max_workers=_blend_workers)
  try:
   _source_rejections=[]
   _blend_rows_order=list(range(0,oh,64 if SINGLE_GROUND else 256))
   if SINGLE_GROUND:
    # Only traversal changes: world-coordinate draws and immutable sources make
    # every disjoint core independent. Start near a diagnosed failure so a retry
    # exercises that area first, then still generates every other row exactly once.
    _blend_start=int(os.environ.get('TW3K_BLEND_START_ROW','0'))
    if _blend_start not in _blend_rows_order:raise ValueError('TW3K_BLEND_START_ROW must name a 64-row core inside the map')
    _start_index=_blend_rows_order.index(_blend_start)
    _blend_rows_order=_blend_rows_order[_start_index:]+_blend_rows_order[:_start_index]
    print(f'[single-ground] traversal starts at row {_blend_start}; all {len(_blend_rows_order)} cores required',flush=True)
   for _batch_index,_batch_counts in enumerate(_blend_pool.map(_blend_rows,_blend_rows_order)):
    _source_rejections.append(_batch_counts)
    if SINGLE_GROUND and (_batch_index==0 or (_batch_index+1)%16==0):
     print(f'[single-ground] rows {min((_batch_index+1)*64,oh)}/{oh}',flush=True)
  finally:_blend_pool.shutdown(cancel_futures=SINGLE_GROUND)
  approach_report['source_candidates_rejected']={k:int(sum(v.get(k,0) for v in _source_rejections)) for k in sorted({k for v in _source_rejections for k in v})};del _source_rejections
  if not SINGLE_GROUND:
   # --- FIX-BATCH TONAL BUFFER --- (constants above): dense forest from the blend just
   # written, ring distance from it, then bright donor materials -> buffer materials.
   _f=np.isin(cur[...,0],list(WOODLAND_MATERIALS)).reshape(oh//4,4,ow//4,4).mean((1,3),dtype=np.float32)[::-1]
   _ft=np.zeros((TH,TW),np.float32);_ft[:_f.shape[0],:_f.shape[1]]=_f[:TH,:TW];del _f
   _bdense=box_blur(_ft,BUFFER_DENSE[0])>=BUFFER_DENSE[1];del _ft
   # --- FIX2 BUFFER SHAPE --- the ring distance was the 4-neighbour growth step (L1): every
   # small forest blob in the Kazakh/Tarim desert drew a 70-hex diamond of steppe ground
   # (offline 2048bq canvas). Now: Euclidean distance to the nearest dense-forest cell (BFS
   # nearest seed, then hypot), meandered by fbm, and weighted by the forest mass around that
   # seed (isolated blobs of a few hexes get no buffer; vanilla's bands follow large forests).
   _near,_bd=fastops.propagate_nearest(np.where(_bdense,np.arange(TH*TW,dtype=np.int32).reshape(TH,TW),0),_bdense,int(TONAL_BUFFER[0][-1]*1.45)+2)
   _reach=_bd<=TONAL_BUFFER[0][-1]*1.45+2;_sy,_sx=np.divmod(_near,TW);del _bd
   _de=np.where(_reach,np.hypot(x-_sx,y-_sy)*(1+BUFFER_MEANDER*fbm(q,r,(NQ,NR),8,3,seed=124)),1e6).astype(np.float32);del _sy,_sx
   _mass=box_blur(box_blur(_bdense.astype(np.float32),BUFFER_MASS[0]),BUFFER_MASS[0])
   _mf=np.where(_reach,np.clip((_mass.ravel()[_near.ravel()].reshape(TH,TW)-BUFFER_MASS[1])/BUFFER_MASS[2],0,1),0).astype(np.float32);del _near,_reach,_mass
   # The mass rule applies on real desert only (blurred RESOLVE 'desert' share): the northern steppe's
   # fragmented forest edges keep the full buffer (run 1 with the rule everywhere: north 6-12/12-20 hex
   # luminance 134/148 against vanilla 124/128).
   if REAL_TERRAIN.exists() and 'desert' in _bnames:_mf=np.maximum(_mf,1-box_blur(box_blur((real_bio==_bnames.index('desert')).astype(np.float32),8),8)*real_on)   # --- NONPLAY GROUND --- real_on (was real_w>0)
   buffer_w=(np.interp(_de,*TONAL_BUFFER)*_mf*donor_share*np.clip(real_gw*2,0,1)*physical).astype(np.float32);del _de,_mf
   # --- OASIS GROUND (oasis_ground.arid_share) --- no forest-tonal buffer on arid ground: times 1 - the
   # blurred share of desert and dry alpine/snow. The rule above gave every non-desert cell the full
   # buffer whatever the forest mass, so the dry Bogda/Tian Shan alpine around Turpan took steppe_3/4
   # over the sand (Jushi window: 48 % of its green). Humid steppe and forest edges (northern steppe,
   # Altai: arid share 0) keep it.
   if REAL_TERRAIN.exists() and 'desert' in _bnames:
    import oasis_ground as _og;buffer_w*=1-_og.arid_share(real_bio,_bnames,real_elev=real_elev)*real_on   # --- NONPLAY GROUND --- real_on (was real_w>0)
   _bq=np.quantile(fbm(*(np.mgrid[0:1024,0:1024][::-1]/4.),(NQ,NR),BUFFER_GRAIN,BUFFER_OCTAVES,seed=123),np.linspace(0,1,257))
   # --- end FIX2 BUFFER SHAPE
   def _buffer_rows(row):
    yy=np.arange(row,min(row+256,oh),dtype=np.int32)[:,None];byv=((oh-1-yy)//4).clip(0,TH-1)[:,0]
    blk=cur[row:row+len(yy)];wv=_rowcols(buffer_w,byv)
    # noise only where the weight is set and the material has a buffer swap (most ground has none)
    _m=(wv>0)&(BUFFER_TABLE[blk[...,1],blk[...,0]]!=blk[...,0]);si,sj=np.nonzero(_m)
    if not len(si):return 0
    # gpu_noise.fbm_at: the bytes of fbm on the gathered points (CPU fallback).
    u=np.interp(gpu_noise.fbm_at(xx[0]/8,(oh-1-yy[:,0])/8,_m,(NQ,NR),BUFFER_GRAIN,BUFFER_OCTAVES,seeds=(123,))[0],_bq,np.linspace(0,1,257))   # FIX2 BUFFER SHAPE: octaves
    hit=u<wv[si,sj];si,sj=si[hit],sj[hit]
    blk[si,sj,0]=BUFFER_TABLE[blk[si,sj,1],blk[si,sj,0]]
    return int(hit.sum())
   approach_report['tonal_buffer']={'dense_forest_tiles':int(_bdense.sum()),'weighted_tiles':int((buffer_w>0).sum()),'pixels_remapped':int(sum(_POOL.map(_buffer_rows,range(0,oh,256))))}
   if _src_flag is not None:approach_report['bridge_woodland_overwritten']={'tonal_buffer':_discard_blend_sources(cur,_src_flag)}
   print('Tonal buffer:',approach_report['tonal_buffer'],flush=True);del _bdense,buffer_w,_bq
   # --- end FIX-BATCH TONAL BUFFER
   # --- EDGE CLEANUP (edge_cleanup.py) --- forest/open edges to vanilla's shape statistics: majority
   # smoothing within 15 hexes of the large-scale woodland boundary (pixel dither and sub-hex tongues go,
   # the meander stays), woodland islands / clearings under ~6 hex^2 removed beyond 5 hexes; changed pixels
   # take the nearest kept pixel's material. Real-terrain canvas ground only (real_gw x donor_share x
   # physical > .5); no noise, deterministic. After the tonal buffer, before GROUND MATERIALS (whose
   # high-mountain mix lays cold_1, material 4). TW3K_EDGE_CLEANUP=0 skips.
   if os.environ.get('TW3K_EDGE_CLEANUP','1')!='0':
    import edge_cleanup as _ecl
    _tec=_time.time();approach_report['edge_cleanup_pixels']=_ecl.apply(cur,oh,ow,(real_gw*donor_share*physical).astype(np.float32),TH,TW,_POOL)
    if _src_flag is not None:approach_report['bridge_woodland_overwritten']['edge_cleanup']=_discard_blend_sources(cur,_src_flag)
    print('Edge cleanup pixels:',approach_report['edge_cleanup_pixels'],f'({_time.time()-_tec:.1f}s)',flush=True);del _tec
   # --- end EDGE CLEANUP
   # --- GROUND MATERIALS (ground_materials.py) --- gravel gobi, grey-brown plateau and tundra,
   # vanilla's high-mountain mix and the cold G band above a latitude-dependent snowline, on
   # real-terrain canvas ground only (real_gw x donor_share x physical; oasis excluded).
   # Bright sand/steppe source materials only; native ground untouched. TW3K_GROUND_MATERIALS=0 skips.
   # --- FARMS AND MUD (ground_materials.farm_mud_fields) --- vanilla farm blend materials (8-12) in river
   # valleys, on lake shores, wide lowland plains, around authored towns (beyond farm_fields' own complex)
   # and in desert oases; mud (14) on flat low ground next to river and lake hexes. Same pixel pass,
   # same ground only. gm_farm: the fields laid here, which FARM FIELDS' 'already farmed' rule ignores.
   # TW3K_FARM_MUD=0 skips (the GROUND MATERIALS bytes are then those without it).
   gm_cold=gm_farm=_gban=None
   import ground_materials as _gmw;_gwl=_gmw.WriterLedger(cur)   # --- BLEND WRITERS --- (blend_single_writer baseline, warn-only)
   if os.environ.get('TW3K_GROUND_MATERIALS','1')!='0' and REAL_TERRAIN.exists():
    import ground_materials as _gm
    _gf,_glon,_glat,_ggw,_gdw=_build_ground_fields() if _early_ground is None else _early_ground
    _early_ground=None   # transfer the one fields dictionary, including its ban rows, to the material pass
    # --- LANDFORM TILES (landform_tiles.py) --- the generator's own landform field for the gates, regression and replica
    if variant=='3k_dlc07_main_map':
     import landform_tiles as _lft;print('Landform tiles:',_lft.save(A,_gf,real_bio_eco,{'ground_w':_ggw,'desert_w':_gdw,'oasis':oasis,'keep':alpine,'river':real_river,'forest_density':(_fdt if _fdt is not None else _bp.forest_density(real_bio_eco,_bnames,q,r,(NQ,NR),fill=_ffl)),'forest_fill':_ffl,'biome_names':_bnames}),flush=True)
    del _ggw,_gdw
    _gfarm=os.environ.get('TW3K_FARM_MUD','1')!='0'
    if _gfarm:   # --- FARMS AND MUD ---
     import farm_fields as _gff
     _grk=[i for i,k in enumerate(names) if k.startswith('tw3k_riv_sea_')]
     _griv=np.isin(ids,_grk)[hy,hx]&~physical&outer;del _grk
     _glak=((ids==names.index('3k_main_sea_lake'))[hy,hx]&~physical&outer) if '3k_main_sea_lake' in names else np.zeros((TH,TW),bool)
     _gtw=[(int(t['town'][0])*2,int((t['town'][1]+t['town'][0]%2*.5)*2),_gff.kind_of(t.get('battle_layout'))) for t in plan['regions']]
     _tfm=_time.time()
     _gfm=_gm.farm_mud_fields(real_elev,real_bio_eco,_bnames,_glon,_glat,(real_gw*donor_share*physical).astype(np.float32),biome,surface,_griv,_glak,real_river>0,_gtw,q,r,(NQ,NR),oasis=oasis,cold=_gf['cold'])
     print('Farm/mud fields:',_gfm['parts'],f'({_time.time()-_tfm:.1f}s)',flush=True);_gf.update(_gfm);del _griv,_glak,_gtw,_gfm,_tfm
     _gfl=np.zeros(256,bool);_gfl[8:13]=True;_gpre=_gfl[cur[...,0]]
    del _glon,_glat
    _gq=_gm.make_quantiles((NQ,NR),farm=_gfarm)
    if _src_flag is not None:   # --- TONE BRIDGE / DONOR WOODLAND --- pixels the material pass re-samples as banned (interim, review)
     _sfr=_blend_source_ban_report(cur,_src_flag,_gf,_gm)
     approach_report['bridge_woodland_banned']=_sfr;print('Bridge / woodland pixels re-sampled as banned:',_sfr,flush=True);_src_flag=None
    _gr=list(_POOL.map(lambda row:_gm.apply_rows(cur,row,min(row+256,oh),oh,_gf,_gq),range(0,oh,256)))
    gm_cold=_gf['cold']
    _gwl.check(cur,'ground materials + farms/mud')   # --- BLEND WRITERS ---
    approach_report['ground_materials_fallback']=_gm.fallback_report();print('Ground materials own-mix fallback (intended change vs main):',approach_report['ground_materials_fallback'],flush=True)   # [STEP 4]
    # --- STEP 4 BAN LEDGER --- banned pixels left by the material pass (0: apply_rows asserts) and added by each later writer
    if _gf['L'].any():_gban=_gm.BanLedger(cur,_gf,oh,strict=os.environ.get('TW3K_BAN_ASSERT','')=='strict')
    approach_report['ground_materials']={'materials_px':int(sum(g_[0] for g_ in _gr)),'cold_band_px':int(sum(g_[1] for g_ in _gr)),
     'class_tiles':{k:int((v>.05).sum()) for k,v in _gf['classes'].items()}}
    if _gfarm:
     approach_report['ground_materials'].update({'farm_px':int(sum(g_[2] for g_ in _gr)),'mud_px':int(sum(g_[3] for g_ in _gr)),'farm_mud_tiles':_gf['parts']})
     gm_farm=_gfl[cur[...,0]]&~_gpre;del _gfl,_gpre
    print('Ground materials:',approach_report['ground_materials'],flush=True);del _gf,_gq,_gr,_gfarm
   # --- end GROUND MATERIALS
   chosen=weight=None;del climate_score
   print(f'[blend-timing] row loop: {_time.time()-_tb:.1f}s',flush=True);_tb=_time.time()
   # Town ground: vanilla towns stand on settlement/field materials, never on the
   # woodland material around them. Authored towns inside vanilla forest (Rinan,
   # live check 2026-10-19) kept the forest floor and read as lost in woods.
   # Copy the ground under a same-climate vanilla town (TOWN_GROUND, native hexes),
   # full within TOWN_GROUND_R[0] hexes and dithered out to TOWN_GROUND_R[1].
   r_in,r_out=TOWN_GROUND_R[0]*8,TOWN_GROUND_R[1]*8;span=int(r_out)+1
   oy,ox=np.mgrid[-span:span+1,-span:span+1];dist=np.hypot(ox,oy)
   grain=.5+.5*fbm(ox/8.,oy/8.,(0,0),1.2,3,seed=95)
   falloff=np.clip((r_out-dist)/(r_out-r_in),0,1);take_disc=grain<falloff
   painted=0;_tg_g=0;_tg_dry=0;_tg_forced=0;_tg_ban=0;_tg_ban_dry=0
   _town_dry_clear=os.environ.get('TW3K_WET_PIXEL_MASK','0')=='1'   # follows apply_climate.WET_PIXEL_MASK (same switch, same default)
   _town_clear=None
   if _town_dry_clear:
    import apply_climate as _town_climate
    _town_clear=_town_climate.town_clear_hex(cells)
   # Forest towns (FOREST_TOWN), classified on the blend before any town ground.
   _fr=int(FOREST_TOWN[1]*8);_fy,_fx=np.mgrid[-_fr:_fr+1,-_fr:_fr+1];_fd=np.hypot(_fx,_fy)/8;_fring=(_fd>=FOREST_TOWN[0])&(_fd<FOREST_TOWN[1]);del _fy,_fx,_fd
   _wood=np.array(sorted(WOODLAND_MATERIALS),np.uint8);forest_towns=set()
   for town in plan['regions']:
    tq,tr=town['town'];cx=int(tq*8);cyy=oh-1-int(round((tr+tq%2*.5)*8))
    if _fr<=cx<ow-_fr and _fr<=cyy<oh-_fr and np.isin(cur[cyy-_fr:cyy+_fr+1,cx-_fr:cx+_fr+1,0][_fring],_wood).mean()>=FOREST_TOWN[2]:forest_towns.add(town['region'])
   # Only towns standing on vanilla woodland ground: elsewhere (desert oases of
   # the Western Regions) the copied settlement ground read as repeated green
   # discs (live check 2026-10-19 c). Forest towns keep their forest floor.
   for town in plan['regions']:
    tq,tr=town['town'];cx=int(tq*8);cyy=oh-1-int(round((tr+tq%2*.5)*8))
    if not(0<=cyy<oh and 0<=cx<ow):continue
    _normal_town=int(cur[cyy,cx,0]) in WOODLAND_MATERIALS and town['region'] not in forest_towns
    if not _normal_town and not _town_dry_clear:continue
    if not(span<=cx<ow-span and span<=cyy<oh-span):continue
    win=(slice(cyy-span,cyy+span+1),slice(cx-span,cx+span+1))
    _tty=((oh-1-(cyy+oy))//4).clip(0,TH-1);_ttx=((cx+ox)//4).clip(0,TW-1)
    prot=_protected[_tty,_ttx];_clear=None
    if _town_dry_clear:
     _tq,_tr=_frame.tile_hex(_ttx,_tty);_clear=_town_clear[_tr,_tq]
    # Forest/non-woodland towns retain their floor except actual wetland inside the town clearance.
    if not _normal_town and not((_clear&(cur[win][...,0]==31)&~prot).any()):continue
    cl=int(biome[min(int((tr+tq%2*.5)*2),TH-1),min(tq*2,TW-1)]);
    if cl not in TOWN_GROUND:raise RuntimeError(f"town {town['region']} has tile climate {cl}, not in TOWN_GROUND {sorted(TOWN_GROUND)}: every town must have a known climate")
    dq,dr=TOWN_GROUND[cl]
    dcx=int(dq*8);dcy=hh-1-int(round((dr+dq%2*.5)*8))
    if not(span<=dcx<ww-span and span<=dcy<hh-span):continue
    _template=donor[dcy-span:dcy+span+1,dcx-span:dcx+span+1]
    # TOWN SPEC: the copy never writes a slot banned for the town's own tiles (v18-v20: 926 drymtn pixels, ban ledger)
    _tban=_gban.banned(_template[...,0],cyy-span,cx-span) if _gban is not None else None
    try:m,_dry,_nb=_town_ground_selection(cur[win][...,0],_template[...,0],take_disc,prot,_normal_town,_clear,_town_dry_clear,_tban)
    except ValueError as _error:raise ValueError(f"town {town['region']} (climate {cl}, donor {dq},{dr}): {_error}") from _error
    _tg_ban+=_nb[0];_tg_ban_dry+=_nb[1];_tg_dry+=int(_dry.sum());_tg_forced+=int((_dry&~take_disc).sum()) if _normal_town else int(_dry.sum())
    _tg_g+=int((cur[win][m,1]!=_template[m,1]).sum())   # [STEP 4] G the old two-channel copy would have changed
    cur[win][m,0]=_template[m,0];painted+=1   # [STEP 4] material channel only; normal and dry clearance share this one write
   print(variant,'town ground painted for',painted,'authored towns;',len(forest_towns),'forest towns keep their floor',flush=True)
   approach_report['town_ground_g_kept_px']=_tg_g;print(variant,'town ground: G pixels kept (old copy would change)',_tg_g,flush=True)   # [STEP 4]
   approach_report['town_ground_dry_clear']={'enabled':_town_dry_clear,'wetland_pixels_selected':_tg_dry,'extra_pixels_outside_normal_selection':_tg_forced}
   approach_report['town_ground_banned_skipped']={'pixels':_tg_ban,'dry_clear_pixels':_tg_ban_dry};print(variant,'town ground: banned template pixels kept on the ground materials',approach_report['town_ground_banned_skipped'],flush=True)   # TOWN SPEC
   del _town_clear
   if HH is not None:print(variant,'patch forest-town floor pixels',HH.paint_forest_floor(cur,WOODLAND_MATERIALS),flush=True)   # --- HEHUANG PATCH (hehuang_terrain.py) --- korea forest towns stand on the woodland floor
   print(f'[blend-timing] town ground: {_time.time()-_tb:.1f}s',flush=True)
   if _gban is not None:_gban.check(cur,'town ground')   # --- STEP 4 BAN LEDGER ---
   _gwl.check(cur,'town ground')   # --- BLEND WRITERS ---
   # --- FARM FIELDS (farm_fields.py) --- vanilla farm-town field complexes (materials 8-12) around
   # authored towns, by town kind and climate, cut by vanilla's slope/height/water rules
   import farm_fields,strategic_ink as _ffsi
   from native_edit_mask import edited_tiles as _ffet
   _ffb=_ffsi.load();_ffr=farm_fields.paint(cur,donor,plan,biome,surface,alpha,physical,_protected,authored_plot[hy,hx],oasis,_ffb[0],_ffb[1],edited=_ffet(variant,halo=0)[:TH,:TW],ignore=gm_farm,filler=farm_fields.filler_keep(cells,names,hy,hx) if NONPLAY_GROUND else np.zeros((TH,TW),np.float32));del _ffb,gm_farm   # ignore: --- FARMS AND MUD ---
   (A/f'analysis/farm-fields-{variant}.json').write_text(json.dumps(_ffr,ensure_ascii=False,indent=1),encoding='utf-8')
   print(variant,'farm fields',json.dumps({k_:v_ for k_,v_ in _ffr.items() if k_!='towns'}),flush=True);del _ffr
   print(f'[blend-timing] farm fields: {_time.time()-_tb:.1f}s',flush=True)
   if _gban is not None:_gban.check(cur,'farm fields')   # --- STEP 4 BAN LEDGER ---
   _gwl.check(cur,'farm fields')   # --- BLEND WRITERS ---
   # --- end FARM FIELDS
   # --- CLIMATE PATCH (apply_climate.py) --- wetland blend: material 31 'wetlands' on the
   # map.hex wetland hexes outside the native slice (apply_climate.py --apply sets them), as
   # vanilla paints its marshes; G band from the hex's gameplay climate. TW3K_CLIMATE_WETLANDS=0 skips.
   if os.environ.get('TW3K_CLIMATE_WETLANDS','1')!='0':
    import apply_climate as _climate_patch
    _wet_px=_climate_patch.paint_wetland_blend(cur,cells,_protected,names,filler=NONPLAY_GROUND)
    approach_report['climate_patch_wetland_blend_pixels']=_wet_px;print(variant,'climate patch: wetland blend pixels',_wet_px,flush=True)
   # --- END CLIMATE PATCH ---
   # --- NONPLAY GROUND --- guard: vanilla playable land within 3 hexes of the repainted filler and outside
   # the intended edits (edit alpha 0) keeps its pre-polish pixels; the farm, mud and desert passes sample
   # their tile fields bilinearly and bled up to a hex across the filler edge (11 native hexes, scratch run).
   _np_near=(box_blur((nonplay_hex>0).astype(np.float32),3)>0)&_rf.vanilla_playable_hex()
   _np_guard=(_np_near[hy,hx]&(alpha<=0))[::-1]   # tile grid, rows north-first like cur
   if _np_guard.any():
    _gp=np.repeat(np.repeat(_np_guard,4,axis=0),4,axis=1)[:oh,:ow];_pre=np.frombuffer(src,dtype=np.uint8,offset=128).reshape(oh,ow,2)
    cur[_gp]=_pre[_gp];approach_report['nonplay_guard_px']=int(_gp.sum());del _gp,_pre
   del _np_near,_np_guard
   _gwl.check(cur,'wetlands + nonplay guard');approach_report['blend_writers']=_gwl.finish();print(variant,'blend writers',approach_report['blend_writers'],flush=True)   # --- BLEND WRITERS ---
   if variant=='3k_dlc07_main_map':(A/'analysis/blend-writers.json').write_text(json.dumps(approach_report['blend_writers'],indent=1)+chr(10),encoding='utf-8')
   del _gwl
   if _gban is not None:   # --- STEP 4 BAN LEDGER --- wetlands and the nonplay guard, then the report
    _gban.check(cur,'wetlands + nonplay guard');approach_report['ban_ledger']=_gban.report;print(variant,'ban ledger',_gban.report,flush=True);_gban=None
  else:
   assert _written_rows.all(),'Single selector left output rows unwritten'
   _owner_map.flush()
   _owner_counts=approach_report['source_candidates_rejected']
   _single_report={'schema':2,'mode':'single-selection','pixels_rewritten':0,'pixels_rewritten_2plus':0,
     'selection_count':oh*ow-_owner_counts.get('owner_preserved',0),
     'copy_count':_owner_counts.get('owner_preserved',0),'core_row_ranges':[[i,min(i+64,oh)] for i in range(0,oh,64)],
     'owners':_owner_counts,'protected_pixels':_owner_counts.get('owner_preserved',0),
     'water':_single_bake.water_provenance,'visual_domains':_visual_domains['meta']}
   assert sum(_owner_counts.values())==oh*ow,'Single selector did not own every output pixel'
   approach_report['blend_writers']=_single_report
   approach_report['ground_materials']={'class_tiles':{k_:int((v_>.05).sum()) for k_,v_ in _gf['classes'].items()}}
   approach_report['climate_patch_wetland_blend_pixels']=_owner_counts.get('owner_wetland',0)
   approach_report['nonplay_guard_px']=0
   if variant=='3k_dlc07_main_map':
    (A/'analysis/blend-writers.json').write_text(json.dumps(_single_report,indent=1)+chr(10),encoding='utf-8')
   print('Single material selection:',json.dumps(_owner_counts),flush=True)
   chosen=weight=None
   del _single_bake,_owner_map,_early_ground,_gf,_glon,_glat,_ggw,_gdw,climate_score
   if _water_field is not None:del _water_field,_water_reader,_native_water,_water_lf,_water_keys
  _bb=src[:128]+cur.tobytes();put(k,_bb)
  if variant=='3k_dlc07_main_map':   # --- BLEND CLIMATE TILES (blend_climate_tiles.py) --- the G band per tile for the tile bake (never in --cache-only)
   import blend_climate_tiles as _bct;print('Blend climate tiles:',_bct.save(A,_bb,cur),flush=True)
  del _bb,cur,xx,chosen,weight,donor
  _mark('variant: material blend')
  climate=cm_grid(base(pre+'climate_map.cm')).copy();mask=alpha[::-1]>.5;climate[mask]=biome[::-1][mask]
  if gm_cold is not None:_gcm=(gm_cold>.5)[::-1]&mask;climate[_gcm]=1;del _gcm   # --- GROUND MATERIALS --- climate_map.cm cold (tile_list order) where the blend took the cold band; tile_map/donor climate unchanged
  climate[TH-_frame.NATIVE_TILE_Y1:TH-_frame.NATIVE_TILE_Y0,_frame.NATIVE_TILE_X0:_frame.NATIVE_TILE_X1]=cm_grid(asset(pre+'climate_map.cm'))
  if HH is not None:_hm=HH.climate_mask(q,r)[::-1];climate[_hm]=biome[::-1][_hm];del _hm   # --- HEHUANG PATCH (hehuang_terrain.py) ---
  put(pre+'climate_map.cm',cm_write(base(pre+'climate_map.cm'),climate))
  # Camera and city queries sample exactly the same surface.
  _mark('variant: climate map')
  k=f'campaign_maps/{variant}/camera_heightmap.png';im=Image.open(io.BytesIO(base(k)));scale=float(im.info['height_scale']);original_camera=np.asarray(im);a=np.rint(np.maximum(surface[::-1],0)/scale).clip(0,65535).astype(np.uint16);a[alpha[::-1]==0]=original_camera[alpha[::-1]==0];assert np.all(a[alpha[::-1]==0]==original_camera[alpha[::-1]==0]);png=Image.fromarray(a);meta=PngImagePlugin.PngInfo()
  for key,val in im.info.items():
   if isinstance(val,str):meta.add_text(key,val)
  stream=io.BytesIO();png.save(stream,format='PNG',pnginfo=meta);put(k,stream.getvalue())
  # Land meshes that exist only in the exported canvas (absent from the native
  # packs) are not rebuilt below. Reset them from the geometry baseline so every
  # downstream step starts from the same input; otherwise later passes stack a
  # new lattice onto the previous run's output each time the pipeline runs.
  for base_mesh in sorted((BASE/pre/'global_meshes').glob('land_mesh_*.rigid_model_v2')):
   if pre+'global_meshes/'+base_mesh.name in idx:continue
   for part in (base_mesh,base_mesh.with_suffix('.compressed_map')):
    if part.exists():put(pre+'global_meshes/'+part.name,part.read_bytes())
  # Rebuild perimeter meshes at half-hex spacing; keep native mesh identifiers.
  _mark('variant: camera heightmap + mesh reset')
  # Meshes are independent (own source, own output files): polished on the thread
  # pool, numpy releases the GIL in the sampling. Same bytes as one by one.
  def _polish_mesh(k):
   source=asset(k);mp=struct.unpack_from('<I',source,152)[0];typ,flags,size,vo,vc,io_,ic=struct.unpack_from('<HH5I',source,mp);box=struct.unpack_from('<6f',source,mp+24)
   verts=np.frombuffer(source,dtype='<f4',offset=mp+vo,count=vc*4).reshape(vc,4).copy();verts[:,0]+=SX;verts[:,2]+=SZ
   ax=_frame.tile_coord(verts[:,0]);az=_frame.tile_coord(verts[:,2]);aw=sample(alpha,ax,az);newy=sample(surface,ax,az);changed_vertices=aw>0;seat_on_surface(verts,newy,changed_vertices,faces=np.frombuffer(source,dtype='<u2',offset=mp+io_,count=ic))
   ids_=np.frombuffer(source,dtype='<u2',offset=mp+io_,count=ic).copy().tolist();x0,y0,z0,x1,y1,z1=box;boxes=[]
   if x0<1:boxes.append((0,x0+SX,z0+SZ,z1+SZ))
   if x1>594:boxes.append((x1+SX,_frame.WORLD_WIDTH,z0+SZ,z1+SZ))
   if z0<1:boxes.append((max(0,x0+SX),min(_frame.WORLD_WIDTH,x1+SX),0,z0+SZ))
   # Out to every canvas edge, corners included (the north edge used to stop
   # 1.5 hexes short at (H-1)*DX and the east corners were missing).
   top=_frame.RAW_MESH_HEIGHT
   if z1>468:boxes.append((max(0,x0+SX),min(_frame.WORLD_WIDTH,x1+SX),z1+SZ,top))
   if x0<1 and z0<1:boxes.append((0,SX,0,SZ))
   if x0<1 and z1>468:boxes.append((0,SX,z1+SZ,top))
   if x1>594 and z0<1:boxes.append((x1+SX,_frame.WORLD_WIDTH,0,SZ))
   if x1>594 and z1>468:boxes.append((x1+SX,_frame.WORLD_WIDTH,z1+SZ,top))
   for ax,bx,az,bz in boxes:
    if bx<=ax or bz<=az:continue
    step=DX/2
    while True:
     nx=math.ceil((bx-ax)/step);nz=math.ceil((bz-az)/step)
     if len(verts)+(nx+1)*(nz+1)<65000:break
     step*=1.15
    xx,zz=np.meshgrid(np.linspace(ax,bx,nx+1),np.linspace(az,bz,nz+1))
    # Coarse extension grids (step up to ~4.5 hexes past the vertex budget): one
    # vertex on each town's display position (HEX_DX: where the engine draws the
    # model, not the block corner q*DX), or the town sits inside a tilted triangle.
    if step>DX:snap_grid_to_towns(xx,zz,[c['town'] for c in plan['regions']],_frame.HEX_DX)
    n=len(verts);yy=sample(surface,xx/DX*2,zz/DX*2);extra=np.column_stack((xx.ravel(),yy.ravel(),zz.ravel(),np.ones(xx.size))).astype('<f4');verts=np.vstack((verts,extra))
    rr,cc=np.mgrid[:nz,:nx];aa=n+rr*(nx+1)+cc;bb=aa+nx+1
    # Split each quad on the diagonal whose ends differ least in height. One fixed
    # diagonal on a 4-hex grid cut lake floors into triangles with parallel straight
    # deep/shallow edges (live 2048j, Yangtong lake; grid step 2.669 there).
    _g=yy.reshape(nz+1,nx+1);_d1=np.abs(_g[:-1,:-1]-_g[1:,1:]);_d2=np.abs(_g[:-1,1:]-_g[1:,:-1]);_alt=(_d2<_d1)[...,None]
    _main=np.stack((aa,bb+1,bb,aa,aa+1,bb+1),axis=2);_other=np.stack((aa,aa+1,bb,aa+1,bb+1,bb),axis=2)
    ids_.extend(np.where(_alt,_other,_main).ravel().tolist())
   assert len(verts)<65536;vb=verts.astype('<f4').tobytes();ib=np.asarray(ids_,dtype='<u2').tobytes();hd=bytearray(source[:mp+vo]);newbox=(float(verts[:,0].min()),float(verts[:,1].min()),float(verts[:,2].min()),float(verts[:,0].max()),float(verts[:,1].max()),float(verts[:,2].max()))
   struct.pack_into('<II',hd,144,len(vb),len(ib));struct.pack_into('<HH5I',hd,mp,typ,flags,vo+len(vb)+len(ib),vo,len(verts),vo+len(vb),len(ids_));struct.pack_into('<6f',hd,mp+24,*newbox);put(k,hd+vb+ib)
   ck=k.replace('.rigid_model_v2','.compressed_map');cm=asset(ck);native=cm_grid(cm);lo=struct.unpack_from('<f',cm,30)[0];hi=struct.unpack_from('<f',cm,42)[0]
   cw=max(native.shape[1],math.ceil((newbox[3]-newbox[0])/(DX/4)/16)*16);ch=max(native.shape[0],math.ceil((newbox[5]-newbox[2])/(DX/4)/16)*16);xx,zz=np.meshgrid(np.linspace(newbox[0],newbox[3],cw),np.linspace(newbox[2],newbox[5],ch));px=(xx-SX-box[0])/(box[3]-box[0])*(native.shape[1]-1);py=(zz-SZ-box[2])/(box[5]-box[2])*(native.shape[0]-1)
   rawheight=lo+sample(native,px,py)/65535*(hi-lo);aw=sample(alpha,_frame.tile_coord(xx),_frame.tile_coord(zz));edited_height=aw>0;rawheight[edited_height]=sample(surface,(xx/DX*2)[edited_height],(zz/DX*2)[edited_height])
   nl=min(float(rawheight.min()),newbox[1]);nh=max(float(rawheight.max()),newbox[4],nl+.001);arr=np.rint((rawheight-nl)/(nh-nl)*65535).clip(0,65535).astype('<u2');put(ck,cm_write(cm,arr,(newbox[0],nl,newbox[2],newbox[3],nh,newbox[5])))
   return len(verts)
  _mesh_counts=list(_POOL.map(_polish_mesh,[k for k in idx if k.startswith(pre+'global_meshes/land_mesh_') and k.endswith('.rigid_model_v2')]))
  changed=len(_mesh_counts);peak_vertices=max([0]+_mesh_counts)
  # Keep city safety and elevations as an offline reference for runtime checks. Height:
  # the final DDS at the town's display position, where the engine draws it (up to a block
  # NE of the anchor texel; the tile surface is exact only on whole tiles). encoded was
  # freed after the projection, so read back the lf_height_map.dds put there (rows north-first).
  _lf=(S/(pre+'lf_height_map.dds')).read_bytes();_lf=np.frombuffer(_lf,dtype='<u2',offset=128).reshape(struct.unpack_from('<2I',_lf,12))[::-1]
  def _town_ground(q,r):
   p=np.array(_frame.display_position(q,r))/(DX,_frame.DZ)*8;return float(LEVEL+(sample(_lf,p[:1],p[1:])[0]/65535-NORM)*GAIN)
  towns=[{'region':c['region'],'town':c['town'],'height':_town_ground(*c['town']),'plateau_height':plateau_levels[town_groups[c['region']]],'local_reference_height':town_reference[c['region']],'plateau_adjusted_to_native_anchors':abs(plateau_levels[town_groups[c['region']]]-town_reference[c['region']])>.001} for c in plan['regions']];del _lf
  # Every town stands on its plot level at its display position (quantisation is ~1e-4);
  # a town pixel left off its plot owner used to slope away unnoticed (Hanseong_resource_1, .58 vs 1.0).
  _off_level=[(t['region'],round(t['height'],3),round(t['plateau_height'],3)) for t in towns if abs(t['height']-t['plateau_height'])>1e-3]
  if _off_level:raise RuntimeError(f'Town height off its plot level ({variant}): {_off_level}')
  np.save(A/f'analysis/polished-surface-{variant}.npy',surface)
  report['variants'].append({'variant':variant,'land_meshes':changed,'largest_mesh_vertices':peak_vertices,'raw_height_range':[float(surface.min()),float(surface.max())],'towns':towns,'town_approach_constraints':approach_report})
  _mark('variant: coarse meshes')
  print('Height, blend, climate, camera and coarse geometry synchronized',variant,flush=True)
 alpha=coverage_alpha
 if LOOP_ONLY:
  for v in POLISH_VARIANTS:
   np.save(A/f'analysis/polished-edit-alpha-part-{v}.npy',alpha)
   (A/f'analysis/geographic-terrain-polish-part-{v}.json').write_text(json.dumps([e for e in report['variants'] if e['variant']==v],ensure_ascii=False),encoding='utf-8')
  print('Loop-only polish finished',POLISH_VARIANTS,flush=True);sys.exit(0)
 np.save(A/'analysis/polished-edit-alpha.npy',alpha)
elif MERGE:
 # Same combined state a serial run reaches after its variant loop.
 alpha=np.maximum.reduce([np.load(A/f'analysis/polished-edit-alpha-part-{v}.npy') for v in ALL_VARIANTS]);np.save(A/'analysis/polished-edit-alpha.npy',alpha)
 report['variants']=[e for v in ALL_VARIANTS for e in json.loads((A/f'analysis/geographic-terrain-polish-part-{v}.json').read_text(encoding='utf-8'))]
 surface=np.load(A/f'analysis/polished-surface-{ALL_VARIANTS[-1]}.npy');norm=np.clip(NORM+(surface-LEVEL)/GAIN,0,1)
 variant_dds=None
else:
 alpha=np.load(A/'analysis/polished-edit-alpha.npy')
 surface=np.load(A/f'analysis/polished-surface-{ALL_VARIANTS[-1]}.npy');norm=np.clip(NORM+(surface-LEVEL)/GAIN,0,1)
 report=json.loads((A/'analysis/geographic-terrain-polish.json').read_text(encoding='utf-8')) if (A/'analysis/geographic-terrain-polish.json').exists() else report
(A/'analysis/geographic-terrain-polish.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
_mark('end of variant loop / merge')
# Cover replacement cells with actual 8x8 native footprints and cache templates.
pre=f'terrain/campaigns/{_frame.CACHE_DONOR_VARIANT}/';biome=apply_climate_override(inherit_climate(biome,asset(pre+'climate_map.cm'),asset(pre+'tile_list.bin'),asset))
if HH is not None:biome=HH.climate(biome,q,r)   # --- HEHUANG PATCH (hehuang_terrain.py) ---
# --- TILE CLIMATE RULE (tile bake, pipeline-bake-design 9.2; with step 4, user choice 1 of 2026-10-26) --- canvas
# tiles take blend_climate_tiles.tile_rule: on the native ring (outside the native rectangle, NATIVE_RING) the native
# climate carried out of the rectangle, elsewhere the ground blend's G band climate (analysis/blend-climate-tiles.npz,
# written with the blend; missing/stale/schema 1 -> error). The rule lives there once, read here and by the invariant.
# The fine tiles follow the ground (v18: 1.59M canvas cells of cold tiles on subtropical ground) and meet the native
# tiles without a climate cut. The native rectangle keeps its inherited climate. Invariant: stage_invariants.
# tile_climate_matches_rule. TW3K_TILE_CLIMATE_BAND=0 turns it off (stop-loss switch).
if os.environ.get('TW3K_TILE_CLIMATE_BAND','1')=='1':
 import blend_climate_tiles as _bct
 _bct.add_ring(A,biome,q,r,(NQ,NR))   # ring + native from this tile bake's own climate grid, once (blend_climate_tiles.add_ring)
 _bct_r=_bct.load(A,(S/'terrain/campaigns/3k_dlc07_main_map/global_map/global_blend.dds').read_bytes(),fields=True)
 if _bct_r is None:raise RuntimeError(f'{A/_bct.NAME} missing: polish writes it with the blend (set TW3K_TILE_CLIMATE_BAND=0 to keep the inherited tile climate)')
 _bct_rule=_bct.tile_rule(_bct_r);del _bct_r;assert _bct_rule.shape==biome.shape,(_bct_rule.shape,biome.shape)
 biome=np.where(outer,_bct_rule,biome).astype(biome.dtype);del _bct_rule
np.save(A/'analysis/polished-tile-climate.npy',biome);native=asset(pre+'tile_map.tiles');ow,oh=_frame.NATIVE_TILE_W,_frame.NATIVE_TILE_H;on=ow*oh;n=TW*TH;tail=native[on*60:];pos=4;paths=[]
_mark('cache: climate inheritance + native layers')
for i in range(struct.unpack_from('<I',tail)[0]):length=struct.unpack_from('<H',tail,pos)[0];pos+=2;paths.append(tail[pos:pos+length].decode());pos+=length
import native_tile_grid as _ntg   # the vanilla cache base on the canvas grid (one place, with tile_bake's list base)
DT=_ntg.CACHE_DT;layers=[]
for i in range(3):
 layer=np.zeros((TH,TW),dtype=DT);layer['flags']=16;layer[_frame.NATIVE_TILE_SLICE]=_ntg.cache_layer(native,i);layers.append(layer)
# An 8x8 tile cannot be stretched to 16x16. Use a complete native land template.
native0=np.frombuffer(native,dtype=DT,count=on).reshape(oh,ow)
generic=paths.index('terrain\\tiles\\campaign\\generic\\generic_8x8\\');template=None
for yy in range(0,oh-16,16):
 for xx in range(0,ow-16,16):
  block=native0[yy:yy+16,xx:xx+16];orig=np.argwhere(block['path']==generic)
  if len(orig)==1:
   oy,ox=orig[0];parent=(yy+oy)*ow+xx+ox
   if np.all((block['path']==generic)|(block['path']==-parent-1)):
    template=block.copy();template_xy=(xx,yy);template_parent=(int(ox),int(oy));break
 if template is not None:break
assert template is not None
_mark('cache: template search')
# Replace native cache owners only where the LF really changed (shared edit
# definition without its halo: an owner is removed whole, so the halo alone
# discarded large vanilla tiles that merely border an edit; a bordering owner
# keeps its tile and sync_native_cache_height_ranges recalibrates its range).
# Alpha alone also covers blurred rings of untouched ground.
from native_edit_mask import edited_tiles
lf_edited=np.logical_or.reduce([edited_tiles(v,halo=0) for v in ALL_VARIANTS])
intended=((alpha>.02)&lf_edited[:alpha.shape[0],:alpha.shape[1]])|outer
# Vanilla fine tiles end on the straight native border, so the canvas floor
# tiles met them in a straight line with right angles (south of Dunhuang, live
# check 2026-10-19). On unprotected (non-playable) native land, replace them in
# a meandering ring inside the border as well; vanilla playable land keeps its tiles.
tile_ring=(edge>=0)&(edge<TILE_RING_MIN+TILE_RING_MEANDER*np.clip(.5+.7*gpu_noise.fbm(q,r,(NQ,NR),14,3,seed=93),0,1))&~_protected&physical
tile_ring|=climate_override
intended|=tile_ring
# --- HEHUANG PATCH (hehuang_terrain.py) --- vanilla fine tiles inside the meandered w > .3 contour are replaced by floors of the patch climate
if HH is not None:_hcut=HH.cache_cut(q,r)&physical;intended|=_hcut;report['hehuang']={**HH.report(),'cache_cut_cells':int(_hcut.sum())};del _hcut
_mark('cache: edited tiles + intended mask')
for _v in POLISH_VARIANTS:
 _p=A/f'analysis/polished-edit-alpha-{_v}.npy'
 if _p.exists():_a=np.load(_p);_a[tile_ring]=np.maximum(_a[tile_ring],.03);np.save(_p,_a)
alpha[tile_ring]=np.maximum(alpha[tile_ring],.03)
if '--skip-cache' in sys.argv:
 # Pipeline serial run: sync_final_terrain_surface.py rebuilds the cache from here with --cache-only
 # on the final DDS; everything it reads from this run (surfaces, alphas incl. the ring) is saved.
 (A/'analysis/geographic-terrain-polish.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
 print('Polish done; tile cache left to sync_final_terrain_surface.py --cache-only',flush=True)
 sys.exit(0)
# Land tile cache footprints are twice their nominal database dimensions.
# Water tile footprints use the nominal size. Infer templates from real ownership,
# rather than assuming a cropped quadrant is a whole land tile.
# --- road tiles (road_tiles.py) 1/3: new road hexes on kept native ground join the cache edit (their
# generic floors are re-laid around the road pieces); hexes on other native tiles keep them.
import road_tiles as _rt,map_hex as _rmh
_rt_mask,_rt_dangling=_rt.mutual_masks(_rt.road_masks(cells));_rt_vmask=np.zeros_like(_rt_mask);_rt_vmask[_frame.NATIVE_HEX_SLICE]=_rt.road_masks(_rmh.read(_frame.VANILLA_MAP)['cells'])
_rt_join,_rt_join_hexes,_rt_join_refused=_rt.native_join_blocks((_rt_mask>0)&(_rt_vmask==0)&~_rt.blocks_in(intended,H,W),layers,paths);intended=intended|_rt_join
# --- end road tiles 1/3
replacement,owner_removal_counts=remove_edited_owners(layers,intended)
_mark('cache: remove edited owners')
np.save(A/'analysis/cache-intended-edit-mask.npy',intended)
np.save(A/'analysis/cache-replacement-mask.npy',replacement)
one=paths.index('terrain\\tiles\\campaign\\generic\\generic_1x1\\');single_template=None
for parent in np.flatnonzero(native0['path'].ravel()==one):
 rec=native0.ravel()[parent];tx,ty=int(rec['x']),int(rec['y'])
 if tx+2>ow or ty+2>oh:continue
 block=native0[ty:ty+2,tx:tx+2];oy,ox=divmod(int(parent),ow);oy-=ty;ox-=tx
 if 0<=ox<2 and 0<=oy<2 and np.all((block['path']==one)|(block['path']==-parent-1)):
  single_template=block.copy();single_parent=(ox,oy);break
assert single_template is not None
_mark('cache: single template')
floors=[];fine_coverage=np.zeros_like(replacement)
# Vanilla salt pans (3k_main_map) on real salt flats of rebuilt ground (salt_pools.py): stamped
# first, their cells leave the floor layout, their list records join the floors' below.
import salt_pools
_salt_owners,_salt_shape=salt_pools.donor_cluster(native0,paths)
salt_cover=np.zeros_like(replacement);salt_placed=[]
# The windows planned (and levelled) by the polish run that wrote the LF this cache follows;
# one whose ground is no longer wholly replaced land is dropped, never moved onto unlevelled ground.
_salt_plan=json.loads((A/salt_pools.WINDOWS_FILE).read_text(encoding='utf-8'))
if tuple(_salt_plan['shape'])!=_salt_shape:raise RuntimeError(f'{salt_pools.WINDOWS_FILE} was planned for a {_salt_plan["shape"]} donor window, not {_salt_shape}')
_salt_win=[(R_,C_) for R_,C_ in _salt_plan['windows'] if replacement[R_:R_+_salt_shape[0],C_:C_+_salt_shape[1]].all() and physical[R_:R_+_salt_shape[0],C_:C_+_salt_shape[1]].all()]
_salt_dropped=len(_salt_plan['windows'])-len(_salt_win);_salt_sites=range(_salt_plan['site_windows'])
for _R,_C in _salt_win:
 _cov,_placed=salt_pools.stamp(layers[0],_salt_owners,(_R,_C),int(biome[_R+_salt_shape[0]//2,_C+_salt_shape[1]//2]),TW)
 for _r,_c in _cov:salt_cover[_r,_c]=True
 salt_placed+=_placed
fine_coverage|=salt_cover;_floor_mask=replacement&~salt_cover
# --- MOUNTAIN TILES (mountain_tiles.py) --- vanilla mountain fine tiles (rock faces, scree, cliffs)
# on the steep and very high real-terrain ground of rebuilt canvas land: whole donor windows of
# the family the window's biome calls for (cold / subtropical / temperate / yellow). Their cells
# leave the floor layout; their list records and ranges ride with the salt pans' (salt_placed).
import mountain_tiles as _mt
if REAL_TERRAIN.exists():
 _mreal=np.load(REAL_TERRAIN);_mmask,_mfam=_mt.mountain_mask(_mreal,(TH,TW));_meco=_mt.tile_eco(_mreal,(TH,TW));del _mreal   # TILE PROPS: ecoregion per tile
 # Town plots (map slot bits; full_town_edit is not built in --cache-only runs) and road hexes stay floor.
 _mplot=((((cells[:,:,2]>>4)>0)|((cells[:,:,3]&1)>0))|(_rt_mask>0))[hy,hx]
 _mallow=replacement&physical&outer&~salt_cover&~fastops.max_filter_mask(_mplot,9)
 _mtaken=np.load(A/'analysis/scenery-tile-cells.npy') if (A/'analysis/scenery-tile-cells.npy').exists() else None
 _mpicked=_mt.plan(_mmask,_mfam,_mallow,_mtaken)
 _mcov,_mplaced=_mt.stamp_all(layers[0],native0,paths,_mpicked,biome,TW,eco=_meco)   # TILE PROPS: props only in their vanilla habitat
 mountain_cover=np.zeros_like(replacement)
 for _r,_c in _mcov:mountain_cover[_r,_c]=True
 fine_coverage|=mountain_cover;_floor_mask&=~mountain_cover;salt_placed+=_mplaced
 np.save(A/_mt.COVER_FILE,mountain_cover)
 report['mountain_tiles']={'windows':len(_mpicked),'tiles':len(_mplaced),'cells':int(mountain_cover.sum()),'by_family':{f:sum(1 for *_,g in _mpicked if g==f) for f in _mt.FAMILIES},'tile_props':dict(_mt.LAST_STATS)}
 (A/_mt.REPORT_FILE).write_text(json.dumps({**report['mountain_tiles'],'windows_list':[[int(R_),int(C_),f_] for R_,C_,f_ in _mpicked]}),encoding='utf-8')
 del _mmask,_mfam,_mplot,_mallow,_mtaken,_mcov,_meco
else:
 # No real terrain, no mountain tiles: an empty cover (and report) replaces the previous run's,
 # which repair_native_placements would otherwise still take as mountain-tile cells.
 np.save(A/_mt.COVER_FILE,np.zeros_like(replacement))
 report['mountain_tiles']={'windows':0,'tiles':0,'cells':0,'by_family':{f:0 for f in _mt.FAMILIES},'real_terrain':False}
 (A/_mt.REPORT_FILE).write_text(json.dumps({**report['mountain_tiles'],'windows_list':[]}),encoding='utf-8')
 print('Mountain tiles: no real terrain ('+str(REAL_TERRAIN)+'); mountain-tile cover cleared',flush=True)
# --- end mountain tiles
# --- COAST TILES (coast_tiles.py) --- vanilla sea_coast pieces on the new sea shores: land hexes (kind
# 0/2) beside open sea whose whole 2x2 block is free floor ground outside the native rectangle or on its
# edited ground (guarded from unedited native land), clear of town/port plots, coast_props docks, river mouths (river hexes, river water band) and
# the road, salt, mountain and scenery cells. Piece and quarter turn follow vanilla's layout for the
# hex's water-neighbour mask; each piece gets vanilla's generic_sea underlays on layer 2. Layer-0 cells
# leave the floor layout; the records ride salt_placed (underlays first, see coast_tiles.stamp).
import coast_tiles as _ct,hashlib as _cth
from campaign_ground_textures import VanillaPacks as _CTVP
_ctvp=_CTVP();_ct_learned=_ct.Learned(_ctvp.read('terrain/campaigns/3k_dlc07_main_map/global_map/tile_list.bin'),_rmh.read(_frame.VANILLA_MAP),_ctvp.read)
_ct_meta=read_hex(A/'work/final-map/map.hex')
_ct_lf=_rt.south_up_lf((S/f'terrain/campaigns/{ALL_VARIANTS[-1]}/lf_height_map.dds').read_bytes())
_ct_rl=A/'analysis/river-water-level.npy';_ct_band=None
if _ct_rl.exists():
 _ct_rs=json.loads(_ct_rl.with_suffix('.json').read_text(encoding='utf-8'))
 if _ct_rs.get('map_sha256')!=_cth.sha256((A/'work/final-map/map.hex').read_bytes()).hexdigest() or _ct_rs.get('canvas')!=_frame.canvas_stamp():raise RuntimeError(f'{_ct_rl} was written for another map.hex or canvas')
 _ct_band=np.isfinite(np.load(_ct_rl))
_ct_taken=_rt.road_cells(_rt_mask>0,TH,TW)
if (A/'analysis/scenery-tile-cells.npy').exists():_ct_taken|=np.load(A/'analysis/scenery-tile-cells.npy')
# coast_props.py (phase B, after polish) writes the docks: this reads the previous run's, which follow the same port towns
_ct_docks=_ct.dock_hexes(json.loads((A/'analysis/coast-props.json').read_text(encoding='utf-8')),H,W) if (A/'analysis/coast-props.json').exists() else None
# Inside the native rectangle only on really edited ground (LF changed at halo 0 or a real-terrain patch, with
# the saved edit alpha above repair_native_placements' floor threshold), guarded from unedited native land.
_ct_region=_ct.region_cells(TH,TW,_frame.NATIVE_TILE_X0,_frame.NATIVE_TILE_Y0,_frame.NATIVE_TILE_X1,_frame.NATIVE_TILE_Y1,_ct.edited_native(lf_edited,np.load(A/'analysis/polished-edit-alpha.npy'),_ct.patch_tiles(HH,hy,hx)),physical,_rf.filler_hex(cells,names)[hy,hx] if NONPLAY_GROUND else np.zeros((TH,TW),bool))
_ct_free=_ct.allowed_cells(_floor_mask&(layers[2]['flags']==16),_ct_region,_ct_taken,((((cells[:,:,2]>>4)>0)|((cells[:,:,3]&1)>0)))[hy,hx],_ct_band,TH,TW)
_ct_shapes=_ct.cache_shapes(native0,paths,_ct.SET);_ct_sea_shapes=_ct.cache_shapes(np.frombuffer(native,dtype=DT,count=on,offset=2*on*20).reshape(oh,ow),paths,_ct.SEA)
# Shores (native edited ones and the new canvas shores) take beach pieces only where the real shore is low
# (coast_tiles.shore_types: ETOPO 2022, aligned through korea_patch's coast warp in its window); steep or
# unmatched real shores keep their floors.
_ct_elev=np.load(REAL_TERRAIN)['elevation'] if REAL_TERRAIN.exists() else None
_ct_nblock,_ct_ntypes=_ct.beach_block(cells,_ct_meta['land_ground_types'],_ct_meta['sea_ground_types'],_ct_elev,[_ct.korea_aligned_elevation(A,cells,_frame.VANILLA_MAP)] if _ct_elev is not None else [],names)
_ct_pl,_ct_stats,_=_ct.plan(cells,_ct_meta['land_ground_types'],_ct_meta['sea_ground_types'],_ct_learned,_ct_free,_ct_shapes,_ct_lf,_ct_nblock if _ct_docks is None else (_ct_docks|_ct_nblock))
_ct_stats['shore_types']=_ct_ntypes
coast_cover,_ct_cover2,_ct_placed=_ct.stamp(layers[0],layers[2],_ct_pl,_ct_shapes,_ct_sea_shapes,lambda t_:paths.index(t_)+1,biome,_ct_lf)
fine_coverage|=coast_cover;_floor_mask&=~coast_cover;salt_placed+=_ct_placed
_ct.save_npy(A/_ct.COVER_FILE,coast_cover)   # repair_native_placements lists no floor under these cells
report['coast_tiles']=_ct.report(_ct_pl,_ct_stats,_ct_learned,coast_cover);_ct.save_text(A/_ct.REPORT_FILE,json.dumps(report['coast_tiles'],indent=1))
print('Coast tiles:',{k_:v_ for k_,v_ in report['coast_tiles'].items() if k_ not in ('unmatched_masks','by_piece','vanilla')},flush=True)
del _ct_elev,_ct_nblock,_ct_region,_ctvp,_ct_learned,_ct_meta,_ct_lf,_ct_band,_ct_taken,_ct_docks,_ct_free,_ct_shapes,_ct_sea_shapes,_ct_pl,_ct_cover2,_ct_placed
# --- end coast tiles
report['salt_pans']={'tiles':len(salt_placed),'cells':int(salt_cover.sum()),'site_windows':len(_salt_sites),'windows':len(_salt_win),'planned_windows_dropped':_salt_dropped,'playa_windows_skipped':_salt_plan['playa_windows_skipped']}
# [l2048-salt-split] BEGIN. The pan stamp splits aligned 2x2 floor groups: the 13-row window leaves one
# row of split groups just outside it (R+13 for even R, R-1 for odd R), and the donor's dropped road cells
# leave pairs inside it. The floor layout skips split groups, so these cells stay inactive in the cache
# while repair_native_placements lists a generic_1x1 there. That gave the black shards and bars in live 2048l
# (Shanshan, Xihai, Piaoyue, empty-1; 348 cells over the 14 windows, all of the inactive-cache 1x1 cells
# there). Use the remedy the road pieces already use for their split cells: 1-cell generic_1x1 cache owners.
_salt_split=_rt.split_floor_cells(replacement,_floor_mask)
report['salt_pans']['split_floor_cells']=_rt.stamp_singles(layers[0],_salt_split,one+1,biome,_rt.south_up_lf((S/f'terrain/campaigns/{ALL_VARIANTS[-1]}/lf_height_map.dds').read_bytes()))
fine_coverage|=_salt_split;_floor_mask&=~_salt_split
# [l2048-salt-split] END
# repair_native_placements.py rebuilds tile_list.bin from terrain-before-polish; it re-adds these.
np.save(A/'analysis/salt-pan-cover.npy',salt_cover)
(A/'analysis/salt-pan-records.json').write_text(json.dumps([[paths[pv-1],x_,y_,fl,cl_,list(rows),list(cols)] for pv,x_,y_,fl,cl_,rows,cols,_ in salt_placed]),encoding='utf-8')
# --- road tiles (road_tiles.py) 2/3: vanilla roads_tracks pieces on every road hex of floor ground
# (pattern and lane rules learned from the vanilla dlc07 layout); their cells and the 2x2 floor cells
# an odd-column road block splits (1-cell generic_1x1 owners) leave the floor layout.
from campaign_ground_textures import VanillaPacks as _RVP
_rvp=_RVP();_rt_patterns,_rt_pieces,_rt_span_max=_rt.learn(_rvp.read('terrain/campaigns/3k_dlc07_main_map/global_map/tile_list.bin'),_rmh.read(_frame.VANILLA_MAP)['cells'],_rvp.read)
_rt_pieces.ports={k_:v_ for k_,v_ in _rt_pieces.ports.items() if _rt.piece_path(k_[0]) in paths}
_rt_lf=_rt.south_up_lf((S/f'terrain/campaigns/{ALL_VARIANTS[-1]}/lf_height_map.dds').read_bytes())
def _rt_span(x_,y_,fw_,fh_):
 b_=_rt_lf[max(0,(y_-2)*4):(y_+fh_+2)*4,max(0,(x_-2)*4):(x_+fw_+2)*4];return (int(b_.max())-int(b_.min()))/65535
_rt_placements,_rt_stats=_rt.layout(_rt_mask,(_rt_mask>0)&_rt.blocks_in(_floor_mask,H,W),_floor_mask,_rt.road_cells(_rt_mask,TH,TW),_rt_patterns,_rt_pieces,_rt.existing_road_tiles(layers[0],paths),_rt_span,_rt_span_max,_rt_vmask)
road_cover,road_placed=_rt.stamp(layers[0],_rt_placements,_rt.cache_shapes(native0,paths),lambda pc_:paths.index(_rt.piece_path(pc_))+1,biome,_rt_lf)
_rt_split=_rt.split_floor_cells(_floor_mask,_floor_mask&~road_cover)
_rt_singles=_rt.stamp_singles(layers[0],_rt_split,one+1,biome,_rt_lf)
fine_coverage|=road_cover|_rt_split;_floor_mask=_floor_mask&~road_cover&~_rt_split
report['road_tiles']={**_rt_stats,'pieces':len(road_placed),'cells':int(road_cover.sum()),'split_floor_cells':_rt_singles,'native_join_hexes':_rt_join_hexes,'native_hexes_refused':_rt_join_refused,'dangling_road_bits_ignored':_rt_dangling}
print('Road tiles:',{k_:v_ for k_,v_ in report['road_tiles'].items() if k_!='inexact_by_mask'},flush=True)
# repair_native_placements.py and clear_obsolete_terrain_overlays.py keep these (records: path, x, y, flags, climate, rows, cols).
np.save(A/'analysis/road-tile-cover.npy',road_cover)
(A/'analysis/road-tile-records.json').write_text(json.dumps([[p_,x_,y_,fl,cl_,list(rows),list(cols)] for p_,x_,y_,fl,cl_,rows,cols,_ in road_placed]),encoding='utf-8')
# --- end road tiles 2/3
# [fix-batch floor-split] BEGIN. The same split-group mechanism as the salt windows, map-wide: any
# cell of the final floor mask whose aligned 2x2 group is not whole (owners removed off the 2x2 grid,
# coast and Hehuang cut edges, ...) was skipped by the 2x2 floor layout and stayed inactive in the
# cache, while repair_native_placements listed a generic_1x1 on 4,727 of them (live 2048l black
# shards outside the pans too; 4,401 inside the native map). Same remedy: 1-cell generic_1x1 cache
# owners. The cells are saved so repair_native_placements lists every one of them (no cache owner
# without its list record).
_fs_split=_rt.split_floor_cells(np.ones_like(_floor_mask),_floor_mask)   # floor cells outside whole 2x2 groups
report['floor_split_cells']=_rt.stamp_singles(layers[0],_fs_split,one+1,biome,_rt_lf)
fine_coverage|=_fs_split;_floor_mask&=~_fs_split
np.save(A/'analysis/floor-split-cells.npy',_fs_split|_salt_split|_rt_split)
print('Floor split cells re-stamped as single cells:',report['floor_split_cells'],flush=True)
# [fix-batch floor-split] END
def floor_tile(tile,xx,yy,size,template_,parent_offset,rotation):
 eh,ew=min(size,TH-yy),min(size,TW-xx);ox,oy=parent_offset;ox=min(ox,ew-1);oy=min(oy,eh-1);parent=(yy+oy)*TW+xx+ox;block=template_[:eh,:ew].copy();block['path']=-parent-1;block['x']=xx;block['y']=yy;block['climate']=biome[yy+eh//2,xx+ew//2];block['lo']=0;block['hi']=0
 block['path'][oy,ox]=tile;block['flags'][oy,ox]=int(template_['flags'][parent_offset[1],parent_offset[0]]);mini=max(0,float(norm[yy:yy+eh,xx:xx+ew].min())-.002);maxi=min(1,float(norm[yy:yy+eh,xx:xx+ew].max())+.002);block['lo'][oy,ox]=mini;block['hi'][oy,ox]=maxi;layers[0][yy:yy+eh,xx:xx+ew]=block
 fine_coverage[yy:yy+eh,xx:xx+ew]=True
 floors.append((tile,xx,yy,int(block['climate'][oy,ox]),mini,maxi,int(block['flags'][oy,ox]),rotation))
# Vectorised floor placement: the same floors in the same order with the same
# fields as the per-block loop (_floor_loop_reference below). Whole uniform
# 16x16 blocks take one generic 8x8 tile (floor_tile, few); every other 2x2
# cell that is fully replaced takes a generic 1x1 tile, written for all cells at once.
def _floor_layout():
 nby,nbx=-(-TH//16),-(-TW//16);PH,PW=nby*16,nbx*16
 rf=np.zeros((PH,PW),bool);rf[:TH,:TW]=_floor_mask;rt=np.ones((PH,PW),bool);rt[:TH,:TW]=_floor_mask
 bio=np.zeros((PH,PW),biome.dtype);bio[:TH,:TW]=biome
 b4=rf.reshape(nby,16,nbx,16);bb=bio.reshape(nby,16,nbx,16)
 full=np.zeros((nby,nbx),bool);full[:TH//16,:TW//16]=True
 gen=b4.any(axis=(1,3))&full&b4.all(axis=(1,3))&(bb==bb[:,:1,:,:1]).all(axis=(1,3))
 single_blocks=b4.any(axis=(1,3))&~gen
 c2=rt.reshape(PH//2,2,PW//2,2).all(axis=(1,3))
 cy,cx=np.nonzero(c2&np.repeat(np.repeat(single_blocks,8,axis=0),8,axis=1))
 keep=(cy*2<TH)&(cx*2<TW);cy,cx=cy[keep],cx[keep]
 gy,gx=np.nonzero(gen)
 # loop order: blocks row-major, then (dy, dx) inside a block
 by=np.concatenate((gy,cy//8));bx=np.concatenate((gx,cx//8));dy=np.concatenate((np.zeros(len(gy),np.int64),cy%8*2));dx=np.concatenate((np.zeros(len(gx),np.int64),cx%8*2))
 isgen=np.concatenate((np.ones(len(gy),bool),np.zeros(len(cy),bool)))
 order=np.lexsort((dx,dy,bx,by))
 return (by*16+dy)[order],(bx*16+dx)[order],isgen[order]
F_y,F_x,F_gen=_floor_layout()
# --- SEA FLOORS (sea_floor_tiles.py) --- open-sea floor cells take vanilla's generic_sea floors, not
# land generic ones (the drowned vanilla shore off Lin Yi read live as a long pale strip, v14-v16):
# all-sea 16x16 floors become four sea 8x8, a 16x16 over sea and land its 2x2 groups (sea or land),
# all-sea 2x2 floors sea 2x2; the land floors keep the layout order (blocks row-major, then dy, dx).
import sea_floor_tiles as _sf
_sf_water=_sf.water_cells(cells,names,hy,hx)
(_sf_ls,F_x,F_y),(S_size,S_x,S_y)=_sf.split(np.where(F_gen,16,2),F_x,F_y,_sf_water)
_o=np.lexsort((F_x%16,F_y%16,F_x//16,F_y//16));F_x,F_y,F_gen=F_x[_o],F_y[_o],_sf_ls[_o]==16;del _o,_sf_ls
# --- end sea floors 1/3
F_climate=np.zeros(len(F_y),np.int64);F_flags=np.zeros(len(F_y),np.int64);F_tile=np.where(F_gen,generic,one)
_gi=np.flatnonzero(F_gen)
for _i in _gi:
 floor_tile(generic,int(F_x[_i]),int(F_y[_i]),16,template,template_parent,2);F_climate[_i]=floors[-1][3];F_flags[_i]=floors[-1][6]
_si=np.flatnonzero(~F_gen)
if len(_si):
 _xs,_ys=F_x[_si],F_y[_si];_eh=np.minimum(2,TH-_ys);_ew=np.minimum(2,TW-_xs)
 _ox=np.minimum(single_parent[0],_ew-1);_oy=np.minimum(single_parent[1],_eh-1);_parent=(_ys+_oy)*TW+_xs+_ox
 _clim=biome[_ys+_eh//2,_xs+_ew//2].astype(np.uint8)
 _np=np.full((TH+1,TW+1),np.inf,norm.dtype);_np[:TH,:TW]=norm;_nm=np.full((TH+1,TW+1),-np.inf,norm.dtype);_nm[:TH,:TW]=norm
 _mn=np.minimum.reduce([_np[_ys+a,_xs+b] for a in (0,1) for b in (0,1)]);_mx=np.maximum.reduce([_nm[_ys+a,_xs+b] for a in (0,1) for b in (0,1)])
 _mini=np.maximum(0,_mn.astype(np.float64)-.002);_maxi=np.minimum(1,_mx.astype(np.float64)+.002)
 _pflag=int(single_template['flags'][single_parent[1],single_parent[0]])
 L0=layers[0]
 for a in (0,1):
  for b in (0,1):
   s=(a<_eh)&(b<_ew);ys,xs=_ys[s]+a,_xs[s]+b
   L0[ys,xs]=single_template[a,b];L0['path'][ys,xs]=-_parent[s]-1;L0['x'][ys,xs]=_xs[s];L0['y'][ys,xs]=_ys[s];L0['climate'][ys,xs]=_clim[s];L0['lo'][ys,xs]=0;L0['hi'][ys,xs]=0
   fine_coverage[ys,xs]=True
 _py,_px=_ys+_oy,_xs+_ox
 L0['path'][_py,_px]=one;L0['flags'][_py,_px]=_pflag;L0['lo'][_py,_px]=_mini;L0['hi'][_py,_px]=_maxi
 F_climate[_si]=_clim;F_flags[_si]=_pflag
 del _xs,_ys,_eh,_ew,_ox,_oy,_parent,_clim,_np,_nm,_mn,_mx,_mini,_maxi
floors=[(int(t),int(x_),int(y_),int(c),0.,0.,int(f),2 if g else 1) for t,x_,y_,c,f,g in zip(F_tile,F_x,F_y,F_climate,F_flags,F_gen)]
# --- SEA FLOORS (sea_floor_tiles.py) 2/3: stamp the sea floors (owner at the turn's corner, children
# 0x710, climate at the footprint centre as the land floors); the 1-cell generic_1x1 owners polish laid
# in split floor cells (salt / road / floor-split singles) on open sea become sea 1x1. Ranges per variant below.
_sf_value={s_:paths.index(_sf.SEA_TILE[s_])+1 for s_ in _sf.SEA_TILE}   # cache path values are one-based
assert all(paths[v_-1]==_sf.SEA_TILE[s_] for s_,v_ in _sf_value.items())
_sf_sy,_sf_sx=np.nonzero((layers[0]['path']==one+1)&replacement&_sf_water)
S_size=np.concatenate((S_size,np.ones(len(_sf_sx),np.int64)));S_x=np.concatenate((S_x,_sf_sx));S_y=np.concatenate((S_y,_sf_sy))
S_climate=_sf.climate_at(S_size,S_x,S_y,biome).astype(np.uint8)
S_turn=_sf.stamp(layers[0],S_size,S_x,S_y,S_climate,_sf_value);S_py,S_px=_sf.owner_cells(S_size,S_x,S_y)
for _s in (8,2,1):
 _i=np.flatnonzero(S_size==_s)
 for _dy in range(_s):
  for _dx in range(_s):fine_coverage[S_y[_i]+_dy,S_x[_i]+_dx]=True
np.save(A/_sf.WATER_FILE,_sf_water);np.save(A/_sf.CACHE_ANCHORS_FILE,_sf.anchors((TH,TW),S_size,S_x,S_y))
report['sea_floors']={'open_sea_cells':int(_sf_water.sum()),'by_size':{str(_s):int((S_size==_s).sum()) for _s in (8,2,1)},'from_singles':int(len(_sf_sx)),'land_floors':int(len(F_x))}
print('Sea floors (generic_sea):',report['sea_floors'],flush=True)
# Cache ranges (owner cells): LF extrema over the footprint plus two cells, raised to vanilla's sea plane,
# from the final DDS (the land floors take theirs per variant below; sea owners keep these).
_sf_t=_sf.lf_extremes(_rt.south_up_lf((S/f'terrain/campaigns/{ALL_VARIANTS[-1]}/lf_height_map.dds').read_bytes()))
S_lo,S_hi=_sf.ranges(_sf_t[0],_sf_t[1],S_size,S_x,S_y);layers[0]['lo'][S_py,S_px]=S_lo;layers[0]['hi'][S_py,S_px]=S_hi
del _sf_sy,_sf_sx,_sf_t
# --- end sea floors 2/3
def _floor_loop_reference():
 for yy in range(0,TH,16):
  for xx in range(0,TW,16):
   eh,ew=min(16,TH-yy),min(16,TW-xx);mask=replacement[yy:yy+eh,xx:xx+ew]
   if not np.any(mask):continue
   if eh==16 and ew==16 and np.all(mask) and np.all(biome[yy:yy+eh,xx:xx+ew]==biome[yy,xx]):floor_tile(generic,xx,yy,16,template,template_parent,2)
   else:
    for dy in range(0,eh,2):
     for dx in range(0,ew,2):
      # Preserve neighbouring native owners at irregular borders. Thin cells
      # which cannot hold a complete fine tile are supplied by coarse ground.
      if np.all(mask[dy:dy+2,dx:dx+2]):floor_tile(one,xx+dx,yy+dy,2,single_template,single_parent,1)
_mark('cache: floor tile placement')
for layer in layers:
 child=layer['path']<0;parents=-layer['path'][child]-1;assert np.all((parents>=0)&(parents<n));assert np.all(layer['path'].ravel()[parents]>=0);assert np.all(layer['flags'].ravel()[parents]!=16)
for variant in POLISH_VARIANTS:
 variant_dds=(S/f'terrain/campaigns/{variant}/lf_height_map.dds').read_bytes();variant_h,variant_w=struct.unpack_from('<2I',variant_dds,12);variant_lf=np.frombuffer(variant_dds,dtype='<u2',offset=128).reshape(variant_h,variant_w)[::-1]
 # Vectorised cache ranges (the same values as the per-floor loop below): 4x4 LF texel min/max per tile
 # cell, then clipped sliding windows of tile cells.
 vectorised=variant_h==4*TH and variant_w==4*TW
 if vectorised:
  _lf4=np.ascontiguousarray(variant_lf).reshape(TH,4,TW,4);TMIN=_lf4.min(axis=(1,3));TMAX=_lf4.max(axis=(1,3));del _lf4
  def _win(a,lo,hi,fn,fill):
   out=a
   for axis in (0,1):
    n=a.shape[axis];acc=None
    for k in range(lo,hi):
     s=np.full(out.shape,fill,a.dtype)
     if k>=0:
      if k<n:(s[:n-k] if axis==0 else s[:,:n-k])[...]=(out[k:] if axis==0 else out[:,k:])
     else:
      if -k<n:(s[-k:] if axis==0 else s[:,-k:])[...]=(out[:n+k] if axis==0 else out[:,:n+k])
     acc=s if acc is None else fn(acc,s)
    out=acc
   return out
  _si=np.flatnonzero(~F_gen);_gi=np.flatnonzero(F_gen)
 # Generated cache ranges follow this variant's final DDS, with cache rather
 # than placement footprints. Native retained roots are calibrated separately.
 vd=variant_dds;vh,vw=struct.unpack_from('<2I',vd,12);vp=np.frombuffer(vd,dtype='<u2',offset=128).reshape(vh,vw)[::-1]
 cache_layers=[layers[0].copy()]+layers[1:]   # only layer 0 ranges change below
 if vectorised:
  _ys,_xs=F_y[_si],F_x[_si];_eh=np.minimum(2,TH-_ys);_ew=np.minimum(2,TW-_xs);_ox=np.minimum(single_parent[0],_ew-1);_oy=np.minimum(single_parent[1],_eh-1)
  _w=_win(TMIN,-2,4,np.minimum,65535);cache_layers[0]['lo'][_ys+_oy,_xs+_ox]=_w[_ys,_xs].astype(np.float64)/65535
  _w=_win(TMAX,-2,4,np.maximum,0);cache_layers[0]['hi'][_ys+_oy,_xs+_ox]=_w[_ys,_xs].astype(np.float64)/65535;del _w
  _cache_floors=[floors[i] for i in _gi]
 else:_cache_floors=floors
 for tile,xx,yy,climate,lo,hi,flags,rotation in _cache_floors:
  size=16 if tile==generic else 2;offset=template_parent if tile==generic else single_parent;ew,eh=min(size,TW-xx),min(size,TH-yy);ox,oy=min(offset[0],ew-1),min(offset[1],eh-1)
  block=vp[max(0,(yy-2)*4):min(vh,(yy+eh+2)*4),max(0,(xx-2)*4):min(vw,(xx+ew+2)*4)];assert block.size
  cache_layers[0]['lo'][yy+oy,xx+ox]=float(block.min())/65535;cache_layers[0]['hi'][yy+oy,xx+ox]=float(block.max())/65535
 for pv,x_,y_,fl,cl_,rows,cols,(pr,pc) in salt_placed:
  block=vp[max(0,(rows[0]-2)*4):min(vh,(rows[1]+3)*4),max(0,(cols[0]-2)*4):min(vw,(cols[1]+3)*4)]
  cache_layers[0]['lo'][pr,pc]=float(block.min())/65535;cache_layers[0]['hi'][pr,pc]=float(block.max())/65535
 put_parts(f'terrain/campaigns/{variant}/tile_map.tiles',[np.ascontiguousarray(layer).reshape(-1).view(np.uint8) for layer in cache_layers]+[tail])
 del cache_layers,vd,vp
 put(f'terrain/campaigns/{variant}/tile_map.index',struct.pack('<3I',11,TW,TH))
 _mark('cache: tile_map write')
# tile_list.bin (both lists) is not written here: repair_native_placements.py builds them from
# terrain-before-polish with the floors, sea floors (sea-floor water cells), salt / mountain / coast pieces
# (salt-pan-records.json and the covers) and road pieces (road-tile-records.json) saved above.
# Validate every active parent, native height range, actual coverage and geometry.
for i,layer in enumerate(layers):
 child=layer['path']<0;parents=-layer['path'][child]-1;assert np.all((parents>=0)&(parents<n));assert np.all(layer['path'].ravel()[parents]>=0)
assert np.all(layers[0]['flags'][fine_coverage]!=16)
np.save(A/'analysis/cache-fine-coverage-mask.npy',fine_coverage)
_mark('cache: final checks')
report['native_cache_owner_removal']=owner_removal_counts
report['cache_boundary_cells_using_coarse_ground']=int((replacement&~fine_coverage).sum())
report['floor_tiles']=len(floors);report['land_floor_footprints']={'generic_8x8':[16,16],'generic_1x1':[2,2]};report['native_height_pixels_preserved_outside_edit']=True;report['replacement_cells']=int(replacement.sum());report['cache_parent_checks']='passed';report['sdk_region_count']=len(names);assert len(names)<=511
(A/'analysis/geographic-terrain-polish.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
# --- TILE BAKE (tile_bake.py, pipeline-bake-design step 1) --- the tile lists' runtime floor placements and the
# obsolete native overlays, on the LF this cache was built from (the former pipeline steps
# repair_native_placements.py and clear_obsolete_terrain_overlays.py, unchanged): they read this segment's
# masks, covers and records above and rewrite both tile_list.bin files.
import tile_bake;tile_bake.bake()
_mark('cache: tile bake (placements + overlays)')
# Offline engineering preview; this is not substituted for game acceptance.
s=np.load(A/f'analysis/polished-surface-{ALL_VARIANTS[-1]}.npy');grey=np.clip(s/8,0,1);rgb=np.stack((55+grey*145,85+grey*115,52+grey*138),axis=2).astype('uint8');rgb[~physical]=[28,65,91];pic=Image.fromarray(rgb[::-1]).resize((W,H),Image.Resampling.LANCZOS);draw=ImageDraw.Draw(pic)
for town in plan['regions']:
 tq,tr=town['town'];draw.ellipse((tq-1,H-1-tr-1,tq+1,H-1-tr+1),fill='white')
pic.save(ROOT/'output/campaign-terrain-polish-height-preview.png')
print('Native coverage and cache checks passed;',len(floors),'native land tiles; geographic terrain staged',flush=True)

_mark('cache: report + preview')
