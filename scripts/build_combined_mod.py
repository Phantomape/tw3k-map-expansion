"""Build matching SDK exports and both compiler/runtime packs for the merged mod."""
import tw3k_paths
import sys,json,struct,time,shutil,subprocess,xml.etree.ElementTree as ET
from pathlib import Path
import os,tempfile,hashlib
from collections import deque
from concurrent.futures import ThreadPoolExecutor
_T0=time.time()
def _mark(label):print(f'[pack-timing] {time.time()-_T0:7.1f}s {label}',flush=True)
from native_compilation_state import atomic_json,current_sources,verify_generated,check_map_data_export
from campaign_world import WORLD_WIDTH,WORLD_HEIGHT
from analyze_reference import ROOT,inventory,digest,read_hex
import campaign_world as _frame
A=_frame.WORKSPACE;KIT=A/'work/standalone_kit';STAGE=A/'work/release-input';GAME=tw3k_paths.GAME_DATA
RPFM=tw3k_paths.RPFM;SCHEMA=tw3k_paths.RPFM_SCHEMA
PACK_TIMESTAMP=1760000000   # 2025-10-09; any constant works, the game does not order packs by it
SMALL_FILE=64*1024*1024     # inputs below this are fingerprinted by content, larger ones by size+mtime
# Interpreter for the camera check and release gates: TW3K_PYTHON as given, else the single
# tw3k env (env/python-path.txt, also read by tw3k_env.ps1) if it exists, else this interpreter.
_DEFAULT_PYTHON=Path((ROOT/'env/python-path.txt').read_text(encoding='utf-8').strip())
TW3K_PYTHON=Path(os.environ.get('TW3K_PYTHON') or (_DEFAULT_PYTHON if _DEFAULT_PYTHON.exists() else sys.executable))


def input_fingerprint(files):
 """Fingerprint pack inputs: content hash for small files (the SDK re-exports the
 tables every run, so their mtimes change), size+mtime for large terrain files."""
 h=hashlib.sha256()
 for name,p in sorted(files.items()):
  st=p.stat();h.update(name.encode()+b'\0'+str(st.st_size).encode()+b'\0')
  h.update(digest(p).encode() if st.st_size<SMALL_FILE else str(st.st_mtime_ns).encode())
 return h.hexdigest()


def battle_entrances():
 """Run build_expanded_battle_entrances.py unless its inputs and output are as last built."""
 script=ROOT/'scripts/build_expanded_battle_entrances.py';out=STAGE/'terrain/battles/3k_main_map/battle_locations_map.bin'
 stamp=A/'analysis/battle-entrances-stamp.json'
 key={'script':digest(script),'design':digest(A/'work/current-region-design.json'),'map':digest(A/'work/final-map/map.hex'),
      'vanilla':digest(ROOT/'reference/vanilla-terrain/terrain/battles/3k_main_map/battle_locations_map.bin'),'campaign_world':digest(ROOT/'scripts/campaign_world.py'),
      # parse/enc and _hex_cells shape the output as much as the script itself.
      'fastbin':digest(ROOT/'scripts/fastbin.py'),'all_directions_build':digest(ROOT/'scripts/all_directions_build.py')}
 if out.exists() and stamp.exists():
  last=json.loads(stamp.read_text(encoding='utf-8'))
  if last.get('inputs')==key and last.get('output')==digest(out):print('Battle entrances unchanged; reused',flush=True);return
 subprocess.run([sys.executable,'-B',str(script)],check=True)
 atomic_json(stamp,{'inputs':key,'output':digest(out)})

def exports():
 p=KIT/'raw_data/db/campaign_maps.xml';doc=ET.parse(p)
 for row in doc.getroot().findall('campaign_maps'):
  if row.findtext('mapname') in _frame.TERRAIN_VARIANTS:
   row.find('maxx').text=format(WORLD_WIDTH,'.12g');row.find('maxy').text=format(WORLD_HEIGHT,'.12g')
 doc.write(p,encoding='utf-8',xml_declaration=True)
 import native_db_export as n
 n.ROOT=A;n.BIN=KIT/'binaries';sys.argv=['native_db_export.py','--all','--split'];n.main()
 subprocess.run([sys.executable,__file__,'world-table'],check=True)


PREFETCH_JOBS=16;PREFETCH_FILE=8*1024*1024;PREFETCH_BYTES=256*1024*1024
def prefetched(ordered,sizes):
 """Yield (name,path,body) in order. Files up to PREFETCH_FILE are read ahead on
 threads (each cold open costs 10-35 ms of Defender scanning; 11k of them ran serially),
 at most PREFETCH_BYTES of them in flight; body is None for a larger file (streamed)."""
 def read(p):
  with p.open('rb') as f:return f.read()
 with ThreadPoolExecutor(PREFETCH_JOBS) as pool:
  ahead=deque();inflight=0;nxt=0
  for i,(name,p) in enumerate(ordered):
   while nxt<len(ordered) and (nxt==i or inflight+sizes[ordered[nxt][0]]<=PREFETCH_BYTES and len(ahead)<64*PREFETCH_JOBS):
    s=sizes[ordered[nxt][0]];small=s<=PREFETCH_FILE
    ahead.append(pool.submit(read,ordered[nxt][1]) if small else None);inflight+=s if small else 0;nxt+=1
   future=ahead.popleft()
   if future is None:yield name,p,None;continue
   body=future.result();inflight-=sizes[name];yield name,p,body

def pack(dest,files,before_commit=None):
 dest=Path(dest);ordered=sorted(files.items());stats={name:(p.stat().st_size,p.stat().st_mtime_ns) for name,p in ordered};sizes={name:v[0] for name,v in stats.items()}
 ix=b''.join(struct.pack('<IB',sizes[name],0)+name.replace('/','\\').encode()+b'\0' for name,p in ordered)
 dest.parent.mkdir(parents=True,exist_ok=True)
 fd,tmpname=tempfile.mkstemp(prefix=dest.name+'.',suffix='.partial',dir=dest.parent);tmp=Path(tmpname)
 try:
  # The sha256 is computed while writing: re-reading the finished pack doubled the I/O.
  # It runs on one worker thread, in chunk order (hashlib and file I/O release the GIL),
  # so hashing (~190-450 MB/s here) overlaps the copy instead of adding to it.
  hasher=hashlib.sha256();queued=deque()
  def hash_chunk(chunk):
   queued.append(hashing.submit(hasher.update,chunk))
   if len(queued)>4:queued.popleft().result()   # at most ~40 MB of chunks in flight
  with ThreadPoolExecutor(1) as hashing,os.fdopen(fd,'wb') as out:
   # Fixed header timestamp: the pack hash then depends only on its contents, so the
   # compiled startpos provenance (compiler_sha256) still matches an identical rebuild.
   head=b'PFH5'+struct.pack('<6I',3,0,0,len(ordered),len(ix),PACK_TIMESTAMP)+ix
   out.write(head);hash_chunk(head)
   for name,p,body in prefetched(ordered,sizes):
    n=0
    if body is not None:
     if body:out.write(body);hash_chunk(body);n=len(body)
    else:
     with p.open('rb') as src:
      while chunk:=src.read(8*1024*1024):
       out.write(chunk);hash_chunk(chunk);n+=len(chunk)
    if n!=sizes[name]:raise RuntimeError('Source size changed while packing: '+name)
   while queued:queued.popleft().result()
   out.flush();os.fsync(out.fileno())
  result=inventory(tmp,False);result['sha256']=hasher.hexdigest()
  if result['dependencies']:raise RuntimeError('Unexpected pack dependencies')
  if any((p.stat().st_size,p.stat().st_mtime_ns)!=stats[name] for name,p in ordered):raise RuntimeError('Staged source changed while packing')
  if before_commit is not None:before_commit()
  if dest.exists():
   prior=dest.with_suffix('.previous-'+str(time.time_ns())+'.pack')
   try:os.link(dest,prior)
   except OSError:shutil.copy2(dest,prior)
  os.replace(tmp,dest);result['source']=str(dest)
  print('Pack:',dest,len(ordered),dest.stat().st_size,result['sha256'],flush=True);return result
 finally:
  if tmp.exists():tmp.unlink()


def logic():
 files={};base=KIT/'working_data/campaign_maps/3k_dlc07_main_map'
 for rel in ('map_data.esf','pathfinding.ppd','trade_routes.ptd','display/borders/borders.pbd'):files['campaign_maps/3k_dlc07_main_map/'+rel]=base/rel
 return files


def native_tables():
 # Re-export the binary start-pos/region tables from the current split XML every
 # time: a stale export (pre-rebase regions) against a fresh map crashes the compiler.
 subprocess.run([sys.executable,'-B',str(ROOT/'scripts/all_directions_build.py'),'native'],check=True)
 xml=A/'work/native-split-xml';db=A/'work/native-split-db';stale=[]
 for x in xml.glob('*.xml'):
  out=db/(x.stem+'_tables')/'!tw3k_exp'
  if out.exists() and out.stat().st_mtime_ns<x.stat().st_mtime_ns:stale.append(x.stem)
 if stale:raise RuntimeError('Native tables older than their XML: '+', '.join(sorted(stale)))


def compiler():
 # Logic only: the native startpos/AI-map compile reads no terrain. A pack without
 # terrain/ and the display campaign_maps files compiled an equivalent romance startpos
 # (only per-run timestamps, GUID and counters differ) and identical hlp/spd data
 # (startpos-terrain-experiment, 2026-10-22). So a terrain-only change leaves this
 # pack, and the compiled startpos, current. Battle entrances belong to runtime().
 # First: a map_data.esf exported from another map.hex fails here, before any table export,
 # pack or game session (verify_generated/current_sources refuse it as well).
 check_map_data_export(A)
 native_tables();_mark('compiler: native tables (SDK export)')
 def sources():return {'source_map_sha256':digest(A/'work/final-map/map.hex'),'source_design_sha256':digest(A/'work/current-region-design.json'),'source_map_data_sha256':digest(logic()['campaign_maps/3k_dlc07_main_map/map_data.esf'])}
 source_before=sources()
 def before_commit():
  if sources()!=source_before:raise RuntimeError('Compiler inputs changed while packing')
 files=logic()
 for p in (A/'work/native-split-db').glob('*_tables/!tw3k_exp'):files['db/'+p.parent.name+'/data_3k_main_campaign_map']=p
 assert sum(k.startswith('db/') for k in files)==57
 # Vanilla regions keep their vanilla provinces: the shipped vanilla junction
 # table stays in force (the old layout-v3 override copy re-provinced 9 of them).
 files['db/victory_objectives.txt']=A/'work/build-input/db/victory_objectives.txt'
 _mark('compiler: inputs gathered')
 # Reuse the existing pack when every input is as it was when it was written; its
 # unchanged sha then lets rebuild_section14_candidate.ps1 skip the native compilation too.
 fingerprint=input_fingerprint(files);manifest_path=A/'analysis/compiler-manifest.json';out=A/'compiler.pack'
 last=json.loads(manifest_path.read_text(encoding='utf-8-sig')) if manifest_path.exists() else {}
 if (out.exists() and last.get('input_fingerprint')==fingerprint and last.get('pack_size')==out.stat().st_size
     and last.get('pack_mtime_ns')==out.stat().st_mtime_ns and all(last.get(k)==v for k,v in source_before.items())):
  print('Compiler pack inputs unchanged; reused',out,last.get('sha256'),flush=True);_mark('compiler: pack reused');return
 m=pack(out,files,before_commit);_mark('compiler: pack written');m.update(source_before)
 m.update(input_fingerprint=fingerprint,pack_size=out.stat().st_size,pack_mtime_ns=out.stat().st_mtime_ns);atomic_json(manifest_path,m)


def extract_current_pack(dbpack,runtime):
 files={}
 with dbpack.open('rb') as f:
  for e in inventory(dbpack,False)['entries']:
   if e['compressed']:raise RuntimeError('Compressed runtime table unsupported')
   rel=e['path'];p=runtime/rel
   if '..' in Path(rel).parts or Path(rel).is_absolute():raise RuntimeError('Unsafe runtime table path')
   p.parent.mkdir(parents=True,exist_ok=True);f.seek(e['offset']);p.write_bytes(f.read(e['size']))
   if not rel.endswith('.rpfm_reserved'):files[rel]=p
 return files

def _started(cmd,**kw):
 """Start cmd now with its output held back; the returned call waits for it, prints the
 output at that point (log order unchanged) and raises as subprocess.run(check=True)."""
 p=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,**kw)
 def finish():
  out,err=p.communicate();sys.stdout.flush();sys.stdout.buffer.write(out);sys.stdout.flush();sys.stderr.flush();sys.stderr.buffer.write(err);sys.stderr.flush()
  if p.returncode:raise subprocess.CalledProcessError(p.returncode,cmd)
 return finish

def runtime():
 # --modes=romance builds a test candidate with one compiled start (the other
 # mode's startpos is left out and that mode must not be started).
 modes=tuple(next((a.split('=',1)[1].split(',') for a in sys.argv if a.startswith('--modes=')),('romance','historical')))
 # Intro cameras, cutscene placements and start-camera literals translated into
 # the expanded world frame. --check re-derives every file from vanilla first; it only
 # reads (vanilla packs, both map.hex, the camera stage), so its 6-19 s run overlaps the
 # provenance check and the runtime tables, and is collected where it used to run.
 camera_root=A/'camera-translation/stage';camera_python=TW3K_PYTHON
 camera_check=_started([str(camera_python),'-B',str(ROOT/'scripts/translate_campaign_cameras.py'),str(camera_root),'--check'],cwd=ROOT/'scripts')
 source_fingerprint=verify_generated(A,modes)
 # The strategic map art (region lookup, minimap, parchment) is drawn from map.hex by the terrain
 # build (rebuild_campaign_strategic_maps.py, which records the map it read); a map edit since then
 # would ship the old region lookup beside the new startpos.
 strategic=A/'analysis/strategic-map-rebuild.json'
 if not strategic.exists():raise RuntimeError(f'{strategic} is missing; run rebuild_campaign_strategic_maps.py (terrain build)')
 if json.loads(strategic.read_text(encoding='utf-8')).get('source_map_sha256')!=source_fingerprint['source_map_sha256']:
  raise RuntimeError('Strategic map art was drawn from another map.hex (analysis/strategic-map-rebuild.json); rerun rebuild_campaign_strategic_maps.py or the terrain build')
 battle_entrances()
 dbpack=A/'runtime-tables.pack'
 if dbpack.exists():dbpack.replace(A/('runtime-tables-'+str(time.time_ns())+'.pack'))
 # Camera zoom-out limits scale with the expanded world (campaign_variables fragment).
 subprocess.run([sys.executable,'-B',str(ROOT/'scripts/write_campaign_camera_variables.py')],check=True)
 subprocess.run([str(RPFM),'--game','three_kingdoms','pack','create','--pack-path',str(dbpack)],check=True)
 subprocess.run([str(RPFM),'--game','three_kingdoms','pack','add','--pack-path',str(dbpack),'--tsv-to-binary',str(SCHEMA),'--folder-path',str(STAGE/'db')+';db',str(STAGE/'text')+';text'],check=True)
 with tempfile.TemporaryDirectory(prefix='runtime-db-current-',dir=A) as temporary:
  runtime=Path(temporary);files=logic()
  for p in STAGE.rglob('*'):
   if not p.is_file() or p.name.endswith(('.tsv','.rpfm_reserved')):continue
   # props_bundle/map_hex write <name>.<pid>.tmp then os.replace: a leftover means that
   # step was killed before the replace and its target is still the previous version.
   if p.name.endswith('.tmp'):raise SystemExit(f'Interrupted write left {p}; rerun the step that writes it')
   rel=p.relative_to(STAGE).as_posix()
   if rel.startswith(('db/','text/')):continue
   if not _frame.ships_terrain_path(rel):continue   # 3k_main_map stays vanilla (190 renders dlc07)
   if '--omit-display-cache' in sys.argv and rel.startswith('terrain/campaigns/') and rel.endswith(('/tile_map.tiles','/tile_map.index')):continue
   files[rel]=p
  files.update(extract_current_pack(dbpack,runtime))
  for name in tuple('startpos_'+m+'.esf' for m in modes)+('hlp_data.esf','spd_data.esf'):
   files[('campaigns/3k_main_campaign_map/' if name.startswith('startpos_') else 'campaign_maps/3k_dlc07_main_map/')+name]=A/'generated'/name
  battlepack=ROOT/'mods/terrain_scenes/output/tw3k_terrain_scenes_offline.pack';battlefiles=runtime/'battle-scenes';overlaps=[]
  battleentries=extract_current_pack(battlepack,battlefiles)
  for name,p in battleentries.items():
   if name in files:raise RuntimeError('Unexpected source overlap: '+name)
   files[name]=p
  camera_check()
  # Only the 190 campaign is supported; other campaign folders stay vanilla to limit mod conflicts.
  camera_files={p.relative_to(camera_root).as_posix():p for p in camera_root.rglob('*') if p.is_file()}
  # The 190 Liu Yan start plays dlc07_guandu/.../scenes/liu_yan_190.CindyScene; ship the assets it references.
  liu_yan_190=tuple('script/campaign/dlc07_guandu/factions/campaing_intro_cutscenes/'+n for n in ('cameras/liu_yan_cam.xml','csc_ref/dong_zhuo_intro.kfcs','particles/smoke.kfe'))
  camera_files={k:p for k,p in camera_files.items() if k.startswith(('script/campaign/three_kingdoms_early/','script/benchmarks/campaign_benchmark/')) or k in liu_yan_190}
  if not all(k in camera_files for k in liu_yan_190):raise RuntimeError('190 Liu Yan intro camera assets missing from the camera translation')
  for name,p in camera_files.items():
   if name in files:raise RuntimeError('Unexpected source overlap: '+name)
   files[name]=p
  # The strategic-view faction colour layer reads the root lookup copies, not only
  # display/regions/: leaving them vanilla misaligned the colours and left new
  # regions uncoloured. Ship identical copies at the root paths.
  for name in ('3k_main_lookup.dds','3k_main_lookup.tga','3k_main_lookup_minimap.tga'):
   src='campaign_maps/3k_dlc07_main_map/display/regions/'+name;dst='campaign_maps/3k_dlc07_main_map/'+name
   if src not in files:raise RuntimeError('Missing current lookup '+src)
   if dst in files:raise RuntimeError('Unexpected source overlap: '+dst)
   files[dst]=files[src]
  if any(k.endswith('.lua') and k not in camera_files for k in files):raise RuntimeError('Validation Lua must not enter release')
  dest=ROOT/('output/tw3k_regions_and_battles_terrain_repair_candidate.pack' if '--terrain-repair-candidate' in sys.argv else 'output/tw3k_regions_and_battles_layout_v5_candidate.pack' if '--layout-v5-candidate' in sys.argv else 'output/tw3k_regions_and_battles_layout_v4_candidate.pack' if '--layout-v4-candidate' in sys.argv else 'output/tw3k_regions_and_battles_layout_v3_candidate.pack' if '--layout-v3-candidate' in sys.argv else 'output/tw3k_regions_and_battles_layout_v2_candidate.pack' if '--layout-v2-candidate' in sys.argv else 'output/tw3k_regions_and_battles.pack')
  def before_commit():
   if verify_generated(A,modes)!=source_fingerprint:raise RuntimeError('Source changed while packaging; existing candidate preserved')
  final=dest
  # Gate before publishing: the session installer takes whatever pack sits at
  # the release path, so a candidate is written beside it and only renamed onto
  # it after every gate below has passed.
  dest=final.with_name(final.stem+'.ungated.pack')   # every runtime pack is gated, including the default release name
  _mark('runtime: inputs gathered');m=pack(dest,files,before_commit);_mark('runtime: pack written')
 atomic_json(A/'analysis/combined-release-manifest.json',m)
 plan=json.loads((A/'work/current-region-design.json').read_text(encoding='utf-8'))
 atomic_json(A/'analysis/combined-source-equivalence.json',{'battle_pack_sha256':digest(battlepack),'battle_entries_identically_included':len(battleentries),'unexpected_overlaps':overlaps,'lua_in_release':'camera translations only','camera_translation_files':len(camera_files),'authored_regions':len(plan['regions']),'province_groups':len(plan['provinces']),'battle_scenes':54,'release_sha256':m['sha256'],'runtime_verified':False})
 if True:   # gates apply to every runtime pack (was: --terrain-repair-candidate only)
  terrain_python=TW3K_PYTHON
  # The gates only read the finished pack and write their own reports: run them
  # in parallel (they ran one after another), then fail on any non-zero exit.
  gates=[[str(terrain_python),'-B',str(ROOT/'scripts/repair_character_placeholder_positions.py')],
         [str(terrain_python),'-B',str(ROOT/'scripts/rebuild_campaign_strategic_maps.py'),'--audit','--pack',str(dest)],
         [str(terrain_python),'-B',str(ROOT/'scripts/sync_authored_camera_heights.py'),'--audit'],
         [str(terrain_python),'-B',str(ROOT/'scripts/repair_campaign_river_roots.py'),'--audit','--pack',str(dest)],
         [str(terrain_python),'-B',str(ROOT/'scripts/translate_campaign_lake_instances.py'),'--audit','--pack',str(dest)],
         [str(terrain_python),'-B',str(ROOT/'scripts/audit_map_asset_coverage.py'),'--pack',str(dest),'--output',str(A/'analysis/map-asset-coverage.json')],
         [str(terrain_python),'-B',str(ROOT/'scripts/audit_ground_seams.py'),'--pack',str(dest),'--out',str(ROOT/'analysis/ground-seam-section11-final-pack'),'--strict-first-five']]
  procs=[subprocess.Popen(c) for c in gates]
  codes=[p.wait() for p in procs]
  if any(codes):raise RuntimeError('Final pack gates failed: '+', '.join(Path(c[2]).name for c,k in zip(gates,codes) if k))
  _mark('runtime: gates')
  if final.exists():
   prior=final.with_suffix('.previous-'+str(time.time_ns())+'.pack')
   try:os.link(final,prior)
   except OSError:shutil.copy2(final,prior)
  os.replace(dest,final);m['source']=str(final);atomic_json(A/'analysis/combined-release-manifest.json',m)
  print('Gates passed; published',final,flush=True)

def world_table():
 values=[{c.tag:c.text for c in row} for row in ET.parse(KIT/'raw_data/db/campaign_maps.xml').getroot().findall('campaign_maps') if row.findtext('mapname') in _frame.TERRAIN_VARIANTS]
 (A/'analysis/expanded-world-bounds.json').write_text(json.dumps(values,indent=2))
 print('World bounds are stored by the native map_data exporter; campaign_maps is an SDK source table')

if __name__=='__main__':{'exports':exports,'compiler':compiler,'runtime':runtime,'world-table':world_table}[sys.argv[1]]()



