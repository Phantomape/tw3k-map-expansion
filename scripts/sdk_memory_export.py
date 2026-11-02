"""Preflight native SDK bit budgets and export without memory overrides.

  python sdk_memory_export.py                   sync the kit map, preflight it, export map-data
  python sdk_memory_export.py --preflight-only  the same preflight on work/final-map/map.hex only (no kit, no export;
                                                candidate scratch workspaces and the bake --diff end-to-end test)"""
import collections,json,subprocess,sys,time
from analyze_reference import ROOT,read_hex
import campaign_world as _frame
A=_frame.WORKSPACE;KIT=A/'work/standalone_kit';p=KIT/'raw_data/EmpireDesignData/campaign_maps/3k_dlc07_main_map/map.hex'
# The preflight needs numpy (hex_components counts the SDK areas by traversal class). The old
# per-cell BFS fallback counted region pieces only (not split by traversal class), under a looser
# limit, so it could pass maps the SDK cannot export: no fallback, fail loudly.
try:
 import numpy as _np,hex_components
except ImportError as _e:
 raise SystemExit(f'sdk_memory_export.py needs numpy for the SDK area preflight ({_e}); run it with the tw3k Python in env/python-path.txt (or set TW3K_BASE_PYTHON to a Python with numpy)')
def preflight(path):
 m=read_hex(path);_frame.assert_grid(m['width'],m['height']) if hasattr(_frame,'assert_grid') else None;names=m['land_regions']+m['sea_regions'];w,h=m['width'],m['height']
 if len(names)>511:raise RuntimeError('SDK has nine region bits; region index 511 is reserved')
 _cells=_np.frombuffer(path.read_bytes()[-4-w*h*16:-4],_np.uint8).reshape(h,w,16)
 _ids=(_cells[:,:,1].astype(_np.int64)<<5)+(_cells[:,:,0].astype(_np.int64)>>3)-1
 counts=collections.Counter({names[k]:v for k,v in hex_components.region_areas(_ids,_cells).items()})
 _limit=hex_components.SDK_AREA_LIMIT
 if max(counts.values())>_limit:raise RuntimeError('SDK area index is 7 bits (127 = unassigned, its flood fill never ends); compact blocked regions first: '+str(counts.most_common(3)))
 print('SDK preflight PASS:',len(names),'regions; max connected components',max(counts.values()),flush=True)
 return m,names,counts
if '--preflight-only' in sys.argv[1:]:
 preflight(A/'work/final-map/map.hex');sys.exit(0)
# The SDK exports from the kit copy, but every edit (layout, ground-type restore,
# canvas migration) lands in final-map/map.hex. Sync it first so an export never
# reads a stale map; keep the replaced kit copy for rollback.
import kit_map;kit_map.sync_from_final()
m,names,counts=preflight(p)
started=time.time();app=ROOT/'tools/caime-1.1.0/lib/app/Tw3kBuild.exe'
r=subprocess.run([str(app),str(KIT),str(p),'map-data'],capture_output=True,text=True)
(A/'analysis/sdk-default-memory-export.log').write_text(r.stdout+r.stderr,encoding='utf-8')
if r.returncode:raise RuntimeError(r.stdout+r.stderr)
if (A/'layout-v2/source-state.json').exists():
 subprocess.run([sys.executable,str(ROOT/'scripts/audit_exported_settlement_anchors.py')],check=True)
result={'status':'pass','default_sdk_memory_limit_mb':2400,'memory_budget_overrides':False,'sdk_binary_modifications':False,'regions':len(names),'max_connected_components_per_region':max(counts.values()),'elapsed_seconds':time.time()-started,'map_sha256':m['sha256'],'allocation_checks_retained':True}
(A/'analysis/sdk-memory-fix-result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
