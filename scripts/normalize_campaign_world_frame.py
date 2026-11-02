"""Match native map-data coordinates to the actual retained terrain cell spacing."""
import json,subprocess,shutil
from analyze_reference import ROOT,read_hex,digest
import campaign_world as _frame
from campaign_world import WORLD_WIDTH,assert_grid
A=_frame.WORKSPACE;K=A/'work/standalone_kit';P=K/'working_data/campaign_maps/3k_dlc07_main_map/map_data.esf';m=read_hex(A/'work/final-map/map.hex');assert_grid(m['width'],m['height']);target_width=WORLD_WIDTH
out=A/'analysis/map-data-coherent-frame.esf';before=digest(P)
subprocess.run([str(ROOT/'tools/caime-1.1.0/lib/app/Tw3kWorldExtent.exe'),str(P),str(out),str(target_width)],check=True)
shutil.copy2(out,P)
(A/'analysis/coherent-world-frame.json').write_text(json.dumps({'source_map_sha256':m['sha256'],'native_width':595.1,'terrain_hex_pitch':_frame.DX,'tile_aligned_frame':_frame.TILE_ALIGNED_FRAME,'target_world_width':target_width,'input_sha256':before,'output_sha256':digest(P),'logical_grid_and_slot_indices_changed':False,'visual_acceptance_inferred':False},indent=2),encoding='utf-8');print('Native map-data world span normalized to retained terrain spacing')
