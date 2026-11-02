"""Keep meshes, foliage and camera on point samples of the final game DDS."""
import io,json,subprocess,sys,time
from pathlib import Path
import numpy as np
from PIL import Image,PngImagePlugin
from analyze_reference import ROOT
import campaign_world as _frame
from final_terrain_surface import load_final_surface,point_resample_low_surface
from smooth_native_canvas_boundary import smooth_variant
from campaign_world import TERRAIN_VARIANTS
# Native camera heights stay vanilla wherever the LF itself is unchanged (shared
# edit definition); the blurred polish alpha alone also covers untouched rings.
from native_edit_mask import edited_tiles
# [sync-timing] lines: each part's wall time (the pipeline log keeps them; the --cache-only polish prints its own)
_TL=[time.time()]
def _mark(label):
 print(f'[sync-timing] {label}: {time.time()-_TL[0]:.1f}s',flush=True);_TL[0]=time.time()
A=_frame.WORKSPACE;S=A/'work/release-input';BASE=A/'terrain-before-polish';rows=[];union=np.zeros((_frame.TILE_H,_frame.TILE_W),np.float32)
for variant in TERRAIN_VARIANTS:
 smooth_variant(variant);_mark('smooth_variant')
 field=load_final_surface(variant,S);surface=point_resample_low_surface(field);old=point_resample_low_surface(load_final_surface(variant,BASE));_mark('resample final + base');previous=np.load(A/f'analysis/polished-surface-{variant}.npy');alpha=np.load(A/f'analysis/polished-edit-alpha-{variant}.npy');changed=np.abs(surface-old)>1e-5;alpha[changed]=np.maximum(alpha[changed],.03);union=np.maximum(union,alpha)
 np.save(A/f'analysis/polished-surface-{variant}.npy',surface);np.save(A/f'analysis/polished-edit-alpha-{variant}.npy',alpha);_mark('save surface + alpha')
 rel=f'campaign_maps/{variant}/camera_heightmap.png';im=Image.open(BASE/rel);scale=float(im.info['height_scale']);native=np.asarray(im);values=np.rint(np.maximum(surface[::-1],0)/scale).clip(0,65535).astype(np.uint16);keep=(alpha==0)|~edited_tiles(variant)[:alpha.shape[0],:alpha.shape[1]];keep=keep[::-1];values[keep]=native[keep];meta=PngImagePlugin.PngInfo()
 for key,value in im.info.items():
  if isinstance(value,str):meta.add_text(key,value)
 stream=io.BytesIO();Image.fromarray(values).save(stream,format='PNG',pnginfo=meta);(S/rel).write_bytes(stream.getvalue());_mark('camera heightmap')
 rows.append({'variant':variant,'sampling':'final DDS pixels (4i,4j): tile t at LF 4t','maximum_difference_from_antialiased_reduction':float(np.max(np.abs(surface-previous))),'changed_low_samples':int(changed.sum()),'camera_synced':True})
np.save(A/'analysis/polished-edit-alpha.npy',union);(A/'analysis/final-surface-sync.json').write_text(json.dumps({'variants':rows,'live_verified':False},indent=2),encoding='utf-8');print(json.dumps(rows),flush=True)
# Generated display-cache ranges must follow the corrected full-resolution DDS.
# The polish beside this script (a scripts copy must not run the main tree's).
subprocess.run([sys.executable,'-B',str(Path(__file__).resolve().parent/'polish_campaign_terrain.py'),'--cache-only'],check=True);_mark('polish --cache-only')
