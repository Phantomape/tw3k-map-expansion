"""Synchronize authored town camera/CPU height samples even on unchanged DDS pixels.

An unchanged saturated LF pixel must not preserve a different old camera height
inside a new settlement. Leave every pixel outside the tiny authored cores exact.
"""
import json,io,struct,hashlib,argparse
from pathlib import Path
import numpy as np
from PIL import Image,ImageFilter,PngImagePlugin
from analyze_reference import ROOT
import campaign_world as _frame
from final_terrain_surface import load_final_surface,sample_final_surface
from campaign_world import TERRAIN_VARIANTS

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--audit',action='store_true');args=ap.parse_args()
    area=_frame.WORKSPACE;stage=area/'work/release-input';plan=json.loads((area/'work/current-region-design.json').read_text(encoding='utf-8'))
    centres=np.zeros((_frame.TILE_H,_frame.TILE_W),np.uint8)
    for c in plan['regions']:
        q,r=c['town'];centres[r*2+q%2,q*2]=255
    mask=np.asarray(Image.fromarray(centres).filter(ImageFilter.MaxFilter(5)))>0
    original=(_frame.VANILLA_MAP).read_bytes();cells=np.frombuffer(original[-4-_frame.NATIVE_W*_frame.NATIVE_H*16:-4],np.uint8).reshape(_frame.NATIVE_H,_frame.NATIVE_W,16)
    native_plot=((cells[:,:,2]>>4)>0)|((cells[:,:,3]&1)>0)
    yy,xx=np.mgrid[:_frame.TILE_H,:_frame.TILE_W];q,r=_frame.tile_hex(xx,yy);q=q-_frame.PAD_LEFT;r=r-_frame.PAD_BOTTOM   # tile owner (cw.tile_hex), native hex frame
    valid=(q>=0)&(q<_frame.NATIVE_W)&(r>=0)&(r<_frame.NATIVE_H);native=np.zeros(mask.shape,bool);native[valid]=native_plot[r[valid],q[valid]]
    assert not np.any(mask&native),'Authored camera core overlaps a preserved native plot'
    ys,xs=np.nonzero(mask);rows=[]
    for variant in TERRAIN_VARIANTS:
        path=stage/f'campaign_maps/{variant}/camera_heightmap.png';before=path.read_bytes();im=Image.open(io.BytesIO(before));pixels=np.asarray(im).copy()[::-1];scale=float(im.info['height_scale']);surface=load_final_surface(variant,stage)
        desired=np.rint(np.maximum(sample_final_surface(surface,xs,ys),0)/scale).clip(0,65535).astype(np.uint16)
        if args.audit:assert np.array_equal(pixels[ys,xs],desired),variant+' authored CPU/camera heights stale'
        else:
            backup=area/'terrain-repair/camera-town-before'/(hashlib.sha256(before).hexdigest()+'.png');backup.parent.mkdir(parents=True,exist_ok=True)
            if not backup.exists():backup.write_bytes(before)
            original_pixels=pixels.copy();pixels[ys,xs]=desired;assert np.array_equal(pixels[~mask],original_pixels[~mask])
            info=PngImagePlugin.PngInfo()
            for k,v in im.info.items():
                if isinstance(v,str):info.add_text(k,v)
            out=io.BytesIO();Image.fromarray(np.ascontiguousarray(pixels[::-1])).save(out,format='PNG',pnginfo=info);path.write_bytes(out.getvalue())
        rows.append(dict(variant=variant,authored_towns=len(plan['regions']),core_pixels=int(mask.sum()),native_plot_pixels_touched=0,all_other_camera_pixels_preserved=True))
    (area/'analysis/authored-camera-height-sync.json').write_text(json.dumps(dict(variants=rows,audit=args.audit),indent=2));print(f'Authored camera/CPU cores: PASS ({len(plan["regions"])} towns per variant; native plots untouched)',flush=True)

if __name__=='__main__':main()
