"""Reproject native BC1 water colour into the translated expanded theatre.

Water materials share the Main and DLC06 paths in both campaign variants.
Use the pixel shader's UV=(worldX/W, 1-worldZ/H), not LF row offsets.
"""
import tw3k_paths
import io, json, math, struct, hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from analyze_reference import ROOT, inventory, digest
import campaign_world as _frame
from campaign_world import DX, DZ

A=_frame.WORKSPACE; S=A/'work/release-input'
GAME=tw3k_paths.GAME_DATA
# The engine's g_world_bounds: map_data width, and tile rows x (width / tile
# columns) x ENGINE_Z_RATIO. engine_bounds() is checked against a replayed
# capture in main() before it is trusted for this canvas.
def engine_bounds(width, tile_w=_frame.TILE_W, tile_h=_frame.TILE_H):
    return float(np.float32(width)), float(np.float32(tile_h * width / tile_w * _frame.ENGINE_Z_RATIO))
NEW_BOUNDS=engine_bounds(_frame.WORLD_WIDTH)
# Native bounds read from the original game's actual water pixel shader.
OLD_BOUNDS=(595.0999755859375,541.2066650390625)
SHIFT=_frame.NATIVE_SHIFT_WORLD
KEYS=('terrain/campaigns/3k_main_map/sea_colour.dds',
      'terrain/campaigns/3k_dlc06_main_map/sea_colour.dds')

def decode_srgb_bc1(data):
    # Pillow does not recognise DXGI 72; its BC1 decoder reads the same bytes
    # under DXGI 71. This temporary header change performs no colour conversion.
    readable=bytearray(data);struct.pack_into('<I',readable,128,71)
    return Image.open(io.BytesIO(readable))

def native_assets():
    result={}
    for name in ('terrain.pack','fast.pack','data.pack'):
        pack=GAME/name
        for e in inventory(pack,False)['entries']:
            if e['path'] in KEYS: result.setdefault(e['path'],(pack,e))
    return result

def dds_bc1(image):
    """Pillow's BC1 encoder, explicit DX10 sRGB header and complete mip chain."""
    levels=[]; sizes=[]; first=None
    while True:
        stream=io.BytesIO();image.save(stream,format='DDS',pixel_format='DXT1')
        data=stream.getvalue();first=first or data[:128]
        expected=max(1,(image.width+3)//4)*max(1,(image.height+3)//4)*8
        assert len(data)==128+expected
        levels.append(data[128:]);sizes.append(image.size)
        if image.size==(1,1):break
        image=image.resize((max(1,image.width//2),max(1,image.height//2)),Image.Resampling.BOX)
    header=bytearray(first);struct.pack_into('<I',header,8,0xA1007)
    struct.pack_into('<I',header,20,len(levels[0]));struct.pack_into('<I',header,28,len(levels))
    header[84:88]=b'DX10';struct.pack_into('<I',header,108,0x401008)
    return bytes(header)+struct.pack('<5I',72,3,0,1,0)+b''.join(levels),sizes

RIM_INSET=192     # output px dropped inside the native edge (the native texture has its own frame)
INNER_PX=384      # output px inside the box over which native colour cross-fades into the field
FADE_PX=2400      # output px (~300 hexes at 8 px/hex) from native rim colour to the ocean colour
WORK=8            # the exterior field is smooth: built at 1/WORK resolution
RIM_BLUR=24       # field cells (x WORK output px): blur of the extrapolated rim

def native_box(ow,oh,nw,nh,ax,bx,ay,by):
    """Output-pixel box (x0,y0,x1,y1) holding native texels, inset by RIM_INSET."""
    x0=-bx/ax;x1=(ow-bx)/ax;y0=-by/ay;y1=(oh-by)/ay
    return (int(math.ceil(x0))+RIM_INSET,int(math.ceil(y0))+RIM_INSET,int(x1)-RIM_INSET,int(y1)-RIM_INSET)

def _push_pull(rgb,valid):
    """Fill invalid pixels with a smooth extrapolation of the valid ones (weighted pyramid)."""
    if min(rgb.shape[:2])<=2 or valid.all():
        w=valid[...,None].astype(np.float32);s=(rgb*w).sum((0,1))/max(float(w.sum()),1.)
        return np.where(valid[...,None],rgb,s).astype(np.float32)
    h,w=valid.shape;H2,W2=(h+1)//2*2,(w+1)//2*2
    pr=np.zeros((H2,W2,3),np.float32);pw=np.zeros((H2,W2),np.float32)
    pr[:h,:w]=rgb*valid[...,None];pw[:h,:w]=valid
    sr=pr.reshape(H2//2,2,W2//2,2,3).sum((1,3));sw=pw.reshape(H2//2,2,W2//2,2).sum((1,3))
    coarse=_push_pull(sr/np.maximum(sw,1e-6)[...,None],sw>0)
    up=np.stack([np.asarray(Image.fromarray(np.ascontiguousarray(coarse[...,c],np.float32)).resize((W2,H2),Image.Resampling.BILINEAR)) for c in range(3)],-1)[:h,:w]
    return np.where(valid[...,None],rgb,up).astype(np.float32)

def _box3(a,r):
    """Three separable box blurs of radius r (edge-padded): close to a Gaussian."""
    for _ in range(3):
        for axis in (0,1):
            n=a.shape[axis];pad=[(0,0)]*a.ndim;pad[axis]=(r+1,r)
            cs=np.cumsum(np.pad(a,pad,mode='edge'),axis=axis,dtype=np.float64)
            a=((np.take(cs,np.arange(2*r+1,2*r+1+n),axis=axis)-np.take(cs,np.arange(n),axis=axis))/(2*r+1)).astype(np.float32)
    return a

def feather_exterior(pixels,box,ocean,seed=5):
    """Native colour inside box; outside it, the extrapolated rim fading to the ocean colour."""
    nh,nw=pixels.shape[:2];x0,y0,x1,y1=box
    sw,sh=(nw+WORK-1)//WORK,(nh+WORK-1)//WORK
    small=np.asarray(Image.fromarray(pixels).resize((sw,sh),Image.Resampling.BOX),np.float32)
    yy,xx=np.mgrid[:sh,:sw].astype(np.float32)*WORK+WORK/2
    inside=(xx>=x0)&(xx<x1)&(yy>=y0)&(yy<y1)
    # The push-pull fill steps from exact native texels to its coarse average at
    # the box edge (a faint straight seam in the colour preview); blurred, the
    # field is continuous there and the inner cross-fade brings the detail back.
    rim=_box3(_push_pull(small,inside),RIM_BLUR)
    # Distance outside the inset native box, meandered by smooth noise (no straight fade line).
    d=np.hypot(np.maximum(np.maximum(x0-xx,xx-x1),0),np.maximum(np.maximum(y0-yy,yy-y1),0))
    rng=np.random.default_rng(seed);n=np.zeros((sh,sw),np.float32)
    for k,a in ((48,.6),(16,.3),(6,.1)):
        g=rng.random((max(2,sh//k),max(2,sw//k))).astype(np.float32)
        n+=a*np.asarray(Image.fromarray(g).resize((sw,sh),Image.Resampling.BICUBIC))
    t=np.clip(d*(.6+.8*n)/FADE_PX,0,1);t=t*t*(3-2*t)
    field=rim*(1-t[...,None])+np.asarray(ocean,np.float32)*t[...,None]
    # big = the field resized to full size (nh x nw x 3 float32) and out=np.clip(np.rint(big),0,255) as
    # uint8, one channel and one row band at a time; only the cross-fade bands keep their big values.
    # Same values as the whole-array version; the peak commit falls from ~40 to ~15 bytes per pixel.
    bands=((slice(y0,y0+INNER_PX),slice(x0,x1)),(slice(y1-INNER_PX,y1),slice(x0,x1)),(slice(y0,y1),slice(x0,x0+INNER_PX)),(slice(y0,y1),slice(x1-INNER_PX,x1)))
    out=np.empty((nh,nw,3),np.uint8);kept=[[] for _ in bands]
    for c in range(3):
        channel=np.asarray(Image.fromarray(np.ascontiguousarray(field[...,c])).resize((nw,nh),Image.Resampling.BICUBIC))
        for r in range(0,nh,512):
            out[r:r+512,:,c]=np.clip(np.rint(channel[r:r+512]),0,255).astype(np.uint8)
        for k,(ys,xs) in zip(kept,bands):k.append(channel[ys,xs].copy())
        del channel
    # Inside the box keep native colour, cross-faded over INNER_PX from its edge so
    # the native frame and the box outline leave no straight line.
    out[y0:y1,x0:x1]=pixels[y0:y1,x0:x1]
    for (ys,xs),k in zip(bands,kept):
        big=np.stack(k,-1);Y,X=np.ogrid[ys,xs]
        di=np.minimum(np.minimum(X-x0,x1-1-X),np.minimum(Y-y0,y1-1-Y)).astype(np.float32)
        w=np.clip(di/INNER_PX,0,1);w=(w*w*(3-2*w))[...,None]
        out[ys,xs]=np.clip(np.rint(pixels[ys,xs]*w+big*(1-w)),0,255).astype(np.uint8)
    return out

def main():
    evidence=A/'terrain-repair/shader-audit/water-world-bounds.json'
    bounds=json.loads(evidence.read_text())
    # The replay was captured on the 1116x798 canvas with the legacy 744.7099
    # width; the bounds formula must reproduce it before it is used here.
    assert bounds.get('replay_verified')
    measured=tuple(bounds['constants']['g_world_bounds'])[2:]
    predicted=engine_bounds(measured[0],2*1116,2*798+1)
    if max(abs(a-b) for a,b in zip(predicted,measured))>1e-3:
        raise RuntimeError(f'g_world_bounds formula {predicted} disagrees with the replay {measured}')
    if abs(NEW_BOUNDS[1]-_frame.WORLD_HEIGHT)>1e-3:
        raise RuntimeError(f'Engine world height {NEW_BOUNDS[1]:.4f} differs from campaign_world {_frame.WORLD_HEIGHT:.4f}')
    native_evidence=A/'terrain-repair/shader-audit/native-water-world-bounds.json'
    native_bounds=json.loads(native_evidence.read_text())
    assert native_bounds.get('replay_verified') and tuple(native_bounds['constants']['g_world_bounds'])==(0.,0.,*OLD_BOUNDS)
    reports=[]
    for key,(pack,e) in native_assets().items():
        assert not e['compressed']
        with pack.open('rb') as f:f.seek(e['offset']);source=f.read(e['size'])
        assert source[84:88]==b'DX10' and struct.unpack_from('<I',source,128)[0]==72
        decoded=decode_srgb_bc1(source)
        assert decoded.getchannel('A').getextrema()==(255,255), 'Native water alpha must be preserved'
        original=decoded.convert('RGB');ow,oh=original.size
        nw=int(round(ow*NEW_BOUNDS[0]/OLD_BOUNDS[0]/4))*4
        nh=int(round(oh*NEW_BOUNDS[1]/OLD_BOUNDS[1]/4))*4
        # Sample open ocean in the native eastern edge; retain its native colour.
        ocean=np.asarray(original)[:,int(ow*.97):,:].reshape(-1,3)
        colour=tuple(int(v) for v in np.median(ocean,axis=0))
        ax=NEW_BOUNDS[0]/nw*ow/OLD_BOUNDS[0]
        ay=NEW_BOUNDS[1]/nh*oh/OLD_BOUNDS[1]
        bx=-SHIFT[0]*ow/OLD_BOUNDS[0]
        by=(OLD_BOUNDS[1]-NEW_BOUNDS[1]+SHIFT[1])*oh/OLD_BOUNDS[1]
        expanded=original.transform((nw,nh),Image.Transform.AFFINE,(ax,0,bx,0,ay,by),Image.Resampling.BILINEAR,fillcolor=colour)
        # A flat fill met the native colour in a straight rectangle (live 2048j:
        # Kyushu, Kimhae, Rinan, Lin Yi). Extrapolate the native rim outward and
        # fade it to the open-ocean colour over a meandering band instead.
        expanded=Image.fromarray(feather_exterior(np.asarray(expanded),native_box(ow,oh,nw,nh,ax,bx,ay,by),colour))
        body,mips=dds_bc1(expanded);path=S/key;path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists() and path.read_bytes()!=body:
            backup=A/'terrain-repair/water-colour-before'/digest(path)/key
            backup.parent.mkdir(parents=True,exist_ok=True)
            if not backup.exists():backup.write_bytes(path.read_bytes())
        path.write_bytes(body)
        # Validate the encoded header, every mip length and affine world mapping.
        assert decode_srgb_bc1(body).size==(nw,nh)
        assert len(body)==148+sum(max(1,(w+3)//4)*max(1,(h+3)//4)*8 for w,h in mips)
        errors=[]
        for x,z in ((0,0),(240,373),(500,450),OLD_BOUNDS):
            px=(x+SHIFT[0])/NEW_BOUNDS[0]*nw
            py=(1-(z+SHIFT[1])/NEW_BOUNDS[1])*nh
            errors.append(max(abs(ax*px+bx-x/OLD_BOUNDS[0]*ow),abs(ay*py+by-(1-z/OLD_BOUNDS[1])*oh)))
        assert max(errors)<1e-8
        reports.append({'path':key,'native_size':[ow,oh],'expanded_size':[nw,nh],
            'native_sha256':hashlib.sha256(source).hexdigest(),'output_sha256':digest(path),
            'format':'DX10 BC1_UNORM_SRGB','mips':mips,'ocean_rgb':colour,
            'output_to_native_affine':[ax,0,bx,0,ay,by],'maximum_mapping_error_pixels':max(errors)})
        print(key,nw,nh,len(mips),'mips: PASS',flush=True)
    assert len(reports)==2
    (A/'analysis/water-colour-expansion.json').write_text(json.dumps({
        'old_world_bounds':OLD_BOUNDS,'new_world_bounds':NEW_BOUNDS,'translation_world':SHIFT,
        'expanded_bounds_source':'terrain-repair/shader-audit/water-world-bounds.json',
        'source_map_sha256':digest(A/'work/final-map/map.hex'),
        'source_design_sha256':digest(A/'work/current-region-design.json'),
        'native_bounds_source':'terrain-repair/shader-audit/native-water-world-bounds.json',
        'shared_material_paths':True,'new_exterior_water_colour':'native open-ocean median',
        'live_verified':False,'textures':reports},indent=2),encoding='utf-8')

if __name__=='__main__':main()
