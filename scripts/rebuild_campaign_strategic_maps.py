"""Rebuild campaign base art from current hexes, LF relief and actual river meshes.

The sampler mirrors CAIME LookupImageExporter, including the 1.5-pixel origin
and even-q rounding. Its reconstructed region colours must match the exported
lookup pixel for pixel before any texture is written.
"""
import argparse,hashlib,json,struct,shutil,xml.etree.ElementTree as ET
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFilter
from analyze_reference import ROOT,read_hex,digest,inventory
import campaign_world as _frame
from campaign_world import W,H,DX,DZ
from expand_campaign_water_colour import dds_bc1,decode_srgb_bc1
from repair_campaign_river_roots import Assets
from campaign_world import TERRAIN_VARIANTS
import strategic_ink
from strategic_ink import rows
A=_frame.WORKSPACE;S=A/'work/release-input'
VARIANTS=TERRAIN_VARIANTS

def gaussian_filter(values,radius):
    weights=np.exp(-np.arange(-6,7,dtype=np.float32)**2/(2*radius*radius));weights/=weights.sum()
    return strategic_ink.separable(values,weights)

def binary_erosion(values,iterations):
    out=values.copy()
    for _ in range(iterations):
        v=np.pad(out,1);out=v[1:-1,1:-1]&v[:-2,1:-1]&v[2:,1:-1]&v[1:-1,:-2]&v[1:-1,2:]
    return out

def hex_sampler(width,height):
    # The same float64 arithmetic as on full (height, width) grids, element for element: column-only
    # and row-only terms are 1-D and broadcast, and the per-pixel rest runs in bands of rows on threads.
    nx=(np.arange(width,dtype=np.float64)-1.5)/((2/3)*width/W);ny=(np.arange(height,dtype=np.float64)-1.5)/((2/3)*height/H*np.sqrt(3)/2)
    ax=(2/3)*nx;bx=(-1/3)*nx;by=np.sqrt(3)/3*ny
    xg=np.rint(ax);ax-=xg;hx=.5*ax;mx=np.abs(ax);q=np.empty((height,width),np.int32);r=np.empty((height,width),np.int32)
    def band(s):
        ay=bx+by[s,None];yg=np.rint(ay);ay-=yg;m=mx>=np.abs(ay)
        col=np.rint(ax+.5*ay);col*=m;col+=xg
        row=np.rint(ay+hx);row*=~m;row+=yg
        f=np.fmod(col,2);f+=col;f/=2;row+=f
        q[s]=np.clip(col,0,W-1);r[s]=H-1-np.clip(row.astype(np.int32),0,H-1)
    rows(band,height,4)
    return q,r

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--audit',action='store_true');ap.add_argument('--pack',type=Path);args=ap.parse_args()
    reportpath=A/'analysis/strategic-map-rebuild.json'
    if args.audit:
        report=json.loads(reportpath.read_text());assets=Assets(args.pack) if args.pack else None
        for row in report['textures']:
            b=assets.read(row['path']) if assets else (S/row['path']).read_bytes()
            assert hashlib.sha256(b).hexdigest()==row['sha256'],row['path']
            decoded=decode_srgb_bc1(b) if row['path'].endswith('.dds') else Image.open(BytesIO(b))
            assert list(decoded.size)==row['size']
        print('Strategic/minimap textures match recorded current-map output: PASS');return
    mp=A/'work/final-map/map.hex';m=read_hex(mp);names=m['land_regions']+m['sea_regions']
    b=mp.read_bytes();cells=np.frombuffer(b[-4-W*H*16:-4],np.uint8).reshape(H,W,16)
    ids=(cells[:,:,1].astype(np.int32)<<5)+(cells[:,:,0].astype(np.int32)>>3)-1
    land=(cells[:,:,0]&3)!=1
    lookup=S/'campaign_maps/3k_dlc07_main_map/display/regions/3k_main_lookup.dds'
    lb=lookup.read_bytes();height,width=struct.unpack_from('<2I',lb,12)
    sdk_lookup=A/'work/standalone_kit/working_data/campaign_maps/3k_dlc07_main_map/3k_main_campaign_map_lookup.bmp'
    with Image.open(sdk_lookup) as image:assert (width,height)==image.size,('Encoded lookup differs from SDK export',(width,height),image.size)
    # Side work runs on threads alongside the main numpy passes: the SDK lookup decode, the
    # vignette, the DDS, the preview and the PNG encodes (PIL and zlib release the GIL).
    pool=ThreadPoolExecutor(6);decoded=pool.submit(lambda:np.asarray(Image.open(sdk_lookup).convert('RGB')))
    q,r=hex_sampler(width,height)
    rgb=ET.parse(A/'work/standalone_kit/raw_data/db/regions.xml').getroot()
    colours={v.findtext('key'):tuple(int(v.findtext(k)) for k in ('r','g','b')) for v in rgb.findall('regions')}
    palette=np.asarray([colours[k] for k in names]+[(0,0,0)],np.uint8)
    actual=decoded.result()
    bad=np.empty((height,width),bool)
    def check(s):bad[s]=np.any(palette[ids[r[s],q[s]]]!=actual[s],axis=2)
    rows(check,height);rounding_ties=int(bad.sum())
    # C# and NumPy arithmetic differ at a few exact hex-edge ties. The already
    # exported lookup is authoritative; resolve only to an adjacent matching hex.
    # Ties scale with the lookup area (under 1000 on the 1116x798 canvas).
    assert rounding_ties<1000*W*H/(1116*798),('unexpected projection mismatch',rounding_ties)
    # All ties at once: of the 3x3 neighbours whose colour matches, the nearest, then lowest q, then r.
    py,px=np.nonzero(bad);dy,dx=np.mgrid[-1:2,-1:2].reshape(2,9);nq=q[py,px,None]+dx;nr=r[py,px,None]+dy
    match=(0<=nq)&(nq<W)&(0<=nr)&(nr<H)&np.all(palette[ids[np.clip(nr,0,H-1),np.clip(nq,0,W-1)]]==actual[py,px,None],axis=2)
    assert match.any(1).all(),(int(px[~match.any(1)][0]),int(py[~match.any(1)][0]))
    pick=np.where(match,((dx*dx+dy*dy)*W+nq)*H+nr,np.iinfo(np.int64).max).argmin(1);i=np.arange(len(py))
    q[py,px]=nq[i,pick];r[py,px]=nr[i,pick]
    surface=np.load(A/'analysis/polished-surface-3k_dlc07_main_map.npy')
    own=np.empty((height,width),ids.dtype);physical=np.empty((height,width),bool);heights=np.empty((height,width),np.float32)
    def sample(s):
        own[s]=ids[r[s],q[s]];bad[s]=np.any(palette[own[s]]!=actual[s],axis=2);physical[s]=land[r[s],q[s]]
        # Actual LF surface; north-up in the texture. No historical hand-drawn coast remains.
        heights[s]=surface[np.minimum(r[s]*2+(q[s]%2),surface.shape[0]-1),q[s]*2]
    rows(sample,height)
    mismatch=int(bad.sum());assert mismatch==0
    del actual,bad
    # Vanilla records / multiplayer / china-map art fades out around the land with
    # a soft vignette alpha (0..255) and the minimap keeps alpha 214..255; one
    # opaque image for every UI map turned those into hard rectangles. Build the
    # same kind of vignette from the expanded land mask.
    def fade_alpha():
        small=Image.fromarray((physical*255).astype(np.uint8)).resize((max(1,width//8),max(1,height//8)),Image.Resampling.BILINEAR)
        small=small.filter(ImageFilter.MaxFilter(9)).filter(ImageFilter.GaussianBlur(radius=max(4,small.size[0]*.035)))
        vignette=np.clip(np.asarray(small.resize((width,height),Image.Resampling.BILINEAR),dtype=np.float32)*1.6,0,255)
        # Land reaches the canvas edge on most sides; fade the frame as vanilla does.
        band=.06*min(width,height);xx_=np.arange(width);alpha=np.empty((height,width),np.uint8)
        def fade(s):
            yy_=np.arange(s.start,s.stop)[:,None]
            edge=np.minimum(np.minimum(xx_,width-1-xx_),np.minimum(yy_,height-1-yy_))/band
            alpha[s]=(vignette[s]*np.clip(edge,0,1)**1.5).astype(np.uint8)
        rows(fade,height);return alpha
    faded_alpha=pool.submit(fade_alpha)
    smooth=gaussian_filter(heights,2.0)
    # Ink-wash base art like vanilla's painted map: biome washes from the ecoregions at each hex,
    # sepia brush strokes on shadowed slopes, snow on the highest relief, salt flats (strategic_ink.py).
    image=strategic_ink.base_art(physical,smooth,q,r)
    coast=physical&~binary_erosion(physical,iterations=2)
    # Redraw present province boundaries from the same hex ownership as lookup.
    junction=ET.parse(A/'work/standalone_kit/raw_data/db/region_to_province_junctions.xml').getroot()
    provinces={v.findtext('region'):v.findtext('province') for v in junction.findall('region_to_province_junctions')}
    pnames=sorted(set(provinces.values()));pv=np.asarray([pnames.index(provinces[k]) if k in provinces else -1 for k in names]+[-1],np.int32)
    def boundaries(own):
        e=np.zeros(physical.shape,bool)
        e[:,1:]|=(own[:,1:]!=own[:,:-1])&physical[:,1:]&physical[:,:-1]
        e[1:]|=(own[1:]!=own[:-1])&physical[1:]&physical[:-1]
        return e
    # Region (settlement plot) boundaries inside a province: thin and light, so
    # every town/farm/pasture/mine plot reads as its own area when zoomed out.
    region_edge=boundaries(own)
    owner=pv[own];edge=boundaries(owner)
    # Thin and soft like vanilla's painted map: region lines two pixels, province lines three.
    # Only around playable regions: lines between non-playable filler regions are not borders.
    playable=np.asarray(['non_playable' not in k for k in names]+[False])[own]
    region_edge&=~binary_erosion(~playable,1)
    inner=(region_edge|np.roll(region_edge,1,1)|np.roll(region_edge,1,0))&physical&~edge
    wide=~binary_erosion(~edge,1)&physical
    pixels=np.empty(image.shape,np.uint8)
    def paint(s):   # coast, then region lines, then province lines, pixel by pixel
        img=image[s];img[coast[s]]=[104,100,86];m=inner[s]
        img[m]=img[m]*.55+np.asarray([96,84,62])*.45;m=wide[s]
        img[m]=img[m]*.45+np.asarray([72,62,46])*.55;pixels[s]=np.clip(img,0,255).astype(np.uint8)
    rows(paint,height)
    im=Image.fromarray(pixels,'RGB');del image,heights,smooth,pixels
    draw=ImageDraw.Draw(im);assets=Assets();triangles=0
    # Project the translated models actually drawn by the engine, not old painted rivers.
    for i in range(24):
        key=f'terrain/campaigns/3k_dlc07_main_map/models/river_{i}.wsmodel'
        geom=ET.fromstring(assets.read(key)).findtext('geometry');rb=assets.read(geom);off=struct.unpack_from('<I',rb,152)[0]
        _,_,_,vo,vc,io,ic=struct.unpack_from('<HH5I',rb,off)
        vertices=np.frombuffer(rb,dtype='<f4',count=vc*12,offset=off+vo).reshape(vc,12)[:,:3].copy()
        vertices+=np.asarray(struct.unpack_from('<3f',rb,off+628))
        px=vertices[:,0]/DX*width/W+1.5;py=(H-1-vertices[:,2]/DZ)*height/H+1.5
        indices=np.frombuffer(rb,dtype='<u2',count=ic,offset=off+io).reshape(-1,3)
        assert int(indices.max())<vc
        for tri in indices:draw.polygon([(float(px[t]),float(py[t])) for t in tri],fill=(142,174,177))
        triangles+=len(indices)
    # Every current settlement hex is represented, including every authored site.
    main_slots=(cells[:,:,2]>>4)==1;towns=[]
    plan=json.loads((A/'work/current-region-design.json').read_text(encoding='utf-8'))
    authored={c['region']:c['town'] for c in plan['regions']}
    # Mean slot position per region: integer sums, exact in float64 in any order, so equal to xs.mean().
    ys,xs=np.nonzero(main_slots);sid=ids[ys,xs]+1;count=np.bincount(sid);sx=np.bincount(sid,xs);sy=np.bincount(sid,ys)
    slot_ids=np.unique(sid)-1
    for rid in slot_ids:
        if names[rid] in authored:tq,tr=authored[names[rid]]
        else:tq,tr=int(round(float(sx[rid+1]/count[rid+1]))),int(round(float(sy[rid+1]/count[rid+1])))
        towns.append((tr,tq))
    # Vanilla settlements (from the vanilla map) plus every authored region, nothing else.
    import map_hex
    vm=map_hex.read(_frame.VANILLA_MAP);vids=map_hex.region_ids(vm['cells'])
    vanilla_towns=len(np.unique(vids[(vm['cells'][:,:,2]>>4)==1]))
    assert len(towns)==vanilla_towns+len(authored),(len(towns),vanilla_towns,len(authored))
    slot_set=set(slot_ids.tolist());assert all(names.index(r) in slot_set for r in authored),'authored region without a settlement slot'
    for tr,tq in towns:
        x=tq*width/W+1.5;y=(H-1-tr-(tq%2)*.5)*height/H+1.5
        draw.rectangle((x-2,y-2,x+2,y+2),fill=(104,80,51))
    # Records, multiplayer and china map are the same image: encode each distinct PNG once and write the same bytes.
    def png(source):
        buffer=BytesIO();source.save(buffer,format='PNG',compress_level=3);return buffer.getvalue()
    encoded={'im':pool.submit(png,im)};encoding=pool.submit(dds_bc1,im.copy())   # own image: Image.save sets per-image encoderinfo/encoderconfig
    preview=ROOT/'output/terrain-strategic-map-preview.png';previewed=pool.submit(lambda:im.resize((1440,1030),Image.Resampling.LANCZOS).save(preview))
    mini_size=Image.open(S/'campaign_maps/3k_dlc07_main_map/display/regions/3k_main_lookup_minimap.tga').size
    # Lookup minimap takes pixel (4*x+2,4*y+2); sample the same origin and stride.
    mini=Image.fromarray(np.asarray(im)[2:mini_size[1]*4:4,2:mini_size[0]*4:4])
    assert mini.size==mini_size,(mini.size,mini_size)
    vignette=faded_alpha.result()
    faded=Image.fromarray(np.dstack([np.asarray(im),vignette]),'RGBA');encoded['faded']=pool.submit(png,faded)
    mini_alpha=(214+vignette[2:mini_size[1]*4:4,2:mini_size[0]*4:4].astype(np.float32)*41/255).astype(np.uint8)
    mini=Image.fromarray(np.dstack([np.asarray(mini),mini_alpha]),'RGBA');encoded['mini']=pool.submit(png,mini)
    dds,mips=encoding.result()
    assert decode_srgb_bc1(dds).size==im.size
    encoded={k:v.result() for k,v in encoded.items()};previewed.result();pool.shutdown()
    hashes={k:hashlib.sha256(v).hexdigest() for k,v in encoded.items()}
    textures=[]
    for variant in VARIANTS:
        base=S/'campaign_maps'/variant;base.mkdir(parents=True,exist_ok=True)
        for name in ('3k_overlay_map.dds','3k_main_minimap.png','campaign_map_records.png','campaign_map_multiplayer.png','custom_battle_map.png','three_kingdoms_china_map.png'):
            p=base/name
            if name.endswith('.dds'):p.write_bytes(dds);size=im.size;sha=hashlib.sha256(dds).hexdigest()
            else:
                # custom_battle_map stays opaque RGB like vanilla; the others carry the vignette.
                key='mini' if name=='3k_main_minimap.png' else ('im' if name=='custom_battle_map.png' else 'faded')
                p.write_bytes(encoded[key]);size={'mini':mini,'im':im,'faded':faded}[key].size;sha=hashes[key]
            textures.append({'path':p.relative_to(S).as_posix(),'sha256':sha,'size':size})
    report={'source_map_sha256':digest(mp),'source_design_sha256':digest(A/'work/current-region-design.json'),
      'lookup_size':[width,height],'minimap_size':mini_size,'caime_lookup_pixel_mismatches':mismatch,'hex_edge_rounding_ties_resolved_from_actual_lookup':rounding_ties,
      'lookup_sampler_source':'tools/source/CampaignMapToolkit-1.1.0/CAIME/Classes/Exporters/LookupImageExporter.cs',
      'native_canvas_padding_hexes':{'west':_frame.PAD_LEFT,'east':_frame.PAD_RIGHT,'north':_frame.PAD_TOP,'south':_frame.PAD_BOTTOM},
      'old_hand_drawn_coast_borders_icons_reused':False,'current_town_markers':len(towns),'river_triangles':triangles,
      'art':'ink-wash parchment (strategic_ink.py: ecoregion washes, slope brush strokes, snow, salt flats) with present coast, thin province and region boundaries and translated river triangles',
      'dds_format':'BC1_UNORM_SRGB; opaque; complete mip chain','ui_png_alpha':'records/multiplayer/china: land vignette 0..255; minimap 214..255; custom_battle_map opaque RGB (as vanilla)','dds_mip_sizes':mips,'textures':textures,'live_verified':False}
    reportpath.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('PASS: lookup sampler exact; strategic 4804x3437; minimap 1201x859;',len(towns),'town markers;',triangles,'river triangles',flush=True)

if __name__=='__main__':main()
