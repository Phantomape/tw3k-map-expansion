"""Reject deeply sunk native-footprint towns and polish cuts round towns; compare introduced slopes locally.

Revision 2026-11-01 (coordinator): no exemptions, every town is held to the rule (2 height units). C2's Jiuyuan passed
an exemption at 3.15 below the local native median, yet rendered as a three-sided pit (polish cut the slope down to
the Yellow River bank, up to 4.49). So a second rule, on the cut itself: within CUT_RING tile cells (4 hexes) of a
town, polished minus pre-polish height. Frozen on C2's 89 native-footprint towns: Jiuyuan 190 cells lowered by more
than 2 and 4.49 at most; every other town at most 103 cells (Jincheng west, a gentle bowl in its render) and 3.05.
Frozen 2026-11-01 (coordinator) on release run C2: pack sha256 960e4de9..., terrain run 20261101-191428, main 2b26cdb5.
Not to be changed afterwards.
Revision 2026-11-01 (c21, coordinator approved; thresholds unchanged): "pre-polish" for the cut is the pre-polish target
ground, hehuang_terrain.target_ground: the native LF with the real-geography patches blended in by their height weight
(the same Patch.blend polish lays). A patch (wuyuan: the Hetao north bank under Jiuyuan) lowering vanilla filler to the
real ground is not a cut; what polish cuts below that ground is. Outside every patch's height weight the target is the
native LF, so those towns' values are as before."""
CUT_FROZEN='2026-11-01, release run C2 (pack 960e4de92925, run 20261101-191428, main 2b26cdb5)'
import json,struct
import numpy as np
from PIL import Image
from analyze_reference import ROOT
import campaign_world as _frame
from campaign_world import DX,DZ
A=_frame.WORKSPACE;S=A/'work/release-input';BASE=A/'terrain-before-polish'
GAIN,LEVEL,NORM=_frame.GAIN,_frame.LEVEL,_frame.NORM
CUT_RING=8;CUT_DEPTH=2.0;CUT_CELLS_MAX=120;CUT_MAX=3.5

def town_cut(current,native,q,r):
    """Cells within CUT_RING tile cells of the town lowered by polish more than CUT_DEPTH, and the deepest lowering."""
    cx,cy=int(q*2),int((r+q%2*.5)*2)
    w=(current-native)[max(0,cy-CUT_RING):cy+CUT_RING+1,max(0,cx-CUT_RING):cx+CUT_RING+1]
    return int((w<-CUT_DEPTH).sum()),round(float(max(0.0,-w.min())),3)

def field(path):
    data=path.read_bytes();h,w=struct.unpack_from('<2I',data,12)
    values=np.frombuffer(data,'<u2',offset=128,count=w*h).reshape(h,w)[::-1]
    return np.asarray(Image.fromarray(values.astype(np.float32)).resize((_frame.TILE_W,_frame.TILE_H),Image.Resampling.BILINEAR))/65535*GAIN+LEVEL-NORM*GAIN

def slopes(surface):
    gz,gx=np.gradient(surface,DZ/2,DX/2)
    return np.hypot(gx,gz)

def generic_cache_mask(path):
    data=path.read_bytes();count=_frame.TILE_W*_frame.TILE_H;tail=data[count*60:];pos=4;paths=[]
    for _ in range(struct.unpack_from('<I',tail)[0]):
        size=struct.unpack_from('<H',tail,pos)[0];pos+=2;paths.append(tail[pos:pos+size].decode());pos+=size
    dtype=np.dtype([('path','<i4'),('rest','V16')]);layer=np.frombuffer(data,dtype=dtype,count=count)['path'].copy()
    children=layer<0;parents=-layer[children]-1;assert np.all((parents>=0)&(parents<count));layer[children]=layer[parents]
    generic=[i for i,name in enumerate(paths) if '\\generic\\' in name]
    return np.isin(layer,generic).reshape(_frame.TILE_H,_frame.TILE_W)

def main():
    report=json.loads((A/'analysis/geographic-terrain-polish.json').read_text(encoding='utf-8'))
    variants=[]
    for entry in report['variants']:
        variant=entry['variant'];pre=f'terrain/campaigns/{variant}/lf_height_map.dds'
        current=field(S/pre);native=field(BASE/pre);bad=[];rows=[];cut=[]
        import hehuang_terrain;target=hehuang_terrain.target_ground(native,hehuang_terrain.load())   # c21: cut below the target ground
        for town in entry['towns']:
            q,r=town['town']
            if not(_frame.PAD_LEFT<=q<_frame.PAD_LEFT+_frame.NATIVE_W and _frame.PAD_BOTTOM<=r<_frame.PAD_BOTTOM+_frame.NATIVE_H):continue
            reference=town['local_reference_height'];depth=reference-town['plateau_height']
            row={'region':town['region'],'town':[q,r],'reference':reference,'plateau':town['plateau_height'],'depth_below_local_reference':depth}
            row['cut_cells_over_2'],row['max_cut']=town_cut(current,target,q,r)
            rows.append(row)
            if depth>2:bad.append(row)
            if row['cut_cells_over_2']>CUT_CELLS_MAX or row['max_cut']>CUT_MAX:cut.append(row)
        # The actual cliff_a window from Claude, sampled at half-hex resolution.
        (c0,r0),(c1,r1)=_frame.from_authored_tile(680,910),_frame.from_authored_tile(841,1061)
        region=(slice(r0,r1),slice(c0,c1));before=slopes(native)[region];after=slopes(current)[region]
        generic=generic_cache_mask(S/f'terrain/campaigns/{variant}/tile_map.tiles')[region]
        variants.append({'variant':variant,'native_footprint_towns':len(rows),'towns_sunk_more_than_2':bad,
            'maximum_sinking':max(t['depth_below_local_reference'] for t in rows),'towns_cut':cut,
            'maximum_cut':max(t['max_cut'] for t in rows),'maximum_cut_cells':max(t['cut_cells_over_2'] for t in rows),
            'cliff_a':{'native_slope_over_1_5':int((before>1.5).sum()),'current_slope_over_1_5':int((after>1.5).sum()),
                'native_slope_over_0_8':int((before>.8).sum()),'current_slope_over_0_8':int((after>.8).sum()),
                'current_generic_slope_over_1_5':int(((after>1.5)&generic).sum()),
                'generic_introduced_slope_over_1_5':int(((after>1.5)&(before<=1.5)&generic).sum()),
                'native_mean_height':float(native[region].mean()),'current_mean_height':float(current[region].mean())},'towns':rows})
    result={'criterion':'native-footprint plateau may not be more than 2 height units below local immutable native median',
        'cut_criterion':f'within {CUT_RING} tile cells of a town: at most {CUT_CELLS_MAX} cells lowered by polish more than {CUT_DEPTH}, none more than {CUT_MAX}; frozen {CUT_FROZEN}',
        'variants':variants,'passed':all(not v['towns_sunk_more_than_2'] and not v['towns_cut'] for v in variants),'live_verified':False}
    (A/'analysis/local-town-elevation-audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    for v in variants:print(v['variant'],'sunk:',len(v['towns_sunk_more_than_2']),'maximum:',v['maximum_sinking'],'cut:',len(v['towns_cut']),'maximum cut:',v['maximum_cut'],v['maximum_cut_cells'],'cliff_a:',v['cliff_a'],flush=True)
    assert result['passed'], 'Some new native-footprint towns still form deep artificial pits'

if __name__=='__main__':main()
