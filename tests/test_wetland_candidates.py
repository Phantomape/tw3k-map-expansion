"""Small arrays only: wetland source candidates, frames and legacy compatibility."""
import ast
import atexit
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
# campaign_world needs a design at import; no project stage or game assets are read.
_tmp = tempfile.TemporaryDirectory(prefix='tw3k-wetland-tests-')
atexit.register(_tmp.cleanup)
_design = Path(_tmp.name) / 'design.json'
_design.write_text(json.dumps({'grid': [2048, 2047], 'padding': [888, 375, 268, 970],
                               'old_grid_offset': [888, 375], 'tile_aligned_frame': True}), encoding='utf-8')
os.environ['TW3K_WORKSPACE'] = _tmp.name
os.environ['TW3K_DESIGN'] = str(_design)
import apply_climate as ac


def legacy_three_read_window(self, box):
    """Pre-optimization window body; production helpers and arithmetic unchanged."""
    from scipy.ndimage import distance_transform_edt
    y0,y1,x0,x1=map(int,box);h,w=self.shape;box=(y0,y1,x0,x1)
    valid=self._valid_pixels(box);bands,_=self._hex_pixels(self._band,box)
    weights=np.zeros(valid.shape,np.float32)
    if not valid.size or not valid.any():return weights,bands.astype(np.uint8),valid
    halo=max(self._radius,int(np.ceil(self._fade))+1)
    outer=(max(0,y0-halo),min(h,y1+halo),max(0,x0-halo),min(w,x1+halo))
    a,b,c,d=outer;crop=(slice(y0-a,y1-a),slice(x0-c,x1-c))
    wet,wet_valid=self._hex_pixels(self._wet,outer)
    if not(wet&wet_valid).any():return weights,bands.astype(np.uint8),valid
    support=(ac._box_blur((wet&wet_valid).astype(np.float32),self._radius)[crop]-ac.WET_SOFT[1])/(1-ac.WET_SOFT[1])
    support=np.clip(support,0,1);support=support*support*(3-2*support)
    eligible=self._valid_pixels(outer)
    inside=distance_transform_edt(np.pad(eligible,1,constant_values=False))[1:-1,1:-1][crop]
    inside=np.clip((inside-.5)/self._fade,0,1).astype(np.float32);inside=inside*inside*(3-2*inside)
    proximity=np.interp(self._water_distance(box)/8.,*ac.WET_WATER).astype(np.float32)
    xx,yy=np.meshgrid(np.arange(x0,x1)/8.,(h-1-np.arange(y0,y1))/8.)
    noise=self._noise_scale*ac.WET_NOISE[2]*ac.fbm(xx,yy,(0,0),ac.WET_NOISE[0],ac.WET_NOISE[1],seed=311)
    field=inside*(support*proximity+noise)
    t=np.clip((field-ac.WET_NOISE[3])/self._threshold_width+.5,0,1)
    weights=(t*t*(3-2*t)).astype(np.float32);weights[~valid|(support<=0)]=0
    return weights,bands.astype(np.uint8),valid


class WetlandTests(unittest.TestCase):
    def fixture(self, patterned=False):
        H, W = 24, 32
        cells = np.zeros((H, W, 16), np.uint8)
        ac.set_ground(cells, np.ones((H, W), bool), 13)
        ac.set_climate(cells, np.ones((H, W), bool), 3)
        types = [f'other_{i}' for i in range(16)]
        types[0] = next(n for n in ac.WETLAND_GROUND_FROM if n != 'wetland')
        types[13] = 'wetland'
        domain = np.ones((H, W), bool)
        protected = np.zeros((H*2, W*2), bool)
        shape = (H*8, W*8)
        distance = np.ones(shape, np.float32)
        if patterned:
            mask = np.zeros((H, W), bool); mask[8:12, 8:12] = True
            ac.set_ground(cells, mask, 0)
            cells[10, 15, 2] = 0x10
            ac.set_ground(cells, np.indices((H,W))[1] == 20, 2)
            domain[3:7, 4:7] = False
            protected[20:24, 30:35] = True
            yy, xx = np.indices(shape)
            water = xx < 15 + yy//12
            distance = distance_transform_edt(~water).astype(np.float32)
        return cells, types, domain, protected, distance

    def model(self, fixture=None, **kw):
        cells, types, domain, protected, distance = fixture or self.fixture()
        water = kw.pop('water_field', {'shape': distance.shape, 'distance_px': distance, 'source': 'synthetic test'})
        return ac.WetlandCandidates.from_inputs(cells, [], protected, False, water_field=water,
                                                domain=domain, ground_types=types, **kw)

    def test_requires_explicit_authoritative_water_and_frame(self):
        cells, types, domain, prot, dist = self.fixture()
        with self.assertRaisesRegex(ValueError, 'authoritative'):
            ac.WetlandCandidates.from_inputs(cells, [], prot, False, domain=domain, ground_types=types)
        with self.assertRaisesRegex(ValueError, 'frame'):
            self.model(water_field={'shape': (100,100), 'distance_px': np.ones((100,100))})

    def test_no_blend_access_input_immutability_and_adapter(self):
        f=self.fixture(); before=[x.copy() if isinstance(x,np.ndarray) else list(x) for x in f]
        m=self.model(f); out=ac.wetland_candidate_rows({'wetland_candidates':m},(64,96,32,96))
        self.assertEqual(out['slot'],31)
        self.assertEqual(out['weight'].shape,(32,64))
        self.assertEqual(out['weight'].dtype,np.float32)
        self.assertEqual(out['band'].dtype,np.uint8)
        self.assertEqual(out['hard_valid'].dtype,np.bool_)
        self.assertTrue(np.isfinite(out['weight']).all())
        self.assertTrue(((out['weight']>=0)&(out['weight']<=1)).all())
        for a,b in zip(f,before):np.testing.assert_array_equal(a,b)

    def test_hard_exclusions_and_inside_fade(self):
        f=self.fixture(); cells,types,domain,prot,dist=f
        domain[:, :4]=False
        cells[10,10,2]=0x10
        ac.set_ground(cells,np.indices(cells.shape[:2])[1]==18,2)
        prot[14:18,24:28]=True
        dist[40:48,70:80]=0  # physical water, even though logical ground is land
        m=self.model(f,noise_scale=0)
        weights,bands,valid=m.rows(0,m.shape[0],m.shape[1])
        self.assertFalse(valid[:, :32].any())
        self.assertFalse(valid[40:48,70:80].any())
        self.assertTrue((weights[~valid]==0).all())
        # A hard edge's interior weight enters the field before thresholding.
        self.assertTrue(valid[80,32]);self.assertEqual(weights[80,32],0)
        self.assertGreater(weights[80,48],weights[80,35])
        yy,xx=np.indices(valid.shape);q,r=ac.cw.tile_hex(xx//4,(m.shape[0]-1-yy)//4)
        ok=(r>=0)&(r<cells.shape[0])
        town=ac.town_clear_hex(cells)
        on_town=town[np.clip(r,0,cells.shape[0]-1),q]&ok
        self.assertFalse(valid[on_town].any())
        for ty,tx in zip(*np.nonzero(prot)):
            sl=(slice(m.shape[0]-(ty+1)*4,m.shape[0]-ty*4),slice(tx*4,(tx+1)*4))
            self.assertFalse(valid[sl].any())

    def test_mask_distance_reader_is_continuous_and_bounded(self):
        f=self.fixture();shape=f[-1].shape
        water=np.zeros(shape,bool);water[80,80]=True
        requests=[]
        def read(box):
            requests.append(box);y0,y1,x0,x1=box;return water[y0:y1,x0:x1]
        m=self.model(f,water_field={'shape':shape,'mask':read})
        a=m._water_distance((80,85,80,85))
        self.assertEqual(a[0,0],0)
        self.assertEqual(a[4,3],5)  # Euclidean3-4-5, not hex rings
        self.assertAlmostEqual(float(a[1,1]),np.sqrt(2),places=6)
        self.assertLess(max(y1-y0 for y0,y1,x0,x1 in requests),shape[0])
        direct=distance_transform_edt(~water).astype(np.float32)
        n=self.model(f,water_field={'shape':shape,'distance_px':direct})
        for left,right in zip(m.window((60,105,60,105)),n.window((60,105,60,105))):np.testing.assert_array_equal(left,right)
        far=self.model(f,water_field={'shape':shape,'mask':np.zeros(shape,bool)})
        self.assertTrue((far._water_distance((70,90,70,90))==40).all())

    def test_rows_x_partition_and_order_are_bit_identical(self):
        m=self.model(self.fixture(patterned=True));H,W=m.shape
        full=m.rows(0,H,W)
        chunks=[(0,17),(17,48),(48,83),(83,129),(129,H)]
        for order in (chunks,chunks[::-1]):
            out=[np.empty_like(v) for v in full]
            for y0,y1 in order:
                for dst,value in zip(out,m.rows(y0,y1,W)):dst[y0:y1]=value
            for a,b in zip(out,full):np.testing.assert_array_equal(a,b)
        for a,b in ((0,35),(35,99),(99,170),(170,W)):
            for crop,value in zip(m.window((0,H,a,b)),full):np.testing.assert_array_equal(crop,value[:,a:b])
        for v in m.rows(5,5,W):self.assertEqual(v.shape,(0,W))

    def test_bands_follow_hex_climate_and_do_not_sample_material(self):
        f=self.fixture();cells=f[0]
        for q,cl in ((5,1),(6,4),(7,3)):
            mask=np.zeros(cells.shape[:2],bool);mask[:,q]=True;ac.set_climate(cells,mask,cl)
        m=self.model(f);weights,bands,valid=m.rows(64,96,m.shape[1])
        for q,band in ((5,ac.BLEND_BAND['cold']),(6,ac.BLEND_BAND['temperate']),(7,ac.BLEND_BAND['subtropical'])):
            self.assertTrue((bands[:,q*8:q*8+8]==band).all())

    def test_legacy_painter_source_unchanged_and_new_prepare_has_no_side_effect(self):
        source=(ROOT/'scripts/apply_climate.py').read_text(encoding='utf-8')
        node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='paint_wetland_blend')
        text='\n'.join(source.splitlines()[node.lineno-1:node.end_lineno])+'\n'
        self.assertEqual(hashlib.sha256(text.encode()).hexdigest(),'13946b72bdb2371a215aacb806b228687bb5289a344294b032473d5898569ae5')
        f=self.fixture();cells,types,domain,prot,dist=f
        def legacy(flag):
            cur=np.zeros((*dist.shape,2),np.uint8);cur[...,0]=4
            with patch.object(ac,'WET_PIXEL_MASK',flag),patch.object(ac,'_LGT',list(types)):
                count=ac.paint_wetland_blend(cur,cells,prot,[],filler=False)
            return cur,count
        for flag in (False,True):
            old,count=legacy(flag);m=self.model(f);m.rows(0,dist.shape[0],dist.shape[1]);new,n=legacy(flag)
            self.assertEqual(count,n);np.testing.assert_array_equal(old,new)

    def test_one_water_read_matches_three_read_path_for_random_windows(self):
        f=self.fixture(patterned=True);distance=f[-1];rng=np.random.default_rng(318)
        for mode in ('mask','distance_px'):
            data=distance==0 if mode=='mask' else distance
            calls=[]
            def reader(box):
                calls.append(box);a,b,c,d=box;return data[a:b,c:d]
            for fade in (2,8,55):  # also exercise eligibility halo wider than water EDT cap
                model=self.model(f,water_field={'shape':data.shape,mode:reader},eligible_fade_px=fade)
                for _ in range(12):
                    y0=int(rng.integers(0,data.shape[0]-1));x0=int(rng.integers(0,data.shape[1]-1))
                    box=(y0,min(data.shape[0],y0+int(rng.integers(1,70))),x0,min(data.shape[1],x0+int(rng.integers(1,80))))
                    calls.clear();old=legacy_three_read_window(model,box);old_count=len(calls)
                    calls.clear();new=model.window(box);new_count=len(calls)
                    for a,b in zip(old,new):np.testing.assert_array_equal(a,b)
                    self.assertLessEqual(new_count,1)
                    if old_count==3:self.assertEqual(new_count,1)
        # Entirely logical-ineligible windows need no physical-water read.
        f=self.fixture();f[2][:]=False;calls=[]
        model=self.model(f,water_field={'shape':f[-1].shape,'mask':lambda box:calls.append(box) or np.zeros((box[1]-box[0],box[3]-box[2]),bool)})
        weight,band,valid=model.window((30,70,30,70))
        self.assertEqual(len(calls),0);self.assertFalse(valid.any());self.assertFalse(weight.any())
        # No wet support still returns the exact established hard_valid mask.
        f=self.fixture();ac.set_ground(f[0],np.ones(f[0].shape[:2],bool),0);calls=[]
        model=self.model(f,water_field={'shape':f[-1].shape,'mask':lambda box:calls.append(box) or np.zeros((box[1]-box[0],box[3]-box[2]),bool)})
        model.window((30,70,30,70));self.assertEqual(len(calls),1)


if __name__=='__main__':unittest.main()
