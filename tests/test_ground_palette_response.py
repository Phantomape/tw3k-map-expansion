"""Ground-only conditional palette contracts; no assets, Q calibration or GPU."""
from concurrent.futures import ThreadPoolExecutor
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import ground_bake as gb
import ground_materials as gm
import biome_profiles as bp
import test_ground_candidates_integration as integration


class GroundPalette(unittest.TestCase):
    def setUp(self):
        self.shape=(2,4)
        self.D=np.array([[0.,.5,1.,.5],[.25,.5,.75,1.]],np.float32)
        self.keep=np.zeros(self.shape,bool);self.keep[0,3]=True
        self.picked=np.zeros(self.shape,np.uint8)
        self.tones=np.zeros((32,3),np.float32)
        self.tones[4,0]=30;self.tones[5,0]=20
        self.tones[24,0]=31;self.tones[25,0]=40;self.tones[26,0]=40
        self.palette={0:(26,25,24)}
        self.banned=np.zeros((32,)+self.shape,bool)
        self.banned[24]=True
        self.banned[26,1,0]=True

    def forbidden(self,slot):
        mat=np.broadcast_to(np.asarray(slot,np.uint8),self.shape)
        y,x=np.indices(self.shape)
        return self.banned[mat,y,x]

    def apply(self,candidates,**overrides):
        arguments=dict(enabled=True,sources=(3,5),wood_slots=(4,5))
        arguments.update(overrides)
        return gb.density_ground_palette(candidates,self.D,self.picked,self.keep,
                self.forbidden,self.tones,self.palette,**arguments)

    def masses(self,candidates):
        masses={}
        for c in candidates:
            band=np.broadcast_to(np.asarray(c.band),self.shape)
            for g in np.unique(band):
                key=(c.source,int(g))
                masses.setdefault(key,np.zeros(self.shape,np.float64))
                masses[key]+=np.where(band==g,np.broadcast_to(c.weight,self.shape),0)
        return masses

    def test_off_is_original_set_and_exact_selector_result(self):
        candidates=[gb.Candidate(3,4,np.full(self.shape,.4,np.float32),3),
                    gb.Candidate(4,26,np.full(self.shape,.6,np.float32),3)]
        off=self.apply(candidates,enabled=False)
        self.assertIs(off,candidates)
        preserved=np.full(self.shape+(2,),2,np.uint8)
        draw=lambda purpose,slot,mask:np.full(int(mask.sum()),.37,np.float32)
        a=gb.select(candidates,preserved,self.keep,self.forbidden,self.tones,np.ones(4)*30,draw)
        b=gb.select(off,preserved,self.keep,self.forbidden,self.tones,np.ones(4)*30,draw)
        np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1])

    def test_D_endpoints_legal_nearest_tie_G_and_authored_exclusions(self):
        w=np.full(self.shape,.4,np.float32);w.flags.writeable=False
        sources=(1,2,3,4,5,6,7,8,9,10,11,12)
        candidates=[gb.Candidate(source,4,w,np.array([[0,1,2,3],[3,2,1,0]],np.uint8)) for source in sources]
        result=self.apply(candidates)
        for i,source in enumerate(sources):
            if source in (3,5):
                np.testing.assert_array_equal(result[i].weight[~self.keep],(w*self.D)[~self.keep])
                self.assertEqual(result[i].weight[0,3],w[0,3])
            else:
                self.assertIs(result[i],candidates[i])
        residuals=result[len(candidates):]
        self.assertEqual({c.source for c in residuals},{3,5})
        # Slot24 is nearest but illegal; exact tie goes to original first26.
        for source in (3,5):
            by_slot={c.slot:c.weight for c in residuals if c.source==source}
            self.assertGreater(by_slot[26][0,0],0)
            self.assertGreater(by_slot[25][1,0],0)
            self.assertEqual(sum(c.weight[0,2] for c in residuals if c.source==source),0)
        before,after=self.masses(candidates),self.masses(result)
        for key in before:np.testing.assert_allclose(before[key],after[key],atol=2e-7,rtol=0)
        np.testing.assert_array_equal(w,np.full(self.shape,.4,np.float32))

    def test_no_legal_open_and_unknown_climate_keep_original_wood(self):
        self.banned[25:27]=True
        candidates=[gb.Candidate(5,5,np.full(self.shape,.3,np.float32),2)]
        result=self.apply(candidates)
        self.assertEqual(len(result),1)
        np.testing.assert_array_equal(result[0].weight,candidates[0].weight)
        self.banned[:]=False;self.picked[:]=15
        result=self.apply(candidates)
        np.testing.assert_array_equal(result[0].weight,candidates[0].weight)

    def test_illegal_original_wood_is_not_rescued_as_new_open(self):
        self.banned[4]=True
        candidates=[gb.Candidate(3,4,np.full(self.shape,.4,np.float32),3)]
        self.assertIs(self.apply(candidates)[0],candidates[0])

    def model(self,**kwargs):
        fixture=integration.GroundIntegration()
        b=fixture.model(**kwargs)
        b.dw['ground_palette_response']['enabled']=True
        return fixture,b

    def test_actual_rows_hook_is_after_farm_reservations_before_single_select(self):
        fixture,b=self.model(F=.8,M=.4,forest=.5,town=integration.Patches(8,[4],.2))
        b.authored_forest=np.full((2,2),.6,np.float32)
        original=gb.density_ground_palette
        with patch.object(gb,'density_ground_palette',wraps=original) as hook:
            on=fixture.run_rows(b,raw=24,picked_climate=0,capture=True)
        self.assertEqual(hook.call_count,1)
        before=hook.call_args.args[0]
        self.assertTrue(hook.call_args.kwargs['enabled'])
        b.dw['ground_palette_response']['enabled']=False
        off=fixture.run_rows(b,raw=24,picked_climate=0,capture=True)
        # The sole selector sees the completed palette; source6/7 weights do not
        # acquire new support from the brighter forest-floor material.
        for source in (6,7,8,12):
            expected=[c for c in before if c.source==source]
            actual=[c for c in on if c[0]==source]
            self.assertEqual(len(expected),len(actual))
            self.assertTrue(any(np.any(c.weight>0) for c in expected))
            for a,c in zip(expected,actual):
                np.testing.assert_array_equal(a.slot,c[1])
                np.testing.assert_array_equal(a.weight,c[2])
                np.testing.assert_array_equal(a.band,c[3])
            for a,c in zip([v for v in off if v[0]==source],actual):np.testing.assert_array_equal(a[2],c[2])
        # town=.2, F=.8, M=.4, af=.6*.5 (wood and raw24 are both
        # existing FARM_SOURCE). Palette receives the
        # forest .5 share AFTER natural mass and the authored-forest deduction.
        wood_before=sum((np.broadcast_to(c.weight,(8,8)) for c in before if c.source==3),np.zeros((8,8)))
        np.testing.assert_allclose(wood_before,.5*.8*.2*.6*.7,atol=2e-7,rtol=0)
        wood_after=sum((v[2] for v in on if v[0]==3 and np.all(v[1]==4)),np.zeros((8,8)))
        np.testing.assert_allclose(wood_after,wood_before*.5,atol=2e-7,rtol=0)
        self.assertTrue(any(source==3 and np.all(slot==26) and weight.any() for source,slot,weight,_ in on))

    def test_positive_geological_wood_in_actual_rows_uses_same_source5(self):
        fixture,b=self.model(W=1,forest=.5)
        b.fields['P'][:]=0;b.fields['P'][gm.MATERIALS.index(integration.GEO_WOOD)]=1
        on=fixture.run_rows(b,raw=0,picked_climate=0,capture=True)
        b.dw['ground_palette_response']['enabled']=False
        off=fixture.run_rows(b,raw=0,picked_climate=0,capture=True)
        np.testing.assert_allclose(fixture.mass(on,5),fixture.mass(off,5),atol=2e-7,rtol=0)
        self.assertTrue(any(source==5 and np.all(slot==26) and weight.any() for source,slot,weight,_ in on))
        wood=sum((weight for source,slot,weight,_ in on if source==5 and np.all(slot==integration.GEO_WOOD)),np.zeros((8,8)))
        np.testing.assert_allclose(wood,.125,atol=2e-7,rtol=0)

    def test_changed_actual_rows_serial_threads_and_halo_blocks_match(self):
        fixture,b=self.model(forest=.6)
        b.forest=np.array([[.1,.9],[.4,.6]],np.float32)
        rgba=np.empty((8,8,2),np.uint8);rgba[...,0]=24;rgba[...,1]=3
        share=np.zeros((8,8),np.float32);picked=np.zeros((8,8),np.uint8)
        ban=lambda fields,tx,ty:(np.zeros((len(ty),len(tx)),np.uint8),None)
        def rows(bounds):
            a,z=bounds
            return b.rows(a,z,rgba[a:z],rgba[a:z],share[a:z],picked[a:z],rgba[a:z])
        with patch.object(gm,'_ban_rows_fields',ban),patch.object(gm,'_banned',lambda lb,al,slot:np.zeros(np.broadcast_shapes(lb.shape,np.shape(slot)),bool)):
            whole=rows((0,8))
            b.dw['ground_palette_response']['enabled']=False
            off=rows((0,8))
            self.assertTrue(np.any(whole[0]!=off[0]))
            b.dw['ground_palette_response']['enabled']=True
            blocks=[(0,6),(2,8)]
            serial=[rows(v) for v in blocks]
            with ThreadPoolExecutor(2) as pool:parallel=list(pool.map(rows,blocks))
        for a,z in zip(serial,parallel):
            np.testing.assert_array_equal(a[0],z[0]);np.testing.assert_array_equal(a[1],z[1])
        for whole_array,parts in ((whole[0],[v[0] for v in serial]),(whole[1],[v[1] for v in serial])):
            joined=np.concatenate((parts[0][:4],parts[1][2:]),axis=0)
            np.testing.assert_array_equal(whole_array,joined)

    def test_profile_enables_display_only_generated_sources(self):
        rule=bp.CONSTS['ground_materials']['DONOR_WOODLAND']['ground_palette_response']
        self.assertEqual(rule,{'enabled':True,'sources':(3,5)})


if __name__=='__main__':unittest.main()
