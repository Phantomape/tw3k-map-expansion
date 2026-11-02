"""Tiny recipe/ownership integration tests; no assets, quantile grids or bake."""
import sys
import os
from pathlib import Path
import unittest
import threading
import copy
from unittest.mock import patch
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
# Reuse the small import-only campaign frame fixture, not any stage inputs.
import test_wetland_candidates
import ground_bake as gb
import ground_materials as gm
import biome_profiles as bp
# A woodland slot the landform mixes still carry (geological wood): r8 took cold_1 (4) out of the high / alpine mixes,
# tundra keeps cold_2 (5); TW3K_GROUND_RECIPE=v23 has 4 again. The tests take whichever is in MATERIALS.
GEO_WOOD = next(s for s in bp.WOODLAND_SLOTS if s in gm.MATERIALS)


class FixedDraw:
    def rows(self,y0,height):
        return lambda source,slot,mask: np.full(int(mask.sum()),.5,np.float32)
    final_rows=rows


class Patches:
    def __init__(self,source,slots,coverage):self.source,self.slots,self.coverage=source,slots,coverage
    def rows(self,y0,y1,width,band):
        return [gb.Candidate(self.source,m,np.full((y1-y0,width),self.coverage,np.float32),band) for m in self.slots]


class GroundIntegration(unittest.TestCase):
    def model(self,*,W=0,F=0,M=0,Psum=1,forest=0,takeover=1,farms=None,town=None):
        z=lambda value:np.full((2,2),value,np.float32)
        b=gb.GroundCandidates.__new__(gb.GroundCandidates)
        b.fields={k:z(v) for k,v in [('W',W),('Wd',0),('Wo',0),('cold',0),('F',F),('M',M)]}
        b.fields['P']=np.zeros((len(gm.MATERIALS),2,2),np.float32)
        b.fields['P'][gm.MATERIALS.index(7)]=Psum
        b.fields['K']=np.zeros((3,2,2),np.float32);b.fields['K'][0]=1
        b.domains={'native_keep':np.zeros((2,2),bool),'material_takeover':z(takeover)}
        b.forest=z(forest);b.height=b.width=8;b.gm=gm;b.draw=FixedDraw()
        b.wood=tuple(bp.WOODLAND_SLOTS);b.dw=copy.deepcopy(bp.CONSTS['ground_materials']['DONOR_WOODLAND'])
        # Existing tests isolate the pre-display ownership/recipe contracts.
        # The enabled post-reservation palette has separate integration tests.
        b.dw['ground_palette_response']['enabled']=False
        b.tones=np.zeros((32,3),np.float32);b.thresholds=np.ones(4,np.float32)
        b.farms,b.town,b.wetland=farms,town,None;b.town_clear=b.authored_forest=None
        b._wetland_lock=threading.Lock()
        return b

    def run_rows(self,b,raw=0,capture=False,banned_slots=(),picked_climate=2,reference_raw=None,ban_row=0):
        inherited=np.zeros((8,8,2),np.uint8);inherited[...,0]=raw;inherited[...,1]=2
        donor=inherited.copy();share=np.zeros((8,8),np.float32);picked=np.full((8,8),picked_climate,np.uint8)
        reference=inherited.copy()
        if reference_raw is not None:reference[...,0]=reference_raw
        ban=lambda fields,tx,ty:(np.full((len(ty),len(tx)),ban_row,np.uint8),None)
        result=[]
        def collect(candidates,preserved,keep,*args,**kw):
            result.extend((c.source,np.array(c.slot),np.broadcast_to(c.weight,(8,8)).copy(),np.array(c.band)) for c in candidates)
            return preserved.copy(),np.ones((8,8),np.uint8),{}
        with patch.object(gm,'_ban_rows_fields',ban),patch.object(gm,'_banned',lambda lb,al,mat:np.isin(mat,banned_slots)):
            if capture:
                with patch.object(gb,'select',collect):b.rows(0,8,inherited,donor,share,picked,reference)
                return result
            return b.rows(0,8,inherited,donor,share,picked,reference)

    def mass(self,candidates,source=None):
        return sum((c[2] for c in candidates if source is None or c[0]==source),np.zeros((8,8),np.float32))

    def test_geological_recipe_mass_survives_probability_normalization(self):
        c=self.run_rows(self.model(W=.5,Psum=.2),capture=True)
        np.testing.assert_allclose(self.mass(c,5),.5,atol=1e-7)
        np.testing.assert_allclose(self.mass(c),1,atol=1e-7)

    def test_authored_farm_does_not_invent_gm_farm_when_F_zero(self):
        c=self.run_rows(self.model(F=0,farms=Patches(9,[10],.5)),capture=True)
        np.testing.assert_allclose(self.mass(c,6),0,atol=1e-7)
        np.testing.assert_allclose(self.mass(c,9),.5,atol=1e-7)
        np.testing.assert_allclose(self.mass(c),1,atol=1e-7)

    def test_road_has_no_gm_farm_or_mud_support(self):
        c=self.run_rows(self.model(F=.8,M=.5),raw=13,capture=True)
        np.testing.assert_allclose(self.mass(c,6),0,atol=1e-7)
        np.testing.assert_allclose(self.mass(c,7),0,atol=1e-7)

    def test_overlap_material_alternatives_do_not_inflate_authored_coverage(self):
        c=self.run_rows(self.model(F=0,farms=Patches(9,[10,11],.5)),capture=True)
        # rows() reserves max(slot coverage)=.5; the conditional material choices
        # of that one source must sum to that same reserved coverage.
        np.testing.assert_allclose(self.mass(c,9),.5,atol=1e-7)
        np.testing.assert_allclose(self.mass(c),1,atol=1e-7)

    def test_same_R_different_G_is_enumeration_independent(self):
        keep=np.zeros((2,3),bool);pres=np.zeros((2,3,2),np.uint8)
        tones=np.zeros((32,3),np.float32);thresholds=np.ones(4,np.float32)
        a=[gb.Candidate(3,4,.5,0),gb.Candidate(3,4,.5,1)]
        draw=lambda source,slot,mask:np.full(int(mask.sum()),.5)
        first=gb.select(a,pres,keep,lambda slot:np.zeros(slot.shape,bool),tones,thresholds,draw,tone_radius=0)
        second=gb.select(a[::-1],pres,keep,lambda slot:np.zeros(slot.shape,bool),tones,thresholds,draw,tone_radius=0)
        np.testing.assert_array_equal(first[0],second[0]);np.testing.assert_array_equal(first[1],second[1])

    def test_zero_forest_density_cannot_create_woodland_at_full_takeover(self):
        result,owner,report=self.run_rows(self.model(forest=0,takeover=1),raw=4)
        self.assertFalse(np.isin(result[...,0],bp.WOODLAND_SLOTS).any())
        self.assertTrue((result[...,1]==2).all())

    def test_zero_takeover_retains_original_source_family_and_G(self):
        result,owner,report=self.run_rows(self.model(forest=0,takeover=0),raw=4)
        self.assertTrue((result[...,0]==4).all());self.assertTrue((result[...,1]==2).all())

    def test_keep_has_no_generated_owner_and_preserves_both_channels(self):
        b=self.model(F=1,M=1,forest=1);b.domains['native_keep'][:]=True
        result,owner,report=self.run_rows(b,raw=29)
        self.assertTrue((result[...,0]==29).all());self.assertTrue((result[...,1]==2).all())
        self.assertFalse(owner.any())

    def test_illegal_full_town_or_authored_farm_returns_mass_to_natural(self):
        for source,slot in ((8,4),(9,10)):
            kwargs={'town':Patches(source,[slot],1)} if source==8 else {'farms':Patches(source,[slot],1)}
            result,owner,_=self.run_rows(self.model(**kwargs),banned_slots=(slot,))
            self.assertTrue((result[...,0]==0).all())
            self.assertFalse((owner==source).any())

    def test_illegal_full_wetland_returns_mass_to_natural(self):
        class Wet:
            def rows(self,y0,y1,width):
                shape=(y1-y0,width)
                return np.ones(shape,np.float32),np.full(shape,3,np.uint8),np.ones(shape,bool)
        b=self.model();b.wetland=Wet()
        result,owner,_=self.run_rows(b,banned_slots=(31,))
        self.assertTrue((result[...,0]==0).all());self.assertFalse((owner==10).any())

    def test_illegal_full_authored_forest_returns_mass_to_natural(self):
        b=self.model(forest=1);b.authored_forest=np.ones((2,2),np.float32)
        result,owner,_=self.run_rows(b,banned_slots=(4,))
        self.assertFalse((result[...,0]==4).any());self.assertFalse((owner==12).any())

    def test_illegal_all_farm_recipe_or_mud_does_not_consume_coverage(self):
        result,owner,_=self.run_rows(self.model(F=1),banned_slots=(8,9))
        self.assertTrue((result[...,0]==0).all());self.assertFalse((owner==6).any())
        result,owner,_=self.run_rows(self.model(M=1),banned_slots=(14,))
        self.assertTrue((result[...,0]==0).all());self.assertFalse((owner==7).any())

    def test_full_density_k_one_banned_wood_returns_all_mass_to_local_recipe(self):
        b=self.model(forest=1,takeover=1)
        c=self.run_rows(b,capture=True,picked_climate=0,banned_slots=(4,5))
        np.testing.assert_allclose(self.mass(c),1,atol=1e-7)
        np.testing.assert_allclose(self.mass(c,5),1,atol=1e-7)
        result,owner,_=self.run_rows(b,picked_climate=0,banned_slots=(4,5))
        self.assertTrue((result[...,0]==7).all());self.assertTrue((owner==5).all())
        self.assertTrue((result[...,1]==2).all())

    def test_partial_banned_forest_probability_is_conserved(self):
        c=self.run_rows(self.model(forest=.5),capture=True,picked_climate=0,banned_slots=(4,))
        np.testing.assert_allclose(self.mass(c),1,atol=1e-7)
        np.testing.assert_allclose(self.mass(c,5),.5,atol=1e-7)
        for source,slot,weight,band in c:
            if np.all(slot==4):self.assertFalse(weight.any())

    def test_sparse_forest_does_not_invent_a_darker_edge_material(self):
        import ground_tone
        tones=ground_tone.tone_table()
        self.assertGreater(float(tones[4,0]),float(tones[5,0]))
        for density in (.1,.5,.9,1.):
            c=self.run_rows(self.model(forest=density),raw=24,capture=True,picked_climate=0)
            forest=[p for p in c if p[0]==3 and p[2].any()]
            self.assertTrue(all(np.all(p[1]==4) for p in forest))
            np.testing.assert_allclose(self.mass(c,3),density,atol=1e-7)

    def test_dense_actual_woodland_family_survives_then_thins_to_preferred_edge(self):
        dense=self.run_rows(self.model(forest=1),raw=5,capture=True,picked_climate=0)
        np.testing.assert_allclose(sum((p[2] for p in dense if p[0]==3 and np.all(p[1]==5)),np.zeros((8,8))),1)
        thin=self.run_rows(self.model(forest=.5),raw=5,capture=True,picked_climate=0)
        np.testing.assert_allclose(sum((p[2] for p in thin if p[0]==3 and np.all(p[1]==5)),np.zeros((8,8))),.25)
        np.testing.assert_allclose(sum((p[2] for p in thin if p[0]==3 and np.all(p[1]==4)),np.zeros((8,8))),.25)

    def test_zero_takeover_illegal_inherited_material_uses_local_recipe(self):
        c=self.run_rows(self.model(takeover=0),raw=4,capture=True,banned_slots=(4,5))
        np.testing.assert_allclose(self.mass(c),1,atol=1e-7)
        np.testing.assert_allclose(self.mass(c,5),1,atol=1e-7)
        result,owner,_=self.run_rows(self.model(takeover=0),raw=4,banned_slots=(4,5))
        self.assertTrue((result[...,0]==7).all());self.assertTrue((owner==5).all())
        self.assertTrue((result[...,1]==2).all())

    def test_reflected_31_in_clearance_uses_recipe_even_if_reference_is_dry(self):
        b=self.model(takeover=0);b.town_clear=np.ones((2,2),bool)
        c=self.run_rows(b,raw=31,reference_raw=7,capture=True)
        np.testing.assert_allclose(self.mass(c),1,atol=1e-7)
        np.testing.assert_allclose(self.mass(c,5),1,atol=1e-7)
        result,owner,_=self.run_rows(b,raw=31,reference_raw=7)
        self.assertTrue((result[...,0]==7).all());self.assertTrue((owner==5).all())
        self.assertTrue((result[...,1]==2).all())
        b.town_clear[:]=False
        result,owner,_=self.run_rows(b,raw=31,reference_raw=7)
        self.assertTrue((result[...,0]==31).all());self.assertTrue((owner==1).all())

    def test_rejected_geological_wood_open_split_uses_nonwood_local_recipe(self):
        b=self.model(W=1,forest=0)
        b.fields['P'][:]=0
        b.fields['P'][gm.MATERIALS.index(GEO_WOOD)]=.5
        b.fields['P'][gm.MATERIALS.index(7)]=.5
        c=self.run_rows(b,capture=True,picked_climate=0,banned_slots=(24,25,26))
        np.testing.assert_allclose(self.mass(c),1,atol=1e-7)
        np.testing.assert_allclose(self.mass(c,5),1,atol=1e-7)
        result,owner,_=self.run_rows(b,picked_climate=0,banned_slots=(24,25,26))
        self.assertTrue((result[...,0]==7).all());self.assertTrue((owner==5).all())
        self.assertTrue((result[...,1]==2).all())

    def test_missing_legal_nonwood_recipe_fails_without_slot_zero_default(self):
        b=self.model(W=1,forest=0);b.fields['P'][:]=0
        b.fields['P'][gm.MATERIALS.index(GEO_WOOD)]=1
        with self.assertRaisesRegex(ValueError,'No legal local nonforest landform/climate recipe'):
            self.run_rows(b,picked_climate=0,banned_slots=(24,25,26))

    def test_zero_takeover_clearance_31_without_landform_uses_existing_climate_open(self):
        # P=0/L=0 is a normal no-landform domain, including geographical T=0.
        b=self.model(takeover=0,Psum=0);b.town_clear=np.ones((2,2),bool)
        c=self.run_rows(b,raw=31,reference_raw=7,capture=True)
        np.testing.assert_allclose(self.mass(c),1,atol=1e-7)
        np.testing.assert_allclose(self.mass(c,4),1,atol=1e-7)
        result,owner,_=self.run_rows(b,raw=31,reference_raw=7)
        self.assertTrue(np.isin(result[...,0],b.dw['open'][2]).all())
        self.assertTrue((owner==4).all());self.assertTrue((result[...,1]==2).all())

    def test_empty_landform_P_with_positive_L_requires_own_legal_mix(self):
        b=self.model(takeover=0,Psum=0);b.town_clear=np.ones((2,2),bool)
        row=list(bp.LANDFORMS).index('drymtn')+1
        result,owner,_=self.run_rows(b,raw=31,reference_raw=7,ban_row=row)
        self.assertTrue((owner==5).all())
        # A deliberately broken own mix cannot be hidden by climate fallback.
        with patch.dict(gm.MIX,{'drymtn':{4:1.}}):
            with self.assertRaisesRegex(ValueError,'No legal local nonforest landform/climate recipe'):
                self.run_rows(b,raw=31,reference_raw=7,ban_row=row)

    def test_empty_P_climate_open_all_banned_fails_without_slot_zero_default(self):
        b=self.model(takeover=0,Psum=0);b.town_clear=np.ones((2,2),bool)
        with self.assertRaisesRegex(ValueError,'No legal local nonforest landform/climate recipe'):
            self.run_rows(b,raw=31,reference_raw=7,picked_climate=0,banned_slots=(24,25,26))

    def test_empty_P_climate_open_filters_then_normalizes_legal_members(self):
        b=self.model(takeover=0,Psum=0);b.town_clear=np.ones((2,2),bool)
        c=self.run_rows(b,raw=31,reference_raw=7,picked_climate=0,banned_slots=(24,25),capture=True)
        np.testing.assert_allclose(self.mass(c),1,atol=1e-7)
        positive=[item for item in c if item[2].any()]
        self.assertTrue(all(item[0]==4 and np.all(item[1]==26) for item in positive))
        result,owner,_=self.run_rows(b,raw=31,reference_raw=7,picked_climate=0,banned_slots=(24,25))
        self.assertTrue((result[...,0]==26).all());self.assertTrue((owner==4).all())

    def test_empty_P_without_rejected_source_retains_native_bytes(self):
        b=self.model(takeover=0,Psum=0)
        result,owner,_=self.run_rows(b,raw=7)
        self.assertTrue((result[...,0]==7).all());self.assertTrue((owner==1).all())
        self.assertTrue((result[...,1]==2).all())

    def test_missing_error_reports_global_coordinates_and_candidate_metadata(self):
        shape=(2,3);keep=np.zeros(shape,bool);pres=np.zeros(shape+(2,),np.uint8)
        candidates=[gb.Candidate(10,31,np.ones(shape,np.float32),3)]
        with self.assertRaises(ValueError) as error:
            gb.select(candidates,pres,keep,lambda slot:slot==31,np.zeros((32,3),np.float32),
                np.ones(4,np.float32),lambda *args:None,y0=13822,
                context={'takeover':np.ones(shape),'fd':np.zeros(shape),
                         'F':np.zeros(shape),'M':np.zeros(shape),
                         'clearance':np.ones(shape,bool),'L':np.ones(shape,np.uint8)})
        text=str(error.exception)
        for expected in ("'y': 13822","'source': 10","'R': 31","'G': 3",
                         "'weight': 1.0","'bad': True","'takeover': 1.0",
                         "'fd': 0.0","'clearance': True","'L': 1"):
            self.assertIn(expected,text)

    def prepare_fixture(self,towns):
        """Real prepare/paint/footprints/wet constructor; only assets/noise mocked."""
        import farm_fields as ff
        import strategic_ink
        import map_hex
        import terrain_noise
        ac=test_wetland_candidates.ac
        H=W=64;shape=(H*8,W*8);ts=(H*2,W*2)
        a=lambda v:np.full(ts,v,np.float32)
        cells=np.zeros((H,W,16),np.uint8)
        ac.set_ground(cells,np.ones((H,W),bool),13)
        ac.set_climate(cells,np.ones((H,W),bool),4)
        types=[f'other_{i}' for i in range(16)];types[0]='grassland';types[13]='wetland'
        fields={k:a(0) for k in ('W','Wd','Wo','cold')}
        fields['P']=np.zeros((len(gm.MATERIALS),)+ts,np.float32);fields['P'][gm.MATERIALS.index(7)]=1
        fields['L']=np.zeros(ts,np.uint8);fields['allowed']=np.zeros(ts,np.uint32)
        inherited=np.empty(shape+(2,),np.uint8);inherited[...,0]=7;inherited[...,1]=2
        inherited[256:260,256:260,0]=11
        original=inherited.copy();inherited.flags.writeable=False
        donor=np.empty_like(inherited);donor[...,0]=7;donor[...,1]=2
        donor[250:263,250:263,0]=8  # real donor_footprints extracts this dab
        domains={'native_keep':np.zeros(ts,bool),'native_filler':np.zeros(ts,bool),
                 'canvas_land':np.ones(ts,bool),'material_takeover':a(1),'vegetation_takeover':a(1)}
        rb=np.zeros((H,W),np.uint8);rn=['temperate']
        calls=[]
        def water_reader(box):
            calls.append(box);y0,y1,x0,x1=box;return np.zeros((y1-y0,x1-x0),bool)
        water={'shape':shape,'mask':water_reader,'source':{'fixture':'authoritative synthetic pixels'}}
        settings=bp.consts('polish');settings['TOWN_GROUND']={2:(32,32)}
        K=np.zeros((3,)+ts,np.float32);K[0]=1
        fm={'F':a(0),'K':K,'M':a(0),'parts':{'test':True}}
        donors={key:[] for key in ('rice','grain_cold','grain_arid','grain_temperate','tea')};donors['grain_temperate']=['test_farm']
        noise=lambda q,r,*args,**kw:np.full(np.broadcast_shapes(np.shape(q),np.shape(r)),-.25,np.float32)
        with patch.dict(os.environ,{'TW3K_FARM_MUD':'1','TW3K_CLIMATE_WETLANDS':'1'}), \
             patch.object(gb,'OrganicDraw',lambda origin,**kwargs:FixedDraw()), \
             patch.object(gm,'farm_mud_fields',autospec=True,return_value=fm) as farm_source, \
             patch.object(strategic_ink,'load',return_value=(rb,rn,object(),object())) as load, \
             patch.object(ff,'vanilla_towns',return_value={'test_farm':(32,32)}), \
             patch.object(ff,'DONORS',donors),patch.object(ff,'fbm',noise), \
             patch.object(terrain_noise,'fbm',noise),patch.object(ac,'_LGT',types), \
             patch.object(ac,'filler_hex',lambda c,n:np.zeros(c.shape[:2],bool)), \
             patch.object(map_hex,'read',side_effect=AssertionError('Unexpected asset read')), \
             patch.object(ff,'paint',wraps=ff.paint) as paint:
            result=gb.prepare(fields=fields,lon=a(80),lat=a(25),domains=domains,forest=a(0),origin=(0,0),
                cells=cells,names=['test_land'],plan={'regions':towns},biome=np.full(ts,2,np.uint8),
                surface=a(2),alpha=a(1),physical=np.ones(ts,bool),protected=domains['native_keep'],
                plots=np.zeros(ts,bool),oasis=a(1),real_bio=np.zeros(ts,np.uint8),bnames=rn,
                real_elev=a(10),real_river=a(0),q=a(0),r=a(0),inherited=inherited,donor=donor,
                water_field=water,settings=settings)
            load.assert_called_once_with();farm_source.assert_called_once()
            self.assertIs(paint.call_args.args[0],inherited)
            self.assertIs(paint.call_args.args[10],rb);self.assertIs(paint.call_args.args[11],rn)
            self.assertTrue(callable(paint.call_args.kwargs['candidate_sink']))
            self.assertIs(paint.call_args.kwargs['domains'],domains)
            self.assertIsInstance(result.wetland,ac.WetlandCandidates)
            self.assertIs(result.wetland._water,water_reader)
            self.assertEqual(calls,[])  # prepare never rasterises or reads the final blend
            result.wetland.rows(200,220,shape[1])
            self.assertEqual(len(calls),1)
        np.testing.assert_array_equal(inherited,original)
        self.assertFalse(inherited.flags.writeable)
        return result

    def test_actual_prepare_accepts_four_item_strategic_load_with_empty_plan(self):
        result=self.prepare_fixture([])
        self.assertEqual(result.farm_report['painted_towns'],0)

    def test_actual_prepare_runs_real_farm_candidate_sink_without_writing_reference(self):
        result=self.prepare_fixture([{'region':'test_farm','town':[24,24],'battle_layout':'farms_grain'}])
        self.assertGreater(len(result.farms.patches),0)
        self.assertGreater(result.farm_report['pixels'],0)
        self.assertTrue(all(np.any(p[2]>0) for p in result.farms.patches))


if __name__=='__main__':unittest.main()
