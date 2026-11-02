"""Farm margin semantics and real rows() mass contracts, with tiny arrays only."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import ground_bake as gb
import test_ground_candidates_integration as integration


class FarmMargin(unittest.TestCase):
    def test_cdf_matches_integrated_frayed_uniform_and_knots(self):
        points=np.r_[[-1,-.05,0,.05,.95,1,1.05,2],np.linspace(-.06,1.06,37)]
        v=(np.arange(20000,dtype=np.float64)+.5)/20000
        integrated=np.mean(np.clip(points[:,None]-.1*(v[None,:]-.5),0,1),axis=1)
        np.testing.assert_allclose(gb.frayed_uniform_cdf(points,.1),integrated,atol=2e-9)
        np.testing.assert_allclose(gb.frayed_uniform_cdf(np.array([-.05,.05,.95,1.05]),.1),[0,.05,.95,1],atol=1e-14)
        np.testing.assert_allclose(gb.frayed_uniform_cdf(points,0),np.clip(points,0,1))
        ordered=gb.frayed_uniform_cdf(np.linspace(-.06,1.06,301),.1)
        self.assertTrue((np.diff(ordered)>=0).all());self.assertTrue(((ordered>=0)&(ordered<=1)).all())

    def test_conditional_tail_and_stable_low_weights(self):
        f=np.array([0,.02,.03,.05,.07,.1,.15],np.float32);original=f.copy()
        inner=.55*gb.farm_inner_fraction(f,.07,.1)
        np.testing.assert_allclose(inner,[0,0,.00859375,.0495,.0982142857,.176,.2933333333],atol=2e-8)
        np.testing.assert_array_equal(f,original)
        self.assertEqual(inner.dtype,np.float32)
        tiny=gb.farm_inner_fraction(np.array([1e-12,1e-8,1e-5],np.float32),.07,.1)
        np.testing.assert_array_equal(tiny,0)
        self.assertTrue(np.isfinite(tiny).all())
        with self.assertRaises(ValueError):gb.frayed_uniform_cdf(f,-.1)

    def test_actual_rows_change_only_farm_recipe_not_source_masses(self):
        fixture=integration.GroundIntegration();b=fixture.model(F=.06,M=.1)
        b.fields['K'][...]=0;b.fields['K'][1]=1
        fields={key:value.copy() for key,value in b.fields.items()}
        old=lambda F,rim,fray:np.clip(F/max(float(rim),.05),0,1)
        with patch.object(gb,'farm_inner_fraction',old):before=fixture.run_rows(b,capture=True)
        after=fixture.run_rows(b,capture=True)
        for source in range(1,13):
            np.testing.assert_allclose(fixture.mass(after,source),fixture.mass(before,source),atol=1e-7)
        for key,value in fields.items():np.testing.assert_array_equal(b.fields[key],value)
        before10=sum(c[2] for c in before if c[0]==6 and int(c[1])==10)
        after10=sum(c[2] for c in after if c[0]==6 and int(c[1])==10)
        self.assertTrue((after10<before10).all())
        for c,d in zip(before,after):
            self.assertEqual(c[0],d[0]);np.testing.assert_array_equal(c[1],d[1]);np.testing.assert_array_equal(c[3],d[3])
            if c[0]!=6:np.testing.assert_array_equal(c[2],d[2])
        # With no farm ownership the exact old/new candidate bytes still agree.
        b=fixture.model(F=0,M=.1)
        with patch.object(gb,'farm_inner_fraction',old):before=fixture.run_rows(b,capture=True)
        after=fixture.run_rows(b,capture=True)
        for c,d in zip(before,after):
            for x,y in zip(c,d):np.testing.assert_array_equal(x,y)


if __name__=='__main__':unittest.main()
