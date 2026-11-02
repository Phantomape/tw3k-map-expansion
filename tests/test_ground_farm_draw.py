"""ActualOrganicDraw opt-in/routing/world-coordinate contracts with tiny fake noise."""
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import ground_bake as gb


def noise(x,y,origin,scale,octaves,seed):
    return np.mod(np.asarray(x)*.017+np.asarray(y)*.023+scale*.11+octaves*.07+seed*.003,1).astype(np.float32)


class Q:
    def __init__(self,origin,specs,**kwargs):
        self.origin,self.specs=origin,specs
        self.q={k:np.linspace(-1,1,257)+sum(v) for k,v in specs.items()}
    def u(self,k,x,y):return noise(x,y,self.origin,*self.specs[k])
    def uniform(self,k,v):return np.asarray(v,np.float32)


class FarmDraw(unittest.TestCase):
    def test_opt_in_and_source_G_material_dispatch_in_both_branches(self):
        gm=SimpleNamespace(Quantiles=Q,FARM_GRAIN=4.5,FARM_DAB=1.,FARM_KINDS=((8,9),(10,11),(12,12)))
        def at(x,y,mask,origin,scale,octaves,seeds):
            X,Y=np.meshgrid(x,y)
            return np.stack([noise(X[mask],Y[mask],origin,scale,octaves,s) for s in seeds])
        gpu=SimpleNamespace(active=lambda:False,fbm_at=at)
        with patch.dict(sys.modules,{'ground_materials':gm,'gpu_noise':gpu}):
            old=gb.OrganicDraw((696,343));farm=gb.OrganicDraw((696,343),farm_grain=4.5)
            new=gb.OrganicDraw((696,343),farm_grain=4.5,inner_grain=1.)
            legacy=gb.OrganicDraw.__new__(gb.OrganicDraw);legacy.q=old.q
            self.assertIsNone(old.farm_q)
            self.assertEqual(farm.farm_q.specs,{m:(4.5,4,450+m) for m in range(8,13)})
            self.assertEqual(new.farm_q.specs,{8:(1.,2,458),9:(4.5,4,459),10:(1.,2,460),11:(4.5,4,461),12:(4.5,4,462)})
            self.assertEqual(new.q.specs,old.q.specs)
            for k in old.q.q:np.testing.assert_array_equal(new.q.q[k],old.q.q[k])
            gc=gb.GroundCandidates({}, {}, np.zeros((2,3),np.float32),(696,343),(8,12))
            self.assertEqual(gc.draw.farm_q.specs,new.farm_q.specs)
            y,x=np.mgrid[:17,:23];mask=(x+y)%3!=0
            for active in (False,True):
                gpu.active=lambda:active
                for purpose in [s+16*g for g in range(4) for s in range(1,13)]+[101]:
                    for material in (8,9,10,11,12):
                        slots=np.full(mask.shape,material,np.uint8)
                        a=farm.rows(11390,16380)(purpose,slots,mask)
                        b=new.rows(11390,16380)(purpose,slots,mask)
                        changed=purpose%16==6 and material in (8,10)
                        self.assertEqual(np.array_equal(a,b),not changed,(active,purpose,material))
                        np.testing.assert_array_equal(old.rows(11390,16380)(purpose,slots,mask),legacy.rows(11390,16380)(purpose,slots,mask))
                # Mixed slots and absolute rows remain identical under halo splitting.
                slots=(8+x%5).astype(np.uint8);whole=np.zeros(mask.shape,np.float32)
                whole[mask]=new.rows(11390,16380)(54,slots,mask)
                blocked=np.zeros_like(whole)
                for a,b in ((0,5),(5,11),(11,17)):
                    lo,hi=max(0,a-2),min(17,b+2);m=mask[lo:hi];v=np.zeros(m.shape,np.float32)
                    v[m]=new.rows(11390+lo,16380)(54,slots[lo:hi],m)
                    blocked[a:b]=v[a-lo:b-lo]
                np.testing.assert_array_equal(whole,blocked)


if __name__=='__main__':unittest.main()
