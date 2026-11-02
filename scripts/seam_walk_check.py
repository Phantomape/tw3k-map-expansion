"""Stop the seam repair loop early when a coarse defect walks instead of closing.

python seam_walk_check.py --variant V --previous SCAN_DIR --current SCAN_DIR
Exit 0: no coarse class is walking. Exit 3: a coarse class did not shrink, cleared some of the previous
pass's defect hexes, and every hex it gained lies within WALK_RING hexes of them (the repair moved the
defect onto the next triangle out); the message names the meshes and triangle extents there.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np
import campaign_world as _frame
from campaign_world import DX,W,H
from audit_ground_seams import mesh_triangles
from repair_local_coarse_seams import CLASSES,S

# Hexes (4-neighbour index dilation): one repaired box is 2 hexes wide and the next
# triangle out of a big coarse grid can span several more.
WALK_RING=12

def grow(m,n):
    for _ in range(n):
        g=m.copy();g[1:]|=m[:-1];g[:-1]|=m[1:];g[:,1:]|=m[:,:-1];g[:,:-1]|=m[:,1:];m=g
    return m

def walking(previous,current):
    """Coarse classes whose defect hexes did not shrink and moved only next to the previous ones."""
    with np.load(previous/'ground-seam-masks.npz') as a,np.load(current/'ground-seam-masks.npz') as b:
        rows=[]
        # The coarse classes that fail the scan (coarse_hole_new_land is repaired but does not gate it).
        for key in (k for k in CLASSES if k!='coarse_hole_new_land'):
            p=a[key]>0;c=b[key]>0;gained=c&~p
            if p.any() and c.sum()>=p.sum() and gained.any() and (p&~c).any() and not (gained&~grow(p,WALK_RING)).any():rows.append((key,int(p.sum()),int(c.sum()),c))
    return rows

def triangles_at(variant,mask):
    """Coarse surface triangles overlapping the defect hexes' repair boxes: the 4 meshes with the largest, per mesh its count and largest extents (hexes)."""
    out=[];r,q=np.nonzero(mask)
    x0,x1=(q.min()-.5)*DX,(q.max()+1.5)*DX;z0,z1=(r.min()-.5)*DX,(r.max()+2)*DX
    for path in sorted((S/f'terrain/campaigns/{variant}/global_meshes').glob('land_mesh_*.rigid_model_v2')):
        t=mesh_triangles(path.read_bytes());lo=t.min(1);hi=t.max(1)
        hit=(hi[:,0]>=x0)&(lo[:,0]<=x1)&(hi[:,2]>=z0)&(lo[:,2]<=z1)&(lo[:,1]>-40)
        if not hit.any():continue
        size=np.maximum(hi[hit,0]-lo[hit,0],hi[hit,2]-lo[hit,2]);k=np.argsort(-size)[:3]
        out.append((float(size.max()),path.name,int(hit.sum()),[(round(float(lo[hit][i,0]/DX),1),round(float(hi[hit][i,0]/DX),1),round(float(lo[hit][i,2]/DX),1),round(float(hi[hit][i,2]/DX),1)) for i in k]))
    # The biggest faces first: those are the ones the clip keeps cutting.
    return [row[1:] for row in sorted(out,key=lambda row:-row[0])[:4]]

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--variant',default='3k_dlc07_main_map');ap.add_argument('--previous',type=Path,required=True);ap.add_argument('--current',type=Path,required=True);args=ap.parse_args()
    rows=walking(args.previous,args.current)
    if not rows:sys.exit(0)
    for key,before,now,mask in rows:
        r,q=np.nonzero(mask)
        print(f'{args.variant}: {key} is walking, not closing: {before} -> {now} hexes, now at q {q.min()}..{q.max()} r {r.min()}..{r.max()}',flush=True)
        for name,count,extents in triangles_at(args.variant,mask):
            print(f'  {name}: {count} surface triangles there; largest x/z extents (hexes) '+'; '.join(f'x {a}..{b} z {c}..{d}' for a,b,c,d in extents),flush=True)
    sys.exit(3)
