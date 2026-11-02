"""Match the exported theatre to native terrain cell spacing.

The SDK emits a fixed 595.1-wide theatre even for a wider hex grid. Scale its
world-coordinate bounds and polygons through EsfLibrary, retaining logical
hex data. Renderer decompression is an explicit one-time recovery option.
"""
import json, struct, shutil, hashlib, subprocess, sys
from pathlib import Path
import numpy as np
from analyze_reference import ROOT, digest
import campaign_world as _frame
from parse_global_props import parse
from read_root_prefabs import decode
from read_tile_list import parse as tile_parse
A=_frame.WORKSPACE; S=A/'work/release-input'
BACK=A/'before-world-extent-repair'; _frame.check_snapshot(BACK); MARK=A/'analysis/world-extent-repair.json'
FACTOR=(_frame.W-1)/(_frame.NATIVE_W-1)
prior=json.loads(MARK.read_text()) if MARK.exists() else {'hashes':{}}
hashes=prior['hashes']
# Files are keyed and backed up relative to the checkout that owns the workspace (<root>/work/<area>): the same
# keys as before in a checkout's own workspace, and a worktree's scripts can run on the main workspace.
BASE=A.resolve().parents[1]
def edit(p, transform, current_source=False):
    rel=p.resolve().relative_to(BASE).as_posix()
    if hashes.get(rel)==digest(p): return
    backup=BACK/p.resolve().relative_to(BASE)
    if not backup.exists(): backup.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,backup)
    b=transform(bytearray(p.read_bytes() if current_source else backup.read_bytes()));p.write_bytes(b);hashes[rel]=digest(p)
def mesh(b):
    mp=struct.unpack_from('<I',b,152)[0]; t,f,sz,vo,vc,io,ic=struct.unpack_from('<HH5I',b,mp)
    assert t==101
    v=np.frombuffer(b,dtype='<f4',offset=mp+vo,count=vc*4).reshape(vc,4)
    v[:,0]*=FACTOR;v[:,2]*=FACTOR
    box=list(struct.unpack_from('<6f',b,mp+24))
    for i in (0,2,3,5):box[i]*=FACTOR
    struct.pack_into('<6f',b,mp+24,*box);return b
def cm(b):
    box=list(struct.unpack_from('<6f',b,26))
    for i in (0,2,3,5):box[i]*=FACTOR
    struct.pack_into('<6f',b,26,*box);return b
def root_scene(root):
    _,records=decode(root)
    for r in records:
        mat=list(r['matrix'])
        for i in (0,2,4,6,8,10,12,14):mat[i]*=FACTOR
        struct.pack_into('<16f',root,r['matrix_pos'],*mat)
    return root
def props(b):
    es,base=parse(b);off=next(o for k,o in es if k.endswith('/bmd_objects.bin'))
    assert off==max(o for k,o in es)
    b[base+off:]=root_scene(bytearray(b[base+off:]));return b
def trees(b):
    pos=24
    for g in range(struct.unpack_from('<I',b,20)[0]):
        n=struct.unpack_from('<H',b,pos)[0];pos+=2+n
        num=struct.unpack_from('<I',b,pos)[0];pos+=4
        for i in range(num):
            x,y,z=struct.unpack_from('<3f',b,pos);struct.pack_into('<3f',b,pos,x*FACTOR,y,z*FACTOR)
            seasons=struct.unpack_from('<I',b,pos+14)[0];pos+=18+seasons*4
    assert pos==len(b);return b
# The tile lists have one writer, the tile bake (pipeline-bake-design step 1); the one-time recovery below
# (renderer decompression) also rewrote both tile_list.bin detail scales and is retired with that rule.
assert '--undo-old-compression' not in sys.argv,'retired one-time recovery: it rewrites the tile lists outside the tile bake'
for variant in (('3k_main_map','3k_dlc07_main_map') if '--undo-old-compression' in sys.argv else ()):
    pre=S/f'terrain/campaigns/{variant}'
    for p in (pre/'global_meshes').glob('*.rigid_model_v2'):edit(p,mesh)
    for p in (pre/'global_meshes').glob('*.compressed_map'):edit(p,cm)
    edit(pre/'global_props.bin',props);edit(pre/'bmd_objects.bin',root_scene)
    edit(S/f'campaign_maps/{variant}/display/trees/trees.campaign_tree_list',trees)
    # Restore the native setting after the unsuccessful detail-scale probe.
    for suffix in ('tile_list.bin','global_map/tile_list.bin'):
        p=pre/suffix;b=bytearray(p.read_bytes());info=tile_parse(b)[-1]
        struct.pack_into('<f',b,info+20,1.3329999446868896);p.write_bytes(b)
p=A/'work/standalone_kit/working_data/campaign_maps/3k_dlc07_main_map/map_data.esf'
def theatre(b):
    original=A/'analysis/world-extent-native-input.esf';result=A/'analysis/world-extent-native-output.esf'
    original.write_bytes(b)
    subprocess.run([str(ROOT/'tools/caime-1.1.0/lib/app/Tw3kWorldExtent.exe'),str(original),str(result),str(_frame.WORLD_WIDTH)],check=True)
    return result.read_bytes()
edit(p,theatre,current_source=True)
prior.update(hashes=hashes,world_scale=_frame.WORLD_WIDTH/_frame.NATIVE_WORLD_WIDTH,world_width=_frame.WORLD_WIDTH,world_height=_frame.WORLD_HEIGHT,
             reason='SDK theatre bounds were fixed while native terrain cell spacing was retained',live_verified=False)
MARK.write_text(json.dumps(prior,indent=2));print('Expanded theatre and native render coordinates synchronized',prior['world_width'],prior['world_height'])
