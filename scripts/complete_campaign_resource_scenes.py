"""Complete native resource scenery and seat existing clones on town terraces.

Root transforms, owning region and parent reference paths change. A renamed scene
gets its own copies of the donor leaves, named after its parent, so the
world-space leaf step bakes its translation into them and not into the donor's.
Embedded native models, building-level variants and opaque metadata remain byte
exact. The legacy decoder's "uid" field is not a proven identifier and is
preserved verbatim.
"""
import json,struct,math,hashlib,shutil,io
from pathlib import Path
import numpy as np
from PIL import Image
from analyze_reference import ROOT
import campaign_world as _frame
from parse_global_props import parse
from read_root_prefabs import decode,string
from terrain_checkpoint import StepSource,inputs
from final_terrain_surface import surface_at_display
from campaign_world import DX,DZ
from campaign_world import TERRAIN_VARIANTS
import prefab_source
A=_frame.WORKSPACE;S=A/'work/release-input';BASE=A/'terrain-before-polish'
_prefab_source=prefab_source.load(A,TERRAIN_VARIANTS,_frame.canvas_stamp(),(_frame.TILE_W,_frame.TILE_H))
BACK=A/'before-resource-scenes-native-v2';LEVEL=_frame.LEVEL
plan=json.loads((A/'work/current-region-design.json').read_text(encoding='utf-8'))
known=json.loads((ROOT/'analysis/saibei-town-prop-positions.json').read_text())
donor_positions={p['donor']:p for p in known.values()}
templates={'3k_main_anding_resource_1':[2412,2476], '3k_main_daijun_resource_1':[1912,1913],
           '3k_main_youzhou_resource_1':[1852], '3k_main_yanmen_resource_1':[1972,1973]}
report=[]


def surface_at_world(surface,x,z):
    """Bilinear height of a tile-resolution surface (rows south-up) at raw world (x, z)."""
    tx,ty=x/DX*2,z/DZ*2;h,w=surface.shape;x_,y_=min(max(tx,0),w-1),min(max(ty,0),h-1);x0,y0=int(x_),int(y_);x1,y1=min(x0+1,w-1),min(y0+1,h-1);fx,fy=x_-x0,y_-y0
    return (float(surface[y0,x0])*(1-fx)+float(surface[y0,x1])*fx)*(1-fy)+(float(surface[y1,x0])*(1-fx)+float(surface[y1,x1])*fx)*fy


def seated_y(parent,matrix,surface,assets):
    """TOWN SPEC (scene seating, 2026-10-31): the root height that seats a scene on the final ground under its own
    pieces: the median over the scene's pieces of (ground at the piece's world x, z - its y). The pieces are the
    ones stage_invariants prop_scene_offset counts (prop_instances.leaf_records + scene_piece_mask: BHM_IGNORE, not
    parked, no vegetation or decals); world = leaf position + root translation here (world_space_leaves adds the root
    translation into the leaves later). Scenes used to move with the town tile's height (surface_at_display), which
    hung or buried a scene off its plateau where the ground moved differently (v21: Gumo farm -2.80, Wusun lumber +.9).
    None (the town rule stays) when the root rotates or scales or the scene has no such pieces."""
    import prop_instances as pi
    if any(abs(matrix[i]-v)>1e-4 for i,v in ((0,1),(1,0),(2,0),(4,0),(5,1),(6,0),(8,0),(9,0),(10,1))):return None
    _,children=decode(parent);d=[]
    for c in children:
        leaf=assets.get(c['path'])
        if leaf is None or b'rigidmodels/' not in leaf:continue
        try:recs=pi.leaf_records(leaf)
        except (AssertionError,struct.error,UnicodeDecodeError):continue
        for model,m,mode in recs:
            if mode!='BHM_IGNORE' or m[10]<=pi.PARKED_Y or pi.family(model.lower()) in pi.SCENE_SKIP_FAMILIES:continue
            d.append(surface_at_world(surface,m[9]+matrix[12],m[11]+matrix[14])-m[10])
    return float(np.median(d)) if d else None


step=StepSource("resource-scenes", {**inputs(), **{v:hashlib.sha256((A/f"analysis/polished-surface-{v}.npy").read_bytes()).hexdigest() for v in TERRAIN_VARIANTS}, **_prefab_source.context()})
for variant in TERRAIN_VARIANTS:
    pre=S/f'terrain/campaigns/{variant}';p=pre/'global_props.bin';backup=BACK/p.relative_to(S)
    if not backup.exists():backup.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,backup)
    data=step.read(p)
    if data is None:
        print(variant,"resource scenes already current",flush=True);continue
    es,base=parse(data);off=next(o for k,o in es if k.endswith('/bmd_objects.bin'))
    assert off==max(o for k,o in es);root=data[base+off:];cp,rs=decode(root)
    offsets=sorted({o for k,o in es})+[len(data)-base]
    assets={k:data[base+o:base+offsets[offsets.index(o)+1]] for k,o in es};new_assets={}
    def bind_parent(source,owner):
        number=source.rsplit('.',2)[-2];newpath=f'terrain/campaigns/{variant}/bmd_objects.{owner}.{number}.bin'
        assert newpath not in new_assets,('Two scenes bound to one parent',newpath)
        body=assets[source];pc,children=decode(body);chunks=[];leaves={}
        for child in children:
            part=bytearray(body[child['start']:child['end']]);_,end=string(body,child['region_pos']);a=child['region_pos']-child['start'];z=end-child['start'];key=owner.encode()
            part[a:z]=struct.pack('<H',len(key))+key   # region first: the path precedes it, so offsets are still valid
            # The engine places leaves in world space and bakes each root's translation
            # into the leaves under its own stem (world_space_leaves.py): a renamed
            # scene gets its own copies of the donor leaves, named after its parent.
            leaf=newpath[:-4]+'.'+child['path'].rsplit('.',2)[-2]+'.bin'
            if leaf!=child['path']:
                assert leaf not in assets and b'/props/lakes/' not in assets[child['path']],leaf
                part=root_path(part,leaf);leaves[leaf]=assets[child['path']]
            chunks.append(bytes(part))
        rebound=body[:children[0]['start']]+b''.join(chunks)+body[children[-1]['end']:]
        _,checked=decode(rebound);assert all(c['region']==owner and c['path'].startswith(newpath[:-4]+'.') for c in checked)
        for k,b in {newpath:rebound,**leaves}.items():
            output=S/k;output.parent.mkdir(parents=True,exist_ok=True);output.write_bytes(b);new_assets[k]=b
        return newpath
    def root_path(blob,newpath):
        old_length=struct.unpack_from('<H',blob,2)[0];name=newpath.encode();blob[2:4+old_length]=struct.pack('<H',len(name))+name
        return blob
    owners={r['region'] for r in rs};counties={c['region']:c for c in plan['regions']}
    anchor=_prefab_source.camera(variant)
    im=Image.open(anchor);camera=np.asarray(im);scale=float(im.info['height_scale'])
    current_camera=Image.open(S/f'campaign_maps/{variant}/camera_heightmap.png').size
    if im.size!=current_camera:raise RuntimeError(f'Resource-scene camera anchor {anchor} is {im.size}, the canvas camera map is {current_camera}; it belongs to another canvas')
    surface=np.load(A/f'analysis/polished-surface-{variant}.npy')
    kept=[];moved=0;seated_on_ground=0
    for rec in rs:
        blob=bytearray(root[rec['start']:rec['end']]);county=counties.get(rec['region'])
        if county:
            q,r=county['town'];old_q,old_r=county.get('previous_town',county['town'])
            old_y=float(camera[max(0,min(camera.shape[0]-1,camera.shape[0]-1-round((old_r+old_q%2*.5)*2))),old_q*2])*scale
            # Baseline scenes sit at a fixed offset from the previous town's block corner
            # (old_q*DX, the same for every county, not from its display position): move
            # that corner onto the town's display position, the frame of the added scenes.
            tx,tz=_frame.display_position(q,r);target_y=surface_at_display(surface,q,r);matrix=list(rec['matrix']);matrix[13]+=target_y-old_y
            matrix[12]+=tx-old_q*DX;matrix[14]+=tz-(old_r+old_q%2*.5)*DZ
            _seat=seated_y(assets[rec['path']],matrix,surface,assets)   # TOWN SPEC (scene seating): on the ground under its models
            if _seat is not None:matrix[13]=_seat;seated_on_ground+=1
            struct.pack_into('<16f',blob,rec['matrix_pos']-rec['start'],*matrix);moved+=1
            blob=root_path(blob,bind_parent(rec['path'],county['region']))
        kept.append(bytes(blob))
    added=[];instances=[];uids={r['uid'] for r in rs}
    for county in plan['regions']:
        donor=county['donor']
        if donor not in templates:continue
        info=donor_positions[donor];sx,sz=info['source_world'];source_y=min(v[2] for m in info['models'] for v in m['positions'])
        # source_world is the donor town's display position (vanilla q*NATIVE_DX spacing), so
        # the target is the county town's display position too, not its block corner q*DX.
        q,r=county['town'];tx,tz=_frame.display_position(q,r);target_y=surface_at_display(surface,q,r)
        for number in templates[donor]:
            if any(rec['region']==county['region'] and rec['path'].endswith('.'+str(number)+'.bin') for rec in rs):continue
            source=f'terrain/campaigns/{variant}/bmd_objects.{donor}.{number}.bin'
            matches=[r for r in rs if r['path']==source and r['region']==donor]
            if not matches:continue
            rec=matches[0];blob=bytearray(root[rec['start']:rec['end']]);matrix=list(rec['matrix'])
            # These native scenes contain absolute child coordinates.
            matrix[12]=tx-sx;matrix[13]=target_y-source_y;matrix[14]=tz-sz
            _seat=seated_y(assets[source],matrix,surface,assets)   # TOWN SPEC (scene seating): on the ground under its models
            if _seat is not None:matrix[13]=_seat;seated_on_ground+=1
            struct.pack_into('<16f',blob,rec['matrix_pos']-rec['start'],*matrix)
            # This legacy decoder field is not a proven unique identifier:
            # every native root has value 1. Preserve it and nested metadata.
            uid=rec['uid']
            _,end=string(root,rec['region_pos']);a=rec['region_pos']-rec['start'];z=end-rec['start'];owner=county['region'].encode()
            blob[a:z]=struct.pack('<H',len(owner))+owner;blob=root_path(blob,bind_parent(source,county['region']));added.append(bytes(blob));instances.append({'region':county['region'],'source':source,'uid':uid,'nested_region_bindings_updated':True})
    head=bytearray(root[:rs[0]['start']]);struct.pack_into('<I',head,cp,len(kept)+len(added))
    updated=bytes(head)+b''.join(kept+added)+root[rs[-1]['end']:];_,check=decode(updated)
    assert len(check)==len(rs)+len(added)
    # Register the new parents in the native bundle as well as exposing them
    # as loose pack assets. Existing embedded child payloads remain byte exact.
    rootkey=next(k for k,o in es if k.endswith('/bmd_objects.bin'))
    bodies=[(k,new_assets.get(k,assets[k])) for k,_ in sorted(es,key=lambda v:v[1]) if k!=rootkey]
    bodies.extend((k,v) for k,v in sorted(new_assets.items()) if k not in assets)
    bodies.append((rootkey,updated));cursor=0;index=[]
    for key,body in bodies:index.append((key,cursor));cursor+=len(body)
    assert len({k for k,_ in index})==len(index), 'Duplicate scene bundle asset'
    header=struct.pack('<I',len(index))+b''.join(k.encode()+b'\0'+struct.pack('<I',o) for k,o in index)
    bundled=header+b''.join(body for _,body in bodies)
    checked_index,checked_base=parse(bundled);assert len(checked_index)==len(index)
    p.write_bytes(bundled);(pre/'bmd_objects.bin').write_bytes(updated)
    # Loose copies left by an earlier layout (deleted or re-donored regions) are
    # no longer in the bundle; drop them so they do not ship as orphans.
    bundled_keys={k for k,_ in index}
    for loose in pre.glob('bmd_objects.*.bin'):
        if f'terrain/campaigns/{variant}/{loose.name}' not in bundled_keys:loose.unlink()
    report.append({'variant':variant,'existing_instances_seated':moved,'seated_on_scene_ground':seated_on_ground,'added_instances':instances,'native_model_payloads_preserved_except_explicit_region_bindings':True})
    if _prefab_source.explicit:report[-1]['prefab_source']=_prefab_source.provenance()
    print(variant,'seated',moved,'existing; added',len(added),'native resource scenes',flush=True)
step.finish()
if report:(A/'analysis/resource-scene-completion.json').write_text(json.dumps({'variants':report,'live_verified':False},indent=2))
