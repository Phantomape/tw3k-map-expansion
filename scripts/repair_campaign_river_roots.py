"""Place rivers through the model pivots actually consumed by the renderer.

GPU replay proved that root translations were absent from river instance matrices.
Bake the world translation into each unique river model's pivot and bounds, and
clear the corresponding root translation. Local vertices and indices stay exact.
"""
import tw3k_paths
import argparse, hashlib, json, re, struct
from pathlib import Path
import xml.etree.ElementTree as ET
from analyze_reference import ROOT, inventory, digest
import campaign_world as _frame
from campaign_world import DX, DZ
from props_bundle import embedded   # also imported from here by many steps
from read_root_prefabs import decode
from campaign_world import TERRAIN_VARIANTS

A = _frame.WORKSPACE
S = A / 'work/release-input'
G = tw3k_paths.GAME_DATA
RIVER = re.compile(rb'terrain/campaigns/3k_(?:dlc07_)?main_map/models/river_\d+\.wsmodel')

class Assets:
    def __init__(self, candidate=None, original_only=False):
        self.index = {}
        for p in ([candidate] if candidate else []) + [G / n for n in ('terrain.pack','fast.pack','data.pack')]:
            for e in inventory(p, False)['entries']:
                self.index.setdefault(e['path'], (p, e))
        self.candidate = candidate
        self.original_only = original_only
    def read(self, key):
        if not self.original_only and not self.candidate and (S / key).exists():
            return (S / key).read_bytes()
        p, e = self.index[key]
        assert not e['compressed'], key
        with p.open('rb') as f:
            f.seek(e['offset']); return f.read(e['size'])

def references(rec, assets):
    prefix = rec['path'][:-4] + '.'
    return sorted({m.group().decode() for k,b in assets.items()
                   if k.startswith(prefix) for m in RIVER.finditer(b)})

def corrected(data):
    assets, entries, base = embedded(data)
    key = next(k for k,_ in entries if k.endswith('/bmd_objects.bin'))
    root = bytearray(assets[key]); _, records = decode(root)
    from translate_campaign_lake_instances import records as leaf_records
    rows = []; body = bytearray(data); offset_of = dict(entries); leaf_edits = []
    for rec in records:
        models = references(rec, assets)
        if not models: continue
        matrix = list(rec['matrix'])
        assert min(abs(matrix[12]),abs(matrix[12]-_frame.NATIVE_SHIFT_WORLD[0]))<1e-4 and abs(matrix[13])<1e-5
        assert min(abs(matrix[14]),abs(matrix[14]-_frame.NATIVE_SHIFT_RAW[1]),abs(matrix[14]-_frame.NATIVE_SHIFT_WORLD[1]))<1e-4
        dx, dz = matrix[12], matrix[14]
        before = matrix[14]; matrix[12] = matrix[13] = matrix[14] = 0.
        struct.pack_into('<16f', root, rec['matrix_pos'], *matrix)
        # River water is placed by its model pivot, but other models in the same
        # root (port walkways, props) are placed by the root translation: carry the
        # translation removed above into their own positions, or they stay at
        # vanilla coordinates (only when it is removed, so a rerun is a no-op).
        if abs(dx) > 1e-4 or abs(dz) > 1e-4:
            stem = rec['path'][:-4] + '.'
            for k, b in assets.items():
                if not (k.startswith(stem) and k.endswith('.bin')) or k == key: continue
                try: instances = leaf_records(b)
                except Exception: continue
                for inst in instances:
                    m = inst['model'].lower()
                    if '/river' in m or '/lakes/' in m: continue
                    at = base + offset_of[k] + inst['position_offset']
                    x, y, z = struct.unpack_from('<3f', body, at)
                    struct.pack_into('<3f', body, at, x+dx, y, z+dz)
                    leaf_edits.append({'leaf':k,'model':inst['model'],'before':[x,y,z],'after':list(struct.unpack_from('<3f', body, at))})
        rows.append({'root':rec['path'],'models':models,'old_z':before,'new_z':0.,'placement':'translated model pivot','non_water_instances_translated':sum(e['leaf'].startswith(rec['path'][:-4]+'.') for e in leaf_edits)})
    assert len(rows)==23, len(rows)
    off = dict(entries)[key]; assert off == max(o for _,o in entries)
    out = bytes(body[:base+off]) + bytes(root)
    # Prove that only the selected translation fields and leaf positions changed.
    restored = bytearray(out)
    for rec in records:
        if references(rec,assets):
            p = base+off+rec['matrix_pos']+12*4
            restored[p:p+12] = data[p:p+12]
    for e in leaf_edits:
        at = base + offset_of[e['leaf']]
        restored[at:at+len(assets[e['leaf']])] = assets[e['leaf']]
    assert bytes(restored)==data
    return out, key, bytes(root), rows

def translated_model(original):
    b=bytearray(original);mp=struct.unpack_from('<I',b,152)[0]
    typ,flags,size,vo,vc,io,ic=struct.unpack_from('<HH5I',b,mp)
    assert typ==68 and io-vo==vc*48
    fields=((mp+24,_frame.NATIVE_SHIFT_WORLD[0]),(mp+32,_frame.NATIVE_SHIFT_WORLD[1]),(mp+36,_frame.NATIVE_SHIFT_WORLD[0]),(mp+44,_frame.NATIVE_SHIFT_WORLD[1]),(mp+628,_frame.NATIVE_SHIFT_WORLD[0]),(mp+636,_frame.NATIVE_SHIFT_WORLD[1]))
    for pos,delta in fields:
        value=struct.unpack_from('<f',b,pos)[0];struct.pack_into('<f',b,pos,value+delta)
    check=bytearray(b)
    for pos,_ in fields:check[pos:pos+4]=original[pos:pos+4]
    assert bytes(check)==original
    return bytes(b)

def patches(data):
    assert data[:10]==b'FASTBIN0\x01\x00'
    pos=14; rows=[]
    for _ in range(struct.unpack_from('<I',data,10)[0]):
        version,n=struct.unpack_from('<2H',data,pos); assert version==1
        pos+=4; name=data[pos:pos+n].decode(); pos+=n
        box=struct.unpack_from('<4f',data,pos); pos+=16
        rows.append((name,box))
    assert pos==len(data)
    return rows

def translated_patches(original,zshift=_frame.NATIVE_SHIFT_WORLD[1]):
    out=bytearray(original); pos=14
    assert len(patches(original))==55
    for _ in range(55):
        _,n=struct.unpack_from('<2H',original,pos);pos+=4+n
        x0,z0,x1,z1=struct.unpack_from('<4f',original,pos)
        struct.pack_into('<4f',out,pos,x0+_frame.NATIVE_SHIFT_WORLD[0],z0+zshift,x1+_frame.NATIVE_SHIFT_WORLD[0],z1+zshift);pos+=16
    assert pos==len(original)
    return bytes(out)

def audit(assets,variant):
    pre=f'terrain/campaigns/{variant}/'
    native=Assets(original_only=True)
    original=native.read(pre+'height_patches/rivers.height_patch_collection')
    expected=translated_patches(original)
    patch_data=assets.read(pre+'height_patches/rivers.height_patch_collection')
    assert patch_data==expected, 'River height patches must match the native world Z frame'
    ab,_,_=embedded(assets.read(pre+'global_props.bin'))
    roots=decode(ab[pre+'bmd_objects.bin'])[1]; pr=patches(patch_data); checked=[]
    for rec in roots:
        for model in references(rec,ab):
            assert max(abs(rec['matrix'][i]) for i in (12,13,14))<1e-5, rec['path']
            geom=ET.fromstring(assets.read(model)).findtext('geometry')
            data=assets.read(geom); mp=struct.unpack_from('<I',data,152)[0]
            assert data==translated_model(native.read(geom)), geom+' must contain only the proved pivot/bounds translation'
            bb=struct.unpack_from('<6f',data,mp+24)
            pivot=struct.unpack_from('<3f',data,mp+628)
            centre_error=max(abs(pivot[i]-(bb[i]+bb[i+3])*.5) for i in range(3))
            assert centre_error<1e-4,(geom,'pivot/bounds centre',centre_error)
            stem=Path(model).stem
            boxes=[box for name,box in pr if ('/'+stem+'_patch_') in name]
            assert boxes, model
            union=(min(b[0] for b in boxes),min(b[1] for b in boxes),max(b[2] for b in boxes),max(b[3] for b in boxes))
            placed=(bb[0]+rec['matrix'][12],bb[2]+rec['matrix'][14],bb[3]+rec['matrix'][12],bb[5]+rec['matrix'][14])
            assert all((placed[i]>=union[i]-1e-3 if i<2 else placed[i]<=union[i]+1e-3) for i in range(4)), (model,placed,union)
            checked.append({'model':model,'placed_bounds':placed,'patch_union':union,'pivot':pivot,'pivot_bbox_centre_error':centre_error})
    assert len(checked)==24, len(checked)
    return {'variant':variant,'river_roots':23,'models_checked':checked,'world_frame_patches_verified':True,'patch_pixels_preserved':True,'placement':'model pivot; root translation zero','local_vertices_and_indices_preserved':True}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--audit',action='store_true');ap.add_argument('--pack',type=Path);args=ap.parse_args()
    if args.pack: assert args.audit
    assets=Assets(args.pack); reports=[]
    for v in TERRAIN_VARIANTS:
        key=f'terrain/campaigns/{v}/global_props.bin'
        if not args.audit:
            data=assets.read(key); out,rk,root,changes=corrected(data)
            backup=A/'terrain-repair/river-roots-before'/hashlib.sha256(data).hexdigest()
            backup.mkdir(parents=True,exist_ok=True); p=backup/(v+'.bin')
            if not p.exists():p.write_bytes(data)
            (S/key).write_bytes(out);(S/rk).write_bytes(root)
            # Loose copies of leaves translated above follow the bundle.
            before_assets=embedded(data)[0]
            for leaf,body in embedded(out)[0].items():
                if body!=before_assets.get(leaf) and leaf!=rk and (S/leaf).exists():(S/leaf).write_bytes(body)
            native_assets=Assets(original_only=True)
            for row in changes:
                for model in row['models']:
                    geom=ET.fromstring(assets.read(model)).findtext('geometry')
                    p=S/geom;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(translated_model(native_assets.read(geom)))
            patch_key=f'terrain/campaigns/{v}/height_patches/rivers.height_patch_collection'
            native=Assets(original_only=True).read(patch_key)
            current=assets.read(patch_key)
            expected=(translated_patches(native,_frame.NATIVE_SHIFT_RAW[1]),translated_patches(native))
            base_copy=_frame.WORKSPACE/'terrain-before-polish'/patch_key
            if current not in expected and not args.pack and base_copy.exists() and base_copy.read_bytes() in expected:
                # A stage copy left from another canvas frame: the base is current.
                current=base_copy.read_bytes();(S/patch_key).write_bytes(current)
                print(v,'stale river height patches reseeded from terrain-before-polish',flush=True)
            assert current in expected, 'Unexpected river patch source'
            pb=A/'terrain-repair/river-patches-before'/hashlib.sha256(current).hexdigest()
            pb.mkdir(parents=True,exist_ok=True); pp=pb/(v+'.bin')
            if not pp.exists():pp.write_bytes(current)
            (S/patch_key).write_bytes(translated_patches(native))
        reports.append(audit(assets,v))
    report={'source_pack':str(args.pack) if args.pack else None,'coordinate_frame':'world Z','coordinate_evidence':'native-river-coordinate-evidence.json','runtime_evidence':'terrain-repair/shader-audit/river-translation-baked-probe.json','live_verified':False,'variants':reports}
    (A/'analysis/native-river-root-frame.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(f'River model pivots, zero roots and world patch bounds: PASS ({23*len(TERRAIN_VARIANTS)} roots / {24*len(TERRAIN_VARIANTS)} models)',flush=True)

if __name__=='__main__': main()
