"""Bounded diagnostic: move native lake leaf instances into expanded world space."""
import argparse,json,struct,re,math,time,hashlib
from pathlib import Path
from repair_campaign_river_roots import Assets,embedded
from analyze_reference import ROOT,inventory,digest
import campaign_world as _frame
from campaign_world import DX,DZ
from campaign_world import TERRAIN_VARIANTS

MODEL=re.compile(rb'(?:rigidmodels|terrain)/[A-Za-z0-9_./-]+\.(?:wsmodel|rigid_model_v2)',re.I)

def records(b):
    starts=[]
    for m in MODEL.finditer(b):
        s=m.start()
        if s>=8 and struct.unpack_from('<H',b,s-8)[0]==2 and struct.unpack_from('<H',b,s-2)[0]==len(m.group()) and 0<struct.unpack_from('<I',b,s-6)[0]<10000:starts.append(s)
    assert len(starts)==1,('model table',len(starts))
    s=starts[0];n=struct.unpack_from('<I',b,s-6)[0];pos=s-2;models=[]
    for _ in range(n):
        le=struct.unpack_from('<H',b,pos)[0];models.append(b[pos+2:pos+2+le].decode());pos+=2+le
    count=struct.unpack_from('<I',b,pos)[0];p=pos+4;out=[]
    for _ in range(count):
        version,idx=struct.unpack_from('<HI',b,p);assert version==20 and idx<n,(p,version,idx)
        mode=p+107;le=struct.unpack_from('<H',b,mode-2)[0];assert 4<le<100 and b[mode:mode+4]==b'BHM_'
        matrix=struct.unpack_from('<12f',b,p+24);assert all(math.isfinite(x) and abs(x)<10000 for x in matrix)
        out.append(dict(model=models[idx],matrix=matrix,position_offset=p+60))
        p+=131+le
    assert p<=len(b)
    return out

def translate(b):
    out=bytearray(b);changes=[]
    for r in records(b):
        if '/props/lakes/' not in r['model'].lower():continue
        pos=r['position_offset'];x,y,z=struct.unpack_from('<3f',b,pos)
        struct.pack_into('<f',out,pos,x+_frame.NATIVE_SHIFT_WORLD[0]);struct.pack_into('<f',out,pos+8,z+_frame.NATIVE_SHIFT_WORLD[1])
        changes.append(dict(model=r['model'],offset=pos,before=[x,y,z],after=list(struct.unpack_from('<3f',out,pos))))
    back=bytearray(out)
    for r in changes:
        p=r['offset'];back[p:p+4]=b[p:p+4];back[p+8:p+12]=b[p+8:p+12]
    assert bytes(back)==b
    return bytes(out),changes

def translated_preserving_staged_edits(b,native):
    original={r['position_offset']:r for r in records(native) if '/props/lakes/' in r['model'].lower()}
    out=bytearray(b);baseline=bytearray(b);changes=[]
    for r in records(b):
        if '/props/lakes/' not in r['model'].lower():continue
        p=r['position_offset'];n=original.pop(p)
        assert r['model']==n['model'] and r['matrix'][:9]==n['matrix'][:9]
        x,y,z=r['matrix'][9:];nx,ny,nz=n['matrix'][9:]
        assert (abs(x-nx)<.001 and abs(z-nz)<.001) or (abs(x-nx-_frame.NATIVE_SHIFT_WORLD[0])<.001 and abs(z-nz-_frame.NATIVE_SHIFT_WORLD[1])<.001),('Unknown lake position',r['model'],x,z)
        struct.pack_into('<f',out,p,nx+_frame.NATIVE_SHIFT_WORLD[0]);struct.pack_into('<f',out,p+8,nz+_frame.NATIVE_SHIFT_WORLD[1])
        baseline[p:p+4]=native[p:p+4];baseline[p+8:p+12]=native[p+8:p+12]
        changes.append(dict(model=r['model'],offset=p,before=[nx,y,nz],after=list(struct.unpack_from('<3f',out,p)),disabled_before_repair=y<-100))
    assert not original
    check=bytearray(out)
    for r in changes:
        p=r['offset'];check[p:p+4]=baseline[p:p+4];check[p+8:p+12]=baseline[p+8:p+12]
    assert bytes(check)==bytes(baseline)
    return bytes(out),bytes(baseline),changes

# --- HEHUANG PATCH: authored lake leaves skipped by the vanilla translation and its counts ---
from hehuang_lake_prop import AUTHORED_LEAVES as AUTHORED_LAKE_LEAVES
# Vanilla lake leaf files / lake instances per variant.
LAKE_LEAVES={'3k_main_map':(88,108),'3k_dlc07_main_map':(79,97)}

def source_check(apply=False,candidate=None):
    area=_frame.WORKSPACE;stage=area/'work/release-input';assets=Assets(candidate);native=Assets(original_only=True);changes=[]
    for v in TERRAIN_VARIANTS:
        key=f'terrain/campaigns/{v}/global_props.bin';current=assets.read(key);ab,entries,base=embedded(current)
        original_ab,_,_=embedded(native.read(key));out=bytearray(current)
        for k,b in ab.items():
            if b'/props/lakes/' not in b:continue
            # --- HEHUANG PATCH: authored Qinghai lake leaf (hehuang_lake_prop.py) is already in world space ---
            if k in AUTHORED_LAKE_LEAVES:continue
            assert k in original_ab,('Unknown authored lake leaf',k)
            expected,baseline,rows=translated_preserving_staged_edits(b,original_ab[k])
            if not rows:continue
            baseline_hash=hashlib.sha256(baseline).hexdigest();proof=area/'terrain-repair/lake-leaf-before'/(baseline_hash+'.bin')
            if apply:
                proof.parent.mkdir(parents=True,exist_ok=True)
                if proof.exists():assert proof.read_bytes()==baseline
                else:proof.write_bytes(baseline)
                off=dict(entries)[k];assert len(expected)==len(b);out[base+off:base+off+len(b)]=expected
                path=stage/k;path.parent.mkdir(parents=True,exist_ok=True)
                if path.exists():assert path.read_bytes() in (baseline,expected),k
                path.write_bytes(expected)
            else:
                assert proof.read_bytes()==baseline,k+' pre-repair content changed'
                assert b==expected,k+' embedded lake coordinates remain native'
                assert assets.read(k)==expected,k+' loose leaf override missing or stale'
            changes.append(dict(path=k,baseline_sha256=baseline_hash,records=rows))
        if apply:(stage/key).write_bytes(out)
    result=dict(source_pack=str(candidate) if candidate else None,leaf_files=len(changes),lake_instances=sum(len(r['records']) for r in changes),world_translation=[_frame.NATIVE_SHIFT_WORLD[0],0,_frame.NATIVE_SHIFT_WORLD[1]],all_other_leaf_bytes_preserved=True,changes=changes,live_evidence=['output/terrain-lake-leaf-probe-changjiang.png','output/terrain-lake-leaf-probe-xiangshui.png'])
    expected_files=sum(LAKE_LEAVES[v][0] for v in TERRAIN_VARIANTS);expected_instances=sum(LAKE_LEAVES[v][1] for v in TERRAIN_VARIANTS)
    assert (result['leaf_files'],result['lake_instances'])==(expected_files,expected_instances),(result['leaf_files'],result['lake_instances'])
    (area/'analysis/lake-instance-world-frame.json').write_text(json.dumps(result,indent=2));print(f'Lake leaf world coordinates: PASS ({expected_files} files / {expected_instances} instances)',flush=True)
    if apply:source_check(candidate=candidate)
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--apply',action='store_true');ap.add_argument('--audit',action='store_true');ap.add_argument('--pack',type=Path);ap.add_argument('--build-probe',action='store_true');args=ap.parse_args()
    if not args.build_probe:
        assert not (args.apply and args.pack);source_check(apply=args.apply,candidate=args.pack);return
    area=_frame.WORKSPACE;src_report=json.loads((area/'terrain-repair/river-baked-translation-probe.json').read_text());src=Path(src_report['path']);assert digest(src)==src_report['sha256']
    a=Assets(src);replacement={};changes=[]
    for v in TERRAIN_VARIANTS:
        key=f'terrain/campaigns/{v}/global_props.bin';original=a.read(key);ab,entries,base=embedded(original);out=bytearray(original)
        for k,b in ab.items():
            if b'/props/lakes/' not in b:continue
            updated,rows=translate(b)
            if not rows:continue
            replacement[k]=updated;off=dict(entries)[k];assert len(updated)==len(b);out[base+off:base+off+len(b)]=updated
            changes.append(dict(path=k,records=rows))
        replacement[key]=bytes(out)
    dest=ROOT/'output/tw3k_lake_leaf_translation_probe.pack';rows=inventory(src,False)['entries'];existing={e['path'] for e in rows}
    rows += [dict(path=k,size=len(b),compressed=False) for k,b in replacement.items() if k not in existing];rows.sort(key=lambda e:e['path'])
    ix=b''.join(struct.pack('<IB',len(replacement[e['path']]) if e['path'] in replacement else e['size'],0)+e['path'].replace('/','\\').encode()+b'\0' for e in rows)
    with src.open('rb') as inp,dest.open('xb') as out:
        out.write(b'PFH5'+struct.pack('<6I',3,0,0,len(rows),len(ix),int(time.time()))+ix)
        for e in rows:
            if e['path'] in replacement:out.write(replacement[e['path']]);continue
            inp.seek(e['offset']);left=e['size']
            while left:
                b=inp.read(min(left,1024*1024));assert b;out.write(b);left-=len(b)
    result=inventory(dest);assert set(replacement)<={e['path'] for e in result['entries']}
    report=dict(path=str(dest),sha256=result['sha256'],entries=len(rows),diagnostic_only=True,source_modified=False,leaf_files=len(changes),lake_instances=sum(len(r['records']) for r in changes),changes=changes)
    (area/'terrain-repair/lake-leaf-translation-probe.json').write_text(json.dumps(report,indent=2));print(dest,result['sha256'],'leaf files',len(changes),'lake instances',report['lake_instances'],flush=True)

if __name__=='__main__':main()
