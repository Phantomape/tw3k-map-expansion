"""Isolated SDK/database build stages for the 30-group atlas."""
import json,shutil,sys,xml.etree.ElementTree as ET
from pathlib import Path
from analyze_reference import ROOT,digest
import campaign_world as _frame
LEFT,BOTTOM=_frame.NATIVE_OFFSET_HEX
CHARACTER_PLACEHOLDERS=frozenset({(0.,0.),(1.,1.)})

def shifted_character_position(x,y,dx,dy,placeholders):
    """Keep explicit off-map sentinels outside the translated playable canvas."""
    return (x,y) if (x,y) in placeholders else (x+dx,y+dy)
A=_frame.WORKSPACE;KIT=A/'work/standalone_kit'
_maps={}
def _hex_cells(path):
    """(names, cells[h,w,16]) of a map.hex; rows are south-up hex r."""
    if path not in _maps:
        import numpy as np
        from analyze_reference import read_hex
        m=read_hex(path);b=path.read_bytes();w,h=m['width'],m['height']
        _maps[path]=(m['land_regions']+m['sea_regions'],np.frombuffer(b[-4-w*h*16:-4],np.uint8).reshape(h,w,16))
    return _maps[path]
def _region(names,cells,q,r):
    c=cells[r,q];i=(int(c[1])<<5)+(int(c[0])>>3)-1
    return names[i] if 0<=i<len(names) else None
def _standable(names,cells,q,r,region):
    c=cells[r,q];return (c[0]&3)!=1 and (c[2]&8)==0 and _region(names,cells,q,r)==region
def deployed_start_position(x,y):
    """Shift a vanilla start position onto the canvas and check it still stands on
    passable land of its vanilla region. Authored regions took some vanilla hexes
    (e.g. Meng Huo's start became an impassable hex of another faction's county);
    such a character moves to the nearest passable hex of his vanilla region.
    Returns (x, y, moved_from or None)."""
    nx,ny=shifted_character_position(x,y,LEFT,BOTTOM,CHARACTER_PLACEHOLDERS)
    if (nx,ny)==(x,y):return nx,ny,None
    vnames,vcells=_hex_cells(_frame.VANILLA_MAP);fnames,fcells=_hex_cells(A/'work/final-map/map.hex')
    region=_region(vnames,vcells,int(round(x)),int(round(y)))
    q,r=int(round(nx)),int(round(ny))
    # Vanilla itself starts some characters on impassable hexes; keep those as is.
    if not _standable(vnames,vcells,int(round(x)),int(round(y)),region):return nx,ny,None
    if _standable(fnames,fcells,q,r,region):return nx,ny,None
    h,w=fcells.shape[:2]
    for radius in range(1,40):
        best=None
        for dr in range(-radius,radius+1):
            for dq in range(-radius,radius+1):
                if max(abs(dq),abs(dr))!=radius:continue
                qq,rr=q+dq,r+dr
                if 0<=qq<w and 0<=rr<h and _standable(fnames,fcells,qq,rr,region):
                    d=dq*dq+dr*dr
                    if best is None or d<best[0]:best=(d,qq,rr)
        if best:return float(best[1]),float(best[2]),[nx,ny]
    raise RuntimeError(f'No passable hex of {region} near start {(nx,ny)}')
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def setup():
    if (A/'kit-source-copied.json').exists():raise RuntimeError('Kit already initialized')
    source=ROOT/'work/standalone_kit'
    for rel in ('raw_data/db','binaries','raw_data/EmpireDesignData/campaigns/3k_main_campaign_map','raw_data/EmpireDesignData/campaign_maps/3k_dlc07_main_map'):
        shutil.copytree(source/rel,KIT/rel,dirs_exist_ok=True)
    shutil.copy2(A/'work/final-map/map.hex',KIT/'raw_data/EmpireDesignData/campaign_maps/3k_dlc07_main_map/map.hex')
    dump(A/'kit-source-copied.json',{'source':str(source),'map_sha256':digest(A/'work/final-map/map.hex')})
    print('Isolated all-directions kit prepared; game configuration untouched')
def database():
    import prepare_province,prepare_standalone as base
    prepare_province.ROOT=A;base.ROOT=A;base.KIT=KIT;base.TREES.clear();base.CHANGES.clear()
    design_path=A/'work/current-region-design.json'
    design_sha_before=digest(design_path)   # design_untouched_by_build (single-source audit 3): checked at the end
    # Byte 15 (region-border sides): checked, never written here. final-map/map.hex has one writer, bake_map.py
    # --diff (single-source audit 4, coordinator 2026-11-01; stage_invariants map_hex_is_l0); stale masks stop the
    # build and are fixed in the layout (refresh_border_masks.py --apply on the source map, then re-extract the L0).
    import refresh_border_masks
    _masks=refresh_border_masks.refresh(A/'work/final-map/map.hex',apply=False)
    print('Border masks:',_masks,flush=True)
    if _masks['hexes_changed']:raise RuntimeError(f"map.hex border masks are stale on {_masks['hexes_changed']} hexes: fix the layout source, not the built map")
    # analysis/standalone-prototype.json is written once, below (its one writer). The JSON copy keeps what the
    # report was when it went through the file: the steps below edit it without touching prepare_province's tables.
    report=json.loads(json.dumps(prepare_province.main(),ensure_ascii=False))
    camera=KIT/'raw_data/db/campaign_camera_map_bounds.xml';doc=ET.parse(camera)
    row=next(r for r in doc.getroot().findall('campaign_camera_map_bounds') if r.findtext('campaign')==base.CAMPAIGN)
    for k,v in {k:str(v) for k,v in _frame.CAMERA_BOUNDS.items()}.items():
        if row.find(k) is not None:row.find(k).text=v
    doc.write(camera,encoding='utf-8',xml_declaration=True)
    factions=ET.parse(KIT/'raw_data/db/start_pos_factions.xml').getroot()
    ids={r.findtext('ID') for r in factions.findall('start_pos_factions') if r.findtext('campaign')==base.CAMPAIGN}
    source=base.SOURCE/'raw_data/db/start_pos_characters.xml';chars=ET.parse(source);shifted=[];defaults=[]
    for r in chars.getroot().findall('start_pos_characters'):
        if r.findtext('faction') not in ids or r.findtext('start_on_map')!='1':continue
        x=float(r.findtext('startx'));y=float(r.findtext('starty'))
        nx,ny,moved=deployed_start_position(x,y)
        if (nx,ny)==(x,y):defaults.append(r.findtext('ID'));continue
        r.find('startx').text=format(nx,'.15g');r.find('starty').text=format(ny,'.15g')
        shifted.append({'id':r.findtext('ID'),'old':[x,y],'new':[nx,ny],'relocated_from':moved})
    chars.write(KIT/'raw_data/db/start_pos_characters.xml',encoding='utf-8',xml_declaration=True)
    # Area-of-interest labels (Mount Wutai, the Yellow River, Great Wall...) sit at
    # vanilla logical hex positions; the native map moved by PAD_LEFT/PAD_BOTTOM on
    # the canvas, so this map's labels move with it (they stayed behind, live 2048k).
    labels=ET.parse(base.SOURCE/'raw_data/db/campaign_map_areas_of_interest_label_positions.xml');moved_labels=0
    for r in labels.getroot().findall('campaign_map_areas_of_interest_label_positions'):
        if not (r.findtext('campaign_map_area_of_interest') or '').startswith('3k_dlc07_'):continue
        r.find('logical_position_x').text=str(int(r.findtext('logical_position_x'))+_frame.PAD_LEFT)
        r.find('logical_position_y').text=str(int(r.findtext('logical_position_y'))+_frame.PAD_BOTTOM);moved_labels+=1
    labels.write(KIT/'raw_data/db/campaign_map_areas_of_interest_label_positions.xml',encoding='utf-8',xml_declaration=True)
    moved_label_rows=[{c.tag:c.text for c in r} for r in labels.getroot().findall('campaign_map_areas_of_interest_label_positions') if (r.findtext('campaign_map_area_of_interest') or '').startswith('3k_dlc07_')]
    print(moved_labels,'area-of-interest labels moved with the native map',flush=True)
    reportpath=A/'analysis/standalone-prototype.json'
    report['changed_rows']['campaign_camera_map_bounds']=[{c.tag:c.text for c in row}]
    report['changed_rows']['campaign_map_areas_of_interest_label_positions']=moved_label_rows   # shipped as a keyed fragment over vanilla
    # Landmarks of the new regions (landmarks_spec.py): AOI definitions, map binding and
    # canvas-frame label rows appended to the kit XML and to the same fragments.
    import landmarks_db
    print(landmarks_db.apply_kit(KIT,report),'landmark area-of-interest rows added',flush=True)
    # Region gameplay rows of the new regions (region_gameplay_db.py): effect bundles and more.
    import region_gameplay_db
    print(region_gameplay_db.apply_kit(KIT,report,A,json.loads((A/'work/current-region-design.json').read_text(encoding='utf-8'))),'region gameplay rows added',flush=True)
    # Jibin's Buddhist monastery: the mod's copy of the vanilla temple chain (buddhist_temple_db.py).
    import buddhist_temple_db
    print(buddhist_temple_db.apply_kit(KIT,report),'Buddhist temple rows added',flush=True)
    report['changed_rows']['start_pos_characters']=[{c.tag:c.text for c in r} for r in chars.getroot().findall('start_pos_characters') if r.findtext('faction') in ids]
    missing=[t for t in prepare_province.RESTORE_TABLES if t not in report['changed_rows']]+[t for t in report['changed_rows'] if not (KIT/'raw_data/db'/f'{t}.xml').exists()]
    if missing:raise RuntimeError(f'changed_rows: restore tables missing or tables without kit XML: {missing}')
    dump(reportpath,report);dump(A/'analysis/startpos-coordinate-conversion.json',{'shifted':shifted,'zero_defaults_preserved':defaults,'native_startpos_generated':False,'live_verified':False})
    import export_build_tables as exporter
    exporter.ROOT=A;exporter.KIT=KIT;exporter.OUT=A/'work/build-input';sys.argv=['export_build_tables.py','--campaign-split'];exporter.main()
    print(len(json.loads((A/'work/current-region-design.json').read_text(encoding='utf-8'))['regions']),'authored-region source DB prepared;',len(shifted),'deployed characters shifted;',len(defaults),'placeholder positions preserved; no game launched')
    # design_untouched_by_build: the database step reads the design and never writes it (one design writer, 2026-11-01;
    # prepare_province only checks it is normalized). normalize-design is the one permitted writer.
    import os
    if os.environ.get('TW3K_NORMALIZE_DESIGN')!='1' and digest(design_path)!=design_sha_before:
        raise RuntimeError('design_untouched_by_build: the database step changed current-region-design.json')
    bad=_frame.design_nested_mismatch(json.loads(design_path.read_text(encoding='utf-8')))
    if bad:raise RuntimeError(f'design_single_source after the database step: {bad[:5]}')
def native(split=True):
    import native_db_export as n
    n.ROOT=A;n.BIN=KIT/'binaries';sys.argv=['native_db_export.py','--all']+(['--split'] if split else []);n.main()

def official_headers():
    """Bootstrap from official KIT XML before the first database/TSV export.

    The converter writes work/native-db, without requiring split XML or reading
    workshop inputs. Run after setup and before database on a fresh workspace.
    Only its verified zero-row containers supply missing-schema table formats.
    """
    native(split=False)

def normalize_design():
    """The database step with permission to write the design once (prepare_province.normalized_design: building donors,
    battle layouts, names, gameplay templates, keys-only provinces[].regions). 'database' alone never writes the design:
    it stops when the design is not normalized (one design writer, 2026-11-01). Review and commit the design after."""
    import os
    os.environ['TW3K_NORMALIZE_DESIGN']='1'
    database()

if __name__=='__main__':{'setup':setup,'official-headers':official_headers,'database':database,'native':native,'normalize-design':normalize_design}[sys.argv[1]]()
