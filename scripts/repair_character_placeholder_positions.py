"""Repair only native main-190 character coordinate sentinels in current XML."""
import argparse,json,xml.etree.ElementTree as ET
from pathlib import Path
from analyze_reference import ROOT,digest
import campaign_world as _frame
from all_directions_build import deployed_start_position,CHARACTER_PLACEHOLDERS,LEFT,BOTTOM
A=_frame.WORKSPACE; K=A/'work/standalone_kit'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--apply',action='store_true');args=ap.parse_args()
    factions=ET.parse(K/'raw_data/db/start_pos_factions.xml').getroot()
    ids={r.findtext('ID') for r in factions.findall('start_pos_factions') if r.findtext('campaign')=='3k_main_campaign_map'}
    from prepare_standalone import SOURCE   # the shipped (vanilla) assembly kit; work/assembly_kit carries an earlier experiment's edits
    native=ET.parse(SOURCE/'raw_data/db/start_pos_characters.xml').getroot()
    original={r.findtext('ID'):r for r in native.findall('start_pos_characters') if r.findtext('faction') in ids}
    reports=[]
    for rel in ('work/standalone_kit/raw_data/db/start_pos_characters.xml','work/native-split-xml/start_pos_characters.xml'):
        path=A/rel;tree=ET.parse(path);changed=[];parked=[];deployed=[]
        before={r.findtext('ID'):{c.tag:c.text for c in r} for r in tree.getroot().findall('start_pos_characters')}
        for r in tree.getroot().findall('start_pos_characters'):
            key=r.findtext('ID')
            if key not in original or original[key].findtext('start_on_map')!='1':continue
            src=original[key];pos=tuple(float(src.findtext(c)) for c in ('startx','starty'))
            expected=deployed_start_position(*pos)[:2]   # shifted, relocated off lost/impassable hexes
            actual=tuple(float(r.findtext(c)) for c in ('startx','starty'))
            if pos in CHARACTER_PLACEHOLDERS:
                assert actual in (pos,(pos[0]+LEFT,pos[1]+BOTTOM)),(key,pos,actual)
                parked.append({'id':key,'position':expected})
                if actual!=expected:
                    changed.append({'id':key,'before':actual,'after':expected})
                    if args.apply:
                        for c,v in zip(('startx','starty'),expected):r.find(c).text=format(v,'.15g')
                    else:raise AssertionError(('Placeholder still shifted',key,actual))
            else:
                if actual!=expected:
                    changed.append({'id':key,'before':actual,'after':expected})
                    if args.apply:
                        for c,v in zip(('startx','starty'),expected):r.find(c).text=format(v,'.15g')
                    else:raise AssertionError(('Deployed start not on a valid hex',key,actual,expected))
                deployed.append(key)
        assert len(parked)==48 and len(deployed)==102,(len(parked),len(deployed))
        for r in tree.getroot().findall('start_pos_characters'):
            key=r.findtext('ID');after={c.tag:c.text for c in r}
            allowed={'startx','starty'} if key in {x['id'] for x in changed} else set()
            assert all(after[k]==v for k,v in before[key].items() if k not in allowed)
        if args.apply and changed:
            backup=A/'terrain-repair/character-placeholders-before'/digest(path)/path.name
            backup.parent.mkdir(parents=True,exist_ok=True)
            if not backup.exists():backup.write_bytes(path.read_bytes())
            tree.write(path,encoding='utf-8',xml_declaration=True)
        reports.append({'path':rel,'sha256':digest(path),'changed':changed,'placeholders':parked,'real_positions_unchanged':len(deployed)})
    p=A/'analysis/character-placeholder-repair.json'
    previous=json.loads(p.read_text()) if p.exists() else {}
    original_changes=previous.get('original_repair_changes') or [dict(path=row['path'],changes=row['changed']) for row in previous.get('files',reports) if row['changed']]
    out={'sentinels':sorted(CHARACTER_PLACEHOLDERS),'main_190_parked_characters':48,'real_deployed_characters':102,'original_repair_changes':original_changes,'files':reports,'live_verified':False}
    p.write_text(json.dumps(out,indent=2),encoding='utf-8')
    print('PASS: 48 placeholders preserved, 102 deployed positions unchanged in both current XML sources')

if __name__=='__main__':main()
