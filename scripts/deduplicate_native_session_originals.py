"""Keep byte-identical immutable native backups as NTFS links, never installed files."""
import json,os
from pathlib import Path
from analyze_reference import ROOT,digest
import campaign_world as _frame
A=_frame.WORKSPACE

def main():
    state=json.loads((A/'combined-session.json').read_text(encoding='utf-8-sig'))
    assert state['owner']=='regions-and-battles' and not state['restored']
    assert state.get('lastMode') in (None,'romance','historical','runtime','vanilla')
    session=Path(state['session']).resolve();assert session==A.resolve()/('game-session-'+state['sessionId'])
    # The expected bytes are whatever this session backed up for slot 9 (the
    # installed release pack); never a hard-coded hash of one particular release.
    slot=next((f for f in state['files'] if int(f['index'])==9),None)
    if slot is None or not slot.get('existed') or not slot.get('sha256'):
        print('Native session has no slot-9 original to deduplicate',flush=True);return
    EXPECTED=slot['sha256'].lower()
    # Canonical copy: the oldest surviving native backup with that hash
    # (session folders are cleaned up over time, so never hard-code one).
    owned=sorted(A.glob('game-session-*/9.original'),key=lambda q:q.stat().st_mtime)
    CANONICAL=next((q for q in owned if digest(q)==EXPECTED),None)
    if CANONICAL is None:raise RuntimeError('No native 9.original backup with the ledger hash '+EXPECTED)
    files=[session/'9.original',*session.glob('9.*.compiled')];records=[]
    for p in files:
        assert p.resolve().parent==session and p.is_file()
        if digest(p)!=EXPECTED:raise RuntimeError('Foreign original in native compile session: '+str(p))
        if os.path.samefile(p,CANONICAL):continue
        temporary=p.with_name(p.name+'.verified-native-link');assert not temporary.exists()
        os.link(CANONICAL,temporary)
        assert digest(temporary)==EXPECTED
        os.replace(temporary,p)
        assert os.path.samefile(p,CANONICAL)
        records.append({'path':str(p),'sha256':EXPECTED,'bytes_preserved':True})
    journal=session/'native-backup-deduplication.json'
    if records or not journal.exists():journal.write_text(json.dumps(records,indent=2))
    print('Native session immutable originals deduplicated:',len(records),flush=True)

if __name__=='__main__':main()
