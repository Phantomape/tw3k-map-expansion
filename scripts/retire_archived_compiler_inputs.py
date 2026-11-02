"""Retire only this task's obsolete compiler copies after verified ZIP archival."""
import json,hashlib,zipfile
from pathlib import Path
from analyze_reference import ROOT
import campaign_world as _frame

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    area=_frame.WORKSPACE;records=[]
    meta=area/'terrain-repair/pre-camera-build-archive.json'
    if meta.exists():
        data=json.loads(meta.read_text());assert data['roundtrip_verified']
        item=next(r for r in data['contents'] if r['entry']=='compiler.pack')
        records.append(dict(archive=data['archive'],archive_sha256=data['archive_sha256'],entry=item['entry'],sha256=item['sha256']))
    meta2=area/'terrain-repair/sections14-20-archive.json'
    if meta2.exists():records.extend(r for r in json.loads(meta2.read_text()) if r['entry']=='compiler.pack' and r['roundtrip_verified'])
    sessions=[];active=area/'combined-session.json'
    if active.exists():
        state=json.loads(active.read_text(encoding='utf-8-sig'));session=Path(state['session']).resolve()
        assert session.parent==area.resolve() and not state['restored'] and state['owner']=='regions-and-battles';sessions.append(session)
    # Completed sessions owned by this repair, found by their restored ledgers
    # (a hard-coded session id silently matched nothing once it was cleaned up).
    for ledger in sorted(area.glob('game-session-*/ledger-restored.json')):
        state=json.loads(ledger.read_text(encoding='utf-8-sig'))
        if state.get('restored') and state.get('owner')=='regions-and-battles':sessions.append(ledger.parent.resolve())
    candidates=list(area.glob('compiler.previous-*.pack'))+[p for session in sessions for p in session.glob('0.*.compiled')];retired=[]
    current=sha(area/'compiler.pack')
    for rec in records:
        expected=rec['sha256']
        if current==expected:continue
        archive=Path(rec['archive'])
        with zipfile.ZipFile(archive) as z:size=z.getinfo(rec['entry']).file_size
        matches=[p for p in candidates if p.exists() and p.stat().st_size==size and sha(p)==expected]
        if not matches:continue
        assert sha(archive)==rec['archive_sha256']
        with zipfile.ZipFile(archive) as z:
            with z.open(rec['entry']) as f:assert hashlib.file_digest(f,'sha256').hexdigest()==expected
        for p in matches:
            assert p.resolve().parent in [area.resolve(),*sessions]
            retired.append({'path':str(p),'sha256':expected,'archive':str(archive)});p.unlink()
    # Append: the journal is the only record of earlier permanent deletions.
    journal=area/'terrain-repair/retired-archived-compiler-copies.json'
    history=json.loads(journal.read_text()) if journal.exists() else []
    journal.write_text(json.dumps(history+retired,indent=2))
    print('Archived obsolete compiler copies retired:',len(retired),flush=True)

if __name__=='__main__':main()
