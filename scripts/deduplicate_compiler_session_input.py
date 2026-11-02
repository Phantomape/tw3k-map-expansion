"""Deduplicate only retired compiler inputs against the immutable atomic source."""
import json,os,hashlib
from pathlib import Path
from analyze_reference import ROOT
import campaign_world as _frame

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    area=_frame.WORKSPACE;state=json.loads((area/'combined-session.json').read_text(encoding='utf-8-sig'));session=Path(state['session']).resolve()
    assert session.parent==area.resolve() and state['owner']=='regions-and-battles' and not state['restored']
    assert state['lastMode'] in ('romance','historical')
    entry=next(e for e in state['files'] if e['index']==0);assert entry['expectedSha256'] is None and not Path(entry['path']).exists()
    source=area/'compiler.pack';expected=sha(source);rows=[]
    for p in session.glob('0.*.compiled'):
        assert p.resolve().parent==session
        # Installed as a hard link (game_session.ps1): already the same file, no re-hash.
        if os.path.samefile(p,source):pass
        elif sha(p)!=expected:continue
        else:
            tmp=p.with_name(p.name+'.dedup-link');assert not tmp.exists();os.link(source,tmp);os.replace(tmp,p)
        assert os.path.samefile(p,source);rows.append(dict(path=str(p),source=str(source),sha256=expected))
    assert rows,'No current retired compiler input found'
    (session/'compiler-input-deduplication.json').write_text(json.dumps(rows,indent=2));print('Retired compiler input deduplicated; no installed file linked',flush=True)

if __name__=='__main__':main()
