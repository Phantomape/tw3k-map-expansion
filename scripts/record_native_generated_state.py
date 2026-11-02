"""Record verified native outputs from the matching session launch."""
import argparse,json
from datetime import datetime,timezone
from analyze_reference import ROOT,digest
import campaign_world as _frame
from native_compilation_state import atomic_json,current_sources,state_path

def record_capture(area,mode,launch):
 current=current_sources(area)
 if launch.get('schema')!=2 or launch.get('restored') or launch.get('lastMode')!=mode:raise RuntimeError('Capture requires active matching version-2 build session')
 if launch.get('compilerSources')!=current:raise RuntimeError('Compiler/map/design changed since native build launch')
 started=datetime.fromisoformat(launch['lastStartedUtc'].replace('Z','+00:00')).timestamp()
 names=('startpos_'+mode+'.esf','hlp_data.esf','spd_data.esf');hashes={}
 for name in names:
  path=area/'generated'/name;stat=path.stat()
  if stat.st_size<1024 or stat.st_mtime<started:raise RuntimeError('Missing, too small, or stale native output: '+name)
  hashes[name]=digest(path)
 path=state_path(area);state=json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else {'modes':{}}
 timestamp=datetime.now(timezone.utc).isoformat()
 state.setdefault('modes',{})[mode]=dict(current,startpos_sha256=hashes[names[0]],capturedUtc=timestamp)
 state['ai_map']=dict(current,producer_mode=mode,hlp_data_sha256=hashes['hlp_data.esf'],spd_data_sha256=hashes['spd_data.esf'],capturedUtc=timestamp)
 atomic_json(path,state);return state

def main():
 parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('romance','historical'));parser.add_argument('--capture-ledger');args=parser.parse_args()
 area=_frame.WORKSPACE;ledger=args.capture_ledger or str(area/'combined-session.json')
 record_capture(area,args.mode,json.loads(open(ledger,encoding='utf-8-sig').read()));print('Recorded startpos and HLP/SPD provenance for '+args.mode)
if __name__=='__main__':main()
