"""Read the native 3K root prefab records without rewriting unknown fields."""
import re,struct,json
from pathlib import Path
from parse_global_props import parse
from analyze_reference import ROOT
def string(data,pos):
 n=struct.unpack_from('<H',data,pos)[0];return data[pos+2:pos+2+n].decode(),pos+2+n
def decode(root):
 matches=[]
 for m in re.finditer(rb'terrain/campaigns/3k_(?:dlc[0-9]+_)?main_map/bmd_objects\.[a-zA-Z0-9_.]+\.bin',root):
  if struct.unpack_from('<H',root,m.start()-4)[0]!=9:continue
  if struct.unpack_from('<H',root,m.start()-2)[0]!=len(m.group()):continue
  matches.append(m)
 first=matches[0].start()-4;count_pos=first-4;count=struct.unpack_from('<I',root,count_pos)[0]
 assert count==len(matches),(count,len(matches))
 records=[]
 for i,m in enumerate(matches):
  start=m.start()-4;path,p=string(root,start+2);matrix_pos=p;matrix=struct.unpack_from('<16f',root,p);p+=64
  overrides=struct.unpack_from('<I',root,p)[0];p+=4;assert overrides==0,(i,overrides)
  mask=struct.unpack_from('<I',root,p)[0];p+=4
  region_pos=p;region,p=string(root,p);flag=root[p];p+=1;height_mode,p=string(root,p);uid_pos=p;uid=struct.unpack_from('<Q',root,p)[0];p+=8
  # The legacy "uid" label is unproven: native roots consistently store 1.
  # Preserve this eight-byte field and the following 20 opaque bytes verbatim.
  end=p+20
  if i+1<len(matches):assert end==matches[i+1].start()-4,(i,end,matches[i+1].start()-4)
  records.append(dict(start=start,end=end,path=path,matrix_pos=matrix_pos,matrix=matrix,mask=mask,region_pos=region_pos,region=region,height_flag=flag,height_mode=height_mode,uid_pos=uid_pos,uid=uid))
 return count_pos,records
if __name__=='__main__':
 root=(ROOT/'work/props-inspection/bmd_objects.bin').read_bytes()
 count_pos,records=decode(root)
 print('Decoded',len(records),'count offset',count_pos,'tail',len(root)-records[-1]['end'])
 relevant=[r for r in records if r['region'] in ('3k_main_youzhou_capital','3k_main_daijun_resource_1','3k_main_youzhou_resource_1','3k_main_yanmen_resource_1')]
 (ROOT/'analysis/saibei-root-prefab-source.json').write_text(json.dumps(relevant,indent=2)+'\n')
 for r in relevant:print(Path(r['path']).name,'uid',r['uid'],'matrix',r['matrix'],'region',r['region'],'height',r['height_mode'])
