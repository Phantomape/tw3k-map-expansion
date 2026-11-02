import struct,json
from pathlib import Path
from analyze_reference import ROOT
def parse(data):
 count=struct.unpack_from('<I',data)[0];pos=4;entries=[]
 for i in range(count):
  end=data.index(b'\0',pos);path=data[pos:end].decode('ascii');pos=end+1
  offset=struct.unpack_from('<I',data,pos)[0];pos+=4;entries.append((path,offset))
 return entries,pos
if __name__=='__main__':
 b=(ROOT/'work/global_props.bin').read_bytes();entries,pos=parse(b)
 print('entries',len(entries),'data start',pos,'first',entries[:3],'last',entries[-3:],'data head',b[pos:pos+100].hex(' '))
 print('offset bounds',min(e[1] for e in entries),max(e[1] for e in entries),'body length',len(b)-pos)
 relevant=[e for e in entries if any(k in e[0] for k in ('3k_main_youzhou_capital.','3k_main_daijun_resource_1.','3k_main_youzhou_resource_1.','3k_main_yanmen_resource_1.'))]
 (ROOT/'analysis/saibei-props-source-index.json').write_text(json.dumps({'data_start':pos,'entries':relevant},indent=2)+'\n')
 offsets=sorted({e[1] for e in entries})+[len(b)-pos]
 for path,offset in relevant[:4]:
  size=offsets[offsets.index(offset)+1]-offset
  chunk=b[pos+offset:pos+offset+size]
  out=ROOT/'work/props-inspection'/Path(path).name;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(chunk)
  print(path,'offset',offset,'size',size,'head',chunk[:100].hex(' '))
