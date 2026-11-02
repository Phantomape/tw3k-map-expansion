import struct,json,collections
from pathlib import Path
from analyze_reference import ROOT,read_hex
def parse(data):
 count=struct.unpack_from('<I',data,12)[0];pos=16;paths=[]
 for i in range(count):
  n=struct.unpack_from('<H',data,pos)[0];pos+=2;paths.append(data[pos:pos+n].decode());pos+=n
 climate_count=struct.unpack_from('<I',data,pos)[0];pos+=4;climates=[]
 for i in range(climate_count):
  n=struct.unpack_from('<H',data,pos)[0];pos+=2;climates.append(data[pos:pos+n].decode());pos+=n
 info_pos=pos;pos+=24+44+1;count_pos=pos;tiles=struct.unpack_from('<I',data,pos)[0];pos+=4
 assert len(data)-pos==tiles*21,(len(data),pos,tiles)
 records=[struct.unpack_from('<HIBhhHff',data,pos+i*21) for i in range(tiles)]
 assert all(r[0]==1 and 0<=r[1]<len(paths) for r in records)
 return paths,climates,count_pos,pos,records,info_pos
def identity_sha256(data):
 """sha256 of tile_list.bin bytes with every record's lo/hi height range (the last 8 of its 21 bytes) zeroed:
 equal for lists whose records differ only in their ranges (repair_calibrated_native_tile_ranges rewrites them
 after the overlay audit), different for any added, removed, reordered or otherwise changed record or header."""
 import hashlib,numpy as np
 count=struct.unpack_from('<I',data,12)[0];pos=16
 for i in range(count):pos+=2+struct.unpack_from('<H',data,pos)[0]
 climate_count=struct.unpack_from('<I',data,pos)[0];pos+=4
 for i in range(climate_count):pos+=2+struct.unpack_from('<H',data,pos)[0]
 pos+=24+44+1;tiles=struct.unpack_from('<I',data,pos)[0];pos+=4
 assert len(data)-pos==tiles*21,(len(data),pos,tiles)
 rec=np.frombuffer(data,np.uint8,count=tiles*21,offset=pos).reshape(tiles,21).copy();rec[:,13:]=0
 return hashlib.sha256(bytes(data[:pos])+rec.tobytes()).hexdigest()
if __name__=='__main__':
 b=(ROOT/'work/mesh-inspection/tile_list.bin').read_bytes();paths,climates,count_pos,pos,records,info=parse(b)
 print('Records',len(records),'first',records[:6],'rotations',collections.Counter(r[2] for r in records))
 print('XY bounds',min(r[3] for r in records),max(r[3] for r in records),min(r[4] for r in records),max(r[4] for r in records))
 print('Float bounds',min(r[6] for r in records),max(r[6] for r in records),min(r[7] for r in records),max(r[7] for r in records))
 print('Info floats',struct.unpack_from('<6f',b,info),'ints',struct.unpack_from('<11i',b,info+24))
 selected=[(i,paths[r[1]],r) for i,r in enumerate(records) if 1000<=r[3]<=1150 and 1220<=r[4]<=1380 and 'mountain' in paths[r[1]]]
 print('Province mountain tiles',len(selected));print(selected[:6])
 (ROOT/'analysis/saibei-terrain-tile-candidates.json').write_text(json.dumps(selected,indent=2)+'\n')
