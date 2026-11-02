"""Read native TABLE_INDEXED 16x16 height blocks; preserve other headers."""
import struct
def parse(data):
 assert data[:10]==b'FASTBIN0\x03\x00' and data[52:65]==b'TABLE_INDEXED'
 width,height,bw,bh=struct.unpack_from('<4I',data,10);assert (bw,bh)==(16,16)
 count=struct.unpack_from('<I',data,65)[0];offsets=struct.unpack_from('<'+str(count)+'I',data,69)
 lp=69+count*4;assert struct.unpack_from('<I',data,lp)[0]==count
 lengths=struct.unpack_from('<'+str(count)+'H',data,lp+4);bp=lp+4+count*2+4
 assert len(data)-bp==struct.unpack_from('<I',data,bp-4)[0]
 return width,height,[data[bp+o:bp+o+n] for o,n in zip(offsets,lengths)]
def decode(block):
 if block[0]<128:
  count=block[0]+1;bits=(count-1).bit_length();palette=struct.unpack_from('<'+str(count)+'H',block,1)
  assert len(block)==1+count*2+bits*32,(block[0],len(block),bits)
  packed=int.from_bytes(block[1+count*2:],'little');mask=(1<<bits)-1
  return [palette[(packed>>(bits*i))&mask] for i in range(256)]
 assert block[0]&0xf0==0x80,block[0]
 bits=(block[0]&15)+1
 if bits==16:
  assert len(block)==513;return list(struct.unpack_from('<256H',block,1))
 assert len(block)==3+bits*32,(len(block),bits)
 base=struct.unpack_from('<H',block,1)[0];packed=int.from_bytes(block[3:],'little');mask=(1<<bits)-1
 return [base+((packed>>(bits*i))&mask) for i in range(256)]
def grid(data):
 width,height,blocks=parse(data);cols=(width+15)//16;out=[0]*(width*height)
 for i,block in enumerate(blocks):
  values=decode(block);ox=i%cols*16;oy=i//cols*16
  for y in range(min(16,height-oy)):
   for x in range(min(16,width-ox)):out[(oy+y)*width+ox+x]=values[y*16+x]
 return width,height,out
def rebuild(data,blocks):
 offsets=[];offset=0
 for b in blocks:offsets.append(offset);offset+=len(b)
 n=len(blocks)
 return data[:65]+struct.pack('<I',n)+struct.pack('<'+str(n)+'I',*offsets)+struct.pack('<I',n)+struct.pack('<'+str(n)+'H',*[len(b) for b in blocks])+struct.pack('<I',offset)+b''.join(blocks)
if __name__=='__main__':
 from pathlib import Path
 dds=Path('work/saibei-height.dds').read_bytes();cm=Path('work/height-assets/terrain/lf_height_map.compressed_map').read_bytes();w,h,blocks=parse(cm)
 for i in range(0,len(blocks),151):
  values=decode(blocks[i]);ox=i%((w+15)//16)*16;oy=i//((w+15)//16)*16
  for y in range(min(16,h-oy)):
   expected=struct.unpack_from('<16H',dds,128+2*((oy+y)*w+ox))
   assert values[y*16:y*16+16]==list(expected),(i,y,values[y*16:y*16+3],expected[:3])
 print('Native block decoder matches original DDS at',len(range(0,len(blocks),151)),'independent sampled tiles')
