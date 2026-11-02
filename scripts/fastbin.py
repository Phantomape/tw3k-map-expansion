"""The battle-locations FASTBIN codec (terrain/battles/.../battle_locations_map.bin): parse() splits the
file into header, groups of entrance rows and tail; enc() encodes one string field."""
import struct


def parse(data):
    if data[:12]!=b'FASTBIN0\x02\x00\x01\x00':raise ValueError('Unexpected FASTBIN format')
    pos=12
    def integer():
        nonlocal pos
        v=struct.unpack_from('<I',data,pos)[0];pos+=4;return v
    def string():
        nonlocal pos
        n=struct.unpack_from('<H',data,pos)[0];pos+=2
        v=data[pos:pos+n].decode();pos+=n;return v
    types=[string() for _ in range(integer())]
    groups_count=integer();header=data[:pos];groups=[]
    for _ in range(groups_count):
        name=string();rows=[]
        for _ in range(integer()):
            start=pos;kind=struct.unpack_from('<H',data,pos)[0];pos+=42
            strings=[string() for _ in range(4)];pos+=6
            if kind!=5:raise ValueError('Unexpected battle entrance record version')
            rows.append(dict(raw=data[start:pos],strings=strings,coordinates=struct.unpack_from('<6i',data,start+2)))
        groups.append((name,rows))
    return header,groups,data[pos:]

def enc(value):
    b=value.encode();return struct.pack('<H',len(b))+b
