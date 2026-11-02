"""grid(path): a logical map's read_hex summary and a cell(x, y) -> (region name, 16 cell bytes) lookup."""
from analyze_reference import read_hex
def grid(path):
 m=read_hex(path); raw=path.read_bytes(); data=raw[-4-m['width']*m['height']*16:-4]
 names=m['land_regions']+m['sea_regions']
 def cell(x,y):
  c=data[(y*m['width']+x)*16:(y*m['width']+x+1)*16]
  r=((c[1]<<5)|(c[0]>>3))-1
  return names[r] if r>=0 else '',c
 return m,cell
