"""Isolated call to the installed SDK's exported terrain builder (no patching)."""
import ctypes as C,os
from analyze_reference import ROOT
BIN=ROOT/'work/standalone_kit/binaries';EX=ROOT/'work/canvas-expansion'
out=EX/'native-terrain-probe';out.mkdir(exist_ok=True);os.chdir(out)
dll_dir=os.add_dll_directory(str(BIN))
calibs=C.WinDLL(str(BIN/'calibs.modder.x64.dll'))
builder=C.WinDLL(str(BIN/'tooldatabuilder.modder.x64.dll'))
constructor=getattr(calibs,'??0String@CA@@QEAA@PEBD@Z')
constructor.restype=C.c_void_p;constructor.argtypes=[C.c_void_p,C.c_char_p]
source=C.create_string_buffer(64);target=C.create_string_buffer(64)
constructor(source,str(EX/'work/release-input/terrain/campaigns/3k_main_map').encode())
constructor(target,str(out).encode())
process=getattr(builder,'?process_terrain_global_mesh@TOOLDATABUILDER@@YA_NAEBVString@CA@@0M@Z')
process.restype=C.c_bool;process.argtypes=[C.c_void_p,C.c_void_p,C.c_float]
print('Calling installed SDK terrain builder with isolated source and output',flush=True)
result=process(source,target,C.c_float(.5))
print('Native terrain result:',result,flush=True)
print('Output files:',len(list(out.rglob('*'))),flush=True)
