"""Invoke the isolated official SDK's own table-conversion commands.

All inputs/outputs stay in this workspace. DLLs are used through their exported
constructors and command dispatcher, without changing their code or memory.
"""
import ctypes as C
import os
import sys
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / 'work/standalone_kit/binaries'
DLL_DIR = os.add_dll_directory(str(BIN))

def function(dll, name, result, *args):
    f = getattr(dll, name)
    f.restype = result
    f.argtypes = list(args)
    return f

def main():
    os.chdir(BIN)
    calibs=C.WinDLL(str(BIN/'calibs.modder.x64.dll'))
    utility=C.WinDLL(str(BIN/'utilitydll.modder.x64.dll'))
    builder=C.WinDLL(str(BIN/'tooldatabuilder.modder.x64.dll'))
    qt=C.WinDLL(str(BIN/'Qt5Core.dll'))
    shared=C.WinDLL(str(BIN/'qtshared.modder.x64.dll'))
    print('SDK libraries loaded',flush=True)
    application=C.create_string_buffer(256)
    argc=C.c_int(1);argv=(C.c_char_p*2)(b'tw3k-native-db',None)
    app_ctor=function(qt,'??0QCoreApplication@@QEAA@AEAHPEAPEADH@Z',C.c_void_p,C.c_void_p,C.c_void_p,C.c_void_p,C.c_int)
    app_ctor(application,C.byref(argc),argv,0x050f02)
    print('Qt initialized',flush=True)
    keep=[]
    def qstring(text=''):
        obj=C.create_string_buffer(8)
        f=function(qt,'?fromUtf8@QString@@SA?AV1@PEBDH@Z',C.c_void_p,C.c_void_p,C.c_char_p,C.c_int)
        b=text.encode('utf-8');f(obj,b,len(b));keep.append(obj);return obj
    def qtext(obj):
        f=function(qt,'?utf16@QString@@QEBAPEBGXZ',C.c_void_p,C.c_void_p)
        return C.wstring_at(f(obj))
    db=C.create_string_buffer(4096)
    ctor_db=function(shared,'??0TW_XML_QUERY_INTERFACE@@QEAA@PEAVPROGRESS_CALLBACK@@AEBVQString@@_N2@Z',C.c_void_p,C.c_void_p,C.c_void_p,C.c_void_p,C.c_bool,C.c_bool)
    ctor_db(db,None,qstring('tw3k-build'),True,True)
    print('XML interface constructed',flush=True)
    result=qstring()
    connect=function(shared,'?connect_without_merge@TW_XML_QUERY_INTERFACE@@UEAA?AVQString@@AEBV2@_N@Z',C.c_void_p,C.c_void_p,C.c_void_p,C.c_void_p,C.c_bool)
    connect(db,result,qstring((ROOT/'work/standalone_kit/raw_data/db').as_posix()),True)
    print('Database connect:',qtext(result),flush=True)
    connected=function(shared,'?is_connected@TW_XML_QUERY_INTERFACE@@UEBA_NXZ',C.c_bool,C.c_void_p)
    if not connected(db):raise RuntimeError('Official XML database did not connect')
    C.c_void_p.in_dll(builder,'?s_tdb_database@DATABASE_DATA@TOOLDATABUILDER@@2PEAVTW_XML_QUERY_INTERFACE@@EA').value=C.addressof(db)
    def register_probe_output():
        console=C.WinDLL(str(BIN/'empireconsole.modder.x64.dll'))
        spool=function(console,'?cli_out@CONSOLE@@YAAEAVSPOOL@WIN_CON@@XZ',C.c_void_p)()
        output=C.create_string_buffer(64)
        function(calibs,'??0UniString@CA@@QEAA@PEB_W@Z',C.c_void_p,C.c_void_p,C.c_wchar_p)(output,'tw3k-terrain-probe-output')
        keep.append(output)
        register=function(utility,'?register_output@UTILITYDLL@@YA_NAEBVUniString@CA@@AEAVUniCharOStream@UTILITYLIB@@@Z',C.c_bool,C.c_void_p,C.c_void_p)
        redirect=function(utility,'?redirect_output@UTILITYDLL@@YA_NAEBVUniString@CA@@@Z',C.c_bool,C.c_void_p)
        print('Native terrain probe output:',register(output,spool),redirect(output),flush=True)
    if '--sdk-path-context-probe' in sys.argv:
        get_vfs=function(utility,'?vfs@UTILITYDLL@@YAAEAUVFS@1@XZ',C.c_void_p)
        def vfs_reader_ready():
            address=get_vfs()
            return bool(address and C.c_void_p.from_address(address+8).value)
        print('Native VFS reader before loading BOB shared module:',vfs_reader_ready(),flush=True)
        bob_shared=C.WinDLL(str(BIN/'bob_shared.modder.x64.dll'))
        print('Native VFS reader after loading BOB shared module:',vfs_reader_ready(),flush=True)
        init_paths=function(bob_shared,'?initialise@PATHS@@CA_NAEBVQString@@@Z',C.c_bool,C.c_void_p)
        initialized=bool(init_paths(qstring((ROOT/'work/standalone_kit').as_posix())))
        print('Official BOB PATHS initialization returned:',initialized,flush=True)
        print('Native VFS reader after PATHS initialization:',vfs_reader_ready(),flush=True)
        if '--polish-terrain-probe' in sys.argv:
            if not initialized or not vfs_reader_ready():raise RuntimeError('BOB SDK file-access context is unavailable')
            register_probe_output()
            import native_polish_terrain_probe
            native_polish_terrain_probe.run(BIN)
        return
    if '--polish-terrain-probe' in sys.argv:
        register_probe_output()
        import native_polish_terrain_probe
        native_polish_terrain_probe.run(BIN)
        return
    if '--canvas-terrain-probe' in sys.argv:
        # Keep Qt and the official XML interface alive while testing the native
        # terrain entry point. Default database export behavior is unchanged.
        import probe_native_terrain_export
        return
    if '--core' in sys.argv:
        expansion=C.create_string_buffer(64)
        native_string=function(calibs,'??0String@CA@@QEAA@PEBD@Z',C.c_void_p,C.c_void_p,C.c_char_p)
        native_assign=function(calibs,'??4String@CA@@QEAAAEAV01@AEBV01@@Z',C.c_void_p,C.c_void_p,C.c_void_p)
        native_string(expansion,b'core')
        destination=C.addressof(C.c_char.in_dll(builder,'?s_current_expansion_for_export@DATABASE_DATA@TOOLDATABUILDER@@2VString@CA@@A'))
        native_assign(destination,expansion)
        print('Official export expansion: core',flush=True)
    string=C.create_string_buffer(64)
    ctor=function(calibs,'??0UniString@CA@@QEAA@PEB_W@Z',C.c_void_p,C.c_void_p,C.c_wchar_p)
    execute=function(utility,'?execute@UTILITYDLL@@YAXAEBVUniString@CA@@PEAVPROGRESS_CALLBACK@@@Z',None,C.c_void_p,C.c_void_p)
    console=C.WinDLL(str(BIN/'empireconsole.modder.x64.dll'))
    spool=function(console,'?cli_out@CONSOLE@@YAAEAVSPOOL@WIN_CON@@XZ',C.c_void_p)()
    output_name=C.create_string_buffer(64);ctor(output_name,'tw3k-native-output')
    register=function(utility,'?register_output@UTILITYDLL@@YA_NAEBVUniString@CA@@AEAVUniCharOStream@UTILITYLIB@@@Z',C.c_bool,C.c_void_p,C.c_void_p)
    redirect=function(utility,'?redirect_output@UTILITYDLL@@YA_NAEBVUniString@CA@@@Z',C.c_bool,C.c_void_p)
    print('Native output:',register(output_name,spool),redirect(output_name),flush=True)
    if '--all-db' in sys.argv:
        raw=(BIN/'tooldatabuilder.modder.x64.dll').read_bytes()
        known={s.decode('utf-16le') for s in re.findall(rb'(?:[\x20-\x7e]\x00){6,}',raw)}
        known.update(s.decode() for s in re.findall(rb'[\x20-\x7e]{6,}',raw))
        tables=[p.stem for p in (ROOT/'work/standalone_kit/raw_data/db').glob('*.xml') if 'add_'+p.stem+'_table' in known]
    elif '--all' in sys.argv:
        tables=[p.stem for p in (ROOT/'work/standalone_kit/raw_data/db').glob('start_pos_*.xml') if p.stem!='start_pos_family_relationship_types']
        tables+=['campaigns','regions','campaign_map_regions','campaign_map_settlements','region_to_province_junctions','campaign_settlement_display_settlement_rotations','campaign_settlement_display_settlement_layouts','provinces','season_province_effect_bundles','campaign_camera_map_bounds']
    else:tables=[sys.argv[1] if len(sys.argv)>1 else 'campaigns']
    for table in tables:
        source=ROOT/('work/native-split-xml' if '--split' in sys.argv else 'work/standalone_kit/raw_data/db')/f'{table}.xml'
        out=ROOT/('work/native-core-db' if '--core' in sys.argv else 'work/native-all-db' if '--all-db' in sys.argv else 'work/native-split-db' if '--split' in sys.argv else 'work/native-db')/f'{table}_tables'/'!tw3k_exp'
        out.parent.mkdir(parents=True,exist_ok=True)
        if out.exists():out.unlink()   # success must mean this run wrote it, not a stale export
        command=f'add_{table}_table "{source.as_posix()}" "{out.as_posix()}";'
        ctor(string,command)
        execute(string,None)
        if not out.exists():raise RuntimeError('Native export missing: '+table)
        print('Output:',table,out.stat().st_size,flush=True)
    print('Official native exports completed:',len(tables),flush=True)

if __name__=='__main__':main()
