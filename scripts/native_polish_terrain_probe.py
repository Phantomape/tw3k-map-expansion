"""Bounded SDK terrain export with its native tile database present on disk."""
import tw3k_paths
import ctypes as C, os, struct, threading, time, json
from pathlib import Path
from analyze_reference import ROOT, inventory
import campaign_world as _frame
A=_frame.WORKSPACE; OUT=A/'native-polish-terrain'; ASSETS=OUT/'assets'
def run(bin_dir):
    ASSETS.mkdir(parents=True,exist_ok=True)
    pack=tw3k_paths.GAME_DATA / 'fast.pack'
    selected=[e for e in inventory(pack,False)['entries'] if e['path'].startswith('terrain/tiles/campaign/')]
    with pack.open('rb') as f:
        for e in selected:
            p=ASSETS/e['path'];p.parent.mkdir(parents=True,exist_ok=True)
            if not p.exists():f.seek(e['offset']);p.write_bytes(f.read(e['size']))
    assert (ASSETS/'terrain/tiles/campaign/_tile_database/_settings.bin').is_file()
    print('SDK tile database staged:',len(selected),'assets',flush=True)
    os.chdir(ASSETS)
    cal=C.WinDLL(str(bin_dir/'calibs.modder.x64.dll'));builder=C.WinDLL(str(bin_dir/'tooldatabuilder.modder.x64.dll'))
    ctor=getattr(cal,'??0String@CA@@QEAA@PEBD@Z');ctor.argtypes=[C.c_void_p,C.c_char_p];ctor.restype=C.c_void_p
    # VFS searches CA's registered source paths; changing the OS working
    # directory alone does not register a terrain asset root.
    paths=getattr(cal,'?search_paths@SearchPathCollection@CA@@SAAEAU12@XZ');paths.argtypes=[];paths.restype=C.c_void_p
    prepend=getattr(cal,'?prepend_path@SearchPathCollection@CA@@QEAAHAEBVString@2@@Z');prepend.argtypes=[C.c_void_p,C.c_void_p];prepend.restype=C.c_int
    root_string=C.create_string_buffer(64);ctor(root_string,str(ASSETS).encode());collection=paths();path_index=prepend(collection,root_string)
    settings_string=C.create_string_buffer(64);ctor(settings_string,b'terrain/tiles/campaign/_tile_database/_settings.bin')
    exists=getattr(cal,'?fexist@SearchPathCollection@CA@@QEBA_NAEBVString@2@@Z');exists.argtypes=[C.c_void_p,C.c_void_p];exists.restype=C.c_bool
    resolved=bool(exists(collection,settings_string));print('Native asset search path registered:',path_index,'tile database resolved:',resolved,flush=True)
    if not resolved:raise RuntimeError('Native SDK cannot resolve its tile database from the registered source path')
    utility=C.WinDLL(str(bin_dir/'utilitydll.modder.x64.dll'))
    file_ctor=getattr(cal,'??0FileName@CA@@QEAA@AEBVString@1@@Z');file_ctor.argtypes=[C.c_void_p,C.c_void_p];file_ctor.restype=C.c_void_p
    vfs_exists=getattr(utility,'?fexist_vfs@UTILITYDLL@@YA_NVFileName@CA@@@Z');vfs_exists.argtypes=[C.c_void_p];vfs_exists.restype=C.c_bool
    file_name=C.create_string_buffer(64);file_ctor(file_name,settings_string)
    vfs_relative=bool(vfs_exists(file_name))
    absolute_string=C.create_string_buffer(64);ctor(absolute_string,str(ASSETS/'terrain/tiles/campaign/_tile_database/_settings.bin').encode())
    absolute_name=C.create_string_buffer(64);file_ctor(absolute_name,absolute_string)
    vfs_absolute=bool(vfs_exists(absolute_name))
    print('Native VFS tile database existence: relative=',vfs_relative,'absolute=',vfs_absolute,flush=True)
    if not vfs_relative:raise RuntimeError('CA SearchPathCollection sees the tile database, but native VFS does not; terrain export is not attempted')
    if os.environ.get('TW3K_TILEMAP_AUDIT')=='1':
        # Use native readers in this isolated process to check whether the
        # processed list and its tile definitions are accepted before TQT export.
        warscape=C.WinDLL(str(bin_dir/'warscape.modder.x64.dll'))
        def exported(name,result,*args):
            fn=getattr(warscape,name);fn.restype=result;fn.argtypes=list(args);return fn
        database=C.create_string_buffer(32768);database_path=C.create_string_buffer(64)
        ctor(database_path,b'terrain/tiles/campaign/_tile_database/_settings.bin')
        exported('??0TILE_DATABASE@WARSCAPE@@QEAA@AEBVString@CA@@@Z',C.c_void_p,C.c_void_p,C.c_void_p)(database,database_path)
        tiles=exported('?tile_count@TILE_DATABASE@WARSCAPE@@QEBAIXZ',C.c_uint,C.c_void_p)(database)
        error=exported('?get_load_error@TILE_DATABASE@WARSCAPE@@QEBA_NXZ',C.c_bool,C.c_void_p)(database)
        print('Native tile database:',tiles,'definitions; load_error=',error,flush=True)
        if error or not tiles:raise RuntimeError('Native tile database did not load; no terrain export attempted')
        shared=(C.c_void_p*2)(C.addressof(database),None);tilemap=C.create_string_buffer(32768)
        exported('??0BATTLE_TILE_MAP@WARSCAPE@@QEAA@AEBV?$shared_ptr@$$CBVTILE_DATABASE@WARSCAPE@@@std@@_N@Z',C.c_void_p,C.c_void_p,C.c_void_p,C.c_bool)(tilemap,shared,True)
        source_path=C.create_string_buffer(64);ctor(source_path,((A/'work/release-input/terrain/campaigns/3k_dlc07_main_map').as_posix()+'/').encode())
        loaded=exported('?load@BATTLE_TILE_MAP@WARSCAPE@@QEAA_NAEBVString@CA@@@Z',C.c_bool,C.c_void_p,C.c_void_p)(tilemap,source_path)
        count=exported('?tile_count@BATTLE_TILE_MAP@WARSCAPE@@QEBAIXZ',C.c_uint,C.c_void_p)(tilemap)
        report={'database_definitions':tiles,'database_load_error':bool(error),'tile_list_loaded':bool(loaded),'native_tile_count':count}
        if os.environ.get('TW3K_NATIVE_READER')=='1':
            import struct
            values=[]
            for label,path in [('current',A/'work/release-input/terrain/campaigns/3k_dlc07_main_map'),('native',A/'terrain-repair/native-reader/reference/3k_dlc07_main_map')]:
                obj=C.create_string_buffer(32768)
                exported('??0BATTLE_TILE_MAP@WARSCAPE@@QEAA@AEBV?$shared_ptr@$$CBVTILE_DATABASE@WARSCAPE@@@std@@_N@Z',C.c_void_p,C.c_void_p,C.c_void_p,C.c_bool)(obj,shared,True)
                name=C.create_string_buffer(64);ctor(name,(path.as_posix()+'/').encode())
                ok=exported('?load@BATTLE_TILE_MAP@WARSCAPE@@QEAA_NAEBVString@CA@@@Z',C.c_bool,C.c_void_p,C.c_void_p)(obj,name)
                item={'label':label,'loaded':bool(ok)}
                for field in ('infield','map_area','low_frequency_world_position','low_frequency_world_scale','origin_offset'):
                    suffix='AABB_2i' if field in ('infield','map_area') else 'VECTOR_2'
                    function='?'+field+'@BATTLE_TILE_MAP@WARSCAPE@@QEBAAEBV'+suffix+'@UTILITYLIB@@XZ'
                    ptr=exported(function,C.c_void_p,C.c_void_p)(obj)
                    item[field]=struct.unpack('<4i' if suffix=='AABB_2i' else '<2f',C.string_at(ptr,16 if suffix=='AABB_2i' else 8))
                params=exported('?render_params@TILE_DATABASE@WARSCAPE@@QEBAAEBUTERRAIN_RENDER_PARAMS@2@XZ',C.c_void_p,C.c_void_p)(database)
                scale=C.c_float.from_address(params+16).value
                item['database_world_scale']=scale
                item['fallback_world_size']=[(item['infield'][axis+2]-item['infield'][axis]+1)*128*scale for axis in (0,1)]
                values.append(item)
            report['sampling_metadata']=values
            (A/'terrain-repair/native-reader/metadata.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        (OUT/'native-tile-reader-audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
        return
    # Match WARSCAPE_RENDER_WIDGET::initialise_thread_manager's public calls.
    # Geometry export submits task::Batch work and cannot run with a null scheduler.
    manager=getattr(utility,'?thread_manager@UTILITYDLL@@YAAEAVTHREAD_MANAGER@CA@@XZ');manager.argtypes=[];manager.restype=C.c_void_p
    initialize=getattr(utility,'?initialize@THREAD_MANAGER@CA@@QEAAXH_N@Z');initialize.argtypes=[C.c_void_p,C.c_int,C.c_bool];initialize.restype=None
    initialize(manager(),0,True)
    initialized=getattr(cal,'?is_initialized@query@task@@YA_NXZ');initialized.argtypes=[];initialized.restype=C.c_bool
    if not initialized():raise RuntimeError('Native terrain task scheduler did not initialize')
    workers=getattr(cal,'?number_of_worker_threads@query@task@@YAIXZ');workers.argtypes=[];workers.restype=C.c_uint
    print('Official terrain task scheduler initialized; workers=',workers(),flush=True)
    source=C.create_string_buffer(64);dest=C.create_string_buffer(64)
    constructor_source=A/'work/release-input/terrain/campaigns/3k_dlc07_main_map'
    output=OUT/'generated';output.mkdir(exist_ok=True)
    # Native BATTLE_TILE_MAP::load concatenates "tile_list.bin" directly.
    # Its input is a directory prefix and must retain the trailing separator.
    ctor(source,(constructor_source.as_posix()+'/').encode());ctor(dest,(output.as_posix()+'/').encode())
    process=getattr(builder,'?process_terrain_global_mesh@TOOLDATABUILDER@@YA_NAEBVString@CA@@0M@Z')
    process.argtypes=[C.c_void_p,C.c_void_p,C.c_float];process.restype=C.c_bool
    timeout=int(os.environ.get('TW3K_TERRAIN_TIMEOUT','100'));assert 1<=timeout<=600
    def guard():
        time.sleep(timeout);print('SDK terrain probe exceeded '+str(timeout)+' seconds',flush=True);os._exit(4)
    threading.Thread(target=guard,daemon=True).start()
    started=time.time();print('Native terrain export with disk assets started',flush=True)
    if os.environ.get('TW3K_SDK_CAPTURE_WAIT')=='10':time.sleep(10)
    ok=process(source,dest,.5)
    report={'returned':bool(ok),'elapsed_seconds':time.time()-started,'files':[str(p.relative_to(output)) for p in output.rglob('*') if p.is_file()]}
    (OUT/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
