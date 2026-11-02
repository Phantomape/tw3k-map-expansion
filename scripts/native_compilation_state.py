"""Mandatory source and artifact provenance shared by capture and packaging."""
import tw3k_paths
import hashlib
import json
import os
import struct
import tempfile
from pathlib import Path
from analyze_reference import digest

GAME_DATA = tw3k_paths.GAME_DATA
# Inputs the compile reads besides the compiler pack: compile mode lists only that pack
# (mod packs load only when listed, used_mods is emptied), so the game's own packs and
# executable are the rest. Their size and write time change with a game update.
COMPILER_MANIFEST_KEYS = ('source_map_sha256', 'source_design_sha256', 'source_map_data_sha256', 'compiler_sha256')


def game_data_fingerprint(data=GAME_DATA):
    """Name, size and write time of every non-mod pack in the game data folder and of the
    game executable (a Steam update changes them and forces a recompile)."""
    h = hashlib.sha256()
    for p in sorted(data.glob('*.pack'), key=lambda p: p.name.lower()):
        with p.open('rb') as f:
            head = f.read(16)
        if head[:3] == b'MFH':
            head = head[8:]
        if head[:4] == b'PFH5' and struct.unpack_from('<I', head, 4)[0] & 0xF == 3:
            continue   # mod pack: loaded only when listed
        st = p.stat()
        h.update(f'{p.name.lower()}|{st.st_size}|{st.st_mtime_ns}\n'.encode())
    exe = data.parent / 'Three_Kingdoms.exe'
    st = exe.stat()
    h.update(f'{exe.name.lower()}|{st.st_size}|{st.st_mtime_ns}\n'.encode())
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as out:
            json.dump(value, out, ensure_ascii=False, indent=2)
            out.write('\n')
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def state_path(area):
    for version in (5, 4, 3, 2):
        folder = area / ('layout-v' + str(version))
        if (folder / 'source-state.json').exists():
            return folder / 'native-generated-state.json'
    return area / 'analysis/native-generated-state.json'


def check_map_data_export(area, map_sha256=None):
    """Refuse a map_data.esf the SDK exported from another map.hex than the current one.

    map.hex and map_data.esf are hashed independently below, so a map edit without a new
    map-data export would compile the old logical map (regions, settlements, passability)
    under the new map's provenance. sdk_memory_export.py records the map it exported in
    analysis/sdk-memory-fix-result.json; recompiling cannot repair a stale export, the
    export must be redone. An area without that record (test fixtures) is not checked."""
    record = area / 'analysis/sdk-memory-fix-result.json'
    if not record.exists():
        return
    exported = json.loads(record.read_text(encoding='utf-8-sig')).get('map_sha256')
    current = map_sha256 or digest(area / 'work/final-map/map.hex')
    if exported != current:
        raise RuntimeError('map_data.esf was exported from map.hex ' + str(exported) + ', current map.hex is ' + current
                           + ': rerun the SDK map export (build_polished_campaign_terrain.ps1 -RebuildGeometry) before compiling')


def current_sources(area):
    manifest = json.loads((area / 'analysis/compiler-manifest.json').read_text(encoding='utf-8-sig'))
    current = {
        'source_map_sha256': digest(area / 'work/final-map/map.hex'),
        'source_design_sha256': digest(area / 'work/current-region-design.json'),
        'source_map_data_sha256': digest(area / 'work/standalone_kit/working_data/campaign_maps/3k_dlc07_main_map/map_data.esf'),
        'compiler_sha256': digest(area / 'compiler.pack'),
    }
    for name in COMPILER_MANIFEST_KEYS:
        recorded = manifest.get('sha256' if name == 'compiler_sha256' else name)
        if recorded != current[name]:
            raise RuntimeError('Compiler manifest does not match current ' + name)
    check_map_data_export(area, current['source_map_sha256'])
    current['game_data_fingerprint'] = game_data_fingerprint()
    return current


def verify_generated(area, modes=('romance', 'historical')):
    current = current_sources(area)
    path = state_path(area)
    if not path.exists():
        raise RuntimeError('Native compilation provenance is required: ' + str(path))
    state = json.loads(path.read_text(encoding='utf-8-sig'))
    for mode in modes:
        record = state.get('modes', {}).get(mode, {})
        for name, value in current.items():
            if record.get(name) != value:
                raise RuntimeError('Native ' + mode + ' does not match ' + name)
        if record.get('startpos_sha256') != digest(area / 'generated' / ('startpos_' + mode + '.esf')):
            raise RuntimeError('Captured ' + mode + ' startpos changed after compilation')
    ai = state.get('ai_map', {})
    if ai.get('producer_mode') not in ('romance', 'historical'):
        raise RuntimeError('HLP/SPD lack recorded native compilation provenance; recapture them')
    for name, value in current.items():
        if ai.get(name) != value:
            raise RuntimeError('Native AI map does not match ' + name)
    for name in ('hlp_data.esf', 'spd_data.esf'):
        if ai.get(name.replace('.esf', '_sha256')) != digest(area / 'generated' / name):
            raise RuntimeError('Captured ' + name + ' changed after compilation')
    return current
