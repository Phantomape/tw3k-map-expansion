"""Cell fields of map.hex and the tracked layout base (L0) of the map bake.

One table of the 16-byte cell layout (map_hex docstring; analysis/map-bake-inventory.md section 0), used by
extract_map_layout.py (writes data/map-layout/), bake_map.py (builds map.hex from vanilla + L0 + design + rules)
and verify_map_bake.py (field-by-field diff). A field is a list of (byte, mask) parts read as one unsigned
value, low part first: the value of 'region' is the region id + 1 as map.hex stores it (0 = no region), the
value of 'ground' the ground index + 1, 'climate' byte 7 as is ((index + 1) << 1), 'aoi' the AOI index + 1.

UNUSED holds the bits that are zero on every vanilla and every current cell (checked by the extraction and
by every bake): byte 0 bit 2, byte 2 bits 0-2, byte 6 bits 3-7, byte 8 bits 0-2, byte 9 bits 1-7, bytes 10-14.
"""
import json
import os
from pathlib import Path

import numpy as np

FIELDS = {
    'kind': ((0, 0x03),),                 # 0 land, 1 water, 2 beach, 3 cliff
    'region': ((0, 0xF8), (1, 0xFF)),     # region id + 1 (land list, then sea list)
    'impassable': ((2, 0x08),),
    'slot': ((2, 0xF0),),                 # settlement slot: 1 primary, 2 port
    'sprawl': ((3, 0x01),),
    'road': ((3, 0x7E),),                 # road edge mask (CAIME directions)
    'bridge': ((3, 0x80),),               # bridge / ford
    'river_edges': ((4, 0xFF), (5, 0x0F)),
    'ground': ((5, 0xF0), (6, 0x07)),     # ground index + 1 (land list, then sea list)
    'climate': ((7, 0xFF),),              # (climate index + 1) << 1
    'aoi': ((8, 0xF8), (9, 0x01)),        # AOI index + 1
    'border': ((15, 0xFF),),              # region-border side mask (map_hex.border_masks)
}
UNUSED = ((0, 0x04), (2, 0x07), (6, 0xF8), (8, 0x07), (9, 0xFE), (10, 0xFF), (11, 0xFF), (12, 0xFF), (13, 0xFF), (14, 0xFF))


def _shift(mask):
    return (mask & -mask).bit_length() - 1


def get(cells, name):
    """(H, W) int32 value of a field."""
    out = np.zeros(cells.shape[:2], np.int32); pos = 0
    for byte, mask in FIELDS[name]:
        part = (cells[..., byte].astype(np.int32) & mask) >> _shift(mask)
        out |= part << pos
        pos += bin(mask).count('1')
    return out


def put(cells, name, value, where=None):
    """Write a field (value: scalar or (H, W) array) into cells in place, on where (bool (H, W)) or everywhere."""
    value = np.broadcast_to(np.asarray(value, np.int64), cells.shape[:2])
    if where is None:
        where = np.ones(cells.shape[:2], bool)
    pos = 0
    for byte, mask in FIELDS[name]:
        width = bin(mask).count('1')
        part = ((value >> pos) & ((1 << width) - 1)) << _shift(mask)
        b = cells[..., byte]
        b[where] = ((b[where] & (0xFF ^ mask)) | part[where]).astype(np.uint8)
        pos += width


def unused_bits(cells):
    """Number of cells with any UNUSED bit set (0 on a valid map)."""
    bad = np.zeros(cells.shape[:2], bool)
    for byte, mask in UNUSED:
        bad |= (cells[..., byte] & mask) != 0
    return int(bad.sum())


def region_names(m):
    return list(m['land_regions']) + list(m['sea_regions'])


# ---------------------------------------------------------------- town stamps (add_ports cube geometry)
def shape_cells(town, shape):
    """Hexes of a no-port stamp round town: 'city' the 19-hex radius-2 hexagon, 'resource' the 7-hex flower."""
    from add_ports import FLOWER, HEXAGON, cadd, to_cube, to_hex
    T = to_cube(*town)
    return sorted(to_hex(cadd(T, o)) for o in (HEXAGON if shape == 'city' else FLOWER))


def stamp_shape(town, cells_set):
    """'city' / 'resource' when cells_set is exactly that footprint round town, else None."""
    for shape in ('city', 'resource'):
        if set(shape_cells(town, shape)) == set(cells_set):
            return shape
    return None


# ---------------------------------------------------------------- tracked files
LAYOUT_DIR = Path(__file__).resolve().parents[1] / 'data/map-layout'
LAYOUT_NPZ, MANIFEST, ROADS_NPZ, STAMPS, LAKES, GEODATA = ('layout.npz', 'manifest.json', 'roads.npz', 'stamps.json', 'lakes.json',
                                                         'geodata.json')
# Untracked inputs the bake's rule steps read (design section 3 item 5): (key, base, relative path, readers).
# base 'workspace' = campaign_world.WORKSPACE, 'root' = the repository (work/geodata is a junction to the main checkout's).
GEODATA_FILES = (
    ('west_real_terrain', 'workspace', 'analysis/west-real-terrain.npz', 'apply_climate.hex_geo (ground types, climate), apply_shore_kinds'),
    ('eco_grid', 'root', 'work/geodata/eco-grid-v2.npz', 'apply_climate.wetland_hexes'),
    ('strategic_ink', 'root', 'work/geodata/strategic-ink-2048x2047-888-375.npz', 'apply_forest_ground (forest biome)'),
    ('ne_rivers', 'root', 'work/geodata/ne_10m_rivers_lake_centerlines.zip', 'apply_rivers.raster_rivers'),
    ('etopo', 'root', 'work/geodata/ETOPO_2022_v1_60s_N90W180_surface.nc', 'apply_rivers (town reroutes, WR traces)'),
    ('vanilla_map', 'root', 'tools/caime-1.1.0/lib/app/Templates/3k_dlc07_main_map/map.hex', 'every step (vanilla slice, learned tables)'),
)


# Patch caches the bake itself writes (step 10, ensure_patch_caches, from the map it is building; P6 2026-11-01): not
# geodata. Patch.load keys them on the design, the map window, west-real-terrain and their rules; a geodata.json written
# before them may still list them, and the bake's geodata check skips them.
BAKE_WRITTEN = ('hehuang_patch', 'korea_patch', 'korea_patch_forests', 'wuyuan_patch')


def geodata_active(key):
    """False for the korea cache the switch does not read (TW3K_KOREA_FORESTS picks korea_patch_forests over
    korea_patch): bake_map skips its check, extract_map_layout records it only when present."""
    import os
    on = os.environ.get('TW3K_KOREA_FORESTS', '0') == '1'
    return {'korea_patch': not on, 'korea_patch_forests': on}.get(key, True)


def geodata_paths(workspace, root):
    return {k: (Path(workspace) if base == 'workspace' else Path(root)) / rel for k, base, rel, _ in GEODATA_FILES}


def sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def write_bytes_atomic(path, data):
    path = Path(path)
    tmp = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def write_json_atomic(path, doc):
    write_bytes_atomic(path, (json.dumps(doc, ensure_ascii=False, indent=1) + '\n').encode('utf-8'))


def savez_atomic(path, **arrays):
    """A compressed npz (np.load reads it) through a temporary file, byte-deterministic: members in name order,
    a fixed zip timestamp (np.savez_compressed stamps the current time), deflate level 9."""
    import io
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for k in sorted(arrays):
            a = io.BytesIO()
            np.lib.format.write_array(a, np.ascontiguousarray(arrays[k]), allow_pickle=False)
            info = zipfile.ZipInfo(f'{k}.npy', date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            z.writestr(info, a.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    write_bytes_atomic(path, buf.getvalue())


def load_npz(path):
    """All arrays of an npz, read whole and the file closed (never a memory map)."""
    with np.load(Path(path)) as z:
        return {k: np.array(z[k]) for k in z.files}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))
