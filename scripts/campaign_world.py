"""Single source of truth for the expanded campaign canvas and every asset frame.

Mesh Z is pre-shader ("raw"); world Z is stretched by ZS. The canvas size and
padding come from the design file, never from literals in individual scripts.
Import the names below instead of writing 1116/798/192/384/1536/... again.

Frames (odd-q hex grid, r grows north; native = original 892x702 game map):
  hex       logical cell (q, r)
  raw       land/sea meshes, LF and tile grids: x = q*DX, z = (r + odd(q)/2)*DX
            is a hex's anchor, the SW corner of its tile block (see tile_hex)
  world     shader/prefab space: same x, z = raw_z * ZS. Prefab root matrices,
            river model pivots and height-patch bounds, g_world_bounds and the
            water-colour UVs live here.
  tile      half-hex placement grid (tile_list.bin, tile_map.tiles):
            tx = 2q, ty = 2r + odd(q); grid TILE_W x TILE_H
  lf_px     lf_height_map.dds, LF_PX_PER_HEX per hex; DDS file rows are north-first,
            code flips them ([::-1]) to work south-up
  sea_px    lf_sea_height_map.dds, SEA_PX_PER_HEX per hex
  camera_px camera_heightmap.png, CAMERA_PX_PER_HEX per hex, PNG rows north-first
  lookup_px region lookup, strategic overlay: LOOKUP_PX_PER_HEX per hex
ASSET_FRAMES records which frame each asset family uses; keep it current when a
new asset family is transformed.
"""
import json
import math
import os
from pathlib import Path

# Canvas rasters reach 16384 x 16380 (2048-column canvas), beyond PIL's default
# decompression-bomb limit; every pipeline script imports this module first.
try:
    from PIL import Image as _Image
    _Image.MAX_IMAGE_PIXELS = None
except ImportError:
    pass

_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = Path(os.environ.get('TW3K_WORKSPACE', _ROOT / 'work/all-directions-30'))
DESIGN_PATH = Path(os.environ.get('TW3K_DESIGN', WORKSPACE / 'work/current-region-design.json'))

# The pure vanilla 3k_dlc07_main_map logical map (CAIME's export of the shipped
# game). work/original-map/map.hex already carries one early mod region (848
# hexes of tw3k_exp_wudu_west_lumber) and the assembly-kit copy was replaced by
# the map of an earlier experiment, so neither is a vanilla reference.
VANILLA_MAP = _ROOT / 'tools/caime-1.1.0/lib/app/Templates/3k_dlc07_main_map/map.hex'
# Only the 190 campaign is supported, and it renders 3k_dlc07_main_map; every
# terrain stage processes and ships just this variant. 3k_main_map stays vanilla.
TERRAIN_VARIANTS = ('3k_dlc07_main_map',)
# dlc07 has no tile_map of its own in vanilla; the vanilla 3k_main_map cache (with
# its climate_map/tile_list/LF) is the native donor for the dlc07 display cache.
CACHE_DONOR_VARIANT = '3k_main_map'
# dlc07 has no sea_colour.dds; its water materials read the 3k_main_map and
# 3k_dlc06_main_map copies, so these are the only paths outside the dlc07 folders
# that the release still overrides.
SHARED_TERRAIN_OUTPUTS = ('terrain/campaigns/3k_main_map/sea_colour.dds',
                          'terrain/campaigns/3k_dlc06_main_map/sea_colour.dds')
# The campaign lighting groups the 190 campaign uses (snow_bake.py writes them: the 3k_main_map groups with the
# snow post-process in every season; translate_campaign_environment_spheres.py points environment_collection here).
VANILLA_LIGHTING_DIR = 'terrain/campaigns/3k_main_map/lighting'
SNOW_LIGHTING_DIR = 'terrain/campaigns/3k_dlc07_main_map/tw3k_lighting'


def snow_bake_on():
    """TW3K_SNOW_BAKE=1 turns the snow bake on (default off: one change per test pack). Off, environment_collection
    keeps the vanilla lighting references and the stage carries no snow products (snow_bake removes its own)."""
    return os.environ.get('TW3K_SNOW_BAKE', '0') == '1'



def arctic_treeline_on():
    """TW3K_ARCTIC_TREELINE=1 turns the arctic treeline on (polish MATERIALS ECOTONE: ecotone_bands.tundra_forest, tundra
    out of the USR3 open set; §9.3). Default off: tundra stays in the USR3 open set and the classes are byte-identical
    to the earlier main (one change per test pack; the coordinator switches it on for its pack)."""
    return os.environ.get('TW3K_ARCTIC_TREELINE', '0') == '1'


def ships_terrain_path(rel):
    """Whether a stage path under terrain/campaigns or campaign_maps belongs in a pack."""
    if not rel.startswith(('terrain/campaigns/', 'campaign_maps/')):
        return True
    parts = rel.split('/')
    variant = parts[2] if parts[0] == 'terrain' else parts[1]
    return variant in TERRAIN_VARIANTS or rel in SHARED_TERRAIN_OUTPUTS

# Native (vanilla) campaign grid and world scale.
NATIVE_W, NATIVE_H = 892, 702
NATIVE_WORLD_WIDTH = 595.1
# Vanilla settlement/character display spacing (world width / (columns - 1)).
NATIVE_DX = NATIVE_WORLD_WIDTH / (NATIVE_W - 1)
ZS = 2 / math.sqrt(3)
# The engine renders terrain on a tile grid of world width / tile columns per
# cell (vanilla: 595.1 / 1784) and derives world height, height-map scale and
# vertical scale from it. ENGINE_Z_RATIO is its raw -> world z stretch for the
# tile rows, taken from vanilla's g_world_bounds (541.2067 over 1405 rows); the
# same formula reproduces the replayed expanded bounds to 1e-5.
VANILLA_TILE_CELL = 595.0999755859375 / 1784
ENGINE_Z_RATIO = 541.2066650390625 / (1405 * VANILLA_TILE_CELL)


def _load_canvas():
    design = json.loads(DESIGN_PATH.read_text(encoding='utf-8'))
    width, height = (int(v) for v in design['grid'])
    left, bottom, right, top = (int(v) for v in design['padding'])
    if (left + NATIVE_W + right, bottom + NATIVE_H + top) != (width, height):
        raise ValueError(f'Design padding {design["padding"]} does not add up to grid {design["grid"]}')
    if tuple(design.get('old_grid_offset', (left, bottom))) != (left, bottom):
        raise ValueError('Design old_grid_offset disagrees with padding')
    if left % 2:
        # Odd-q stagger is decided by column parity; an odd western padding
        # would shift every native asset by half a row.
        raise ValueError(f'Western padding {left} must be even to preserve odd-q hex parity')
    return width, height, left, bottom, right, top, bool(design.get('tile_aligned_frame', True))


W, H, PAD_LEFT, PAD_BOTTOM, PAD_RIGHT, PAD_TOP, TILE_ALIGNED_FRAME = _load_canvas()

# Terrain frame. A hex is two tile cells wide; DX is the terrain hex pitch every
# raster <-> world conversion uses (x / DX * 2 = tile column). With the
# tile-aligned frame (default) the cell is vanilla's, so the world is W * DX
# wide, the native map moves by whole vanilla cells and the engine's derived
# world height and height scales equal vanilla's. The legacy frame (design
# 'tile_aligned_frame': false, diagnostics only) used vanilla's display spacing
# as DX, which put every engine scale 0.0225% off vanilla and opened slits.
if TILE_ALIGNED_FRAME:
    DX = 2 * VANILLA_TILE_CELL
    WORLD_WIDTH = W * DX
    WORLD_HEIGHT = (2 * H + 1) * VANILLA_TILE_CELL * ENGINE_Z_RATIO
else:
    DX = NATIVE_DX
    WORLD_WIDTH = (W - 1) * DX
    WORLD_HEIGHT = (H + .5) * DX * ZS
DZ = DX * ZS
RAW_MESH_HEIGHT = (H + .5) * DX
# Engine display spacing of logical hexes (settlements, characters, cameras on
# a town): world width / (columns - 1). It is DX * W / (W - 1), so a town shows
# up to half a hex from its tile block centre at the map edges, as in vanilla.
HEX_DX = WORLD_WIDTH / (W - 1)
HEX_DZ = HEX_DX * ZS

if TILE_ALIGNED_FRAME:
    # Slit guard: the engine derives its tile cell, LF texel size and height
    # scales from world width / tile columns. Any drift from vanilla's cell
    # reopens the vertical slits (live-check-padded-diagnostics), and native
    # content must move by whole vanilla cells.
    assert abs(WORLD_WIDTH / (2 * W) - VANILLA_TILE_CELL) < 1e-12, 'world width is not tile-aligned'
    assert abs(WORLD_HEIGHT / (2 * H + 1) - VANILLA_TILE_CELL * ENGINE_Z_RATIO) < 1e-12, 'world height is not tile-aligned'

# Campaign camera limits (campaign_camera_map_bounds): vanilla keeps the camera
# target 45/55/140/46 units inside the west/south/east/north world edges
# (bounds 45..455 x 55..495 on a 595.1 x 541.2 world) so the view never shows
# past the terrain. Vanilla's wide eastern margin covers open sea; authored
# towns stand further east (Wa Nu at x 828 on the 1400 canvas, unreachable
# with it: live-check-canvas-1400), so the east takes the western margin.
VANILLA_WORLD = (595.0999755859375, 541.2066650390625)
VANILLA_CAMERA_BOUNDS = {'min_x': 45, 'min_y': 55, 'max_x': 455, 'max_y': 495}
CAMERA_BOUNDS = {'min_x': VANILLA_CAMERA_BOUNDS['min_x'], 'min_y': VANILLA_CAMERA_BOUNDS['min_y'],
                 'max_x': int(WORLD_WIDTH - VANILLA_CAMERA_BOUNDS['min_x']),
                 'max_y': int(WORLD_HEIGHT - (VANILLA_WORLD[1] - VANILLA_CAMERA_BOUNDS['max_y']))}

# Native canvas placement inside the expanded grid, in every unit.
NATIVE_OFFSET_HEX = (PAD_LEFT, PAD_BOTTOM)
NATIVE_HEX_SLICE = (slice(PAD_BOTTOM, PAD_BOTTOM + NATIVE_H), slice(PAD_LEFT, PAD_LEFT + NATIVE_W))

TILE_PER_HEX = 2
TILE_W, TILE_H = W * TILE_PER_HEX, H * TILE_PER_HEX + 1
NATIVE_TILE_W, NATIVE_TILE_H = NATIVE_W * TILE_PER_HEX, NATIVE_H * TILE_PER_HEX + 1
NATIVE_OFFSET_TILE = (PAD_LEFT * TILE_PER_HEX, PAD_BOTTOM * TILE_PER_HEX)
# Native tile/cache rectangle inside the expanded tile grid (rows south-up).
NATIVE_TILE_X0, NATIVE_TILE_Y0 = NATIVE_OFFSET_TILE
NATIVE_TILE_X1, NATIVE_TILE_Y1 = NATIVE_TILE_X0 + NATIVE_TILE_W, NATIVE_TILE_Y0 + NATIVE_TILE_H
NATIVE_TILE_SLICE = (slice(NATIVE_TILE_Y0, NATIVE_TILE_Y1), slice(NATIVE_TILE_X0, NATIVE_TILE_X1))

# Native content (meshes, props, trees, cameras, spheres) moves by the padding.
# In the tile-aligned frame that is whole vanilla tile cells, so it lands on the
# same tile/LF texels as in vanilla.
NATIVE_SHIFT_RAW = (PAD_LEFT * DX, PAD_BOTTOM * DX)
if TILE_ALIGNED_FRAME:
    NATIVE_SHIFT_WORLD = (PAD_LEFT * DX, NATIVE_TILE_Y0 * VANILLA_TILE_CELL * ENGINE_Z_RATIO)
    for _shift, _cells in ((NATIVE_SHIFT_RAW[0], NATIVE_TILE_X0), (NATIVE_SHIFT_RAW[1], NATIVE_TILE_Y0)):
        assert abs(_shift - _cells * VANILLA_TILE_CELL) < 1e-9, 'native shift is not whole vanilla tile cells'
else:
    NATIVE_SHIFT_WORLD = (PAD_LEFT * DX, PAD_BOTTOM * DZ)


def native_cache_owner(parent):
    """Re-encode a native tile_map child pointer (-index-1) for the expanded grid."""
    return -((parent // NATIVE_TILE_W + NATIVE_TILE_Y0) * TILE_W + parent % NATIVE_TILE_W + NATIVE_TILE_X0) - 1

LF_PX_PER_HEX = 8
LF_W, LF_H = W * LF_PX_PER_HEX, int((H + .5) * LF_PX_PER_HEX)
# D3D11 caps a texture side at 16384; the LF height map and global blend are
# LF_PX_PER_HEX texels per hex, so the canvas tops out at 2048 x ~2047 hexes.
MAX_TEXTURE_SIDE = 16384
if max(LF_W, LF_H) > MAX_TEXTURE_SIDE:
    raise ValueError(f'Canvas {W}x{H} needs a {LF_W}x{LF_H} height map; D3D11 allows {MAX_TEXTURE_SIDE} per side')
NATIVE_LF_W, NATIVE_LF_H = NATIVE_W * LF_PX_PER_HEX, int((NATIVE_H + .5) * LF_PX_PER_HEX)
NATIVE_OFFSET_LF = (PAD_LEFT * LF_PX_PER_HEX, PAD_BOTTOM * LF_PX_PER_HEX)
# Same placement in DDS file row order (north-first): rows start below the top padding.
NATIVE_OFFSET_LF_FILE = (PAD_LEFT * LF_PX_PER_HEX, PAD_TOP * LF_PX_PER_HEX)

SEA_PX_PER_HEX = 4
NATIVE_OFFSET_SEA = (PAD_LEFT * SEA_PX_PER_HEX, PAD_BOTTOM * SEA_PX_PER_HEX)

CAMERA_PX_PER_HEX = 2
# camera_heightmap.png shares the tile grid size; PNG rows are north-first.
NATIVE_OFFSET_CAMERA_FILE = (PAD_LEFT * CAMERA_PX_PER_HEX, PAD_TOP * CAMERA_PX_PER_HEX)
NATIVE_CAMERA_FILE_SLICE = (slice(NATIVE_OFFSET_CAMERA_FILE[1], NATIVE_OFFSET_CAMERA_FILE[1] + NATIVE_H * 2 + 1),
                            slice(NATIVE_OFFSET_CAMERA_FILE[0], NATIVE_OFFSET_CAMERA_FILE[0] + NATIVE_W * 2))
LOOKUP_PX_PER_HEX = 3840 / NATIVE_W

# Runtime height encoding of the LF map (normalised u16 -> polish height units).
LEVEL = 1.2026508780887113
NORM = .30209811551079574
GAIN = 14.3

# Placeholder coordinates the game uses for characters that are not really on
# the map. Never translate these.
CHARACTER_PLACEHOLDERS = {(0, 0), (1, 1)}

ASSET_FRAMES = {
    'land_mesh / sea_mesh vertices': 'raw',
    'land_mesh .compressed_map header box': 'raw',
    'lf_height_map.dds / lf_sea_height_map.dds': 'lf_px / sea_px, north-first file rows (flipped to south-up in code)',
    'tile_list.bin / tile_map.tiles': 'tile; tile_map child cells store -(row*TILE_W+col)-1',
    'camera_heightmap.png': 'camera_px, north-first rows, height_scale text chunk',
    'global_props root matrices (bmd_objects.bin)': 'world',
    'models/river_*.rigid_model_v2 (mesh+24 box, mesh+628 pivot)': 'world; engine uses the pivot, not the root matrix',
    'height_patches/rivers.height_patch_collection bounds': 'world',
    'sea_colour.dds / water shader UVs': 'world / g_world_bounds',
    'region lookup / 3k_overlay_map / minimap': 'lookup_px, north-first rows',
    'start_pos_characters startx/starty': 'hex; CHARACTER_PLACEHOLDERS untouched',
    'battle_locations_map.bin entrances': 'hex with y flipped (H - y)',
}


def assert_grid(width, height):
    if (width, height) != (W, H):
        raise ValueError(f'Unsupported theatre grid {(width, height)}, expected {(W, H)}')


def assert_native_shape(shape, kind):
    """Raise if a native asset does not have the expected vanilla dimensions."""
    expected = {'lf': (NATIVE_LF_H, NATIVE_LF_W), 'tile': (NATIVE_TILE_H, NATIVE_TILE_W)}[kind]
    if tuple(shape[:2]) != expected:
        raise ValueError(f'Native {kind} shape {shape[:2]} != {expected}')


def load_canvas_array(path, grid='hex'):
    """np.load an authored canvas-sized array and refuse stale shapes.

    grid='hex' expects (H, W); grid='tile' expects (TILE_H, TILE_W). Arrays saved
    on an older canvas must be migrated, not broadcast or cropped.
    """
    import numpy as np
    array = np.load(path)
    expected = {'hex': (H, W), 'tile': (TILE_H, TILE_W)}[grid]
    if array.shape[:2] != expected:
        raise ValueError(f'{path} has shape {array.shape[:2]}, current {grid} canvas is {expected}')
    return array


AUTHORED_COORDINATES_PATH = DESIGN_PATH.with_name('canvas-authored-coordinates.json')
# Per-design gameplay traits (province classes, rice / walled provinces, reviewed resources, historical town sites):
# beside the design, so a design candidate (TW3K_DESIGN) carries its own. Read by region_gameplay_db and
# check_town_positions; no key of a design is written in their code.
REGION_TRAITS_PATH = DESIGN_PATH.with_name('region-traits.json')


def region_traits():
    if not REGION_TRAITS_PATH.exists():
        raise FileNotFoundError(f'{REGION_TRAITS_PATH}: a design needs its region-traits.json beside it (copy it with '
                                'current-region-design.json into a workspace or design candidate)')
    return json.loads(REGION_TRAITS_PATH.read_text(encoding='utf-8'))


def design_nested_mismatch(design):
    """design_single_source (2026-11-01): the region records live only in the top-level design['regions']; each
    provinces[].regions entry is the bare key {'region': <key>} of a top-level region whose province_key is that group's
    province, and every top-level region is in exactly one group. Returns the problems ('<key>: body' for an entry with
    other fields, 'nested only <key>', 'top only <key>', '<key>: twice', '<key>: province_key'); empty = one source.
    The nested copies had drifted (139 regions in the 10-30 design) and the c4 generator built its list from them."""
    if not design.get('provinces'):
        return []
    top = {r['region']: r for r in design.get('regions', [])}
    out, seen = [], set()
    for g in design['provinces']:
        for r in g.get('regions', []):
            k = r.get('region')
            if set(r) != {'region'}:
                out.append(f'{k}: body')
            if k in seen:
                out.append(f'{k}: twice')
            seen.add(k)
            if k not in top:
                out.append(f'nested only {k}')
            elif top[k].get('province_key') != g['province']['key']:
                out.append(f'{k}: province_key')
    out += [f'top only {k}' for k in sorted(set(top) - seen)]
    return out


def design_capital_problems(design):
    """design_capitals (2026-11-01): is_capital marks a province capital, not a city: every authored province
    (tw3k_*) has exactly one capital region, a region in a vanilla province (3k_*) is never one, and a capital's key
    ends in _capital. (Cities that are no province capital keep is_capital false and a city stamp: Nanjiang and Ailao in
    the vanilla provinces Zangke and Yongchang, single-source audit 14.)"""
    out, caps = [], {}
    for r in design.get('regions', []):
        caps.setdefault(r.get('province_key', ''), []).append(r)
        if r.get('is_capital') and not r['region'].endswith('_capital'):
            out.append(f"{r['region']}: capital without the _capital key")
    for pk, rs in sorted(caps.items()):
        n = sum(bool(r.get('is_capital')) for r in rs)
        if pk.startswith('tw3k_') and n != 1:
            out.append(f'{pk}: {n} capitals')
        if not pk.startswith('tw3k_') and n:
            out.append(f'{pk}: vanilla province with {n} authored capitals')
    return out


def design_map_problems(design, m, vanilla):
    """design_matches_map (2026-11-01): the authored seas the design lists (imported_sea_regions, plus new_sea_regions
    where present) are exactly the map's non-vanilla sea regions (prepare_province builds one regions row per listed
    sea: c5p listed 4 removed seas and missed 3 new ones), and sdk_region_budget.total_regions is the map header's
    land + sea count. m, vanilla: map_hex.read maps."""
    listed = {s['region'] for s in design.get('imported_sea_regions', []) + design.get('new_sea_regions', [])}
    on_map = set(m['sea_regions']) - set(vanilla['sea_regions'])
    out = [f'listed sea not on the map: {k}' for k in sorted(listed - on_map)]
    out += [f'map sea not listed: {k}' for k in sorted(on_map - listed)]
    total = design.get('sdk_region_budget', {}).get('total_regions')
    n = len(m['land_regions']) + len(m['sea_regions'])
    if total is not None and total != n:
        out.append(f'sdk_region_budget.total_regions {total} != map {n}')
    return out


def design_keys_only(design):
    """The design with provinces[].regions reduced to bare keys (design_single_source); the top level unchanged."""
    import copy
    d = copy.deepcopy(design)
    for g in d.get('provinces', []):
        g['regions'] = [{'region': r['region']} for r in g.get('regions', [])]
    return d


def load_authored_coordinates():
    """Relief lines, valleys, biome bands and check points in current-canvas hexes.

    The file carries the canvas it was written for; scripts/migrate_canvas.py
    shifts it together with the design when the padding changes.
    """
    data = json.loads(AUTHORED_COORDINATES_PATH.read_text(encoding='utf-8'))
    require_canvas(data['canvas']['grid'], data['canvas']['padding'], str(AUTHORED_COORDINATES_PATH))
    return data


def relief_features(data, coast_revision):
    """(name, points, height, width) tuples for the given coast coverage revision."""
    result = []
    for e in data['relief_features']:
        if e.get('unless_coast_revision') == coast_revision:
            continue
        points = e.get('points_by_coast_revision', {}).get(str(coast_revision), e['points'])
        result.append((e['name'], [tuple(p) for p in points], e['height'], e['width']))
    return result


def canvas_stamp():
    stamp = {'grid': [W, H], 'padding': [PAD_LEFT, PAD_BOTTOM, PAD_RIGHT, PAD_TOP]}
    if TILE_ALIGNED_FRAME:
        # Snapshots hold world positions; legacy-frame ones need
        # migrate_world_frame.py before this frame may read them.
        stamp['frame'] = 'tile-aligned'
    return stamp


def stamp_snapshot(directory):
    """Record which canvas a snapshot/backup directory was taken on."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / '.canvas.json').write_text(json.dumps(canvas_stamp()), encoding='utf-8')


def check_snapshot(directory):
    """Refuse to read a first-run snapshot that belongs to another canvas.

    Unstamped directories that already hold files predate this guard and are
    adopted for the current canvas once; after a canvas change they must be
    refreshed (geometry export) or deleted, never silently reused.
    """
    directory = Path(directory)
    stamp = directory / '.canvas.json'
    if stamp.exists():
        found = json.loads(stamp.read_text(encoding='utf-8'))
        if found != canvas_stamp():
            raise RuntimeError(f'{directory} was captured on canvas {found}, current is {canvas_stamp()}; '
                               'refresh or remove this snapshot before rebuilding')
    elif directory.exists() and any(directory.iterdir()):
        stamp.write_text(json.dumps(canvas_stamp()), encoding='utf-8')


def seed_from_stage_allowed(directory):
    """Whether a first-run snapshot may be filled from the current stage.

    Only into a snapshot already stamped for this canvas. A missing snapshot
    (e.g. quarantined by migrate_canvas.py) must not be re-seeded from a stage
    that still holds the previous canvas; set TW3K_SEED_BASE_FROM_STAGE=1 only
    when the stage is known to be on the current canvas (a first build).
    """
    directory = Path(directory)
    if (directory / '.canvas.json').exists():
        check_snapshot(directory)
        return True
    if os.environ.get('TW3K_SEED_BASE_FROM_STAGE'):
        return True
    raise RuntimeError(f'{directory} is missing or unstamped; after a canvas change rebuild it for the '
                       'new canvas (canvas snapshot migration) instead of seeding it from the old stage. '
                       'For a first build on a fresh stage set TW3K_SEED_BASE_FROM_STAGE=1.')


def require_canvas(grid, padding, what):
    """Guard data written in absolute canvas coordinates (e.g. authored relief lines).

    Such literals are only valid for the canvas they were authored on; when the
    canvas changes they must be migrated by the padding delta, not silently reused.
    """
    if (tuple(grid), tuple(padding)) != ((W, H), (PAD_LEFT, PAD_BOTTOM, PAD_RIGHT, PAD_TOP)):
        raise RuntimeError(f'{what} were authored for grid {grid} padding {padding}; '
                           f'current canvas is {(W, H)} padding {(PAD_LEFT, PAD_BOTTOM, PAD_RIGHT, PAD_TOP)}. '
                           'Migrate these coordinates before rebuilding.')


# Least-squares fit of the vanilla capitals: canvas hex = f(lon, lat), fitted with the native map at padding
# (658, 230); lonlat_fit follows the current padding. The one copy of the fit (plan_frontier_regions.fit,
# check_town_positions.fit use it).
LONLAT_FIT_Q = (28.54, -0.85, -2110)
LONLAT_FIT_R = (-0.89, 25.72, -86)
LONLAT_FIT_PADDING = (658, 230)


def lonlat_fit(lon, lat):
    """(q, r) of the fit; the same arithmetic, in the same order, as the copies it replaced."""
    (a, b, c), (d, e, f) = LONLAT_FIT_Q, LONLAT_FIT_R
    return (a * lon + b * lat + c + PAD_LEFT - LONLAT_FIT_PADDING[0],
            d * lon + e * lat + f + PAD_BOTTOM - LONLAT_FIT_PADDING[1])


def world_position(q, r):
    """Terrain-frame world position of hex (q, r)'s tile anchor: the SW corner of
    its tile block (tile (2q, 2r+odd(q)), LF texel (8q, 8r+4*odd(q))), not its
    centre. world_hex maps it back to (q, r); block_centre is the centre and
    display_position is where the engine draws towns and characters."""
    return q * DX, (r + (int(q) % 2) * .5) * DZ


def block_centre(q, r):
    """World-space centre of hex (q, r)'s tile block, ((q+.5)*DX, (r+odd/2+.5)*DZ);
    raw z is this z / ZS."""
    return (q + .5) * DX, (r + (int(q) % 2) * .5 + .5) * DZ


def display_position(q, r):
    """Where the engine shows a settlement or character standing on hex (q, r)."""
    return q * HEX_DX, (r + (int(q) % 2) * .5) * HEX_DZ


def raw_position(q, r):
    """As world_position in raw (mesh) coordinates: the tile anchor (SW corner of
    the hex's tile block), z unit DX."""
    return q * DX, (r + (int(q) % 2) * .5) * DX


# --- Lattice-safe conversions -------------------------------------------
# Native coarse vertices lie on the DX/4 lattice (within float32 noise ~5e-4 of
# it) in the tile-aligned frame, so they sit exactly on hex and tile borders.
# Every lookup goes through these helpers: float64, snapped within
# LATTICE_EPS of the lattice, ties always resolved the same way.
LATTICE_EPS = 1e-3


def _snap(t, eps=LATTICE_EPS):
    """t in lattice units (float64); values within eps of an integer become it."""
    import numpy as np
    t = np.asarray(t, np.float64)
    k = np.rint(t)
    return np.where(np.abs(t - k) <= eps, k, t)


def world_hex(x, z):
    """Canvas hex (q, r) owning world-space points: tile_hex of the tile holding
    them, i.e. world x in [q, q+1)*DX (the engine's and vanilla's tile, climate and
    display-position rule; the old centre-at-q*DX rule was half a hex off).
    A point on a block edge (native vertices on the DX/4 lattice) belongs to the
    block east/north of it, the block of the LF texel it sits on."""
    import numpy as np
    return tile_hex(np.floor(_snap(np.asarray(x, np.float64) / DX * 2)).astype(int),
                    np.floor(_snap(np.asarray(z, np.float64) / DZ * 2)).astype(int))


def raw_hex(x, z_raw):
    """As world_hex for raw (mesh) coordinates, whose z unit is DX."""
    import numpy as np
    return tile_hex(np.floor(_snap(np.asarray(x, np.float64) / DX * 2)).astype(int),
                    np.floor(_snap(np.asarray(z_raw, np.float64) / DX * 2)).astype(int))


def lattice_keys(xz):
    """Vectorised lattice_key for an (n, 2) array: int64 (n, 4) rows of
    (on-lattice flag, value) per axis, value = lattice index or 1e-4 grid."""
    import numpy as np
    xz = np.asarray(xz, np.float64)
    t = xz / (DX / 4)
    k = np.rint(t)
    on = np.abs(t - k) <= LATTICE_EPS
    value = np.where(on, k, np.rint(xz / 1e-4)).astype(np.int64)
    return np.column_stack((on[:, 0], value[:, 0], on[:, 1], value[:, 1])).astype(np.int64)


def tile_coord(v):
    """Continuous tile coordinate of world/raw x (or raw z), snapped to the
    half-tile lattice where native vertices sit, so bilinear samples there do
    not pick up a neighbour through float32 noise."""
    import numpy as np
    return _snap(np.asarray(v, np.float64) / DX * 4) / 2


def tile_hex(tx, ty):
    """Hex owning tile (tx, ty): tiles 2q and 2q+1, rows 2r+odd(q) and +1."""
    import numpy as np
    q = np.clip(np.asarray(tx) // 2, 0, W - 1)
    return q, np.clip((np.asarray(ty) - q % 2) // 2, 0, H - 1)


def lattice_key(x, z):
    """Hashable position key: lattice index for points on the DX/4 lattice,
    a 1e-4 grid otherwise (kept apart from lattice keys)."""
    out = []
    for v in (x, z):
        t = float(v) / (DX / 4)
        k = round(t)
        out.append(('L', k) if abs(t - k) <= LATTICE_EPS else ('P', round(float(v), 4)))
    return tuple(out)


# Hand-written check windows and preview sites were authored on the
# 1116x798 canvas (native map at hex 192, 32). These map them onto the
# current canvas so they follow the native map when it moves.
AUTHORED_ON_PADDING = (192, 32)


def from_authored_hex(q, r):
    return q + PAD_LEFT - AUTHORED_ON_PADDING[0], r + PAD_BOTTOM - AUTHORED_ON_PADDING[1]


def from_authored_tile(tx, ty):
    return (tx + TILE_PER_HEX * (PAD_LEFT - AUTHORED_ON_PADDING[0]),
            ty + TILE_PER_HEX * (PAD_BOTTOM - AUTHORED_ON_PADDING[1]))


def native_to_canvas_hex(q, r):
    return q + PAD_LEFT, r + PAD_BOTTOM
