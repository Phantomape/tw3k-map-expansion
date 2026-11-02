"""Terrain regression check: known-problem windows of a finished workspace, baseline vs candidate.

MERGE-TIME PROCEDURE (every output-changing terrain change; analysis/terrain-regression.md):
  1. Baseline: a snapshot of main's workspace after its last full pipeline run
       python terrain_regression.py snapshot --out <base-dir>
  2. Candidate: run the full pipeline (build_polished_campaign_terrain.ps1) on a throwaway
     workspace copy with TW3K_WORKSPACE set to it (or use the next pipeline run), then
       python terrain_regression.py snapshot --workspace <copy> --out <new-dir>
  3.   python terrain_regression.py compare <base-dir> <new-dir> [--out <dir>]
     and attach the printed table (also <dir>/compare.txt, compare.json, side-by-side PNGs of
     the flagged windows) to the merge. Exit 1 = regressions; explain or fix each before merging.
  At the end of the pipeline it can run warn-only: snapshot never fails on missing or broken
  inputs (the section is skipped with a note) and compare --warn-only always exits 0.

  python terrain_regression.py snapshot --out DIR [--workspace WS] [--strict]
  python terrain_regression.py compare BASE_DIR NEW_DIR [--out DIR] [--warn-only]
  python terrain_regression.py copy-inputs --to WS [--workspace SRC] [--wait S]
      copy just the files snapshot reads (~1.4 GB) into a new workspace dir, only while
      work/.pipeline-lock and -2 are absent (a lock appearing mid-copy aborts it)

Read-only on the workspace. Reads, with normal full reads (no mmap, no file kept open):
the stage LF (lf_height_map.dds), global_blend.dds, both tile lists, the stage land meshes,
the stage tree list, work/final-map/map.hex, analysis/river-water-level.npy, the newest
analysis/ground-seam-pipeline-<variant>-*/ scan, the coverage-hole scan and the run reports.
~15 s, 2.4 GB peak (2048 canvas); compare ~2 s. Frames: hex (q, r) on the current canvas (r grows north), tile t = LF
pixel 4t (final_terrain_surface), heights in polish units (LF and mesh y agree).

WINDOWS: a centre/radius part is the square hex box q +- R, r +- R; a window may hold several
parts (metrics are summed / maxed over them). Per window (all parts):
  seam.<class>     hexes of each ground-seam class (audit_ground_seams masks) + coverage holes
  lf_px_*          |dh| between neighbouring LF pixels (max, p99, count > 0.5 units)
  tile_step_*      |dh| between neighbouring land tile points (max, p99); shore_step_* the same
                   on land tiles within 2 tiles of water (stairs, shore walls)
  mesh_dev_*       stage coarse land mesh (top / bottom face) vs bilinear LF at quarter-hex
                   samples on land (max, p99, samples > 0.3); tri_edge_max_hex: longest xz edge of
                   a triangle touching the window (the 8-16 hex land_mesh_154 triangles);
                   mesh_uncovered_share: land samples without a stage triangle (native land is
                   covered by vanilla meshes from the packs, not read here)
  trees, wet_trees canvas trees of the stage tree list whose footprint ground lies below the
                   water plane or the local river level (polish_campaign_foliage's WET test)
  water_tiles, lake_iou (tile water by LF < max(plane, river level) vs map.hex lake hexes;
                   a proxy of audit_river_water_visibility's rendered IoU), below_plane_inland
                   (tiles under the plane on land / river hexes: flat 'diamond' water),
                   straight_coast_run_max / _runs8 (longest straight water edge, tiles)
  river_hexes, river_dry_centres (frontier river hex centres with LF >= water), fords, ford_groups
  towns            per design town in the window: LF median and range over its hex block, range
                   over the 3x3-hex plot; positions (a moved town is flagged)
  materials        land-tile shares at tile centres (green, forest4, woodland, farm, mud, sand, wet),
                   edges_per_hex / islands_per_1000 (woodland 4/5 vs open, blend pixels)
  sea_coast, beach sea_coast records of the local tile_list; beach hexes (map.hex kind 2) and those
                   whose centre LF is above BEACH_HIGH (a beach on a high shore)
  patch_*          ground patchiness of the stage blend (ground_patchiness.py; warn-only): hard edges per 100
                   land hexes, mean tone contrast, straight hard segments, dark islands, farm / wetland squares,
                   'lit' = above vanilla's alarm level; compare also lists them against the v18 numbers
  window extras    river_shape (bank raggedness, width CV, end narrowing), port (land, LF >= .4),
                   climate (climate cut at the native west border + native-edge-audit failures),
                   rows (water / dry centres in a row band), salt (0x7xx global records),
                   lake_level (a raised lake prop: Qinghai at hehuang-patch.json's water level)
GLOBAL: seam class totals of the final scan, seam-loop passes, repair / normalize / river-bed /
island / coverage-patch report counts, wet trees, dry river hex centres (per river), mud share of
the new regions, byte-15 border-mask mismatches, one-sided road edges, global-list 0x7xx bare
records, sea_coast records, beach hexes. BIOMES: per real biome of analysis/west-real-terrain.npz
(the RESOLVE classes polish reads), over new canvas land and over vanilla land (reference): share
of each global_blend material family and green / forest4 / woodland / farm / mud shares; gated on
the new land: green up in desert / alpine, forest up in paddy / grass, mud up, farmland down.
EXPECT holds the documented absolute rules (trees-in-river 0, ...).
"""
import argparse, collections, ctypes, json, os, re, shutil, struct, sys, time, traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VARIANT = '3k_dlc07_main_map'
STAGE = 'work/release-input'
PRE = f'{STAGE}/terrain/campaigns/{VARIANT}/'
TREES = f'{STAGE}/campaign_maps/{VARIANT}/display/trees/trees.campaign_tree_list'
INPUTS = ('work/current-region-design.json', 'work/canvas-authored-coordinates.json', 'work/final-map/map.hex',
          PRE + 'lf_height_map.dds', PRE + 'tile_list.bin', PRE + 'global_map/global_blend.dds',
          PRE + 'global_map/tile_list.bin', PRE + 'global_meshes/land_mesh_*.rigid_model_v2', TREES,
          'analysis/river-water-level.npy', 'analysis/river-water-level.json',
          f'analysis/ground-seam-pipeline-{VARIANT}-*/ground-seam-scan.json',
          f'analysis/ground-seam-pipeline-{VARIANT}-*/ground-seam-masks.npz',
          f'analysis/coverage-hole-scan-{VARIANT}/ground-seam-masks.npz',
          f'analysis/local-coarse-seam-repair-{VARIANT}.json', f'analysis/coarse-partition-height-normalization-{VARIANT}.json',
          'analysis/river-bed-mesh.json', 'analysis/island-ground-patches.json', 'analysis/terrain-native-foliage.json',
          'analysis/native-edge-audit.json', 'analysis/coast-tiles.json', 'analysis/salt-pan-windows.json', 'analysis/hehuang-patch.json',
          'analysis/west-real-terrain.npz', 'analysis/landform-tiles.npz',
          'analysis/pipeline-timing.log')

PLANE = -.015                 # campaign water plane (polish LAKE_WATER, foliage LAKE_PLANE)
WET_MARGIN = .065             # tree_bake WET_LEVEL - LAKE_PLANE (props bake dry predicate)
STRICT_CLASSES = ('lf_cliff', 'coarse_hole_native_removed', 'coarse_hole_new_land', 'coarse_hole_prefab_removed',
                  'coarse_stack', 'coarse_step', 'fine_uncovered', 'fine_range_stale')
NOISY_CLASSES = ('fine_uncovered_land', 'fine_climate_patch', 'fine_special_removed', 'steep_generic_authored')
GREEN = tuple(range(4, 8)) + (21, 22) + tuple(range(23, 31))   # audit_terrain_realism: cold/dark steppe/subtropical/temperate
FARM, MUD, SAND, WOODLAND, WETM = range(8, 13), 14, range(0, 4), (4, 5), 31
# global_blend material families (audit_terrain_realism) for the per-biome shares
FAMILIES = {'sand': range(0, 4), 'cold': range(4, 8), 'farm': range(8, 13), 'road': (13,), 'mud': (14,), 'other': range(15, 19),
            'steppe_light': (19, 20), 'steppe_dark': (21, 22), 'subtropical': range(23, 27), 'temperate': range(27, 31), 'wet': (31,)}
DESERT_CLEAR_HEX = 6           # desert_banned: hexes kept clear of water and settlement slots
BIOME_MIN_TILES = 500         # biomes with fewer land tiles on a side are not reported
DRY_BIOMES, PLAIN_BIOMES = ('desert', 'alpine'), ('paddy', 'grass')
BEACH_HIGH = 1.0              # beach hex centre above this (polish units): a beach on a high shore
BARE = re.compile(r'\\(generic|generic_sea|sea_coast|mountains_[^\\]+)\\')

# name, parts (q, r, R) or box (q0, q1, r0, r1), what to watch, extras. Hex coordinates on the 2048
# canvas (pad 888/375); camera world positions of the live-check plans converted with world_hex.
WINDOWS = (
    ('amur-banks', [(1524, 1221, 40)], 'bank raggedness, width CV, narrowing at the Hailar source (east end)', {'river_shape': 1}),
    ('funan-delta', [(1105, 268, 60)], 'gap north of the capital (rows 266-270), branches to the sea, diamond water, farm share', {'rows': (266, 270), 'gate': ('farm_share',)}),
    ('mekong-ford', [(1113, 292, 6)], 'ford passable (groups of 4), water at river-hex centres', {}),
    ('ganges-mouth', [(689, 608, 8)], 'mouth diamond (LF < -0.015 off the sea)', {}),
    ('irtysh-lakes', [(727, 1261, 10), (750, 1207, 10), (752, 1264, 10)], 'water vs lake outline (rendered IoU 0.99)', {}),
    ('bosten', [(640, 675, 1035, 1065)], 'lake IoU, shore stairs, fishing port (661,1050) on land LF>=0.4, plateaus of 南河城/尉犁/危须', {'port': (661, 1050)}),
    ('issyk-kul-shore', [(519, 1081, 8)], 'coarse_step on the shore', {}),
    ('lop-nur-shore', [(723, 1012, 12)], 'stepped shore, coarse holes', {}),
    ('m154-step-lineage', [(839, 1296, 6), (735, 1300, 6), (727, 1296, 6), (719, 1304, 6), (607, 1259, 6), (767, 1332, 6), (1109, 1397, 6)],
     'coarse_step and LF deviation of the 8-hex land_mesh_154 triangles', {}),
    ('native-west-climate', [(887, 674, 4), (887, 722, 4)], 'climate cut on the native west border (rebuilt tile_list)', {'climate': 1}),
    ('kyushu-nw-bay', [(1775, 1790, 768, 792)], 'right-angle coast on the native rectangle east edge', {}),
    ('kyushu-inland', [(1792, 790, 10)], 'inland cliffs / beach hexes re-marked as land (ddbf57b2)', {}),
    ('edge-cleanup', [(1250, 1250, 20), (1147, 1296, 20), (652, 1272, 20), (1700, 1724, 20)], 'woodland edges per hex and island density (vanilla ~0.25/0.02/3.8)', {'gate': ('edges_per_hex', 'islands_per_1000')}),
    ('turpan-green', [(702, 1085, 30)], 'green steppe/cold share (real geography: desert and alpine only)', {'gate': ('green_share',)}),
    ('dunhuang', [(810, 1000, 35)], 'green share, the four town positions, farms', {'gate': ('green_share',)}),
    ('bengal-forest', [(720, 575, 35)], 'forest material 4 share (40%, too high on paddy plains)', {'gate': ('forest4_share',)}),
    ('farm-ganges', [(736, 592, 45)], 'farmland share (must not fall back to 5-9%)', {'gate': ('farm_share',)}),
    ('salt-pans', [(659, 1022, 6), (841, 552, 6), (914, 908, 6)], 'Shanshan / Hanlin-Piaoyue / Xihai (Chaka) salt: black blocks (0x7xx records)', {'salt': 1}),
    ('hehuang-lake', [(930, 985, 870, 930)], 'Qinghai lake shore, lake towns (临羌, 破羌, 安夷), patch relief', {'lake_level': ('analysis/hehuang-patch.json', 'water')}),
    ('korea-patch', [(1620, 1700, 830, 950)], 'Korea patch ground stairs, town rims, coast walls (2ff35bf8)', {}),
    ('west-tarim-buffer', [(470, 1040, 20)], 'west-Tarim buffer lattice (2ff35bf8), ground', {}),
    ('bailongdui-shore', [(737, 1016, 10)], 'stepped steep shore (live 2048o)', {}),
    ('irrawaddy-mouth', [(821, 382, 10)], 'river mouth, dry river hexes (shallow channel)', {}),
    ('indus-mouth', [(344, 607, 10)], 'river mouth, dry river hexes (shallow channel)', {}),
    ('dayuan-wusun-plateaus', [(398, 1050, 8), (519, 1073, 8)], 'plateau rim round the Dayuan / Wusun capitals', {}),
    ('native-north-border', [(1076, 1077, 8), (1284, 1077, 8)], 'native border seam (stripes / straight slope segments)', {'climate': 1}),
    ('yangtze-control', [(1319, 726, 12)], 'vanilla Yangtze (Jiangling): must stay unchanged', {}),
    ('yellow-control', [(1274, 843, 12)], 'vanilla Yellow River (Luoyang): must stay unchanged', {}),
    ('xihai-forest-edge', [(965, 940, 20)], 'long straight forest/desert edge north of the lake', {'gate': ('edges_per_hex',)}),
    ('piedmont-plateaus', [(650, 1107, 22), (679, 1104, 22), (719, 1109, 16)],
     'town plateau rims of the 单桓 / 卑陆 capitals and 车师后部 (town floor blended once, fix/town-plateau-trees)', {}),
    ('ailao-native-edge-trees', [(878, 898, 520, 630)], 'tree density step at the native west edge (q=888, Ailao): canvas vs vanilla trees', {}),
    # ground patchiness (ground_patchiness.py): the v19 live close-ups (live-canvas-2048s), box q +- 24,
    # r - 20 .. r + 16 round the camera target hex (the close camera looks north, steeply)
    ('ring-west-ground', [(858, 906, 680, 716)], 'straight / stair-step hard edge: mirror-band temperate 27/28 against cold woodland 4 (v18 and v19)', {}),
    ('funan-ground', [(1071, 1119, 248, 284)], 'dark-olive slot-4 blotches in light ground, wetland / farm specks', {}),
    ('indus-ground', [(446, 494, 580, 616)], 'dark-olive slot-4 blotches, slot-11 farm specks', {}),
    ('taklamakan-ground', [(572, 620, 962, 998)], 'patchiness control: plain desert, oasis farms', {}),
    ('mongolia-ground', [(1016, 1064, 1170, 1206)], 'patchiness control: steppe mottling', {}),
    ('siberia-ground', [(1656, 1704, 1730, 1766)], 'patchiness control: cold woodland under forest', {}),
)

# Documented absolute expectations on a snapshot: (metric path, op, value, note). A failing
# expectation is a regression only if the baseline met it or the candidate is worse.
EXPECT = (
    ('global.trees.wet_canvas', '==', 0, 'trees below the local river level / plane must be 0'),
    ('global.map.border_mask_mismatch', '==', 0, 'byte 15 = map_hex.border_masks(cells) since 7e8baadb'),
    ('global.tiles.global_0x7xx_bare_records', '==', 0, 'no generic/sea/coast/mountain global record with 0x7xx flags (salt black blocks)'),
    ('global.map.one_sided_road_edges', '<=', 1182, 'one-sided road edges (1182 on 2026-10-24) may only decrease'),
    ('global.seams.first_five_passed', '==', True, 'seam loop converged'),
    ('global.materials.desert_banned_share_new', '<=', .005, 'biome step 6: banned slots on the enforced landforms of the generator (landform-tiles.npz; the same count as the --gate). Fixed by the user: never widen'),
    ('global.materials.mud_share_new', '<=', .035, 'mud on new land (3.3 % of land tiles on 2026-10-24; native 0.34 %) must not grow past 3.5 %'),
)

# leaf metric -> (bad direction: +1 up, -1 down, 0 info), abs tolerance, relative tolerance
RULES = {**{f'seam.{c}': (1, 0, 0) for c in STRICT_CLASSES + ('coverage_hole',)}, **{f'seam.{c}': (1, 2, .05) for c in NOISY_CLASSES},
         'lf_px_step_max': (1, .1, .1), 'lf_px_step_p99': (1, .02, .1), 'lf_px_steps_over_half': (1, 4, .1),
         'tile_step_max': (1, .1, .1), 'tile_step_p99': (1, .02, .1), 'shore_step_max': (1, .1, .1), 'shore_step_p99': (1, .02, .1),
         'mesh_dev_max': (1, .05, .1), 'mesh_dev_p99': (1, .02, .1), 'mesh_dev_over_03': (1, 4, .1), 'mesh_uncovered_share': (1, .002, 0),
         'tri_edge_max_hex': (0, 0, 0),   # info only since the mesh bake (3.2.1): big triangles stand only where they keep the LF (mesh_dev_* gates the error itself)
         'wet_trees': (1, 0, 0), 'river_dry_centres': (1, 0, 0), 'fords': (-1, 0, 0),
         'lake_iou': (-1, .01, 0), 'below_plane_inland': (1, 0, 0), 'straight_coast_run_max': (1, 2, 0), 'straight_coast_runs8': (1, 1, .1),
         'town_range': (1, .05, 0), 'town_plot3_range': (1, .1, 0), 'town_positions': (1, 0, 0), 'port_ok': (-1, 0, 0),
         'climate_cut': (1, .05, 0), 'edge_audit_failing': (1, 0, 0), 'raggedness': (1, .02, 0), 'width_cv': (1, .05, 0),
         'band_dry_centres': (1, 0, 0), 'salt_0x7xx_records': (1, 0, 0), 'beach_high': (1, 0, 0),
         # gated only where a window lists them in 'gate' (info elsewhere)
         'green_share': (1, .01, 0), 'forest4_share': (1, .02, 0), 'farm_share': (-1, .01, 0), 'edges_per_hex': (1, 0, .1),
         'islands_per_1000': (1, 0, .15),
         # global
         # seam-repair work counts (triangles clipped, faces added, vertices normalised, coverage-hole hexes
         # patched) are information: they follow the input and do not mark the result worse
         'seam_passes': (1, 0, 0), 'first_five_passed': (-1, 0, 0), 'problem_hexes': (1, 0, 0), 'triangles_clipped': (0, 0, 0),
         'new_faces': (0, 0, 0), 'repaired_meshes': (1, 0, 0), 'normalized_vertices': (0, 0, 0), 'normalized_meshes': (0, 0, 0),
         'coverage_hole_hexes': (0, 0, 0), 'wet_canvas': (1, 0, 0), 'dry_river_centres': (1, 0, 0), 'mud_share_new': (1, .002, 0), 'desert_banned_share_new': (1, .001, 0), 'desert_banned_tiles_new': (0, 0, 0), 'desert_tiles_new': (0, 0, 0),
         'desert_banned_source': (0, 0, 0),
         'biome_green_dry': (1, .01, 0), 'biome_forest_plain': (1, .02, 0), 'biome_mud': (1, .005, 0), 'biome_farm': (-1, .01, 0),
         'border_mask_mismatch': (1, 0, 0), 'one_sided_road_edges': (1, 0, 0), 'global_0x7xx_bare_records': (1, 0, 0)}
GATED_ONLY = ('green_share', 'forest4_share', 'farm_share', 'edges_per_hex', 'islands_per_1000')
# Ground patchiness per window (ground_patchiness.py: hard straight edges, dark islands, farm / wetland squares),
# warn-only: a rise beyond the tolerance is a 'warning', never a regression, and compare also lists every window
# against the v18 numbers (PATCH_BASELINE, audit_ground_patchiness.py baseline) and vanilla's alarm levels.
PATCH_KEYS = ('hard_edge_per100', 'contrast_de_mean', 'straight_segments', 'straight_segment_per100', 'straight_segment_max_hex', 'straight_tile_share',
              'dark_islands', 'dark_island_share', 'dark_island_max_hexes', 'square_blobs', 'square_per100', 'lit')
PATCH_RULES = {'patch_straight_segment_per100': (1, .25, .15), 'patch_straight_segment_max_hex': (1, 1., .15), 'patch_dark_island_share': (1, .01, .15),
               'patch_square_per100': (1, .25, .15), 'patch_hard_edge_per100': (1, 1., .15), 'patch_contrast_de_mean': (1, .5, .05)}
RULES.update(PATCH_RULES)
PATCH_BASELINE = ROOT / 'analysis/ground-patchiness-baseline.json'


# --------------------------------------------------------------------------- helpers
def peak_mb():
    """Peak commit (pagefile usage) of this process in MB (Windows), else None."""
    try:
        class PMC(ctypes.Structure):
            _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong)] + [(n, ctypes.c_size_t) for n in (
                'PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
        c = PMC(); c.cb = ctypes.sizeof(c)
        k32 = ctypes.windll.kernel32; k32.GetCurrentProcess.restype = ctypes.c_void_p
        ps = ctypes.windll.psapi; ps.GetProcessMemoryInfo.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong)
        if not ps.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
            return None
        return round(max(c.PeakPagefileUsage, c.PeakWorkingSetSize) / 2 ** 20)
    except Exception:
        return None


def stats(v, over=None):
    v = np.asarray(v, np.float64)
    if not v.size:
        return None, None, 0 if over is not None else None
    return round(float(v.max()), 4), round(float(np.percentile(v, 99)), 4), (int((v > over).sum()) if over is not None else None)


def rnd(x, k=4):
    return None if x is None else round(float(x), k)


def label(mask, eight=False):
    from scipy import ndimage
    return ndimage.label(mask, structure=np.ones((3, 3)) if eight else None)


def parse_tile_list(data):
    """read_tile_list.parse with the 21-byte records as one numpy structured array."""
    count = struct.unpack_from('<I', data, 12)[0]; pos = 16; paths = []
    for _ in range(count):
        n = struct.unpack_from('<H', data, pos)[0]; pos += 2; paths.append(data[pos:pos + n].decode()); pos += n
    climates = struct.unpack_from('<I', data, pos)[0]; pos += 4
    for _ in range(climates):
        pos += 2 + struct.unpack_from('<H', data, pos)[0]
    pos += 24 + 44 + 1; tiles = struct.unpack_from('<I', data, pos)[0]; pos += 4
    assert len(data) - pos == tiles * 21, (len(data), pos, tiles)
    dt = np.dtype({'names': ['one', 'path', 'climate', 'x', 'y', 'flags', 'lo', 'hi'],
                   'formats': ['<u2', '<u4', 'u1', '<i2', '<i2', '<u2', '<f4', '<f4'], 'offsets': [0, 2, 6, 7, 9, 11, 13, 17], 'itemsize': 21})
    return paths, np.frombuffer(data, dt, count=tiles, offset=pos)


# --------------------------------------------------------------------------- inputs
class Snapshot:
    """Everything snapshot reads, loaded once (missing inputs -> None + a note)."""

    def __init__(self, ws, strict):
        global cw, map_hex
        import campaign_world as cw, map_hex
        self.ws, self.strict, self.notes, self.inputs = ws, strict, [], {}
        self.t0 = time.time()

    def note(self, msg):
        print('note:', msg, flush=True); self.notes.append(msg)

    def path(self, rel):
        p = self.ws / rel
        if p.is_file():
            st = p.stat(); self.inputs[rel] = {'bytes': st.st_size, 'mtime': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(st.st_mtime))}
            return p
        self.note(f'missing input {rel}')
        return None

    def section(self, name, fn, *a):
        t = time.time()
        try:
            out = fn(*a)
        except Exception as e:
            if self.strict:
                raise
            self.note(f'section {name} failed, skipped: {type(e).__name__}: {e}')
            traceback.print_exc()
            out = None
        print(f'[{time.time() - t:5.1f}s] {name}', flush=True)
        return out

    # ---- loaders
    def load(self):
        A = self
        p = A.path('work/final-map/map.hex')
        self.map = map_hex.read(p) if p else None
        if self.map is not None:
            c = self.map['cells']
            self.land = (c[:, :, 0] & 3) != 1
            ids = map_hex.region_ids(c)
            names = self.map['land_regions'] + self.map['sea_regions']
            self.names, self.ids = names, np.clip(ids, 0, len(names) - 1)
            isriv = np.array([n.startswith('tw3k_riv_sea_') for n in names])
            islake = np.array([n == '3k_main_sea_lake' for n in names])
            self.river = ~self.land & isriv[self.ids]
            self.lake = ~self.land & islake[self.ids]
            self.ford = self.river & ((c[:, :, 3] & 0x80) > 0)
            self.beach = (c[:, :, 0] & 3) == 2          # coast_tiles: land beside water is beach (2) or cliff (3)
        p = A.path(PRE + 'lf_height_map.dds')
        self.lf = None
        if p:
            d = p.read_bytes(); h, w = struct.unpack_from('<2I', d, 12)
            if (w, h) != (cw.TILE_W * 4, cw.TILE_H * 4):
                self.note(f'LF is {w}x{h}, canvas wants {cw.TILE_W * 4}x{cw.TILE_H * 4}; LF metrics skipped')
            else:
                self.lf = np.frombuffer(d, '<u2', offset=128, count=h * w).reshape(h, w)[::-1]
        p = A.path('analysis/river-water-level.npy')
        self.level = None
        if p:
            lv = np.load(p, allow_pickle=False)
            if hasattr(lv, 'files'):
                lv = next(lv[k] for k in sorted(lv.files, key=lambda k: k != 'level') if lv[k].dtype.kind == 'f')
            if lv.shape != (cw.TILE_H, cw.TILE_W):
                self.note(f'river-water-level.npy shape {lv.shape} is not the tile grid; plane only')
            else:
                self.level = lv
        lvl = np.full((cw.TILE_H, cw.TILE_W), PLANE, np.float32) if self.level is None else np.fmax(self.level, np.float32(PLANE))
        self.water_level = lvl
        if self.lf is not None:
            self.lf_tile = self.h(self.lf[::4, ::4])
            self.water = self.lf_tile < lvl
        else:
            self.lf_tile = self.water = None
        p = A.path(PRE + 'global_map/global_blend.dds')
        self.mat = self.blend_px = None
        if p:
            d = p.read_bytes(); bh, bw = struct.unpack_from('<2I', d, 12)
            if (bw, bh) != (cw.TILE_W * 4, cw.TILE_H * 4):
                self.note(f'global_blend is {bw}x{bh}; material metrics skipped')
            else:
                b = np.frombuffer(d, np.uint8, offset=128, count=bh * bw * 2).reshape(bh, bw, 2)[::-1]
                self.mat = np.ascontiguousarray(b[2::4, 2::4, 0])   # tile centres (audit_terrain_realism)
                self.blend_px = b[..., 0]                            # view of d (dropped after the windows)
        self.tones = None
        if self.blend_px is not None and self.lf is not None:
            try:
                import ground_patchiness
                self.tones = ground_patchiness.Tones()
            except (SystemExit, Exception) as e:   # tone table not measured yet: patchiness skipped
                self.note(f'ground patchiness skipped: {e}')
        p = A.path('work/current-region-design.json')
        self.towns = []
        if p:
            for r in json.loads(p.read_text(encoding='utf-8')).get('regions', []):
                if r.get('town'):
                    self.towns.append((r['region'], r.get('name', ''), int(r['town'][0]), int(r['town'][1])))
        self.tiles_local = self.tiles_global = None
        for attr, rel in (('tiles_local', PRE + 'tile_list.bin'), ('tiles_global', PRE + 'global_map/tile_list.bin')):
            p = A.path(rel)
            if p:
                setattr(self, attr, parse_tile_list(p.read_bytes()))
        p = A.path('analysis/west-real-terrain.npz')
        self.biome = None
        if p:
            with np.load(p) as z:   # only the two members are read
                if z['biome'].shape == (cw.TILE_H, cw.TILE_W):
                    self.biome, self.biome_names = z['biome'], [str(n) for n in z['biome_names']]
                else:
                    self.note(f"west-real-terrain biome {z['biome'].shape} is not the tile grid; biome shares skipped")
        self.load_seams()
        self.load_trees()

    @staticmethod
    def h(raw):
        return (cw.LEVEL + (raw.astype(np.float32) / 65535 - cw.NORM) * cw.GAIN).astype(np.float32)

    def load_seams(self):
        """Newest ground-seam scan of the variant (by its json mtime) + passes of the last loop."""
        self.seam_masks, self.seam_scan, self.seam_dir, self.seam_passes, self.coverage = {}, None, None, None, None
        dirs = [d for d in self.ws.glob(f'analysis/ground-seam-pipeline-{VARIANT}-*') if (d / 'ground-seam-scan.json').is_file()]
        if not dirs:
            self.note('no analysis/ground-seam-pipeline-* scan; seam metrics skipped')
            return
        mt = lambda d: (d / 'ground-seam-scan.json').stat().st_mtime
        last = max(dirs, key=mt)
        self.seam_dir = last.name
        self.seam_scan = json.loads((last / 'ground-seam-scan.json').read_text(encoding='utf-8'))
        self.path(f'analysis/{last.name}/ground-seam-scan.json')
        if self.path(f'analysis/{last.name}/ground-seam-masks.npz'):
            with np.load(last / 'ground-seam-masks.npz') as z:
                self.seam_masks = {k: z[k] > 0 for k in z.files if z[k].shape == (cw.H, cw.W)}
        num = {int(d.name.rsplit('-', 1)[1]): mt(d) for d in dirs if d.name.rsplit('-', 1)[1].isdigit()}
        if last.name.rsplit('-', 1)[1].isdigit():
            n = int(last.name.rsplit('-', 1)[1])
            # passes 0..n of one loop have non-decreasing scan times (older passes are kept)
            if all(k in num for k in range(n + 1)) and all(num[k] <= num[k + 1] for k in range(n)):
                self.seam_passes = n + 1
            else:
                self.note(f'seam pass dirs 0..{n} are not one run; pass count unknown')
        else:
            self.note(f'newest seam scan is {last.name} (not a numbered loop pass)')
        p = self.ws / f'analysis/coverage-hole-scan-{VARIANT}/ground-seam-masks.npz'
        self.coverage = None
        if p.is_file() and p.stat().st_mtime >= mt(last) - 3600:
            self.path(f'analysis/coverage-hole-scan-{VARIANT}/ground-seam-masks.npz')
            with np.load(p) as z:
                self.coverage = np.any([z[k] > 0 for k in z.files], axis=0)
        elif p.is_file():
            self.note('coverage-hole scan older than the seam loop; ignored')

    def load_trees(self):
        self.tree_xz, self.tree_wet = None, None
        p = self.path(TREES)
        if p is None or self.lf is None:
            return
        b = p.read_bytes(); pos = 24; xyz = []
        for _ in range(struct.unpack_from('<I', b, 20)[0]):
            n = struct.unpack_from('<H', b, pos)[0]; pos += 2 + n; cnt = struct.unpack_from('<I', b, pos)[0]; pos += 4
            for _ in range(cnt):
                xyz.append(b[pos:pos + 12]); pos += 18 + struct.unpack_from('<I', b, pos + 14)[0] * 4
        assert pos == len(b), 'tree list layout'
        p3 = np.frombuffer(b''.join(xyz), '<f4').reshape(-1, 3).astype(np.float64)
        tx, tz = p3[:, 0] / cw.DX * 2, p3[:, 2] / cw.DZ * 2
        from final_terrain_surface import sample_final_surface
        lo = np.full(len(p3), np.inf); lv = np.full(len(p3), -np.inf)
        for ox in (-.35, 0, .35):
            for oz in (-.35, 0, .35):
                lo = np.minimum(lo, sample_final_surface(self.lf, tx + ox, tz + oz))
                if self.level is not None:
                    rl = self.level[np.clip((tz + oz).astype(np.int64), 0, cw.TILE_H - 1), np.clip((tx + ox).astype(np.int64), 0, cw.TILE_W - 1)]
                    lv = np.fmax(lv, rl.astype(np.float64))
        q, zr = p3[:, 0] / cw.DX, p3[:, 2] / cw.DZ
        native = (cw.PAD_LEFT <= q) & (q < cw.PAD_LEFT + cw.NATIVE_W) & (cw.PAD_BOTTOM <= zr) & (zr < cw.PAD_BOTTOM + cw.NATIVE_H)
        self.tree_xz = np.column_stack((tx, tz))
        self.tree_wet = ~native & ((lo < PLANE + WET_MARGIN) | (lo < lv + WET_MARGIN))
        self.tree_native = native

    def load_meshes(self, boxes):
        """Stage land-mesh triangles whose xz box meets any part box (raw frame), per part."""
        from audit_ground_seams import mesh_triangles
        files = sorted(self.ws.glob(PRE + 'global_meshes/land_mesh_*.rigid_model_v2'))
        if not files:
            self.note('no stage land meshes; mesh metrics skipped')
            return None
        self.inputs[PRE + 'global_meshes/land_mesh_*.rigid_model_v2'] = {'files': len(files), 'bytes': sum(f.stat().st_size for f in files)}
        bx = np.array(boxes, np.float64)   # (n, 4) x0 x1 z0 z1
        per = [[] for _ in boxes]
        for f in files:
            t = mesh_triangles(f.read_bytes())
            lo, hi = t[:, :, [0, 2]].min(1), t[:, :, [0, 2]].max(1)
            for i, (x0, x1, z0, z1) in enumerate(bx):
                s = (hi[:, 0] >= x0) & (lo[:, 0] <= x1) & (hi[:, 1] >= z0) & (lo[:, 1] <= z1)
                if s.any():
                    per[i].append(t[s])
        return [np.concatenate(p) if p else np.zeros((0, 3, 3)) for p in per]


# --------------------------------------------------------------------------- window metrics
def part_box(p):
    if len(p) == 3:
        q, r, R = p; box = (q - R, q + R, r - R, r + R)
    else:
        box = p
    q0, q1, r0, r1 = box
    return max(0, q0), min(cw.W - 1, q1), max(0, r0), min(cw.H - 1, r1)


class Part:
    def __init__(self, S, box):
        self.box = q0, q1, r0, r1 = box
        self.ty0, self.ty1 = 2 * r0, min(cw.TILE_H, 2 * r1 + 3)
        self.tx0, self.tx1 = 2 * q0, min(cw.TILE_W, 2 * q1 + 2)
        self.ts = (slice(self.ty0, self.ty1), slice(self.tx0, self.tx1))
        ty, tx = np.mgrid[self.ty0:self.ty1, self.tx0:self.tx1]
        self.tq, self.tr = cw.tile_hex(tx, ty)
        self.inside = (self.tq >= q0) & (self.tq <= q1) & (self.tr >= r0) & (self.tr <= r1)
        self.hs = (slice(r0, r1 + 1), slice(q0, q1 + 1))
        self.px = (slice(4 * self.ty0, 4 * self.ty1), slice(4 * self.tx0, 4 * self.tx1))

    def at_tiles(self, hexgrid):
        return hexgrid[self.tr, self.tq] & self.inside

    def hexes_in(self, hexmask):
        return int(hexmask[self.hs].sum())


def tile_steps(h, mask):
    """|dh| between 4-neighbouring tiles that are both in mask."""
    out = [np.abs(np.diff(h, axis=a))[(mask[1:] & mask[:-1]) if a == 0 else (mask[:, 1:] & mask[:, :-1])] for a in (0, 1)]
    return np.concatenate(out)


def straight_runs(w, inside):
    """Longest run of a water/land boundary along one tile row or column, and runs >= 8 tiles."""
    best, n8 = 0, 0
    for a in (0, 1):
        e = (w[1:] != w[:-1]) & inside[1:] & inside[:-1] if a == 0 else ((w[:, 1:] != w[:, :-1]) & inside[:, 1:] & inside[:, :-1]).T
        for row in e:
            if not row.any():
                continue
            d = np.diff(np.r_[0, row.astype(np.int8), 0])
            runs = np.flatnonzero(d == -1) - np.flatnonzero(d == 1)
            best = max(best, int(runs.max())); n8 += int((runs >= 8).sum())
    return best, n8


def mesh_metrics(S, part, tris):
    from audit_ground_seams import raster, SAMPLE_OFFSET
    q0, q1, r0, r1 = part.box
    step = cw.DX / 4
    x0, z0 = q0 * cw.DX, r0 * cw.DX
    nx, nz = (q1 - q0 + 1) * 4, int((r1 - r0 + 1.5) * 4)
    if not len(tris):
        return {'mesh_dev_max': None, 'mesh_dev_p99': None, 'mesh_dev_over_03': None, 'mesh_uncovered_share': None, 'tri_edge_max_hex': None}
    t = tris - np.array([x0, 0, z0])
    cnt, low, high = raster(t, nx, nz, step, SAMPLE_OFFSET)
    xs = x0 + (np.arange(nx) + SAMPLE_OFFSET[0]) * step
    zs = z0 + (np.arange(nz) + SAMPLE_OFFSET[1]) * step
    X, Z = np.meshgrid(xs, zs)
    from final_terrain_surface import sample_final_surface
    lf = sample_final_surface(S.lf, X / cw.DX * 2, Z / cw.DX * 2)   # raw z / DX * 2 = tile row
    q, r = cw.raw_hex(X, Z)
    land = S.land[r, q] & (q >= q0) & (q <= q1) & (r >= r0) & (r <= r1)
    cov = (cnt > 0) & land
    dev = np.maximum(np.abs(high - lf), np.abs(low - lf))[cov]
    mx, p99, over = stats(dev, .3)
    e = np.linalg.norm(tris[:, :, [0, 2]] - np.roll(tris[:, :, [0, 2]], 1, 1), axis=2).max(1) / cw.DX
    return {'mesh_dev_max': mx, 'mesh_dev_p99': p99, 'mesh_dev_over_03': over,
            'mesh_uncovered_share': rnd(1 - cov.sum() / max(1, land.sum())), 'tri_edge_max_hex': rnd(e.max(), 2)}


def window_metrics(S, name, parts_spec, extra, tris_by_part):
    parts = [Part(S, part_box(p)) for p in parts_spec]
    m = collections.defaultdict(float)
    acc = collections.defaultdict(list)
    arrays = []
    towns = {}
    for k, P in enumerate(parts):
        ins = P.inside
        land_t = P.at_tiles(S.land) if S.map is not None else ins
        # seams
        for c, mask in S.seam_masks.items():
            m[f'seam.{c}'] += P.hexes_in(mask)
        if S.coverage is not None:
            m['seam.coverage_hole'] += P.hexes_in(S.coverage)
        defect = np.zeros(ins.shape, np.uint8)
        for c, mask in S.seam_masks.items():
            if c in STRICT_CLASSES:
                defect |= np.where(P.at_tiles(mask), np.uint8(1 if c.startswith(('coarse', 'lf')) else 2), np.uint8(0))
        if S.coverage is not None:
            defect |= np.where(P.at_tiles(S.coverage), np.uint8(4), np.uint8(0))
        lf_t = water = None
        if S.lf is not None:
            hp = S.h(S.lf[P.px])
            inpx = ins.repeat(4, 0).repeat(4, 1)[:hp.shape[0], :hp.shape[1]]
            d = np.concatenate([np.abs(np.diff(hp, axis=0))[inpx[1:] & inpx[:-1]], np.abs(np.diff(hp, axis=1))[inpx[:, 1:] & inpx[:, :-1]]])
            acc['lf_px'].append(d)
            lf_t = S.lf_tile[P.ts]; water = S.water[P.ts] & ins
            if 'lake_level' in extra and S.map is not None:
                # a raised lake prop (Hehuang: Qinghai lake at the patch's water level), not the plane
                rel, key = extra['lake_level']; lp = S.path(rel)
                if lp:
                    from scipy import ndimage
                    lvl = float(json.loads(lp.read_text(encoding='utf-8'))[key])
                    water = water | ((lf_t < lvl) & ndimage.binary_dilation(P.at_tiles(S.lake), iterations=2) & ins)
            acc['tile'].append(tile_steps(lf_t, land_t & ~water))
            from scipy import ndimage
            shore = land_t & ~water & ndimage.binary_dilation(water, iterations=2)
            acc['shore'].append(tile_steps(lf_t, shore))
            m['water_tiles'] += int(water.sum())
            m['below_plane_inland'] += int(((lf_t < PLANE) & (P.at_tiles(S.land) | P.at_tiles(S.river))).sum()) if S.map is not None else 0
            run, n8 = straight_runs(water, ins)
            m['straight_coast_run_max'] = max(m['straight_coast_run_max'], run); m['straight_coast_runs8'] += n8
            if S.map is not None:
                lake_t = P.at_tiles(S.lake); notriv = ~P.at_tiles(S.river)
                wl = water & notriv & (lake_t | land_t)   # lake water, not the open sea
                if lake_t.any():
                    m['_lake_i'] += int((wl & lake_t).sum()); m['_lake_u'] += int((wl | lake_t).sum())
                # river hex centres (tile 2q+1, 2r+odd+1 = LF pixel 4x that)
                rr, qq = np.nonzero(S.river[P.hs]); rr, qq = rr + P.box[2], qq + P.box[0]
                cy, cx = np.minimum(2 * rr + qq % 2 + 1, cw.TILE_H - 1), np.minimum(2 * qq + 1, cw.TILE_W - 1)
                dry = S.h(S.lf[4 * cy, 4 * cx]) >= S.water_level[cy, cx]
                m['river_hexes'] += len(rr); m['river_dry_centres'] += int(dry.sum())
                if 'rows' in extra:
                    a, b = extra['rows']; band = (rr >= a) & (rr <= b)
                    m['band_dry_centres'] += int((dry & band).sum())
                    m['band_water_tiles'] += int((water & (P.tr >= a) & (P.tr <= b)).sum())
                fm = S.ford[P.hs]; m['fords'] += int(fm.sum())
                lab, n = label(fm, True)
                acc['ford_groups'] += sorted(np.bincount(lab.ravel())[1:].tolist())
        if S.mat is not None:
            mt = S.mat[P.ts]
            nl = max(1, int(land_t.sum()))
            for key, sel in (('green_share', GREEN), ('forest4_share', (4,)), ('woodland_share', WOODLAND), ('farm_share', FARM),
                             ('mud_share', (MUD,)), ('sand_share', SAND), ('wet_share', (WETM,))):
                m['_n.' + key] += int((np.isin(mt, sel) & land_t).sum())
            m['_land_tiles'] += nl
            bp = S.blend_px[P.px]
            lp = land_t.repeat(4, 0).repeat(4, 1)[:bp.shape[0], :bp.shape[1]]
            wd = np.isin(bp, WOODLAND)
            m['_edges'] += int(((wd[1:] != wd[:-1]) & lp[1:] & lp[:-1]).sum() + ((wd[:, 1:] != wd[:, :-1]) & lp[:, 1:] & lp[:, :-1]).sum())
            m['_land_hexes'] += nl / 4
            isl = 0
            for cls in (wd & lp, ~wd & lp):
                lab, n = label(cls, True)
                if n:
                    size = np.bincount(lab.ravel())
                    edge = np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])
                    small = np.flatnonzero(size < 6 * 64); small = small[(small > 0) & ~np.isin(small, edge)]
                    isl += len(small)
            m['_islands'] += isl
        else:
            mt = None
        # towns
        for key, nm, q, r in S.towns:
            if P.box[0] <= q <= P.box[1] and P.box[2] <= r <= P.box[3] and S.lf is not None:
                px, pz = 8 * q, 8 * r + 4 * (q % 2)
                blk = S.h(S.lf[pz:pz + 8, px:px + 8]); plot = S.h(S.lf[max(0, pz - 8):pz + 16, max(0, px - 8):px + 16])
                towns[key] = {'name': nm, 'hex': [q, r], 'median': rnd(np.median(blk)), 'range': rnd(np.ptp(blk)), 'plot3_range': rnd(np.ptp(plot)),
                              'land': bool(S.land[r, q]) if S.map is not None else None}
        # trees
        if S.tree_xz is not None:
            tq, tr = cw.tile_hex(S.tree_xz[:, 0].astype(np.int64), S.tree_xz[:, 1].astype(np.int64))
            sel = (tq >= P.box[0]) & (tq <= P.box[1]) & (tr >= P.box[2]) & (tr <= P.box[3])
            m['trees'] += int((sel & ~S.tree_native).sum()); m['wet_trees'] += int((sel & S.tree_wet).sum())
            wet_xy = S.tree_xz[sel & S.tree_wet] - (P.tx0, P.ty0)
        else:
            wet_xy = np.zeros((0, 2))
        # coast tiles
        if S.tiles_local is not None:
            paths, rec = S.tiles_local
            if not hasattr(S, '_coast_idx'):
                S._coast_idx = np.array(['\\sea_coast\\' in p for p in paths])
            sel = (rec['x'] >= P.tx0) & (rec['x'] < P.tx1) & (rec['y'] >= P.ty0) & (rec['y'] < P.ty1)
            m['sea_coast'] += int(S._coast_idx[rec['path'][sel]].sum())
        if S.map is not None and S.lf is not None:
            rr, qq = np.nonzero(S.beach[P.hs]); rr, qq = rr + P.box[2], qq + P.box[0]
            m['beach_hexes'] += len(rr)
            m['beach_high'] += int((S.h(S.lf[8 * rr + 4 * (qq % 2) + 4, 8 * qq + 4]) > BEACH_HIGH).sum())
        # mesh
        if tris_by_part is not None and S.lf is not None and S.map is not None:
            mm = mesh_metrics(S, P, tris_by_part[k])
            for key, v in mm.items():
                if v is not None:
                    m[key] = max(m.get(key, 0), v) if key != 'mesh_dev_over_03' else m.get(key, 0) + v
        # extras
        if 'climate' in extra and S.tiles_local is not None:
            m.update(climate_cut(S, P, m))
        if 'salt' in extra and S.tiles_global is not None:
            paths, rec = S.tiles_global
            bare = np.array([bool(BARE.search(p)) for p in paths])
            sel = (rec['x'] >= P.tx0) & (rec['x'] < P.tx1) & (rec['y'] >= P.ty0) & (rec['y'] < P.ty1) & ((rec['flags'] & 0x700) > 0)
            m['salt_0x7xx_records'] += int(bare[rec['path'][sel]].sum())
        if 'river_shape' in extra and water is not None and S.map is not None:
            m.update(river_shape(S, P, water))
        if 'port' in extra and S.lf is not None:
            q, r = extra['port']
            if P.box[0] <= q <= P.box[1] and P.box[2] <= r <= P.box[3]:
                px, pz = 8 * q, 8 * r + 4 * (q % 2)
                med = float(np.median(S.h(S.lf[pz:pz + 8, px:px + 8])))
                m['port_lf_median'] = rnd(med); m['port_ok'] = int(bool(S.land[r, q]) and med >= .4)
        arrays.append({'lf': lf_t, 'mat': mt, 'water': water, 'defect': defect, 'inside': ins, 'wet': wet_xy,
                       'towns': [(q - P.box[0], r - P.box[2]) for _, _, q, r in S.towns if P.box[0] <= q <= P.box[1] and P.box[2] <= r <= P.box[3]]})
    out = {}
    for key in ('lf_px', 'tile', 'shore'):
        if acc[key]:
            mx, p99, over = stats(np.concatenate(acc[key]), .5)
            pre = {'lf_px': 'lf_px_step', 'tile': 'tile_step', 'shore': 'shore_step'}[key]
            out[pre + '_max'], out[pre + '_p99'] = mx, p99
            if key == 'lf_px':
                out['lf_px_steps_over_half'] = over
    if acc['ford_groups'] or m.get('fords'):
        out['ford_groups'] = acc['ford_groups']
    for key, v in m.items():
        if key.startswith('_'):
            continue
        out[key] = int(v) if isinstance(v, float) and v.is_integer() and not key.startswith(('mesh_dev', 'tri_', 'port_lf', 'raggedness', 'width', 'narrow', 'climate_cut', 'mesh_uncovered')) else v
    if m.get('_lake_u'):
        out['lake_iou'] = rnd(m['_lake_i'] / m['_lake_u'])
    if getattr(S, 'tones', None) is not None:
        out.update(patch_metrics(S, [P.box for P in parts]))
    if m.get('_land_tiles'):
        for key in ('green_share', 'forest4_share', 'woodland_share', 'farm_share', 'mud_share', 'sand_share', 'wet_share'):
            out[key] = rnd(m['_n.' + key] / m['_land_tiles'])
        out['edges_per_hex'] = rnd(m['_edges'] / 8 / max(1, m['_land_hexes']))
        out['islands_per_1000'] = rnd(m['_islands'] / max(1, m['_land_hexes']) * 1000, 2)
    out['town_positions'] = sorted(f"{k}@{v['hex'][0]},{v['hex'][1]}" for k, v in towns.items())
    out['towns'] = towns
    return out, parts, arrays


def patch_metrics(S, boxes):
    """Ground patchiness of a window (all its parts) on the stage blend, land = stage LF above the water plane."""
    import ground_patchiness as gp

    def band(y0, y1, x0, x1):
        y0, x0 = max(0, y0), max(0, x0)
        return np.ascontiguousarray(S.blend_px[y0:y1, x0:x1]), S.h(S.lf[y0:y1, x0:x1]) >= gp.PLANE
    tot, isl = None, []
    for q0, q1, r0, r1 in boxes:
        _, _, inbox, sums = gp.analyse_box(band, q0, q1, r0, r1, S.tones)
        isl += inbox
        tot = gp.merge_sums(tot, sums)
    nums = gp.numbers(tot, isl)
    return {f'patch_{k}': nums[k] for k in PATCH_KEYS}


def patch_vs_baseline(rep):
    """Warn-only rows: each window's patchiness against the v18 numbers and vanilla's alarm levels."""
    import ground_patchiness as gp
    base = json.loads(PATCH_BASELINE.read_text(encoding='utf-8'))['windows'] if PATCH_BASELINE.is_file() else {}
    alarm = gp.alarms()
    rows = []
    for name, w in rep.get('windows', {}).items():
        met = w.get('metrics') or {}
        for k in gp.ALARM_METRICS:
            v = met.get(f'patch_{k}')
            if v is None:
                continue
            b = (base.get(name) or {}).get(k)
            status = judge(f'patch_{k}', b, v, True)[0] if b is not None else 'no-baseline'
            status = 'warning' if status == 'regression' else status
            lit = k in alarm and v > alarm[k]
            if status in ('warning', 'improved') or lit:
                rows.append({'window': name, 'metric': k, 'v18': b, 'new': v, 'alarm': alarm.get(k), 'status': status, 'lit': lit})
    return rows


def edge_counts(fail, box, margin=8):
    """(failing, known) native-edge-audit segments near a part box (q0, q1, r0, r1): a segment whose fails are all
    registered entries ('*_known') counts as known, any other failing segment as failing."""
    near = [f for (q, r), f in fail if box[0] - margin <= q <= box[1] + margin and box[2] - margin <= r <= box[3] + margin]
    known = sum(1 for f in near if f and all(str(x).endswith('_known') for x in f))
    return len(near) - known, known


def climate_cut(S, P, m):
    """Climate flips at the native west border (tile column NATIVE_TILE_X0) in this part, from the local
    tile_list record footprints (generic NxM; other tiles as 1x1), and native-edge-audit failures here."""
    paths, rec = S.tiles_local
    if not hasattr(S, '_dims'):
        S._dims = np.array([tuple(map(int, mm.groups())) if (mm := re.search(r'generic_(\d+)x(\d+)', p)) else (1, 1) for p in paths])
        S._gsea = np.array(['generic_sea' in p for p in paths])
    y0, y1, x0, x1 = P.ty0 - 16, P.ty1 + 16, P.tx0 - 16, P.tx1 + 16
    sel = (rec['x'] >= x0) & (rec['x'] < x1) & (rec['y'] >= y0) & (rec['y'] < y1) & ~S._gsea[rec['path']]
    clim = np.full((y1 - y0, x1 - x0), 255, np.uint8)
    for r in rec[sel]:
        w, h = S._dims[r['path']]
        if r['flags'] & 0x10 or (r['flags'] & 0x30) == 0x30:   # quarter turn swaps the footprint (approximation)
            w, h = h, w
        clim[max(0, r['y'] - y0):max(0, r['y'] - y0 + h), max(0, r['x'] - x0):max(0, r['x'] - x0 + w)] = r['climate']
    out = {}
    X = cw.NATIVE_TILE_X0 - x0
    if 0 < X < clim.shape[1]:
        a, b = clim[16:-16, X - 1], clim[16:-16, X]
        valid = (a != 255) & (b != 255)
        if valid.any():
            out['climate_cut'] = max(m.get('climate_cut', 0), rnd(((a != b) & valid).sum() / valid.sum()))
    fail = getattr(S, '_edge_fail', None)
    if fail is not None:
        # A segment failing only on registered entries ('*_known', e.g. tone_cut_known) is not counted: the known list
        # may only shrink, which the native_edge_tone_cut invariant guards (coordinator 2026-10-30, v20 precheck 5 -> 7).
        failing, known = edge_counts(fail, P.box)
        out['edge_audit_failing'] = int(m.get('edge_audit_failing', 0)) + failing
        out['edge_audit_known'] = int(m.get('edge_audit_known', 0)) + known
    return out


def river_shape(S, P, water):
    """Bank raggedness (share of shoreline tiles not on a river-hex outline), width CV and east-end narrowing."""
    riv_t = P.at_tiles(S.river)
    band = np.isfinite(S.level[P.ts]) if S.level is not None else riv_t
    w = water & (band | riv_t)
    if not w.any():
        return {}
    nb = np.zeros_like(w)
    nb[1:] |= ~w[:-1]; nb[:-1] |= ~w[1:]; nb[:, 1:] |= ~w[:, :-1]; nb[:, :-1] |= ~w[:, 1:]
    shore = w & nb
    out_line = np.zeros_like(riv_t)
    nr = ~riv_t
    out_line[1:] |= nr[:-1]; out_line[:-1] |= nr[1:]; out_line[:, 1:] |= nr[:, :-1]; out_line[:, :-1] |= nr[:, 1:]
    outline = riv_t & out_line
    rag = (shore & ~outline).sum() / max(1, shore.sum())
    width = w.sum(0).astype(np.float64); width = width[width > 0]
    k = max(1, len(width) // 10)
    return {'raggedness': rnd(rag), 'width_cv': rnd(width.std() / width.mean()), 'width_median': rnd(np.median(width), 2),
            'east_end_narrowing': rnd(width[-k:].mean() / np.median(width))}


# --------------------------------------------------------------------------- global metrics
def global_metrics(S):
    g = {'seams': {}, 'repair': {}, 'trees': {}, 'rivers': {}, 'materials': {}, 'map': {}, 'tiles': {}}
    if S.seam_scan is not None:
        g['seams'] = {'scan': S.seam_dir, 'first_five_passed': S.seam_scan.get('first_five_passed'),
                      'seam_passes': S.seam_passes, **{f'seam.{k}': v.get('hexes') for k, v in S.seam_scan.get('classes', {}).items()},
                      'steep_authored_ratio': (S.seam_scan.get('steep_generic') or {}).get('authored_ratio')}
    if S.coverage is not None:
        g['seams']['coverage_hole_hexes'] = int(S.coverage.sum())
    rep = {}
    for rel, fn in ((f'analysis/local-coarse-seam-repair-{VARIANT}.json',
                     lambda d: {'problem_hexes': d.get('problem_hexes'), 'problem_hex_list': d.get('problem_hex_list'), 'quarter_cells': d.get('quarter_cells'), 'triangles_clipped': d.get('triangles_clipped'),
                                'new_faces': d.get('new_faces'), 'repaired_meshes': len(d.get('meshes', [])), 'repair_input': Path(d.get('input_scan', '')).name}),
                    (f'analysis/coarse-partition-height-normalization-{VARIANT}.json',
                     lambda d: {'normalized_meshes': len(d.get('meshes', [])), 'normalized_vertices': sum(x.get('changed_vertices', 0) for x in d.get('meshes', [])),
                                'lf_pixels_unchanged': d.get('lf_pixels_unchanged')}),
                    ('analysis/river-bed-mesh.json',
                     lambda d: {'river_bed_meshes_changed': sum(len(v.get('meshes_changed', [])) for v in d.get('variants', []) if v.get('variant') == VARIANT),
                                'river_bed_footprint_corners': sum(v.get('footprint_corners', 0) for v in d.get('variants', []) if v.get('variant') == VARIANT)}),
                    ('analysis/island-ground-patches.json', lambda d: {'island_patches': len(d.get('patches', []))}),
                    ('analysis/terrain-native-foliage.json',
                     lambda d: {'foliage_wet_dropped': d.get('wet_trees_dropped'), 'foliage_river_bank_dropped': d.get('river_bank_trees_dropped'),
                                'foliage_total_instances': d.get('total_instances'),
                                'foliage_post_filter_removals': (d.get('props_bake') or {}).get('post_filter_removals')}),
                    ('analysis/coast-tiles.json', lambda d: {'coast_pieces': d.get('pieces'), 'coast_tiled_hexes': d.get('tiled_hexes'),
                                                             'coast_unmatched_mask_hexes': d.get('unmatched_mask_hexes')})):
        p = S.path(rel)
        if p:
            rep.update(fn(json.loads(p.read_text(encoding='utf-8'))))
    p = S.path('analysis/pipeline-timing.log')
    if p:
        lines = p.read_text(encoding='utf-8', errors='replace').splitlines()
        tot = [i for i, l in enumerate(lines) if '\tTOTAL\t' in l]
        # Run inside the pipeline (build_polished_campaign_terrain.ps1 sets TW3K_PIPELINE_STARTED and
        # TW3K_PIPELINE_ELAPSED_S): this snapshot runs before the run's own TOTAL line is written, so the
        # last TOTAL in the log is the previous run's. The current run is then the log lines stamped at or
        # after its start, its time the pipeline's elapsed time so far. A standalone snapshot (no such
        # variables) reads the last finished run and its TOTAL line, as before.
        started = os.environ.get('TW3K_PIPELINE_STARTED')
        run = []
        if started:
            stamp = re.compile(r'(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\t')
            run = [l for l in lines if (m := stamp.match(l)) and m.group(1) >= started]
            rep['pipeline_total_s'] = float(os.environ.get('TW3K_PIPELINE_ELAPSED_S') or 'nan')
            rep['pipeline_finished'] = run[-1].split('\t')[0] if run else started
            rep['pipeline_complete'] = False
        elif tot:
            prev = tot[-2] if len(tot) > 1 else -1
            run = lines[prev + 1:tot[-1] + 1]
            rep['pipeline_total_s'] = float(lines[tot[-1]].split('\t')[-1].rstrip('s').replace(',', ''))
            rep['pipeline_finished'] = lines[tot[-1]].split('\t')[0]
            rep['pipeline_complete'] = True
        if run:
            rep['timing_seam_scans'] = sum('audit_ground_seams.py' in l for l in run)
            rep['timing_seam_repairs'] = sum('repair_local_coarse_seams.py' in l for l in run)
    g['repair'] = rep
    if S.tree_wet is not None:
        g['trees'] = {'canvas': int((~S.tree_native).sum()), 'wet_canvas': int(S.tree_wet.sum())}
    if S.map is not None and S.lf is not None:
        rr, qq = np.nonzero(S.river)
        cy, cx = np.minimum(2 * rr + qq % 2 + 1, cw.TILE_H - 1), np.minimum(2 * qq + 1, cw.TILE_W - 1)
        dry = S.h(S.lf[4 * cy, 4 * cx]) >= S.water_level[cy, cx]
        names = np.array(['_'.join(n.split('_')[3:-1]) for n in S.names])[S.ids[rr, qq]]
        g['rivers'] = {'river_hexes': len(rr), 'dry_river_centres': int(dry.sum()), 'fords': int(S.ford.sum()),
                       'dry_by_river': dict(sorted(collections.Counter(names[dry].tolist()).items()))}
    if S.mat is not None and S.map is not None:
        ty, tx = np.mgrid[:cw.TILE_H, :cw.TILE_W]
        q, r = cw.tile_hex(tx, ty)
        land = S.land[r, q]
        native = np.zeros((cw.TILE_H, cw.TILE_W), bool); native[cw.NATIVE_TILE_SLICE] = True
        new = land & ~native
        g['materials'] = {'mud_share_new': rnd((S.mat[new] == MUD).sum() / max(1, new.sum())),
                          'farm_share_new': rnd(np.isin(S.mat[new], FARM).sum() / max(1, new.sum())),
                          'mud_share_native': rnd((S.mat[land & native] == MUD).sum() / max(1, (land & native).sum()))}
        if S.biome is not None:
            g['biomes'] = biome_shares(S, land, native)
            g['materials'].update(desert_banned(S, new, q, r))
    if S.map is not None:
        c = S.map['cells']
        g['map'] = {'border_mask_mismatch': int((map_hex.border_masks(c) != c[:, :, 15]).sum()), 'one_sided_road_edges': one_sided_road_edges(c)}
        if S.lf is not None:
            rr, qq = np.nonzero(S.beach)
            g['map'].update({'beach_hexes': len(rr), 'beach_high': int((S.h(S.lf[8 * rr + 4 * (qq % 2) + 4, 8 * qq + 4]) > BEACH_HIGH).sum())})
    if S.tiles_global is not None:
        paths, rec = S.tiles_global
        bare = np.array([bool(BARE.search(p)) for p in paths])
        g['tiles']['global_0x7xx_bare_records'] = int(bare[rec['path'][(rec['flags'] & 0x700) > 0]].sum())
        g['tiles']['global_records'] = len(rec)
    if S.tiles_local is not None:
        paths, rec = S.tiles_local
        coast = np.array(['\\sea_coast\\' in p for p in paths])
        g['tiles'].update({'local_records': len(rec), 'sea_coast_records': int(coast[rec['path']].sum())})
        tl = S.ws / (PRE + 'tile_list.bin'); ea = S.ws / 'analysis/native-edge-audit.json'
        if ea.is_file() and ea.stat().st_mtime < tl.stat().st_mtime:
            S.note('native-edge-audit.json is older than tile_list.bin (stale climate_cut)')
    return g


def desert_banned(S, new, q, r):
    """Biome step 6 gate: share of the slots banned on every desert landform (biome_profiles.AUDIT) on new
    land tiles of real desert away from water and towns (DESERT_CLEAR_HEX: the oasis belts, river and
    lake shores and the towns' own fields keep their greens)."""
    import biome_profiles as bp
    import landform_tiles
    # [STEP 2] with the generator's landform field: the banned share on the enforced landforms (the same
    # count as audit_terrain_realism --gate), not a re-classification of the raw RESOLVE desert
    lt = landform_tiles.load(S.ws) if (S.ws / landform_tiles.NAME).is_file() else None
    if lt is not None:
        res = landform_tiles.banned_share(S.mat, lt, sel=new)
        n = sum(v['tiles'] for v in res.values()); b = sum(v['banned'] for v in res.values())
        return {'desert_banned_share_new': rnd(b / max(1, n)), 'desert_banned_tiles_new': b, 'desert_tiles_new': n,
                'desert_banned_source': 'landform-tiles', 'desert_banned_by_landform': {k: v['share'] for k, v in res.items()}}
    S.note(f'no {landform_tiles.NAME}: desert_banned on the raw RESOLVE desert (pre step-2 definition)')
    from hex_components import hex_dilate
    c = S.map['cells']
    keep = (~S.land) | ((c[:, :, 2] >> 4) > 0)
    for _ in range(DESERT_CLEAR_HEX):
        keep = hex_dilate(keep)
    des = new & (S.biome == S.biome_names.index('desert')) & ~keep[r, q]
    n = int(des.sum())
    # slots banned on every desert landform (drymtn keeps its own steppe_3, 21)
    ban = np.isin(S.mat[des], sorted(set(bp.AUDIT['gobi']['banned']) & set(bp.AUDIT['drymtn']['banned'])))
    return {'desert_banned_share_new': rnd(ban.sum() / max(1, n)), 'desert_banned_tiles_new': int(ban.sum()), 'desert_tiles_new': n}


def biome_shares(S, land, native):
    """Per real biome (west-real-terrain.npz, RESOLVE classes polish reads), over the new canvas land
    and over vanilla land as the reference: share of each global_blend material family at tile
    centres, plus green / forest4 / woodland / farm / mud shares."""
    out = {}
    fam = np.full(32, -1, np.int8)
    for i, (_, sel) in enumerate(FAMILIES.items()):
        fam[list(sel)] = i
    for side, sel in (('new', land & ~native), ('native', land & native)):
        b, m = S.biome[sel], S.mat[sel]
        res = {}
        for k, name in enumerate(S.biome_names):
            mk = m[b == k]
            if len(mk) < BIOME_MIN_TILES:
                continue
            n = len(mk)
            f = np.bincount(fam[np.minimum(mk, 31)].astype(np.int64) + 1, minlength=len(FAMILIES) + 1)[1:]
            res[name] = {'tiles': n, **{f'fam_{fn}': rnd(f[i] / n) for i, fn in enumerate(FAMILIES)},
                         'green_share': rnd(np.isin(mk, GREEN).sum() / n), 'forest4_share': rnd((mk == 4).sum() / n),
                         'woodland_share': rnd(np.isin(mk, WOODLAND).sum() / n), 'farm_share': rnd(np.isin(mk, FARM).sum() / n),
                         'mud_share': rnd((mk == MUD).sum() / n)}
        out[side] = res
    return out


def one_sided_road_edges(cells):
    """move_town.one_sided_road_edges, vectorised: road bits (byte 3 bits 1-6) without the partner bit."""
    m = ((cells[:, :, 3] >> 1) & 63).astype(np.uint8)
    H, W = m.shape
    dirs = (((0, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0)), ((0, 1), (1, 1), (1, 0), (0, -1), (-1, 0), (-1, 1)))
    r, q = np.nonzero(m)
    bad = 0
    for d in range(6):
        has = (m[r, q] >> d) & 1 > 0
        dq = np.where(q & 1, dirs[1][d][0], dirs[0][d][0]); dr = np.where(q & 1, dirs[1][d][1], dirs[0][d][1])
        a, b = q + dq, r + dr
        ok = (a >= 0) & (a < W) & (b >= 0) & (b < H)
        partner = np.zeros(len(q), bool)
        partner[ok] = (m[b[ok], a[ok]] >> ((d + 3) % 6)) & 1 > 0
        bad += int((has & ~partner).sum())
    return bad


# --------------------------------------------------------------------------- renders
def compose(images, height=240, gap=4):
    from PIL import Image
    ims = []
    for a in images:
        im = Image.fromarray(a)
        s = height / im.height
        ims.append(im.resize((max(1, round(im.width * s)), height), Image.NEAREST if s >= 1 else Image.LANCZOS))
    out = Image.new('RGB', (sum(i.width for i in ims) + gap * (len(ims) - 1), height), (20, 20, 20))
    x = 0
    for i in ims:
        out.paste(i, (x, 0)); x += i.width + gap
    return out


def render_part(S, P, a):
    """North-up RGB: LF hillshade, water blue, seam defects red (coarse/LF) / orange (fine) / magenta (holes),
    wet trees magenta dots, towns white boxes. Tile resolution x2."""
    if S.lf is None:
        return np.zeros((8, 8, 3), np.uint8)
    hp = S.h(S.lf[P.px][::2, ::2])
    gz, gx = np.gradient(hp)
    shade = np.clip(.55 + (gz - gx) * 1.6, 0, 1)
    rel = np.clip((hp - np.percentile(hp, 2)) / max(.5, np.ptp(hp)), 0, 1)
    rgb = np.stack([shade * (150 + 80 * rel), shade * (150 + 50 * rel), shade * (130 + 20 * rel)], -1)
    up = lambda t: t.repeat(2, 0).repeat(2, 1)[:rgb.shape[0], :rgb.shape[1]]
    if a['water'] is not None:
        w = up(a['water'])
        rgb[w] = rgb[w] * .35 + np.array([30, 90, 200]) * .65
    for bit, col in ((2, (255, 150, 0)), (1, (255, 30, 30)), (4, (255, 0, 255))):
        d = up((a['defect'] & bit) > 0)
        rgb[d] = rgb[d] * .3 + np.array(col) * .7
    out = ~up(a['inside'])
    rgb[out] *= .35
    rgb = rgb.astype(np.uint8)
    for x, y in a['wet']:
        i, j = int(y * 2), int(x * 2)
        rgb[max(0, i - 1):i + 2, max(0, j - 1):j + 2] = (255, 0, 255)
    for q, r in a['towns']:
        i, j = (2 * r + q % 2) * 2, 2 * q * 2
        for di in range(-1, 6):
            for dj in (-1, 5):
                if 0 <= i + di < rgb.shape[0] and 0 <= j + dj < rgb.shape[1]:
                    rgb[i + di, j + dj] = 255
                if 0 <= j + di < rgb.shape[1] and 0 <= i + dj < rgb.shape[0]:
                    rgb[i + dj, j + di] = 255
    return rgb[::-1]


def contact_sheet(items, path, cols=5, w=300, h=230):
    from PIL import Image, ImageDraw
    rows = (len(items) + cols - 1) // cols
    sheet = Image.new('RGB', (cols * w, rows * h), (12, 12, 12))
    d = ImageDraw.Draw(sheet)
    for k, (name, png, tag) in enumerate(items):
        im = Image.open(png); im.thumbnail((w - 8, h - 24))
        x, y = (k % cols) * w, (k // cols) * h
        sheet.paste(im, (x + 4, y + 20))
        d.text((x + 4, y + 4), name + (f'  {tag}' if tag else ''), fill=(255, 90, 90) if tag else (230, 230, 230))
    sheet.save(path)


# --------------------------------------------------------------------------- snapshot
def snapshot(args):
    out = args.out.resolve()
    if args.workspace.resolve() in (out, *out.parents):
        sys.exit('--out must lie outside the workspace')
    (out / 'windows').mkdir(parents=True, exist_ok=True)
    try:
        S = Snapshot(args.workspace, args.strict)
    except Exception as e:   # campaign_world needs the workspace's design file
        if args.strict:
            raise
        note = f'workspace unusable, nothing measured: {type(e).__name__}: {e}'
        print('note:', note)
        (out / 'metrics.json').write_text(json.dumps({'kind': 'terrain-regression-snapshot', 'version': 1, 'workspace': str(args.workspace),
                                                      'windows': {}, 'global': {}, 'notes': [note]}, indent=1) + '\n', encoding='utf-8')
        np.savez_compressed(out / 'windows.npz')
        return
    S.section('inputs', S.load)
    canvas = {'grid': [cw.W, cw.H], 'padding': [cw.PAD_LEFT, cw.PAD_BOTTOM, cw.PAD_RIGHT, cw.PAD_TOP]}
    if canvas != {'grid': [2048, 2047], 'padding': [888, 375, 268, 970]}:
        S.note(f'canvas {canvas} differs from the one the windows were chosen on (2048x2047, pad 888/375/268/970)')
    p = S.path('analysis/native-edge-audit.json')
    if p:
        ea = json.loads(p.read_text(encoding='utf-8'))
        S._edge_fail = [(tuple(s['hex']), s.get('fails')) for e in ea.get('edges', {}).values() for s in e.get('failing', [])]
    boxes = []
    for _, parts, _, _ in WINDOWS:
        for p in parts:
            q0, q1, r0, r1 = part_box(p)
            boxes.append((q0 * cw.DX, (q1 + 1) * cw.DX, r0 * cw.DX, (r1 + 1.5) * cw.DX))
    tris = S.section('meshes', S.load_meshes, boxes)
    report = {'kind': 'terrain-regression-snapshot', 'version': 1, 'workspace': str(S.ws), 'variant': VARIANT, 'canvas': canvas,
              'created': time.strftime('%Y-%m-%d %H:%M:%S'), 'windows': {}}
    arrays, sheet = {}, []
    k = 0
    for name, parts, watch, extra in WINDOWS:
        tb = tris[k:k + len(parts)] if tris is not None else None
        k += len(parts)
        res = S.section(f'window {name}', window_metrics, S, name, parts, extra, tb)
        if res is None:
            report['windows'][name] = {'watch': watch, 'parts': [list(part_box(p)) for p in parts], 'metrics': None}
            continue
        met, P, arr = res
        report['windows'][name] = {'watch': watch, 'parts': [list(x.box) for x in P], 'gate': list(extra.get('gate', ())), 'metrics': met}
        for i, a in enumerate(arr):
            for key in ('lf', 'mat', 'water', 'defect', 'inside'):
                if a[key] is not None:
                    arrays[f'{name}|{i}|{key}'] = a[key]
        try:
            png = out / 'windows' / f'{name}.png'
            compose([render_part(S, x, a) for x, a in zip(P, arr)]).save(png)
            sheet.append((name, png, ''))
        except Exception as e:
            S.note(f'render {name} failed: {e}')
    S.blend_px = None
    report['global'] = S.section('global', global_metrics, S) or {}
    np.savez_compressed(out / 'windows.npz', **arrays)
    if sheet:
        contact_sheet(sheet, out / 'contact-sheet.png')
    report.update({'inputs': S.inputs, 'notes': S.notes, 'runtime_s': round(time.time() - S.t0, 1), 'peak_mb': peak_mb()})
    (out / 'metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)) + '\n', encoding='utf-8')
    print(f"snapshot: {len(report['windows'])} windows, {len(S.notes)} notes, {report['runtime_s']} s, peak {report['peak_mb']} MB -> {out / 'metrics.json'}")


# --------------------------------------------------------------------------- compare
def flatten(rep):
    """metric path -> (leaf rule key, value, gated)."""
    out = {}
    for name, w in rep.get('windows', {}).items():
        met = w.get('metrics') or {}
        gate = set(w.get('gate') or ())
        for key, v in met.items():
            if key == 'towns':
                for t, d in v.items():
                    out[f'windows.{name}.towns.{t}.range'] = ('town_range', d.get('range'), True)
                    out[f'windows.{name}.towns.{t}.plot3_range'] = ('town_plot3_range', d.get('plot3_range'), True)
                    out[f'windows.{name}.towns.{t}.median'] = ('town_median', d.get('median'), False)
            elif key in ('town_positions', 'ford_groups'):
                out[f'windows.{name}.{key}'] = (key, v, key == 'town_positions')
            else:
                out[f'windows.{name}.{key}'] = (key, v, key not in GATED_ONLY or key in gate)
    for sec, d in rep.get('global', {}).items():
        for key, v in (d or {}).items():
            if sec == 'biomes':
                for b, mets in v.items():
                    for kk, vv in mets.items():
                        out[f'global.biomes.{key}.{b}.{kk}'] = (biome_rule(key, b, kk), vv, True)
            elif isinstance(v, dict):
                for kk, vv in v.items():
                    out[f'global.{sec}.{key}.{kk}'] = ('dry_river_centres' if key == 'dry_by_river' else kk, vv, True)
            else:
                out[f'global.{sec}.{key}'] = (key, v, True)
    return out


def biome_rule(side, biome, metric):
    """Gated per-biome shares on the new land: green in dry biomes up, forest in plains up, mud up,
    farmland down. Everything else (and the native reference) is information."""
    if side != 'new':
        return 'info'
    if metric == 'green_share' and biome in DRY_BIOMES:
        return 'biome_green_dry'
    if metric in ('forest4_share', 'woodland_share') and biome in PLAIN_BIOMES:
        return 'biome_forest_plain'
    return {'mud_share': 'biome_mud', 'farm_share': 'biome_farm'}.get(metric, 'info')


def judge(rule_key, b, n, gated):
    """('regression' | 'improved' | 'changed' | 'same' | 'skipped', delta)."""
    if b is None or n is None:
        return ('skipped' if b is None and n is None else 'unmeasured'), None
    if isinstance(b, list) or isinstance(n, list):
        if rule_key == 'town_positions':
            return ('regression' if set(b) != set(n) and gated else ('same' if b == n else 'changed')), None
        return ('same' if b == n else 'changed'), None
    if isinstance(b, str) or isinstance(n, str):
        return ('same' if b == n else 'changed'), None
    b, n = float(b), float(n)
    d = n - b
    if d == 0:
        return 'same', 0.
    rule = RULES.get(rule_key, (0, 0, 0))
    if rule_key.startswith('seam.') and rule_key not in RULES:
        rule = (1, 0, 0)
    sign, tol_a, tol_r = rule
    if not sign or not gated:
        return 'changed', d
    tol = max(tol_a, tol_r * abs(b))
    if sign * d > tol:
        return 'regression', d
    return ('improved' if sign * d < 0 else 'changed'), d


def get(rep, path):
    cur = rep
    for part in path.split('.'):
        if part == 'windows' and cur is rep:
            cur = rep.get('windows', {}); continue
        if isinstance(cur, dict) and 'metrics' in cur and part not in cur:
            cur = cur['metrics'] or {}
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def check_expect(rep, op, value, path):
    v = get(rep, path)
    if v is None:
        return None, v
    return {'==': v == value, '<=': v <= value, '>=': v >= value}[op], v


def side_by_side(base, new, name, nb, nn, out):
    """base | new | diff (LF change per tile: red up, blue down, +-0.5 full; yellow = new defect) for one window."""
    from PIL import Image
    pb, pn = base / 'windows' / f'{name}.png', new / 'windows' / f'{name}.png'
    panels = []
    for p in (pb, pn):
        panels.append(np.asarray(Image.open(p).convert('RGB')) if p.is_file() else np.zeros((240, 240, 3), np.uint8))
    diffs = []
    i = 0
    while f'{name}|{i}|lf' in nb and f'{name}|{i}|lf' in nn:
        lb, ln = nb[f'{name}|{i}|lf'], nn[f'{name}|{i}|lf']
        if lb.shape != ln.shape:
            break
        d = np.clip((ln - lb) / .5, -1, 1)
        rgb = np.full(d.shape + (3,), 60, np.float32)
        rgb[..., 0] += np.maximum(d, 0) * 195; rgb[..., 2] += np.maximum(-d, 0) * 195
        rgb[..., 1] += 0
        db, dn = nb.get(f'{name}|{i}|defect'), nn.get(f'{name}|{i}|defect')
        if db is not None and dn is not None:
            rgb[(dn > 0) & (db == 0)] = (255, 230, 0)
        diffs.append(rgb.astype(np.uint8)[::-1].repeat(2, 0).repeat(2, 1))
        i += 1
    im = compose(panels + ([np.asarray(compose(diffs))] if diffs else []), height=300)
    im.save(out / f'{name}.png')


def compare(args):
    base, new = args.base.resolve(), args.new.resolve()
    rb = json.loads((base / 'metrics.json').read_text(encoding='utf-8'))
    rn = json.loads((new / 'metrics.json').read_text(encoding='utf-8'))
    out = (args.out or new / 'compare').resolve()
    (out / 'side-by-side').mkdir(parents=True, exist_ok=True)
    fb, fn = flatten(rb), flatten(rn)
    rows, flagged_windows = [], set()
    for path in sorted(set(fb) | set(fn)):
        key, vb, gated = fb.get(path, fn.get(path))
        key2, vn, gated2 = fn.get(path, (key, None, gated))
        if path not in fb:
            vb = None
        status, d = judge(key, vb, vn, gated and gated2)
        if status == 'same' or status == 'skipped':
            continue
        if status == 'regression' and key.startswith('patch_'):
            status = 'warning'          # ground patchiness is warn-only
        rows.append({'metric': path, 'base': vb, 'new': vn, 'delta': rnd(d) if d is not None else None, 'status': status})
        if status == 'regression' and path.startswith('windows.'):
            flagged_windows.add(path.split('.')[1])
    expect = []
    for path, op, value, note in EXPECT:
        ok_n, vn = check_expect(rn, op, value, path)
        ok_b, vb = check_expect(rb, op, value, path)
        if ok_n is None:
            status = 'unmeasured'
        elif ok_n:
            status = 'ok'
        elif ok_b is False and judge(path.split('.')[-1], vb, vn, True)[0] != 'regression':
            status = 'known'           # failed in the baseline too, not worse
        else:
            status = 'regression'
            if path.startswith('windows.'):
                flagged_windows.add(path.split('.')[1])
        expect.append({'rule': f'{path} {op} {value}', 'note': note, 'base': vb, 'new': vn, 'status': status})
    # tiles / materials changed per window (snapshot arrays)
    changed = {}
    nb, nn = {}, {}
    for d, z in ((nb, base / 'windows.npz'), (nn, new / 'windows.npz')):
        if z.is_file():
            with np.load(z) as f:
                d.update({k: f[k] for k in f.files})
    for name in rn.get('windows', {}):
        c = {'lf_tiles_changed': 0, 'material_tiles_changed': 0}
        i = 0
        while f'{name}|{i}|inside' in nb and f'{name}|{i}|inside' in nn:
            ins = nb[f'{name}|{i}|inside']
            for key, ck, tol in (('lf', 'lf_tiles_changed', .005), ('mat', 'material_tiles_changed', 0)):
                a, b = nb.get(f'{name}|{i}|{key}'), nn.get(f'{name}|{i}|{key}')
                if a is not None and b is not None and a.shape == b.shape:
                    c[ck] += int((np.abs(a.astype(np.float32) - b.astype(np.float32)) > tol)[ins].sum())
            i += 1
        changed[name] = c
    for name in sorted(flagged_windows):
        try:
            side_by_side(base, new, name, nb, nn, out / 'side-by-side')
        except Exception as e:
            print('side-by-side', name, 'failed:', e)
    regress = [r for r in rows if r['status'] == 'regression'] + [e for e in expect if e['status'] == 'regression']
    lines = [f'terrain regression compare  base {base}  new {new}', '']
    fmt = lambda v: '-' if v is None else (f'{v:.4g}' if isinstance(v, float) else (str(len(v)) + ' items' if isinstance(v, list) and len(str(v)) > 30 else str(v)))
    order = {'regression': 0, 'warning': 1, 'unmeasured': 2, 'changed': 3, 'improved': 4}
    lines.append(f"{'status':11s} {'metric':72s} {'base':>12s} {'new':>12s} {'delta':>10s}")
    for r in sorted(rows, key=lambda r: (order.get(r['status'], 9), r['metric'])):
        lines.append(f"{r['status']:11s} {r['metric'][:72]:72s} {fmt(r['base']):>12s} {fmt(r['new']):>12s} {fmt(r['delta']):>10s}")
    lines += ['', 'expectations:']
    for e in expect:
        lines.append(f"{e['status']:11s} {e['rule']:60s} base {fmt(e['base'])}  new {fmt(e['new'])}  ({e['note']})")
    patch = patch_vs_baseline(rn)
    lines += ['', f'ground patchiness, warn-only (vs v18 {PATCH_BASELINE.name}; lit = above the vanilla alarm level):']
    lines += [f"  {r['status']:11s} {r['window'] + '.' + r['metric']:60s} v18 {fmt(r['v18'])}  new {fmt(r['new'])}  alarm {fmt(r['alarm'])}"
              + ('  LIT' if r['lit'] else '') for r in patch] or ['  none']
    lines += ['', 'changed tiles per window (|dLF| > 0.005 / material):']
    lines += [f"  {n:24s} lf {c['lf_tiles_changed']:6d}  material {c['material_tiles_changed']:6d}" for n, c in changed.items() if any(c.values())] or ['  none']
    notes = [f'base: {n}' for n in rb.get('notes', [])] + [f'new: {n}' for n in rn.get('notes', [])]
    if notes:
        lines += ['', 'snapshot notes:'] + ['  ' + n for n in notes]
    warns = sum(r['status'] == 'warning' for r in rows) + sum(r['status'] == 'warning' for r in patch)
    lines += ['', f'{len(regress)} regression(s)' + (f" in windows: {', '.join(sorted(flagged_windows))}" if flagged_windows else '')
              + (f"; side-by-side PNGs in {out / 'side-by-side'}" if flagged_windows else '') + f'; {warns} patchiness warning(s) (warn-only)']
    text = '\n'.join(lines)
    print(text)
    (out / 'compare.txt').write_text(text + '\n', encoding='utf-8')
    (out / 'compare.json').write_text(json.dumps({'base': str(base), 'new': str(new), 'regressions': len(regress), 'flagged_windows': sorted(flagged_windows),
                                                  'rows': rows, 'expectations': expect, 'patchiness': patch, 'changed_tiles': changed, 'notes': notes},
                                                 ensure_ascii=False, indent=1, default=str) + '\n', encoding='utf-8')
    return 1 if regress and not args.warn_only else 0


# --------------------------------------------------------------------------- copy-inputs
def copy_inputs(args):
    """Copy INPUTS (mtimes kept) from --workspace into --to, only while the pipeline locks are absent."""
    src, dst = args.workspace.resolve(), args.to.resolve()
    import pipeline_lock
    busy = lambda: pipeline_lock.busy(src.parent, pipeline_lock.PIPELINE)
    t0 = time.time()
    while busy():
        if time.time() - t0 > args.wait:
            sys.exit(f'pipeline locks still held after {args.wait} s: {busy()}')
        time.sleep(10)
    files = sorted({p for pat in INPUTS for p in src.glob(pat) if p.is_file()})
    total = 0
    for p in files:
        o = dst / p.relative_to(src)
        o.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, o)
        total += p.stat().st_size
        if busy():
            sys.exit(f'a pipeline lock appeared during the copy ({busy()}); {dst} is not consistent')
    manifest = {'source': str(src), 'files': len(files), 'bytes': total, 'seconds': round(time.time() - t0, 1),
                'missing': [pat for pat in INPUTS if not any(src.glob(pat))]}
    (dst / 'terrain-regression-copy.json').write_text(json.dumps(manifest, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(manifest))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('snapshot'); s.add_argument('--out', type=Path, required=True)
    s.add_argument('--workspace', type=Path); s.add_argument('--strict', action='store_true', help='raise instead of skipping a failing section')
    c = sub.add_parser('compare'); c.add_argument('base', type=Path); c.add_argument('new', type=Path)
    c.add_argument('--out', type=Path); c.add_argument('--warn-only', action='store_true', help='exit 0 even with regressions')
    k = sub.add_parser('copy-inputs'); k.add_argument('--to', type=Path, required=True); k.add_argument('--workspace', type=Path)
    k.add_argument('--wait', type=float, default=3600)
    args = ap.parse_args()
    if getattr(args, 'workspace', None):
        os.environ['TW3K_WORKSPACE'] = str(args.workspace.resolve())
        os.environ.pop('TW3K_DESIGN', None)
    else:
        args.workspace = Path(os.environ.get('TW3K_WORKSPACE', ROOT / 'work/all-directions-30'))
    if args.cmd == 'compare':
        sys.exit(compare(args))
    if args.cmd == 'copy-inputs':
        return copy_inputs(args)
    snapshot(args)


if __name__ == '__main__':
    main()
