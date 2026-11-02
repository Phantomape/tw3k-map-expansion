"""Stage invariants: cheap read-only checks after each terrain pipeline step, so a step that undoes an
earlier step's fix fails at that step, by name, instead of in an end-of-run audit or a live test.

Why: a fix kept getting undone by a later step and patched again further downstream (the Lin Yi seabed
holes: polish floored the sea, repair_native_placements kept vanilla generic_sea, clear_obsolete_terrain_
overlays culled it and refilled land only, 38,458 sea cells without a tile for three releases; dry river
centres beside towns: RIVER CHANNELS carved them, the town slope projection lifted them again; desert green
specks; rocks over the flat Dunhuang basin). Every such fix gets an invariant here; the pipeline checks the
applicable ones after each step and stops at the first step where one fails.

  python stage_invariants.py --after STEP [--workspace WS] [--run ID] [--warn-only] [--only A,B]
  python stage_invariants.py --after map --map PATH   (any map.hex; read-only: prints, writes no record)
  python stage_invariants.py --list
  python stage_invariants.py --build-cache [--workspace WS]   (the desert landform classes: ~40 s / 6 GB, once)

STEPS (checkpoints, in pipeline order; build_polished_campaign_terrain.ps1 calls --after after each):
  map          the input map.hex, at the pipeline start (a guard: the map stage wrote it, not this pipeline)
  geometry     after terrain_checkpoint.py capture geometry (-RebuildGeometry): the refreshed terrain-before-polish
  polish       polish_campaign_terrain.py (and its checkpoint / early edge audit)
  sync_final   sync_final_terrain_surface.py (boundary smoothing, polish --cache-only), the authored camera
               sync and the side lane (sea height, water colour, spheres, ocean background)
  resource_scenes  in the graph, graph node scene-check after world-space-leaves (a65cd554): the props as the
               resource scenes seated them, with the scene roots' translation baked into the leaves (earlier, every
               copy of a template still reads at its donor position: v21b run 20261031-034627, 92 false scene
               failures), against the final LF (gap-fill is still rebuilding the meshes). The name stays for the
               records; the writer it checks is complete_campaign_resource_scenes
  graph        the step graph (foliage, prefab chain, placements, overlays, ...), at its join: a failure is
               attributed to the graph nodes that write the invariant's artifacts (GRAPH_WRITERS)
  seam_pass0   the first ground-seam scan (audit_ground_seams.py pass 0)
  mesh_final   after the seam repair loop and patch_coverage_holes: the last land-mesh writer (repair passes re-
               partition triangles: run 20261030-123911 pass 1 added 0.42M triangles to 62 meshes after seam_pass0)

Kinds: exact (value above its threshold fails), metric (also compared with the previous checkpoint of this
run: a regression beyond its tolerance fails; with no earlier checkpoint in the run, the same step of the
previous run is the reference and a regression there only warns, since maps and designs change between
runs). warn_only invariants (thresholds not calibrated yet) never fail. A rollout invariant (ROLLOUT) only
warns until its own setting says hard: river_centre_wet, TW3K_INVARIANT_RIVER_CENTRE_WET=hard (default warn).
A check that cannot run (missing artifact, checker error) is recorded as skip / error and never fails the pipeline.

Record: one JSON line per invariant and checkpoint in <ws>/analysis/stage-invariants.jsonl (run id from
--run, else TW3K_INVARIANT_RUN, set by the ps1 per pipeline run), plus a '*' line with the checkpoint's
cost. TW3K_INVARIANTS=warn makes every failure a warning, =off skips the checks.
Exit: 0 pass / warnings, 1 a hard invariant failed (the message names the step, the invariant and the
writers of its artifacts), 3 the checker itself failed (the ps1 only warns).

Read-only: workspace files are read whole into memory (no mmap, nothing kept open) and closed; the only
files written are the record and the landform cache analysis/stage-invariants-landforms.npz (no pipeline
step reads either).
"""
import argparse, hashlib, json, os, re, struct, sys, time, traceback
from pathlib import Path

import numpy as np

STEPS = ('map', 'geometry', 'polish', 'sync_final', 'resource_scenes', 'graph', 'seam_pass0', 'mesh_final')
STEP_SCRIPTS = {'map': 'the map stage (map.hex writers: apply_rivers, apply_ground_types, apply_forest_ground, apply_climate, '
                       'apply_shore_kinds, ...)',
                'geometry': 'terrain_checkpoint.py capture geometry (the SDK export and its base refresh)',
                'polish': 'polish_campaign_terrain.py',
                'sync_final': 'sync_final_terrain_surface.py (boundary smoothing + polish --cache-only with the tile bake: tile lists), sync_authored_camera_heights.py, side lane',
                'resource_scenes': 'complete_campaign_resource_scenes.py (graph node resource-scenes; checked by graph node scene-check)',
                'graph': 'the terrain step graph',
                'seam_pass0': 'the stage the first seam scan reads (LF: polish / sync_final; tile ranges: graph nodes '
                              'calibrated-ranges, cache-heights; tile lists: the sync_final tile bake)',
                'mesh_final': 'the seam repair loop (repair_local_coarse_seams.py exact partition, '
                              'normalize_coarse_partition_heights.py) and patch_coverage_holes.py'}
VARIANT = '3k_dlc07_main_map'   # the 190 campaign's only terrain variant (campaign_world.TERRAIN_VARIANTS)
PRE = f'work/release-input/terrain/campaigns/{VARIANT}/'
MAP = 'work/final-map/map.hex'
POLISH_REPORT = 'analysis/geographic-terrain-polish.json'   # polish approach report (towns: plateau heights)
CHANNELS = 'work/final-map/river-channels.json'   # apply_rivers / bake_map step 7
LF = PRE + 'lf_height_map.dds'
TILES = PRE + 'tile_list.bin'
GTILES = PRE + 'global_map/tile_list.bin'
TILEMAP = PRE + 'tile_map.tiles'   # the display cache (layer 0 = the fine tiles the campaign draws)
BLEND = PRE + 'global_map/global_blend.dds'
TREES = f'work/release-input/campaign_maps/{VARIANT}/display/trees/trees.campaign_tree_list'
LEVEL = 'analysis/river-water-level.npy'
REAL = 'analysis/west-real-terrain.npz'
OVERLAY = 'analysis/native-overlay-preservation.json'
EDGE_AUDIT = 'analysis/native-edge-audit.json'   # audit_native_edges.py (the early edge audit after polish)
SEAM0 = f'analysis/ground-seam-pipeline-{VARIANT}-0/ground-seam-scan.json'
MESHES = PRE + 'global_meshes/'   # land_mesh_*.rigid_model_v2: the coarse ground the game draws outside the fine tiles
RECORD = 'analysis/stage-invariants.jsonl'
LANDFORM_CACHE = 'analysis/stage-invariants-landforms.npz'
LANDFORM_TILES = 'analysis/landform-tiles.npz'          # landform_tiles.py (polish GROUND MATERIALS)
BLEND_CLIMATE = 'analysis/blend-climate-tiles.npz'      # blend_climate_tiles.py (polish, after global_blend)
# Graph nodes (Invoke-Tw3kGraph in the ps1) that write each artifact an invariant reads at the join.
# The tile lists are written by the tile bake in sync_final (tile_bake.py); in the graph only calibrated-ranges (lo/hi).
FOLIAGE_REPORT = 'analysis/terrain-native-foliage.json'   # polish_campaign_foliage report
GRAPH_WRITERS = {TILES: ('calibrated-ranges',), GTILES: ('calibrated-ranges',),
                 TILEMAP: ('cache-heights',), TREES: ('foliage',), OVERLAY: ('overlay-audit',), FOLIAGE_REPORT: ('foliage',)}   # tile_map.tiles: cache-heights rewrites ranges (the display-cache node is gone since step 0)
# Graph nodes that write tile_list.bin after overlay-audit (they may change lo/hi only).
TILES_AFTER_OVERLAY_AUDIT = ('calibrated-ranges',)

PLANE = -.015                    # campaign water plane (terrain_regression.PLANE, polish LAKE_WATER)
VANILLA_SEA_HOLES = 134          # vanilla dlc07: open-sea cells without a floor record (same definition, 2026-10-25)
B = chr(92)
SIZE = re.compile(r'(\d+)x(\d+)')     # a floor's footprint from its last path part (generic_16x16, generic_sea 8x8)
QUARTER_TURN = 0x20 | 0x80       # native_tile_dimensions.QUARTER_TURN_FLAGS: footprint w/h swap


class Missing(Exception):
    pass


# --------------------------------------------------------------------------- inputs
class Ctx:
    """Workspace inputs, each read once per process (whole file into memory, then closed)."""

    def __init__(self, ws):
        self.ws, self._c, self.reads = ws, {}, {}
        self.over = {}   # rel -> another file (--map)

    def path(self, rel):
        p = self.over.get(rel) or self.ws / rel
        if not p.is_file():
            raise Missing(rel)
        st = p.stat()
        self.reads[rel] = {'bytes': st.st_size, 'mtime': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(st.st_mtime))}
        return p

    def get(self, key, fn):
        if key not in self._c:
            self._c[key] = fn()
        return self._c[key]

    def map(self):
        def load():
            import map_hex
            m = map_hex.read(self.path(MAP))
            return m['cells'], m['land_regions'] + m['sea_regions']
        return self.get('map', load)

    def lf(self):
        """Stage LF raw uint16, south-up (row 0 = south), as terrain_regression reads it (not kept: 0.5 GB)."""
        d = self.path(LF).read_bytes(); h, w = struct.unpack_from('<2I', d, 12)
        return np.frombuffer(d, '<u2', offset=128, count=h * w).reshape(h, w)[::-1]

    def tiles(self, rel):
        def load():
            from terrain_regression import parse_tile_list
            return parse_tile_list(self.path(rel).read_bytes())
        return self.get(rel, load)


def tile_hex_grid(TH, TW, H, W):
    """(rows r, cols q) int32 of the hex owning each tile cell (campaign_world.tile_hex, broadcast)."""
    q = np.minimum(np.arange(TW, dtype=np.int32) // 2, W - 1)
    r = np.clip((np.arange(TH, dtype=np.int32)[:, None] - (q % 2)[None, :]) // 2, 0, H - 1)
    return r, np.broadcast_to(q[None, :], (TH, TW))


# --------------------------------------------------------------------------- checks
def sea_floor_coverage(ctx):
    """Open-sea tile cells (map.hex sea hexes outside river channels and lakes, sea_floor_tiles.open_sea_hexes)
    that no floor record (sea_floor_tiles.FLOOR_FAMILY, footprint from the name, quarter turns swap w/h) of
    the tile list covers; the larger count of the local and global list. Vanilla dlc07: 134 (all inside its
    sea, none from the edits); v16 (the Lin Yi strip): 38,244."""
    import campaign_world as cw, sea_floor_tiles as sf
    cells, names = ctx.map()
    H, W = cells.shape[:2]; TH, TW = 2 * H + 1, 2 * W
    r, q = tile_hex_grid(TH, TW, H, W)
    sea = sf.open_sea_hexes(cells, names)[r, q]
    nat = np.zeros((TH, TW), bool); nat[cw.NATIVE_TILE_SLICE] = True
    detail = {}; last = None
    for label, rel in (('local', TILES), ('global', GTILES)):
        if rel == GTILES and not (ctx.ws / rel).is_file():
            detail['global'] = 'missing'; continue
        paths, rec = ctx.tiles(rel)
        dims = np.zeros((len(paths), 2), np.int64)
        for i, p in enumerate(paths):
            m = SIZE.search(p.rstrip(B).split(B)[-1]) if sf.FLOOR_FAMILY.search(p.replace(B, '/')) else None
            if m:
                dims[i] = int(m.group(1)), int(m.group(2))
        pi = rec['path'].astype(np.int64)
        sel = dims[pi, 0] > 0
        w, h = dims[pi[sel], 0], dims[pi[sel], 1]
        turn = (rec['flags'][sel] & QUARTER_TURN) > 0
        w, h = np.where(turn, h, w), np.where(turn, w, h)
        x, y = rec['x'][sel].astype(np.int64), rec['y'][sel].astype(np.int64)
        fp = np.stack((x, y, w, h))
        if last is not None and last[0].shape == fp.shape and np.array_equal(last[0], fp):
            detail[label] = {**last[1], 'same_floors_as': 'local'}; continue   # the lists differ in flags only
        x0, x1 = np.clip(x, 0, TW), np.clip(x + w, 0, TW); y0, y1 = np.clip(y, 0, TH), np.clip(y + h, 0, TH)
        # footprint counts by 2D prefix sums of the corner marks (+1 at (y0, x0), (y1, x1); -1 at (y0, x1), (y1, x0))
        n = (TH + 1) * (TW + 1)
        plus = np.bincount(np.concatenate((y0 * (TW + 1) + x0, y1 * (TW + 1) + x1)), minlength=n).astype(np.int32)
        plus -= np.bincount(np.concatenate((y0 * (TW + 1) + x1, y1 * (TW + 1) + x0)), minlength=n).astype(np.int32)
        cov = plus.reshape(TH + 1, TW + 1).cumsum(0, dtype=np.int32).cumsum(1, dtype=np.int32)[:TH, :TW]
        holes = sea & (cov == 0)
        detail[label] = {'holes': int(holes.sum()), 'native': int((holes & nat).sum()), 'canvas': int((holes & ~nat).sum()),
                         'floor_records': int(sel.sum())}
        last = (fp, detail[label])
        del plus, cov, holes
    return {'value': max(v['holes'] for v in detail.values() if isinstance(v, dict)), 'detail': {**detail, 'open_sea_cells': int(sea.sum())}}


def river_centre_wet(ctx):
    """Frontier river hexes (map.hex sea regions tw3k_riv_sea_*) whose centre LF pixel (tile 2q+1, 2r+odd(q)+1,
    LF 4x that) is at or above the water there (max of analysis/river-water-level.npy and the plane):
    terrain_regression's river_dry_centres over the whole map. State: the dry hexes (r * W + q)."""
    import campaign_world as cw
    cells, names = ctx.map()
    H, W = cells.shape[:2]
    lv = np.load(ctx.path(LEVEL), allow_pickle=False)
    if lv.shape != (cw.TILE_H, cw.TILE_W):
        raise Missing(f'{LEVEL} shape {lv.shape} is not the tile grid')
    ids = (cells[:, :, 1].astype(np.int64) << 5) + (cells[:, :, 0].astype(np.int64) >> 3) - 1
    isriv = np.array([n.startswith('tw3k_riv_sea_') for n in names] + [False])
    river = ((cells[:, :, 0] & 3) == 1) & isriv[np.where((ids >= 0) & (ids < len(names)), ids, len(names))]
    rr, qq = np.nonzero(river)
    cy, cx = np.minimum(2 * rr + qq % 2 + 1, cw.TILE_H - 1), np.minimum(2 * qq + 1, cw.TILE_W - 1)
    raw = ctx.lf()[4 * cy, 4 * cx]
    hgt = cw.LEVEL + (raw.astype(np.float32) / 65535 - cw.NORM) * cw.GAIN
    dry = hgt >= np.fmax(lv[cy, cx], np.float32(PLANE))
    ids_dry = (rr[dry] * W + qq[dry]).astype(int).tolist()
    return {'value': int(dry.sum()), 'detail': {'river_hexes': int(len(rr)), 'dry_centres': int(dry.sum())}, 'state': ids_dry}


def river_regress(res, ref):
    """Newly dry centres against the reference record's dry set."""
    if ref.get('state') is None:
        return res['value'] - ref['value']
    return len(set(res['state']) - set(ref['state']))


def landform_classes(ctx, build=True):
    """uint8 tile grid: index of the hard landform class (audit_terrain_realism.real_landforms: classify argmax,
    weight > .5, >= REAL_GROUND_FULL tiles from the native map, real weight > .5; land applied later), 255
    elsewhere; and the names. Cached in LANDFORM_CACHE under a key of the real-terrain npz (size, mtime), the
    classify code and the canvas: classify costs ~20 s and ~6 GB, so a checkpoint reuses the cache."""
    import campaign_world as cw
    src = ctx.path(REAL); st = src.stat()
    here = Path(__file__).resolve().parent
    code = hashlib.sha256(b''.join((here / n).read_bytes() for n in ('biome_profiles.py', 'audit_terrain_realism.py', 'oasis_ground.py')
                                   if (here / n).is_file())).hexdigest()
    key = json.dumps([st.st_size, st.st_mtime_ns, code, cw.W, cw.H, cw.PAD_LEFT, cw.PAD_BOTTOM,
                      list(cw.load_authored_coordinates()['noise_origin'])])
    cache = ctx.ws / LANDFORM_CACHE
    if cache.is_file():
        with np.load(cache, allow_pickle=False) as z:
            if str(z['key']) == key:
                return z['cls'], [str(n) for n in z['names']], False
    if not build:
        raise Missing(f'{LANDFORM_CACHE} (stage_invariants.py --build-cache)')
    import audit_terrain_realism as atr
    with np.load(src) as z:
        zz = {k: z[k] for k in ('biome_names', 'biome', 'elevation', 'lon', 'lat', 'weight')}
    names, top, topw, real = atr.real_landforms(np.ones(zz['biome'].shape, bool), zz)
    del zz
    cls = np.where(real & (topw > .5), top, 255).astype(np.uint8)
    tmp = cache.with_name(cache.name + '.tmp')
    with open(tmp, 'wb') as f:
        np.savez_compressed(f, cls=cls, names=np.array(names), key=np.array(key))
    os.replace(tmp, cache)
    return cls, names, True


def desert_banned_share(ctx):
    """Largest banned-slot share away from oases on a desert floor landform (audit_terrain_realism --gate:
    'banned_open' on GATE_LANDFORMS, GATE_BANNED .002): the stage global_blend at tile centres on real canvas
    ground of each landform, outside the audit's oasis mask (real water / towns nearby). Warn only: the generator's own
    mask (ground_materials.tile_fields 'L': class x gain x real-ground reach, oasis excluded) lives inside polish
    and is not saved, so this is the gate's definition, not a third one; the post-repair --gate stays the hard gate."""
    import audit_terrain_realism as atr
    cells, _ = ctx.map()
    H, W = cells.shape[:2]
    d = ctx.path(BLEND).read_bytes(); bh, bw = struct.unpack_from('<2I', d, 12)
    TH, TW = 2 * H + 1, 2 * W
    if (bw, bh) != (4 * TW, 4 * TH):
        raise Missing(f'global_blend is {bw}x{bh}')
    mat = np.ascontiguousarray(np.frombuffer(d, np.uint8, offset=128, count=bh * bw * 2).reshape(bh, bw, 2)[::-1][2::4, 2::4, 0])
    del d
    cls, names, built = landform_classes(ctx)
    q, r = atr.tile_hexes(TH, TW, H, W)
    land = ((cells[:, :, 0] & 3) != 1)[r, q]; town = ((cells[:, :, 2] >> 4) == 1)[r, q]; del q, r
    with np.load(ctx.path(REAL)) as z:
        water = z['lake'] | z['river'] | z['salt']
    oasis = atr.oasis_tiles(water, land, town); del water, town
    real = land & (cls != 255)
    shares = atr.desert_banned_shares(mat, names, cls, np.ones(cls.shape, np.float32), real, oasis)
    vals = [v for v in shares.values() if v is not None]
    return {'value': max(vals) if vals else 0., 'detail': {'shares': shares, 'landform_cache_built': built}}


# The land meshes of the base snapshot (terrain-before-polish): the game packs' land_mesh_0..169 and the canvas meshes
# 170..183 of the exported canvas; nothing else (mesh bake 3.2.1 overflow files are graph output).
BASE_LAND_MESHES = 184


def base_land_mesh_set(ctx):
    """land_mesh files of the base snapshot (terrain-before-polish) outside land_mesh_0..BASE_LAND_MESHES-1, or missing
    from it (.rigid_model_v2 and .compressed_map each): 0. On 10-30 polish's snapshot loop seeded the previous run's
    overflow files (184..260) into the base, and every later run reset them as models (land_mesh_budget 15.5M)."""
    d = ctx.ws / 'terrain-before-polish' / f'terrain/campaigns/{VARIANT}/global_meshes'
    if not d.is_dir():
        raise Missing(str(d))
    want = {f'land_mesh_{i}{ext}' for i in range(BASE_LAND_MESHES) for ext in ('.rigid_model_v2', '.compressed_map')}
    have = {f.name for f in d.glob('land_mesh_*')}
    extra, missing = sorted(have - want), sorted(want - have)
    return {'value': len(extra) + len(missing), 'detail': {'extra': extra[:40], 'missing': missing[:40], 'files': len(have)}}


LF_FILE = 'lf_height_map'


def side_lane_lf_readers(ps1_text, scripts_dir):
    """Scripts in the pipeline's sync_final side lane ($tw3kSideLaneSteps) whose source names the LF height map.
    sync_final rewrites lf_height_map.dds in place (smooth_variant) while the side lane runs, so a reader there gets
    the old, the new or a half-written LF by timing (snow_bake in the side lane, integration review 2026-11-01)."""
    m = re.search(r"\$tw3kSideLaneSteps=@\(([^)]*)\)", ps1_text)
    if not m:
        return None
    steps = re.findall(r"'([^']+\.py)'", m.group(1))
    return [s for s in steps if LF_FILE in (Path(scripts_dir) / s).read_text(encoding='utf-8', errors='replace')]


def side_lane_lf(ctx):
    """Pipeline start: no LF reader runs in the sync_final side lane (side_lane_lf_readers). Value: such steps, or 1
    if the side-lane list cannot be found in build_polished_campaign_terrain.ps1."""
    here = Path(__file__).resolve().parent
    found = side_lane_lf_readers((here / 'build_polished_campaign_terrain.ps1').read_text(encoding='utf-8'), here)
    if found is None:
        return {'value': 1, 'detail': {'side_lane': 'not found in build_polished_campaign_terrain.ps1'}}
    return {'value': len(found), 'detail': {'lf_readers_in_side_lane': found}}


def design_single_source(ctx):
    """Pipeline start: the region records live only in the top-level design['regions'] (campaign_world.
    design_nested_mismatch: provinces[].regions are bare keys grouped by province_key), is_capital is consistent
    (design_capital_problems) and no field names the earlier reference mod (design_release_form.leftovers). Value:
    problems."""
    import campaign_world as cw
    import design_release_form as drf
    d = json.loads(cw.DESIGN_PATH.read_text(encoding='utf-8'))
    bad = cw.design_nested_mismatch(d) + cw.design_capital_problems(d) + [f'reference mod named at {p}' for p in drf.leftovers(d)]
    return {'value': len(bad), 'detail': {'problems': bad[:40]}}


def design_matches_map(ctx):
    """Pipeline start: the design's authored sea list and region budget agree with the final map
    (campaign_world.design_map_problems). Value: problems."""
    import campaign_world as cw
    import map_hex
    d = json.loads(cw.DESIGN_PATH.read_text(encoding='utf-8'))
    bad = cw.design_map_problems(d, map_hex.read(cw.WORKSPACE / 'work/final-map/map.hex'), map_hex.read(cw.VANILLA_MAP))
    return {'value': len(bad), 'detail': {'problems': bad[:40]}}


def map_hex_is_l0(ctx):
    """The final map.hex is the release L0's map (single-source audit 4, coordinator 2026-11-01): its sha256 equals
    data/map-layout-diff/manifest.json map_sha256, the map bake_map.py --diff writes (l0_roundtrip). Checked at the
    pipeline start and end, so no step and no authoring tool rewrites the map in between unnoticed. Missing until the
    release L0 exists in the repository. Value: 0 equal, 1 differs."""
    import campaign_world as cw
    man = cw._ROOT / 'data/map-layout-diff/manifest.json'
    if not man.exists():
        raise Missing('no release L0 (data/map-layout-diff) in this checkout')
    want = json.loads(man.read_text(encoding='utf-8'))['map_sha256']
    h = hashlib.sha256(ctx.path(MAP).read_bytes()).hexdigest()
    return {'value': int(h != want), 'detail': {'map_sha256': h, 'l0_map_sha256': want}}


def filler_passable_off_road(c, names):
    """(H, W) bool: passable land hexes of the non-playable filler without road edges (map.hex cells)."""
    import map_hex
    fill = (map_hex.region_ids(c) == names.index('3k_main_reg_non_playable')) & ((c[:, :, 0] & 3) != 1)
    return fill & ((c[:, :, 2] & 8) == 0) & (((c[:, :, 3] >> 1) & 63) == 0)


def filler_impassable(ctx):
    """The non-playable filler keeps no passable hex off its roads beyond the ones the production map already had
    (coordinator 2026-11-01): land handed back to the filler by the open-source rebuild went in passable, 82 -> 123
    SDK areas, and only the full pipeline's SDK preflight caught it (rehearsal 20261101-123241). Road hexes are the
    accepted exception (a new road crosses the filler only where nothing else connects). The known passable off-road
    filler hexes (none of them vanilla: earlier canvas and frontier work) are frozen from the release map in
    data/oss-rebuild/filler-open-known.json. Value: passable off-road filler land hexes not in that list."""
    import campaign_world as cw
    import map_hex
    known_p = cw._ROOT / 'data/oss-rebuild/filler-open-known.json'
    if not known_p.exists():
        raise Missing('data/oss-rebuild/filler-open-known.json')
    known = {tuple(h) for h in json.loads(known_p.read_text(encoding='utf-8'))['hexes']}
    m = map_hex.read(ctx.path(MAP)); c = m['cells']
    names = m['land_regions'] + m['sea_regions']
    rr, qq = np.nonzero(filler_passable_off_road(c, names))
    new = [(int(q), int(r)) for q, r in zip(qq, rr) if (int(q), int(r)) not in known]
    return {'value': len(new), 'detail': {'open': int(len(qq)), 'known': len(known), 'new': new[:20]}}


def settlement_primaries_ground(ctx):
    """Every region's primary settlement hexes (byte 2 slot nibble 1) include plain or beach land: the SDK exports no
    settlement whose primaries are all cliff (kind 3), and audit_exported_settlement_anchors then stops terrain step 1
    after the SDK export (release run 2026-11-01: North Okjeo, rebuilt town on the base map's cliff shore). Vanilla has
    no primary on a cliff. Value: regions with primaries and none off cliff."""
    import map_hex
    m = map_hex.read(ctx.path(MAP)); c = m['cells']
    names = m['land_regions'] + m['sea_regions']
    ids = map_hex.region_ids(c)
    prim = (c[:, :, 2] >> 4) == 1
    ok = prim & ((c[:, :, 0] & 3) != 3)
    have, good = set(np.unique(ids[prim]).tolist()), set(np.unique(ids[ok]).tolist())
    bad = sorted(names[i] for i in have - good if 0 <= i < len(names))
    return {'value': len(bad), 'detail': {'regions': bad[:20]}}


def dock_capacity(ctx):
    """The new regions' ports (coast_props.ports, the docks terrain step coast_props lays) fit the dock prefab
    numbering (coast_props.DOCK_PORTS ranges of 10 from DOCK_BASE): release run C 2026-11-01 stopped 30 minutes in
    at 13 ports with room for 10. Checked at the pipeline start. Value: ports over capacity."""
    import campaign_world as cw
    import coast_props as cp
    import map_hex
    m = map_hex.read(ctx.path(MAP))
    n = len(cp.ports(m, json.loads(cw.DESIGN_PATH.read_text(encoding='utf-8'))))
    return {'value': max(0, n - cp.DOCK_PORTS), 'detail': {'ports': n, 'capacity': cp.DOCK_PORTS}}


def river_channels_is_l0(ctx):
    """final-map/river-channels.json is the release L0's channel list (coordinator 2026-11-01, as map_hex_is_l0):
    byte-identical to data/map-layout-diff/river-channels.json, which bake_map.py --diff copies beside map.hex. Polish
    and the riparian belts only read it; apply_rivers --apply is an authoring tool outside the pipeline. Checked at
    the pipeline start and end. Missing until the release L0 carries a channel list. Value: 0 equal, 1 differs."""
    import campaign_world as cw
    ref = cw._ROOT / 'data/map-layout-diff/river-channels.json'
    if not ref.exists():
        raise Missing('no river channel list in the release L0 (data/map-layout-diff)')
    a, b = ctx.path(CHANNELS).read_bytes(), ref.read_bytes()
    return {'value': int(a != b), 'detail': {'sha256': hashlib.sha256(a).hexdigest(), 'l0_sha256': hashlib.sha256(b).hexdigest()}}


def biome_profiles_frozen(ctx):
    """Pipeline start: the biome profile table matches analysis/biome-profiles-frozen.json, is consistent and its review
    JSON is current (check_biome_profiles.status). Value: differing values + inconsistencies + stale review JSON. A
    REGIONS entry merged without --refreeze (P6 korea_patch, 2026-11-01) is the second miss of this kind; any difference
    stops the run. Fix: python scripts/check_biome_profiles.py --refreeze in the commit that changes the table."""
    import check_biome_profiles as cbp
    _, _, diff, bad, stale = cbp.status()
    return {'value': len(diff) + len(bad) + int(stale),
            'detail': {'differ': diff[:20], 'inconsistent': bad[:20], 'review_json_stale': stale}}


def snow_switches(ctx):
    """TW3K_SNOW_BAKE=1 needs TW3K_SNOW_COVERAGE=1 (coordinator 2026-11-01): the binary mask reads winter_snow as a
    coverage share, and the old intensity weights read that way put 20-40 % snow patches on the Tarim and Junggar
    basins (snow-v22b-findings §6). Checked at the pipeline start. 0 = consistent, 1 = SNOW_BAKE without COVERAGE."""
    import campaign_world as cw, biome_profiles as bp
    if not cw.snow_bake_on():
        raise Missing('snow bake switched off (TW3K_SNOW_BAKE=1 turns it on)')
    return {'value': int(not bp.snow_coverage_on()), 'detail': {'TW3K_SNOW_BAKE': '1', 'TW3K_SNOW_COVERAGE': os.environ.get('TW3K_SNOW_COVERAGE', '0')}}


def korea_patch_rules(ctx):
    """The korea patch cache polish will load (analysis/korea-patch.npz, korea-patch-forests.npz with
    TW3K_KOREA_FORESTS=1) carries the active forest rules: rules_sha == biome_profiles.korea_forests_sha(), the rule
    hehuang_terrain.check_rules stops polish on. 0 = current, 1 = stale or missing. Checked at the pipeline start: v22
    run 20261031-154638 died 20 s into polish on the 10-24 cache (no rules_sha, d85a1c98). Rebuild: python
    scripts/korea_patch.py build (same TW3K_KOREA_FORESTS)."""
    import biome_profiles as bp
    if os.environ.get('TW3K_KOREA', '1') == '0':
        raise Missing('korea patch switched off (TW3K_KOREA=0)')
    forests = bp.korea_forests_on()
    rel = 'analysis/korea-patch-forests.npz' if forests else 'analysis/korea-patch.npz'
    want = bp.korea_forests_sha()
    path = ctx.ws / rel
    if not path.is_file():
        if not forests:
            raise Missing(rel)   # polish runs without the patch then (hehuang_terrain.Patch.load)
        return {'value': 1, 'detail': {'path': rel, 'missing': True, 'want': want[:12]}}
    with np.load(path) as z:
        got = str(z['rules_sha']) if 'rules_sha' in z.files else None
    return {'value': int(got != want), 'detail': {'path': rel, 'rules_sha': str(got)[:12], 'want': want[:12],
                                                  'korea_forests': forests}}


def patch_towns_flattened(ctx):
    """Every design town in a patch cache's window that the patch weight covers is levelled by it (coordinator 11-01,
    P6). For each cache present (hehuang_terrain.patch_path) and each design town whose hex is in its hex_box: the weight
    on the town hex's own tiles (hehuang_terrain.town_weight, the builds' rule; korea height_weight, hehuang weight).
    Fails: a town at weight > .5 missing from the cache's towns (a build that sampled elsewhere: c7's two shore towns,
    sampled on their water east neighbour, 11-01), and a Korea peninsula-class town (region-traits province_classes
    'peninsula') at weight <= .5 with no reason. Towns at weight <= .5 are listed with the reason: not in the patch
    class, town hex water, vanilla town plot, or the weight's edge (<= .5 on land). Value = failures."""
    import hehuang_terrain as ht
    import campaign_world as cw
    import map_hex
    plan = json.loads(cw.DESIGN_PATH.read_text(encoding='utf-8'))
    F = map_hex.read(ctx.path(MAP))
    V = map_hex.read(cw.VANILLA_MAP)
    stems = set(cw.region_traits()['province_classes'].get('peninsula', ())) if cw.REGION_TRAITS_PATH.exists() else set()
    det, bad = {}, []
    for name in ht.PATCHES:
        if os.environ.get('TW3K_' + name.upper(), '1') == '0':
            continue
        path = ht.patch_path(name)
        if not path.is_file():
            continue
        with np.load(path) as z:
            hb = [int(v) for v in z['hex_box']]; x0, y0 = int(z['box'][0]), int(z['box'][2])
            w = (z['height_weight'] if 'height_weight' in z.files else z['weight']).astype(np.float32)
            levelled = {t['region'] for t in json.loads(str(z['towns']))}
        skipped = {}
        for region, _, (q, r), _ in ht.design_towns(hb, plan):
            tw = ht.town_weight(w, q, r, x0, y0)
            if region in levelled:
                continue
            t = next(t for t in plan['regions'] if t['region'] == region)
            in_class = t.get('province_key', '').split('_province_', 1)[-1] in stems
            if tw > .5:
                bad.append(f'{name}: {region} at weight {tw:.2f} not levelled'); continue
            vq, vr = q - cw.PAD_LEFT, r - cw.PAD_BOTTOM
            vc = V['cells'][vr, vq] if 0 <= vq < cw.NATIVE_W and 0 <= vr < cw.NATIVE_H else None
            if (F['cells'][r, q, 0] & 3) == 1:
                why = 'town hex water'
            elif vc is not None and (((vc[2] >> 4) > 0) or ((vc[3] & 1) > 0)):
                why = 'vanilla town plot'
            elif name == 'korea' and not in_class:
                why = 'not in the peninsula class'
            else:
                why = f'patch weight {tw:.2f} (outside the patch, its edge or protected)'
            skipped[region] = why
            if name == 'korea' and in_class and why.startswith('patch weight'):
                bad.append(f'{name}: peninsula-class {region} not levelled ({why})')
        det[name] = {'levelled': len(levelled), 'skipped': skipped}
    if not det:
        raise Missing('no patch cache')
    if bad:
        det['failures'] = bad[:20]
    return {'value': len(bad), 'detail': det}


def patch_design(ctx):
    """Patch caches polish will load (hehuang_terrain.patch_path: hehuang, korea / korea-forests) built from the current
    region design: hehuang_terrain.design_mismatch (npz design_sha, or the levelled towns of older caches) for each one
    present, and from the current inputs and build rules: hehuang_terrain.source_mismatch (npz inputs_sha: the final
    map inside the window and west-real-terrain; build_rules_sha; single-source audit item 8). A cache built before
    those keys is stale (rebuild). Value = stale caches. Checked at the pipeline start (P6, 2026-11-01: the redesigned regions' towns are baked
    into the npz; Patch.load stops on the same rule). Rebuild: python scripts/<name>_patch.py build."""
    import hehuang_terrain as ht
    import campaign_world as cw
    plan = json.loads(cw.DESIGN_PATH.read_text(encoding='utf-8'))
    det, bad = {}, 0
    for name in ht.PATCHES:
        if os.environ.get('TW3K_' + name.upper(), '1') == '0':
            continue
        path = ht.patch_path(name)
        if not path.is_file():
            continue
        with np.load(path) as z:
            why = ht.design_mismatch(z, plan) or ht.source_mismatch(name, z, ctx.ws)
        det[name] = why or 'current'
        bad += why is not None
    if not det:
        raise Missing('no patch cache')
    return {'value': bad, 'detail': det}


def _wuyuan_off():
    """Why the wuyuan patch is not there, or None. TW3K_WUYUAN=0 is for diagnostics only, never a fallback: without the
    patch Jiuyuan stands in its pit again, so the wuyuan invariants fail instead of skipping (integrator 2026-11-01)."""
    import hehuang_terrain as ht
    if os.environ.get('TW3K_WUYUAN', '1') == '0':
        return 'wuyuan patch switched off (TW3K_WUYUAN=0: diagnostics only)'
    if not ht.patch_path('wuyuan').is_file():
        return 'wuyuan patch cache missing (analysis/wuyuan-patch.npz; bake step 10 / ensure_patch_caches builds it)'
    return None


def _wuyuan_required(fn):
    def run(ctx):
        why = _wuyuan_off()
        return {'value': 1, 'reason': why, 'detail': {'patch': why}} if why else fn(ctx)
    run.__name__, run.__doc__ = fn.__name__, fn.__doc__
    return run


def _wuyuan_npz(ctx):
    import hehuang_terrain as ht
    with np.load(ht.patch_path('wuyuan')) as z:
        return {k: z[k] for k in z.files}


@_wuyuan_required
def wuyuan_patch_scope(ctx):
    """The wuyuan patch (wuyuan_patch.py; coordinator 2026-11-01) edits only its declared box and only vanilla
    3k_main_reg_non_playable filler land: height weight > 0 outside wuyuan_patch.BOX, on vanilla playable land, on a
    vanilla plot or on water counts (tile cells). 0 = in scope."""
    import wuyuan_patch as wp
    z = _wuyuan_npz(ctx)
    q0, q1, r0, r1 = (int(v) for v in z['hex_box'])
    out_box = int(z['hex_mask'].sum()) if (q0, q1, r0, r1) != wp.BOX else 0
    cells, names = ctx.map()
    tx, ty, hq, hr, land, filler, protected, plots, water, _ = wp._fields(cells, names)
    hw = z['height_weight'].astype(np.float32)
    bad = (hw > 0) & (protected | plots | water | ~filler)
    return {'value': out_box + int(bad.sum()), 'detail': {'box': [q0, q1, r0, r1], 'declared': list(wp.BOX), 'hexes_weight_gt0': int(z['hex_mask'].sum()),
                                                      'cells_out_of_scope': int(bad.sum())}}


@_wuyuan_required
def wuyuan_core_real(ctx):
    """In the wuyuan core (wuyuan_patch.CORE, full height weight, more than BANK_SEAM + .5 hexes from the vanilla bank)
    the patch relief stays within CORE_MAX_DIFF (frozen 2026-11-01) of the real ground LEVEL + to_relief(ETOPO) the build
    stores ('real'), outside the town flats (npz town_flat > 0; wuyuan_patch.core_checked, user 2026-11-01). Value =
    checked core tile cells beyond it; the detail gives the cells excluded and their largest difference."""
    import campaign_world as cw
    import wuyuan_patch as wp
    z = _wuyuan_npz(ctx)
    if 'town_flat' not in z:
        return {'value': 1, 'reason': 'wuyuan patch cache predates town_flat (rebuild it)', 'detail': {}}
    x0, x1, y0, y1 = (int(v) for v in z['box'])
    ty, tx = np.mgrid[y0:y1, x0:x1]
    hq, hr = cw.tile_hex(tx, ty)
    core = wp.core_checked(hq, hr, z['height_weight'], z['bank_distance'], z['town_flat'])
    flats = wp.core_checked(hq, hr, z['height_weight'], z['bank_distance'], None) & (z['town_flat'].astype(np.float32) > 0)
    diff = np.abs(z['relief'] - z['real'])
    d = diff[core]
    return {'value': int((d > wp.CORE_MAX_DIFF).sum()), 'detail': {'core_cells': int(core.sum()), 'max_abs': round(float(d.max()), 4) if d.size else None,
                                                                 'limit': wp.CORE_MAX_DIFF, 'town_flat_cells_excluded': int(flats.sum()),
                                                                 'town_flat_max_abs': round(float(diff[flats].max()), 4) if flats.any() else None}}


def _lf_tiles(raw):
    """Raw LF texels (south-up) -> relief on the tile grid (mean of each tile cell's texel block)."""
    import campaign_world as cw
    k = raw.shape[1] // cw.TILE_W
    a = raw[:cw.TILE_H * k, :cw.TILE_W * k].astype(np.float32).reshape(cw.TILE_H, k, cw.TILE_W, k).mean((1, 3))
    return cw.LEVEL + (a / 65535 - cw.NORM) * cw.GAIN


def _before_polish_lf(ctx):
    p = ctx.ws / 'terrain-before-polish' / f'terrain/campaigns/{VARIANT}/lf_height_map.dds'
    if not p.is_file():
        raise Missing(str(p))
    d = p.read_bytes(); h, w = struct.unpack_from('<2I', d, 12)
    return np.frombuffer(d, '<u2', offset=128, count=h * w).reshape(h, w)[::-1]


@_wuyuan_required
def wuyuan_playable_identical(ctx):
    """Vanilla playable land round the wuyuan box (BOX grown 3 hexes; the Shuofang bank) keeps the pre-polish LF
    byte for byte: stage LF texels differing from terrain-before-polish there. 0 = identical (coordinator 2026-11-01;
    the native audits exempt the patch mask grown 2 hexes, which reaches the bank)."""
    import campaign_world as cw
    import wuyuan_patch as wp
    _wuyuan_npz(ctx)
    cur, base = ctx.lf(), _before_polish_lf(ctx)
    k = cur.shape[1] // cw.TILE_W
    cells, names = ctx.map()
    tx, ty, hq, hr, land, filler, protected, plots, water, _ = wp._fields(cells, names, 3)
    ys, xs = ty[protected], tx[protected]
    n = 0
    for y, x in zip(ys.tolist(), xs.tolist()):
        n += int((cur[y * k:(y + 1) * k, x * k:(x + 1) * k] != base[y * k:(y + 1) * k, x * k:(x + 1) * k]).sum())
    return {'value': n, 'detail': {'playable_tile_cells': int(protected.sum()), 'texels_differing': n}}


def vanilla_playable_lf(ctx):
    """Tile cells of vanilla playable land (vanilla map.hex land not 3k_main_reg_non_playable, native frame) with any
    stage LF texel differing from terrain-before-polish. Baseline 380 (release run C2, 2026-11-01: polish's own writers
    before the wuyuan patch); it only goes down (integrator 2026-11-01: the wuyuan check covers its box exactly, this
    one catches a write anywhere else on vanilla playable ground)."""
    import campaign_world as cw
    import map_hex
    cur, base = ctx.lf(), _before_polish_lf(ctx)
    k = cur.shape[1] // cw.TILE_W
    V = map_hex.read(cw.VANILLA_MAP); vn = V['land_regions'] + V['sea_regions']; vid = map_hex.region_ids(V['cells'])
    names = np.array(vn + ['?'])[np.where(vid >= 0, vid, len(vn))]
    play = ((V['cells'][..., 0] & 3) != 1) & (names != '3k_main_reg_non_playable')
    ty, tx = np.mgrid[cw.NATIVE_TILE_Y0:cw.NATIVE_TILE_Y1, cw.NATIVE_TILE_X0:cw.NATIVE_TILE_X1]
    hq, hr = cw.tile_hex(tx, ty)
    m = play[np.clip(hr - cw.PAD_BOTTOM, 0, play.shape[0] - 1), np.clip(hq - cw.PAD_LEFT, 0, play.shape[1] - 1)]
    sl = np.s_[cw.NATIVE_TILE_Y0 * k:cw.NATIVE_TILE_Y1 * k, cw.NATIVE_TILE_X0 * k:cw.NATIVE_TILE_X1 * k]
    d = (cur[sl] != base[sl]).reshape(m.shape[0], k, m.shape[1], k).any((1, 3))
    n = int((d & m).sum())
    return {'value': n, 'detail': {'playable_tile_cells': int(m.sum()), 'changed': n}}


@_wuyuan_required
def wuyuan_band_slope(ctx):
    """The wuyuan transition band (0 < height weight < .999) after polish is no steeper than the vanilla filler round
    the box (frozen rule, coordinator 2026-11-01): largest step between 4-neighbour tile cells in the band (stage LF)
    minus the largest such step in pre-polish filler land in the 4-hex ring outside the box. Value = the excess
    (0 = not steeper)."""
    import wuyuan_patch as wp
    z = _wuyuan_npz(ctx)
    G = 4
    cells, names = ctx.map()
    tx, ty, hq, hr, land, filler, protected, plots, water, _ = wp._fields(cells, names, G)
    cur, base = _lf_tiles(ctx.lf()), _lf_tiles(_before_polish_lf(ctx))
    y0, x0 = int(ty[0, 0]), int(tx[0, 0])
    sl = np.s_[y0:y0 + ty.shape[0], x0:x0 + tx.shape[1]]
    c, b = cur[sl], base[sl]
    hw = np.zeros(c.shape, np.float32)
    bx0, bx1, by0, by1 = (int(v) for v in z['box'])
    hw[by0 - y0:by1 - y0, bx0 - x0:bx1 - x0] = z['height_weight'].astype(np.float32)
    in_box = np.zeros(c.shape, bool); in_box[by0 - y0:by1 - y0, bx0 - x0:bx1 - x0] = True

    def steps(a, m):
        dy = np.abs(np.diff(a, axis=0)); my = m[1:] & m[:-1]
        dx = np.abs(np.diff(a, axis=1)); mx = m[:, 1:] & m[:, :-1]
        return max(float(dy[my].max()) if my.any() else 0., float(dx[mx].max()) if mx.any() else 0.)
    band = steps(c, (hw > 0) & (hw < .999) & land)
    ring = steps(b, land & filler & ~in_box)
    return {'value': round(max(0., band - ring), 4), 'detail': {'band_max_step': round(band, 4), 'ring_vanilla_max_step': round(ring, 4)}}


def land_mesh_strays(ctx):
    """Stage land_mesh files (.rigid_model_v2 / .compressed_map) that are neither base meshes (land_mesh_0..
    BASE_LAND_MESHES-1) nor listed in the overflow manifest (analysis/coarse-mesh-overflow-<variant>.json): 0. Checked at
    the pipeline start: a stray here is what polish's snapshot loop seeded into the base on 10-30. Manifest entries
    whose files or parent model are gone (stale, 10-30 run 4: gap-fill FileNotFoundError) are reported in the detail
    (gap-fill drops them; mesh bake 3.2.1)."""
    d = ctx.ws / MESHES
    if not d.is_dir():
        raise Missing(MESHES)
    man = ctx.ws / f'analysis/coarse-mesh-overflow-{VARIANT}.json'
    files = json.loads(man.read_text(encoding='utf-8')).get('files', {}) if man.is_file() else {}
    listed = {n for names in files.values() for n in names}
    have = {f.name for f in d.glob('land_mesh_*')}

    def known(n):
        stem = n.split('.')[0]
        return int(stem.rsplit('_', 1)[1]) < BASE_LAND_MESHES or stem + '.rigid_model_v2' in listed
    strays = sorted(n for n in have if not known(n))
    stale = sorted(parent for parent, names in files.items() if parent not in have
                   or any(n not in have or n.replace('.rigid_model_v2', '.compressed_map') not in have for n in names))
    return {'value': len(strays), 'detail': {'strays': strays[:40], 'stale_manifest_entries': stale[:40],
                                             'overflow_listed': len(listed), 'files': len(have)}}


def oasis_town_trees(ctx):
    """Oasis towns below their TOWN OASIS floor (foliage_biomes.town_oasis_rule: planted hexes in the town's ring >=
    ceil(min_hits_per_oasis_hex x oasis hexes there)), as the foliage step reports them (town_oasis). A report
    without the block counts as a failure: the floor did not run."""
    rep = json.loads(ctx.path(FOLIAGE_REPORT).read_text(encoding='utf-8'))
    to = rep.get('town_oasis')
    if to is None:
        return {'value': 1, 'detail': {'error': 'terrain-native-foliage.json has no town_oasis block'}}
    below = {k: v for k, v in to['towns'].items() if v['planted_hexes'] < v['floor']}
    return {'value': len(below), 'detail': {'oasis_towns': to['oasis_towns'], 'floor_added_hexes': to['floor_added_hexes'], 'below': below}}


def flat_arid_rocks(ctx):
    """Canvas general_rock_* rows of the stage tree list on flat arid hexes (foliage_biomes ROCK RELIEF:
    flat_rock_hexes, canvas = outside the native rectangle as polish_campaign_foliage.native_many)."""
    import campaign_world as cw, foliage_biomes as fb
    b = ctx.path(TREES).read_bytes(); pos = 24; xz = []; by = {}
    for _ in range(struct.unpack_from('<I', b, 20)[0]):
        n = struct.unpack_from('<H', b, pos)[0]; name = b[pos + 2:pos + 2 + n].decode(); pos += 2 + n
        cnt = struct.unpack_from('<I', b, pos)[0]; pos += 4
        rock = name.startswith(fb.ROCK_PREFIX)
        for _ in range(cnt):
            if rock:
                x, _, z = struct.unpack_from('<3f', b, pos); xz.append((x, z)); by.setdefault(name, []).append(len(xz) - 1)
            pos += 18 + struct.unpack_from('<I', b, pos + 14)[0] * 4
    if pos != len(b):
        raise ValueError('tree list layout')
    if not xz:
        return {'value': 0, 'detail': {'rock_rows': 0}}
    p = np.array(xz, np.float64)
    qx, zr = p[:, 0] / cw.DX, p[:, 1] / cw.DZ
    canvas = ~((cw.PAD_LEFT <= qx) & (qx < cw.PAD_LEFT + cw.NATIVE_W) & (cw.PAD_BOTTOM <= zr) & (zr < cw.PAD_BOTTOM + cw.NATIVE_H))
    q, r = cw.world_hex(p[:, 0], p[:, 1])
    flat = fb.flat_rock_hexes()
    ok = (q >= 0) & (q < flat.shape[1]) & (r >= 0) & (r < flat.shape[0])
    bad = canvas & ok
    bad[bad] = flat[r[bad], q[bad]]
    per = {k: int(bad[v].sum()) for k, v in by.items() if bad[v].any()}
    return {'value': int(bad.sum()), 'detail': {'rock_rows': len(xz), 'canvas_rock_rows': int(canvas.sum()), 'on_flat_by_species': per}}


def frontier_mountain_stable(ctx):
    """Frontier mountain props (frontier_mountain_props leaves, prefabs FIRST_PREFAB..+899 of the stage
    global_props.bin) identical, instance by instance, to the previous run's wherever the terrain the step reads did
    not change. Per BLOCK x BLOCK hex block: a terrain hash over the block plus MARGIN hexes (map.hex cells, polished
    surface, strategic_ink biome / elevation) and a props hash (sorted model + matrix rounded to 1e-3 of the sites
    centred in the block). State: {block: [terrain, props]}. Value: blocks of the reference run (the last record row
    of this invariant from another run, TW3K_INVARIANT_RUN) whose terrain hash is unchanged but props hash differs;
    0 with no reference row."""
    import hashlib, os
    import campaign_world as cw, props_bundle as pb, strategic_ink
    import frontier_mountain_props as fmp
    from translate_campaign_lake_instances import records
    BLOCK, MARGIN = 32, 12
    cells, _ = ctx.map()
    H, W = cells.shape[:2]
    surf = np.load(ctx.path(f'analysis/polished-surface-{VARIANT}.npy'), allow_pickle=False)
    biome, _, _, elev = strategic_ink.load()
    _, assets = pb.unbundle(ctx.path(PRE + 'global_props.bin').read_bytes())
    leaf = re.compile(r'bmd_objects\.[^/]+\.(\d+)\.31\.bin$')
    sites, n_sites = {}, 0
    for k, b in assets.items():
        mt = leaf.search(k)
        if not mt or not fmp.FIRST_PREFAB <= int(mt.group(1)) < fmp.FIRST_PREFAB + 900:
            continue
        rs = records(b)
        n_sites += 1
        x, z = float(np.mean([r['matrix'][9] for r in rs])), float(np.mean([r['matrix'][11] for r in rs]))
        q, r = (int(v) for v in cw.world_hex(x, z))
        sites.setdefault((r // BLOCK, q // BLOCK), []).extend(
            sorted(f"{r_['model'].lower()}|{','.join(f'{v:.3f}' for v in r_['matrix'])}" for r_ in rs))
    imp = (cells[..., 2] & 8) > 0
    imp[cw.NATIVE_HEX_SLICE] = False
    state = {}
    for br in range((H + BLOCK - 1) // BLOCK):
        for bq in range((W + BLOCK - 1) // BLOCK):
            if not imp[br * BLOCK:(br + 1) * BLOCK, bq * BLOCK:(bq + 1) * BLOCK].any() and (br, bq) not in sites:
                continue
            r0, r1 = max(0, br * BLOCK - MARGIN), min(H, (br + 1) * BLOCK + MARGIN)
            q0, q1 = max(0, bq * BLOCK - MARGIN), min(W, (bq + 1) * BLOCK + MARGIN)
            th = hashlib.sha1()
            for a in (cells[r0:r1, q0:q1], surf[2 * r0:2 * r1 + 2, 2 * q0:2 * q1 + 2], biome[r0:r1, q0:q1], elev[r0:r1, q0:q1]):
                th.update(np.ascontiguousarray(a).tobytes())
            ph = hashlib.sha1('\n'.join(sites.get((br, bq), [])).encode()).hexdigest()[:16]
            state[f'{br},{bq}'] = [th.hexdigest()[:16], ph]
    run = os.environ.get('TW3K_INVARIANT_RUN')
    ref = next((o for o in reversed([] if ctx.over.get(MAP) else load_record(ctx.ws))
                if o.get('invariant') == 'frontier_mountain_stable' and o.get('state') and o.get('run') != run), None)
    bad = []
    if ref is not None:
        bad = sorted(b for b, (t, p) in ref['state'].items() if b in state and state[b][0] == t and state[b][1] != p)
    return {'value': len(bad), 'state': state,
            'detail': {'sites': n_sites, 'instances': sum(len(v) for v in sites.values()), 'site_blocks': len(sites), 'blocks': len(state),
                       'reference_run': ref and ref.get('run'), 'terrain_changed_blocks': 0 if ref is None else
                       sum(1 for b, (t, _) in ref['state'].items() if b in state and state[b][0] != t),
                       'unchanged_terrain_props_differ': bad[:40]}}


def frontier_mountain_family_count(ctx):
    """Frontier mountain instances per model family (frontier_mountain_props leaves, prefabs FIRST_PREFAB..+899 of
    the stage global_props.bin) against the family's vanilla-rate target in the step's report: value = families
    off by more than props MOUNTAIN_FAMILY_COUNT_TOL (warn-only: the stateless step has no cap). Detail: counts,
    targets, deviations, instances and LOD0 triangles (models x instances, frontier mountain props)."""
    import biome_profiles as bp, props_bundle as pb
    import frontier_mountain_props as fmp
    from translate_campaign_lake_instances import records
    tol = bp.CONSTS['props']['MOUNTAIN_FAMILY_COUNT_TOL']
    rep = json.loads(ctx.path('analysis/frontier-mountain-props.json').read_text(encoding='utf-8'))
    targets = {f: v.get('target') for f, v in rep['variants'][0]['by_family'].items() if v.get('target')}
    _, assets = pb.unbundle(ctx.path(PRE + 'global_props.bin').read_bytes())
    leaf = re.compile(r'bmd_objects\.[^/]+\.(\d+)\.31\.bin$')
    models = {}
    for k, b in assets.items():
        mt = leaf.search(k)
        if mt and fmp.FIRST_PREFAB <= int(mt.group(1)) < fmp.FIRST_PREFAB + 900:
            for r in records(b):
                models[r['model']] = models.get(r['model'], 0) + 1
    counts = {}
    for mdl, n in models.items():
        counts[fmp.family(mdl)] = counts.get(fmp.family(mdl), 0) + n
    dev = fmp.family_deviation(counts, targets, tol)
    va = fmp.vanilla_assets()
    tri = {mdl: fmp.model_triangles(va, mdl) for mdl in models}
    return {'value': sum(1 for v in dev.values() if v[3]),
            'detail': {'tol': tol, 'families': {f: {'count': c, 'target': t, 'deviation': d} for f, (c, t, d, _) in dev.items()},
                       'counts': counts, 'instances': sum(models.values()),
                       'triangles_lod0': sum((tri[m] or 0) * n for m, n in models.items()),
                       'unreadable_models': sorted(m for m, t in tri.items() if t is None)}}


def _canvas_blend(ctx, include_band=False):
    """(blend R south-up, canvas land mask on the pixel grid) of the stage: canvas = outside the native rectangle."""
    import campaign_world as cw
    cells, _ = ctx.map()
    d = ctx.path(BLEND).read_bytes(); bh, bw = struct.unpack_from('<2I', d, 12)
    rg = np.frombuffer(d, np.uint8, offset=128, count=bh * bw * 2).reshape(bh, bw, 2)[::-1]
    r = rg[:, :, 0].copy()
    if include_band:
        g = rg[:, :, 1].copy()
    del rg, d
    ty, tx = np.mgrid[:bh // 4, :bw // 4]
    q, rr = cw.tile_hex(tx, ty)
    land_t = ((cells[:, :, 0] & 3) != 1)[rr, q]
    canvas_t = land_t & ~((tx >= cw.NATIVE_TILE_X0) & (tx < cw.NATIVE_TILE_X1) & (ty >= cw.NATIVE_TILE_Y0) & (ty < cw.NATIVE_TILE_Y1))
    m = np.repeat(np.repeat(canvas_t, 4, 0), 4, 1)[:bh, :bw]
    return (r, m, g) if include_band else (r, m)


def tone_contrast_edges(ctx):
    """Canvas neighbour contrast by each region's own vanilla G-band p99.

    Per-band counts and vanilla reference rates remain visible; a global threshold
    would apply the wrong regional distribution. Skipped until tones are measured.
    """
    import ground_tone as gt, biome_profiles as bp
    t, mde = gt.tone_table(), gt.max_dE()
    if t is None or mde is None:
        raise Missing('MATERIAL_TONE / CONTRAST not measured (build_material_tones.py)')
    r, m, g = _canvas_blend(ctx, include_band=True)
    # _canvas_blend serves south-up tile consumers; vanilla contrast is measured
    # north-first, including which endpoint supplies a vertical pair's G band.
    by_band = gt.contrast_by_band(r[::-1], g[::-1], t, m[::-1])
    vanilla = bp.CONSTS['ground_materials']['CONTRAST']
    expected = 0.
    for band, stats in by_band.items():
        reference = vanilla.get('by_band', {}).get(band, vanilla).get('share_above_max')
        stats['share'] = stats['above'] / max(stats['pairs'], 1)
        stats['vanilla_share'] = reference
        if reference is not None:
            expected += stats['pairs'] * reference
    above, pairs = sum(s['above'] for s in by_band.values()), sum(s['pairs'] for s in by_band.values())
    return {'value': round(above / max(pairs, 1), 6),
            'detail': {'pairs': pairs, 'above': above, 'by_band': by_band,
                       'vanilla_weighted_share': expected / max(pairs, 1)}}


def woodland_off_forest(ctx):
    """Share of canvas land tiles whose blend centre pixel is a woodland floor slot (CONSTS DONOR_WOODLAND slots)
    while the shared forest density (landform-tiles forest_density) is below .2: the dark islands of woodland floor
    on open ground (ground contrast fix §3, funan / indus). Skipped without the forest density field."""
    import biome_profiles as bp
    with np.load(ctx.path(LANDFORM_TILES)) as z:
        if 'forest_density' not in z.files:
            raise Missing('landform-tiles has no forest_density (polish before ground contrast fix §3)')
        fd = z['forest_density'].astype(np.float32)
    r, m = _canvas_blend(ctx)
    rt, mt = r[2::4, 2::4], m[2::4, 2::4]
    h, w = min(rt.shape[0], fd.shape[0]), min(rt.shape[1], fd.shape[1])
    wood = np.isin(rt[:h, :w], bp.CONSTS['ground_materials']['DONOR_WOODLAND']['slots']) & mt[:h, :w]
    off = wood & (fd[:h, :w] < .2)
    return {'value': round(float(off.sum() / max(int(mt[:h, :w].sum()), 1)), 5),
            'detail': {'woodland_tiles': int(wood.sum()), 'off_forest_tiles': int(off.sum())}}


def tundra_temperate_share(ctx):
    """Share of tundra tiles (landform field: tundra dominant above ban_min, audit-only ban) whose blend centre pixel
    holds a temperate slot (biome_profiles AUDIT tundra 'banned', minus the per-slot whitelist; landform_tiles.
    banned_share, the one definition): the global relief recipe painted tundra valleys temperate_2 (eighteenth
    pack .456, sixteenth .034) until LANDFORMS['tundra'] took its own relief (§9.3)."""
    import landform_tiles
    lt = landform_tiles.load(ctx.path(LANDFORM_TILES).parent.parent, check=False)
    r, _ = _canvas_blend(ctx)
    mat = r[2::4, 2::4]; th, tw = lt['dom'].shape
    if mat.shape != (th, tw):
        raise Missing(f'blend tile grid {mat.shape} != landform field {(th, tw)}')
    s = landform_tiles.banned_share(mat, lt, landforms=('tundra',))['tundra']
    return {'value': s['share'], 'detail': s}



def tundra_forest_straight(ctx):
    """Share of tundra/forest boundary windows whose large-scale contour is one straight line, on the ecotone-edited
    classes polish painted with (landform-tiles eco_bio; ecotone_bands.straight_share, the one definition): RESOLVE drew
    the arctic treeline as long straight vector edges (main 2026-10-30 .251 of 275 windows, the 'NE straight tundra
    band' of live checks 2048n-p) until ecotone_bands.tundra_forest (§9.3)."""
    import landform_tiles, ecotone_bands, campaign_world as cw
    if not cw.arctic_treeline_on():
        raise Missing('arctic treeline switched off (TW3K_ARCTIC_TREELINE=1 turns it on)')
    lt = landform_tiles.load(ctx.path(LANDFORM_TILES).parent.parent, check=False)
    s, n, hits = ecotone_bands.straight_share(lt['eco_bio'], list(lt['meta']['biome_names']))
    return {'value': round(s, 4), 'detail': {'windows': n, 'straight': len(hits), 'centres': hits[:40]}}

def native_edge_tone_cut(ctx):
    """Native-border segments failing the tone CUT (audit_native_edges 'tone_cut': the slot colour jumps on the border
    line beyond vanilla's own interior lines, any season; ground contrast fix design §10) in the early edge audit run
    right after polish, outside the shrink-only known list (audit_native_edges TONE_CUT_KNOWN, design 10.1): 0. The listed
    segments' values are reported every run (known_passing: fixed, remove from the list). Missing when that report is
    older than the blend or predates the tone CUT."""
    p, blend = ctx.path(EDGE_AUDIT), ctx.path(BLEND)
    if p.stat().st_mtime < blend.stat().st_mtime:
        raise Missing(f'{EDGE_AUDIT} is older than the blend (audit_native_edges.py not run on this polish)')
    rep = json.loads(p.read_text(encoding='utf-8'))
    if not rep.get('tone_cut_limits'):
        raise Missing(f'{EDGE_AUDIT} has no tone CUT (TONE_CUT not measured or audit before ground contrast fix §10)')
    if 'tone_cut_known' not in rep:
        raise Missing(f'{EDGE_AUDIT} predates the tone CUT known list (design 10.1)')
    bad = [{'edge': n, 'hex': s['hex'], 'seasons': s.get('tone_cut_fail_seasons'), 'tone_cut': s.get('tone_cut')}
           for n, e in rep['edges'].items() for s in e['failing'] if 'tone_cut' in s['fails']]
    known = rep['tone_cut_known']
    return {'value': len(bad), 'detail': {'segments': bad[:40], 'known': known,
                                          'known_passing': [k for k in known if not k['fail_seasons']]}}


def wetland_in_town_clear(ctx):
    """Wetland material (31) pixels on the town clearance hexes (apply_climate.town_clear_hex: settlement slots
    and TOWN_CLEAR around them) outside the native rectangle: 0 (the town spec keeps its ground; ground contrast
    fix §4: the soft wetland edge must not reach a town)."""
    import campaign_world as cw, apply_climate as ac
    if not ac.WET_PIXEL_MASK:
        # switch off (the default until validated): the old wetland painter and town ground run, which leave 364
        # pixels on main (run 20261030-014133); the invariant guards the switch's change, as tile_climate_matches_rule
        raise Missing('TW3K_WET_PIXEL_MASK=0: the wetland pixel mask and the town dry-clear are off')
    cells, _ = ctx.map()
    tc = ac.town_clear_hex(cells)
    r, m = _canvas_blend(ctx)
    bh, bw = r.shape
    ty, tx = np.mgrid[:bh // 4, :bw // 4]
    q, rr = cw.tile_hex(tx, ty)
    tct = np.repeat(np.repeat(tc[rr, q], 4, 0), 4, 1)[:bh, :bw]
    n = int(((r == 31) & m & tct).sum())
    return {'value': n, 'detail': {'town_clear_hexes': int(tc.sum())}}


def blend_single_writer(ctx):
    """Material pixels of the blend rewritten by a later writer after the mirror/donor blend (polish
    analysis/blend-writers.json: ground materials, town ground, farm fields, wetlands): the baseline for the single
    selection point (ground contrast fix §1 end state, bake design: one writer per product). Warn-only."""
    d = json.loads(ctx.path('analysis/blend-writers.json').read_text(encoding='utf-8'))
    return {'value': int(d.get('pixels_rewritten', 0)), 'detail': d}


def hex_squareness(ctx):
    """Share of the wetland (31) and farm (8-12) material outlines on canvas land that run along tile grid lines
    (every 4 px): about .25 for outlines that ignore the grid, 1 for tile-square paint (ground contrast fix §4)."""
    r, m = _canvas_blend(ctx)
    out = {}
    for name, slots in (('wetland', (31,)), ('farm', (8, 9, 10, 11, 12))):
        k = np.isin(r, slots) & m
        ev = k[:, :-1] != k[:, 1:]; eh = k[:-1, :] != k[1:, :]
        on = int(ev[:, 3::4].sum()) + int(eh[3::4, :].sum()); tot = int(ev.sum()) + int(eh.sum())
        out[name] = round(on / tot, 4) if tot else None
    vals = [v for v in out.values() if v is not None]
    return {'value': max(vals) if vals else 0., 'detail': out}


def town_plateau_lift(ctx):
    """Largest lift of a town plateau over its own reference height (analysis/geographic-terrain-polish.json towns:
    plateau_height - local_reference_height, the projection's plot coupling raising a plot to its neighbours): a lifted
    plot stands as a mesa above its own clearance ground, with grade-cap walls and V slots towards its neighbours
    (v20 town-wusun: 3.77, the lake-shore cap stopping dead at LAKE_TOWN_REACH; town spec, lake cap continuity)."""
    d = json.loads(ctx.path(POLISH_REPORT).read_text(encoding='utf-8'))

    def towns(o):
        if isinstance(o, dict):
            t = o.get('towns')
            if isinstance(t, list) and t and isinstance(t[0], dict) and 'plateau_height' in t[0]:
                return t
            o = list(o.values())
        if isinstance(o, list):
            for v in o:
                r = towns(v)
                if r:
                    return r
        return None
    t = towns(d)
    if not t:
        raise Missing(f'{POLISH_REPORT}: no towns with plateau heights')
    lift = sorted(((x['plateau_height'] - x['local_reference_height']), x['region']) for x in t)
    return {'value': round(float(lift[-1][0]), 3),
            'detail': {'towns': len(t), 'over_.3': sum(v > .3 for v, _ in lift),
                       'top': [[n, round(float(v), 3)] for v, n in lift[::-1][:8]]}}


def _lake_shore(ctx):
    """The lake-shore zone of polish's floor ramp (height bake L): lake tiles (3k_main_sea_lake hexes outside the native
    map), zone = land within 40 tile cells of them, outside the native map, not within 24 cells of a frontier river
    hex; the LF heights (south-up) and per-lake windows (LF pixels) round each lake component. Cached on ctx.
    Missing (the lake shore checks skip) with the lake basin switch off (TW3K_LAKE_BOWL, lake stop-loss 2026-10-31):
    the lakes are the v20 ones then, shore flooding included."""
    if os.environ.get('TW3K_LAKE_BOWL', '0') != '1':
        raise Missing('lake basin switch off (TW3K_LAKE_BOWL=0, lake stop-loss 2026-10-31): v20 lakes')
    def load():
        import campaign_world as cw
        from scipy import ndimage
        cells, names = ctx.map()
        H, W = cells.shape[:2]; TH, TW = cw.TILE_H, cw.TILE_W
        ids = (cells[:, :, 1].astype(np.int64) << 5) + (cells[:, :, 0].astype(np.int64) >> 3) - 1
        water = (cells[:, :, 0] & 3) == 1
        ok = (ids >= 0) & (ids < len(names))
        lake = water & (ids == names.index('3k_main_sea_lake')) if '3k_main_sea_lake' in names else np.zeros((H, W), bool)
        isr = np.array([n.startswith('tw3k_riv_sea_') for n in names] + [False])
        river = water & isr[np.where(ok, ids, len(names))]
        r, q = tile_hex_grid(TH, TW, H, W)
        nat = np.zeros((TH, TW), bool); nat[cw.NATIVE_TILE_SLICE] = True
        lake_t = lake[r, q] & ~nat
        near_river = ndimage.maximum_filter(river[r, q].astype(np.uint8), size=2 * 24 + 1) > 0
        zone = (~water)[r, q] & ~nat & (ndimage.distance_transform_edt(~lake_t) < 40) & ~near_river if lake_t.any() else np.zeros((TH, TW), bool)
        lab, n = ndimage.label(ndimage.binary_dilation(lake_t, iterations=12))
        boxes = [(4 * sl[0].start, 4 * sl[0].stop, 4 * sl[1].start, 4 * sl[1].stop) for sl in ndimage.find_objects(lab)]
        raw = ctx.lf()
        return {'zone': zone, 'lake': lake_t, 'raw': raw, 'boxes': boxes, 'cw': cw}
    return ctx.get('lake_shore', load)


def lake_shore_dry(ctx):
    """Land hex centres of the lake-shore zone under the water plane (LF at the hex's centre pixel 4 * (2q + 1),
    4 * (2r + q % 2 + 1) < PLANE): 0. polish's floor ramp keeps a dry disc round each (LAKE_CENTRE_R); the waterline
    may bend into a land hex's edge (v21b art review: the hex-tile check cut the lakes back to their hex outline).
    v21: 15,370 land tiles under the plane on the lake shores."""
    L = _lake_shore(ctx); cw = L['cw']; zone, raw = L['zone'], L['raw']
    cells, _ = ctx.map(); H, W = cells.shape[:2]
    rr, qq = np.mgrid[:H, :W]
    cy, cx = np.minimum(2 * rr + qq % 2 + 1, cw.TILE_H - 1), np.minimum(2 * qq + 1, cw.TILE_W - 1)
    sel = zone[cy, cx]
    hgt = cw.LEVEL + (raw[4 * cy[sel], 4 * cx[sel]].astype(np.float32) / 65535 - cw.NORM) * cw.GAIN
    low = hgt < PLANE
    return {'value': int(low.sum()), 'detail': {'shore_hexes': int(sel.sum()),
                                                'hexes': sorted(zip(qq[sel][low].tolist(), rr[sel][low].tolist()))[:30]}}


def lake_centre_wet(ctx):
    """Lake hex centres (3k_main_sea_lake outside the native map) at or above the water plane: the lake shrank there
    (a shore rule drying shallow arms: 84 in a copy run of the smooth-waterline fix, Lop Nur west). Salt pans and
    the like keep some dry (v21b: 184); the count only goes down. Metric."""
    L = _lake_shore(ctx); cw = L['cw']; raw = L['raw']
    cells, _ = ctx.map(); H, W = cells.shape[:2]
    rr, qq = np.mgrid[:H, :W]
    cy, cx = np.minimum(2 * rr + qq % 2 + 1, cw.TILE_H - 1), np.minimum(2 * qq + 1, cw.TILE_W - 1)
    sel = L['lake'][cy, cx]
    hgt = cw.LEVEL + (raw[4 * cy[sel], 4 * cx[sel]].astype(np.float32) / 65535 - cw.NORM) * cw.GAIN
    dry = hgt >= PLANE
    return {'value': int(dry.sum()), 'detail': {'lake_hexes': int(sel.sum()),
                                                'hexes': sorted(zip(qq[sel][dry].tolist(), rr[sel][dry].tolist()))[:30]}}


def _shore_px(ctx):
    """Per lake window: water (LF < PLANE) and shore pixels (dry, 4-adjacent to water, in the zone)."""
    L = _lake_shore(ctx); cw = L['cw']
    zt = L['zone']; out = []
    for y0, y1, x0, x1 in L['boxes']:
        h = cw.LEVEL + (L['raw'][y0:y1, x0:x1].astype(np.float32) / 65535 - cw.NORM) * cw.GAIN
        zl = zt | L['lake']   # the shore zone and the lake tiles (v21b: the water edge lay inside the lake hexes)
        z = np.repeat(np.repeat(zl[y0 // 4:(y1 + 3) // 4, x0 // 4:(x1 + 3) // 4], 4, 0), 4, 1)[:h.shape[0], :h.shape[1]]
        w = h < PLANE
        from scipy import ndimage
        near = ~w & (ndimage.distance_transform_edt(~w) <= 8) & z   # dry pixels within a hex (8 px) of the water
        out.append((h, w, z, near, (y0, x0)))
    return out


def lake_waterline_straight(ctx):
    """Longest straight run (LF pixels) of the water edge along one pixel row or column inside the lake-shore zone:
    a lake cut back to its hex outline draws runs of a hex or more (v21b art review, Bosten west). Metric."""
    import terrain_regression as tr
    best, where = 0, None
    for h, w, z, shore, (y0, x0) in _shore_px(ctx):
        inside = z | w
        b, _ = tr.straight_runs(w, inside)
        if b > best:
            best, where = b, [int(x0), int(y0)]
    return {'value': int(best), 'detail': {'window_px': where, 'hex_px': 8}}


def lake_shore_step(ctx):
    """95th percentile slope (world units per world unit) of the dry pixels within a hex (8 LF px) of the lake water
    (lake-shore zone and lake tiles): a floor at the full height right to the lake hexes left the basin wall inside
    them standing at the projection's grade cap, a dark step between water and shore (v21b art review); the floor
    ramp from the smooth waterline keeps the shore low and gentle there. Metric."""
    import campaign_world as cw
    px = cw.DX / 8                      # world units per LF pixel
    v = []
    for h, w, z, near, _ in _shore_px(ctx):
        if near.any():
            gy, gx = np.gradient(h, px); v.append(np.hypot(gx, gy)[near])
    if not v:
        raise Missing('no lake shore pixels')
    v = np.concatenate(v)
    return {'value': round(float(np.percentile(v, 95)), 4), 'detail': {'shore_px': int(len(v)), 'median': round(float(np.median(v)), 4)}}


def _lake_level(ctx):
    """The lake water level product and field (lake_level.py; polish TW3K_LAKE_LEVEL=1), cached on ctx: product, tile
    masks, waterline signed distances (LF grid of each lake's window) and the LF in world units over each window.
    Missing (the lake level checks skip) with the switch off: every lake on the one water plane."""
    if os.environ.get('TW3K_LAKE_LEVEL', '0') != '1':
        raise Missing('lake level switch off (TW3K_LAKE_LEVEL=0): every lake on the one water plane')
    def load():
        import campaign_world as cw
        import lake_level
        from analyze_reference import digest
        try:
            d = lake_level.read_product(ctx.ws, digest(ctx.path(MAP)))
        except FileNotFoundError as e:
            raise Missing(str(e))
        ctx.reads[lake_level.PRODUCT] = {'bytes': (ctx.ws / lake_level.PRODUCT).stat().st_size}
        ctx.reads[lake_level.FIELD] = {'bytes': (ctx.ws / lake_level.FIELD).stat().st_size}
        fields = lake_level.read_fields(ctx.ws, d)
        raw = ctx.lf()
        win = {}
        for n, (s, (y0, y1, x0, x1)) in fields.items():
            F = lake_level.FINE
            win[n] = _lf_world(cw, raw[F * y0:F * y0 + s.shape[0], F * x0:F * x0 + s.shape[1]])
        masks = {n: lake_level.cells_from_list((cw.TILE_H, cw.TILE_W), e['cells_yx']) for n, e in d['lakes'].items()}
        return {'product': d, 'masks': masks, 'fields': fields, 'h': win, 'cw': cw, 'LL': lake_level}
    return ctx.get('lake_level', load)


def _lf_world(cw, raw):
    return cw.LEVEL + (raw.astype(np.float64) / 65535 - cw.NORM) * cw.GAIN


def _lake_level_plan(ctx):
    """river_water_mesh.plan as river_water_props builds the lake models (lake_surface from the product, bank cap)."""
    L = _lake_level(ctx)
    def load():
        import map_hex
        import river_water_mesh as M
        import river_water_props as P
        m = map_hex.read(ctx.path(MAP))
        _, _, river, lake, other = P.hex_masks(m)
        lf_c, lf_lo = M.lf_cells(ctx.lf())
        F, pad = M.lake_surface(L['product'], L['fields'])
        p = M.plan(np.load(ctx.path(LEVEL)), lf_c, lf_lo, river, lake, other, cap=True, floor=F, lake_pad=pad)
        p['lf_centre'] = lf_c
        return p, M
    return ctx.get('lake_level_plan', load)


def _quad_vertices(q):
    v = np.zeros((q.shape[0] + 1, q.shape[1] + 1), bool)
    for dy in (0, 1):
        for dx in (0, 1):
            v[dy:dy + q.shape[0], dx:dx + q.shape[1]] |= q
    return v


def _lake_level_mesh(ctx):
    """name -> (vertex mask of the lake's quads (TH, TW), vertex signed distance (TH, TW)) over its window."""
    L = _lake_level(ctx); p, M = _lake_level_plan(ctx)
    out = {}
    shape = p['H'].shape
    for n, (s, (y0, y1, x0, x1)) in L['fields'].items():
        sv = np.full(shape, np.inf); sv[y0:y1, x0:x1] = L['LL'].vertex_samples(s)[:y1 - y0, :x1 - x0]
        v = _quad_vertices(p['lake_quad'])[:shape[0], :shape[1]]
        win = np.zeros(shape, bool); win[y0:y1, x0:x1] = True
        # this lake's quads: the connected lake-quad vertices that reach its pad
        from scipy import ndimage
        lab, _ = ndimage.label(v & win, np.ones((3, 3), bool))
        ks = np.unique(lab[(sv < float(L['product']['lakes'][n]['spec']['mesh_pad'])) & v]); ks = ks[ks > 0]
        out[n] = (np.isin(lab, ks), sv)
    return out


def lake_level_mesh_height(ctx):
    """Largest |water mesh height - (solved level + SURFACE_MARGIN)| over the vertices of each level lake's quads
    (river_water_mesh.plan with the product's lake_surface, bank cap included): 0."""
    L = _lake_level(ctx); p, M = _lake_level_plan(ctx)
    worst, detail = 0., {}
    for n, (v, sv) in _lake_level_mesh(ctx).items():
        want = float(L['product']['lakes'][n]['level']) + M.SURFACE_MARGIN
        h = p['H'][v]
        dev = float(np.abs(h - want).max()) if h.size else float('inf')
        worst = max(worst, dev)
        detail[n] = {'level': L['product']['lakes'][n]['level'], 'mesh_height_want': round(want, 4), 'vertices': int(v.sum()),
                     'height_min_max': [round(float(h.min()), 4), round(float(h.max()), 4)] if h.size else None}
    return {'value': round(worst, 6), 'detail': detail}


def lake_level_bed_wet(ctx):
    """LF pixels inside each level lake's waterline (signed distance s < 0) at or above its level, plus those not
    under one of its lake quads: 0 (the water covers the whole bed)."""
    L = _lake_level(ctx); mesh = _lake_level_mesh(ctx)
    out, total = {}, 0
    for n, (s, (y0, y1, x0, x1)) in L['fields'].items():
        lv = float(L['product']['lakes'][n]['level']); h = L['h'][n]
        inside = s < 0
        dry = inside & (h >= lv)
        # quad (y, x) spans the vertices (cell centres, LF pixel 4t+2) y..y+1, x..x+1: LF pixels 4y+2 .. 4y+6
        v, _ = mesh[n]
        q = v[:-1, :-1] & v[1:, :-1] & v[:-1, 1:] & v[1:, 1:]   # quads with all corners on this lake's surface
        F = L['LL'].FINE
        cov = np.zeros(s.shape, bool)
        qy, qx = np.nonzero(q[y0:y1, x0:x1])
        for dy in range(F):
            for dx in range(F):
                yy, xx = F * qy + 2 + dy, F * qx + 2 + dx
                ok = (yy < s.shape[0]) & (xx < s.shape[1])
                cov[yy[ok], xx[ok]] = True
        unc = inside & ~cov
        total += int(dry.sum()) + int(unc.sum())
        d = lv - h[inside]
        out[n] = {'level': lv, 'inside_px': int(inside.sum()), 'dry_px': int(dry.sum()), 'uncovered_px': int(unc.sum()),
                  'depth_p5_p50_max': [round(float(x), 3) for x in np.percentile(d, [5, 50, 100])] if d.size else None}
    return {'value': total, 'detail': out}


def lake_level_mesh_edge(ctx):
    """Water-mesh vertices of each level lake at or past its mesh_pad from the waterline (every rim vertex of its
    surface) whose ground (LF at the vertex, pixel 4t+2) lies under level + freeboard: 0. The vertices between the
    waterline and mesh_pad (on the bank, under the visible water edge) are reported."""
    L = _lake_level(ctx); p, M = _lake_level_plan(ctx)
    out, total = {}, 0
    for n, (v, sv) in _lake_level_mesh(ctx).items():
        e = L['product']['lakes'][n]; lv = float(e['level']); fb = float(e['spec']['freeboard']); pad = float(e['spec']['mesh_pad'])
        g = p['lf_centre']
        outer_v = v & (sv >= pad)
        bad = outer_v & (g < lv + fb)
        band = v & (sv >= 0) & (sv < pad)
        total += int(bad.sum())
        fbv = g[outer_v] - lv
        out[n] = {'freeboard': fb, 'outer_vertices': int(outer_v.sum()), 'under': int(bad.sum()),
                  'outer_ground_over_level_min_p50': [round(float(fbv.min()), 3), round(float(np.median(fbv)), 3)] if fbv.size else None,
                  'bank_vertices': int(band.sum()),
                  'bank_ground_over_level_min': round(float((g[band] - lv).min()), 3) if band.any() else None}
    return {'value': total, 'detail': out}


def lake_level_waterline(ctx):
    """LF pixels on the land side of each level lake's waterline (0 <= s <= ring_width) under its level, off the river
    band (river level finite): 0. The terrain crosses the level on the designed waterline."""
    L = _lake_level(ctx)
    lvl = np.load(ctx.path(LEVEL))
    out, total = {}, 0
    for n, (s, (y0, y1, x0, x1)) in L['fields'].items():
        e = L['product']['lakes'][n]; lv = float(e['level']); h = L['h'][n]
        F = L['LL'].FINE
        band = np.repeat(np.repeat(np.isfinite(lvl[y0:y1, x0:x1]), F, 0), F, 1)[:s.shape[0], :s.shape[1]]
        land = (s >= 0) & (s <= e['spec']['bank']['ring_width'])
        wet = land & (h < lv) & ~band
        total += int(wet.sum())
        out[n] = {'land_px': int(land.sum()), 'wet_land_px': int(wet.sum()), 'river_band_px': int((land & band).sum())}
    return {'value': total, 'detail': out}


LAKE_LEVEL_STEP_SLOPE = 3.   # frozen 2026-10-31 before the v2 run (criteria.md A5): v1 walls 1.7-6, design bed slope 2.4
LAKE_LEVEL_STEP_BAND = .5    # world units either side of the waterline


def lake_level_shore_step(ctx):
    """Largest height step between 4-neighbour LF pixels within LAKE_LEVEL_STEP_BAND of each level lake's waterline,
    as a slope (step / pixel pitch): <= LAKE_LEVEL_STEP_SLOPE (no vertical wall)."""
    L = _lake_level(ctx)
    pz, px = L['LL'].lf_pitch()
    worst, out = 0., {}
    for n, (s, _) in L['fields'].items():
        h = L['h'][n]
        nb = np.abs(s) <= LAKE_LEVEL_STEP_BAND
        sx = np.abs(np.diff(h, axis=1)) / px; mx = nb[:, 1:] & nb[:, :-1]
        sz = np.abs(np.diff(h, axis=0)) / pz; mz = nb[1:] & nb[:-1]
        m = max(float(sx[mx].max()) if mx.any() else 0., float(sz[mz].max()) if mz.any() else 0.)
        worst = max(worst, m)
        out[n] = {'max_slope': round(m, 3), 'p99_slope': round(float(np.percentile(np.concatenate([sx[mx], sz[mz]]), 99)), 3),
                  'limit': LAKE_LEVEL_STEP_SLOPE}
    return {'value': round(worst, 3), 'detail': out}


LAKE_LEVEL_AXIS_SHARE = .35   # frozen 2026-10-31 before the v2 run (criteria A5): vanilla .08-.32, hex block edge .49-.67


def _river_band_px(ctx, shape, win):
    import lake_level
    y0, y1, x0, x1 = win
    lvl = np.load(ctx.path(LEVEL))
    F = lake_level.FINE
    return np.repeat(np.repeat(np.isfinite(lvl[y0:y1, x0:x1]), F, 0), F, 1)[:shape[0], :shape[1]]


LAKE_LEVEL_AXIS_EXCESS = .05   # frozen 2026-10-31 before the v4 run (lakelevel-runs/v4/criteria.md A5b; v5: reference amended, gate kept): < half of v3's
                               # allowance over a round outline (.35 - .222) and of vanilla's (.32 - .222); hex edge +.27..+.45


def _axis_share(gz, gx):
    ang = np.degrees(np.arctan2(np.abs(gz), np.abs(gx)))
    return float(((ang <= 10) | (ang >= 80)).mean()) if ang.size else 0.


def lake_level_waterline_sawtooth(ctx):
    """Excess hex / tile sawtooth on the terrain's actual level crossing over the designed surface (v5 A5b, amended by
    user approval 2026-10-31, lakelevel-runs/v5/criteria.md; v4 compared with the outline normals grad s, but the
    designed profile itself scores .095 there: the slope bend at the waterline, bank floor .2 / bed 2.4, skews central
    differences towards the axes; v3 gated the raw share against .35): crossing pixels = the LF pixels at or above each
    level lake's level with a 4-neighbour under it, within LAKE_LEVEL_STEP_BAND of the waterline, off the river band,
    |grad h| > 1e-3; at them, the share of terrain-gradient directions within 10 degrees of the x / z axes minus the
    same share of the designed surface's gradient, f(s) = level + bank_floor(s) (s >= 0) / level - bed_depth(s)
    (s < 0), same np.gradient: <= LAKE_LEVEL_AXIS_EXCESS. Detail: both shares, the v4 reference (grad s) share and
    excess, the raw share against the v3 .35, and the median / p90 angle between grad h and grad f there."""
    L = _lake_level(ctx)
    pz, px = L['LL'].lf_pitch()
    worst, out = -1., {}
    for n, (s, win) in L['fields'].items():
        h = L['h'][n]; e = L['product']['lakes'][n]; lv = float(e['level'])
        s = np.asarray(s, np.float64)
        up = h >= lv
        low = np.zeros(h.shape, bool)
        low[1:] |= ~up[:-1]; low[:-1] |= ~up[1:]; low[:, 1:] |= ~up[:, :-1]; low[:, :-1] |= ~up[:, 1:]
        gz, gx = np.gradient(h, pz, px)
        sz, sx = np.gradient(s, pz, px)
        f = np.where(s >= 0, lv + L['LL'].bank_floor(s, e['spec']['bank'], float(e['spec']['freeboard'])),
                     lv - L['LL'].bed_depth(s, e['spec']['bed']))
        fz, fx = np.gradient(f, pz, px)
        wl = up & low & (np.abs(s) <= LAKE_LEVEL_STEP_BAND) & ~_river_band_px(ctx, s.shape, win) & (np.hypot(gz, gx) > 1e-3)
        sh_h, sh_f, sh_s = _axis_share(gz[wl], gx[wl]), _axis_share(fz[wl], fx[wl]), _axis_share(sz[wl], sx[wl])
        dev = np.degrees(np.abs(np.angle(np.exp(1j * (np.arctan2(gz[wl], gx[wl]) - np.arctan2(fz[wl], fx[wl]))))))
        worst = max(worst, sh_h - sh_f)
        out[n] = {'crossing_px': int(wl.sum()), 'axis_share': round(sh_h, 3), 'designed_surface_axis_share': round(sh_f, 3),
                  'excess': round(sh_h - sh_f, 3), 'limit': LAKE_LEVEL_AXIS_EXCESS,
                  'v4_outline_axis_share': round(sh_s, 3), 'v4_excess_vs_outline': round(sh_h - sh_s, 3), 'v3_raw_limit': LAKE_LEVEL_AXIS_SHARE,
                  'angle_to_design_deg_p50_p90': [round(float(v), 2) for v in np.percentile(dev, [50, 90])] if dev.size else None}
    return {'value': round(max(worst, 0.), 3), 'detail': out}


def lake_level_bed_profile(ctx):
    """v5 (polish LAKE LEVEL BED, lake_level.cut_bed): every LF pixel under each level lake's waterline (s < 0), off the
    river band (RIVER BEDS may cut deeper there), lies on the designed bed, level - bed_depth(s), within one LF
    quantization step (GAIN / 65535; the cut rounds to the nearest step: <= half of it). Value: pixels off it.
    Detail: pixel counts, largest deviation off and on the river band."""
    L = _lake_level(ctx)
    lsb = L['cw'].GAIN / 65535
    out, total = {}, 0
    for n, (s, win) in L['fields'].items():
        e = L['product']['lakes'][n]; lv = float(e['level'])
        s = np.asarray(s, np.float64); wet = s < 0
        dev = np.abs(L['h'][n] - (lv - L['LL'].bed_depth(s, e['spec']['bed'])))
        band = _river_band_px(ctx, s.shape, win)
        m, mb = wet & ~band, wet & band
        bad = int((dev[m] > lsb).sum()); total += bad
        out[n] = {'bed_px': int(m.sum()), 'off_profile_px': bad, 'max_dev': round(float(dev[m].max()), 6) if m.any() else None,
                  'tolerance': round(lsb, 6), 'river_band_bed_px': int(mb.sum()),
                  'river_band_max_dev': round(float(dev[mb].max()), 6) if mb.any() else None}
    return {'value': total, 'detail': out}


def lake_level_bank_envelope(ctx):
    """No terrace on the bank (v3): LF pixels of each level lake's window with 0 <= s <= ring_width, off the river band,
    higher than level + town_envelope(s) + 2 LSB: 0 (the bank and the town ground on the water side follow the
    vanilla bank envelope)."""
    L = _lake_level(ctx); cw = L['cw']
    tol = 2 * cw.GAIN / 65535
    out, total = {}, 0
    for n, (s, win) in L['fields'].items():
        e = L['product']['lakes'][n]; lv = float(e['level']); h = L['h'][n]
        band = (s >= 0) & (s <= e['spec']['bank']['ring_width']) & ~_river_band_px(ctx, s.shape, win)
        over = h - lv - L['LL'].town_envelope(np.maximum(s, 0), e['spec'])
        bad = band & (over > tol)
        total += int(bad.sum())
        out[n] = {'bank_px': int(band.sum()), 'over_envelope_px': int(bad.sum()),
                  'max_over': round(float(over[band].max()), 4) if band.any() else None,
                  'worst_s': round(float(s[band][np.argmax(over[band])]), 3) if band.any() else None}
    return {'value': total, 'detail': out}


def plots_on_lakes(ctx):
    """map.hex settlement / resource plot hexes (slot or plot flag) on 3k_main_sea_lake water hexes: 0."""
    cells, names = ctx.map()
    import map_hex
    ids = map_hex.region_ids(cells)
    land = (cells[:, :, 0] & 3) != 1
    if '3k_main_sea_lake' not in names:
        return {'value': 0, 'detail': {'lake_hexes': 0}}
    lake = (ids == names.index('3k_main_sea_lake')) & ~land
    plot = ((cells[:, :, 2] >> 4) > 0) | ((cells[:, :, 3] & 1) > 0)
    rr, qq = np.nonzero(plot & lake)
    return {'value': int(len(rr)), 'detail': {'lake_hexes': int(lake.sum()), 'plot_hexes_on_lake': sorted(zip(qq.tolist(), rr.tolist()))[:20]}}


def _lake_level_plateaus(ctx):
    """name -> {town: (plateau LF pixel rows, cols in the lake window grid, centre height, centre s)} for every design town
    whose plateau (the connected LF region within .005 of the LF at the town centre, at most 40 tile cells out) has
    pixels inside the lake's distance window."""
    L = _lake_level(ctx)
    def load():
        from scipy import ndimage
        design = json.loads(ctx.path('work/current-region-design.json').read_text(encoding='utf-8'))
        centres = L['LL'].town_centres(design['regions'])
        raw = ctx.lf(); cw = L['cw']; F = L['LL'].FINE
        out = {}
        for n, (s, (y0, y1, x0, x1)) in L['fields'].items():
            out[n] = {}
            for t, (cy, cx) in sorted(centres.items()):
                if not (y0 - 40 <= cy < y1 + 40 and x0 - 40 <= cx < x1 + 40):
                    continue
                wy0, wx0 = max(0, F * (cy - 40)), max(0, F * (cx - 40))
                win = _lf_world(cw, raw[wy0:F * (cy + 41), wx0:F * (cx + 41)])
                c = (F * cy + 2 - wy0, F * cx + 2 - wx0)
                hc = float(win[c])
                lab, _ = ndimage.label(np.abs(win - hc) <= .005)
                py, pxx = np.nonzero(lab == lab[c])
                gy, gx = py + wy0 - F * y0, pxx + wx0 - F * x0
                ok = (gy >= 0) & (gy < s.shape[0]) & (gx >= 0) & (gx < s.shape[1])
                if not ok.any():
                    continue
                sc = (F * cy + 2 - F * y0, F * cx + 2 - F * x0)
                sv = float(s[sc]) if 0 <= sc[0] < s.shape[0] and 0 <= sc[1] < s.shape[1] else None
                out[n][t] = (gy[ok], gx[ok], hc, sv, int(len(py)))
        return out
    return ctx.get('lake_level_plateaus', load)


def lake_level_towns(ctx):
    """Town plateaus inside each level lake's distance window (v3 scope; v2 judged the centre of a town 23 world away),
    v4 rule (lakelevel-runs/v4/criteria.md A7): plateau pixels inside the waterline (s < 0), plateau pixels under the
    water mesh (s < mesh_pad) below the drawn water surface (level + SURFACE_MARGIN: flooded), and plateau-edge steps
    (4-neighbour pairs plateau / non-plateau whose outside pixel lies on the bank, 0 <= s <= ring_width, off the river
    band) steeper than LAKE_LEVEL_STEP_SLOPE: 0 (with lake_level_town_envelope: the plateau never over the bank
    envelope). Detail per town: centre height over the level, centre and nearest plateau pixel distance to the
    waterline, plateau pixels under level + freeboard (v3's rule, reported), largest edge slope."""
    L = _lake_level(ctx)
    pz, px = L['LL'].lf_pitch()
    out, total = {}, 0
    for n, towns in _lake_level_plateaus(ctx).items():
        s, win = L['fields'][n]; e = L['product']['lakes'][n]; lv = float(e['level'])
        fb, pad = float(e['spec']['freeboard']), float(e['spec']['mesh_pad'])
        rw = float(e['spec']['bank']['ring_width'])
        h = L['h'][n]
        bank = (s >= 0) & (s <= rw) & ~_river_band_px(ctx, s.shape, win)
        for t, (gy, gx, hc, sc, npx) in towns.items():
            sv, hv = s[gy, gx], h[gy, gx]
            wet = int((sv < 0).sum()); flood = int(((sv >= 0) & (sv < pad) & (hv < lv + L['LL'].SURFACE_MARGIN)).sum())
            under_fb = int(((sv >= 0) & (sv < pad) & (hv < lv + fb)).sum())
            pl = np.zeros(s.shape, bool); pl[gy, gx] = True
            edge = 0.; nedge = 0
            for ax, pitch in ((0, pz), (1, px)):
                a = [slice(None)] * 2; b = [slice(None)] * 2
                for lo_, hi_ in ((slice(None, -1), slice(1, None)), (slice(1, None), slice(None, -1))):
                    a[ax], b[ax] = lo_, hi_
                    m = pl[tuple(a)] & ~pl[tuple(b)] & bank[tuple(b)]
                    if m.any():
                        st = np.abs(h[tuple(a)][m] - h[tuple(b)][m]) / pitch
                        nedge += int((st > LAKE_LEVEL_STEP_SLOPE).sum()); edge = max(edge, float(st.max()))
            total += wet + flood + nedge
            out[t] = {'centre_over_level': round(hc - lv, 3), 'centre_to_waterline': None if sc is None else round(sc, 2),
                      'plateau_px': npx, 'plateau_px_in_window': int(len(gy)), 'nearest_plateau_s': round(float(sv.min()), 3),
                      'plateau_px_in_water': wet, 'plateau_px_flooded': flood, 'plateau_px_under_freeboard_v3': under_fb,
                      'edge_slope_max': round(edge, 3), 'edge_steps_over_limit': nedge}
    return {'value': total, 'detail': out}


def lake_level_town_envelope(ctx):
    """Town ground obeys the bank envelope (v3): plateau pixels of the lake_level_towns towns with s >= 0 higher than
    level + town_envelope(s) + 2 LSB: 0."""
    L = _lake_level(ctx); cw = L['cw']
    tol = 2 * cw.GAIN / 65535
    out, total = {}, 0
    for n, towns in _lake_level_plateaus(ctx).items():
        s = L['fields'][n][0]; e = L['product']['lakes'][n]; lv = float(e['level']); h = L['h'][n]
        for t, (gy, gx, hc, sc, npx) in towns.items():
            sv = s[gy, gx]; m = sv >= 0
            over = h[gy, gx][m] - lv - L['LL'].town_envelope(sv[m], e['spec'])
            bad = int((over > tol).sum())
            total += bad
            if m.any():
                out[t] = {'over_px': bad, 'max_over': round(float(over.max()), 4), 'at_s': round(float(sv[m][np.argmax(over)]), 3)}
    return {'value': total, 'detail': out}


LAKE_LEVEL_WET_SHARE = .986   # vanilla: lake hexes 98.6-100% wet (lessons.md 1)


def lake_level_lake_hex_wet(ctx):
    """Share of each level lake's lake-hex LF pixels NOT under the water surface (LF >= level + SURFACE_MARGIN): <=
    1 - LAKE_LEVEL_WET_SHARE (vanilla). Detail: lake-hex pixels on the land side of the designed waterline (0 by
    construction, lake_level.outline_sdf)."""
    L = _lake_level(ctx)
    worst, out = 0., {}
    for n, (s, (y0, y1, x0, x1)) in L['fields'].items():
        e = L['product']['lakes'][n]; lv = float(e['level']); h = L['h'][n]; F = L['LL'].FINE
        lk = np.repeat(np.repeat(L['masks'][n][y0:y1, x0:x1], F, 0), F, 1)[:s.shape[0], :s.shape[1]]
        dry = lk & (h >= lv + L['LL'].SURFACE_MARGIN)
        share = float(dry.sum()) / max(1, int(lk.sum()))
        worst = max(worst, share)
        out[n] = {'lake_hex_px': int(lk.sum()), 'dry_px': int(dry.sum()), 'wet_share': round(1 - share, 4),
                  'designed_dry_px': int((lk & (s >= 0)).sum())}
    return {'value': round(worst, 4), 'detail': out}


def lake_level_outside_identical(ctx):
    """LF pixels outside each level lake's reach (lake_level.reach_mask: the reach polish derived from the cluster mix's
    input and wrote to the product, v3) that differ from the switch-off run's LF (TW3K_LAKE_LEVEL_REF_LF: a stage
    lf_height_map.dds of the same stage with TW3K_LAKE_LEVEL=0; Missing without it): 0."""
    L = _lake_level(ctx); cw = L['cw']
    ref = os.environ.get('TW3K_LAKE_LEVEL_REF_LF')
    if not ref or not Path(ref).is_file():
        raise Missing('TW3K_LAKE_LEVEL_REF_LF (switch-off LF) not set')
    d = Path(ref).read_bytes(); h, w = struct.unpack_from('<2I', d, 12)
    a = np.frombuffer(d, '<u2', offset=128, count=h * w).reshape(h, w)[::-1]
    b = ctx.lf()
    if a.shape != b.shape:
        raise Missing(f'reference LF {a.shape} is not {b.shape}')
    near, towns = L['LL'].reach_mask(ctx.ws, L['product'], (cw.TILE_H, cw.TILE_W))
    F = L['LL'].FINE
    diff = a != b
    ys, xs = np.nonzero(diff)
    inside = near[np.minimum(ys // F, near.shape[0] - 1), np.minimum(xs // F, near.shape[1] - 1)]
    oy, ox = ys[~inside], xs[~inside]
    return {'value': int(len(oy)), 'detail': {'differing_px': int(len(ys)), 'inside_reach_px': int(inside.sum()),
                                              'reach_cells': int(near.sum()), 'reach_towns': {k: len(v) for k, v in towns.items()},
                                              'differing_bbox_lf_yx': [int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())] if len(ys) else None,
                                              'outside_bbox_tile_yx': [int(oy.min()) // F, int(oy.max()) // F, int(ox.min()) // F, int(ox.max()) // F] if len(oy) else None}}


def town_rim_step(ctx):
    """The step round the canvas town plateaus: per town (design town outside the native map), the plateau is the
    connected LF region within .005 of the LF at the town centre; the rim is the LF pixels within RIM_WU (.5 world
    units) outside it; the town's value is the 95th percentile of |LF - plateau level| on the rim. The invariant is
    the median over the towns (detail: the 95th percentile and the worst towns; the worst are coast and lake-shore
    towns whose ground falls to the water, which no town mix changes). v21 (copy): median .1107, p95 .5009; option A (copy run 2026-10-31): median .0512. A plateau whose surrounding ground is the town's own
    level (polish town ground mix, option A) has a soft rim; a grade-cap step round it (v21: every plateau, Wusun's
    creases) reads as a line in the fine mesh."""
    import campaign_world as cw
    from scipy import ndimage
    plan = json.loads(ctx.path('work/current-region-design.json').read_text(encoding='utf-8'))
    raw = ctx.lf(); oh, ow = raw.shape
    px = cw.DX / 2 / 4                      # world units per LF pixel (x); z is close enough for a band width
    rim = max(1, int(round(.5 / px)))
    vals = []
    nx0, nx1, ny0, ny1 = 4 * cw.NATIVE_TILE_X0, 4 * cw.NATIVE_TILE_X1, 4 * cw.NATIVE_TILE_Y0, 4 * cw.NATIVE_TILE_Y1
    for t in plan['regions']:
        tq, tr = t['town']
        cx, cy = int(tq * 8), int(round((tr + tq % 2 * .5) * 8))
        if nx0 <= cx < nx1 and ny0 <= cy < ny1:
            continue
        R = 120 if t.get('is_capital') else 80
        if not (R <= cx < ow - R and R <= cy < oh - R):
            continue
        w = (cw.LEVEL + (raw[cy - R:cy + R + 1, cx - R:cx + R + 1].astype(np.float32) / 65535 - cw.NORM) * cw.GAIN)
        lev = float(np.median(w[R - 2:R + 3, R - 2:R + 3]))
        flat, _ = ndimage.label(np.abs(w - lev) < .005)
        k = flat[R, R]
        if k == 0:
            continue
        plat = flat == k
        band = ~plat & (ndimage.distance_transform_edt(~plat) <= rim)
        if band.sum() < 8:
            continue
        vals.append((float(np.percentile(np.abs(w[band] - lev), 95)), t['region']))
    if not vals:
        raise Missing('no canvas town plateaus measured')
    v = np.array([a for a, _ in vals])
    vals.sort(reverse=True)
    return {'value': round(float(np.median(v)), 4),
            'detail': {'towns': len(vals), 'p95': round(float(np.percentile(v, 95)), 4), 'worst': [[n, round(a, 3)] for a, n in vals[:10]]}}


def ground_banned_px(ctx):
    """Pixels of the stage global_blend holding a slot banned for their tile's landform (the generator's own
    field: landform-tiles L = dom where domw > BAN_MIN, whitelist bits 'allowed'): bake step 4's invariant.
    ground_materials asserts 0 after its own pass; polish's ban ledger (approach_report ban_ledger) names the
    later writer that adds any. Hard since the town spec fix: v18-v20's 926 (all drymtn, all within 34 px of
    tw3k_hist_pulei_capital) were polish town ground copying its donor's slots; it now skips banned template slots."""
    import biome_profiles as bp, ground_materials as gm
    ctx.path(LANDFORM_TILES)
    with np.load(ctx.path(LANDFORM_TILES)) as z:
        dom, domw = z['dom'], z['domw']
        allowed = z['allowed'] if 'allowed' in z.files else None
    L = np.where((domw > bp.BAN_MIN) & gm.BAN[1:].any(), dom, 0).astype(np.uint8); del dom, domw
    d = ctx.path(BLEND).read_bytes(); bh, bw = struct.unpack_from('<2I', d, 12)
    cur = np.frombuffer(d, np.uint8, offset=128, count=bh * bw * 2).reshape(bh, bw, 2)
    n = gm.count_banned(cur, L, allowed, bh)
    return {'value': int(n), 'detail': {'enforced': list(bp.BAN_ENFORCED)}}



def tundra_pale_share(ctx):
    """Share of pale sand slots (arid_1-4, steppe_1/2) in the stage global_blend on tundra-dominated tiles (landform
    tiles dom = tundra, domw > BAN_MIN): the pale 'camouflage' blobs of art review tundra-mid (§9.3). v21 .15 (5 frontier
    windows .137-.153); TW3K_TUNDRA_BAN .077; + TW3K_TUNDRA_SCREE .033 (windowed replica 2026-10-31)."""
    import biome_profiles as bp
    if not bp.tundra_scree_on():
        raise Missing('tundra scree mix switched off (TW3K_TUNDRA_SCREE=1 turns it on)')
    with np.load(ctx.path(LANDFORM_TILES)) as z:
        t = (z['dom'] == list(bp.LANDFORMS).index('tundra') + 1) & (z['domw'] > bp.BAN_MIN)
    d = ctx.path(BLEND).read_bytes(); bh, bw = struct.unpack_from('<2I', d, 12)
    r = np.frombuffer(d, np.uint8, offset=128, count=bh * bw * 2).reshape(bh, bw, 2)[::-1, :, 0]   # south-up: row 4t.. = tile t
    th, tw = min(t.shape[0], bh // 4), min(t.shape[1], bw // 4)
    m = np.repeat(np.repeat(t[:th, :tw], 4, 0), 4, 1)
    rr = r[:4 * th, :4 * tw][m]
    n = int(rr.size)
    pale = int(np.isin(rr, (0, 1, 2, 3, 19, 20)).sum())
    return {'value': round(pale / max(n, 1), 4), 'detail': {'tundra_px': n, 'pale_px': pale}}


def tundra_arid3(ctx):
    """arid_3 pixels in the stage global_blend on tundra-dominated tiles (as tundra_pale_share). TUNDRA_SCREE has no
    arid_3 since 2026-11-01: the v22b live check found the pale camouflage blobs on exactly those pixels (65 % of them
    pale live, <scratch>/tundra-live-20261101/README.md). Fix: biome_profiles TUNDRA_SCREE / TUNDRA_BAN_EXTRA."""
    import biome_profiles as bp
    if not bp.tundra_scree_on():
        raise Missing('tundra scree mix switched off (TW3K_TUNDRA_SCREE=1 turns it on)')
    with np.load(ctx.path(LANDFORM_TILES)) as z:
        t = (z['dom'] == list(bp.LANDFORMS).index('tundra') + 1) & (z['domw'] > bp.BAN_MIN)
    d = ctx.path(BLEND).read_bytes(); bh, bw = struct.unpack_from('<2I', d, 12)
    r = np.frombuffer(d, np.uint8, offset=128, count=bh * bw * 2).reshape(bh, bw, 2)[::-1, :, 0]
    th, tw = min(t.shape[0], bh // 4), min(t.shape[1], bw // 4)
    m = np.repeat(np.repeat(t[:th, :tw], 4, 0), 4, 1)
    n = int((r[:4 * th, :4 * tw][m] == 2).sum())
    return {'value': n, 'detail': {'tundra_px': int(m.sum())}}


def tundra_cold2_treeless(ctx):
    """cold_2 (woodland floor) pixels in the stage global_blend on treeless tundra tiles (tundra-dominated as
    tundra_pale_share, landform tiles forest_density < .2 as woodland_off_forest). TUNDRA_SCREE has no cold_2 since
    2026-11-01: on open tundra it drew hard-edged dark-green spots (v22b live). Fix: biome_profiles TUNDRA_SCREE /
    TUNDRA_BAN_EXTRA."""
    import biome_profiles as bp
    if not bp.tundra_scree_on():
        raise Missing('tundra scree mix switched off (TW3K_TUNDRA_SCREE=1 turns it on)')
    with np.load(ctx.path(LANDFORM_TILES)) as z:
        t = (z['dom'] == list(bp.LANDFORMS).index('tundra') + 1) & (z['domw'] > bp.BAN_MIN) & (z['forest_density'] < .2)
    d = ctx.path(BLEND).read_bytes(); bh, bw = struct.unpack_from('<2I', d, 12)
    r = np.frombuffer(d, np.uint8, offset=128, count=bh * bw * 2).reshape(bh, bw, 2)[::-1, :, 0]
    th, tw = min(t.shape[0], bh // 4), min(t.shape[1], bw // 4)
    m = np.repeat(np.repeat(t[:th, :tw], 4, 0), 4, 1)
    n = int((r[:4 * th, :4 * tw][m] == 5).sum())
    return {'value': n, 'detail': {'treeless_tundra_px': int(m.sum())}}


def tundra_oasis(ctx):
    """Tiles of the 'no_oasis' classes (tundra, snow, grass north of TUNDRA_GRASS_LAT) north of NO_OASIS_LINE with an oasis weight, plus
    'no_alpine_belt' tiles with an alpine (keep) weight, in the landform tiles polish saved. Arctic rivers took desert
    oasis belts (tile climate 3 x low valley): the tundra rule was zeroed there and the river corridor stayed pale
    (v22b tundra-mid). Fix: polish [TUNDRA OASIS OFF] (biome_profiles no_oasis_masks, TW3K_TUNDRA_OASIS_OFF)."""
    import biome_profiles as bp
    if not bp.tundra_oasis_off_on():
        raise Missing('tundra oasis gate switched off (TW3K_TUNDRA_OASIS_OFF=1 turns it on)')
    with np.load(ctx.path(LANDFORM_TILES)) as z:
        meta = json.loads(str(z['meta'].reshape(-1)[0])); eco = z['eco_bio']; oasis = z['oasis']; keep = z['keep']
    with np.load(ctx.path(REAL)) as g:
        lat = np.zeros(eco.shape, np.float32); x0, x1, y0, y1 = (int(v) for v in g['box']); lat[y0:y1, x0:x1] = g['lat']
        lon = np.zeros(eco.shape, np.float32); lon[y0:y1, x0:x1] = g['lon']
    origin = json.loads(ctx.path('work/canvas-authored-coordinates.json').read_text(encoding='utf-8'))['noise_origin']
    yy, xx = np.mgrid[:eco.shape[0], :eco.shape[1]].astype(np.float32)
    no_o, no_a = bp.no_oasis_masks(eco, list(meta['biome_names']), lat, xx / 2, yy / 2, tuple(origin)); del yy, xx
    n_o, n_a = int(((oasis > 0) & no_o).sum()), int(((keep > 0) & no_a).sum())
    (lo0, lo1), (la0, la1) = bp.NO_OASIS_LINE['ladakh_box']
    n_l = int(((no_o | no_a) & (lon >= lo0) & (lon <= lo1) & (lat >= la0) & (lat <= la1)).sum())   # the gate never reaches Ladakh
    return {'value': n_o + n_a + n_l, 'detail': {'oasis_tiles': n_o, 'alpine_belt_tiles': n_a, 'gate_tiles_in_ladakh_box': n_l}}

def forest_fill_floor(ctx):
    """Tiles of the fill_floor classes whose forest fill lies strictly between 0 and FOREST_FILL_FLOOR in the landform tiles polish saved: with
    TW3K_FOREST_FILL_FLOOR the fade (frontier_forest.fill_weight) never drops below the floor, and the exclusions
    (native rectangle / native_keep, west band) set 0, so anything in between means a writer bypassed the shared fade
    (ground and tree fill would read different weights). Fix: frontier_forest.fill_weight / biome_profiles.forest_fill."""
    import biome_profiles as bp
    if not bp.forest_fill_floor_on():
        raise Missing('forest fill floor switched off (TW3K_FOREST_FILL_FLOOR=1 turns it on)')
    with np.load(ctx.path(LANDFORM_TILES)) as z:
        if 'forest_fill' not in z.files:
            raise Missing('landform tiles carry no forest_fill')
        f = z['forest_fill']; meta = json.loads(str(z['meta'].reshape(-1)[0]))
        cls = bp.fill_floor(z['eco_bio'], list(meta['biome_names'])) > 0   # the floor's classes (taiga, conifer)
    n = int((cls & (f > 0) & (f < bp.FOREST_FILL_FLOOR - 1e-6)).sum())
    return {'value': n, 'detail': {'tiles_at_floor': int(np.isclose(f, bp.FOREST_FILL_FLOOR).sum()), 'tiles_filled': int((f > 0).sum())}}


TREELINE_MEMO = 'analysis/treeline-straight-memo.json'   # written only by treeline_straight


def _treeline_key(src, sel, rule, tones, band_rows=1024):
    """sha256 of what line_numbers reads: the blend file, the land mask band() derives from the LF (same arithmetic,
    row band by row band), the class-line hexes, the rule, the tones and the code of both modules."""
    import campaign_world as cw
    import ground_patchiness as gp
    h = hashlib.sha256()
    with open(src.src['blend'][0], 'rb') as f:
        f.seek(src.src['blend'][1])
        h.update(hashlib.file_digest(f, 'sha256').digest())
    for y0 in range(0, src.h, band_rows):
        raw = src.rows('lf', y0, min(src.h, y0 + band_rows))
        h.update(np.packbits((cw.LEVEL + (raw.astype(np.float32) / 65535 - cw.NORM) * cw.GAIN) >= gp.PLANE).tobytes())
    h.update(np.packbits(sel).tobytes() + repr(sel.shape).encode())
    h.update(json.dumps(rule, sort_keys=True).encode())
    h.update(np.ascontiguousarray(tones.lab).tobytes())
    for m in (gp, sys.modules[__name__]):
        h.update(hashlib.sha256(Path(m.__file__).read_bytes()).digest())
    return h.hexdigest()


def treeline_straight(ctx):
    """Straight hard-edge segments (ground_patchiness straight segment rule) per 100 land hexes in the tundra/forest
    transition zone (ground_patchiness.class_line_hexes: hexes within TREELINE_DONOR band + 2 tiles of a 'from' tile
    touching a cold 'forest' tile, so an edge moved off the class line still counts; line_numbers, blocks of 96
    hexes): the donor climate switched on the tile-grid class line (art review dc7a9361 section 1: staircases and
    straight runs at mid range, which tundra_forest_straight, the large-scale contour, did not see). Fix
    ground_donor.treeline_weight / treeline_pick (TW3K_TREELINE_BLEND)."""
    import biome_profiles as bp
    import ground_patchiness as gp
    rule = bp.consts('polish')['TREELINE_DONOR']
    with np.load(ctx.path(LANDFORM_TILES)) as z:
        meta = json.loads(str(z['meta'].reshape(-1)[0])); eco = z['eco_bio']
    sel = gp.class_line_hexes(eco, list(meta['biome_names']), rule['from'], rule['forest'], reach=int(rule['band']) + 2)
    if not sel.any():
        raise Missing('no tundra/forest class line')
    src, tones = gp.BlendSource(blend=ctx.path(BLEND), lf=ctx.path(LF)), gp.Tones('harvest')
    # line_numbers takes ~110 s; its inputs are the blend, the LF land mask (not the heights: sync_final smooths
    # LF near the native border), sel, the tones and this code. Unchanged inputs (sync_final after polish) give the
    # recorded numbers, the same values by construction.
    key = _treeline_key(src, sel, rule, tones)
    memo = ctx.ws / TREELINE_MEMO
    try:
        rec = json.loads(memo.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        rec = {}
    if rec.get('key') == key:
        n = rec['numbers']
    else:
        n = gp.line_numbers(src.band, sel, tones)
        n = {k: n[k] for k in ('straight_segment_per100', 'land_hexes', 'hard_edge_per100', 'hard_tile_share',
                               'straight_segments', 'straight_segment_max_hex', 'blocks')}
        tmp = memo.with_name(memo.name + '.tmp')
        tmp.write_text(json.dumps({'key': key, 'numbers': n}), encoding='utf-8')
        tmp.replace(memo)
    return {'value': n['straight_segment_per100'],
            'detail': {k: n[k] for k in ('land_hexes', 'hard_edge_per100', 'hard_tile_share', 'straight_segments',
                                         'straight_segment_max_hex', 'blocks')}}


def taiga_light_share(ctx):
    """Share of the light cold slots cold_3/cold_4 (6, 7) in the stage global_blend on taiga tiles (landform tiles
    eco_bio): the light-green 'camouflage' islands of the rugged cold donor patches on the conifer floor (art review
    arctic-yenisei, section 9.3; ground_donor.compress_choice, TW3K_TAIGA_DONOR). v21 canvas .26; exact donor replica
    (four windows) .04-.135."""
    import biome_profiles as bp
    if not bp.taiga_donor_on():
        raise Missing('taiga donor compression switched off (TW3K_TAIGA_DONOR=1 turns it on)')
    with np.load(ctx.path(LANDFORM_TILES)) as z:
        meta = json.loads(str(z['meta'].reshape(-1)[0]))
        t = z['eco_bio'] == list(meta['biome_names']).index('taiga')
    d = ctx.path(BLEND).read_bytes(); bh, bw = struct.unpack_from('<2I', d, 12)
    r = np.frombuffer(d, np.uint8, offset=128, count=bh * bw * 2).reshape(bh, bw, 2)[::-1, :, 0]   # south-up: row 4t.. = tile t
    th, tw = min(t.shape[0], bh // 4), min(t.shape[1], bw // 4)
    rr = r[:4 * th, :4 * tw][np.repeat(np.repeat(t[:th, :tw], 4, 0), 4, 1)]
    light = int(np.isin(rr, (6, 7)).sum())
    return {'value': round(light / max(rr.size, 1), 4), 'detail': {'taiga_px': int(rr.size), 'light_px': light}}

FRAG_BASELINE = Path(__file__).resolve().parents[1] / 'analysis/ground-fragmentation-baseline.json'
FRAG_REBUILT = Path(__file__).resolve().parents[1] / 'analysis/ground-fragmentation-rebuilt.npz'
FRAG_MEMO = 'analysis/ground-fragmentation-memo.json'   # written only by ground_fragmentation


def _frag_key(src, dom, domw, tones, lfs, groups):
    """sha256 of what fragmentation reads: the blend file, the landform tile field (dom, domw, names), the group
    tiles, the tones and the code of both modules."""
    import ground_patchiness as gp
    h = hashlib.sha256()
    with open(src.src['blend'][0], 'rb') as f:
        f.seek(src.src['blend'][1])
        h.update(hashlib.file_digest(f, 'sha256').digest())
    for a in (dom, domw, groups):
        a = np.ascontiguousarray(a)
        h.update(repr((a.dtype.str, a.shape)).encode() + a.tobytes())
    h.update(json.dumps(lfs).encode())
    h.update(np.ascontiguousarray(tones.lab).tobytes())
    for m in (gp, sys.modules[__name__]):
        h.update(hashlib.sha256(Path(m.__file__).read_bytes()).digest())
    return h.hexdigest()


FARM_DRY_LANDFORMS = ('gobi', 'erg', 'drymtn', 'salt_crust', 'yardang', 'sgravel', 'alpine_steppe')
# farm_on_dry_ground threshold (pixels) per ground recipe (biome_profiles.ground_recipe, TW3K_GROUND_RECIPE): v24 the
# arid farm fix run, release C3 pack 42388bf0 (run 20261102-001340: 315; C2 pack 960e4de9 under this definition 34056);
# v23 (the rollback, the old 9-hex arid-town disc) the v21e pack 6b5a6234 (38415), so the rollback switch is not locked
# by the check. Lower only, never raise.
FARM_DRY_MAX_BY_RECIPE = {'v24': 315, 'v23': 38415}


def _farm_dry_max():
    import biome_profiles as bp
    return FARM_DRY_MAX_BY_RECIPE[bp.ground_recipe()]


FARM_DRY_MAX = _farm_dry_max()
WEST_REAL_TERRAIN = 'analysis/west-real-terrain.npz'


def farm_dry_tiles(ctx):
    """(bad tile mask, nearest-town tile index per tile, {tile index: town region}) of farm_on_dry_ground: canvas land
    tiles of a dry landform (FARM_DRY_LANDFORMS, domw > .5) with no water source a field may stand on - polish oasis
    belts <= .2, outside the OASES discs that water fields (radius x 1.35: their fbm warp; not FARM_OASES_SKIP) and
    OASIS_AREAS, farther than FARM_REAL_RIVER[3]
    hexes from a Natural Earth river, FARM_RIVER[3] from a map river hex, FARM_LAKE[3] from a lake hex - and farther
    than an arid town's own fields reach (ground_materials ARID_TOWN_REACH: farm_fields ARID_OWN_OASIS + the ramp's
    .5, over 1 - TOWN_WARP) from every authored town."""
    import biome_profiles as bp, ground_materials as gm, campaign_world as cw
    from scipy import ndimage
    K = bp.CONSTS['ground_materials']
    with np.load(ctx.path(LANDFORM_TILES)) as z:
        dom, domw, oasis = z['dom'], z['domw'].astype(np.float32), z['oasis'].astype(np.float32)
        real_river = z['river'] > 0
        lfs = json.loads(str(z['meta'].reshape(-1)[0]))['landforms']
    TH, TW = dom.shape
    ids = [lfs.index(n) + 1 for n in FARM_DRY_LANDFORMS if n in lfs]
    bad = np.isin(dom, ids) & (domw > .5) & (oasis <= .2); del dom, domw, oasis
    with np.load(ctx.path(WEST_REAL_TERRAIN)) as z:
        x0, x1, y0, y1 = (int(v) for v in z['box'])
        lon = np.zeros((TH, TW), np.float32); lat = np.zeros((TH, TW), np.float32)
        lon[y0:y1, x0:x1] = z['lon']; lat[y0:y1, x0:x1] = z['lat']
    for lo, la, rad in (v for n, v in gm.OASES.items() if n not in gm.FARM_OASES_SKIP):
        bad &= np.hypot((lon - lo) * 111.32 * np.cos(np.radians(la)), (lat - la) * 110.6) > 1.35 * rad
    for poly in gm.OASIS_AREAS.values():
        pa = np.asarray(poly)
        bi = np.nonzero((lon >= pa[:, 0].min()) & (lon <= pa[:, 0].max()) & (lat >= pa[:, 1].min()) & (lat <= pa[:, 1].max()))
        if len(bi[0]):
            bad[bi] &= ~gm._in_poly(lon[bi].astype(np.float64), lat[bi].astype(np.float64), poly)
    del lon, lat
    cells, names = ctx.map()
    H, W = cells.shape[:2]
    rr, qq = tile_hex_grid(TH, TW, H, W)
    hid = (cells[:, :, 1].astype(np.int64) << 5) + (cells[:, :, 0].astype(np.int64) >> 3) - 1
    water = (cells[:, :, 0] & 3) == 1
    ok = (hid >= 0) & (hid < len(names))
    isr = np.array([n.startswith('tw3k_riv_sea_') for n in names] + [False])
    river_t = (water & isr[np.where(ok, hid, len(names))])[rr, qq]
    lake_t = ((water & (hid == names.index('3k_main_sea_lake'))) if '3k_main_sea_lake' in names else np.zeros((H, W), bool))[rr, qq]
    bad &= (~water)[rr, qq]; del hid, water, ok
    for seed, reach in ((real_river, K['FARM_REAL_RIVER'][3]), (river_t, K['FARM_RIVER'][3]), (lake_t, K['FARM_LAKE'][3])):
        if seed.any():
            bad &= ndimage.distance_transform_edt(~seed) * .5 > reach
    plan = json.loads(ctx.path('work/current-region-design.json').read_text(encoding='utf-8'))
    towns = np.zeros((TH, TW), bool); tname = {}
    for t in plan['regions']:
        tq, tr = t['town']; ty, tx = min(TH - 1, int((tr + tq % 2 * .5) * 2)), min(TW - 1, tq * 2)
        towns[ty, tx] = True; tname[ty * TW + tx] = t['region']
    dist, (iy, ix) = ndimage.distance_transform_edt(~towns, return_indices=True)
    bad &= dist * .5 > gm.ARID_TOWN_REACH
    bad[cw.NATIVE_TILE_SLICE] = False
    return bad, (iy.astype(np.int64) * TW + ix), tname


def farm_on_dry_ground(ctx):
    """Farm pixels (blend R 8-12) on dry canvas ground with no water source (farm_dry_tiles): the arid rule of both farm
    writers (ground_materials.farm_mud_fields: arid ground watered only by valleys, lake shores and oases, a town's own
    fields within farm_fields ARID_OWN_OASIS; farm_fields: irrigated belts or its own ARID_OWN_OASIS disc). C2 review
    2026-11-01: an arid-town disc of 9 hexes that watered itself (ARID_TOWN, v23) drew fields on the gobi round Jushi
    and Weixu. Threshold FARM_DRY_MAX (lower only). Detail: the 10 nearest towns by pixels. Fix: biome_profiles
    CONSTS ground_materials ARID_TOWN / farm_fields ARID_OWN_OASIS, or the oasis data (OASES, oasis seeds,
    turpan_han_oasis)."""
    bad, near, tname = farm_dry_tiles(ctx)
    d = ctx.path(BLEND).read_bytes(); bh, bw = struct.unpack_from('<2I', d, 12)
    TH, TW = bad.shape
    th, tw = min(TH, bh // 4), min(TW, bw // 4)
    R = np.frombuffer(d, np.uint8, offset=128, count=bh * bw * 2).reshape(bh, bw, 2)[::-1, :, 0]
    cnt = np.zeros((TH, TW), np.int32)
    for y in range(0, th, 512):
        y2 = min(th, y + 512); b = R[4 * y:4 * y2, :4 * tw]
        cnt[y:y2, :tw] = ((b >= 8) & (b <= 12)).reshape(y2 - y, 4, tw, 4).sum(axis=(1, 3))
    del R, d
    hit = bad & (cnt > 0)
    by = {}
    for k, n in zip(near[hit].tolist(), cnt[hit].tolist()):
        by[tname[k]] = by.get(tname[k], 0) + n
    return {'value': int(cnt[hit].sum()), 'detail': {'dry_tiles': int(bad.sum()), 'farm_tiles': int(hit.sum()),
                                                     'top_towns': dict(sorted(by.items(), key=lambda kv: -kv[1])[:10])}}


def ground_fragmentation(ctx):
    """Landforms whose ground is more fragmented than allowed (ground mottle step 1, user decision 2026-11-01: after
    the recipe change the fragmentation of every landform may only improve). Per landform (ground_patchiness.fragmentation
    on the stage global_blend and landform tiles): E (colour-weighted edge density, dE76 per hex) and n100 (same-slot
    patches per 100 hexes). Outside the rebuilt area (analysis/ground-fragmentation-rebuilt.npz: regions whose layout
    changed since the baseline + band) against analysis/ground-fragmentation-baseline.json (the v21e stop-loss pack,
    old recipe, switches off); inside it against the stage's own value outside (the baseline has the old layout there;
    criteria amendment 2). Worse = above the reference by more than 'tolerance'. Value: the number of (landform,
    metric) pairs worse. Fix: the landform's mix / relief in biome_profiles LANDFORMS, or CONSTS ground_materials GRAIN
    (TW3K_GROUND_RECIPE=v23 is the old recipe)."""
    import ground_patchiness as gp
    if not FRAG_BASELINE.is_file():
        raise Missing(f'no fragmentation baseline ({FRAG_BASELINE.name})')
    base = json.loads(FRAG_BASELINE.read_text(encoding='utf-8'))
    if not FRAG_REBUILT.is_file():
        raise Missing(f'no rebuilt-area mask ({FRAG_REBUILT.name})')
    with np.load(ctx.path(LANDFORM_TILES)) as z:
        dom, domw = z['dom'], z['domw'].astype(np.float32)
        lfs = json.loads(str(z['meta'].reshape(-1)[0]))['landforms']
    with np.load(FRAG_REBUILT) as z:
        hexmask = np.unpackbits(z['mask'], count=int(np.prod(z['shape'])))[:int(np.prod(z['shape']))].reshape(tuple(z['shape'])).astype(bool)
        rebuilt_meta = json.loads(str(z['meta']))
    groups = gp.hex_to_tiles(hexmask, dom.shape)
    src, tones = gp.BlendSource(blend=ctx.path(BLEND)), gp.Tones('harvest')
    # fragmentation takes ~25 s; unchanged inputs (sync_final after polish leaves the blend and the landform tiles)
    # give the recorded numbers, the same values by construction (groups kept in order: json keys are strings).
    key = _frag_key(src, dom, domw, tones, lfs, groups)
    memo = ctx.ws / FRAG_MEMO
    try:
        rec = json.loads(memo.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        rec = {}
    if rec.get('key') == key:
        res = dict(enumerate(rec['groups']))
    else:
        res = gp.fragmentation(src, dom, domw, tones, lfs, groups=groups)
        tmp = memo.with_name(memo.name + '.tmp')
        tmp.write_text(json.dumps({'key': key, 'groups': [res[g] for g in range(len(res))]}), encoding='utf-8')
        tmp.replace(memo)
    out_, in_ = res[0], res[1]
    tol = float(base['tolerance'])
    worse = [f'{c}.{k} {out_[c][k]} > baseline {b[k]}' for c, b in base['landforms'].items() if c in out_
             for k in ('E', 'n100') if out_[c][k] > b[k] * (1 + tol)]
    worse += [f'rebuilt {c}.{k} {v[k]} > outside {out_[c][k]}' for c, v in in_.items() if c in out_
              for k in ('E', 'n100') if v[k] > out_[c][k] * (1 + tol)]
    stage_map = ctx.path(MAP)
    return {'value': len(worse), 'detail': {
        'worse': worse, 'outside': {c: {k: v[k] for k in ('E', 'n100', 'hexes')} for c, v in out_.items()},
        'rebuilt': {c: {k: v[k] for k in ('E', 'n100', 'hexes')} for c, v in in_.items()},
        'baseline_product': base.get('product'), 'rebuilt_mask': rebuilt_meta,
        'stage_map_is_mask_map': hashlib.sha256(stage_map.read_bytes()).hexdigest() == rebuilt_meta.get('map_b_sha256')}}


def blend_climate_fresh(ctx):
    """1 when analysis/blend-climate-tiles.npz was written for another global_blend than the stage's (the tile
    bake would choose tile climates from a stale G band), else 0."""
    import blend_climate_tiles as bct
    ctx.path(BLEND_CLIMATE)
    try:
        _, m = bct.load(ctx.ws, ctx.path(BLEND).read_bytes())
    except RuntimeError as e:
        return {'value': 1, 'detail': {'stale': str(e)}}
    return {'value': 0, 'detail': {'blend_sha256': m['blend_sha256'][:16]}}


def tile_climate_matches_rule(ctx):
    """Canvas floor records (generic floors of both tile lists, anchored outside the native rectangle) whose climate
    differs from blend_climate_tiles.tile_rule (carried native climate on the native ring, the blend G band's climate
    elsewhere; analysis/blend-climate-tiles.npz) at the cell their climate is taken from: native_tile_dimensions.
    floor_relay takes a floor's climate at its centre cell, (s//2, s//2) of an s x s floor (a 2x2 at its (1,1) cell),
    so a floor across a band edge follows that cell. The floors inside polish's special-piece windows (PLACED records)
    keep their donor window's climate and are skipped. The larger count of the local and global list. Skipped when
    the rule is switched off (TW3K_TILE_CLIMATE_BAND=0: canvas tiles keep the inherited climate). The tile bake lays
    canvas floors in the ground's climate (v18: 1.59M canvas cells of cold tiles on subtropical ground)."""
    import os
    import campaign_world as cw, blend_climate_tiles as bct
    if os.environ.get('TW3K_TILE_CLIMATE_BAND', '1') != '1':
        raise Missing('tile climate rule switched off (TW3K_TILE_CLIMATE_BAND=0)')
    ctx.path(BLEND_CLIMATE)
    f = bct.load(ctx.ws, ctx.path(BLEND).read_bytes(), fields=True)
    want = bct.tile_rule(f); TH, TW = want.shape
    # the floors inside polish's special-piece windows (salt pools, mountain and coast tiles: PLACED records) are
    # copied with their donor window and keep its climate; they are the window's, not the bake's floors
    placed = set()
    for rel in PLACED:
        if (ctx.ws / rel).is_file():
            placed |= {(r[0], int(r[1]), int(r[2])) for r in json.loads(ctx.path(rel).read_text(encoding='utf-8'))}
    detail = {}
    for label, rel in (('local', TILES), ('global', GTILES)):
        paths, rec = ctx.tiles(rel)
        gen = np.array([(B + 'generic' + B) in p for p in paths], bool)[rec['path'].astype(np.int64)]
        x, y = rec['x'].astype(np.int64), rec['y'].astype(np.int64)
        canvas = gen & (x >= 0) & (y >= 0) & (x < TW) & (y < TH)
        canvas &= (x < cw.NATIVE_TILE_X0) | (x >= cw.NATIVE_TILE_X1) | (y < cw.NATIVE_TILE_Y0) | (y >= cw.NATIVE_TILE_Y1)
        i = np.flatnonzero(canvas)
        i = i[[(paths[int(rec['path'][k])], int(x[k]), int(y[k])) not in placed for k in i]]
        size = np.ones(len(paths), np.int64)
        for k, p in enumerate(paths):
            m = SIZE.search(p.rstrip(B).split(B)[-1])
            if m:
                size[k] = int(m.group(1))
        c = size[rec['path'][i].astype(np.int64)] // 2   # floor_relay's climate cell: the centre
        cy, cx = np.clip(y[i] + c, 0, TH - 1), np.clip(x[i] + c, 0, TW - 1)
        bad = i[rec['climate'][i] != want[cy, cx]]
        detail[label] = {'bad': int(len(bad)), 'canvas_floor_records': int(len(i)),
                         'examples': [[int(x[k]), int(y[k]), int(rec['climate'][k])] for k in bad[:5]]}
    return {'value': max(v['bad'] for v in detail.values()), 'detail': detail}

def tile_cache_matches_list(ctx):
    """Canvas cells (outside the native rectangle) covered by a generic floor of the local tile list whose display-cache
    owner (tile_map.tiles layer 0: the cell itself, or the owner its negative path points at) carries another climate
    than that floor. The tile bake writes the cache floors and the list floors from one climate grid, so they agree;
    tile_climate_matches_rule checks the list against the rule, this the cache against the list (v19 live check: the
    cache is what the campaign draws). Main workspace run 221230: 0 of 7.58M canvas cells under a generic floor."""
    import campaign_world as cw, native_tile_grid as ntg
    TH, TW = cw.TILE_H, cw.TILE_W; n = TH * TW
    data = ctx.path(TILEMAP).read_bytes()
    L = np.frombuffer(data, dtype=ntg.CACHE_DT, count=n).reshape(TH, TW); del data
    paths, rec = ctx.tiles(TILES)
    size = np.ones(len(paths), np.int64)
    for k, p in enumerate(paths):
        m = SIZE.search(p.rstrip(B).split(B)[-1])
        if m:
            size[k] = int(m.group(1))
    gen = np.array([(B + 'generic' + B) in p for p in paths], bool)[rec['path'].astype(np.int64)]
    lc = np.full((TH, TW), 255, np.uint8)
    for sz in np.unique(size):   # floors do not overlap; paint each size class by its offsets
        k = np.flatnonzero(gen & (size[rec['path'].astype(np.int64)] == sz))
        x, y, c = rec['x'][k].astype(np.int64), rec['y'][k].astype(np.int64), rec['climate'][k]
        for dy in range(int(sz)):
            for dx in range(int(sz)):
                yy, xx = y + dy, x + dx; ok = (yy >= 0) & (yy < TH) & (xx >= 0) & (xx < TW)
                lc[yy[ok], xx[ok]] = c[ok]
    own = np.where(L['path'] >= 0, np.arange(n, dtype=np.int64).reshape(TH, TW), -L['path'].astype(np.int64) - 1)
    oc = L['climate'].ravel()[np.clip(own, 0, n - 1)]; del own
    nat = np.zeros((TH, TW), bool); nat[cw.NATIVE_TILE_SLICE] = True
    m = (L['flags'] != 16) & ~nat & (lc != 255)
    bad = m & (oc != lc)
    ys, xs = np.nonzero(bad)
    return {'value': int(bad.sum()), 'detail': {'canvas_cells_with_floor': int(m.sum()),
                                              'examples': [[int(xs[i]), int(ys[i]), int(oc[ys[i], xs[i]]), int(lc[ys[i], xs[i]])] for i in range(min(5, len(xs)))]}}

def seam_static_classes(ctx):
    """lf_cliff + fine_range_stale hexes of the first seam scan: the seam loop repairs coarse meshes only, so
    these never close in later passes (fine_uncovered / fine_special_removed / fine_climate_patch stay ungated).
    fine_range_stale is the tile bake's range gate: a plain "lo/hi brackets the LF under the footprint" check fails on
    46k records of vanilla dlc07 itself (stored ranges follow vanilla's own padding), so none is added at the graph join."""
    rep = json.loads(ctx.path(SEAM0).read_text(encoding='utf-8'))
    c = {k: int(rep['classes'][k]['hexes']) for k in ('lf_cliff', 'fine_range_stale')}
    return {'value': sum(c.values()), 'detail': c}


def native_overlay_preservation(ctx):
    """audit_native_overlay_preservation.py's own result (protected vanilla special tiles missing + Hehuang
    leaks), checked against the current stage tile list. calibrated-ranges rewrites the lo/hi ranges after the
    audit, so the full sha never matches at the graph join: the audit's tile_list_identity_sha256 (records with
    lo/hi zeroed, read_tile_list.identity_sha256) is compared. Range-only rewrites match; a mismatch means a graph
    node after overlay-audit added, removed or changed tile records, the audited result was undone: a FAILURE
    (value 1 + the audit's own count), attributed to TILES_AFTER_OVERLAY_AUDIT. Only a legacy report without the
    identity field falls back to the full tile_list_sha256, and a mismatch there is a skip."""
    from read_tile_list import identity_sha256
    rep = json.loads(ctx.path(OVERLAY).read_text(encoding='utf-8'))
    data = ctx.path(TILES).read_bytes()
    v = next((x for x in rep['variants'] if x['variant'] == VARIANT), None)
    if v is None:
        raise Missing(f'{OVERLAY} has no {VARIANT}')
    reason = None
    if 'tile_list_identity_sha256' in v:
        now = identity_sha256(data)
        if v['tile_list_identity_sha256'] != now:
            reason = (f'tile_list records changed after the overlay audit (identity digest mismatch: audit '
                      f'{v["tile_list_identity_sha256"][:16]}, now {now[:16]}); written after overlay-audit by graph '
                      f'node(s) {", ".join(TILES_AFTER_OVERLAY_AUDIT)}')
    elif v.get('tile_list_sha256') != hashlib.sha256(data).hexdigest():
        raise Missing(f'{OVERLAY} is stale (legacy report without tile_list_identity_sha256; tile_list.bin bytes changed)')
    del data
    missing = sum(len(x['missing']) + len(x.get('hehuang', {}).get('leaks', [])) for x in rep['variants'])
    value = missing if missing or rep['passed'] else 1
    out = {'value': value + (1 if reason else 0),
           'detail': {'passed': rep['passed'], 'protected': v['protected_special_tiles'], 'missing': len(v['missing']),
                      'records_changed_after_audit': bool(reason)}}
    if reason:
        out['reason'] = reason
    return out


# --------------------------------------------------------------------------- tile bake (pipeline-bake-design 3.3)
PLACED = ('analysis/salt-pan-records.json', 'analysis/road-tile-records.json')
_DIMS = {}


def tile_dims(paths):
    """(n, 2) placement width/height per tile path: generic floors from the name, the rest from the vanilla tile
    database (audit_ground_seams.dims; 1x1 when a path has no database record)."""
    import re as _re
    from audit_ground_seams import Packs, GAME
    from native_tile_dimensions import decode_dimensions
    out = np.ones((len(paths), 2), np.int64)
    for i, p in enumerate(paths):
        if p not in _DIMS:
            m = _re.search(r'generic(?:_sea)?.(\d+)x(\d+)', p)
            if m:
                _DIMS[p] = (int(m.group(1)), int(m.group(2)))
            else:
                if 'packs' not in _DIMS:
                    _DIMS['packs'] = Packs([GAME / 'terrain.pack', GAME / 'fast.pack', GAME / 'data.pack'])
                parts = p.strip(B).split(B)
                key = 'terrain/tiles/campaign/_tile_database/tiles/' + parts[-2] + '_' + parts[-1] + '.bin'
                van = _DIMS['packs']
                _DIMS[p] = decode_dimensions(van.read(key)) if key in van.index else (1, 1)
        out[i] = _DIMS[p]
    return out


def footprints(paths, rec):
    """x, y, w, h (int64) of each record, quarter turns swapping w/h."""
    d = tile_dims(paths)[rec['path'].astype(np.int64)]
    turn = (rec['flags'] & QUARTER_TURN) > 0
    return (rec['x'].astype(np.int64), rec['y'].astype(np.int64),
            np.where(turn, d[:, 1], d[:, 0]), np.where(turn, d[:, 0], d[:, 1]))


def cover_counts(TH, TW, x, y, w, h):
    """Records covering each tile cell (2D prefix sums of the footprint corners)."""
    x0, x1 = np.clip(x, 0, TW), np.clip(x + w, 0, TW); y0, y1 = np.clip(y, 0, TH), np.clip(y + h, 0, TH)
    n = (TH + 1) * (TW + 1)
    plus = np.bincount(np.concatenate((y0 * (TW + 1) + x0, y1 * (TW + 1) + x1)), minlength=n).astype(np.int32)
    plus -= np.bincount(np.concatenate((y0 * (TW + 1) + x1, y1 * (TW + 1) + x0)), minlength=n).astype(np.int32)
    return plus.reshape(TH + 1, TW + 1).cumsum(0, dtype=np.int32).cumsum(1, dtype=np.int32)[:TH, :TW]


def tile_placed_first(ctx):
    """Occupancy order of the tile bake (pipeline-bake-design 3.3.1 item 2): the special pieces polish lays (salt
    pools, mountain and coast tiles in salt-pan-records.json, road pieces in road-tile-records.json) take their
    cells first; the floors fill the rest, and the obsolete-overlay cull (tile_bake.clear_overlays) never removes
    them. Count of those records missing from either tile list."""
    want = set()
    for rel in PLACED:
        p = ctx.ws / rel
        if p.is_file():
            ctx.path(rel)
            want |= {(r[0], int(r[1]), int(r[2])) for r in json.loads(p.read_text(encoding='utf-8'))}
    detail = {'placed_records': len(want)}; missing = 0
    for label, rel in (('local', TILES), ('global', GTILES)):
        paths, rec = ctx.tiles(rel)
        have = {(paths[int(a)], int(b), int(c)) for a, b, c in zip(rec['path'], rec['x'], rec['y'])}
        lost = want - have; detail[label] = len(lost); missing = max(missing, len(lost))
        if lost:
            detail[label + '_examples'] = [list(k) for k in sorted(lost)[:5]]
    return {'value': missing, 'detail': detail}


def tile_land_coverage(ctx):
    """Land tile cells (map.hex land hexes) with no land fine tile (generic_sea ignored), the seam scan's
    fine_uncovered_land, counted in tile cells; the tile bake's floors cover every land cell polish edited and
    keep vanilla's elsewhere."""
    import campaign_world as cw
    cells, names = ctx.map()
    H, W = cells.shape[:2]; TH, TW = 2 * H + 1, 2 * W
    r, q = tile_hex_grid(TH, TW, H, W)
    land = ((cells[:, :, 0] & 3) != 1)[r, q]
    paths, rec = ctx.tiles(TILES)
    k = ~np.array(['generic_sea' in p for p in paths], bool)[rec['path'].astype(np.int64)]
    x, y, w, h = footprints(paths, rec[k])
    bare = land & (cover_counts(TH, TW, x, y, w, h) == 0)
    nat = np.zeros((TH, TW), bool); nat[cw.NATIVE_TILE_SLICE] = True
    return {'value': int(bare.sum()), 'detail': {'native': int((bare & nat).sum()), 'canvas': int((bare & ~nat).sum()),
                                                'land_cells': int(land.sum())}}


NATIVE_RULE_INPUTS = ('analysis/polished-edit-alpha.npy', 'analysis/polished-tile-climate.npy')
NATIVE_BLOCK_COVERS = ('analysis/road-tile-cover.npy', 'analysis/coast-tile-cover.npy', 'analysis/floor-split-cells.npy',
                       'analysis/salt-pan-cover.npy', 'analysis/mountain-tile-cover.npy')


# --------------------------------------------------------------------------- snow bake (analysis/snow-bake-design.md 4)
SNOW_MASK = 'work/release-input/terrain/textures/campaign/default/snow/tw3k_exp_snowmask.dds'
SNOW_MATERIAL = 'work/release-input/materials/environment/campaign/tw3k_exp_campaign_snow.xml.material'
SNOW_JSON = 'analysis/snow-bake.json'
ENV_COLLECTION = PRE + 'environment_collection.xml'


def _snow(ctx):
    import campaign_world as cw
    if not cw.snow_bake_on():
        raise Missing('snow bake switched off (TW3K_SNOW_BAKE=1 turns it on)')
    def load():
        import snow_bake as sb
        rgba, mips = sb.read_mask(ctx.path(SNOW_MASK).read_bytes())
        return sb, rgba.astype(np.int16), mips
    return ctx.get('snow', load)


def snow_native_parity(ctx):
    """Native rectangle outside the edge ring and the peaks: winter snow (A) equal to vanilla's reprojected A within
    2 levels; R, G and B carry the peaks alone, so they are equal everywhere (vanilla's own R / G / B, never shown
    before snow ran in every season, are not copied). Value: pixels breaking either."""
    from scipy import ndimage
    sb, m, _ = _snow(ctx)
    van, _ = sb.vanilla_mask(); va = sb.reproject(van, m.shape[:2])[..., 3].astype(np.int16)
    nat = sb.native_box(m.shape[:2])
    core = ndimage.distance_transform_edt(nat) > sb.RING   # the blend reaches RING px inside, no deeper
    bad_a = core & (m[..., 1] == 0) & (np.abs(m[..., 3] - va) > 2)
    bad_rgb = (m[..., 0] != m[..., 1]) | (m[..., 1] != m[..., 2])
    return {'value': int(bad_a.sum() + bad_rgb.sum()),
            'detail': {'a_off_vanilla': int(bad_a.sum()), 'rgb_unequal': int(bad_rgb.sum()), 'core_px': int(core.sum())}}


def snow_peaks_all_channels(ctx):
    """Year-round snow is winter snow too: no pixel whose summer channel (G, the peaks) exceeds its winter channel
    (A); and the peak count is the bake's own (snow-bake.json peak_px, G >= 128). Value: pixels with G > A, plus 1
    if the count disagrees."""
    sb, m, _ = _snow(ctx)
    rep = json.loads(ctx.path(SNOW_JSON).read_text(encoding='utf-8'))
    over = int((m[..., 1] > m[..., 3]).sum()); peaks = int((m[..., 1] >= 128).sum())
    return {'value': over + (peaks != rep['peak_px']), 'detail': {'g_over_a': over, 'peak_px': peaks, 'json_peak_px': rep['peak_px']}}


def snow_frame(ctx):
    """The mask fits this canvas (size = snow_bake.mask_shape(), full mip chain) and was baked from the current
    map.hex, real terrain and rule code (snow-bake.json inputs). Value: the number of mismatches."""
    sb, m, mips = _snow(ctx)
    rep = json.loads(ctx.path(SNOW_JSON).read_text(encoding='utf-8'))
    ctx.path(MAP); ctx.path('analysis/west-real-terrain.npz'); ctx.path(LF)
    now = sb.fingerprint(ctx.ws)
    bad = [k for k in now if rep.get('inputs', {}).get(k) != now[k]]
    shape = list(m.shape[:2]) != list(sb.mask_shape())
    full = mips != int(np.log2(max(m.shape[:2]))) + 1
    return {'value': len(bad) + shape + full, 'detail': {'stale_inputs': bad, 'shape': list(m.shape[:2]), 'mips': mips}}


def snow_lighting(ctx):
    """environment_collection references only the snow bake's lighting groups, each present in the stage and equal
    to snow_bake.lighting_groups() (snow in every entry, spring rain kept); the material is snow_bake.material().
    A reference to a missing group would break the campaign start. Value: the number of failures.
    Switched off (TW3K_SNOW_BAKE unset): every reference is a vanilla lighting group and the stage carries none of
    the snow bake's products (the stage as before the bake)."""
    import campaign_world as cw, snow_bake as sb
    refs = set(re.findall(r"lighting='([^']*)'", ctx.path(ENV_COLLECTION).read_text(encoding='utf-8')))
    if not cw.snow_bake_on():
        S = ctx.ws / 'work/release-input'
        foreign = sorted(r for r in refs if not r.startswith(cw.VANILLA_LIGHTING_DIR + '/'))
        left = [p for p in (SNOW_MASK, SNOW_MATERIAL, 'work/release-input/' + cw.SNOW_LIGHTING_DIR) if (ctx.ws / p).exists()]
        return {'value': len(foreign) + len(left), 'detail': {'switch': 'off', 'refs': len(refs), 'foreign_refs': foreign, 'snow_products_left': left}}
    groups, _ = sb.lighting_groups(); pre = cw.SNOW_LIGHTING_DIR + '/'
    foreign = sorted(r for r in refs if not r.startswith(pre) or r[len(pre):] not in groups)
    differ = []
    for rel, text in groups.items():
        p = ctx.ws / 'work/release-input' / pre / rel
        if not p.is_file() or ctx.path(f'work/release-input/{pre}{rel}').read_bytes() != text.encode('utf-8'):   # bytes: CRLF as vanilla
            differ.append(rel)
    mat = ctx.path(SNOW_MATERIAL).read_bytes() != sb.material().encode('utf-8')
    return {'value': len(foreign) + len(differ) + mat, 'detail': {'refs': len(refs), 'foreign_refs': foreign, 'groups_differ': differ, 'material_differs': mat}}


def snow_mask_binary_diff(levels, core):
    """Bytes strictly between 0 and 232 outside the native core, summed over the mip levels (levels: RGBA uint8, top
    first; core: bool at the top level). The game shows snow only from 232 (170 with the best noise): such a byte is a
    shade the shader drops (analysis/snow-v22b-findings.md §3)."""
    from PIL import Image
    bad, per = 0, []
    for lv in levels:
        h, w = lv.shape[:2]
        c = core if core.shape == (h, w) else np.asarray(Image.fromarray(core.astype(np.float32), 'F').resize((w, h), Image.Resampling.BOX)) >= .5
        n = int(((lv > 0) & (lv < 232) & ~c[..., None]).sum())
        per.append(n); bad += n
    return bad, per


def snow_mask_binary(ctx):
    """Outside the native core (vanilla's own bytes, snow_native_parity) every mask byte on every mip level is 0 or
    >= 232: the shader is a switch (smoothstep(.80, .85) on the sRGB-decoded mask), so a shade between would show as
    no snow (v22b: dry plateau at 102, peak ramps). Value: such bytes."""
    sb, _, _ = _snow(ctx)
    levels = sb.read_levels(ctx.path(SNOW_MASK).read_bytes())
    bad, per = snow_mask_binary_diff(levels, sb.native_core(levels[0].shape[:2]))
    return {'value': bad, 'detail': {'by_level': per[:6], 'levels': len(levels)}}


def snow_peaks_visible_count(g, lat, lon, summits, exceptions, win, need):
    """Per named summit: G >= 232 pixels within win px of its nearest mask pixel. Returns (failures, counts)."""
    import snow_bake as sb
    counts, bad = {}, []
    for name, (r, c) in sb.summit_pixels(lat, lon, summits).items():
        n = int((g[max(r - win, 0):r + win + 1, max(c - win, 0):c + win + 1] >= 232).sum())
        counts[name] = n
        if n < need and name not in exceptions:
            bad.append(name)
    return bad, counts


def snow_peaks_visible(ctx):
    """Every named summit (perennial_summits + snow_bake.EXTRA_SUMMITS) shows a white top: >= PEAK_MIN_PX pixels of
    G >= 232 within PEAK_WIN px (31 x 31), but the PEAK_EXCEPTIONS (LF lake basins at the summit, a terrain item).
    Peaks follow the visible mountains (user 2026-11-01); v22b's gradient peaks left Bogda 1 px. Value: failing summits."""
    sb, m, _ = _snow(ctx)
    z = np.load(ctx.path('analysis/west-real-terrain.npz'))
    lat = sb.to_mask(z['lat'].astype(np.float32), m.shape[:2]); lon = sb.to_mask(z['lon'].astype(np.float32), m.shape[:2])
    bad, counts = snow_peaks_visible_count(m[..., 1], lat, lon, sb.named_summits(), sb.PEAK_EXCEPTIONS, sb.PEAK_WIN, sb.PEAK_MIN_PX)
    return {'value': len(bad), 'detail': {'failing': bad, 'counts': counts, 'exceptions': sorted(sb.PEAK_EXCEPTIONS)}}


def snow_on_water(ctx):
    """No snow on the canvas sea and lakes (mask pixels wholly water outside the native rectangle; vanilla's
    own mask over its sea is kept; shore pixels part land keep their land share). Value: such pixels with A > 0."""
    sb, m, _ = _snow(ctx)
    ctx.path(MAP)
    water = sb.to_mask(sb.water_tiles(ctx.ws).astype(np.float32), m.shape[:2]) > .999
    bad = water & ~sb.native_box(m.shape[:2]) & (m[..., 3] > 0)
    return {'value': int(bad.sum()), 'detail': {'water_px': int(water.sum())}}




def native_vanilla_records(ctx):
    """Vanilla special records (not generic floors, not seabed) on unedited native ground that the tile lists do not
    carry field for field (path, climate, x, y, flags): the tile bake's rule "unedited native ground = vanilla"
    (tile_bake.vanilla_specials). Unedited as place_floors reads it: wholly inside the native rectangle, no cell under
    polish's edit (alpha > .02 on changed LF), its road / coast / salt / mountain pieces or split floors, and the
    tile climate rule equal to the record's climate. Before the fix (run 20261030-123911): 330 per list, the
    Jiangyang / Yunnan mountain pieces the export lost. Records within clear_overlays' cull reach are left out too
    (tile_bake.cull_reach, the same CULL_APRON as its cull_window; a superset, the cull also needs changed ground):
    run 20261031-001956's 21 per list that this check missed before (19 sea_coast, 1 river_mouth, 1 cliff) all lay there."""
    import tile_bake as tb
    from read_tile_list import parse
    from repair_campaign_river_roots import Assets
    from native_edit_mask import edited_tiles
    ctx.path(LF)   # edited_tiles reads the stage LF
    alpha, climate = (np.load(ctx.path(rel)) for rel in NATIVE_RULE_INPUTS)
    edit = (alpha > .02) & edited_tiles(VARIANT, stage=ctx.ws / 'work/release-input', halo=0); del alpha
    blocked = tb.cull_reach(edit)
    for rel in NATIVE_BLOCK_COVERS:
        if (ctx.ws / rel).is_file():
            blocked |= np.load(ctx.path(rel))
    missing = {}; examples = []
    for rel in (TILES, GTILES):
        paths, _, _, _, rs, _ = parse(ctx.path(rel).read_bytes())
        have = {(paths[r[1]], r[2], r[3], r[4], r[5]) for r in rs}
        vp, _, _, _, vr, _ = parse(Assets(original_only=True).read(rel[len('work/release-input/'):]))
        ids = {(vp[v[1]], v[2], *tb._ntg.list_xy(v[3], v[4]), v[5]) for v in vr}
        # as place_floors: the listed records that are not vanilla (polish's river / coast repairs) block too
        own = tb.land_cover(paths, [r for r in rs if (paths[r[1]], r[2], r[3], r[4], r[5]) not in ids])
        out, _ = tb.vanilla_specials(vr, vp, vp, have, blocked | own, climate)   # the vanilla records the list lacks
        missing[rel] = len(out)
        examples += [[rel, vp[r[1]], r[2], r[3], r[4], r[5]] for r in out[:3]]
    return {'value': sum(missing.values()), 'detail': {'missing_by_list': missing, 'examples': examples}}

# Tile families whose global_map/tile_list.bin records keep the 0x700 flag bits, and those whose records drop them
# (rotation 0x?0 only); the local tile_list.bin keeps 0x7?0 on every record. Vanilla survey 2026-10-31, 3k_dlc07_main_map
# and 3k_main_map global lists: each family is all one way in both (mines / salt_pools only in 3k_main_map).
GLOBAL_KEEPS_700 = frozenset(('blockout_cliff', 'blockout_cliff_ends', 'canal', 'canal_links', 'mines', 'river', 'river_crossing',
                              'river_crossing_imperial', 'river_crossing_track', 'river_mouth', 'river_start', 'roads_imperial',
                              'roads_paved', 'roads_tracks', 'salt_pools'))
GLOBAL_DROPS_700 = frozenset(('generic', 'generic_sea', 'mountains_cold', 'mountains_subtropical', 'mountains_tea', 'mountains_temperate',
                              'mountains_terrace_farm', 'mountains_yellow', 'sea_coast'))

def tile_list_pair_diff(local, glob):
    """Mismatches between the two tile lists, each (paths, records) as read_tile_list.parse gives them: records of one
    list without a twin in the other by (path, climate, x, y, rotation, lo, hi), local records without 0x700, and
    global records whose 0x700 bits break the vanilla family convention (unknown families count too)."""
    import collections
    def fam(p):
        s = p.split(B)
        return s[3] if len(s) > 3 else p
    (pl, rl), (pg, rg) = local, glob
    key = lambda P, r: (P[r[1]], r[2], r[3], r[4], r[5] & 0xff, r[6], r[7])
    kl = collections.Counter(key(pl, r) for r in rl); kg = collections.Counter(key(pg, r) for r in rg)
    only_l, only_g = kl - kg, kg - kl
    local_bits = [r for r in rl if r[5] & 0xf00 != 0x700]
    global_bits = [r for r in rg if not ((fam(pg[r[1]]) in GLOBAL_KEEPS_700 and r[5] & 0xf00 == 0x700)
                                         or (fam(pg[r[1]]) in GLOBAL_DROPS_700 and r[5] & 0xf00 == 0))]
    detail = {'records': [len(rl), len(rg)], 'only_local': sum(only_l.values()), 'only_global': sum(only_g.values()),
              'local_without_700': len(local_bits), 'global_off_convention': len(global_bits),
              'examples': [list(k) for k in list(only_l)[:2] + list(only_g)[:2]]
                          + [[pg[r[1]], r[3], r[4], r[5]] for r in global_bits[:3]]}
    return detail['only_local'] + detail['only_global'] + detail['local_without_700'] + detail['global_off_convention'], detail

def tile_list_pair(ctx):
    """The two tile lists carry the same records field for field except the 0x700 flag bits (vanilla 3k_dlc07_main_map:
    195,490 records each, 181,930 differing in that one byte, lo/hi identical), so whichever the engine reads draws the
    same tiles. Every list writer writes both; this checks they did, after each step that writes them."""
    from read_tile_list import parse
    lists = []
    for rel in (TILES, GTILES):
        paths, _, _, _, rs, _ = parse(ctx.path(rel).read_bytes())
        lists.append((paths, rs))
    value, detail = tile_list_pair_diff(*lists)
    return {'value': value, 'detail': detail}


def _lf_mesh(ctx, step=.5, thr=.25):
    """|LF - land mesh| (world units) on canvas land over 0.5 wu cells, measured once per check (cached on ctx): the
    LF bilinear as the offline view and the seam scan sample it, the mesh as the highest land_mesh_* triangle
    (campaign_water_meshes.raster; holes = fine tiles, left out). Canvas split by map.hex region: playable land vs the
    rest (non-playable regions, sea); the vanilla interior (native rectangle) for reference. Also the land-mesh files,
    vertices and triangles (geometry budget). pipeline-bake-design 9.1 / 3.2.1."""
    def load():
        import campaign_world as cw, map_hex
        from campaign_water_meshes import _mesh, raster
        d = ctx.ws / MESHES
        files = sorted(d.glob('land_mesh_*.rigid_model_v2')) if d.is_dir() else []
        if not files:
            raise Missing(MESHES + 'land_mesh_*.rigid_model_v2')
        tris, nv, nt = [], 0, 0
        for f in files:
            typ, v, t, _ = _mesh(ctx.path(MESHES + f.name).read_bytes(), 'f4')
            nv += len(v); nt += len(t)
            if typ == 101:                       # land groups (sea / skirt-less parts are other types)
                v[:, 2] *= cw.ZS                 # raw frame z -> world z
                tris.append(v[t].astype(np.float32))
        tris = np.concatenate(tris)
        cells, names = ctx.map()
        ids = map_hex.region_ids(cells)
        pl = np.array([not n.startswith('3k_main_sea') and 'non_playable' not in n for n in names] + [False], bool)
        play_hex = pl[np.where((ids >= 0) & (ids < len(names)), ids, len(names))]
        lf = ctx.lf()                            # raw uint16, south-up
        h, w = lf.shape
        per = cw.LF_PX_PER_HEX
        nx, nz = int(cw.WORLD_WIDTH / step), int(cw.WORLD_HEIGHT / step)
        nat_x0, nat_z0 = cw.NATIVE_SHIFT_WORLD
        nat_x1, nat_z1 = nat_x0 + cw.VANILLA_WORLD[0], nat_z0 + cw.VANILLA_WORLD[1]
        lo, hi = tris[:, :, 2].min(1), tris[:, :, 2].max(1)
        diffs = {'canvas': [], 'canvas_playable': [], 'canvas_nonplayable': [], 'native': []}
        txyz, terr = _tree_xyz_safe(ctx)   # None without a usable tree list; sampled in the same bands (tree_mesh_offset)
        if txyz is not None:
            tmesh = np.full(len(txyz), np.nan)
            torder = np.argsort(txyz[:, 2]); tzs = txyz[torder, 2]
        band = 400
        bands = []   # the whole raster, kept for prop_mesh_offset / prop_scene_offset (one rasterisation per check)
        gx = (np.arange(nx) + .5) * step
        for j0 in range(0, nz, band):
            nb = min(band, nz - j0)
            z0 = j0 * step
            sel = (hi >= z0) & (lo <= z0 + nb * step)
            land = raster(tris[sel], 0., z0, step, nx, nb)
            bands.append(land)
            gz = z0 + (np.arange(nb) + .5) * step
            X, Z = np.meshgrid(gx, gz)
            u = np.clip(X / cw.DX * per, 0, w - 1); v_ = np.clip(Z / cw.DZ * per, 0, h - 1)
            u0, v0 = np.minimum(u.astype(np.intp), w - 2), np.minimum(v_.astype(np.intp), h - 2)
            fu, fv = u - u0, v_ - v0
            raw = ((lf[v0, u0] * (1 - fu) + lf[v0, u0 + 1] * fu) * (1 - fv) + (lf[v0 + 1, u0] * (1 - fu) + lf[v0 + 1, u0 + 1] * fu) * fv)
            lfh = cw.LEVEL + (raw / 65535 - cw.NORM) * cw.GAIN
            ok = np.isfinite(land)
            nat = (X >= nat_x0) & (X < nat_x1) & (Z >= nat_z0) & (Z < nat_z1)
            q, r = cw.world_hex(X, Z)
            ph = play_hex[np.clip(np.asarray(r), 0, play_hex.shape[0] - 1), np.clip(np.asarray(q), 0, play_hex.shape[1] - 1)]
            dd = np.abs(lfh - land).astype(np.float32)
            can = ok & ~nat
            diffs['canvas'].append(dd[can]); diffs['canvas_playable'].append(dd[can & ph])
            diffs['canvas_nonplayable'].append(dd[can & ~ph]); diffs['native'].append(dd[ok & nat])
            if txyz is not None:
                idx = torder[np.searchsorted(tzs, z0):np.searchsorted(tzs, z0 + nb * step)]
                tmesh[idx] = _bilinear(land, txyz[idx, 0] / step - .5, (txyz[idx, 2] - z0) / step - .5)
        out = {}
        for k, v in diffs.items():
            a = np.concatenate(v)
            out[k] = {'cells': int(len(a)), 'p95': round(float(np.percentile(a, 95)), 3) if len(a) else None,
                      'median': round(float(np.median(a)), 3) if len(a) else None,
                      'share_over_25': round(float((a > thr).mean()), 4) if len(a) else None}
        out['meshes'] = {'files': len(files), 'vertices': nv, 'triangles': nt}
        ctx._c['land_raster'] = (step, np.concatenate(bands))   # (nz, nx) float32, -inf = no land triangle
        if terr:
            out['trees_error'] = terr
        if txyz is not None:
            q, r = cw.world_hex(txyz[:, 0], txyz[:, 2])
            tph = play_hex[np.clip(np.asarray(r), 0, play_hex.shape[0] - 1), np.clip(np.asarray(q), 0, play_hex.shape[1] - 1)]
            d = txyz[:, 1] - tmesh
            ctx._c['tree_mesh'] = {'xyz': txyz, 'mesh': tmesh, 'playable': tph}   # per tree, for art_review heatmaps
            ok = np.isfinite(d)
            over = ok & (np.abs(d) > thr)

            def cnt(m):
                n = int((m & ok).sum())
                return {'trees': n, 'over': int((m & over).sum()), 'share': round(float((m & over).sum() / max(n, 1)), 4),
                        'p95': round(float(np.percentile(np.abs(d[m & ok]), 95)), 3) if n else None}
            # native rectangle (vanilla rows: vanilla's own coarse mesh sits ~.3 above its LF forests, an old offset, not a
            # regression) vs canvas, each split by playable; the value stays the playable share over both
            tn = (txyz[:, 0] >= nat_x0) & (txyz[:, 0] < nat_x1) & (txyz[:, 2] >= nat_z0) & (txyz[:, 2] < nat_z1)
            out['trees'] = {'threshold': thr, 'all': cnt(np.ones(len(d), bool)), 'playable': cnt(tph), 'other': cnt(~tph),
                            'native_playable': cnt(tn & tph), 'canvas_playable': cnt(~tn & tph),
                            'native': cnt(tn), 'canvas': cnt(~tn),
                            'no_mesh_under': int((~ok).sum()),
                            'floating': int((over & (d > 0)).sum()), 'buried': int((over & (d < 0)).sum())}
        return out
    return ctx.get('lf_mesh', load)


def _tree_xyz(ctx):
    """Every stage tree-list row's world (x, height, z) (the layout flat_arid_rocks reads; canvas frame, no shift), or
    None without a tree list (lf_mesh_diff still runs)."""
    try:
        b = ctx.path(TREES).read_bytes()
    except Missing:
        return None
    pos = 24; xyz = []
    for _ in range(struct.unpack_from('<I', b, 20)[0]):
        n = struct.unpack_from('<H', b, pos)[0]; pos += 2 + n
        cnt = struct.unpack_from('<I', b, pos)[0]; pos += 4
        for _ in range(cnt):
            xyz.append(struct.unpack_from('<3f', b, pos))
            pos += 18 + struct.unpack_from('<I', b, pos + 14)[0] * 4
    if pos != len(b):
        raise ValueError('tree list layout')
    return np.array(xyz, np.float64).reshape(-1, 3)


def _tree_xyz_safe(ctx):
    """(_tree_xyz or None, reason): a tree list it cannot parse must not take the mesh gates of _lf_mesh down with it."""
    try:
        return _tree_xyz(ctx), None
    except (ValueError, struct.error) as e:
        return None, f'{type(e).__name__}: {e}'


def _bilinear(a, fx, fy):
    """a sampled at fractional cell (fx, fy); a NaN neighbour (mesh hole) falls back to the nearest cell."""
    h, w = a.shape
    x0 = np.clip(np.floor(fx).astype(np.intp), 0, w - 2); y0 = np.clip(np.floor(fy).astype(np.intp), 0, h - 2)
    tx = np.clip(fx - x0, 0, 1); ty = np.clip(fy - y0, 0, 1)
    with np.errstate(invalid='ignore'):   # NaN (hole) * 0 weight
        v = (a[y0, x0] * (1 - tx) + a[y0, x0 + 1] * tx) * (1 - ty) + (a[y0 + 1, x0] * (1 - tx) + a[y0 + 1, x0 + 1] * tx) * ty
    near = a[np.clip(np.rint(fy).astype(np.intp), 0, h - 1), np.clip(np.rint(fx).astype(np.intp), 0, w - 1)]
    return np.where(np.isfinite(v), v, near)


def lf_mesh_diff(ctx):
    """p95 of |LF - land mesh| on playable canvas land (_lf_mesh): <= .12, the vanilla interior's p95 (mesh bake 3.2.1:
    the error-bounded lattice, mesh_lattice.TOL_PLAYABLE). Checked after the graph (gap-fill) and seam_pass0; right after
    polish the canvas is still polish's budget lattice (p95 .88 on 10-30), which gap-fill replaces."""
    out = _lf_mesh(ctx)
    return {'value': out['canvas_playable']['p95'], 'detail': out}


def lf_mesh_diff_nonplayable(ctx):
    """p95 of |LF - land mesh| on canvas land outside the playable regions (_lf_mesh): <= .25 (mesh_lattice
    TOL_NONPLAYABLE, the geometry budget's coarser tolerance off the playable map)."""
    out = _lf_mesh(ctx)
    return {'value': out['canvas_nonplayable']['p95'], 'detail': {k: out[k] for k in ('canvas_nonplayable', 'meshes')}}


def land_mesh_budget(ctx):
    """Land-mesh triangles, millions (_lf_mesh): <= 10 (1.5 x main's 6.64M on 2026-10-30; vanilla 2.70M), the
    geometry budget of the mesh bake (3.2.1)."""
    m = _lf_mesh(ctx)['meshes']
    return {'value': round(m['triangles'] / 1e6, 3), 'detail': m}



def tree_mesh_offset(ctx):
    """Share of playable-region trees whose height is more than .25 wu off the land mesh under them (floating or
    buried; _lf_mesh samples the mesh raster at every tree-list row, bilinear over its 0.5 wu cells; trees on a fine
    tile with no mesh under it are left out and counted). Checked after the graph (the tree list is final there) and at
    mesh_final (the seam repair and height normalisation move the meshes after the graph: the meshes as packed).
    Warn only, 1% line (art-review gate section 3, coordinator 2026-10-30; v19 4.9%), and warns when worse than the
    last run; hard once the line is calibrated (the coordinator's call)."""
    out = _lf_mesh(ctx)
    if 'trees_error' in out:
        raise ValueError(f'tree list unreadable ({out["trees_error"]}); the mesh checks still ran')
    if 'trees' not in out:
        raise Missing(TREES)
    return {'value': out['trees']['playable']['share'], 'detail': out['trees']}

PROPS = PRE + 'global_props.bin'   # the prop bundle (prop_instances reads it: world-space leaves)
PROP_FAMILIES = ('mountains', 'rocks')   # prop_mesh_offset: single large props (town scenes are prop_scene_offset)
PROP_FLOAT = .25     # wu a prop's base may sit over the ground (the tree / LF-mesh line; vanilla interior float p95 .38)
PROP_BURIED = -1.    # wu: buried deeper is counted as information (vanilla buries props by design: median -.63)
SCENE_LIMIT = 1.     # wu: a scene whose median |y - ground| exceeds this is off its ground (rigid shift would fix it)
SCENE_SPREAD = 1.    # wu: p95 of |y - ground - median| within a scene above this: on a slope, no rigid shift fits (info)
SCENE_MIN_PIECES = 10   # the second (reconciliation) count: canvas groups with >= 10 settlement pieces


def prop_offset_stats(fam, mode, y, base, ground, playable, native, thr=PROP_FLOAT, buried=PROP_BURIED):
    """Counts of prop_mesh_offset from per-instance arrays (pure; tests). Counted: BHM_IGNORE (absolute y) instances
    of PROP_FAMILIES, not parked (y > prop_instances.PARKED_Y), with ground under them (finite). d = base - ground,
    base = y + the model's bounding-box floor x its y scale (y where the model is unknown). value = share of the
    playable ones with d > thr; buried (d < buried) is information only. Returns (value, detail, d)."""
    import prop_instances as pi
    fam, mode = np.asarray(fam), np.asarray(mode)
    y, base, ground = (np.asarray(a, np.float64) for a in (y, base, ground))
    playable, native = np.asarray(playable, bool), np.asarray(native, bool)
    cand = np.isin(fam, PROP_FAMILIES) & (mode == 'BHM_IGNORE') & (y > pi.PARKED_Y)
    ok = cand & np.isfinite(ground)
    d = np.where(ok, base - ground, np.nan)

    def cnt(m):
        m = m & ok; n = int(m.sum()); f = int((m & (d > thr)).sum())
        return {'props': n, 'floating': f, 'share': round(f / max(n, 1), 4), 'buried_below': int((m & (d < buried)).sum()),
                'float_p95': round(float(np.percentile(d[m], 95)), 3) if n else None}
    out = {'threshold': thr, 'buried_line': buried, 'playable': cnt(playable), 'other': cnt(~playable),
           'native_interior': cnt(native), 'canvas_playable': cnt(playable & ~native), 'no_ground_under': int((cand & ~ok).sum())}
    return out['playable']['share'], out, d


def _scene_groups(keys, d):
    """{key: offsets array} of the instances with a key (None = not a scene piece)."""
    g = {}
    for k, v in zip(keys, d):
        if k is not None:
            g.setdefault(k, []).append(v)
    return {k: np.array(v) for k, v in g.items()}


def scene_offsets(fam, mode, y, ground, region, prefab, native, limit=SCENE_LIMIT, spread=SCENE_SPREAD, min_pieces=SCENE_MIN_PIECES):
    """prop_scene_offset from per-instance arrays (pure; tests).

    The gated count (one entry per placed resource scene): scenes = bmd_objects.<region>.<num> groups whose num is one
    of the templates complete_campaign_resource_scenes places (prop_instances.RESOURCE_TEMPLATES), anywhere on the map,
    any number of pieces; pieces = prop_instances.scene_piece_mask (BHM_IGNORE, not parked, no vegetation / decals)
    with ground under them; offset = median(y - ground) (the writer seats a scene rigidly, one height for all its
    pieces). value = scenes with |offset| > limit. Info: spread = p95 of |y - ground - offset| within a scene, and the
    scenes whose spread exceeds `spread` (on a slope: a rigid shift cannot seat them).
    The reconciliation count (the art-review prop diff of 2026-10-31): canvas groups (outside the native rectangle)
    of any prefab with >= min_pieces settlement pieces, the same median rule.
    Returns (value, detail)."""
    import prop_instances as pi
    fam, mode, region, prefab = np.asarray(fam), np.asarray(mode), np.asarray(region), np.asarray(prefab)
    y, ground, native = np.asarray(y, np.float64), np.asarray(ground, np.float64), np.asarray(native, bool)
    d = y - ground
    piece = pi.scene_piece_mask(fam, mode, y, prefab) & np.isfinite(ground)
    g = _scene_groups([f'{region[i]}.{int(prefab[i])}' if piece[i] else None for i in range(len(y))], d)
    med = {k: float(np.median(v)) for k, v in g.items()}
    spr = {k: float(np.percentile(np.abs(v - med[k]), 95)) for k, v in g.items()}
    off = {k: round(med[k], 3) for k in sorted(med, key=lambda k: -abs(med[k])) if abs(med[k]) > limit}
    sloped = {k: round(spr[k], 3) for k in sorted(spr, key=lambda k: -spr[k]) if spr[k] > spread}
    alt = (fam == 'settlements') & (mode == 'BHM_IGNORE') & (y > pi.PARKED_Y) & np.isfinite(ground) & ~native
    g2 = _scene_groups([f'{region[i]}.{int(prefab[i])}' if alt[i] else None for i in range(len(y))], d)
    med2 = {k: float(np.median(v)) for k, v in g2.items() if len(v) >= min_pieces}
    det = {'scenes': len(med), 'off': off, 'limit': limit, 'templates': list(pi.RESOURCE_TEMPLATES),
           'spread_line': spread, 'spread_over': len(sloped), 'spread': sloped,
           'canvas_settlement_groups': {'min_pieces': min_pieces, 'scenes': len(med2),
                                        'off': sum(1 for v in med2.values() if abs(v) > limit)}}
    return len(off), det


def _model_floors(models):
    """{model: bounding-box floor (LOD0 lo y)} from the vanilla model packs (render_live_objects.Models); NaN when the
    model is not found. Only the PROP_FAMILIES models are asked for (a few dozen)."""
    import campaign_ground_textures as cgt, render_live_objects as lo
    packs = cgt.VanillaPacks(); lo.add_model_packs(packs); M = lo.Models(packs)
    out = {}
    for m in models:
        bb = M.bbox(m)
        out[m] = float(bb[0][1]) if bb is not None else np.nan
    return out


def _prop_ground(ctx, ref):
    """Per-instance arrays of the stage props against a ground (cached on ctx per ref): ref 'mesh' = the land-mesh
    raster of _lf_mesh (bilinear over its 0.5 wu cells, as tree_mesh_offset; mesh holes = fine tiles have no ground),
    ref 'lf' = the final LF, bilinear as _lf_mesh samples it (right after the resource scenes the graph's gap-fill is
    still rebuilding the meshes; the LF is final after sync_final)."""
    def load():
        import campaign_world as cw, map_hex, prop_instances as pi
        inst = pi.instances(ctx.path(PROPS).read_bytes())
        if not inst:
            raise Missing(PROPS + ' (no leaf instances)')
        x = np.array([i.x for i in inst]); y = np.array([i.y for i in inst]); z = np.array([i.z for i in inst])
        fam = np.array([pi.family(i.model) for i in inst]); mode = np.array([i.mode for i in inst])
        if ref == 'mesh':
            _lf_mesh(ctx)
            step, land = ctx._c['land_raster']
            a = np.where(np.isfinite(land), land, np.nan)
            ground = _bilinear(a, x / step - .5, z / step - .5)
        else:
            lf = ctx.lf(); h, w = lf.shape; per = cw.LF_PX_PER_HEX
            u = np.clip(x / cw.DX * per, 0, w - 1); v = np.clip(z / cw.DZ * per, 0, h - 1)
            u0, v0 = np.minimum(u.astype(np.intp), w - 2), np.minimum(v.astype(np.intp), h - 2); fu, fv = u - u0, v - v0
            raw = ((lf[v0, u0] * (1 - fu) + lf[v0, u0 + 1] * fu) * (1 - fv) + (lf[v0 + 1, u0] * (1 - fu) + lf[v0 + 1, u0 + 1] * fu) * fv)
            ground = cw.LEVEL + (raw / 65535 - cw.NORM) * cw.GAIN
        sel = np.isin(fam, PROP_FAMILIES)
        floors = _model_floors(sorted({i.model for i, s in zip(inst, sel) if s}))
        ylo = np.array([floors.get(i.model, np.nan) if s else np.nan for i, s in zip(inst, sel)])
        base = y + np.where(np.isfinite(ylo), ylo, 0.) * np.array([i.scale_y for i in inst])
        cells, names = ctx.map()
        ids = map_hex.region_ids(cells)
        pl = np.array([not n.startswith('3k_main_sea') and 'non_playable' not in n for n in names] + [False], bool)
        play_hex = pl[np.where((ids >= 0) & (ids < len(names)), ids, len(names))]
        q, r = cw.world_hex(x, z)
        playable = play_hex[np.clip(np.asarray(r), 0, play_hex.shape[0] - 1), np.clip(np.asarray(q), 0, play_hex.shape[1] - 1)]
        nx0, nz0 = cw.NATIVE_SHIFT_WORLD
        native = (x >= nx0) & (x < nx0 + cw.VANILLA_WORLD[0]) & (z >= nz0) & (z < nz0 + cw.VANILLA_WORLD[1])
        return dict(ref=ref, x=x, y=y, z=z, fam=fam, mode=mode, base=base, ground=ground, playable=playable, native=native,
                    region=np.array([i.region for i in inst]), prefab=np.array([i.prefab for i in inst]),
                    model=np.array([i.model for i in inst]), no_floor=sorted(k for k, v in floors.items() if not np.isfinite(v)))
    return ctx.get('prop_ground_' + ref, load)


def _prop_offset(ctx, ref):
    p = _prop_ground(ctx, ref)
    value, det, d = prop_offset_stats(p['fam'], p['mode'], p['y'], p['base'], p['ground'], p['playable'], p['native'])
    worst = np.argsort(-np.nan_to_num(d, nan=-np.inf))[:10]
    det['worst_floating'] = [{'model': p['model'][i].rsplit('/', 1)[-1], 'x': round(float(p['x'][i]), 2), 'z': round(float(p['z'][i]), 2),
                              'offset': round(float(d[i]), 2), 'scene': f'{p["region"][i]}.{int(p["prefab"][i])}'}
                             for i in worst if np.isfinite(d[i])]
    det['ground'] = ref; det['models_without_bbox'] = p['no_floor']
    return {'value': value, 'detail': det}


def _scene_offset(ctx, ref):
    p = _prop_ground(ctx, ref)
    n, det = scene_offsets(p['fam'], p['mode'], p['y'], p['ground'], p['region'], p['prefab'], p['native'])
    det['ground'] = ref
    return {'value': n, 'detail': det}


def prop_mesh_offset(ctx):
    """Share of playable-region mountain / rock props (BHM_IGNORE, not parked, land mesh under them) whose base sits
    more than .25 wu over the land mesh as packed (prop_offset_stats, _prop_ground 'mesh'). Buried below -1 wu is
    reported, not gated (vanilla buries props). Warn only, line 5.9% = the vanilla dlc07 native interior on the v20 / v21
    packs (5.87%); v20 11.3%, v21 10.6% (canvas playable 29.9% / 27.0%: the template-2412 rocks, pipeline-bake-design
    9.4). Hard once the resource scene placement is fixed (the coordinator's call)."""
    return _prop_offset(ctx, 'mesh')


def prop_lf_offset(ctx):
    """prop_mesh_offset against the final LF (_prop_ground 'lf'), at resource_scenes: after complete_campaign_resource_scenes
    and world-space-leaves (leaves in world space), before the meshes are final. Its own line, 9.4% = the vanilla native interior against the LF on the v20 / v21 packs
    (the LF is smoother than the mesh under vanilla mountain pieces, and fine-tile holes have LF ground); v20 13.9%,
    v21 13.2%. Warn only."""
    return _prop_offset(ctx, 'lf')


def prop_scene_offset(ctx):
    """Placed resource scenes whose median y - land mesh exceeds 1 wu either way (scene_offsets, _prop_ground 'mesh').
    Definition (one entry per placed scene): every bmd_objects.<region>.<num> group whose num is one of the templates
    complete_campaign_resource_scenes places (prop_instances.RESOURCE_TEMPLATES), anywhere on the map; its pieces are
    prop_instances.scene_piece_mask with mesh under them. A baseline that only goes down: 30 of 112 on the v20 and v21
    packs. Lower the threshold when a fix brings scenes back on their ground, never raise it; a rise against the last
    run warns (metric, tolerance 0). Info in the detail: the in-scene spread (p95 |y - mesh - median| > 1: scenes on
    slopes no rigid shift seats; 24 / 22) and the art-review prop diff's count (canvas groups with >= 10 settlement
    pieces: 20 of 67)."""
    return _scene_offset(ctx, 'mesh')


def prop_scene_lf_offset(ctx):
    """prop_scene_offset against the final LF at resource_scenes (the writer's own check: the scenes as
    complete_campaign_resource_scenes seated them, read once world-space-leaves has baked the root translation in). Baseline 33 of 112 on the v20 and v21 packs (the LF counts
    pieces over fine-tile holes too), only goes down; hard."""
    return _scene_offset(ctx, 'lf')


# name: (function, reads, steps, kind, threshold, tolerance, warn_only, owner)
def river_channels_match(ctx):
    """river-channels.json against map.hex (river spec 6.2 item 4): river-region hexes without a channel, channel hexes
    that are not river water, and drawn-only hexes (option B) that are not land: 0. A river laid as water must be in the
    channels polish carves by; a drawn hex must stay its region's land."""
    cells, names = ctx.map()
    H, W = cells.shape[:2]
    doc = json.loads(ctx.path(CHANNELS).read_text(encoding='utf-8'))
    ids = (cells[:, :, 1].astype(np.int64) << 5) + (cells[:, :, 0].astype(np.int64) >> 3) - 1
    isriv = np.array([n.startswith('tw3k_riv_sea_') for n in names] + [False])
    land = (cells[:, :, 0] & 3) != 1
    river = ~land & isriv[np.where((ids >= 0) & (ids < len(names)), ids, len(names))]

    def mask(d):
        m = np.zeros((H, W), bool)
        for pts in d.values():
            a = np.asarray(pts, np.int64).reshape(-1, 2)
            ok = (a[:, 0] >= 0) & (a[:, 0] < W) & (a[:, 1] >= 0) & (a[:, 1] < H)
            m[a[ok, 1], a[ok, 0]] = True
        return m
    ch, dr = mask(doc['channels']), mask(doc.get('drawn', {}))
    det = {'river_without_channel': int((river & ~ch).sum()), 'channel_not_river': int((ch & ~river).sum()),
           'drawn_not_land': int((dr & ~land).sum()), 'drawn': int(dr.sum()), 'river_hexes': int(river.sum())}
    return {'value': det['river_without_channel'] + det['channel_not_river'] + det['drawn_not_land'], 'detail': det}


def port_stamp_shape(ctx):
    """Port towns whose stamp is not vanilla's (add_ports docstring): a land region with land port hexes (slot 2)
    must have exactly one, beside exactly one water port hex W1 (slot 2) that has exactly one more W2, both in one
    water region that is not a lake, with 4 (resource) or 16 (city) primary hexes. River ports (river spec 6.7) are
    stamped after the rivers; add_ports checked this only before it wrote."""
    def nb(q, r):   # add_ports.neighbours (odd-q); add_ports itself loads the workspace design on import
        for dq, dr in ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1 if q % 2 else -1), (-1, 1 if q % 2 else -1)):
            if 0 <= q + dq < W and 0 <= r + dr < H:
                yield q + dq, r + dr
    cells, names = ctx.map()
    H, W = cells.shape[:2]
    ids = (cells[:, :, 1].astype(np.int64) << 5) + (cells[:, :, 0].astype(np.int64) >> 3) - 1
    water = (cells[:, :, 0] & 3) == 1
    slot = cells[:, :, 2] >> 4
    bad = []
    for r, q in zip(*np.nonzero((slot == 2) & ~water)):
        me = int(ids[r, q]); name = names[me]
        lp = int(((ids == me) & (slot == 2) & ~water).sum()); prim = int(((ids == me) & (slot == 1)).sum())
        w1 = [(a, b) for a, b in nb(int(q), int(r)) if water[b, a] and slot[b, a] == 2]
        w2 = [(a, b) for a, b in nb(*w1[0]) if water[b, a] and slot[b, a] == 2] if len(w1) == 1 else []
        wr = {names[ids[b, a]] for a, b in w1 + w2}
        if lp != 1 or prim not in (4, 16) or len(w1) != 1 or len(w2) != 1 or len(wr) != 1 or any('lake' in n for n in wr):
            bad.append([name, [int(q), int(r)], lp, prim, len(w1), len(w2), sorted(wr)])
    return {'value': len(bad), 'detail': {'ports': int(((slot == 2) & ~water).sum()), 'bad': bad[:20]}}



def vanilla_playable_climate(ctx):
    """Vanilla playable hexes (revealed_filler.vanilla_playable_hex) whose map.hex climate differs from vanilla's (by
    climate name): vanilla playable ground stays identical. Real-geography rules (korea_patch forest zones,
    TW3K_KOREA_FORESTS, §9.3) may change the climate of the non-playable filler and the canvas only. v21: 0 of 222,230."""
    import map_hex, campaign_world as cw, revealed_filler as rf
    m = map_hex.read(ctx.path(MAP)); v = map_hex.read(cw.VANILLA_MAP)
    name = lambda mm: np.array(mm['climates'] + ['?'] * 256)[(mm['cells'][..., 7] >> 1).astype(int) - 1]
    play = rf.vanilla_playable_hex()[cw.NATIVE_HEX_SLICE]
    d = play & (name(m)[cw.NATIVE_HEX_SLICE] != name(v))
    rr, qq = np.nonzero(d)
    return {'value': int(d.sum()), 'detail': {'playable': int(play.sum()),
                                              'first': [(int(q) + cw.PAD_LEFT, int(r) + cw.PAD_BOTTOM) for r, q in zip(rr[:20], qq[:20])]}}

def region_connectivity(ctx):
    """Land regions split into non-island pieces (region_connectivity.py: hex adjacency of map_hex.BORDER_SIDES;
    a stray piece is an island only when no other region's land and no river hex ('_riv_' sea regions) touch it;
    allowance: vanilla dlc07 and the v16 list region_connectivity_known.json). v17: the Western Regions rivers
    cut 43 more regions and the game crashed at load in the region-border code."""
    import region_connectivity as rc
    cells, names = ctx.map()
    errors = rc.region_connectivity_errors(cells, names, rc.default_vanilla(), rc.load_known())
    for line in rc.describe(errors, 60):
        print('[invariant]   split region', line, flush=True)
    return {'value': len(errors), 'detail': {'regions': [{k: (v[:20] if isinstance(v, list) else v) for k, v in e.items()} for e in errors]}}


def native_y0_trees_grounded(ctx):
    """Stage tree rows that are vanilla native rows stored at y == 0 (same name and float32 xz after NATIVE_SHIFT)
    standing more than tree_bake.NATIVE_Y0_TOL off the final LF (sample_final_surface at x / DX * 2, z / DZ * 2: the
    ground procedural rows use). polish_campaign_foliage puts them on that ground (tree_bake.native_row_heights);
    the delta rule left six of them buried on the native south edge (v20: -0.85 to -2.72). The vanilla list is
    the original pack's (ctx key 'vanilla_trees')."""
    import campaign_world as cw, tree_bake as tb
    from final_terrain_surface import sample_final_surface

    def rows(b):
        pos = 24; out = []
        for _ in range(struct.unpack_from('<I', b, 20)[0]):
            n = struct.unpack_from('<H', b, pos)[0]; name = b[pos + 2:pos + 2 + n].decode(); pos += 2 + n
            cnt = struct.unpack_from('<I', b, pos)[0]; pos += 4
            for _ in range(cnt):
                out.append((name,) + struct.unpack_from('<3f', b, pos)); pos += 18 + struct.unpack_from('<I', b, pos + 14)[0] * 4
        if pos != len(b):
            raise ValueError('tree list layout')
        return out

    def vanilla():
        from repair_campaign_river_roots import Assets
        return Assets(original_only=True).read(f'campaign_maps/{VARIANT}/display/trees/trees.campaign_tree_list')
    SX, SZ = cw.NATIVE_SHIFT_WORLD
    want = set()
    for name, x, y, z in rows(ctx.get('vanilla_trees', vanilla)):
        if y == 0:
            xs, zs = np.float64(x) + SX, np.float64(z) + SZ
            if tb.native_many(xs, zs):
                want.add((name, float(np.float32(xs)), float(np.float32(zs))))
    hit = [(n, x, y, z) for n, x, y, z in rows(ctx.path(TREES).read_bytes()) if (n, x, z) in want]
    if not hit:
        return {'value': 0, 'detail': {'vanilla_y0_native': len(want), 'stage_rows': 0}}
    p = np.array([h[1:] for h in hit], np.float64)
    off = p[:, 1] - sample_final_surface(ctx.lf(), p[:, 0] / cw.DX * 2, p[:, 2] / cw.DZ * 2)
    bad = np.flatnonzero(np.abs(off) > tb.NATIVE_Y0_TOL)
    return {'value': int(len(bad)), 'detail': {'vanilla_y0_native': len(want), 'stage_rows': len(hit), 'max_abs': round(float(np.abs(off).max()), 4),
                                               'off': [{'name': hit[i][0], 'x': round(hit[i][1], 2), 'z': round(hit[i][3], 2), 'tree_minus_lf': round(float(off[i]), 3)} for i in bad[:20]]}}


INVARIANTS = {
    'river_channels_match': (river_channels_match, (MAP, CHANNELS), ('map',), 'exact', 0, None, False,
                             'apply_rivers.lay_rivers / bake_map step 7 (river spec 6.2: water hexes in channels, drawn hexes land)'),
    'port_stamp_shape': (port_stamp_shape, (MAP,), ('map',), 'exact', 0, None, False,
                         'add_ports / bake_map 3 stamps and 7.5 river ports (river spec 6.7): vanilla port stamps'),
    'vanilla_playable_climate': (vanilla_playable_climate, (MAP,), ('map',), 'exact', 0, None, False,
                                 'map stage climate (apply_climate, hehuang/korea patches: korea_patch PROTECT_HEXES); v21 0'),
    'region_connectivity': (region_connectivity, (MAP,), ('map',), 'exact', 0, None, False,
                            'the map stage (map.hex writers call region_connectivity.check_map after writing; 东西南北)'),
    'sea_floor_coverage': (sea_floor_coverage, (MAP, TILES, GTILES), ('sync_final', 'graph'), 'exact', VANILLA_SEA_HOLES, None, False,
                           'tile_bake.place_floors / clear_overlays sea floors (fix/xianglin-seafloor, sea_floor_tiles.py)'),
    'river_centre_wet': (river_centre_wet, (MAP, LF, LEVEL), ('polish', 'sync_final'), 'metric', None, 0, False,
                         'polish RIVER CHANNELS / town aprons (e48a5d7f); sync_final boundary smoothing must not dry a centre'),
    'desert_banned_share': (desert_banned_share, (MAP, BLEND, REAL), ('polish', 'sync_final'), 'metric', .002, .002, True,
                            'ground_materials step 6 banned slots (biome_profiles AUDIT); gate: audit_terrain_realism --gate'),
    'tone_contrast_edges': (tone_contrast_edges, (MAP, BLEND), ('polish', 'sync_final'), 'metric', None, 0, True,
                            'ground blend writers (polish mirror/donor tone bridge, ground_materials, wetland, farms); ground contrast fix §1'),
    'woodland_off_forest': (woodland_off_forest, (MAP, BLEND, LANDFORM_TILES), ('polish', 'sync_final'), 'metric', None, 0, True,
                            'polish DONOR WOODLAND (forest density); ground contrast fix §3'),
    'tundra_temperate_share': (tundra_temperate_share, (MAP, BLEND, LANDFORM_TILES), ('polish', 'sync_final'), 'metric', .05, .01, False,
                               'ground_materials relief (biome_profiles LANDFORMS tundra relief, §9.3); sixteenth pack .034'),
    'tundra_forest_straight': (tundra_forest_straight, (MAP, LANDFORM_TILES), ('polish',), 'metric', .15, .03, False,
                               'polish MATERIALS ECOTONE (ecotone_bands.tundra_forest, biome_profiles TUNDRA_FOREST, §9.3); main .251, new .096'),
    'native_edge_tone_cut': (native_edge_tone_cut, (EDGE_AUDIT, BLEND), ('polish',), 'exact', 0, None, False,
                             'native-edge ground writers (polish mirror/donor band, ground_bake edge continuation); ground contrast fix §10'),
    'wetland_in_town_clear': (wetland_in_town_clear, (MAP, BLEND), ('polish', 'sync_final'), 'exact', 0, None, False,
                              'apply_climate.paint_wetland_blend (WET PIXEL MASK spill excludes town_clear_hex); town spec'),
    'blend_single_writer': (blend_single_writer, ('analysis/blend-writers.json',), ('polish',), 'metric', None, 0, True,
                            'polish blend writers after the mirror/donor blend (ground contrast fix §1 end state: one selection point)'),
    'hex_squareness': (hex_squareness, (MAP, BLEND), ('polish', 'sync_final'), 'metric', None, 0, True,
                       'apply_climate.paint_wetland_blend (WET PIXEL MASK), farm_fields.paint (FARM PIXEL MASK); ground contrast fix §4'),
    'town_plateau_lift': (town_plateau_lift, (POLISH_REPORT,), ('polish',), 'exact', .2, None, False,
                          'polish cluster levels (lake cap continuity) and the projection plot coupling; town spec'),
    'lake_shore_dry': (lake_shore_dry, (MAP, LF), ('polish', 'sync_final'), 'exact', 0, None, False,
                       'polish slope projection lower bound (height bake L: shore_floor_tiles at LAKE_WATER + LAKE_FREEBOARD)'),
    'lake_centre_wet': (lake_centre_wet, (MAP, LF), ('polish', 'sync_final'), 'metric', 16, 0, False,
                        'polish lake shore shaping + lake ceilings (height bake U); v21b 184, copy run 16; only goes down'),
    'lake_waterline_straight': (lake_waterline_straight, (MAP, LF), ('polish', 'sync_final'), 'metric', 80, 0, True,
                                'polish smooth waterline; the longest runs follow straight lake hex outlines of the map itself (world 96,819: v21b 78, copy 80): a map bake (lake outline) item, record only'),
    'lake_shore_step': (lake_shore_step, (MAP, LF), ('polish', 'sync_final'), 'metric', .64, .02, False,
                        'polish lake shore shaping (eased relief, shelf from the plane) + lake bowl (flat core, v21d); v21b 1.155, v21c .876, v21d copy .632'),
    # LAKE LEVEL (lake_level.py, TW3K_LAKE_LEVEL=1; skip with it off). Product: analysis/lake-water-levels.json + .npz
    'lake_level_mesh_height': (lake_level_mesh_height, (MAP, LF, LEVEL, 'analysis/lake-water-levels.json'), ('polish', 'sync_final'), 'exact', 1e-6, None, False,
                               'river_water_mesh.lake_surface / plan reading polish LAKE LEVEL'),
    'lake_level_bed_wet': (lake_level_bed_wet, (MAP, LF, LEVEL, 'analysis/lake-water-levels.json'), ('polish', 'sync_final'), 'exact', 0, None, False,
                           'polish LAKE LEVEL SHAPE (lake_level.shaped_ground) + projection ceilings (lake_level.projection_bounds); river_water_mesh mesh pad'),
    'lake_level_mesh_edge': (lake_level_mesh_edge, (MAP, LF, LEVEL, 'analysis/lake-water-levels.json'), ('polish', 'sync_final'), 'exact', 0, None, False,
                             'polish LAKE LEVEL SHAPE bank floor + projection floors; river_water_mesh lake_surface mesh_pad'),
    'lake_level_waterline': (lake_level_waterline, (LF, LEVEL, 'analysis/lake-water-levels.json'), ('polish', 'sync_final'), 'exact', 0, None, False,
                             'polish LAKE LEVEL SHAPE bank + projection floors'),
    'lake_level_shore_step': (lake_level_shore_step, (LF, 'analysis/lake-water-levels.json'), ('polish', 'sync_final'), 'exact', LAKE_LEVEL_STEP_SLOPE, None, False,
                              'polish LAKE LEVEL SHAPE (outline_sdf, shaped_ground) + projection'),
    'lake_level_waterline_sawtooth': (lake_level_waterline_sawtooth, (LF, LEVEL, 'analysis/lake-water-levels.json'), ('polish', 'sync_final'), 'exact', LAKE_LEVEL_AXIS_EXCESS, None, False,
                                      'polish LAKE LEVEL outline (outline_sdf) + LF shape (lf_shape_delta) + projection + LAKE LEVEL BED (cut_bed)'),
    'lake_level_bed_profile': (lake_level_bed_profile, (LF, LEVEL, 'analysis/lake-water-levels.json'), ('polish', 'sync_final'), 'exact', 0, None, False,
                               'polish LAKE LEVEL BED (lake_level.cut_bed after the slope projection)'),
    'lake_level_bank_envelope': (lake_level_bank_envelope, (LF, LEVEL, 'analysis/lake-water-levels.json'), ('polish', 'sync_final'), 'exact', 0, None, False,
                                 'polish LAKE LEVEL SHAPE bank cap + town level solve (envelope cap, owner footprint) + projection ceilings'),
    'lake_level_towns': (lake_level_towns, (LF, 'analysis/lake-water-levels.json', 'work/current-region-design.json'), ('polish', 'sync_final'), 'exact', 0, None, False,
                         'polish LAKE LEVEL outline (stops at plot hexes) + town level solve (owner footprint); town levels'),
    'lake_level_town_envelope': (lake_level_town_envelope, (LF, 'analysis/lake-water-levels.json', 'work/current-region-design.json'), ('polish', 'sync_final'), 'exact', 0, None, False,
                                 'polish town level solve (cluster mix cap, plateau targets: lake_level.town_cap) + projection'),
    'lake_level_lake_hex_wet': (lake_level_lake_hex_wet, (LF, 'analysis/lake-water-levels.json'), ('polish', 'sync_final'), 'exact', round(1 - LAKE_LEVEL_WET_SHARE, 4), None, False,
                                'polish LAKE LEVEL outline (never cuts into the lake) + bed shaping'),
    'plots_on_lakes': (plots_on_lakes, (MAP,), ('map',), 'exact', 0, None, False,
                       'the map stage (town stamps: move_town / design); polish never places plots'),
    'lake_level_outside_identical': (lake_level_outside_identical, (LF, 'analysis/lake-water-levels.json', 'work/current-region-design.json'), ('polish',), 'exact', 0, None, False,
                                     'polish LAKE LEVEL: Bosten out of the bowl, coastal falloff, shaping, river mouth ease, town level solve; reach from the product'),
    'town_rim_step': (town_rim_step, (LF, 'work/current-region-design.json'), ('polish', 'sync_final'), 'exact', .06, None, False,
                      'polish town ground mix (option A: inverse-distance mix, plot edge = own level); v21 .1107, copy run with A .0512'),
    'tundra_pale_share': (tundra_pale_share, (BLEND, LANDFORM_TILES), ('polish',), 'metric', .015, .005, False,
                          'ground_materials tundra (biome_profiles TUNDRA_BAN + TUNDRA_SCREE, §9.3); v21 .15, ban .077, ban+scree .033 (v22b, live fail); no arid_3: cap .015 fixed 2026-11-01'),
    'tundra_arid3': (tundra_arid3, (BLEND, LANDFORM_TILES), ('polish',), 'exact', 0, None, False,
                     'ground_materials tundra (biome_profiles TUNDRA_SCREE without arid_3, TUNDRA_BAN_EXTRA; v22b live camouflage)'),
    'tundra_cold2_treeless': (tundra_cold2_treeless, (BLEND, LANDFORM_TILES), ('polish',), 'exact', 0, None, False,
                              'ground_materials tundra (biome_profiles TUNDRA_SCREE without cold_2, TUNDRA_BAN_EXTRA; v22b live dark spots)'),
    'tundra_oasis': (tundra_oasis, (LANDFORM_TILES, REAL, 'work/canvas-authored-coordinates.json'), ('polish',), 'exact', 0, None, False,
                     'polish [TUNDRA OASIS OFF] (biome_profiles no_oasis / no_alpine_belt, TW3K_TUNDRA_OASIS_OFF; v22b pale river corridor)'),
    'forest_fill_floor': (forest_fill_floor, (LANDFORM_TILES,), ('polish',), 'exact', 0, None, False,
                          'frontier_forest.fill_weight FILL FLOOR (biome_profiles FOREST_FILL_FLOOR, TW3K_FOREST_FILL_FLOOR): ground and tree fill one weight'),
    'treeline_straight': (treeline_straight, (BLEND, LF, LANDFORM_TILES), ('polish', 'sync_final'), 'metric', .5, .1, False,
                          'polish donor climate on the tundra/forest line (ground_donor.treeline_weight/treeline_pick, TREELINE_DONOR, section 9.3); v22 zone see detail'),
    'taiga_light_share': (taiga_light_share, (BLEND, LANDFORM_TILES), ('polish',), 'metric', .15, .03, False,
                          'polish donor choice (ground_donor.compress_choice, biome_profiles DONOR_COMPRESS, section 9.3); v21 .26'),
    'ground_fragmentation': (ground_fragmentation, (BLEND, LANDFORM_TILES, MAP), ('polish', 'sync_final'), 'exact', 0, None, False,
                             'ground_materials landform mixes / relief and GRAIN (biome_profiles LANDFORMS, CONSTS; ground mottle step 1)'),
    'farm_on_dry_ground': (farm_on_dry_ground, (BLEND, LANDFORM_TILES, MAP, WEST_REAL_TERRAIN, 'work/current-region-design.json'),
                           ('polish', 'sync_final'), 'exact', FARM_DRY_MAX, None, False,
                           'ground_materials.farm_mud_fields ARID_TOWN / farm_fields ARID_OWN_OASIS (biome_profiles CONSTS); oasis data'),
    'ground_banned_px': (ground_banned_px, (BLEND, LANDFORM_TILES), ('polish', 'sync_final'), 'exact', 0, None, False,
                         'ground_materials step 4 (apply_rows asserts 0); later blend writers: polish ban ledger names them (town ground: town spec)'),
    'tile_cache_matches_list': (tile_cache_matches_list, (TILEMAP, TILES), ('sync_final', 'graph'), 'exact', 0, None, False,
                                'polish cache segment: the display cache floors carry the list floors\' climate (tile bake)'),
    'tile_climate_matches_rule': (tile_climate_matches_rule, (BLEND, BLEND_CLIMATE, TILES, GTILES) + PLACED, ('sync_final', 'graph'), 'exact', 0, None, False,
                                  'polish cache segment TILE CLIMATE RULE (blend_climate_tiles.tile_rule: native ring carried, G band elsewhere)'),
    'blend_climate_fresh': (blend_climate_fresh, (BLEND, BLEND_CLIMATE), ('polish', 'sync_final'), 'exact', 0, None, False,
                            'polish BLEND CLIMATE TILES (blend_climate_tiles.save after global_blend); a later blend writer must rewrite it'),
    'side_lane_lf': (side_lane_lf, (), ('map',), 'exact', 0, None, False,
                     'pipeline start: build_polished_campaign_terrain.ps1 keeps LF readers out of the sync_final side lane'),
    'design_single_source': (design_single_source, (), ('map',), 'exact', 0, None, False,
                             'design writers (generator, layout tools, all_directions_build database): nested provinces[].regions = top-level regions'),
    'design_matches_map': (design_matches_map, (MAP,), ('map',), 'exact', 0, None, False,
                           'design writers / map bake: imported_sea_regions = the map non-vanilla seas, sdk_region_budget.total_regions = map regions'),
    'map_hex_is_l0': (map_hex_is_l0, (MAP,), ('map', 'mesh_final'), 'exact', 0, None, False,
                      'map.hex writers: final-map/map.hex comes only from bake_map.py --diff (data/map-layout-diff); refresh_border_masks only checks'),
    'filler_impassable': (filler_impassable, (MAP,), ('map',), 'exact', 0, None, False,
                          'oss_rebuild_layout (land handed back to the filler: impassable, roads only on region corridors)'),
    'settlement_primaries_ground': (settlement_primaries_ground, (MAP,), ('map',), 'exact', 0, None, False,
                                    'map writers (oss_rebuild_layout, add_ports, apply_shore_kinds): town slots stay plain land'),
    'dock_capacity': (dock_capacity, (MAP,), ('map',), 'exact', 0, None, False,
                      'coast_props.DOCK_PORTS (dock prefab numbers DOCK_BASE + 10 * port + part)'),
    'river_channels_is_l0': (river_channels_is_l0, (CHANNELS,), ('map', 'mesh_final'), 'exact', 0, None, False,
                             'river-channels.json writers: only bake_map.py --diff (copy of data/map-layout-diff/river-channels.json)'),
    'biome_profiles_frozen': (biome_profiles_frozen, (), ('map',), 'exact', 0, None, False,
                              'pipeline start: check_biome_profiles.py --refreeze in the commit that changes biome_profiles'),
    'snow_switches': (snow_switches, (), ('map',), 'exact', 0, None, False,
                      'pipeline start: TW3K_SNOW_BAKE=1 runs with TW3K_SNOW_COVERAGE=1 (binary mask reads a coverage share)'),
    'korea_patch_rules': (korea_patch_rules, (), ('map',), 'exact', 0, None, False,
                          'pipeline start: korea_patch.py build (rules_sha of biome_profiles.korea_forests; polish stops on a stale cache)'),
    'patch_towns_flattened': (patch_towns_flattened, (MAP,), ('map',), 'exact', 0, None, False,
                              'hehuang_patch.py / korea_patch.py build town levelling (hehuang_terrain.town_weight on the town hex; 2026-11-01)'),
    'patch_design': (patch_design, (), ('map',), 'exact', 0, None, False,
                     'pipeline start: hehuang_patch.py / korea_patch.py build (the npz towns follow the region design; polish stops on a stale cache)'),
    'wuyuan_patch_scope': (wuyuan_patch_scope, (MAP,), ('map',), 'exact', 0, None, False,
                           'wuyuan_patch.py build (bake step 10 / ensure_patch_caches): declared box, filler land only'),
    'wuyuan_core_real': (wuyuan_core_real, (), ('map',), 'exact', 0, None, False,
                         'wuyuan_patch.py build relief: core within CORE_MAX_DIFF (.5, frozen 2026-11-01) of to_relief(ETOPO)'),
    'wuyuan_playable_identical': (wuyuan_playable_identical, (MAP, LF), ('polish', 'sync_final'), 'exact', 0, None, False,
                                  'polish (patch blend / raise_alpha must stay off vanilla playable land; wuyuan PROTECT_HEXES 1)'),
    'vanilla_playable_lf': (vanilla_playable_lf, (LF,), ('polish', 'sync_final'), 'exact', 380, None, False,
                            'polish and the LF writers after it: vanilla playable ground keeps its LF (C2 baseline 380, only goes down)'),
    'wuyuan_band_slope': (wuyuan_band_slope, (MAP, LF), ('polish', 'sync_final'), 'exact', 0, None, False,
                          'wuyuan_patch.py BAND / polish: the transition band no steeper than the vanilla filler round the box'),
    'land_mesh_strays': (land_mesh_strays, (MESHES,), ('map',), 'exact', 0, None, False,
                         'pipeline start: the stage before this run (fill_polished_mesh_gaps overflow manifest; mesh bake 3.2.1)'),
    'base_land_mesh_set': (base_land_mesh_set, (), ('map', 'geometry', 'polish'), 'exact', 0, None, False,
                           'terrain_checkpoint capture geometry / polish snapshot loop (overflow land meshes are graph output; mesh bake 3.2.1)'),
    'oasis_town_trees': (oasis_town_trees, (FOLIAGE_REPORT,), ('graph',), 'exact', 0, None, False,
                         'polish_campaign_foliage TOWN OASIS FLOOR (foliage_biomes.town_oasis_rule; follow-up 2026-10-30)'),
    'flat_arid_rocks': (flat_arid_rocks, (TREES,), ('graph',), 'exact', 0, None, False,
                        'polish_campaign_foliage ROCK RELIEF (foliage_biomes.drop_flat_rocks)'),
    'native_y0_trees_grounded': (native_y0_trees_grounded, (TREES, LF), ('graph',), 'exact', 0, None, False,
                                 'polish_campaign_foliage native rows: vanilla y = 0 on the final LF (tree_bake.native_row_heights; 2026-10-30)'),
    'seam_static_classes': (seam_static_classes, (SEAM0,), ('seam_pass0',), 'exact', 0, None, False,
                            'audit_ground_seams first_five: lf_cliff / fine_range_stale (the seam loop cannot repair them)'),
    'tile_placed_first': (tile_placed_first, (TILES, GTILES), ('sync_final', 'graph'), 'exact', 0, None, False,
                          'tile_bake occupancy order: the special pieces polish lays go before floors, never culled (tile_bake.clear_overlays)'),
    # Threshold: v18 had 7,078 tile cells, 6,996 of them vanilla land in Jiangyang/Yunnan whose vanilla mountain pieces
    # (312 records) the export lost and place_floors did not restore. tile_bake.vanilla_specials restores them (9.2);
    # offline on the run 123911 stage: 943; measured on the first full run with it (20261031-001956, sync_final): 1,051,
    # all native, below vanilla dlc07's own 1,277. A baseline that only goes down: lower it when a change removes bare
    # cells, never raise it (pipeline-bake-design 9.2 has where the 1,051 lie).
    'tile_land_coverage': (tile_land_coverage, (MAP, TILES), ('sync_final', 'graph'), 'exact', 1051, None, False,
                           'tile_bake.place_floors / clear_overlays: land cells without a land fine tile'),
    'native_vanilla_records': (native_vanilla_records, (TILES, GTILES, LF) + NATIVE_RULE_INPUTS, ('sync_final', 'graph'), 'exact', 0, None, False,
                               'tile_bake.place_floors (vanilla_specials): vanilla special records on unedited native ground, field for field'),
    'tile_list_pair': (tile_list_pair, (TILES, GTILES), ('sync_final', 'graph'), 'exact', 0, None, False,
                       'every tile list writer (tile_bake, polish cache segment, range syncs): write tile_list.bin and global_map/tile_list.bin together'),
    'native_overlay_preservation': (native_overlay_preservation, (OVERLAY, TILES), ('graph',), 'exact', 0, None, False,
                                    'audit_native_overlay_preservation.py (graph node overlay-audit)'),
    'lf_mesh_diff': (lf_mesh_diff, (LF, MESHES, MAP), ('graph', 'seam_pass0', 'mesh_final'), 'exact', .12, None, False,
                     'fill_polished_mesh_gaps error-bounded lattice (mesh_lattice TOL_PLAYABLE; mesh bake 3.2.1)'),
    'lf_mesh_diff_nonplayable': (lf_mesh_diff_nonplayable, (LF, MESHES, MAP), ('graph', 'seam_pass0', 'mesh_final'), 'exact', .25, None, False,
                                 'fill_polished_mesh_gaps error-bounded lattice (mesh_lattice TOL_NONPLAYABLE; mesh bake 3.2.1)'),
    'tree_mesh_offset': (tree_mesh_offset, (LF, MESHES, MAP, TREES), ('graph', 'mesh_final'), 'metric', .01, 0, True,
                         'the mesh bake (fill_polished_mesh_gaps lattice, seam repair / height normalisation) vs polish_campaign_foliage tree heights (art-review gate 3)'),
    # Prop base vs ground (art review 2026-10-31, output/offline-view/art-review/10f4d8f6d2f4/prop-diff.md): against the
    # final LF right after the resource scenes (graph node scene-check) and against the packed land mesh at mesh_final.
    # Each name runs at one checkpoint with its own baseline (LF and mesh differ: fine-tile holes, LF smoothness).
    'prop_lf_offset': (prop_lf_offset, (PROPS, MAP, LF), ('resource_scenes',), 'metric', .094, 0, True,
                       'complete_campaign_resource_scenes (scene seat height: surface_at_display at the town hex) and the earlier prop chain'),
    'prop_scene_lf_offset': (prop_scene_lf_offset, (PROPS, MAP, LF), ('resource_scenes',), 'metric', 1, 0, False,
                             'complete_campaign_resource_scenes (scene seating on the ground under its pieces; v21 33, v21b 1)'),
    'prop_mesh_offset': (prop_mesh_offset, (PROPS, MAP, LF, MESHES), ('mesh_final',), 'metric', .059, 0, True,
                         'complete_campaign_resource_scenes, the graph prop writers (mountain / coast / landmark props), the mesh bake'),
    'prop_scene_offset': (prop_scene_offset, (PROPS, MAP, LF, MESHES), ('mesh_final',), 'metric', 2, 0, False,
                          'complete_campaign_resource_scenes (scene seating; v21 30, v21b 2); a later prop writer or the mesh bake moving a scene off its ground'),
    'land_mesh_budget': (land_mesh_budget, (MESHES,), ('graph', 'mesh_final'), 'exact', 10., None, False,
                         'fill_polished_mesh_gaps (mesh_lattice tolerances; mesh bake 3.2.1 geometry budget)'),
    'frontier_mountain_family_count': (frontier_mountain_family_count, (PRE + 'global_props.bin', 'analysis/frontier-mountain-props.json'), ('graph',), 'exact', 0, None, True,
                                       'frontier_mountain_props SEED_OVERSAMPLE (no family cap; biome_profiles props MOUNTAIN_FAMILY_COUNT_TOL)'),
    'frontier_mountain_stable': (frontier_mountain_stable, (MAP, PRE + 'global_props.bin', f'analysis/polished-surface-{VARIANT}.npy'), ('graph',), 'exact', 0, None, False,
                                 'frontier_mountain_props STATELESS DRAWS (graph node mountain-props): props change only where the terrain under them does'),
    'snow_native_parity': (snow_native_parity, (SNOW_MASK,), ('graph',), 'exact', 0, None, False,
                           'snow_bake (graph node snow, after the LF\'s last writer): native winter snow as vanilla, R = G = B = the peaks'),
    'snow_peaks_all_channels': (snow_peaks_all_channels, (SNOW_MASK, SNOW_JSON), ('graph',), 'exact', 0, None, False,
                                'snow_bake: year-round peaks are winter snow too; peak count as baked'),
    'snow_frame': (snow_frame, (SNOW_MASK, SNOW_JSON, MAP, LF), ('graph',), 'exact', 0, None, False,
                   'snow_bake: mask size and inputs of this canvas, map.hex and rule code'),
    'snow_lighting': (snow_lighting, (ENV_COLLECTION, SNOW_MATERIAL), ('graph',), 'exact', 0, None, False,
                      'snow_bake lighting groups + translate_campaign_environment_spheres references (start crash guard)'),
    'snow_on_water': (snow_on_water, (SNOW_MASK, MAP), ('graph',), 'exact', 0, None, False,
                      'snow_bake: no snow on canvas water'),
    'snow_peaks_visible': (snow_peaks_visible, (SNOW_MASK, 'analysis/west-real-terrain.npz'), ('graph',), 'exact', 0, None, False,
                           'snow_bake: a white top (>= 15 px) at every named summit but the LF lake-basin exceptions'),
    'snow_mask_binary': (snow_mask_binary, (SNOW_MASK,), ('graph',), 'exact', 0, None, False,
                         'snow_bake: outside the native core every byte on every mip is 0 or >= 232 (the shader is a switch)'),
}
REGRESS = {'river_centre_wet': river_regress}
# Release exemptions (user approved, one release each): the invariant and its threshold stay; a value up to `max` is
# recorded as 'exempt' instead of failing, anything beyond still fails. Remove the entry once the next release passes.
EXEMPT = {
    'wuyuan_band_slope': {'max': .0366, 'release': 'C3 (2026-11-01)',
                          'why': 'user 2026-11-01 (c22): the east band on the river bank (q 1246-1248, r 996-997) steps .3852 '
                                 '(polish slope cap) against the vanilla ring .3486, because the Tumed plain east of the box '
                                 '(vanilla still ~2 above the real ground) is not in this repaint; next release: Tumed / Hohhot '
                                 'plain repaint and the band rule compared polish to polish'},
}
# Rollout: name -> (setting, default). Hard in the table, but only warns until its setting is 'hard'. The in-run
# fail of river_centre_wet (a centre dry after sync_final that was wet after polish) is not calibrated yet, so the
# first rollout records it without stopping the pipeline. Make it hard per run with $env:TW3K_INVARIANT_RIVER_
# CENTRE_WET='hard', or for good by changing its default here to 'hard' (or deleting the entry) once the records
# of a few runs show its regressions are real.
ROLLOUT = {'river_centre_wet': ('TW3K_INVARIANT_RIVER_CENTRE_WET', 'warn')}


def rollout_warn(name):
    """True while a ROLLOUT invariant's setting is not 'hard'."""
    if name not in ROLLOUT:
        return False
    env, default = ROLLOUT[name]
    return os.environ.get(env, default).strip().lower() != 'hard'


# --------------------------------------------------------------------------- record
def load_record(ws):
    p = ws / RECORD
    if not p.is_file():
        return []
    out = []
    for line in p.read_text(encoding='utf-8').splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def writers(step, reads):
    if step == 'graph':
        w = sorted({n for r in reads for n in GRAPH_WRITERS.get(r, ())})
        return w or ['(graph)']
    return [STEP_SCRIPTS[step]]


def evaluate(ctx, step, run, warn_all, only):
    old = [] if ctx.over else load_record(ctx.ws)   # --map: another map, no record
    order = {s: i for i, s in enumerate(STEPS)}
    rows, failures = [], []
    for name, (fn, reads, steps, kind, thr, tol, warn_only, owner) in INVARIANTS.items():
        if step not in steps or (only and name not in only):
            continue
        t = time.time()
        row = {'run': run, 'step': step, 'invariant': name, 'kind': kind, 'threshold': thr, 'tolerance': tol,
               'warn_only': warn_only or warn_all or rollout_warn(name), 'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'owner': owner}
        try:
            res = fn(ctx)
        except Missing as e:
            row.update(status='skip', reason=f'input missing or unusable: {e}', seconds=round(time.time() - t, 2)); rows.append(row); continue
        except (Exception, SystemExit) as e:   # a checker problem never fails the pipeline (strategic_ink exits)
            traceback.print_exc(file=sys.stdout)
            row.update(status='error', reason=f'{type(e).__name__}: {e}', seconds=round(time.time() - t, 2)); rows.append(row); continue
        row.update(value=res['value'], detail=res.get('detail'))
        if 'state' in res:
            row['state'] = res['state']
        bad, warn = [], []
        if thr is not None and res['value'] > thr:
            bad.append(f'value {res["value"]} > threshold {thr}' + (f': {res["reason"]}' if res.get('reason') else ''))
        if kind == 'metric':
            same = [o for o in old if o.get('invariant') == name and 'value' in o]
            prev = [o for o in same if o.get('run') == run and order.get(o.get('step'), 99) < order[step]]
            ref = prev[-1] if prev else next((o for o in reversed(same) if o.get('run') != run and o.get('step') == step), None)
            if ref is not None:
                reg = REGRESS.get(name, lambda a, b: a['value'] - b['value'])(res, ref)
                row.update(ref={'run': ref['run'], 'step': ref['step'], 'value': ref['value']}, regress=reg)
                if reg > tol:
                    (bad if prev else warn).append(f'regressed by {reg} (> {tol}) against {"this run after " + ref["step"] if prev else "run " + ref["run"] + " after " + ref["step"]}')
        ex = EXEMPT.get(name)
        if bad and ex and res['value'] <= ex['max']:
            row.update(exempt=ex)
            warn, bad = warn + [f'exempt for release {ex["release"]} (value {res["value"]} <= {ex["max"]}): ' + '; '.join(bad)], []
        if bad and row['warn_only']:
            warn, bad = warn + bad, []
        row['status'] = 'fail' if bad else 'warn' if warn else 'pass'
        if bad or warn:
            row['reason'] = '; '.join(bad + warn)
        if bad:
            mine = [o for o in old if o.get('run') == run and o.get('invariant') == name and o.get('status') == 'pass']
            row['writers'] = writers(step, reads)
            since = (f'it passed after {mine[-1]["step"]}, so a step after that broke it' if mine
                     else f'first checked here: the outputs of {step} or an earlier step break it')
            failures.append(f'STAGE INVARIANT FAILED after {step}: {name}: {row["reason"]}; {since}; '
                            f'its artifacts ({", ".join(reads)}) are written by: {", ".join(row["writers"])}. Owner: {owner}')
        row['seconds'] = round(time.time() - t, 2)
        rows.append(row)
    return rows, failures


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--after', choices=STEPS)
    ap.add_argument('--workspace', type=Path)
    ap.add_argument('--run', help='run id (default TW3K_INVARIANT_RUN, else manual-<time>)')
    ap.add_argument('--warn-only', action='store_true', help='report failures as warnings (exit 0)')
    ap.add_argument('--only', help='comma-separated invariant names')
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--build-cache', action='store_true', help='build the landform cache of desert_banned_share')
    ap.add_argument('--node', action='store_true', help='run as a step-graph node: a checker failure (exit 3) exits 0 '
                                                      'with the warning, as Invoke-Tw3kInvariants treats it')
    ap.add_argument('--map', type=Path, help='with --after map: check this map.hex instead of the workspace one '
                                             '(read-only: no record is read or written)')
    args = ap.parse_args()
    if args.list:
        for name, (_, reads, steps, kind, thr, tol, wo, owner) in INVARIANTS.items():
            print(f'{name:28s} {kind:6s} after {",".join(steps):18s} threshold {thr} tol {tol}{" warn-only" if wo else ""}'
                  f'{" warn-only (rollout: " + ROLLOUT[name][0] + "=hard makes it hard)" if rollout_warn(name) else ""}\n    reads {", ".join(reads)}\n    owner {owner}')
        return 0
    mode = os.environ.get('TW3K_INVARIANTS', '').lower()
    if mode == 'off':
        print('stage invariants: TW3K_INVARIANTS=off, skipped'); return 0
    if args.workspace:
        os.environ['TW3K_WORKSPACE'] = str(args.workspace)   # before campaign_world reads the design
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import campaign_world as cw
    ctx = Ctx(cw.WORKSPACE)
    if args.map:
        if args.after != 'map':
            ap.error('--map goes with --after map')
        ctx.over[MAP] = args.map.resolve()
    if args.build_cache:
        t = time.time(); _, names, built = landform_classes(ctx)
        print('landform cache', 'built' if built else 'current', names, f'{time.time() - t:.1f}s'); return 0
    if not args.after:
        ap.error('--after STEP is required')
    run = args.run or os.environ.get('TW3K_INVARIANT_RUN') or time.strftime('manual-%Y%m%d-%H%M%S')
    t0 = time.time()
    rows, failures = evaluate(ctx, args.after, run, args.warn_only or mode == 'warn', set(args.only.split(',')) if args.only else None)
    from terrain_regression import peak_mb
    summary = {'run': run, 'step': args.after, 'invariant': '*', 'seconds': round(time.time() - t0, 2), 'peak_mb': peak_mb(),
               'statuses': {r['invariant']: r['status'] for r in rows}, 'reads': ctx.reads, 'time': time.strftime('%Y-%m-%d %H:%M:%S')}
    if not ctx.over:
        (ctx.ws / RECORD).parent.mkdir(parents=True, exist_ok=True)
        with open(ctx.ws / RECORD, 'a', encoding='utf-8') as f:
            for r in rows + [summary]:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
    for r in rows:
        extra = f' value {r.get("value")} (threshold {r["threshold"]}{", regress " + str(r["regress"]) if "regress" in r else ""})' if 'value' in r else ''
        print(f'[invariant] after {args.after:10s} {r["invariant"]:28s} {r["status"].upper():5s}{extra} {r["seconds"]:.1f}s'
              + (f' -- {r["reason"]}' if r.get('reason') else ''), flush=True)
    print(f'[invariant] after {args.after}: {len(rows)} checked in {summary["seconds"]:.1f}s, peak {summary["peak_mb"]} MB; ' + ('no record (--map)' if ctx.over else f'record {ctx.ws / RECORD}'), flush=True)
    for msg in failures:
        print(msg, flush=True)
    return 1 if failures else 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc(file=sys.stdout)
        print('stage invariants: the checker itself failed (exit 3; the pipeline only warns)', flush=True)
        sys.exit(0 if '--node' in sys.argv else 3)
