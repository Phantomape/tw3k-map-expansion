"""Ground patchiness of the campaign global blend: hard straight edges, dark islands, farm/wetland squares.

The calibration scores of the offline renderer are averages, so nothing alarmed on the three
global_blend faults the v19 close-ups showed (ring-west, funan, indus; all present in v18):
  * long straight / stair-step hard edges where vanilla temperate greens (27/28) carried in by the
    mirror band meet the cold woodland base (4) a few hexes outside the native border;
  * big dark-olive blotches of the slot-4 woodland base brought in by donors;
  * wetland (31) and farmland (11) painted in small blocks that read as squares.
This module measures them on the blend itself. Colour model (analysis/offline-view-research.md):
R = material slot = texture-array layer (identity table), G = climate band (it only enters through
the engine's runtime table, so it does not change the colour here); each layer is reduced to the
mean colour of its texture for one season (CIE Lab, D65), kept in analysis/ground-layer-colours.json.

Frames: blend pixel (px) = LF pixel, 4 per tile, 8 per hex column (campaign_world: tile
tx = 2q, ty = 2r + odd(q)); arrays here are south-up (row 0 = south), like terrain_regression.
Grid lines in px: tile lines every 4 px; hex lines x = 8q and, inside column q, y = 8r + 4*odd(q).

Metrics (thresholds are the module constants below; validation against the v19 live close-ups and
the changes made while tuning them: analysis/ground-patchiness.md):
  contrast     every pair of 4-neighbour land pixels of different layers: mean dE76 and |dL*|
  hard edge    a different-layer pair with dE >= HARD_PAIR_DE between the two pixels and between
               the means of the SIDE_PX pixels on each side (a dithered or gradual transition
               mixes the sides and fails it); length in hex edge lengths (8 px), per 100 land
               hexes, shares on tile / hex grid lines
  straight     hard pairs whose neighbourhood of hard pairs (box of side 2 STRAIGHT_R + 1) is a
  segment      line (position-covariance coherence >= STRAIGHT_COH, >= STRAIGHT_MIN pairs);
               connected straight pairs that are a line as a whole (coherence >= STRAIGHT_COH) of
               length sqrt(12 l1) >= SEG_MIN_PX form a segment; count, length (hexes), length per 100
               land hexes, longest segment, share on tile lines
  dark island  on tile-resolution L* (mean of the 4x4 pixels, ~ the shader's bilinear blend):
               land components darker than a level of ISLAND_LEVELS, ISLAND_MIN_TILES..
               ISLAND_MAX_TILES large, whose ring (ISLAND_RING tiles) is >= ISLAND_DL lighter on
               average with >= ISLAND_RING_SHARE of its tiles lighter than island + ISLAND_DL / 2;
               count, area (hexes), share of land
  square       small 8-connected blobs of slot 11 or 31 (SQUARE_MIN_PX..SQUARE_MAX_PX), i.e.
               farm / wetland painted as specks of a few blocks instead of fields; count, area,
               per 100 land hexes, and the share of their outline on hex lines (per-hex painting 1)
Read-only: the blend / LF are read from a pack or loose DDS with plain reads (never memory-mapped,
no file kept open).
"""
import json, os, struct
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VARIANT = '3k_dlc07_main_map'
BLEND_KEY = f'terrain/campaigns/{VARIANT}/global_map/global_blend.dds'
LF_KEY = f'terrain/campaigns/{VARIANT}/lf_height_map.dds'
COLOURS = ROOT / 'analysis/ground-layer-colours.json'
LAYER_NAMES = ('arid_1', 'arid_2', 'arid_3', 'arid_4', 'cold_1', 'cold_2', 'cold_3', 'cold_4', 'farms_grain_1', 'farms_grain_2',
               'farms_rice_1', 'farms_rice_2', 'farms_tea_1', 'imperial_road', 'mud', 'shroud_1', 'shroud_2', 'shroud_3', 'shroud_4',
               'steppe_1', 'steppe_2', 'steppe_3', 'steppe_4', 'subtropical_1', 'subtropical_2', 'subtropical_3', 'subtropical_4',
               'temperate_1', 'temperate_2', 'temperate_3', 'temperate_4', 'wetlands')   # texture_arrays.xml order
BANDS = ('cold', 'arid', 'temperate', 'sub_tropical')   # G channel order (_settings.bin)
PX_PER_TILE, PX_PER_HEX = 4, 8
PLANE = -.015                  # campaign water plane (polish LAKE_WATER): LF below it is water

# hard pair: dE76 above vanilla's own neighbour contrast (ground_tone.max_dE: p99 of differing adjacent pixel
# pairs on vanilla native land, user decision 2026-10-27), with the SIDE_PX-pixel side means apart by
# HARD_SIDE_FRAC of it (no dithered or gradual transition)
HARD_SIDE_FRAC = .9
# straight segments are traced on strong edges: pairs above vanilla's EDGE_PCT percentile of differing pairs
# (CONTRAST percentiles_of_differing), not only the p99 hard pairs (validation 2026-10-30: the mirror-band edge
# temperate 27/28 | woodland 4 is dE 30-34, below vanilla's p99 36.9, and it is its straightness and length that
# are unlike vanilla, not the contrast of the pair)
EDGE_PCT = '90'
# fields and roads are straight by design (vanilla's farm plots are rectilinear): pairs touching these slots are
# not traced for straight segments
FIELD_SLOTS = tuple(range(8, 14))
# shores follow the water: pairs within SHORE_PX of water are not traced either (vanilla's straight segments
# were river and lake shore strips: sand beside a straight channel)
SHORE_PX = 8
SIDE_PX = 3
STRAIGHT_R = 24                # straightness box half-size (px, 3 hexes)
STRAIGHT_COH = .85             # line coherence of the hard pairs in that box
STRAIGHT_MIN = 36              # at least this many hard pairs in the box (a straight line through it has ~49)
SEG_MIN_PX = 32                # a straight segment counts from 4 hex lengths
ISLAND_DL = 8.                 # L* drop of a dark island below its ring
ISLAND_LEVELS = (34, 38, 42, 46, 50)   # L* levels whose dark components are island candidates
ISLAND_MIN_TILES = 16          # size floor (4 hexes)
ISLAND_MAX_TILES = 2400        # 600 hexes: larger dark components are regions, not islands
ISLAND_RING = 4                # ring width (tiles) round an island
ISLAND_RING_SHARE = .6         # share of the ring that must be lighter by ISLAND_DL / 2
SQUARE_SLOTS = (11, 31)        # farms_rice_2, wetlands
SQUARE_MIN_PX = 16             # a quarter hex
SQUARE_MAX_PX = 6 * 64         # 6 hexes: larger blobs are fields / marshes
STRIP_PX = 1024                # whole-map pixel strips (rows) and their context margin
STRIP_MARGIN = 64
# alarm levels per window: ALARM_PCT[metric] of the same numbers over vanilla dlc07's land regions
# (audit_ground_patchiness.py vanilla -> VANILLA_JSON 'alarm'); a window above one is 'lit'.
# Per metric (user decision 2026-10-30, decisions.md 38010152): the longest straight edge at p95, since at p99
# the ring-west fault (10.7 hex) stays dark and the controls stay dark at p95 (siberia 4.1); the rest at p99,
# since a uniform p95 lights the siberia control on hard edges (3.63 vs p95 2.11)
ALARM_METRICS = ('straight_segment_max_hex', 'straight_segment_per100', 'dark_island_share', 'square_per100', 'hard_edge_per100')
ALARM_PCT = {'straight_segment_max_hex': 95, 'straight_segment_per100': 99, 'dark_island_share': 99, 'square_per100': 99, 'hard_edge_per100': 99}
VANILLA_JSON = ROOT / 'analysis/ground-patchiness-vanilla.json'


# --------------------------------------------------------------------------- tones
def vanilla_packs():
    """campaign_ground_textures.VanillaPacks with its cache taken from the nearest checkout that has one (a
    worktree has no output/: without this the pack index would be rebuilt by reading every game pack)."""
    import campaign_ground_textures as cgt
    for d in (ROOT, *ROOT.parents):
        if (d / 'output/offline-view/cache/pack_index.pkl').is_file():
            cgt.CACHE = d / 'output/offline-view/cache'
            break
    return cgt.VanillaPacks()


def vanilla_map():
    """campaign_world.VANILLA_MAP, or the same file in the nearest checkout that has it (tools/ is not in a worktree)."""
    import campaign_world as cw
    rel = cw.VANILLA_MAP.relative_to(cw._ROOT)
    return next((d / rel for d in (cw._ROOT, *cw._ROOT.parents) if (d / rel).is_file()), cw.VANILLA_MAP)


class Tones:
    """Slot tones (ground_tone / biome_profiles CONSTS MATERIAL_TONE, the table the ground bake reads) and
    the global hard-pair threshold (CONTRAST.max_dE). lab: (256, 3) float32 per R value (rows 32+ zero).
    The threshold retains its original calibration season; it is not a per-season or per-G-band threshold."""

    @staticmethod
    def _table(raw, season, source, flat_season='harvest', required=False):
        """Read {season: 32x3}, legacy {slot: Lab}, or a legacy 32x3 array without changing seasons silently."""
        if raw is None:
            if required:
                raise ValueError(f'{source}: MATERIAL_TONE is missing')
            return None, source
        if isinstance(raw, dict):
            numeric = all(str(k).isdigit() for k in raw)
            if numeric:
                if len(raw) != 32 or {int(k) for k in raw} != set(range(32)):
                    raise ValueError(f'{source}: flat MATERIAL_TONE must contain exactly slots 0-31')
                if season != flat_season:
                    if required:
                        raise ValueError(f'{source}: flat MATERIAL_TONE is for {flat_season!r}, requested {season!r}')
                    return None, source
                raw = [next(v for k, v in raw.items() if int(k) == i) for i in range(32)]
                source += f' MATERIAL_TONE (flat, season={flat_season})'
            else:
                if any(str(k).isdigit() for k in raw):
                    raise ValueError(f'{source}: MATERIAL_TONE mixes season and slot keys')
                if season not in raw:
                    if required:
                        raise ValueError(f'{source}: MATERIAL_TONE has no {season!r} table; available: {sorted(raw)}')
                    return None, source
                raw = raw[season]
                source += f' MATERIAL_TONE[{season!r}]'
        else:
            if season != flat_season:
                if required:
                    raise ValueError(f'{source}: flat MATERIAL_TONE is for {flat_season!r}, requested {season!r}')
                return None, source
            source += f' MATERIAL_TONE (flat, season={flat_season})'
        try:
            table = np.asarray(raw, np.float32)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f'{source}: MATERIAL_TONE must be a numeric 32x3 array') from exc
        if table.shape != (32, 3) or not np.isfinite(table).all():
            raise ValueError(f'{source}: MATERIAL_TONE must be finite with shape (32, 3), got {table.shape}')
        return table, source

    def __init__(self, season='harvest', doc=None):
        """Explicit doc / TW3K_MATERIAL_TONES selects that measurement. Otherwise prefer CONSTS, with the
        default analysis/material-tones.json as fallback. Legacy flat data is harvest unless its document
        declares 'season' / 'start_season'; a requested season must exist or be extracted explicitly."""
        import biome_profiles
        if season not in ('spring', 'summer', 'harvest', 'autumn', 'winter'):
            raise ValueError(f'Unknown material-tone season: {season!r}')
        specified = doc or os.environ.get('TW3K_MATERIAL_TONES')
        path = Path(specified or ROOT / 'analysis/material-tones.json')
        c = biome_profiles.CONSTS['ground_materials']
        t, src = (None, '') if specified else self._table(c.get('MATERIAL_TONE'), season, 'biome_profiles CONSTS')
        contrast = None if specified else c.get('CONTRAST')
        threshold_src = 'biome_profiles CONSTS CONTRAST.max_dE (global, calibration=harvest)'
        if specified and not path.is_file():
            raise FileNotFoundError(f'Material-tone measurement does not exist: {path}')
        if (t is None or contrast is None) and path.is_file():
            d = json.loads(path.read_text(encoding='utf-8'))
            calibration = d.get('start_season', d.get('season', 'harvest'))
            if t is None:
                t, src = self._table(d.get('MATERIAL_TONE'), season, str(path),
                                     flat_season=d.get('season', calibration), required=bool(specified))
            if contrast is None:
                contrast = d.get('CONTRAST')
                threshold_src = f'{path} CONTRAST.max_dE (global, calibration={calibration})'
        if t is None and season != 'harvest' and not specified:
            import build_material_tones
            raw = build_material_tones.tones(season, vanilla_packs())[2]
            t, src = self._table(raw, season, f'build_material_tones.tones({season!r})', flat_season=season, required=True)
        if t is None or contrast is None:
            raise SystemExit('MATERIAL_TONE / CONTRAST not measured for the requested season: run build_material_tones.py')
        try:
            self.max_de = float(contrast['max_dE'])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f'{threshold_src}: expected a finite nonnegative number') from exc
        if not np.isfinite(self.max_de) or self.max_de < 0:
            raise ValueError(f'{threshold_src}: expected a finite nonnegative number')
        self.threshold_source = threshold_src
        pct = (contrast.get('percentiles_of_differing') or {}) if isinstance(contrast, dict) else {}
        self.edge_de = float(pct.get(EDGE_PCT, self.max_de))
        self.lab = np.zeros((256, 3), np.float32); self.lab[:32] = t
        self.season, self.source = season, f'{src}; threshold: {threshold_src}'
        self.srgb = np.zeros((256, 3), np.uint8); self.srgb[:32] = lab_to_srgb(t)


def lab_to_srgb(lab):
    """sRGB 0..255 uint8 of CIE Lab (D65) colours (last axis), for the blend previews."""
    lab = np.asarray(lab, np.float64)
    fy = (lab[..., 0] + 16) / 116; fx = fy + lab[..., 1] / 500; fz = fy - lab[..., 2] / 200
    f = np.stack([fx, fy, fz], -1)
    xyz = np.where(f > 6 / 29, f ** 3, 3 * (6 / 29) ** 2 * (f - 4 / 29)) * (.95047, 1., 1.08883)
    rgb = xyz @ np.array([[3.2404542, -.9692660, .0556434], [-1.5371385, 1.8760108, -.2040259], [-.4985314, .0415560, 1.0572252]])
    rgb = np.clip(rgb, 0, 1)
    return np.round(np.where(rgb <= .0031308, rgb * 12.92, 1.055 * rgb ** (1 / 2.4) - .055) * 255).astype(np.uint8)


def alarms(path=VANILLA_JSON):
    """Alarm level per metric from the vanilla pass ({} until it has run)."""
    if not Path(path).is_file():
        return {}
    return {k: v for k, v in json.loads(Path(path).read_text(encoding='utf-8')).get('alarm', {}).items() if k in ALARM_METRICS}


# --------------------------------------------------------------------------- reading
def pack_entry(pack, key):
    from analyze_reference import inventory
    e = next((x for x in inventory(Path(pack), False)['entries'] if x['path'] == key), None)
    if e is None:
        raise KeyError(f'{key} not in {pack}')
    assert not e['compressed'], key
    return e['offset'], e['size']


class BlendSource:
    """global_blend (+ LF for the land mask) from a pack or loose DDS files, read in row bands with
    plain reads (no memory map, no file kept open). Rows are returned south-up."""

    def __init__(self, pack=None, blend=None, lf=None, entries=None, land_hex=None):
        """entries: {'blend': (file, offset), 'lf': ...} (e.g. from the vanilla pack index); land_hex: a hex
        land grid (H, W) used instead of the LF test (vanilla: its map.hex)."""
        self.src, self.land_hex = dict(entries or {}), land_hex
        for name, key, loose in (('blend', BLEND_KEY, blend), ('lf', LF_KEY, lf)):
            if loose:
                self.src[name] = (Path(loose), 0)
            elif pack:
                self.src[name] = (Path(pack), pack_entry(pack, key)[0])
        with open(self.src['blend'][0], 'rb') as f:
            f.seek(self.src['blend'][1]); head = f.read(128)
        self.h, self.w = struct.unpack_from('<2I', head, 12)
        self.hexes = (self.w // PX_PER_HEX, (self.h - 4) // PX_PER_HEX)   # (W, H) of the hex grid

    @classmethod
    def vanilla(cls):
        """The vanilla dlc07 blend and LF from the game's packs (read-only); land is the LF test, as for the
        packs (map.hex land is per hex: its staircase coasts put seabed pixels on land and read as straight
        edges); the vanilla map.hex cells are kept for the regions."""
        import map_hex
        idx = vanilla_packs().index
        ent = {}
        for name, key in (('blend', BLEND_KEY), ('lf', LF_KEY)):
            p, off, _, comp = idx[key]
            assert not comp, key
            ent[name] = (Path(p), off)
        cells = map_hex.read(vanilla_map())['cells']
        src = cls(entries=ent)
        src.cells = cells
        return src

    def rows(self, name, y0, y1):
        """South-up rows [y0, y1): blend (n, w, 2) uint8 or LF (n, w) uint16."""
        path, off = self.src[name]
        y0, y1 = max(0, y0), min(self.h, y1)
        with open(path, 'rb') as f:
            f.seek(off + 128 + (self.h - y1) * self.w * 2)
            raw = f.read((y1 - y0) * self.w * 2)
        a = np.frombuffer(raw, np.uint8).reshape(y1 - y0, self.w, 2) if name == 'blend' else np.frombuffer(raw, '<u2').reshape(y1 - y0, self.w)
        return a[::-1]

    def band(self, y0, y1, x0=0, x1=None):
        """(R, land) south-up for px rows [y0, y1), cols [x0, x1); land = LF >= PLANE (all land without an LF)."""
        import campaign_world as cw
        x1 = self.w if x1 is None else x1
        R = np.ascontiguousarray(self.rows('blend', y0, y1)[:, x0:x1, 0])
        if self.land_hex is not None:
            hh, hw = self.land_hex.shape
            ys = np.arange(max(0, y0), max(0, y0) + R.shape[0])[:, None]; xs = np.arange(x0, x0 + R.shape[1])[None]
            q = np.clip(xs // PX_PER_HEX, 0, hw - 1); r = np.clip((ys // PX_PER_TILE - q % 2) // 2, 0, hh - 1)
            land = self.land_hex[r, q]
        elif 'lf' in self.src:
            raw = self.rows('lf', y0, y1)[:, x0:x1]
            land = (cw.LEVEL + (raw.astype(np.float32) / 65535 - cw.NORM) * cw.GAIN) >= PLANE
        else:
            land = np.ones(R.shape, bool)
        return R, land


def array_band(R_all, land_all):
    """band() over in-memory south-up arrays (terrain_regression's blend pixels and hex land)."""
    def band(y0, y1, x0, x1):
        y0, x0 = max(0, y0), max(0, x0)
        return np.ascontiguousarray(R_all[y0:y1, x0:x1]), np.ascontiguousarray(land_all[y0:y1, x0:x1])
    return band


# --------------------------------------------------------------------------- geometry
def px_hex(x, y):
    """Hex (q, r) of south-up blend pixels (campaign_world.tile_hex of their tile)."""
    import campaign_world as cw
    return cw.tile_hex(np.asarray(x) // PX_PER_TILE, np.asarray(y) // PX_PER_TILE)


def on_lines(x0, y0, shape):
    """Grid-line flags of the boundaries in a crop at px origin (x0, y0):
    vt/vh: boundary between columns x and x+1 lies on a tile / hex line (shape (h, w-1));
    ht/hh: boundary between rows y and y+1 lies on a tile / hex line (shape (h-1, w))."""
    h, w = shape
    xb = x0 + np.arange(1, w)               # boundary at x = xb (between xb-1 and xb)
    vt = np.broadcast_to(xb % PX_PER_TILE == 0, (h, w - 1))
    vh = np.broadcast_to(xb % PX_PER_HEX == 0, (h, w - 1))
    yb = y0 + np.arange(1, h)
    ht = np.broadcast_to((yb % PX_PER_TILE == 0)[:, None], (h - 1, w))
    q = (x0 + np.arange(w)) // PX_PER_HEX
    hh = ((yb[:, None] - 4 * (q[None] % 2)) % PX_PER_HEX) == 0
    return vt, vh, ht, hh


def line_coherence(m):
    """(coherence, count) of the points of mask m in a box of side 2 STRAIGHT_R + 1 round each pixel:
    coherence = (l1 - l2) / (l1 + l2) of their position covariance (1 on a straight line, ~0 for a blob)."""
    from scipy.ndimage import uniform_filter
    size = 2 * STRAIGHT_R + 1
    h, w = m.shape
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    f = lambda a: uniform_filter(a, size, mode='constant')
    m0 = f(m)
    with np.errstate(invalid='ignore', divide='ignore'):
        mx, my = f(m * x) / m0, f(m * y) / m0
        sxx = f(m * x * x) / m0 - mx * mx; syy = f(m * y * y) / m0 - my * my; sxy = f(m * x * y) / m0 - mx * my
        tr = sxx + syy
        coh = np.where(tr > 1e-6, np.sqrt((sxx - syy) ** 2 + 4 * sxy ** 2) / tr, 0.)
    return np.nan_to_num(coh), m0 * size * size


# --------------------------------------------------------------------------- pixel pass
class Acc:
    """Per-hex accumulators over a hex box (q0..q1, r0..r1) of the canvas."""
    FIELDS = ('land_px', 'L_sum', 'pairs', 'diff_pairs', 'de_sum', 'dl_sum', 'hard', 'hard_tile', 'hard_hex',
              'straight', 'straight_tile', 'seg_n', 'seg_len', 'seg_max', 'island_tiles', 'sq_n', 'sq_px', 'sq_bnd', 'sq_bnd_hex')

    def __init__(self, q0, q1, r0, r1):
        self.q0, self.q1, self.r0, self.r1 = q0, q1, r0, r1
        self.shape = (r1 - r0 + 1, q1 - q0 + 1)
        self.a = {k: np.zeros(self.shape, np.float32) for k in self.FIELDS}   # counts stay exact (< 2**24 per hex)

    def add_max(self, key, q, r, w):
        q, r, w = np.asarray(q), np.asarray(r), np.asarray(w, np.float32)
        sel = (q >= self.q0) & (q <= self.q1) & (r >= self.r0) & (r <= self.r1)
        np.maximum.at(self.a[key], (r[sel] - self.r0, q[sel] - self.q0), w[sel])

    def add(self, key, q, r, w=None):
        q, r = np.asarray(q), np.asarray(r)
        sel = (q >= self.q0) & (q <= self.q1) & (r >= self.r0) & (r <= self.r1)
        if not sel.any():
            return
        idx = (r[sel] - self.r0) * self.shape[1] + (q[sel] - self.q0)
        v = np.bincount(idx, weights=None if w is None else np.asarray(w, np.float64)[sel], minlength=self.shape[0] * self.shape[1])
        self.a[key] += v.reshape(self.shape).astype(np.float32)


def pixel_pass(R, land, x0, y0, T, acc, core=None):
    """Edge, segment and square statistics of a south-up crop with px origin (x0, y0), into acc.
    core = (cy0, cy1): only pixels / pairs whose first pixel row lies in [cy0, cy1) (crop rows), and
    segments / blobs whose centre row does, count. Returns the (h, w, 3) Lab image."""
    from scipy import ndimage
    h, w = R.shape
    cy0, cy1 = core or (0, h)
    L = T.lab[R]
    ys = np.arange(h)[:, None]; xs = np.arange(w)[None]
    q, r = px_hex(x0 + xs, y0 + ys)
    q = np.broadcast_to(q, (h, w)); r = np.broadcast_to(r, (h, w))
    rowok = np.zeros((h, 1), bool); rowok[cy0:cy1] = True
    m = land & rowok
    acc.add('land_px', q[m], r[m]); acc.add('L_sum', q[m], r[m], L[..., 0][m])
    vt, vh, ht, hh = on_lines(x0, y0, (h, w))
    # squares: small blobs of the SQUARE_SLOTS (label image shared by both slots; they never touch as one blob)
    sqlab = np.zeros((h, w), np.int32); nsq = 0
    for s in SQUARE_SLOTS:
        lb, n = ndimage.label((R == s) & land, structure=np.ones((3, 3)))
        sqlab[lb > 0] = lb[lb > 0] + nsq; nsq += n
    sq_size = np.bincount(sqlab.ravel(), minlength=nsq + 1)
    sq_small = (sq_size >= SQUARE_MIN_PX) & (sq_size <= SQUARE_MAX_PX); sq_small[0] = False
    sq_bnd = np.zeros(nsq + 1); sq_bnd_hex = np.zeros(nsq + 1)
    k = SIDE_PX
    cs = np.cumsum(np.pad(L, ((0, 0), (1, 0), (0, 0))), 1)
    csy = np.cumsum(np.pad(L, ((1, 0), (0, 0), (0, 0))), 0)
    pairs = []
    field = np.isin(R, FIELD_SLOTS) | ndimage.maximum_filter(~land, size=2 * SHORE_PX + 1)
    for axis in (1, 0):
        if axis == 1:   # horizontal neighbours: boundary between x and x+1 (a vertical line)
            a, b = (slice(None), slice(None, -1)), (slice(None), slice(1, None))
            i = np.arange(w - 1)
            lo, hi = np.clip(i + 1 - k, 0, w), np.clip(i + 1 + k, 0, w)
            left = (cs[:, i + 1] - cs[:, lo]) / (i + 1 - lo)[None, :, None]
            right = (cs[:, hi] - cs[:, i + 1]) / (hi - i - 1)[None, :, None]
            tline, hline, ok = vt, vh, land[a] & land[b] & rowok
        else:           # vertical neighbours: boundary between y and y+1 (a horizontal line)
            a, b = (slice(None, -1), slice(None)), (slice(1, None), slice(None))
            i = np.arange(h - 1)
            lo, hi = np.clip(i + 1 - k, 0, h), np.clip(i + 1 + k, 0, h)
            left = (csy[i + 1] - csy[lo]) / (i + 1 - lo)[:, None, None]
            right = (csy[hi] - csy[i + 1]) / (hi - i - 1)[:, None, None]
            tline, hline, ok = ht, hh, land[a] & land[b] & rowok[:-1]
        diff = (R[a] != R[b]) & ok
        de = np.sqrt(((L[a] - L[b]) ** 2).sum(-1))
        dl = np.abs(L[a][..., 0] - L[b][..., 0])
        side = np.sqrt(((left - right) ** 2).sum(-1))
        hard = diff & (de > T.max_de) & (side > HARD_SIDE_FRAC * T.max_de)
        edge = diff & (de > T.edge_de) & (side > HARD_SIDE_FRAC * T.edge_de) & ~field[a] & ~field[b]
        qa, ra = q[a], r[a]
        acc.add('pairs', qa[ok], ra[ok]); acc.add('diff_pairs', qa[diff], ra[diff])
        acc.add('de_sum', qa[diff], ra[diff], de[diff]); acc.add('dl_sum', qa[diff], ra[diff], dl[diff])
        acc.add('hard', qa[hard], ra[hard])
        acc.add('hard_tile', qa[hard & tline], ra[hard & tline]); acc.add('hard_hex', qa[hard & hline], ra[hard & hline])
        pairs.append((a, edge, tline))
        # outline of the square blobs: pairs with exactly one side in a blob (land on both sides)
        la, lbb = sqlab[a], sqlab[b]
        bd = (la != lbb) & land[a] & land[b]
        for side_lab in (la, lbb):
            sel = bd & (side_lab > 0)
            sq_bnd += np.bincount(side_lab[sel], minlength=nsq + 1)
            sq_bnd_hex += np.bincount(side_lab[sel & hline], minlength=nsq + 1)
    # straight segments
    hm = np.zeros((h, w), np.float64)
    for a, edge, _ in pairs:
        hm[a] += edge
    coh, cnt = line_coherence(hm)
    straight_px = (hm > 0) & (coh >= STRAIGHT_COH) & (cnt >= STRAIGHT_MIN)
    lab_s, n = ndimage.label(ndimage.binary_closing(straight_px, np.ones((3, 3))) | straight_px, structure=np.ones((3, 3)))
    keep = np.zeros(n + 1, bool)
    for i, sl in enumerate(ndimage.find_objects(lab_s), 1):
        if max(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start) < SEG_MIN_PX / 1.5:
            continue
        # the component as a whole must be a line too (a round blob's rim is locally straight everywhere):
        # coherence of its own position covariance, length = sqrt(12 l1) (a uniform segment of that length)
        yy, xx = np.nonzero(lab_s[sl] == i)
        l2, l1 = np.linalg.eigvalsh(np.cov(np.vstack([xx, yy]).astype(np.float64)))
        ln = float(np.sqrt(12 * max(l1, 0.)))
        if ln < SEG_MIN_PX or (l1 - l2) / max(l1 + l2, 1e-9) < STRAIGHT_COH:
            continue
        cy, cx = (sl[0].start + sl[0].stop - 1) // 2, (sl[1].start + sl[1].stop - 1) // 2
        keep[i] = True              # drawn in every strip that sees it; counted where its centre is
        if cy0 <= cy < cy1:
            sq_, sr_ = px_hex(x0 + cx, y0 + cy)
            acc.add('seg_n', [sq_], [sr_]); acc.add('seg_len', [sq_], [sr_], [ln / PX_PER_HEX])
            acc.add_max('seg_max', [sq_], [sr_], [ln / PX_PER_HEX])
    in_seg = keep[lab_s] & straight_px
    for a, edge, tline in pairs:
        st = edge & in_seg[a]       # edge pairs are core rows only
        acc.add('straight', q[a][st], r[a][st]); acc.add('straight_tile', q[a][st & tline], r[a][st & tline])
    # square blobs: credited to the hex of their centre (core rows); pixels drawn per hex
    if nsq:
        objs = ndimage.find_objects(sqlab)
        for i in np.flatnonzero(sq_small):
            sl = objs[i - 1]
            cy, cx = (sl[0].start + sl[0].stop - 1) // 2, (sl[1].start + sl[1].stop - 1) // 2
            if cy0 <= cy < cy1:
                sq_, sr_ = px_hex(x0 + cx, y0 + cy)
                acc.add('sq_n', [sq_], [sr_]); acc.add('sq_bnd', [sq_], [sr_], [sq_bnd[i]]); acc.add('sq_bnd_hex', [sq_], [sr_], [sq_bnd_hex[i]])
        pm = sq_small[sqlab] & m
        acc.add('sq_px', q[pm], r[pm])
    return L


def tile_lightness(L, land):
    """Tile-resolution mean L* and land share of a crop whose size is a multiple of 4 px."""
    h, w = land.shape
    t = lambda a: a[:h - h % 4, :w - w % 4].reshape(h // 4, 4, w // 4, 4).mean((1, 3))
    return t(L[..., 0]).astype(np.float32), t(land.astype(np.float32))


# --------------------------------------------------------------------------- tile pass
def dark_islands(Lt, land_t):
    """Dark islands on a tile-resolution L* map: (label map, list of dicts: tiles, dl, ring_share, L, centre_tile).

    Land components darker than each level of ISLAND_LEVELS (8-connected) are candidates; one is
    an island when ISLAND_MIN_TILES <= size <= ISLAND_MAX_TILES and its ring (ISLAND_RING tiles,
    land) is >= ISLAND_DL lighter on average with >= ISLAND_RING_SHARE of the ring tiles lighter
    than the island mean + ISLAND_DL / 2 (enclosed by lighter ground, not the edge of a dark
    region). The islands of all levels are merged."""
    from scipy import ndimage
    land = land_t >= .5
    union = np.zeros(Lt.shape, bool)
    box = lambda sl: (slice(max(0, sl[0].start - ISLAND_RING), min(Lt.shape[0], sl[0].stop + ISLAND_RING)),
                      slice(max(0, sl[1].start - ISLAND_RING), min(Lt.shape[1], sl[1].stop + ISLAND_RING)))
    for lev in ISLAND_LEVELS:
        lab, n = ndimage.label((Lt < lev) & land, structure=np.ones((3, 3)))
        if not n:
            continue
        size = np.bincount(lab.ravel(), minlength=n + 1)
        for i, sl in enumerate(ndimage.find_objects(lab), 1):
            if sl is None or not ISLAND_MIN_TILES <= size[i] <= ISLAND_MAX_TILES:
                continue
            bx = box(sl)
            comp = lab[bx] == i
            ring = ndimage.binary_dilation(comp, iterations=ISLAND_RING) & ~comp & land[bx]
            if ring.sum() < 4:
                continue
            lt = Lt[bx]
            lc, lr = float(lt[comp].mean()), lt[ring]
            if float(lr.mean()) - lc >= ISLAND_DL and float((lr >= lc + ISLAND_DL / 2).mean()) >= ISLAND_RING_SHARE:
                union[bx] |= comp
    out, n = ndimage.label(union, structure=np.ones((3, 3)))
    res = []
    for i, sl in enumerate(ndimage.find_objects(out), 1):
        bx = box(sl)
        comp = out[bx] == i
        ring = ndimage.binary_dilation(comp, iterations=ISLAND_RING) & ~comp & land[bx] & ~union[bx]
        lt = Lt[bx]
        lc = float(lt[comp].mean())
        lr = lt[ring] if ring.any() else np.array([lc])
        cy, cx = ndimage.center_of_mass(comp)
        res.append({'tiles': int(comp.sum()), 'dl': round(float(lr.mean()) - lc, 1), 'ring_share': round(float((lr >= lc + ISLAND_DL / 2).mean()), 2),
                    'L': round(lc, 1), 'centre_tile': [int(bx[1].start + cx), int(bx[0].start + cy)]})
    return out, res


# --------------------------------------------------------------------------- numbers / drivers
def numbers(g, islands, alarm=None):
    """Numbers of one area from its accumulator sums g (field -> float) and the islands whose centre lies in
    it; 'lit' lists the ALARM_METRICS above their vanilla alarm level (alarm: {} -> nothing is lit)."""
    land_hex = g['land_px'] / 64
    per100 = lambda v: round(v / max(1., land_hex) * 100, 2)
    hard, dp, st = g['hard'], g['diff_pairs'], g['straight']
    out = {'land_hexes': round(land_hex, 1),
           'contrast_de_mean': round(g['de_sum'] / dp, 2) if dp else None,
           'contrast_dl_mean': round(g['dl_sum'] / dp, 2) if dp else None,
           'material_pair_share': round(dp / g['pairs'], 4) if g['pairs'] else None,
           'hard_edge_hex': round(hard / PX_PER_HEX, 1), 'hard_edge_per100': per100(hard / PX_PER_HEX),
           'hard_tile_share': round(g['hard_tile'] / hard, 3) if hard else None,
           'hard_hexline_share': round(g['hard_hex'] / hard, 3) if hard else None,
           'straight_segments': int(g['seg_n']), 'straight_segment_hex': round(g['seg_len'], 1),
           'straight_segment_per100': per100(g['seg_len']), 'straight_segment_max_hex': round(g['seg_max'], 1),
           'straight_tile_share': round(g['straight_tile'] / st, 3) if st else None,
           'dark_islands': len(islands), 'dark_island_hexes': round(g['island_tiles'] / 4, 1),
           'dark_island_share': round(g['island_tiles'] / 4 / max(1., land_hex), 4),
           'dark_island_max_hexes': round(max((i['tiles'] for i in islands), default=0) / 4, 1),
           'dark_island_dl_mean': round(float(np.mean([i['dl'] for i in islands])), 1) if islands else None,
           'square_blobs': int(g['sq_n']), 'square_hexes': round(g['sq_px'] / 64, 1), 'square_per100': per100(g['sq_n']),
           'square_hexline_share': round(g['sq_bnd_hex'] / g['sq_bnd'], 3) if g['sq_bnd'] else None}
    alarm = alarms() if alarm is None else alarm
    out['lit'] = sorted(k for k, v in alarm.items() if (out[k] or 0) > v)
    return out


def summarise(acc, islands, sel=None, alarm=None):
    """Numbers of the hexes sel (bool mask over acc; all when None)."""
    sel = np.ones(acc.shape, bool) if sel is None else sel
    return numbers(reduce_sel(acc, sel), islands, alarm)


MAX_FIELDS = ('seg_max',)   # accumulators combined by max, the rest by sum


def reduce_sel(acc, sel):
    return {k: float(v[sel].max(initial=0) if k in MAX_FIELDS else v[sel].sum()) for k, v in acc.a.items()}


def merge_sums(a, b):
    """Combine the accumulator sums of two areas (multi-part windows)."""
    return b if a is None else {k: max(a[k], v) if k in MAX_FIELDS else a[k] + v for k, v in b.items()}


def summarise_groups(acc, labels, islands, alarm=None):
    """Numbers per group id of labels (int grid over acc, < 0 = none): {id: numbers}, one bincount per field."""
    lab = labels.ravel(); ok = lab >= 0; n = int(lab.max()) + 1 if ok.any() else 0
    sums = {k: np.bincount(lab[ok], weights=v.ravel()[ok].astype(np.float64), minlength=n) for k, v in acc.a.items() if k not in MAX_FIELDS}
    for k in MAX_FIELDS:
        sums[k] = np.zeros(n); np.maximum.at(sums[k], lab[ok], acc.a[k].ravel()[ok])
    by = {}
    for i in islands:
        g = int(labels[i['centre_hex'][1] - acc.r0, i['centre_hex'][0] - acc.q0])
        by.setdefault(g, []).append(i)
    return {g: numbers({k: float(v[g]) for k, v in sums.items()}, by.get(g, []), alarm) for g in range(n) if sums['land_px'][g] > 0}


def analyse_box(band, q0, q1, r0, r1, T, margin_hex=26, grid=None):
    """All metrics of a hex box. band(y0, y1, x0, x1) -> (R, land) south-up px crop; grid = (W, H) of the
    hex grid (default: the canvas). The crop carries margin_hex hexes of context (island rings, straightness
    boxes); numbers are taken over the box. Returns (numbers, per-hex heat layers of the box, islands in it, accumulator sums)."""
    import campaign_world as cw
    W, H = grid or (cw.W, cw.H)
    q0, q1, r0, r1 = max(0, q0), min(W - 1, q1), max(0, r0), min(H - 1, r1)
    mq0, mq1 = max(0, q0 - margin_hex), min(W - 1, q1 + margin_hex)
    mr0, mr1 = max(0, r0 - margin_hex), min(H - 1, r1 + margin_hex)
    x0, x1 = mq0 * PX_PER_HEX, (mq1 + 1) * PX_PER_HEX
    y0, y1 = mr0 * PX_PER_HEX, min(H * PX_PER_HEX + 4, (mr1 + 1) * PX_PER_HEX + 4)
    y1 -= (y1 - y0) % 4
    R, land = band(y0, y1, x0, x1)
    acc = Acc(mq0, mq1, mr0, mr1)
    L = pixel_pass(R, land, x0, y0, T, acc)
    Lt, lt = tile_lightness(L, land)
    del L
    ilab, isl = dark_islands(Lt, lt)
    ty, tx = np.nonzero(ilab)
    acc.add('island_tiles', *px_hex((tx + x0 // 4) * 4, (ty + y0 // 4) * 4))
    sel = np.zeros(acc.shape, bool)
    sel[r0 - mr0:r1 - mr0 + 1, q0 - mq0:q1 - mq0 + 1] = True
    inbox = []
    for i in isl:
        cq, cr = px_hex((i['centre_tile'][0] + x0 // 4) * 4, (i['centre_tile'][1] + y0 // 4) * 4)
        i['centre_hex'] = [int(cq), int(cr)]
        if q0 <= cq <= q1 and r0 <= cr <= r1:
            inbox.append(i)
    sums = reduce_sel(acc, sel)
    crop = (slice(r0 - mr0, r1 - mr0 + 1), slice(q0 - mq0, q1 - mq0 + 1))
    return numbers(sums, inbox), {k: v[crop] for k, v in heat_layers(acc).items()}, inbox, sums


def class_line_hexes(eco_bio, bnames, a, b, reach=1):
    """Hex mask (H, W) of the hexes holding a tile within `reach` tiles (chessboard) of a class-a tile that touches a
    class-b tile (4-neighbours). eco_bio: tile grid south-up (TH, TW) codes of bnames; a, b: class name tuples."""
    from scipy import ndimage
    import campaign_world as cw
    A = np.isin(eco_bio, [bnames.index(n) for n in a if n in bnames])
    B = np.isin(eco_bio, [bnames.index(n) for n in b if n in bnames])
    nb = ndimage.binary_dilation(B, np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool))
    line = ndimage.binary_dilation(A & nb, np.ones((2 * reach + 1, 2 * reach + 1), bool))
    ty, tx = np.nonzero(line)
    q, r = cw.tile_hex(tx, ty)
    H, W = cw.H, cw.W
    m = np.zeros((H, W), bool)
    ok = (q >= 0) & (q < W) & (r >= 0) & (r < H)
    m[r[ok], q[ok]] = True
    return m


def line_numbers(band, sel, T, block=96, margin_hex=26):
    """Hard-edge and straight-segment numbers of the hexes sel (H, W bool, e.g. class_line_hexes) only: the map is
    cut into block x block hex boxes, each box holding a sel hex is measured as analyse_box does (pixel_pass with
    margin_hex of context) and its sel hexes summed (merge_sums). Dark islands are not measured (a thin line has no
    rings). band(y0, y1, x0, x1) -> (R, land) as BlendSource.band. Returns numbers() with alarm {} plus 'blocks'."""
    H, W = sel.shape
    sums = None; nb = 0
    for r0 in range(0, H, block):
        for q0 in range(0, W, block):
            q1, r1 = min(W - 1, q0 + block - 1), min(H - 1, r0 + block - 1)
            if not sel[r0:r1 + 1, q0:q1 + 1].any():
                continue
            mq0, mq1 = max(0, q0 - margin_hex), min(W - 1, q1 + margin_hex)
            mr0, mr1 = max(0, r0 - margin_hex), min(H - 1, r1 + margin_hex)
            x0, x1 = mq0 * PX_PER_HEX, (mq1 + 1) * PX_PER_HEX
            y0, y1 = mr0 * PX_PER_HEX, min(H * PX_PER_HEX + 4, (mr1 + 1) * PX_PER_HEX + 4)
            y1 -= (y1 - y0) % 4
            R, land = band(y0, y1, x0, x1)
            acc = Acc(mq0, mq1, mr0, mr1)
            pixel_pass(R, land, x0, y0, T, acc)
            del R, land
            s = np.zeros(acc.shape, bool)
            s[r0 - mr0:r1 - mr0 + 1, q0 - mq0:q1 - mq0 + 1] = sel[r0:r1 + 1, q0:q1 + 1]
            sums = merge_sums(sums, reduce_sel(acc, s)); nb += 1
    if sums is None:
        return {'land_hexes': 0, 'hard_edge_per100': 0., 'straight_segment_max_hex': 0., 'blocks': 0, 'lit': []}
    out = numbers(sums, [], alarm={}); out['blocks'] = nb
    return out


class Recorder(Acc):
    """An Acc that only records: every add as (key, flat cells, float32 values), every add_max as (key, flat
    cells, values), for another Acc to replay in the same order, value for value (no arrays of its own)."""
    def __init__(self, q0, q1, r0, r1):
        self.q0, self.q1, self.r0, self.r1 = q0, q1, r0, r1
        self.shape = (r1 - r0 + 1, q1 - q0 + 1)
        self.log = []

    def _idx(self, q, r):
        sel = (q >= self.q0) & (q <= self.q1) & (r >= self.r0) & (r <= self.r1)
        return sel, (r[sel] - self.r0) * self.shape[1] + (q[sel] - self.q0)

    def add_max(self, key, q, r, w):
        q, r, w = np.asarray(q), np.asarray(r), np.asarray(w, np.float32)
        sel, idx = self._idx(q, r)
        self.log.append(('max', key, idx.astype(np.int32), w[sel]))

    def add(self, key, q, r, w=None):
        q, r = np.asarray(q), np.asarray(r)
        sel, idx = self._idx(q, r)
        if not sel.any():
            return
        v = np.bincount(idx, weights=None if w is None else np.asarray(w, np.float64)[sel],
                        minlength=self.shape[0] * self.shape[1]).astype(np.float32)
        nz = np.flatnonzero(v)
        self.log.append(('add', key, nz.astype(np.int32), v[nz]))


def replay(acc, log):
    """Apply a Recorder's adds to acc in order: a[key] += v leaves cells with v == 0 unchanged, so adding the
    non-zero cells gives the float32 sums Acc.add gives, bit for bit."""
    for kind, key, idx, v in log:
        if kind == 'add':
            flat = acc.a[key].reshape(-1)
            flat[idx] += v
        else:
            np.maximum.at(acc.a[key].reshape(-1), idx, v)


def _strip_rows(src, y, H):
    return max(0, y - STRIP_MARGIN), min(H, y + STRIP_PX + STRIP_MARGIN)


def _strip_key(src, T, ya, yb):
    """Cache key of one strip: its input bytes (blend and LF rows), the tone table and threshold, this module."""
    import hashlib
    h = hashlib.sha256(Path(__file__).read_bytes())
    h.update(repr((ya, yb, src.w, src.h, STRIP_PX, STRIP_MARGIN, T.max_de, T.edge_de)).encode())
    h.update(np.ascontiguousarray(T.lab).tobytes())
    for name in ('blend', 'lf'):
        if name in src.src:
            h.update(src.rows(name, ya, yb).tobytes())
    if src.land_hex is not None:
        h.update(np.ascontiguousarray(src.land_hex).tobytes())
    return h.hexdigest()[:24]


def strip_pass(src, T, y, cache=None):
    """One whole-map strip: (recorded adds, tile L*, tile land share), from the cache when its inputs match."""
    import pickle
    H = src.h - src.h % 4
    ya, yb = _strip_rows(src, y, H)
    path = None
    if cache:
        path = Path(cache) / f'strip-{y}-{_strip_key(src, T, ya, yb)}.pkl'
        if path.is_file():
            return pickle.loads(path.read_bytes())
    W, Hh = src.hexes
    rec = Recorder(0, W - 1, 0, Hh - 1)
    R, land = src.band(ya, yb)
    c0, c1 = y - ya, min(H, y + STRIP_PX) - ya
    L = pixel_pass(R, land, 0, ya, T, rec, core=(c0, c1))
    a, b = tile_lightness(L[c0:c1], land[c0:c1])
    out = (rec.log, a, b)
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
        tmp.write_bytes(pickle.dumps(out, protocol=pickle.HIGHEST_PROTOCOL))
        os.replace(tmp, path)
    return out


def analyse_map(src, T, log=print, workers=1, cache=None):
    """Whole-map accumulators and islands of a BlendSource, in pixel strips of STRIP_PX rows with
    STRIP_MARGIN rows of context (a few GB per strip on the 2048 canvas: take a pipeline lock slot first).
    Strips run on `workers` processes and are kept in `cache` (a folder) by their input bytes; each strip's adds
    are replayed in strip order, so the sums equal the one-process pass bit for bit."""
    import time
    W, Hh = src.hexes
    acc = Acc(0, W - 1, 0, Hh - 1)
    H = src.h - src.h % 4
    Lt = np.zeros((H // 4, src.w // 4), np.float32); lt = np.zeros_like(Lt)
    t0 = time.time()
    ys = list(range(0, H, STRIP_PX))
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing as mp
        ex = ProcessPoolExecutor(workers, mp_context=mp.get_context('spawn'))
        res = [ex.submit(strip_pass, src, T, y, cache) for y in ys]
        get = lambda k: res[k].result()
    else:
        ex = None
        get = lambda k: strip_pass(src, T, ys[k], cache)
    try:
        for k, y in enumerate(ys):
            rec, a, b = get(k)
            replay(acc, rec)
            Lt[y // 4:y // 4 + len(a)], lt[y // 4:y // 4 + len(a)] = a, b
            log(f'  rows {y:5d}-{min(H, y + STRIP_PX):5d} [{time.time() - t0:5.0f}s]')
    finally:
        if ex is not None:
            ex.shutdown()
    ilab, isl = dark_islands(Lt, lt)
    ty, tx = np.nonzero(ilab)
    acc.add('island_tiles', *px_hex(tx * 4, ty * 4))
    for i in isl:
        i['centre_hex'] = [int(v) for v in px_hex(i['centre_tile'][0] * 4, i['centre_tile'][1] * 4)]
    return acc, isl


def islands_in(islands, q0, q1, r0, r1):
    return [i for i in islands if q0 <= i['centre_hex'][0] <= q1 and r0 <= i['centre_hex'][1] <= r1]


def heat_layers(acc):
    """Per-hex heat layers: L (mean L*), straight (straight-segment hard pairs), hard (all hard pairs),
    island (island tile share of the hex), square (square-blob pixel share), land."""
    a = acc.a
    land = a['land_px']
    with np.errstate(invalid='ignore', divide='ignore'):
        L = np.where(land > 0, a['L_sum'] / np.maximum(land, 1), np.nan)
        sq = np.where(land > 0, a['sq_px'] / np.maximum(land, 1), 0)
    return {'L': L.astype(np.float32), 'straight': a['straight'].copy(), 'hard': a['hard'].copy(),
            'island': np.clip(a['island_tiles'] / 4, 0, 1).astype(np.float32), 'square': sq.astype(np.float32), 'land': land > 0}


def heat_rgb(maps, scale=4):
    """North-up RGB heatmap: grey = ground L*, orange = hard edges, red = straight hard segments,
    blue = dark islands, magenta = farm / wetland squares, dark blue-grey = water."""
    L = np.nan_to_num(maps['L'], nan=0.)
    g = np.clip((L - 20) / 60, 0, 1) * 150 + 40
    rgb = np.stack([g, g, g], -1)
    rgb[~maps['land']] = (28, 38, 52)
    mix = lambda rgb, wgt, col: rgb * (1 - wgt[..., None]) + np.array(col, np.float64) * wgt[..., None]
    rgb = mix(rgb, .75 * maps['island'], (40, 110, 255))
    rgb = mix(rgb, np.clip(maps['square'] * 3, 0, .85), (255, 0, 220))
    rgb = mix(rgb, .55 * np.clip(maps['hard'] / 8, 0, 1), (255, 170, 0))
    rgb = mix(rgb, np.clip(maps['straight'] / 6, 0, 1), (255, 20, 20))
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)[::-1]
    return rgb.repeat(scale, 0).repeat(scale, 1) if scale > 1 else rgb


# --------------------------------------------------------------------------- fragmentation per landform
# [GROUND MOTTLE, 2026-11-01] the landform mixes' pepper of far-apart tones (Yalong: six or seven materials in
# 300-500 patches per group; the user's shots). Measured per landform on the pixels of the tiles it dominates
# (landform tiles dom, domw > FRAG_DOMW): E = dE76 summed over the 4-neighbour pairs of different slots inside
# the class, per hex (64 px); patches = same-slot 4-connected components inside the class, counted per
# 100 hexes, median area in hexes. Strips of FRAG_STRIP rows: a patch crossing a strip edge counts once per
# strip (deterministic, the same for every product compared). stage_invariants ground_fragmentation: no
# landform worse than the baseline product (analysis/ground-fragmentation-baseline.json) outside the rebuilt area
# (regions whose layout changed since the baseline, + REBUILT_BAND hexes: analysis/ground-fragmentation-rebuilt.npz);
# inside it no landform worse than the product's own value outside (amendment 2 of the criteria, 2026-11-01).
FRAG_DOMW = .5
FRAG_STRIP = 2048
FRAG_MIN_HEX = 50       # landforms with fewer hexes are not reported
REBUILT_BAND = 6        # hexes of band round the rebuilt regions


def fragmentation(src, dom, domw, T, landforms, strip=FRAG_STRIP, groups=None):
    """{landform: {hexes, E, n100, med_hex}} of the blend of src (BlendSource) on the landform tile field
    (dom = 1 + index into landforms, domw; south-up tile grid). groups: optional uint8 tile array (0 .. k-1, e.g. the
    rebuilt-area mask); the result is then {group: {landform: ...}}, pairs and patches never crossing groups."""
    from scipy import ndimage
    nlf0 = len(landforms)
    ng = 1 if groups is None else int(np.max(groups)) + 1
    nlf = nlf0 * ng
    E = np.zeros(nlf + 1); px = np.zeros(nlf + 1, np.int64); areas = [[] for _ in range(nlf + 1)]
    th, tw = dom.shape
    H, W = min(src.h, 4 * th), min(src.w, 4 * tw)
    cls_t = np.where(np.asarray(domw, np.float32) > FRAG_DOMW, dom, 0).astype(np.uint16)
    if groups is not None:
        cls_t = np.where(cls_t > 0, cls_t + nlf0 * np.asarray(groups, np.uint16), 0).astype(np.uint16)
    for y0 in range(0, H, strip):
        y1 = min(H, y0 + strip)
        R = src.rows('blend', y0, y1)[:, :W, 0]
        C = np.repeat(np.repeat(cls_t[y0 // 4:(y1 + 3) // 4, :W // 4], 4, 0), 4, 1)[:y1 - y0, :W]
        px += np.bincount(C.ravel(), minlength=nlf + 1)
        for a, b, ca, cb in ((R[:, 1:], R[:, :-1], C[:, 1:], C[:, :-1]), (R[1:], R[:-1], C[1:], C[:-1])):
            d = (a != b) & (ca == cb) & (ca > 0)
            E += np.bincount(ca[d], weights=np.linalg.norm(T.lab[a[d]] - T.lab[b[d]], axis=1), minlength=nlf + 1)
        K = np.where(C > 0, C.astype(np.int32) * 256 + R, 0).astype(np.int32)
        for k, sl in enumerate(ndimage.find_objects(K), 1):
            if sl is None:
                continue
            lab, n = ndimage.label(K[sl] == k)
            if n:
                areas[k // 256].append(np.bincount(lab.ravel())[1:])
        del R, C, K
    out = {}
    for g in range(ng):
        og = out.setdefault(g, {})
        for j, c in enumerate(landforms, 1):
            i = j + nlf0 * g
            hexes = px[i] / 64.
            if hexes < FRAG_MIN_HEX:
                continue
            a = np.concatenate(areas[i]) / 64.
            og[c] = {'hexes': round(float(hexes), 1), 'E': round(float(E[i] / hexes), 3),
                     'n100': round(float(len(a) / hexes * 100), 2), 'med_hex': round(float(np.median(a)), 4)}
    return out[0] if groups is None else out


# CAIME flat-top neighbours by column parity (oss_rebuild_layout.DIRS / hex_dilate: that module installs read hooks
# on import, so the 12 lines are repeated here)
_HEX_DIRS = (((0, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0)),
             ((0, 1), (1, 1), (1, 0), (0, -1), (-1, 0), (-1, 1)))


def hex_dilate(mask, steps):
    out = mask.copy()
    H, W = mask.shape
    odd = (np.arange(W) % 2 == 1)[None, :]
    for _ in range(steps):
        cur = out.copy()
        for par in (0, 1):
            for dq, dr in _HEX_DIRS[par]:
                sh = np.zeros_like(cur)
                r0, r1 = max(0, -dr), min(H, H - dr); c0, c1 = max(0, -dq), min(W, W - dq)
                sh[r0:r1, c0:c1] = cur[r0 + dr:r1 + dr, c0 + dq:c1 + dq]
                out |= sh & (odd if par else ~odd)
    return out


def rebuilt_hexes(map_a, map_b, band=REBUILT_BAND):
    """Hexes whose region (by key: land and sea lists) differs between two map_hex.read maps, dilated by band."""
    import map_layers as ml
    def keys(m):
        names = np.array([''] + list(m['land_regions']) + list(m['sea_regions']), dtype=object)
        return names[ml.get(m['cells'], 'region').astype(np.int64)]
    return hex_dilate(keys(map_a) != keys(map_b), band)


def hex_to_tiles(hexmask, tile_shape):
    """uint8 tile array (south-up) of a hex mask (H, W rows r, columns q): tile (tx, ty) -> its hex (campaign_world.tile_hex)."""
    th, tw = tile_shape
    H, W = hexmask.shape
    tx = np.arange(tw); ty = np.arange(th)
    q = np.clip(tx // 2, 0, W - 1)
    r = np.clip((ty[:, None] - (q % 2)[None, :]) // 2, 0, H - 1)
    return hexmask[r, q[None, :]].astype(np.uint8)
