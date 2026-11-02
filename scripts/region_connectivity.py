"""Land regions must be one connected piece of land hexes, islands apart.

v17 crashed at load in the region-border code (exe+0x166D0A8): the Western Regions rivers had cut land
regions into disconnected pieces, and apply_rivers only checked the SDK region-count limit. Vanilla dlc07
never lets a river split a region; its only multi-piece regions are islands across the sea (Kuaiji: 5).

Rule (per land region, its land hexes = kind != 1 with that region id):
  * pieces = connected components under the game's hex adjacency (map_hex.BORDER_SIDES, the neighbour
    rule of the byte-15 border masks);
  * the largest piece is the region; every other piece must be an island: none of its outside neighbours
    is a land hex of another region or a river hex. Sea hexes of a region whose name contains '_riv_'
    (vanilla 3k_main_riv_*, the frontier tw3k_riv_sea_*) are rivers, so a piece a river cuts off fails;
    every other sea region (open sea, lakes) counts as sea, and off-map sides are ignored;
  * allowance: a region may keep as many non-island extra pieces as vanilla has for its name (dlc07: none)
    or, while they are being fixed, as KNOWN (region_connectivity_known.json) lists for it.

KNOWN: the 16th 2048 candidate (loaded in game) already had 43 such regions, 136 non-island extra pieces
(Indian / Southeast Asian rivers, the Mongolian steppe, exclaves inside other regions, the non-playable
filler); v17 added 43 more river cuts in the Tarim, Ferghana and Bactria regions. The list only shrinks.
Kept by the user's decision of 2026-10-26 (analysis/decisions.md): never add a region or raise a count; take
entries out as the river spec and the map bake (analysis/pipeline-bake-design.md 3.5, 0') fix the pieces at
the source, until the list is empty.

region_connectivity_errors(cells, names[, vanilla, known]) -> list of offending regions (empty = pass).
check_map(path) -> raises ValueError naming them (one call for every map.hex writer, right after it writes).
"""
import json
from pathlib import Path

import numpy as np

import map_hex

RIVER_TAG = '_riv_'
KNOWN_FILE = Path(__file__).resolve().with_name('region_connectivity_known.json')
_cache = {}


def _neighbours(h, w):
    """Yield (r, q, r2, q2) index arrays of every in-map hex pair across one side (map_hex.BORDER_SIDES)."""
    odd = np.arange(w) % 2 == 1
    for _, dqe, dre, dqo, dro in map_hex.BORDER_SIDES:
        for par, dq, dr in ((False, dqe, dre), (True, dqo, dro)):
            q = np.flatnonzero(odd == par)
            q2 = q + dq
            okq = (q2 >= 0) & (q2 < w)
            q, q2 = q[okq], q2[okq]
            r = np.arange(max(0, -dr), min(h, h - dr))
            yield r[:, None], q[None, :], (r + dr)[:, None], q2[None, :]


def pieces(cells, names):
    """Per land region with several pieces: {region index: [(size, (q, r), island), ...]} largest first.
    island: the piece touches no land hex of another region and no river hex."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    h, w = cells.shape[:2]
    ids = map_hex.region_ids(cells)
    land = (cells[..., 0] & 3) != 1
    river_ids = np.array([RIVER_TAG in n for n in names] + [False])
    river = ~land & river_ids[np.where((ids >= 0) & (ids < len(names)), ids, len(names))]
    flat = np.arange(h * w).reshape(h, w)
    a, b = [], []
    for r, q, r2, q2 in _neighbours(h, w):
        same = land[r, q] & land[r2, q2] & (ids[r, q] == ids[r2, q2])
        a.append(flat[r, q][same]); b.append(flat[r2, q2][same])
    a, b = np.concatenate(a), np.concatenate(b)
    n, lab = connected_components(coo_matrix((np.ones(len(a), np.int8), (a, b)), shape=(h * w, h * w)), directed=False)
    lab = lab.reshape(h, w)
    # components of land hexes only, each with its region id
    comp_region = np.full(n, -1, np.int64); comp_region[lab[land]] = ids[land]
    size = np.bincount(lab[land], minlength=n)
    multi = np.bincount(comp_region[size > 0][comp_region[size > 0] >= 0], minlength=len(names)) > 1
    if not multi.any():
        return {}
    blocked = np.zeros(n, bool)   # touches another region's land or a river
    for r, q, r2, q2 in _neighbours(h, w):
        src = land[r, q] & multi[np.clip(ids[r, q], 0, len(names) - 1)]
        other = (land[r2, q2] & (ids[r2, q2] != ids[r, q])) | river[r2, q2]
        blocked[lab[r, q][src & other]] = True
    out = {}
    sel = land & multi[np.clip(ids, 0, len(names) - 1)]
    rr, qq = np.nonzero(sel)
    first = {}
    for c, r_, q_ in zip(lab[rr, qq].tolist(), rr.tolist(), qq.tolist()):
        first.setdefault(c, (q_, r_))
    for c, at in first.items():
        out.setdefault(int(comp_region[c]), []).append((int(size[c]), at, not blocked[c]))
    for v in out.values():
        v.sort(key=lambda t: -t[0])
    return out


def _non_island_extra(parts):
    return sum(1 for s, at, island in parts[1:] if not island)


def _vanilla_allowance(vanilla):
    if vanilla is None:
        return {}
    key = str(vanilla) if isinstance(vanilla, (str, Path)) else id(vanilla)
    if key not in _cache:
        if isinstance(vanilla, (str, Path)):
            m = map_hex.read(Path(vanilla)); vanilla = (m['cells'], m['land_regions'] + m['sea_regions'])
        _cache[key] = {vanilla[1][i]: _non_island_extra(p) for i, p in pieces(*vanilla).items()}
    return _cache[key]


def default_vanilla():
    import campaign_world
    return campaign_world.VANILLA_MAP


def load_known(path=KNOWN_FILE):
    return json.loads(Path(path).read_text(encoding='utf-8'))['regions'] if path else {}


def region_connectivity_errors(cells, names, vanilla=None, known=None):
    """Offending land regions (empty list = pass): a region with more non-island extra pieces than its
    allowance, max(vanilla's for its name, known.get(name)). vanilla: None, a map.hex path or a
    (cells, names) pair; known: {region name: allowed non-island extra pieces} (load_known())."""
    van, known = _vanilla_allowance(vanilla), known or {}
    errors = []
    for i, parts in sorted(pieces(cells, names).items()):
        allow = max(van.get(names[i], 0), known.get(names[i], 0))
        if _non_island_extra(parts) > allow:
            errors.append({'region': names[i], 'pieces': len(parts), 'sizes': [s for s, _, _ in parts],
                           'main_hex': list(parts[0][1]), 'allowance': allow,
                           'stray': [{'hexes': s, 'hex': list(at), 'island': island} for s, at, island in parts[1:]]})
    return errors


def describe(errors, limit=None):
    """One line per offending region: name, piece sizes, one hex [q, r] (rows south-up) per stray piece."""
    return [f"{e['region']}: pieces {e['sizes'][:12]}{' ...' if len(e['sizes']) > 12 else ''} (allowed extra "
            f"{e['allowance']}), non-island stray at {[s['hex'] for s in e['stray'] if not s['island']][:8]}"
            for e in errors[:limit]]


def check_map(path, vanilla='default', known=KNOWN_FILE):
    """Raise ValueError naming the split regions of map.hex at path: the one call for every map.hex writer,
    right after it writes (vanilla dlc07 and region_connectivity_known.json as the allowance)."""
    m = map_hex.read(Path(path))
    errors = region_connectivity_errors(m['cells'], m['land_regions'] + m['sea_regions'],
                                        default_vanilla() if vanilla == 'default' else vanilla, load_known(known))
    if errors:
        raise ValueError(f'{path}: {len(errors)} land region(s) split into non-island pieces (a river or another '
                         f'region cuts them off; v17 crashed at load in the region-border code):\n  ' + '\n  '.join(describe(errors)))
    return m
