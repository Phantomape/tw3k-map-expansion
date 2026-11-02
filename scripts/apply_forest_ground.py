"""Forest ground types (map.hex) where the frontier forest fill plants vanilla-density forest.

Vanilla ties tree density to the hex ground type (dlc07 tree list per ground, offline survey
2026-10-23): light_forest 0.82 trees/hex, hilly_light_forest 0.81, dense_forest 0.85 (99% of
it impassable), against grassland 0.40, hilly_grassland 0.13, plains (none in the native map's
playable land). frontier_forest.py plants forest by the real biome; this sets the matching
ground on the same hexes so movement/battle ground follows the trees:
  passable grassland/plains          -> light_forest
  passable hilly_grassland/_plains   -> hilly_light_forest
  impassable grassland/plains/desert -> dense_forest
on land hexes of real forest biome (taiga, conifer, temperate forest, tropical moist,
mangrove; not tropical dry) outside the native map (the revealed filler, revealed_filler.revealed_hex, counts
as outside) and the Western Regions band, within
frontier_forest.FILL_REACH hexes of playable land, at least TOWN_CLEAR hexes from towns,
off settlement slots and roads. Dry run by default (counts only); --apply rewrites
work/final-map/map.hex after a backup in backups/. A gameplay change (movement, battle
ground): apply only when the batch is accepted.

  python apply_forest_ground.py [--apply]

FILL FADE: the reach is frontier_forest.fill_mask (soft, dithered fade 30-50 hexes from playable
land, the same mask as the tree fill), no longer the hard FILL_REACH line. Each --apply records the
hexes this step owns (analysis/forest-ground-hexes.npz: hex, ground before, ground set). A later run
clears its own forest ground that lies outside the current mask: a recorded hex goes back to its
recorded ground only while it still holds the ground this step set (later steps that changed it win).
For a map forest-grounded before the record existed (main, 2d77d95f), --previous-backup names the
map.hex backup taken right before that first run: the previous zone is recomputed with the old hard
rule, and a hex outside the new mask is reverted only where the backup holds a source ground whose
CHANGE target is exactly the current ground (anything changed since, e.g. landmarks or climate, stays).

  python apply_forest_ground.py --previous-backup backups/map-before-forest-ground-<first>.hex [--apply]
"""
import argparse
from pathlib import Path
import json
import shutil
import time

import numpy as np

import campaign_world as cw
import map_hex
from fastops import grow_steps
from frontier_forest import FILL_REACH, fill_mask
from hex_components import hex_dilate
from revealed_filler import revealed_hex
import biome_profiles as _bp

_K = _bp.consts('apply_forest_ground')   # values: biome_profiles.CONSTS['apply_forest_ground']

FOREST_BIOMES = _bp.biome_set('forest_ground')   # taiga, conifer, forest, tropical, mangrove (no tropical_dry, no temperate)
TOWN_CLEAR = _K['TOWN_CLEAR']
CHANGE = _K['CHANGE']


def compute(m, record=None, previous=None):
    """Forest ground for map dict m, in memory: {'new': ground index per hex, 'g': ground before, 'changed',
    'revert', 'counts', 'old_zone', 'bg', 'zone', 'imp', 'G', 'H', 'W'}. record: this step's own record
    (q, r, src, dst arrays of analysis/forest-ground-hexes.npz) or None; previous: the --previous-backup map dict
    (FILL FADE for a map grounded before the record existed) or None. With neither (the map bake, which starts
    from the forest-free layout base) nothing is reverted. Reads the strategic-ink biome cache and the design
    towns; writes nothing."""
    import strategic_ink
    c = m['cells']
    H, W = c.shape[:2]
    G = m['land_ground_types']
    g = (c[..., 5].astype(np.int32) >> 4) + ((c[..., 6].astype(np.int32) & 7) << 4) - 1
    land = (g >= 0) & (g < len(G))
    imp = (c[..., 2] & 8) > 0
    slot = ((c[..., 2] >> 4) > 0) | ((c[..., 3] & 1) > 0)
    road = ((c[..., 3].astype(np.int32) >> 1) & 63) > 0
    names = m['land_regions'] + m['sea_regions']
    ids = map_hex.region_ids(c)
    npl = np.array(['non_playable' in n for n in names] + [True])[np.where((ids >= 0) & (ids < len(names)), ids, len(names))]
    reach = fill_mask(land & ~npl)   # FILL FADE (was grow_steps(land & ~npl, FILL_REACH) <= FILL_REACH)
    plan = json.loads(cw.DESIGN_PATH.read_text(encoding='utf-8'))
    towns = np.zeros((H, W), bool)
    for t in plan['regions']:
        towns[t['town'][1], t['town'][0]] = True
    town_seed = towns | ((c[..., 2] >> 4) == 1)
    near_town = town_seed.copy()               # hex distance < TOWN_CLEAR (fastops.grow_steps is a 4-neighbour
    for _ in range(TOWN_CLEAR - 1):            # square-grid growth: it missed 2 of the 6 hex neighbours a step)
        near_town = hex_dilate(near_town)
    biome, bnames, _, _ = strategic_ink.load()
    forest = np.isin(biome, [bnames.index(b) for b in FOREST_BIOMES if b in bnames])
    rr, qq = np.mgrid[:H, :W]
    nat = np.zeros((H, W), bool)
    nat[cw.NATIVE_HEX_SLICE] = True
    nat_rect = nat.copy()              # the first (hard-reach) run's rule: the whole native rectangle
    nat &= ~revealed_hex(c, names)     # the revealed filler (vanilla non-playable land now playable) is canvas
    west = _bp.west_band(qq, rr, cw)   # Western Regions band: q < PAD_LEFT at the native rows
    zone = land & forest & reach & ~nat & ~west & ~near_town & ~slot & ~road
    new = g.copy()
    counts = {}
    for hard, table in CHANGE.items():
        for src, dst in table.items():
            sel = zone & (imp == hard) & (g == G.index(src))
            new[sel] = G.index(dst)
            counts[f"{'impassable' if hard else 'passable'} {src} -> {dst}"] = int(sel.sum())
    # --- FILL FADE --- clear this step's own forest ground outside the current zone.
    revert = np.zeros((H, W), bool); restore = g.copy(); old_zone = bg = None
    if record is not None:
        oq, orr, osrc, odst = record
        sel = (g[orr, oq] == odst) & ~zone[orr, oq]
        revert[orr[sel], oq[sel]] = True; restore[orr[sel], oq[sel]] = osrc[sel]
    elif previous is not None:
        bc = previous['cells']
        assert bc.shape == c.shape and list(previous['land_ground_types']) == list(G)
        bg = (bc[..., 5].astype(np.int32) >> 4) + ((bc[..., 6].astype(np.int32) & 7) << 4) - 1
        old_reach = grow_steps(land & ~npl, FILL_REACH) <= FILL_REACH
        old_near_town = grow_steps(town_seed, TOWN_CLEAR - 1) < TOWN_CLEAR   # the first run's rule, as it ran
        old_zone = land & forest & old_reach & ~nat_rect & ~west & ~old_near_town & ~slot & ~road
        for hard, table in CHANGE.items():
            for src, dst in table.items():
                sel = old_zone & ~zone & (imp == hard) & (bg == G.index(src)) & (g == G.index(dst))
                revert |= sel; restore[sel] = G.index(src)
    new[revert] = restore[revert]
    counts['reverted outside the faded reach'] = int(revert.sum())
    # --- end FILL FADE
    return {'new': new, 'g': g, 'changed': new != g, 'revert': revert, 'counts': counts, 'old_zone': old_zone, 'bg': bg,
            'zone': zone, 'imp': imp, 'G': G, 'H': H, 'W': W}


def set_ground(c, changed, new):
    """Cells c (copied) with the ground index new written where changed (bytes 5/6 only)."""
    v = (new + 1).astype(np.int32)
    c = c.copy()
    c[..., 5] = np.where(changed, (c[..., 5] & 15) | ((v & 15) << 4), c[..., 5]).astype(np.uint8)
    c[..., 6] = np.where(changed, (c[..., 6] & 0xF8) | ((v >> 4) & 7), c[..., 6]).astype(np.uint8)
    return c


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--previous-backup', help='FILL FADE: map.hex backup from right before the first (hard-reach) run')
    args = ap.parse_args()
    A = cw.WORKSPACE
    path = A / 'work/final-map/map.hex'
    m = map_hex.read(path)
    c = m['cells']
    owned_path = A / 'analysis/forest-ground-hexes.npz'
    record = previous = None; source = 'none'
    if owned_path.exists():
        o = np.load(owned_path)
        record = (o['q'], o['r'], o['src'], o['dst']); source = str(owned_path)
    elif args.previous_backup:
        bp = Path(args.previous_backup); bp = bp if bp.is_absolute() else A / bp
        previous = map_hex.read(bp)
        source = str(bp)
    res = compute(m, record=record, previous=previous)
    new, g, changed, revert, counts, old_zone, bg, zone, imp, G, H, W = (res[k] for k in (
        'new', 'g', 'changed', 'revert', 'counts', 'old_zone', 'bg', 'zone', 'imp', 'G', 'H', 'W'))
    report = {'map': str(path), 'changed_hexes': int(changed.sum()), 'by_change': counts, 'applied': args.apply,
              'reach': 'frontier_forest.fill_mask (FILL_FADE)', 'previous_record': source}
    print(json.dumps(report, indent=1))
    if args.apply and changed.any():
        bak = A / 'backups' / f'map-before-forest-ground-{time.strftime("%Y%m%d-%H%M%S")}.hex'
        bak.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, bak)
        m['cells'] = set_ground(c, changed, new)
        map_hex.write(m, path)
        import region_connectivity; region_connectivity.check_map(path)   # land regions stay whole (v17 crash); [to be absorbed by the map bake]
        report['backup'] = str(bak)
        # --- FILL FADE --- record the hexes this step owns (set now, or still holding what it set before)
        own = (new != g) & ~revert
        parts = []
        if owned_path.exists():
            o = np.load(owned_path); keep = new[o['r'], o['q']] == o['dst']
            parts.append((o['q'][keep], o['r'][keep], o['src'][keep], o['dst'][keep]))
        elif old_zone is not None:
            keep_old = np.zeros((H, W), bool)
            for hard, table in CHANGE.items():
                for src, dst in table.items():
                    keep_old |= old_zone & zone & (imp == hard) & (bg == G.index(src)) & (new == G.index(dst))
            keep_old &= ~own
            rr_, qq_ = np.nonzero(keep_old); parts.append((qq_, rr_, bg[keep_old], new[keep_old]))
        rr_, qq_ = np.nonzero(own); parts.append((qq_, rr_, g[own], new[own]))
        np.savez_compressed(owned_path, **{k: np.concatenate([x[i] for x in parts]).astype(np.int32) for i, k in enumerate(('q', 'r', 'src', 'dst'))})
        report['owned_hexes'] = int(sum(len(x[0]) for x in parts))
    (A / 'analysis/forest-ground.json').write_text(json.dumps(report, indent=1), encoding='utf-8')

if __name__ == '__main__':
    main()
