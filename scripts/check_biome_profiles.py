"""Frozen-snapshot check of the biome profile table (biome_profiles.py).

biome_profiles.snapshot() lists every value a pipeline consumer reads from the table (as repr: value,
type and order). analysis/biome-profiles-frozen.json holds that list as it was when the migration
(steps 1-4, byte-identical to main 38c7ffe6) was made; this check fails on any difference, so every
later edit of a pipeline-facing value is deliberate: rerun with --refreeze in the commit that changes
it, and the diff of the frozen file shows the change in review. Audit-only fields (AUDIT, FAMILIES),
karez lines and river_oasis are outside the snapshot. It also checks the table's own consistency
(ranks, set flags, landform order) and that analysis/biome-profiles.json is current.

  python check_biome_profiles.py              # check (exit 1 on a difference)
  python check_biome_profiles.py --refreeze   # write the frozen file and the review JSON from the table
"""
import json
import sys
from pathlib import Path

import biome_profiles as bp

ROOT = Path(__file__).resolve().parent.parent
# [GROUND MOTTLE] TW3K_GROUND_RECIPE=v23 (biome_profiles.ground_recipe, the old ground recipe kept as a named version)
# is checked against its own frozen copy and review JSON, which hold the table as it was before the change
_SUFFIX = '' if bp.ground_recipe() == bp.GROUND_RECIPE_DEFAULT else f'-{bp.ground_recipe()}'
FROZEN = ROOT / f'analysis/biome-profiles-frozen{_SUFFIX}.json'
REVIEW = ROOT / f'analysis/biome-profiles{_SUFFIX}.json'


def consistency():
    bad = []
    if tuple(bp.BIOMES) != bp.BIOME_CODES:
        bad.append('BIOMES order != BIOME_CODES')
    for b, rec in bp.BIOMES.items():
        bad += [f'{b}: unknown flag {f}' for f in rec['sets'] if f not in bp.SETS]
    for f in bp.SETS:
        ranks = sorted(bp.BIOMES[b]['sets'][f] for b in bp.BIOMES if f in bp.BIOMES[b]['sets'])
        if ranks != list(range(len(ranks))):
            bad.append(f'{f}: ranks {ranks} are not 0..n-1')
    if sorted(bp.MIX_ORDER) != sorted(bp.LANDFORMS) or set(bp.AUDIT) != set(bp.LANDFORMS):
        bad.append('MIX_ORDER / AUDIT do not name the landforms')
    for k, v in bp.AUDIT.items():
        for fam, lohi in (v['bounds'] or {}).items():
            if fam not in bp.FAMILIES or not (0 <= lohi[0] <= lohi[1] <= 1):
                bad.append(f'AUDIT {k}: bad bound {fam} {lohi}')
    seeds = bp.REGIONS['oasis_seeds']['seeds']
    bad += [f'oasis seed {n}: no disc {s["at"]}' for n, s in seeds.items() if 'at' in s and s['at'] not in bp.REGIONS['oasis_discs']['discs']]
    rg = bp.REGIONS.get('perennial_summits')
    if rg:   # the listing rule (biome_profiles REGIONS perennial_summits)
        import numpy as np
        for n, lo, la, h, et in rg['points']:
            line = float(bp.perennial_snowline_m(np.float32(la), np.float32(lo)))
            if h < line + rg['min_above_line'] or h - et < rg['min_flattening']:
                bad.append(f'perennial_summits {n}: real {h} m, line {line:.0f}, ETOPO {et}: outside the listing rule')
    return bad


def status():
    """(snapshot, frozen, differing keys, inconsistencies, review JSON stale): the check of main(), also run at the
    pipeline start (stage_invariants biome_profiles_frozen, 2026-11-01: a REGIONS entry merged without refreeze)."""
    snap = bp.snapshot()
    frozen = json.loads(FROZEN.read_text(encoding='utf-8'))['snapshot']
    diff = sorted(k for k in set(snap) | set(frozen) if snap.get(k) != frozen.get(k))
    stale = REVIEW.read_text(encoding='utf-8') != bp._fmt(bp._jsonable(bp.table())) + '\n'
    return snap, frozen, diff, consistency(), stale


def main():
    if '--refreeze' in sys.argv:
        snap = bp.snapshot()
        FROZEN.write_text(json.dumps({'note': 'biome_profiles.snapshot(); refreeze only with a deliberate pipeline-facing change',
                                      'snapshot': snap}, indent=1, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')
        bp.dump_json(REVIEW)
        print('frozen', len(snap), 'values ->', FROZEN)
        return 0
    snap, frozen, diff, bad, stale = status()
    for k in diff:
        print('DIFF', k, '\n  frozen:', frozen.get(k, '<missing>')[:300], '\n  table: ', snap.get(k, '<missing>')[:300])
    for b in bad:
        print('INCONSISTENT', b)
    if stale:
        print('STALE', REVIEW, '(python biome_profiles.py)')
    print(f'{len(snap)} values, {len(diff)} differ from the frozen snapshot, {len(bad)} inconsistencies, review JSON {"stale" if stale else "current"}')
    return 1 if diff or bad or stale else 0


if __name__ == '__main__':
    sys.exit(main())
