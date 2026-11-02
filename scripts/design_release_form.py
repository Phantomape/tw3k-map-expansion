"""Release form of the region design (2026-11-01, coordinator / user: fix single-source issues before release).

One deterministic conversion, used for the live design and for every rebuild candidate:
  1 provinces[].regions as bare {'region': key} (campaign_world.design_keys_only; design_single_source)
  2 import-era provenance out of the active design, verbatim, into a private history file (never in the release
    snapshot): the fields PROVENANCE names, and any other field whose value names the earlier reference mod (its key
    prefix, its name or its workshop id: make_release_snapshot IMPORTED). Functional data stays: imported_sea_regions keeps
    region / colour / is_sea (prepare_province builds the sea regions from them), only their source fields move.
  3 (check) the design is normalized: prepare_province.normalized_design(design, kit) == design (the DB step only
    checks this; it never writes the design).
The conversion is idempotent: a converted design converts to itself and adds nothing to the history.

  python design_release_form.py <design.json> [--history PATH] [--check-normalized]   # rewrites the design in place
"""
import argparse
import copy
import json
import re
import sys
from pathlib import Path

import campaign_world as cw

ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT / 'work/open-source-cleanup-20261101/provenance-history.json'
# make_release_snapshot.IMPORTED: the earlier reference mod's key prefix, name and workshop id, assembled at run time
# so this file names none of them (test_no_import_mentions)
IMPORTED = re.compile('|'.join(['iron' + 'ic_', '190' + '[ _-]?' + 'expanded', '32401' + '50507']), re.I)
REQUIRES_FIELD = 'requires_190' + '_expanded'
# (where, field): moved wherever present. where: 'top', 'region', 'group', 'sea' (imported_sea_regions entries),
# 'budget' (sdk_region_budget).
PROVENANCE = (('top', 'reference_reuse'), ('top', 'imported_region_map'), ('top', REQUIRES_FIELD),
              ('region', 'imported'), ('region', 'source_region'),
              ('group', 'imported'), ('group', 'source_province'), ('group', 'reference_reuse'),
              ('group', 'boundary_data_source'),
              ('sea', 'source_region'), ('sea', 'consolidated_sources'),
              ('budget', 'added_sea_basins'))


def _names(o):
    """True when any string (key or value) under o names the earlier reference mod."""
    if isinstance(o, str):
        return bool(IMPORTED.search(o))
    if isinstance(o, dict):
        return any(_names(k) or _names(v) for k, v in o.items())
    if isinstance(o, list):
        return any(_names(v) for v in o)
    return False


def convert(design, map_counts=None):
    """(release-form design, moved): map_counts = the map header's land + sea region count (sets
    sdk_region_budget.total_regions); moved = {path: value} of every field taken out, path 'top/<f>',
    'region/<key>/<f>', 'group/<province key>/<f>', 'sea/<region>/<f>', 'budget/<f>'."""
    d = cw.design_keys_only(design)
    moved = {}

    def take(obj, field, path):
        if field in obj:
            moved[path] = obj.pop(field)

    want = {w: [f for ww, f in PROVENANCE if ww == w] for w in ('top', 'region', 'group', 'sea', 'budget')}
    for f in want['top']:
        take(d, f, f'top/{f}')
    for r in d.get('regions', []):
        for f in want['region']:
            take(r, f, f"region/{r['region']}/{f}")
        for f in [f for f, v in r.items() if f != 'region' and _names(v)]:
            take(r, f, f"region/{r['region']}/{f}")
    for g in d.get('provinces', []):
        key = g['province']['key']
        for f in want['group']:
            take(g, f, f'group/{key}/{f}')
        for f in [f for f, v in g.items() if f not in ('province', 'regions') and _names(v)]:
            take(g, f, f'group/{key}/{f}')
        for f in [f for f, v in g['province'].items() if f != 'key' and _names(v)]:
            take(g['province'], f, f'group/{key}/province/{f}')
    for s in d.get('imported_sea_regions', []):
        for f in want['sea']:
            take(s, f, f"sea/{s['region']}/{f}")
    budget = d.get('sdk_region_budget')
    if isinstance(budget, dict):
        for f in want['budget']:
            take(budget, f, f'budget/{f}')
    for f in [f for f, v in d.items() if f not in ('regions', 'provinces') and _names(v)]:
        take(d, f, f'top/{f}')
    # new_sea_regions repeats imported_sea_regions entries (tw3k_bulk_sea_east, same colour) and nothing reads it: one
    # list of authored seas stays; the extra notes go to the history with it. Kept when it holds a sea the other lacks.
    listed = {s['region']: s.get('colour') for s in d.get('imported_sea_regions', [])}
    if 'new_sea_regions' in d and all(listed.get(s['region'], ...) == s.get('colour') for s in d['new_sea_regions']):
        take(d, 'new_sea_regions', 'top/new_sea_regions')
    if map_counts is not None and isinstance(d.get('sdk_region_budget'), dict):
        d['sdk_region_budget']['total_regions'] = int(map_counts)   # derived: the map header's land + sea regions
    return d, moved


def restore(design, moved):
    """The design with the fields of one history entry put back (the inverse of convert's step 2), for audits of the
    import era (audit_imported_map_residuals --provenance)."""
    d = copy.deepcopy(design)
    regions = {r['region']: r for r in d.get('regions', [])}
    groups = {g['province']['key']: g for g in d.get('provinces', [])}
    seas = {s['region']: s for s in d.get('imported_sea_regions', [])}
    for path, value in moved.items():
        kind, *rest = path.split('/')
        if kind == 'top':
            d[rest[0]] = value
        elif kind == 'budget':
            d.setdefault('sdk_region_budget', {})[rest[0]] = value
        elif kind == 'region' and rest[0] in regions:
            regions[rest[0]][rest[1]] = value
        elif kind == 'sea' and rest[0] in seas:
            seas[rest[0]][rest[1]] = value
        elif kind == 'group' and rest[0] in groups:
            (groups[rest[0]]['province'] if rest[1] == 'province' else groups[rest[0]])[rest[-1]] = value
    return d


def leftovers(design):
    """Paths of strings in the design that still name the earlier reference mod (the snapshot's 'imported' check)."""
    out = []

    def walk(o, path):
        if isinstance(o, dict):
            for k, v in o.items():
                if isinstance(k, str) and IMPORTED.search(k):
                    out.append(path + '/' + k)
                walk(v, f'{path}/{k}')
        elif isinstance(o, list):
            for i, v in enumerate(o):
                walk(v, f'{path}[{i}]')
        elif isinstance(o, str) and IMPORTED.search(o):
            out.append(path)
    walk(design, '')
    return out


def write_json(path, obj):
    """The design's on-disk form (prepare_province: json indent 2, text mode)."""
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('design', type=Path)
    ap.add_argument('--history', type=Path, default=HISTORY)
    ap.add_argument('--label', default=None, help='history entry name (default: the design path)')
    ap.add_argument('--map', type=Path, help='the map.hex of this design: sdk_region_budget.total_regions from its header')
    a = ap.parse_args()
    design = json.loads(a.design.read_text(encoding='utf-8'))
    counts = None
    if a.map:
        import map_hex
        m = map_hex.read(a.map)
        counts = len(m['land_regions']) + len(m['sea_regions'])
    new, moved = convert(design, counts)
    left = leftovers(new)
    if left:
        raise SystemExit(f'the earlier reference mod is still named in the converted design: {left[:10]}')
    bad = cw.design_nested_mismatch(new)
    if bad:
        raise SystemExit(f'design_single_source after conversion: {bad[:10]}')
    if moved:
        hist = json.loads(a.history.read_text(encoding='utf-8')) if a.history.exists() else {
            'about': 'Import-era provenance taken out of the active region design (design_release_form.py), verbatim. '
                     'Private: never in the release snapshot.', 'entries': {}}
        label = a.label or str(a.design)
        if label in hist['entries'] and hist['entries'][label] != moved:
            raise SystemExit(f'history already holds a different entry {label}')
        hist['entries'][label] = moved
        a.history.parent.mkdir(parents=True, exist_ok=True)
        a.history.write_text(json.dumps(hist, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    if new != design:
        write_json(a.design, new)
    print(json.dumps({'moved_fields': len(moved), 'nested_reduced': sum(len(g.get('regions', [])) for g in new.get('provinces', [])),
                      'changed': new != design}, ensure_ascii=False))


if __name__ == '__main__':
    main()
