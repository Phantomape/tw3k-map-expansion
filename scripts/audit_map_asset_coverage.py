"""Fail when any vanilla map asset or coordinate table is not accounted for.

Every vanilla file under terrain/campaigns/<variant>/ and campaign_maps/<variant>/
must match a rule in map_asset_registry.json. Rules marked 'transformed' must be
overridden by the release pack, and raster rules must scale with the canvas.
Database tables with coordinate-like fields must be listed in the registry, and
character placeholders must stay untranslated. Read-only; --output writes JSON.

Run this whenever the canvas grows: a new map asset family that nobody has
transformed shows up here as an error instead of as a live visual defect.
"""
import tw3k_paths
import argparse, json, re, struct, sys
from pathlib import Path

import numpy as np

import campaign_world as cw
from analyze_reference import ROOT, inventory, read_hex
import campaign_world as _frame

GAME = tw3k_paths.GAME_DATA
A = _frame.WORKSPACE
REGISTRY = Path(__file__).with_name('map_asset_registry.json')
COORD_FIELD = re.compile(r'^(startx|starty|x|y|z|pos_x|pos_y|position_x|position_y|x_pos|y_pos|min_x|max_x|min_y|max_y|minx|maxx|miny|maxy|logical_x|logical_y)$', re.I)


class Packs:
    def __init__(self, packs):
        self.index = {}
        for pack in packs:
            for e in inventory(pack, False)['entries']:
                self.index.setdefault(e['path'].lower(), (pack, e))

    def head(self, key, n=64):
        pack, e = self.index[key]
        with pack.open('rb') as f:
            f.seek(e['offset'])
            return f.read(min(n, e['size']))


def dims(head, key):
    if head[:4] == b'DDS ':
        h, w = struct.unpack_from('<2I', head, 12)
        return w, h
    if head[:8] == b'\x89PNG\r\n\x1a\n':
        return struct.unpack_from('>2I', head, 16)
    if key.endswith('.tga'):
        return struct.unpack_from('<2H', head, 12)
    return None


def expected_dims(kind, w, h):
    sy = (cw.H + .5) / (cw.NATIVE_H + .5)
    if kind == 'world':
        return round(w * (cw.W - 1) / (cw.NATIVE_W - 1)), round(h * sy)
    return round(w * cw.W / cw.NATIVE_W), round(h * sy)


def audit_files(registry, release, errors, warnings, rows):
    vanilla = Packs([p for p in sorted(GAME.glob('*.pack')) if not p.name.lower().startswith(('tw3k', '!'))])
    rules = [dict(r, rx=re.compile(r['pattern'])) for r in registry['rules']]
    for variant in registry['variants']:
        for prefix in (f'terrain/campaigns/{variant}/', f'campaign_maps/{variant}/'):
            for key in sorted(k for k in vanilla.index if k.startswith(prefix)):
                norm = key.replace(f'/{variant}/', '/<v>/')
                rule = next((r for r in rules if r['rx'].search(norm) and r.get('variant_only', variant) == variant), None)
                row = {'path': key, 'rule': rule['pattern'] if rule else None, 'status': rule['status'] if rule else 'unclassified'}
                rows.append(row)
                if rule is None:
                    errors.append(f'unclassified vanilla map asset: {key}')
                    continue
                if rule['status'] == 'review':
                    warnings.append(f'unverified ({rule.get("why", "")}): {key}')
                    continue
                if rule['status'] != 'transformed':
                    continue
                if key not in release.index:
                    errors.append(f'transformed asset missing from release pack: {key}')
                    continue
                kind = rule.get('raster')
                if kind:
                    vd, rd = dims(vanilla.head(key), key), dims(release.head(key), key)
                    row['dims'] = {'vanilla': vd, 'release': rd}
                    if not vd or not rd:
                        errors.append(f'cannot read raster size: {key}')
                    elif kind == 'aspect':
                        want = cw.W / (cw.H + .5)
                        if abs(rd[0] / rd[1] - want) > .01 * want:
                            errors.append(f'UI map aspect {rd[0] / rd[1]:.4f} != canvas {want:.4f}: {key} {rd}')
                    else:
                        want = expected_dims(kind, *vd)
                        row['dims']['expected'] = want
                        # BC formats round sizes up to 4-pixel blocks.
                        if max(abs(rd[0] - want[0]), abs(rd[1] - want[1])) > 3:
                            errors.append(f'raster not scaled with canvas ({kind}): {key} vanilla {vd} release {rd} expected ~{want}')


def audit_tables(registry, xml_dir, errors, rows):
    known = registry['db_tables']
    for f in sorted(xml_dir.glob('*.xml')):
        text = f.read_text(encoding='utf-8', errors='replace')
        fields = {m for m in re.findall(r'<([A-Za-z_][A-Za-z0-9_]*)>-?[0-9.]+</\1>', text) if COORD_FIELD.match(m)}
        if not fields:
            continue
        table = f.stem
        rows.append({'table': table, 'fields': sorted(fields), 'status': known.get(table, {}).get('status', 'unclassified')})
        if table not in known:
            errors.append(f'unclassified coordinate table: {table} fields {sorted(fields)}')
        elif set(fields) - set(known[table]['fields']):
            errors.append(f'new coordinate fields in {table}: {sorted(set(fields) - set(known[table]["fields"]))}')


def audit_characters(registry, xml_dir, map_path, errors, rows):
    spec = registry['db_tables']['start_pos_characters']
    placeholders = {tuple(p) for p in spec['placeholders']}
    shifted_placeholders = {(x + cw.PAD_LEFT, y + cw.PAD_BOTTOM) for x, y in placeholders}
    meta = read_hex(map_path)
    cw.assert_grid(meta['width'], meta['height'])
    cells = np.frombuffer(map_path.read_bytes()[-4 - cw.W * cw.H * 16:-4], np.uint8).reshape(cw.H, cw.W, 16)
    land = (cells[:, :, 0] & 3) != 1
    text = (xml_dir / 'start_pos_characters.xml').read_text(encoding='utf-8', errors='replace')
    bad = []
    for m in re.finditer(r'<start_pos_characters [^>]*>(.*?)</start_pos_characters>', text, re.S):
        body = m.group(1)
        get = lambda k: (re.search(rf'<{k}>([^<]*)</{k}>', body) or [None, None])[1]
        if get('start_on_map') != '1':
            continue
        x, y = int(float(get('startx'))), int(float(get('starty')))
        if (x, y) in placeholders:
            continue
        if (x, y) in shifted_placeholders:
            bad.append((get('template'), x, y, 'translated placeholder'))
        elif not (0 <= x < cw.W and 0 <= y < cw.H) or not land[y, x]:
            bad.append((get('template'), x, y, 'not on land'))
    rows.append({'characters_checked': True, 'problems': len(bad)})
    for b in bad:
        errors.append(f'character {b[0]} at {b[1:3]}: {b[3]}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pack', type=Path, required=True)
    ap.add_argument('--xml', type=Path, default=A / 'work/native-split-xml')
    ap.add_argument('--map', type=Path, default=A / 'work/final-map/map.hex')
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    registry = json.loads(REGISTRY.read_text(encoding='utf-8'))
    release = Packs([args.pack])
    errors, warnings, rows = [], [], []
    audit_files(registry, release, errors, warnings, rows)
    audit_tables(registry, args.xml, errors, rows)
    audit_characters(registry, args.xml, args.map, errors, rows)
    report = {'pack': str(args.pack), 'errors': errors, 'warnings': warnings, 'rows': rows}
    if args.output:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')
    for w in warnings:
        print('WARNING', w)
    for e in errors:
        print('ERROR', e)
    print(f'{len(rows)} items checked, {len(errors)} errors, {len(warnings)} warnings')
    sys.exit(1 if errors else 0)


if __name__ == '__main__':
    main()
