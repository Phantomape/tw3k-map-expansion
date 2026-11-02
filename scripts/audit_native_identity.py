"""Compare every native-area layer of the stage (or a pack) with vanilla.

The native map must stay identical to vanilla except where terrain was
intentionally edited: the rim along the native border and town plateaus.
Self-consistency audits (seams, coverage) cannot see a native layer silently
drifting from vanilla; this one compares each layer against vanilla placed at
the native offset and reports, per layer, where it differs.

Differences are split into those inside the intended edits (polished-edit-alpha
> 0 grown by 3 hexes, plus the border rim) and those OUTSIDE them, which are
suspected bugs and listed by 16-hex block.

  python audit_native_identity.py [--pack PATH] [--variant V]
Writes analysis/native-identity-audit.json; prints a table.
"""
import argparse, bisect, collections, io, json, struct
from pathlib import Path

import numpy as np
from PIL import Image

import campaign_world as cw
from analyze_reference import inventory
from repair_campaign_river_roots import Assets, embedded
from read_root_prefabs import decode
from read_tile_list import parse as tile_parse
from sync_native_cache_height_ranges import cache
from translate_campaign_lake_instances import records as leaf_records

RIM = 28
S = cw.WORKSPACE / 'work/release-input'
NW, NH = cw.NATIVE_W, cw.NATIVE_H
QX, RY = cw.PAD_LEFT, cw.PAD_BOTTOM


class Source:
    def __init__(self, pack=None):
        self.pack = pack
        self.index = {e['path']: e for e in inventory(pack, False)['entries']} if pack else None

    def read(self, key):
        if self.pack is None:
            p = S / key
            return p.read_bytes() if p.exists() else None
        e = self.index.get(key)
        if e is None:
            return None
        with self.pack.open('rb') as f:
            f.seek(e['offset'])
            return f.read(e['size'])


def rim_mask():
    q, r = np.meshgrid(np.arange(NW), np.arange(NH))
    edge = np.minimum.reduce([q, NW - 1 - q, r, NH - 1 - r])
    return edge < RIM


EDITS = None   # native hex grid of intended edits (set in main)
ROAD = None    # native tile grid cells of the road pieces road_tiles.py laid (exempt; set in main)
HEHUANG = None   # HEHUANG PATCH: native hex grid of the Hehuang patch grown by 2 (set in main), or None
FILLER = None    # NONPLAY GROUND: vanilla non-playable filler now land (revealed_filler.filler_hex, native grid; set in main)


from native_edit_mask import edited_hexes, TOLERANCE


def edit_mask(variant):
    """Intended edits: the shared definition every pipeline step uses
    (native_edit_mask: staged LF differs from vanilla, grown by its halo), plus the rim."""
    return edited_hexes(variant)[cw.NATIVE_HEX_SLICE] | rim_mask()


def summarise(name, diff_hex, rim, detail=None):
    """diff_hex: bool (NH, NW) native hex grid. 'outside' = not inside an intended edit."""
    # --- HEHUANG PATCH: differences inside the patch mask are intended (counted in
    # edit and reported as exempt_hehuang); outside it the rule is unchanged.
    allowed = EDITS if HEHUANG is None else EDITS | HEHUANG
    if FILLER is not None:   # --- NONPLAY GROUND: the filler is repainted from real geography (user 2026-10-25)
        allowed = allowed | FILLER
    inner = diff_hex & ~allowed
    out = {'layer': name, 'differing_hexes': int(diff_hex.sum()), 'inside_edits': int((diff_hex & allowed).sum()), 'outside_edits': int(inner.sum())}
    if HEHUANG is not None:
        out['exempt_hehuang'] = int((diff_hex & HEHUANG).sum())
    if FILLER is not None:
        out['exempt_filler'] = int((diff_hex & FILLER).sum())
    if inner.any():
        # coarse clusters: 16-hex blocks
        r, q = np.nonzero(inner)
        blocks = collections.Counter(zip((q // 16) * 16 + QX, (r // 16) * 16 + RY))
        out['outside_edit_blocks'] = [{'q': int(a), 'r': int(b), 'hexes': n} for (a, b), n in blocks.most_common(12)]
    if detail:
        out['detail'] = detail
    return out


def raster_hex(diff_px, per_hex):
    """Pool a native-frame pixel diff mask (rows south-up) onto the native hex grid
    by tile ownership (cw.tile_hex; pixel x lies in tile x*2//per_hex). The native
    slice starts on an even column, so local parity is global; pixels above the
    last native row are pooled into it."""
    h, w = diff_px.shape
    ys = np.arange(h) * 2 // per_hex
    xs = np.arange(w) * 2 // per_hex
    out = np.zeros((NH, NW), bool)
    r, c = np.nonzero(diff_px)
    q = np.minimum(xs[c] // 2, NW - 1)
    out[np.clip((ys[r] - q % 2) // 2, 0, NH - 1), q] = True
    return out


def dds_native(b, channels, per_hex, ox, oy, nw, nh):
    h, w = struct.unpack_from('<2I', b, 12)
    dt = '<u2' if channels == 0 else np.uint8
    a = np.frombuffer(b, dt, offset=128)
    a = a.reshape(h, w) if channels == 0 else a.reshape(h, w, channels)
    a = a[::-1]
    return a[oy:oy + nh, ox:ox + nw]


def raster_layers(src, van, variant, rim, report):
    pre = f'terrain/campaigns/{variant}/'
    for key, ch, per in ((pre + 'lf_height_map.dds', 0, 8), (pre + 'lf_sea_height_map.dds', 0, 4), (pre + 'global_map/global_blend.dds', 2, 8)):
        sb, vb = src.read(key), van.read(key)
        if sb is None or vb is None:
            report.append({'layer': key, 'missing': sb is None})
            continue
        vh, vw = struct.unpack_from('<2I', vb, 12)
        v = dds_native(vb, ch, per, 0, 0, vw, vh)
        s = dds_native(sb, ch, per, QX * per, RY * per, vw, vh)
        # Heights within the shared edit tolerance are projection rounding, not edits.
        d = (np.abs(s.astype(np.int32) - v) > TOLERANCE) if ch == 0 else np.any(s != v, axis=2)
        report.append(summarise(key, raster_hex(d, per), rim))
    key = f'campaign_maps/{variant}/camera_heightmap.png'
    sb, vb = src.read(key), van.read(key)
    if sb and vb:
        s = np.array(Image.open(io.BytesIO(sb)))[::-1]
        v = np.array(Image.open(io.BytesIO(vb)))[::-1]
        s = s[RY * 2:RY * 2 + v.shape[0], QX * 2:QX * 2 + v.shape[1]]
        report.append(summarise(key, raster_hex(s != v, 2), rim))


def tile_layers(src, van, variant, rim, report):
    pre = f'terrain/campaigns/{variant}/'
    for key in (pre + 'global_map/tile_list.bin',):
        sb, vb = src.read(key), van.read(key)
        if sb is None or vb is None:
            continue
        sp, _, _, _, sr, _ = tile_parse(sb)
        vp, _, _, _, vr, _ = tile_parse(vb)
        ox, oy = cw.NATIVE_OFFSET_TILE
        cur = {(r[3], r[4], sp[r[1]]): r for r in sr}
        miss = np.zeros((NH, NW), bool)
        changed = np.zeros((NH, NW), bool)
        kinds = collections.Counter()
        for r in vr:
            k = (r[3] + ox, r[4] + oy, vp[r[1]])
            q = min(NW - 1, max(0, r[3] // 2))   # anchor tile's owner (cw.tile_hex)
            rr = min(NH - 1, max(0, (r[4] - q % 2) // 2))
            c = cur.get(k)
            if ROAD is not None and 0 <= r[4] < ROAD.shape[0] and 0 <= r[3] < ROAD.shape[1] and ROAD[r[4], r[3]]:
                kinds['road tile cell (exempt)'] += 1
                continue
            if c is None:
                miss[rr, q] = True
                kinds['missing:' + vp[r[1]].split('\\')[-2]] += 1
            elif (c[2], c[5]) != (r[2], r[5]) or abs(c[6] - r[6]) > 1e-4 or abs(c[7] - r[7]) > 1e-4:
                changed[rr, q] = True
                kinds['changed'] += 1
        report.append(summarise(key + ' (vanilla tiles missing)', miss, rim, dict(kinds.most_common(8))))
        report.append(summarise(key + ' (vanilla tiles changed)', changed, rim))
    key = pre + 'tile_map.tiles'
    sb = src.read(key)
    vb = van.read('terrain/campaigns/3k_main_map/tile_map.tiles')
    if sb and vb:
        _, SL = cache(sb, cw.TILE_W, cw.TILE_H)
        _, VL = cache(vb, cw.NATIVE_TILE_W, cw.NATIVE_TILE_H)
        for li in range(3):
            s = SL[li][cw.NATIVE_TILE_SLICE]
            v = VL[li]
            d = (np.abs(s['lo'] - v['lo']) > 1e-4) | (np.abs(s['hi'] - v['hi']) > 1e-4) | (s['flags'] != v['flags'])
            if ROAD is not None:
                d &= ~ROAD                  # road piece cells: new road tiles, not native drift
            # All four tile cells of each hex (cw.tile_hex): columns 2q, 2q+1 and
            # rows 2r+odd(q), +1. The native slice starts on even tiles, so local parity is global.
            c = d[:, ::2] | d[:, 1::2]
            hexd = c[:NH * 2:2] | c[1:NH * 2:2]
            hexd[:, 1::2] = c[1:NH * 2:2, 1::2] | c[2:NH * 2 + 1:2, 1::2]
            report.append(summarise(f'{key} layer{li} (range/flags vs vanilla)', hexd[:NH, :NW], rim))


def prop_layer(src, van, variant, rim, report):
    pre = f'terrain/campaigns/{variant}/'
    sb, vb = src.read(pre + 'global_props.bin'), van.read(pre + 'global_props.bin')
    if not (sb and vb):
        return

    def instances(blob, shift):
        ab, _, _ = embedded(blob)
        roots = {r['path']: r for r in decode(ab[pre + 'bmd_objects.bin'])[1]}
        out = collections.Counter()
        # The keys starting with stem + '.' are one run of the sorted keys, [stem + '.', stem + '/')
        # ('/' follows '.'); visited in ab's own order, as the scan of every key did.
        keys = sorted(ab)
        order = {k: i for i, k in enumerate(ab)}
        for path, rt in roots.items():
            stem = path[:-4]
            t = rt['matrix'][12:15]
            for k in sorted(keys[bisect.bisect_left(keys, stem + '.'):bisect.bisect_left(keys, stem + '/')], key=order.__getitem__):
                b = ab[k]
                if not k.endswith('.bin'):
                    continue
                try:
                    rs = leaf_records(b)
                except Exception:
                    continue
                for i in rs:
                    m = i['model'].lower()
                    # Leaves carry their own world position; the engine ignores
                    # root translations (world_space_leaves.py).
                    water = True
                    # The pipeline shifts root translations (or water leaves) in
                    # float32; apply the shift the same way so rounding matches.
                    f = lambda a, d: float(np.float32(a + d))
                    if water:
                        x, z = f(i['matrix'][9], shift[0]), f(i['matrix'][11], shift[1])
                    else:
                        x, z = i['matrix'][9] + f(t[0], shift[0]), i['matrix'][11] + f(t[2], shift[1])
                    if i['matrix'][10] < -100:
                        continue
                    # Big-river models are placed by their translated model pivot,
                    # not by the instance position (repair_campaign_river_roots.py
                    # audits them); compare their presence only.
                    if '/models/river_' in m:
                        out[(m, None, None)] += 1
                        continue
                    out[(m, round(x, 2), round(z, 2))] += 1
        return out

    sx, sz = cw.NATIVE_SHIFT_WORLD
    s = instances(sb, (0, 0))
    v = instances(vb, (sx, sz))
    missing = v - s
    extra = s - v
    # Leaves carry float32(leaf + float32 root translation); the reference adds
    # the float64 shift. Positions rounded to 0.01 can then differ by one step:
    # pair such missing/extra copies of the same model as the same instance.
    by_model = collections.defaultdict(list)
    for (m, x, z), n in extra.items():
        if x is not None:
            by_model[m].append([x, z, n])
    for key in list(missing):
        m, x, z = key
        if x is None:
            continue
        for cand in by_model.get(m, ()):
            while cand[2] and missing[key] and abs(cand[0] - x) <= .0101 and abs(cand[1] - z) <= .0101:
                cand[2] -= 1; missing[key] -= 1; extra[(m, cand[0], cand[1])] -= 1
    missing, extra = +missing, +extra
    grid = np.zeros((NH, NW), bool)
    kinds = collections.Counter()
    for (m, x, z), n in missing.items():
        if x is None:
            kinds[m] += n
            continue
        q, r = cw.world_hex(x, z)   # owning hex (odd-q, tile_hex rule)
        q, r = int(q) - QX, int(r) - RY
        if 0 <= q < NW and 0 <= r < NH:
            grid[r, q] = True
        kinds[m.split('/')[-2] if '/' in m else m] += n
    report.append(summarise(pre + 'global_props.bin (vanilla instances missing)', grid, rim,
                            {'missing': sum(missing.values()), 'extra': sum(extra.values()), 'missing_by_folder': dict(kinds.most_common(8))}))


def tree_layer(src, van, variant, rim, report):
    key = f'campaign_maps/{variant}/display/trees/trees.campaign_tree_list'
    sb, vb = src.read(key), van.read(key)
    if not (sb and vb):
        return

    def rows(b, shift):
        n = struct.unpack_from('<I', b, 20)[0]
        pos = 24
        out = collections.Counter()
        for g in range(n):
            l = struct.unpack_from('<H', b, pos)[0]
            name = b[pos + 2:pos + 2 + l]
            pos += 2 + l
            c = struct.unpack_from('<I', b, pos)[0]
            pos += 4
            for _ in range(c):
                x, y, z = struct.unpack_from('<3f', b, pos)
                st = pos
                pos += 14
                s = struct.unpack_from('<I', b, pos)[0]
                pos += 4 + s * 4
                # The pipeline writes float32(x + shift); match that rounding exactly.
                out[(name, round(float(np.float32(x + shift[0])), 2), round(y, 2), round(float(np.float32(z + shift[1])), 2), b[st + 12:pos])] += 1
        return out

    sx, sz = cw.NATIVE_SHIFT_WORLD
    s, v = rows(sb, (0, 0)), rows(vb, (sx, sz))
    missing = v - s
    grid = np.zeros((NH, NW), bool)
    for (_, x, _, z, _), n in missing.items():
        q, r = cw.world_hex(x, z)   # owning hex (odd-q, tile_hex rule)
        q, r = int(q) - QX, int(r) - RY
        if 0 <= q < NW and 0 <= r < NH:
            grid[r, q] = True
    report.append(summarise(key + ' (vanilla trees missing/changed)', grid, rim, {'missing_or_changed': sum(missing.values())}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pack', type=Path)
    ap.add_argument('--variant', default='3k_dlc07_main_map')
    args = ap.parse_args()
    src = Source(args.pack)
    native = Assets(original_only=True)

    class Vanilla:
        def read(self, k):
            try:
                return native.read(k)
            except KeyError:
                return None
    van = Vanilla()
    global EDITS, HEHUANG, FILLER
    EDITS = edit_mask(args.variant)
    global ROAD
    road_cover = cw.WORKSPACE / 'analysis/road-tile-cover.npy'
    if road_cover.exists():
        ROAD = np.load(road_cover)[cw.NATIVE_TILE_SLICE]
        print('road tile cells exempt in the native area:', int(ROAD.sum()))
    from native_edit_mask import hehuang_mask, HEHUANG_GROW   # HEHUANG PATCH
    hh = hehuang_mask()
    HEHUANG = None if hh is None else hh[cw.NATIVE_HEX_SLICE]
    import map_hex, revealed_filler   # NONPLAY GROUND: the filler mask on the native grid
    _m = map_hex.read(cw.WORKSPACE / 'work/final-map/map.hex')
    FILLER = revealed_filler.filler_hex(_m['cells'], _m['land_regions'] + _m['sea_regions'])[cw.NATIVE_HEX_SLICE]
    rim = rim_mask()
    report = []
    for fn in (raster_layers, tile_layers, prop_layer, tree_layer):
        fn(src, van, args.variant, rim, report)
    out = cw.WORKSPACE / f'analysis/native-identity-audit-{args.variant}.json'
    import hehuang_terrain   # HEHUANG PATCH: every patch present (hehuang, korea), or-ed
    hh_info = None if HEHUANG is None else {'mask': 'analysis/<name>-patch.npz hex_mask (height weight > 0) of %s, or-ed, grown by %d hexes' % ([p.name for p in hehuang_terrain.load().parts], HEHUANG_GROW), 'native_hexes': int(HEHUANG.sum())}
    out.write_text(json.dumps({'source': str(args.pack or S), 'variant': args.variant, 'rim_hexes': RIM, 'hehuang_exemption': hh_info, 'road_tile_exempt_native_cells': int(ROAD.sum()) if ROAD is not None else 0, 'layers': report}, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'{"layer":78} {"diff":>8} {"in edit":>8} {"OUTSIDE":>8} {"hehuang":>8}')
    for r in report:
        if 'differing_hexes' in r:
            print(f'{r["layer"][-78:]:78} {r["differing_hexes"]:8} {r["inside_edits"]:8} {r["outside_edits"]:8} {r.get("exempt_hehuang", "-"):>8}')
            for b in r.get('outside_edit_blocks', [])[:3]:
                print(f'{"":10}outside-edit block q~{b["q"]} r~{b["r"]}: {b["hexes"]} hexes')
            if r.get('detail'):
                print(f'{"":10}{r["detail"]}')
        else:
            print(r)
    print('->', out)


if __name__ == '__main__':
    main()
