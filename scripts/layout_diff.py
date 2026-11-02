"""Layout diff (release L0, coordinator 2026-11-01, option A): the campaign map as a difference against vanilla.

The public repository ships no map.hex. A user builds it from their own vanilla 3k_dlc07 map.hex plus this diff:

    map.hex = vanilla cells placed at the native offset on the empty canvas, then every stored hex's 16 bytes,
              then the stored header lists

Only hexes that differ from that base are stored (vanilla hexes the mod never changed are not in the repository).
No rule step runs: the rule scripts stay in the source as the record of how the layers were made, and the rule-format
L0 (data/map-layout layout.npz ...) comes back when bake_map.py takes over the map (decisions.md).

  data/map-layout-diff/layout-diff.npz    q, r (int16), mask (N uint16: bit i = byte i differs), bytes (N x 16 uint8,
                                          the differing bytes; 0 where the byte is vanilla's, which is never stored)
  data/map-layout-diff/manifest.json      format, vanilla / map sha256, grid, native offset, header, counts
  data/map-layout-diff/river-channels.json  the map's river channel list (written beside map.hex)

  python layout_diff.py extract --map MAP --out DIR [--river-channels PATH]
  bake_map.py --diff --layout DIR --out MAP        (stops when the vanilla map is not the recorded one)
"""
import argparse, hashlib, json, os, shutil, sys
from pathlib import Path
import numpy as np

FORMAT = 'layout-diff-v2'
NPZ, MANIFEST, CHANNELS = 'layout-diff.npz', 'manifest.json', 'river-channels.json'
HEADER = ('major', 'minor', 'opaque', 'game', 'campaign_map', 'land_regions', 'sea_regions', 'land_ground_types',
          'sea_ground_types', 'climates', 'attritions', 'areas_of_interest', 'palettes')


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def base_cells(vanilla_cells, shape, native_slice):
    """The empty canvas (zeros) with the vanilla cells at the native offset."""
    cells = np.zeros(tuple(shape) + (16,), np.uint8)
    cells[native_slice] = vanilla_cells
    return cells


def diff(cells, base):
    """(q, r, mask, bytes) of every hex with a byte that differs from base: only the differing bytes are kept (the
    others are 0 and their mask bit clear), so no unchanged vanilla byte is stored."""
    d = cells != base
    r, q = np.nonzero(d.any(-1))
    dm = d[r, q]
    mask = (dm.astype(np.uint16) << np.arange(16, dtype=np.uint16)).sum(-1).astype(np.uint16)
    return q.astype(np.int16), r.astype(np.int16), mask, np.where(dm, cells[r, q], 0).astype(np.uint8)


def apply(base, q, r, mask, stored):
    out = base.copy()
    rr, qq = r.astype(np.intp), q.astype(np.intp)
    bits = ((mask[:, None] >> np.arange(16, dtype=np.uint16)) & 1).astype(bool)
    out[rr, qq] = np.where(bits, stored, out[rr, qq])
    return out


def header_to_json(m):
    return {k: (m[k].hex() if k == 'opaque' else [p.hex() for p in m[k]] if k == 'palettes' else m[k]) for k in HEADER}


def header_from_json(h):
    return {k: (bytes.fromhex(h[k]) if k == 'opaque' else [bytes.fromhex(p) for p in h[k]] if k == 'palettes' else h[k]) for k in HEADER}


def field_counts(cells, base, mask):
    """Per byte: hexes inside mask whose byte differs from the base."""
    d = (cells != base) & mask[..., None]
    return {f'byte_{i}': int(d[..., i].sum()) for i in range(16)}


def extract(map_path, vanilla_path, out_dir, native_slice, river_channels=None):
    import map_hex
    m = map_hex.read(Path(map_path)); V = map_hex.read(Path(vanilla_path))
    cells = m['cells']; base = base_cells(V['cells'], cells.shape[:2], native_slice)
    q, r, mask, stored = diff(cells, base)
    nat = np.zeros(cells.shape[:2], bool); nat[native_slice] = True
    in_rect = nat[r.astype(np.intp), q.astype(np.intp)]
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / NPZ, q=q, r=r, mask=mask, bytes=stored)
    man = {'format': FORMAT, 'by': 'layout_diff.py extract', 'vanilla_sha256': sha256(vanilla_path), 'map_sha256': sha256(map_path),
           'grid': [int(cells.shape[1]), int(cells.shape[0])],
           'native_slice': [[native_slice[0].start, native_slice[0].stop], [native_slice[1].start, native_slice[1].stop]],
           'header': header_to_json(m),
           'counts': {'stored_hexes': int(len(q)), 'stored_in_vanilla_rectangle': int(in_rect.sum()),
                      'stored_outside_rectangle': int((~in_rect).sum()), 'vanilla_rectangle_hexes': int(nat.sum()),
                      'stored_bytes_in_vanilla_rectangle': int(sum(bin(int(x)).count('1') for x in mask[in_rect])),
                      'vanilla_rectangle_by_byte': field_counts(cells, base, nat)},
           'npz_sha256': sha256(out_dir / NPZ)}
    if river_channels:
        shutil.copyfile(river_channels, out_dir / CHANNELS); man['river_channels_sha256'] = sha256(out_dir / CHANNELS)
    (out_dir / MANIFEST).write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding='utf-8')
    return man


def build(layout_dir, vanilla_path):
    """The map dict the diff describes; SystemExit when the vanilla map or the npz is not the recorded one."""
    import map_hex
    layout_dir = Path(layout_dir)
    man = json.loads((layout_dir / MANIFEST).read_text(encoding='utf-8'))
    if man.get('format') != FORMAT:
        raise SystemExit(f'{layout_dir / MANIFEST}: not a {FORMAT} layout')
    if sha256(vanilla_path) != man['vanilla_sha256']:
        raise SystemExit(f'vanilla map {vanilla_path} is not the recorded one (sha256 {man["vanilla_sha256"][:16]}...): '
                         'game update? the diff would land on the wrong base')
    if sha256(layout_dir / NPZ) != man['npz_sha256']:
        raise SystemExit(f'{layout_dir / NPZ} is not the recorded one')
    V = map_hex.read(Path(vanilla_path))
    (r0, r1), (q0, q1) = man['native_slice']
    W, H = man['grid']
    z = np.load(layout_dir / NPZ)
    m = header_from_json(man['header'])
    m['cells'] = apply(base_cells(V['cells'], (H, W), (slice(r0, r1), slice(q0, q1))), z['q'], z['r'], z['mask'], z['bytes'])
    return m, man


def layout_problems(layout_dir, vanilla_path):
    """The release snapshot's 'layout' self-check for the diff format: no vanilla content is stored. Inside the native
    rectangle every stored byte differs from the vanilla byte it replaces (an unchanged vanilla byte is never stored)
    and every byte outside the mask is 0. Returns a list of problems (empty: pass)."""
    import map_hex
    layout_dir = Path(layout_dir)
    man = json.loads((layout_dir / MANIFEST).read_text(encoding='utf-8'))
    z = np.load(layout_dir / NPZ)
    V = map_hex.read(Path(vanilla_path))
    (r0, r1), (q0, q1) = man['native_slice']
    q, r, mask, b = z['q'].astype(np.intp), z['r'].astype(np.intp), z['mask'], z['bytes']
    bits = ((mask[:, None] >> np.arange(16, dtype=np.uint16)) & 1).astype(bool)
    out = []
    if (b[~bits] != 0).any():
        out.append(f'{NPZ}: {int((b[~bits] != 0).sum())} bytes outside the mask are not 0')
    inr = (r >= r0) & (r < r1) & (q >= q0) & (q < q1)
    van = V['cells'][r[inr] - r0, q[inr] - q0]
    same = bits[inr] & (b[inr] == van)
    if same.any():
        out.append(f'{NPZ}: {int(same.sum())} stored bytes inside the native rectangle equal vanilla (vanilla copy)')
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    e = sub.add_parser('extract'); e.add_argument('--map', type=Path, required=True); e.add_argument('--out', type=Path, required=True)
    e.add_argument('--river-channels', type=Path)
    a = ap.parse_args()
    import campaign_world as cw
    van = cw.VANILLA_MAP
    man = extract(a.map, van, a.out, cw.NATIVE_HEX_SLICE, a.river_channels)
    print(json.dumps(man['counts'], indent=1))


if __name__ == '__main__':
    main()
