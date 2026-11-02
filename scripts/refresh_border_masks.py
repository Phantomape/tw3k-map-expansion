"""Recompute map.hex byte 15 (region-border edge mask) from the region ids on every hex.

Steps that move hexes between regions (town moves, relayouts, lakes, rivers, frontier regions)
did not all maintain byte 15: on 2026-10-24 about 100k hexes of the final map disagreed with the
rule (18k border hexes outside the native map had no border sides at all), so region borders
between new regions were missing. Vanilla 3k_dlc07 map.hex follows map_hex.border_masks on every
hex, so recomputing is a no-op on unedited vanilla ground. all_directions_build.py database calls
refresh() before the kit sync, so every database build ships consistent masks.

  python refresh_border_masks.py            # dry run: count hexes that would change
  python refresh_border_masks.py --apply    # rewrite work/final-map/map.hex (backup beside it)
"""
import argparse, json, shutil, time

import numpy as np

import campaign_world as cw
import map_hex

FINAL = cw.WORKSPACE / 'work/final-map/map.hex'


def refresh(path=FINAL, apply=True):
    """Report dict; with apply and any change, back up and rewrite path (atomic)."""
    m = map_hex.read(path)
    cells = m['cells']
    masks = map_hex.border_masks(cells)
    diff = masks != cells[..., 15]
    inside = np.zeros(diff.shape, bool)
    inside[cw.NATIVE_HEX_SLICE] = True
    report = {'map': str(path), 'hexes_changed': int(diff.sum()), 'outside_native': int((diff & ~inside).sum()),
              'inside_native': int((diff & inside).sum()),
              'border_hexes_without_sides_before': int(((masks > 0) & (cells[..., 15] == 0)).sum()),
              'applied': False}
    if apply and diff.any():
        backup = path.with_name(f'{path.name}.before-border-masks-{time.strftime("%Y%m%d-%H%M%S")}')
        shutil.copy2(path, backup)
        m['cells'] = cells.copy()
        m['cells'][..., 15] = masks
        map_hex.write(m, path)
        report.update(applied=True, backup=str(backup))
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--apply', action='store_true')
    args = ap.parse_args()
    print(json.dumps(refresh(apply=args.apply), indent=1))
    if not args.apply:
        print('Dry run only; add --apply to rewrite map.hex.')


if __name__ == '__main__':
    main()
