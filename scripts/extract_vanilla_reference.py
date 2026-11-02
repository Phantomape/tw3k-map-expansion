"""Extract the vanilla game files the build reads as references from your own game install into
reference/vanilla-terrain/ (open-source checklist C3; the files are game content and are not distributed).

  python scripts/extract_vanilla_reference.py            extract what is missing or differs
  python scripts/extract_vanilla_reference.py --check    only report; exit 1 if a file is missing or differs

Each file is read from the named vanilla pack in tw3k_paths.GAME_DATA, never from another pack in data/ (a mod
pack there may carry a file of the same path), and checked against the SHA-256 in FILES, recorded from the
game version the release was built with. A different hash stops the run: the game was updated and the
build's vanilla reference changed with it.

The other vanilla inputs come from elsewhere: the logical map from CAIME's template (BUILD.md), the terrain
snapshot terrain-before-polish from the first build's geometry export (-RebuildGeometry with
TW3K_SEED_BASE_FROM_STAGE=1), the database and map folders copied by hand from the Assembly Kit
(BUILD.md, known gap 1).
"""
import argparse
import hashlib
import sys
from pathlib import Path

import tw3k_paths
from analyze_reference import ROOT, inventory

OUT = ROOT / 'reference/vanilla-terrain'
# path in the pack -> (vanilla pack, bytes, sha256); read by build_expanded_battle_entrances / build_combined_mod
FILES = {
    'terrain/battles/3k_main_map/battle_locations_map.bin':
        ('fast.pack', 2592781, '074ac7173582bd8e85f8e8ffbeed479d9cb4060417938a322df4c09c937fdc5d'),
}


def read_entry(pack, path):
    """Bytes of one uncompressed entry of a PFH5 pack."""
    entry = next((e for e in inventory(pack, False)['entries'] if e['path'] == path), None)
    if entry is None:
        raise SystemExit(f'{path} is not in {pack}')
    if entry['compressed']:
        raise SystemExit(f'{path} is compressed in {pack}; not supported')
    with open(pack, 'rb') as f:
        f.seek(entry['offset'])
        return f.read(entry['size'])


def main(argv=None, game_data=None, out=OUT, files=FILES):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args(argv)
    game_data = Path(game_data or tw3k_paths.GAME_DATA)
    bad = 0
    for path, (pack, size, sha) in files.items():
        dest = Path(out) / path
        if dest.is_file() and dest.stat().st_size == size and hashlib.sha256(dest.read_bytes()).hexdigest() == sha:
            print(f'ok       {path}')
            continue
        if a.check:
            print(f'missing  {path}' if not dest.exists() else f'differs  {path}')
            bad += 1
            continue
        data = read_entry(game_data / pack, path)
        got = hashlib.sha256(data).hexdigest()
        if got != sha or len(data) != size:
            raise SystemExit(f'{path} in {pack}: sha256 {got} ({len(data)} bytes), recorded {sha} ({size} bytes); '
                             'the game was updated, so this vanilla reference changed. Not extracted.')
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + '.tmp')
        tmp.write_bytes(data)
        tmp.replace(dest)
        print(f'extracted {path} from {pack}')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
