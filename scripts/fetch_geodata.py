"""Fetch the published geographic source data the build reads (data/geodata-sources.json) into work/geodata/,
each file checked against its recorded sha256 and size.

  python scripts/fetch_geodata.py              fetch what is missing or wrong (required files only)
  python scripts/fetch_geodata.py --optional   also the optional ones (comparison previews)
  python scripts/fetch_geodata.py --check      only report; exit 1 if a required file is missing or differs

A download goes to <file>.part and is renamed only when its sha256 matches. A mismatch (the publisher replaced
the file, e.g. a newer Natural Earth release under the same URL) stops with both hashes and leaves nothing in
place: the derived caches and the bake were checked against the recorded version. A file already present with
the right hash is not downloaded again. The root is the repository root (TW3K_ROOT overrides it).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import urllib.request
import zipfile

ROOT = Path(os.environ.get('TW3K_ROOT', Path(__file__).resolve().parents[1]))
SOURCES = Path(__file__).resolve().parents[1] / 'data/geodata-sources.json'
CHUNK = 1 << 20


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(CHUNK), b''):
            h.update(b)
    return h.hexdigest()


def state(entry, root=ROOT):
    """'ok', 'missing' or 'differs' for the file of one sources entry."""
    p = root / entry['path']
    if not p.is_file():
        return 'missing'
    if p.stat().st_size != entry['bytes'] or sha256(p) != entry['sha256']:
        return 'differs'
    return 'ok'


def fetch(name, entry, root=ROOT, opener=urllib.request.urlopen):
    """Download one entry to <path>.part, check it, rename; returns the final path or raises SystemExit."""
    dest = root / entry['path']
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + '.part')
    print(f'{name}: {entry["url"]}', flush=True)
    try:
        with opener(entry['url']) as r, open(part, 'wb') as f:
            shutil.copyfileobj(r, f, CHUNK)
        got = sha256(part)
        if got != entry['sha256'] or part.stat().st_size != entry['bytes']:
            raise SystemExit(f'{name}: sha256 {got} ({part.stat().st_size} bytes) differs from the recorded '
                             f'{entry["sha256"]} ({entry["bytes"]} bytes); the publisher may have replaced the '
                             'file. Not installed.')
        os.replace(part, dest)
    finally:
        part.unlink(missing_ok=True)
    if entry.get('extract_to'):
        with zipfile.ZipFile(dest) as z:
            z.extractall(root / entry['extract_to'])
    return dest


def main(argv=None, root=ROOT, sources=SOURCES, opener=urllib.request.urlopen):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--optional', action='store_true')
    a = ap.parse_args(argv)
    files = json.loads(Path(sources).read_text(encoding='utf-8'))['files']
    bad = []
    for name, e in files.items():
        if e.get('optional') and not a.optional:
            continue
        s = state(e, root)
        print(f'{name:12s} {s:8s} {e["path"]}', flush=True)
        if s == 'ok':
            continue
        if a.check:
            bad.append(name)
            continue
        if s == 'differs':
            raise SystemExit(f'{name}: {e["path"]} exists but differs from the recorded file; move it aside '
                             'and run again')
        fetch(name, e, root, opener)
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
