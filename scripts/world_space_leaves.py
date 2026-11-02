"""Leaf instances are placed in world space: the engine ignores root translations.

Every leaf instance sits at its own position, whatever its root's translation
(live checks: the great wall, Mount Tai and its walkway buildings, and town
scenery such as Donglai's terraces all rendered at vanilla coordinates until
the translation was written into the leaves; live-check-world-space-props,
live-check-all-leaves). Any root translation, the native shift or a scene
moved to a new town, is therefore written into all of its leaves.

Leaves are FASTBIN0 v35 scene files. Besides the model table (leaf_records)
they hold particle emitters and composite scenes (.csc fauna, living-campaign
and building-level scenes). Nothing in the data places these by their root:
every vanilla root is identity and every leaf, these included, is authored at
its town's absolute coordinates, one copy per place. Their 3x4 transform ends
in the position and they carry no bounds or second copy of it; bake() moves
them with the models.

The same rule is applied by two older steps: translate_campaign_lake_instances.py
(lake leaves, with their own pack audit) and repair_campaign_river_roots.py
(river roots zeroed, translation carried into their non-water leaves). bake()
skips lakes and zero-translation roots so nothing moves twice.
"""
import math
import re
import struct

from read_root_prefabs import decode
from repair_campaign_river_roots import embedded
from translate_campaign_lake_instances import records as leaf_records

FLAG_OFFSET = 101   # leaf flag byte; 1 on mountains, vegetation, area-of-interest models

# Particle emitter: u16 6, name, 12f transform (3x3, position), 18 bytes,
# height mode, 28 bytes. Composite scene: u16 6, 12f transform, .csc path,
# height mode, 35 bytes. Each list is u16 1, u32 count, then the records.
EMITTER = re.compile(rb'\x06\x00(?=[\x01-\x7f]\x00[A-Za-z0-9_])')
SCENE = re.compile(rb'composite_scene/[A-Za-z0-9_./-]+\.csc', re.I)


def is_world_space(blob, record):
    return blob[record['position_offset'] - 60 + FLAG_OFFSET] == 1 and '/props/lakes/' not in record['model'].lower()


def _string(b, p):
    n = struct.unpack_from('<H', b, p)[0]
    return b[p + 2:p + 2 + n], p + 2 + n


def _transform(b, p):
    m = struct.unpack_from('<12f', b, p)
    return m if all(math.isfinite(x) and abs(x) < 10000 for x in m) else None


def _emitter(b, s):
    """(record, end) of the particle emitter starting at s, or None."""
    name, p = _string(b, s + 2)
    if not re.fullmatch(rb'[A-Za-z0-9_]+', name) or p + 68 > len(b):
        return None
    m = _transform(b, p)
    mode, end = _string(b, p + 48 + 18)
    if m is None or not mode.startswith(b'BHM_') or end + 28 > len(b):
        return None
    return dict(kind='emitter', model=name.decode(), matrix=m, position_offset=p + 36), end + 28


def _scene(b, s):
    """(record, end) of the composite scene starting at s, or None."""
    m = _transform(b, s + 2)
    path, p = _string(b, s + 50)
    if m is None or not SCENE.fullmatch(path):
        return None
    mode, end = _string(b, p)
    if not mode.startswith(b'BHM_') or end + 35 > len(b):
        return None
    return dict(kind='scene', model=path.decode(), matrix=m, position_offset=s + 38), end + 35


def _chain(b, starts, parse):
    # Every record found belongs to one list: u16 1, u32 count, then exactly
    # count records back to back.
    if not starts:
        return []
    s = min(starts)
    version, count = struct.unpack_from('<HI', b, s - 6)
    assert version == 1 and count == len(starts), ('list header', s, version, count, len(starts))
    out, p = [], s
    for _ in range(count):
        assert p in starts, ('record chain', p)
        rec, p = parse(b, p)
        out.append(rec)
    return out


def placed_records(b):
    """Particle emitters and composite scenes of a leaf, with their position offsets."""
    emitters = {m.start() for m in EMITTER.finditer(b) if _emitter(b, m.start())}
    scenes = {m.start() - 52 for m in SCENE.finditer(b) if m.start() >= 58
              and struct.unpack_from('<H', b, m.start() - 52)[0] == 6
              and struct.unpack_from('<H', b, m.start() - 2)[0] == len(m.group())}
    assert all(_scene(b, s) for s in scenes), 'composite scene record'
    return _chain(b, emitters, _emitter) + _chain(b, scenes, _scene)


def bake(bundle):
    """Bundle with each root's translation added to every non-lake leaf instance.

    Model-table instances, particle emitters and composite scenes all move.
    A root's leaves are the children listed in its parent, each under the root's
    own stem and owned by no other translated root; anything else is refused.
    Returns (bundle, {leaf path: new bytes}, instances moved). Same length.
    """
    assets, entries, base = embedded(bundle)
    key = next(k for k, _ in entries if k.endswith('/bmd_objects.bin'))
    body = bytearray(bundle)
    offsets = dict(entries)
    changed, moved, owner = {}, 0, {}
    for rec in decode(assets[key])[1]:
        t = rec['matrix'][12:15]
        if max(abs(v) for v in t) < 1e-6:
            continue
        # A root's leaves are the children its parent lists; they must sit under
        # its own stem, and one leaf cannot carry two roots' translations.
        stem = rec['path'][:-4] + '.'
        for k in (c['path'] for c in decode(assets[rec['path']])[1]):
            if not k.startswith(stem) or owner.setdefault(k, rec['path']) != rec['path']:
                raise RuntimeError(f"Leaf {k} of {rec['path']} is not its own (stem or shared with {owner.get(k)}); give the scene its own leaf copies")
            b = assets[k]
            try:
                rows = leaf_records(b)
            except Exception:
                rows = []   # no model table: emitters and composite scenes only
            rows += placed_records(b)
            out = bytearray(changed.get(k, b))
            for r in rows:
                if '/props/lakes/' in r['model'].lower():
                    continue
                p = r['position_offset']
                x, y, z = struct.unpack_from('<3f', out, p)
                struct.pack_into('<3f', out, p, x + t[0], y + t[1], z + t[2])
                moved += 1
            if out != b:
                changed[k] = bytes(out)
    for k, b in changed.items():
        body[base + offsets[k]:base + offsets[k] + len(b)] = b
    return bytes(body), changed, moved
