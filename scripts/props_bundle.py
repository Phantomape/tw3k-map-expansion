"""Campaign prop bundles: terrain/campaigns/<map>/global_props.bin and its root prefab list.

A bundle is u32 count, count x (NUL-terminated asset key, u32 offset), then the asset
bodies; an asset runs to the next larger offset, the last to the end. Its last asset is
the root prefab list terrain/campaigns/<map>/bmd_objects.bin (read_root_prefabs.decode);
root records and the child records of a parent bmd_objects.<region>.<n>.bin are u16 kind,
u16 length, path, ..., owning region string, ...

The prop steps (hehuang_lake_prop, frontier_mountain_props, coast_props, landmarks_props,
place_ambient) unpack the staged bundle, clone vanilla records under new paths and regions,
rebuild the root list and the bundle and write the stage:

  es, assets = unbundle(data, rootkey)                  index and {key: bytes}
  rebind(blob, rec, region, path) / rename(blob, path)  cloned root / child records
  updated, chk = prefab_list(root, cp, rs, kept, new)   root list (or parent): kept records + new record bytes
  bundled = rebundle(es, assets, rootkey, updated, new_assets)
  write_stage(stage, pre, bundled, updated, new_assets) bundle, loose root list, new loose assets
"""
import os
import struct

from parse_global_props import parse
from read_root_prefabs import decode, string


def embedded(data):
    """({key: bytes}, [(key, offset)] index, body start) of a bundle (linear: each asset ends at the next larger offset)."""
    entries, base = parse(data)
    offsets = sorted({o for _, o in entries}) + [len(data) - base]
    ends = dict(zip(offsets, offsets[1:]))
    return {k: data[base + o:base + ends[o]] for k, o in entries}, entries, base


def unbundle(data, rootkey=None):
    """(index, {key: bytes}) of a bundle, in index order; with rootkey, checks the root prefab list is the last asset."""
    assets, es, _ = embedded(data)
    if rootkey is not None:
        assert dict(es)[rootkey] == max(o for _, o in es), 'root prefab list is not the last bundle asset'
    return es, assets


def rename(blob, newpath):
    """Root/child record (u16 kind, u16 length, path ...) with its path replaced."""
    blob = bytearray(blob)
    old = struct.unpack_from('<H', blob, 2)[0]
    name = newpath.encode()
    blob[2:4 + old] = struct.pack('<H', len(name)) + name
    return bytes(blob)


def rebind(blob, rec, region, newpath):
    """Bytes blob of decoded root/child record rec with its owning region and path replaced (region first: it follows the path)."""
    part = bytearray(blob)
    a = rec['region_pos'] - rec['start']
    _, end = string(part, a)
    key = region.encode()
    part[a:end] = struct.pack('<H', len(key)) + key
    return rename(bytes(part), newpath)


def prefab_list(blob, cp, rs, kept, new):
    """Root prefab list or parent blob (decode: count at cp, records rs) holding the kept records of rs,
    then the new record bytes; (bytes, its decoded records)."""
    head = bytearray(blob[:rs[0]['start']])
    struct.pack_into('<I', head, cp, len(kept) + len(new))
    out = bytes(head) + b''.join(blob[r['start']:r['end']] for r in kept) + b''.join(new) + blob[rs[-1]['end']:]
    _, chk = decode(out)
    assert len(chk) == len(kept) + len(new)
    return out, chk


def rebundle(es, assets, rootkey, root, new, replace=False):
    """Bundle bytes: the assets of index es still in assets in their old order, the new assets sorted
    by key, the root prefab list root last. replace: a new asset whose key is in assets takes the old
    one's place (others still follow sorted). Parse round-trip checked; the asset count is its first u32."""
    order = [k for k, _ in sorted(es, key=lambda v: v[1]) if k != rootkey and k in assets]
    if replace:
        bodies = [(k, new.get(k, assets[k])) for k in order] + [(k, v) for k, v in sorted(new.items()) if k not in assets]
    else:
        bodies = [(k, assets[k]) for k in order] + sorted(new.items())
    bodies.append((rootkey, root))
    cursor, index = 0, []
    for k, body in bodies:
        index.append((k, cursor))
        cursor += len(body)
    assert len({k for k, _ in index}) == len(index), 'duplicate bundle asset'
    header = struct.pack('<I', len(index)) + b''.join(k.encode() + b'\0' + struct.pack('<I', o) for k, o in index)
    out = header + b''.join(b for _, b in bodies)
    assert len(parse(out)[0]) == len(index)
    return out


def write_atomic(path, data):
    """path replaced by data through a temporary file in its own directory (os.replace): never half written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def write_stage(stage, pre, bundled, root, new):
    """The bundle, the loose root prefab list and the new loose assets ({key: bytes}) under stage, each atomically."""
    write_atomic(stage / (pre + 'global_props.bin'), bundled)
    write_atomic(stage / (pre + 'bmd_objects.bin'), root)
    for k, v in new.items():
        write_atomic(stage / k, v)
