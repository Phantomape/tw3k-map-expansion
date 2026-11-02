"""Output-identical fast versions of the polish array helpers.

propagate_nearest  frontier BFS instead of four full-array passes per step;
                   same 4-neighbour growth and the same direction priority
                   ((0,1),(0,-1),(1,0),(-1,0)) as native_campaign_climate.
grow_steps         4-neighbour ring distance (the PROTECT_FADE / LAKE_BOWL loops).
box_blur           native_campaign_climate.box_blur, run in independent strips
                   on a thread pool (same per-element float64 arithmetic).
"""
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

_POOL = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)
_DIRS = ((0, 1), (0, -1), (1, 0), (-1, 0))


def _bfs(seed, steps, values=None):
    """4-neighbour BFS from `seed`. Returns (near, dist) like propagate_nearest
    (near is None when values is None). A cell reached in step s takes the value of
    the first direction (in _DIRS order) whose neighbour was reached in step s-1."""
    th, tw = seed.shape
    big = np.iinfo(np.int32).max
    dist = np.full(th * tw, big, np.int32)
    front = np.flatnonzero(seed.ravel())
    dist[front] = 0
    near = None
    if values is not None:
        near = np.zeros(th * tw, values.dtype)
        near[front] = values.ravel()[front]
    for step in range(1, steps + 1):
        if not front.size:
            break
        fy, fx = np.divmod(front, tw)
        new = []
        for dy, dx in _DIRS:
            # cell c = f - (dy, dx) reads its neighbour f = c + (dy, dx)
            ok = np.ones(front.shape, bool)
            if dy > 0: ok &= fy >= dy
            elif dy < 0: ok &= fy < th + dy
            if dx > 0: ok &= fx >= dx
            elif dx < 0: ok &= fx < tw + dx
            src = front[ok]
            cand = src - (dy * tw + dx)
            free = dist[cand] == big
            cand = cand[free]
            if not cand.size:
                continue
            dist[cand] = step
            if near is not None:
                near[cand] = near[src[free]]
            new.append(cand)
        front = np.concatenate(new) if new else np.empty(0, np.int64)
    dist = dist.reshape(th, tw)
    return (near.reshape(th, tw) if near is not None else None), dist


def propagate_nearest(values, seed, steps):
    """native_campaign_climate.propagate_nearest, identical result."""
    near, dist = _bfs(np.asarray(seed, bool), steps, np.asarray(values))
    return near, dist


def grow_steps(mask, steps):
    """Ring index of the iterative 4-neighbour growth loop: 0 on mask, i for the ring
    added in iteration i (1..steps), int32 max beyond."""
    return _bfs(np.asarray(mask, bool), steps)[1]


def _box_axis(out, radius, axis, strip):
    n_other = out.shape[1 - axis]
    res = np.empty(out.shape, np.float32)

    def work(start):
        stop = min(start + strip, n_other)
        sel = (slice(None), slice(start, stop)) if axis == 0 else (slice(start, stop), slice(None))
        a = out[sel]
        pad = [(0, 0), (0, 0)]; pad[axis] = (radius + 1, radius)
        p = np.pad(a, pad, mode='edge'); c = np.cumsum(p, axis=axis, dtype=np.float64)
        k = 2 * radius + 1
        hi = c[k:] if axis == 0 else c[:, k:]
        lo = c[:-k] if axis == 0 else c[:, :-k]
        d = np.subtract(hi, lo); np.divide(d, k, out=d)
        res[sel] = d   # float64 -> float32 rounding, as .astype(np.float32)
    list(_POOL.map(work, range(0, n_other, strip)))
    return res


def box_blur(a, radius, strip=None):
    """Separable box blur of a 2-D float array (edges clamped); identical to
    native_campaign_climate.box_blur."""
    out = a.astype(np.float32)
    for axis in (0, 1):
        s = strip or (8 if axis == 0 else 4)   # tuned: cache-sized strips
        out = _box_axis(out, radius, axis, s)
    return out


def cm_write(b, a, bounds=None):
    """polish_campaign_terrain.cm_write (TABLE_INDEXED 16x16 blocks), vectorised;
    byte-identical output (bytearray, like native_height_field.rebuild)."""
    import struct
    h, w = a.shape
    padded = np.pad(a, ((0, (-h) % 16), (0, (-w) % 16)), mode='edge').astype('<u2', copy=False)
    ny, nx = padded.shape[0] // 16, padded.shape[1] // 16
    blk = np.ascontiguousarray(padded.reshape(ny, 16, nx, 16).transpose(0, 2, 1, 3)).reshape(-1, 256)
    uni = (blk == blk[:, :1]).all(axis=1)
    lengths = np.where(uni, 3, 513).astype(np.int64)
    n = len(lengths)
    offsets = np.zeros(n, np.int64)
    np.cumsum(lengths[:-1], out=offsets[1:])
    total = int(lengths.sum())
    raw = blk.view(np.uint8)
    col = np.arange(513)[None, :]
    parts = []
    step = 65536
    for s in range(0, n, step):
        e = min(s + step, n)
        full = np.empty((e - s, 513), np.uint8)
        full[:, 0] = np.where(uni[s:e], 0, 0x8f)
        full[:, 1:] = raw[s:e]
        parts.append(full[col < lengths[s:e, None]].tobytes())
    hd = bytearray(b); struct.pack_into('<2I', hd, 10, w, h)
    if bounds:
        struct.pack_into('<6f', hd, 26, *bounds)
    return (hd[:65] + struct.pack('<I', n) + offsets.astype('<u4').tobytes() + struct.pack('<I', n)
            + lengths.astype('<u2').tobytes() + struct.pack('<I', total) + b''.join(parts))


_CM_BITLEN = np.array([int(v).bit_length() for v in range(128)])   # palette index bits per head byte < 128
_CM_CHUNK = 8192                                                   # blocks per gather (index array ~35 MB)


def _cm_unpack(raw, bits):
    """(k, 32 bits) packed bytes -> (k, 256) little-endian `bits`-wide fields, as int.from_bytes(.., 'little') >> bits*i."""
    k = len(raw)
    u = np.unpackbits(raw, axis=1, bitorder='little').reshape(k, 256, bits)
    pad = np.zeros((k, 256, 8 if bits <= 8 else 16), np.uint8); pad[:, :, :bits] = u
    v = np.packbits(pad, axis=2, bitorder='little')
    return v[:, :, 0] if bits <= 8 else v.view('<u2')[:, :, 0]


def cm_blocks(b):
    """native_height_field.parse + decode of every block, vectorised: (width, height, (count, 256) uint16
    values in block order), with the same header, length and mode checks (an index past its palette or a
    base + offset over 65535 raises, as the Python decode / uint16 conversion does)."""
    import struct
    assert b[:10] == b'FASTBIN0\x03\x00' and b[52:65] == b'TABLE_INDEXED'
    w, h, bw, bh = struct.unpack_from('<4I', b, 10); assert (bw, bh) == (16, 16)
    n = struct.unpack_from('<I', b, 65)[0]; lp = 69 + n * 4
    offsets = np.frombuffer(b, '<u4', n, 69).astype(np.int64); assert struct.unpack_from('<I', b, lp)[0] == n
    lengths = np.frombuffer(b, '<u2', n, lp + 4).astype(np.int64); bp = lp + 4 + n * 2 + 4
    assert len(b) - bp == struct.unpack_from('<I', b, bp - 4)[0]
    buf = np.frombuffer(b, np.uint8); start = bp + offsets
    assert (lengths > 0).all() and (start + lengths <= len(b)).all()
    head = buf[start]; pal = head < 128
    assert ((head[~pal] & 0xf0) == 0x80).all()
    expect = np.where(pal, 1 + 2 * (head.astype(np.int64) + 1) + 32 * _CM_BITLEN[head & 127],
                      np.where(head == 0x8f, 513, 3 + 32 * ((head.astype(np.int64) & 15) + 1)))
    assert (lengths == expect).all()
    out = np.empty((n, 256), np.uint16)
    def gather(sel, at, size):
        return buf[start[sel, None] + at + np.arange(size)]
    for hv in np.unique(head).tolist():
        every = np.flatnonzero(head == hv)
        for c in range(0, len(every), _CM_CHUNK):
            sel = every[c:c + _CM_CHUNK]
            if hv < 128:                              # palette of hv + 1 values, index fields of bit_length(hv)
                count, bits = hv + 1, int(_CM_BITLEN[hv])
                palette = gather(sel, 1, 2 * count).view('<u2')
                idx = _cm_unpack(gather(sel, 1 + 2 * count, 32 * bits), bits) if bits else np.zeros((len(sel), 256), np.uint8)
                out[sel] = np.take_along_axis(palette, idx.astype(np.intp), 1)
            elif hv == 0x8f:                          # 16 bits: raw values
                out[sel] = gather(sel, 1, 512).view('<u2')
            else:                                     # base + offset fields of (hv & 15) + 1 bits
                bits = (hv & 15) + 1
                v = gather(sel, 1, 2).view('<u2').astype(np.int64) + _cm_unpack(gather(sel, 3, 32 * bits), bits)
                if v.max() > 65535:
                    raise OverflowError('TABLE_INDEXED base + offset over 65535')
                out[sel] = v
    return w, h, out


def cm_read(b):
    """The (height, width) uint16 grid of a TABLE_INDEXED 16x16 compressed_map, vectorised; identical to
    the per-block native_height_field.decode loop into a 16-padded grid cropped to [:h, :w]."""
    w, h, v = cm_blocks(b)
    rows, cols = -(-h // 16), -(-w // 16); assert len(v) == rows * cols, (len(v), rows, cols)
    return np.ascontiguousarray(v.reshape(rows, cols, 16, 16).transpose(0, 2, 1, 3).reshape(rows * 16, cols * 16)[:h, :w])


def max_filter_mask(mask, size):
    """np.asarray(Image.fromarray(mask.astype(uint8)*255).filter(MaxFilter(size)))>0:
    PIL pads by edge replication, so this is a square dilation with the window
    clipped at the border; separable shifted ORs give the same mask."""
    assert size % 2 == 1
    r = size // 2
    out = np.asarray(mask, bool)
    for axis in (0, 1):
        acc = out.copy()
        for k in range(1, min(r, out.shape[axis] - 1) + 1):
            if axis == 0:
                acc[:-k] |= out[k:]; acc[k:] |= out[:-k]
            else:
                acc[:, :-k] |= out[:, k:]; acc[:, k:] |= out[:, :-k]
        out = acc
    return out


def pack_entries(path):
    """analyze_reference.inventory(path, False)['entries'] with the same checks,
    parsing the PFH5 index with bytes.find instead of one read per byte."""
    import struct
    with open(path, 'rb') as stream:
        head = stream.read(4)
        if head[:3] == b'MFH':
            stream.seek(8); head = stream.read(4)
        if head != b'PFH5':
            raise ValueError(f'Only PFH5 supported; got {head!r}')
        flags, dep_count, dep_size, count, index_size, timestamp = struct.unpack('<6I', stream.read(24))
        if flags & ~0x4F:
            raise ValueError('Encrypted/extended/unknown pack flags unsupported')
        deps = stream.read(dep_size).split(b'\0')[:-1]
        if len(deps) != dep_count:
            raise ValueError('Dependency count mismatch')
        index = stream.read(index_size)
        if len(index) != index_size:
            raise ValueError('Truncated file')
        offset = stream.tell()
    entries = []; pos = 0; extra = 4 if flags & 0x40 else 0
    for _ in range(count):
        size = struct.unpack_from('<I', index, pos)[0]; pos += 4 + extra
        compressed = index[pos]; pos += 1
        if compressed not in (0, 1):
            raise ValueError('Invalid compression flag')
        end = index.find(b'\0', pos)
        if end < 0:
            raise ValueError('Truncated file')
        entries.append(dict(path=index[pos:end].decode('utf-8').replace(chr(92), '/'), size=size, offset=offset, compressed=bool(compressed)))
        pos = end + 1; offset += size
    import os
    if pos != index_size or offset != os.path.getsize(path):
        raise ValueError('Index/data bounds do not match pack size')
    if len({e['path'] for e in entries}) != count:
        raise ValueError('Duplicate entry paths')
    return entries
