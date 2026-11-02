"""Read/write a Three Kingdoms map.hex (version 0.19) without losing any field.

Layout: u32 major, u32 minor, 8 opaque bytes, game, campaign_map, seven string
lists (land/sea regions, land/sea ground types, climates, attritions, areas of
interest), two u32 colour palettes (editor display only), u32 width, u32 height,
width*height 16-byte cells (rows south-up), CRC32 of everything before it.

Cell bits used by the pipeline:
  byte0 bits 0-1  land/sea (1 = sea); bits 3-7 region id+1, low 5 bits
  byte1           region id+1, high bits
  byte2 bit 3     impassable; high nibble settlement slot (1 primary, 2 port)
  byte5 >> 4, byte6 & 7  ground type index
"""
import io
import os
import struct
import zlib

import numpy as np

LISTS = ('land_regions', 'sea_regions', 'land_ground_types', 'sea_ground_types', 'climates', 'attritions', 'areas_of_interest')


def _u32(s):
    return struct.unpack('<I', s.read(4))[0]


def _str(s):
    return s.read(_u32(s)).decode('ascii')


def read(path):
    data = path.read_bytes()
    if zlib.crc32(data[:-4]) & 0xFFFFFFFF != struct.unpack('<I', data[-4:])[0]:
        raise ValueError(f'{path}: CRC32 mismatch')
    s = io.BytesIO(data)
    m = {'major': _u32(s), 'minor': _u32(s), 'opaque': s.read(8), 'game': _str(s), 'campaign_map': _str(s)}
    if (m['major'], m['minor']) != (0, 19):
        raise ValueError('Expected a version 0.19 map')
    for name in LISTS:
        m[name] = [_str(s) for _ in range(_u32(s))]
    m['palettes'] = []
    for _ in range(2):
        n = _u32(s)
        m['palettes'].append(s.read(4 * n))
    w, h = _u32(s), _u32(s)
    m['cells'] = np.frombuffer(s.read(w * h * 16), np.uint8).reshape(h, w, 16).copy()
    if len(s.read()) != 4:
        raise ValueError('Unexpected trailing data')
    return m


def write(m, path):
    out = io.BytesIO()

    def u32(v):
        out.write(struct.pack('<I', v))

    def st(v):
        b = v.encode('ascii')
        u32(len(b))
        out.write(b)
    u32(m['major']); u32(m['minor']); out.write(m['opaque']); st(m['game']); st(m['campaign_map'])
    for name in LISTS:
        u32(len(m[name]))
        for v in m[name]:
            st(v)
    for pal in m['palettes']:
        u32(len(pal) // 4)
        out.write(pal)
    h, w = m['cells'].shape[:2]
    u32(w); u32(h)
    out.write(np.ascontiguousarray(m['cells'], np.uint8).tobytes())
    body = out.getvalue()
    # Through a temporary file in the same directory: an interrupted write never leaves a torn map.
    tmp = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    try:
        tmp.write_bytes(body + struct.pack('<I', zlib.crc32(body) & 0xFFFFFFFF))
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def region_ids(cells):
    return (cells[..., 1].astype(np.int32) << 5) + (cells[..., 0].astype(np.int32) >> 3) - 1


# Byte 15: region-border edge mask. Bit per hex side (add_inland_lake.EDGE_BITS order: N, S, NE, SE,
# NW, SW in its neighbours()), set where the neighbour lies in another region; an off-map side stays
# clear. Vanilla 3k_dlc07 map.hex follows this rule on every hex (checked 2026-10-24).
BORDER_SIDES = ((4, 0, 1, 0, 1), (32, 0, -1, 0, -1), (8, 1, 0, 1, 1), (16, 1, -1, 1, 0),
                (128, -1, 0, -1, 1), (64, -1, -1, -1, 0))   # (bit, dq, dr even column, dq, dr odd column)


def border_masks(cells):
    """(H, W) uint8 byte-15 masks recomputed from the region ids of cells."""
    ids = region_ids(cells)
    h, w = ids.shape
    out = np.zeros((h, w), np.uint8)
    odd = (np.arange(w) % 2 == 1)[None, :]
    for bit, dqe, dre, dqo, dro in BORDER_SIDES:
        for par, dq, dr in ((0, dqe, dre), (1, dqo, dro)):
            other = np.zeros((h, w), bool)
            r0, r1 = max(0, -dr), min(h, h - dr)
            c0, c1 = max(0, -dq), min(w, w - dq)
            other[r0:r1, c0:c1] = ids[r0 + dr:r1 + dr, c0 + dq:c1 + dq] != ids[r0:r1, c0:c1]
            out |= np.where(other & (odd if par else ~odd), np.uint8(bit), np.uint8(0))
    return out


def set_region_ids(cells, ids):
    v = np.asarray(ids, np.int32) + 1
    if v.max() >= 1 << 13:
        raise ValueError('Region id out of range')
    cells[..., 0] = (cells[..., 0] & 7) | ((v & 31) << 3).astype(np.uint8)
    cells[..., 1] = (v >> 5).astype(np.uint8)
