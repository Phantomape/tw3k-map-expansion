"""Codec for vanilla's campaign river-water models (rigid_model_v2, mesh type 68, 48-byte vertices).

terrain/campaigns/3k_dlc07_main_map/models/river_<n>.wsmodel.rigid_model_v2 (vanilla terrain.pack)
is an RMV2 v8 file with one LOD holding one mesh:

  file header        0..140   b'RMV2', u32 version 8, u32 lod count 1, 128-byte skeleton name (empty)
  LOD header       140..168   u32 mesh count 1, u32 vertex bytes, u32 index bytes, u32 mesh start (168),
                              f32 visibility distance (1e6), u32 authored LOD 0, u8[4] quality (0,255,255,255)
  mesh header  m+0..m+940     u16 material type 68 ('default'), u16 flags 0, u32 mesh block size,
                              u32 vertex offset (940), u32 vertex count, u32 index offset, u32 index count,
                              f32[6] world bounds (min xyz, max xyz),
                              32 bytes shader params ('rigid_default' + bytes the exporter left as is),
                              u16 vertex format 13, 32-byte mesh name ('River'), 256-byte texture dir,
                              256-byte filters, 2 opaque bytes, f32[3] pivot (= bounds centre),
                              3 x f32[12] identity 3x4 matrices, i32 matrix index -1, i32 parent -1,
                              u32 attachment count 0, u32 texture count 0, 140 opaque bytes
  vertices      48 bytes each f32[4] local position (x, y, z, w=1) relative to the pivot,
                              f32[4] texcoords: uv0 (u, v) = flow coordinates, uv1 (u, v) = second set,
                              u8[4] normal, u8[4] tangent, u8[4] bitangent: bytes (z, y, x, 0), each
                              b / 127.5 - 1 (pack_dir/unpack_dir), u8[4] colour (0)
  indices       u16 triangle list

The engine places a river model by its pivot (repair_campaign_river_roots.py: root translations
are not applied to river instances): world = local + pivot. Opaque header bytes are kept
verbatim, so decode -> encode reproduces every vanilla river model byte for byte
(audit_river_water_codec.py).

Direction byte order (audit_river_water_mesh_look.py, 2026-10-24): the three packed directions store
z first. Read as (z, y, x), vanilla's vertex normals match their faces (area-weighted face normal of
tilted vertices: p50 0.5 deg; read as (x, y, z): 7.5 deg), the tangent is dP/dv (along the river,
upstream; p50 8-11 deg) and the bitangent dP/du (across, to the right of upstream; p50 4-5 deg), so
N.(T x B) = +1. Written as (x, y, z), a frontier normal tilted to +x showed tilted to +z, and the
wave-normal frame was mirrored across the x = z diagonal (right only for rivers flowing along z).
"""
import struct

import numpy as np

MAGIC = b'RMV2'
FILE_HEADER = 140
LOD_HEADER = 28
MESH_TYPE = 68
VERTEX_BYTES = 48
MESH_HEADER = 940
VERTEX = np.dtype([('pos', '<f4', 4), ('uv', '<f4', 4), ('normal', 'u1', 4), ('tangent', 'u1', 4),
                   ('bitangent', 'u1', 4), ('colour', 'u1', 4)])
assert VERTEX.itemsize == VERTEX_BYTES
U16_LIMIT = 65535

# Mesh header layout (offsets from the mesh start).
_BOUNDS = 24
_SHADER = 48
_FORMAT = 80
_NAME = 82
_TEXDIR = 114
_FILTERS = 370
_PAD = 626
_PIVOT = 628
_MATRICES = 640
_INDICES = 784          # i32 matrix index, i32 parent index, u32 attachments, u32 textures
_TAIL = 800


def _cstr(raw):
    return raw.split(b'\0', 1)[0].decode('ascii')


def decode(data):
    """dict of a type-68 single-mesh RMV2: header fields, opaque byte runs, vertices (VERTEX) and
    triangles ((n, 3) u16)."""
    data = bytes(data)
    assert data[:4] == MAGIC, 'not an RMV2 file'
    version, lods = struct.unpack_from('<2I', data, 4)
    assert version == 8 and lods == 1, (version, lods)
    meshes, vbytes, ibytes, start, distance, authored = struct.unpack_from('<4IfI', data, FILE_HEADER)
    quality = data[FILE_HEADER + 24:FILE_HEADER + 28]
    assert meshes == 1 and start == FILE_HEADER + LOD_HEADER, (meshes, start)
    m = start
    typ, flags, size, vo, vc, io, ic = struct.unpack_from('<HH5I', data, m)
    assert typ == MESH_TYPE and vo == MESH_HEADER and io == vo + vc * VERTEX_BYTES, (typ, vo, vc, io)
    assert vbytes == vc * VERTEX_BYTES and ibytes == ic * 2 and size == io + ic * 2 and m + size == len(data)
    assert ic % 3 == 0
    matrix_index, parent_index, attachments, textures = struct.unpack_from('<2i2I', data, m + _INDICES)
    assert attachments == 0 and textures == 0, (attachments, textures)
    out = {
        'skeleton': data[12:FILE_HEADER],
        'distance': distance, 'authored_lod': authored, 'quality': quality,
        'flags': flags,
        'bounds': struct.unpack_from('<6f', data, m + _BOUNDS),
        'shader': data[m + _SHADER:m + _FORMAT],
        'vertex_format': struct.unpack_from('<H', data, m + _FORMAT)[0],
        'name': data[m + _NAME:m + _TEXDIR],
        'texture_dir': data[m + _TEXDIR:m + _FILTERS],
        'filters': data[m + _FILTERS:m + _PAD],
        'pad': data[m + _PAD:m + _PIVOT],
        'pivot': struct.unpack_from('<3f', data, m + _PIVOT),
        'matrices': struct.unpack_from('<36f', data, m + _MATRICES),
        'matrix_index': matrix_index, 'parent_index': parent_index,
        'tail': data[m + _TAIL:m + MESH_HEADER],
        'vertices': np.frombuffer(data, VERTEX, vc, m + vo).copy(),
        'triangles': np.frombuffer(data, '<u2', ic, m + io).reshape(-1, 3).copy(),
    }
    return out


def encode(model):
    """RMV2 bytes of a decode()-shaped dict; sizes and offsets follow from the vertices and triangles."""
    v = np.ascontiguousarray(model['vertices'], VERTEX)
    t = np.ascontiguousarray(model['triangles'], '<u2').reshape(-1)
    vc, ic = len(v), len(t)
    assert 0 < vc <= U16_LIMIT + 1 and ic % 3 == 0 and (not ic or int(t.max()) < vc)
    vo, io = MESH_HEADER, MESH_HEADER + vc * VERTEX_BYTES
    size = io + ic * 2
    head = bytearray(FILE_HEADER + LOD_HEADER + MESH_HEADER)
    head[:12] = MAGIC + struct.pack('<2I', 8, 1)
    assert len(model['skeleton']) == FILE_HEADER - 12
    head[12:FILE_HEADER] = model['skeleton']
    struct.pack_into('<4IfI', head, FILE_HEADER, 1, vc * VERTEX_BYTES, ic * 2, FILE_HEADER + LOD_HEADER,
                     model['distance'], model['authored_lod'])
    head[FILE_HEADER + 24:FILE_HEADER + 28] = model['quality']
    m = FILE_HEADER + LOD_HEADER
    struct.pack_into('<HH5I', head, m, MESH_TYPE, model['flags'], size, vo, vc, io, ic)
    struct.pack_into('<6f', head, m + _BOUNDS, *model['bounds'])
    for off, end, key in ((_SHADER, _FORMAT, 'shader'), (_NAME, _TEXDIR, 'name'), (_TEXDIR, _FILTERS, 'texture_dir'),
                          (_FILTERS, _PAD, 'filters'), (_PAD, _PIVOT, 'pad'), (_TAIL, MESH_HEADER, 'tail')):
        assert len(model[key]) == end - off, key
        head[m + off:m + end] = model[key]
    struct.pack_into('<H', head, m + _FORMAT, model['vertex_format'])
    struct.pack_into('<3f', head, m + _PIVOT, *model['pivot'])
    struct.pack_into('<36f', head, m + _MATRICES, *model['matrices'])
    struct.pack_into('<2i2I', head, m + _INDICES, model['matrix_index'], model['parent_index'], 0, 0)
    return bytes(head) + v.tobytes() + t.tobytes()


def unpack_dir(b):
    """World (x, y, z) vectors of packed u8[4] normals/tangents/bitangents (bytes z, y, x; b / 127.5 - 1)."""
    return b[:, 2::-1].astype(np.float64) / 127.5 - 1


def pack_dir(d):
    """u8[4] of world (x, y, z) unit vectors d (n, 3) as vanilla packs them: bytes (z, y, x, 0)."""
    d = np.asarray(d, np.float64)
    d = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
    out = np.zeros((len(d), 4), np.uint8)
    out[:, :3] = np.clip(np.rint((d[:, ::-1] + 1) * 127.5), 0, 255).astype(np.uint8)
    return out


def world_vertices(model):
    """(n, 3) float64 world positions (local + pivot)."""
    return model['vertices']['pos'][:, :3].astype(np.float64) + np.asarray(model['pivot'], np.float64)


def wsmodel(geometry, material):
    """Vanilla river .wsmodel text (byte layout of river_<n>.wsmodel: two-space indents, CRLF, no final newline)."""
    return ('<model version="1">\r\n'
            f'  <geometry>{geometry}</geometry>\r\n'
            '  <materials>\r\n'
            f'    <material lod_index="0" part_index="0">{material}</material>\r\n'
            '  </materials>\r\n'
            '</model>').encode('ascii')
