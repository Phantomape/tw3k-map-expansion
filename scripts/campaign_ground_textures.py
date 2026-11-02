"""Ground texture array of the campaign global blend, decoded from the vanilla packs.

The campaign far-terrain shader (rigidterrain.hlsl ps_global_blend) reads
global_map/global_blend.dds as R = material slot, G = climate band, looks up
g_global_blend_indices[R + G*K + season] (an engine-built cbuffer table) to get
a layer of the base-colour texture array listed in global_map/texture_arrays.xml,
and samples that layer at world.xz / g_blend_pixel_scale_array[layer].
This module only reads the vanilla packs (never writes them), decodes the
BC1/BC3 textures and caches the decoded arrays under output/offline-view/cache.
"""
import tw3k_paths
import pickle, re, struct
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GAME = tw3k_paths.GAME_DATA
CACHE = ROOT / 'output/offline-view/cache'
PACKS = ('terrain.pack', 'terrain2.pack', 'terrain3.pack', 'terrain4.pack', 'terrain5.pack',
         'fast.pack', 'data.pack', 'vegetation.pack')
VARIANT = '3k_dlc07_main_map'


class VanillaPacks:
    """Read-only index of the vanilla packs (cached on disk; first pack wins)."""
    def __init__(self):
        CACHE.mkdir(parents=True, exist_ok=True)
        cache = CACHE / 'pack_index.pkl'
        if cache.exists():
            self.index = pickle.loads(cache.read_bytes())
        else:
            from analyze_reference import inventory
            self.index = {}
            for n in PACKS:
                for e in inventory(GAME / n, False)['entries']:
                    self.index.setdefault(e['path'], (str(GAME / n), e['offset'], e['size'], e['compressed']))
            cache.write_bytes(pickle.dumps(self.index))

    def read(self, key):
        p, o, s, c = self.index[key]
        assert not c, key
        with open(p, 'rb') as f:
            f.seek(o)
            return f.read(s)

    def keys(self, prefix, suffix=''):
        return sorted(k for k in self.index if k.startswith(prefix) and k.endswith(suffix))


def _bc_colour(blocks):
    """Decode the BC1-style colour part of (n, 8) uint8 blocks -> (n, 16, 3) uint8."""
    c0 = blocks[:, 0].astype(np.uint32) | (blocks[:, 1].astype(np.uint32) << 8)
    c1 = blocks[:, 2].astype(np.uint32) | (blocks[:, 3].astype(np.uint32) << 8)
    bits = (blocks[:, 4].astype(np.uint32) | (blocks[:, 5].astype(np.uint32) << 8) |
            (blocks[:, 6].astype(np.uint32) << 16) | (blocks[:, 7].astype(np.uint32) << 24))

    def rgb(c):
        return np.stack([((c >> 11) & 31) * 255 // 31, ((c >> 5) & 63) * 255 // 63, (c & 31) * 255 // 31],
                        -1).astype(np.int32)
    a, b = rgb(c0), rgb(c1)
    four = (c0 > c1)[:, None]
    p2 = np.where(four, (2 * a + b) // 3, (a + b) // 2)
    p3 = np.where(four, (a + 2 * b) // 3, 0)
    pal = np.stack([a, b, p2, p3], 1)
    idx = (bits[:, None] >> (2 * np.arange(16, dtype=np.uint32))[None]) & 3
    return np.take_along_axis(pal, idx[..., None].astype(np.int64), 1).astype(np.uint8)


def decode_dds(data, mip=0):
    """RGB uint8 array of one mip of a BC1/BC3 (DX10 or legacy) DDS (alpha ignored)."""
    h, w = struct.unpack_from('<2I', data, 12)
    fourcc = data[84:88]
    off = 128
    if fourcc == b'DX10':
        fmt = struct.unpack_from('<I', data, 128)[0]
        off += 20
        bs = 8 if fmt in (70, 71, 72) else 16
    else:
        bs = 8 if fourcc == b'DXT1' else 16
    for _ in range(mip):
        off += max(1, w // 4) * max(1, h // 4) * bs
        w, h = max(1, w // 2), max(1, h // 2)
    bw, bh = max(1, w // 4), max(1, h // 4)
    blk = np.frombuffer(data, np.uint8, bw * bh * bs, off).reshape(-1, bs)
    col = _bc_colour(blk[:, bs - 8:])
    img = col.reshape(bh, bw, 4, 4, 3).transpose(0, 2, 1, 3, 4).reshape(bh * 4, bw * 4, 3)
    return img[:h, :w]


def texture_array(packs=None, variant=VARIANT):
    """(group names, base-colour paths) in texture-array layer order."""
    packs = packs or VanillaPacks()
    x = packs.read(f'terrain/campaigns/{variant}/global_map/texture_arrays.xml').decode()
    names = re.findall(r'<group>(.*?)</group>', x)
    paths = [t.replace('\\', '/').lower() for t in re.findall(r'<texture>(.*?)</texture>', x)]
    assert len(names) == len(paths)
    return names, paths


SEASONS = ('spring', 'summer', 'harvest', 'autumn', 'winter')
# Fallback when a group has no texture for the requested season (assumption:
# only the farm groups ship 'harvest'; the rest are taken as summer then).
SEASON_FALLBACK = {'harvest': ('harvest', 'summer', 'autumn'), 'spring': ('spring', 'summer'),
                   'summer': ('summer',), 'autumn': ('autumn', 'summer'), 'winter': ('winter', 'summer')}


def seasonal_paths(paths, season, packs):
    """Swap the season token of each texture-array path to `season` when that file exists."""
    if season in (None, 'xml'):
        return list(paths)
    out = []
    for p in paths:
        m = re.search(r'_(spring|summer|harvest|autumn|winter)_base_colour\.dds$', p)
        pick = p
        if m:
            for s in SEASON_FALLBACK[season]:
                cand = p[:m.start(1)] + s + p[m.end(1):]
                if cand in packs.index:
                    pick = cand
                    break
        out.append(pick)
    return out


def load_layers(mip=1, packs=None, variant=VARIANT, season='xml'):
    """(names, paths, uint8 [L, S, S, 3] sRGB array), cached per mip and season.

    season 'xml' uses texture_arrays.xml as written (summer/autumn mix); a season
    name swaps each group to that season's texture where the packs have one."""
    packs = packs or VanillaPacks()
    names, paths = texture_array(packs, variant)
    paths = seasonal_paths(paths, season, packs)
    cache = CACHE / f'ground_layers_{season}_mip{mip}.npy'
    if cache.exists():
        return names, paths, np.load(cache)
    arr = np.stack([decode_dds(packs.read(p), mip) for p in paths])
    np.save(cache, arr)
    return names, paths, arr
