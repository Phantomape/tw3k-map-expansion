"""Marker overlay of campaign mountain/rock props for render_campaign_view.py (--props).

The offline renderer does not draw prop models; this marks every instance of a
rigidmodels/campaign/mountains/* or /rocks/* model in a global_props.bin (stage or a given
copy): a filled triangle at its position (colour by family) and a thin circle of
FOOTPRINT_CORE of its model's bounding radius (the ground it covers).
"""
import math
import struct

import numpy as np
from PIL import ImageDraw

import campaign_world as cw

COLOURS = {'cold': (235, 240, 255), 'arid': (190, 120, 60), 'subtropical': (60, 170, 80), 'temperate': (170, 110, 200),
           'rocks': (150, 150, 150)}
_cache = {}


def instances(blob):
    """[(family, model, x, y, z, footprint world radius)] of mountain/rock instances."""
    key = (len(blob), hash(blob[:4096]), hash(blob[-4096:]))
    if key in _cache:
        return _cache[key]
    from repair_campaign_river_roots import embedded
    from translate_campaign_lake_instances import records
    from frontier_mountain_props import model_radius, vanilla_assets
    va = vanilla_assets()
    assets, _, _ = embedded(blob)
    out = []
    for k, b in assets.items():
        if k.endswith('/bmd_objects.bin') or (b'/mountains/' not in b and b'/rocks/' not in b):
            continue
        try:
            rs = records(b)
        except Exception:
            continue
        for r in rs:
            m = r['model'].lower()
            if '/mountains/' in m and 'terrace' not in m:
                fam = m.split('/mountains/')[1].split('/')[0]
            elif '/rocks/' in m:
                fam = 'rocks'
            else:
                continue
            M = r['matrix']
            if M[10] < -100:
                continue
            sc = float(np.mean([math.sqrt(sum(v * v for v in M[i:i + 3])) for i in (0, 3, 6)]))
            out.append((fam, m, M[9], M[10], M[11], .8 * sc * model_radius(va, m)))
    _cache[key] = out
    return out


def draw(im, blob, cx, cz, half, size):
    p = 2 * half / size
    dr = ImageDraw.Draw(im)
    n = 0
    for fam, m, x, y, z, rad in instances(blob):
        if abs(x - cx) > half + rad or abs(z - cz) > half + rad:
            continue
        px, py = (x - cx + half) / p, (cz + half - z) / p
        col = COLOURS.get(fam, (255, 0, 255))
        rr = rad / p
        if rr > 3:
            dr.ellipse([px - rr, py - rr, px + rr, py + rr], outline=col)
        s = 7
        dr.polygon([(px, py - s), (px - s, py + s * .7), (px + s, py + s * .7)], fill=col, outline=(0, 0, 0))
        n += 1
    return n
