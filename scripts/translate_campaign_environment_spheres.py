"""Translate regional lighting spheres into the expanded world frame.

environment_collection.xml places ENVIRONMENT_SPHERES (cold / arid / temperate /
subtropical lighting) by world-space x/z centres. The native file is read from
the game data for every variant, the centres are shifted by NATIVE_SHIFT_WORLD
and the result is written to the release stage. Radii and all other text stay
byte-identical; the lighting references point at the snow bake's copies of the
same groups (campaign_world.SNOW_LIGHTING_DIR: snow post-process in every season,
analysis/snow-bake-design.md). New canvas land outside the native map is
not given new spheres and falls back to the global lighting group.
"""
import tw3k_paths
import argparse, json, re
from pathlib import Path

import campaign_world as cw
from analyze_reference import ROOT, inventory
import campaign_world as _frame
from campaign_world import TERRAIN_VARIANTS

GAME = tw3k_paths.GAME_DATA
A = _frame.WORKSPACE
S = A / 'work/release-input'
VARIANTS = TERRAIN_VARIANTS
SPHERE = re.compile(r"(<SPHERE\b[^>]*?\bx=')([-0-9.]+)('[^>]*?\bz=')([-0-9.]+)(')")


def native(key):
    for name in ('terrain.pack', 'fast.pack', 'data.pack'):
        for e in inventory(GAME / name, False)['entries']:
            if e['path'].lower() == key:
                with (GAME / name).open('rb') as f:
                    f.seek(e['offset'])
                    return f.read(e['size'])
    raise FileNotFoundError(key)


def translate(text):
    dx, dz = cw.NATIVE_SHIFT_WORLD
    moved = []

    def shift(m):
        x, z = float(m.group(2)), float(m.group(4))
        nx, nz = x + dx, z + dz
        moved.append({'from': [x, z], 'to': [round(nx, 6), round(nz, 6)]})
        return f'{m.group(1)}{nx:.6f}{m.group(3)}{nz:.6f}{m.group(5)}'

    out = SPHERE.sub(shift, text)
    if not moved:
        raise RuntimeError('No ENVIRONMENT_SPHERES found; native format changed')
    if text.count('<SPHERE') != len(moved):
        raise RuntimeError('Some spheres did not match the x/z pattern')
    if out.count('<CYLINDER') or re.search(r'<ENVIRONMENT_CYLINDERS>\s*<', out):
        raise RuntimeError('Environment cylinders present; add their translation before continuing')
    for m in moved:
        if not (0 <= m['to'][0] <= cw.WORLD_WIDTH and 0 <= m['to'][1] <= cw.WORLD_HEIGHT):
            raise RuntimeError(f'Sphere centre leaves the world: {m}')
    radii = [float(r) for r in re.findall(r"inner_radius='([-0-9.]+)'", text)]
    if len(radii) != len(moved):
        raise RuntimeError('Sphere radii do not pair with sphere centres')
    for m, r in zip(moved, radii):
        m['inner'] = r
    added = canvas_spheres(moved)
    if added:
        lines = ''.join(f"\t\t<SPHERE serialise_version='1' lighting='{s['lighting']}' x='{s['x']:.6f}' y='{s['y']:.6f}' "
                        f"z='{s['z']:.6f}' inner_radius='{s['inner']:.6f}' outer_radius='{s['outer']:.6f}'/>\n" for s in added)
        # Native files may end lines with CRLF; insert before the closing tag either way.
        close = out.index('</ENVIRONMENT_SPHERES>')
        indent = out.rfind('\n', 0, close) + 1
        out = out[:indent] + lines.replace('\n', '\r\n' if '\r\n' in out else '\n') + out[indent:]
        if out.count('<SPHERE') != len(moved) + len(added):
            raise RuntimeError('Could not append canvas spheres')
    if cw.snow_bake_on():   # off (default): the vanilla lighting groups, as before the snow bake
        old, new = cw.VANILLA_LIGHTING_DIR + '/', cw.SNOW_LIGHTING_DIR + '/'
        if not out.count(old):
            raise RuntimeError(f'No {old} lighting references; native format changed')
        out = out.replace(old, new)
    return out, moved + [{'added': s} for s in added]


# Lighting group per climate code (0 sub_tropical, 1 cold, 2 temperate, 3 arid).
LIGHTING = {0: 'subtropical/subtropical_su', 1: 'cold/cold_su', 2: 'temperate/temperate_su', 3: 'arid/arid_su'}
CANVAS_SPACING, CANVAS_INNER, CANVAS_OUTER = 110.0, 64.0, 90.0
CANVAS_MIN_OUTER = 30.0


def canvas_spheres(native):
    """Regional lighting for canvas land outside the native map.

    Vanilla has six large spheres over the native map only; new land fell back to
    the global summer lighting regardless of climate. Centres sit on a regular
    world grid over canvas land not already inside a native sphere's inner radius;
    the climate comes from the same authored bands that pick the ground tiles.
    """
    import numpy as np
    from native_campaign_climate import band_climate
    authored = cw.load_authored_coordinates()
    cells = np.frombuffer((A / 'work/final-map/map.hex').read_bytes()[-4 - cw.W * cw.H * 16:-4], np.uint8).reshape(cw.H, cw.W, 16)
    land = (cells[:, :, 0] & 3) != 1
    r, q = np.mgrid[:cw.H, :cw.W]
    outside = (q < cw.PAD_LEFT) | (q >= cw.PAD_LEFT + cw.NATIVE_W) | (r < cw.PAD_BOTTOM) | (r >= cw.PAD_BOTTOM + cw.NATIVE_H)
    climate, _, _ = band_climate(q, r, authored['biome_bands'], authored['noise_origin'])
    wx, wz = q * cw.DX, (r + (q % 2) * .5) * cw.DZ
    centres = [(m['to'][0], m['to'][1], m['inner']) for m in native]   # real radii (cold spheres are 64 and 80)
    native_land = land & ~outside
    nx_, nz_ = wx[native_land], wz[native_land]
    added = []
    for z in np.arange(CANVAS_SPACING / 2, cw.WORLD_HEIGHT, CANVAS_SPACING):
        for x in np.arange(CANVAS_SPACING / 2, cw.WORLD_WIDTH, CANVAS_SPACING):
            if any(np.hypot(x - cx, z - cz) < inner for cx, cz, inner in centres):
                continue
            near = (np.hypot(wx - x, wz - z) < CANVAS_INNER) & outside
            if near.sum() < 50 or land[near].mean() < .5:
                continue
            code = int(np.bincount(climate[near & land].ravel(), minlength=4).argmax())
            # A canvas sphere must not relight native land: its whole influence
            # (outer radius) stops short of the nearest native land hex.
            reach = float(np.sqrt(np.min((nx_ - x) ** 2 + (nz_ - z) ** 2))) - 1.0
            outer = min(CANVAS_OUTER, reach)
            if outer < CANVAS_MIN_OUTER:
                continue
            added.append({'lighting': f'terrain/campaigns/3k_main_map/lighting/{LIGHTING[code]}.environment_group',
                          'x': float(x), 'y': 5.0, 'z': float(z), 'inner': round(outer * CANVAS_INNER / CANVAS_OUTER, 3), 'outer': round(outer, 3)})
    return added


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--check', action='store_true', help='verify the stage files, write nothing')
    args = ap.parse_args()
    report = {'shift_world': list(cw.NATIVE_SHIFT_WORLD), 'variants': {}}
    failed = False
    for variant in VARIANTS:
        key = f'terrain/campaigns/{variant}/environment_collection.xml'
        text = native(key).decode('utf-8')
        out, moved = translate(text)
        target = S / key
        if args.check:
            ok = target.exists() and target.read_bytes() == out.encode('utf-8')   # bytes: read_text would fold the native CRLF
            failed |= not ok
            report['variants'][variant] = {'spheres': len(moved), 'stage_matches': ok}
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(out.encode('utf-8'))
        report['variants'][variant] = {'spheres': len(moved), 'moved': moved}
    report['new_canvas_land'] = 'canvas spheres added on new land only; each outer radius stops short of native land'
    if not args.check:
        (A / 'analysis/environment-spheres.json').write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps({v: {k: x for k, x in r.items() if k != 'moved'} for v, r in report['variants'].items()}))
    raise SystemExit(1 if failed else 0)


if __name__ == '__main__':
    main()
