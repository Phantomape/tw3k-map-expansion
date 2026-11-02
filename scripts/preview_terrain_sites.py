"""Top-down previews of known problem sites and failing native-border segments.

Built from the same layers the game renders (LF heights, global blend, fine-tile
climate), so stripes, square patches, straight cuts and walls show up before a
live test. Each site gets three panels, north up:
  1. approximate ground: tile climate colour, blend material tint, hill shading
  2. hill shading with 0.5-height contours and the native border in red
  3. blend channels amplified (red = channel 0, green = channel 1)

  python preview_terrain_sites.py [--pack X.pack] [--edge-audit native-edge-audit.json]
Writes analysis/terrain-preview/<site>.png and contact-sheet.png in the workspace.
"""
import argparse, json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import campaign_world as cw
from terrain_layers import open_source, load_layers

# Authored on the 1116x798 canvas (native map at hex 192, 32); the edge
# sites sit on the native border, so they follow the native map.
SITES = {k: cw.from_authored_hex(*v) for k, v in {
    'guanghan_sand': (370, 496), 'daner': (210, 377), 'shu_dependency': (317, 424), 'yuexi': (291, 352),
    'xiaowan': (165, 536), 'west500': (192, 500), 'north395': (395, 734), 'north500': (500, 734),
    'north649': (649, 726), 'north800': (800, 734), 'north1000': (1000, 734), 'south400': (400, 32),
}.items()}
CLIMATE_RGB = np.array([(70, 120, 60), (200, 205, 212), (125, 150, 80), (190, 165, 110)] + [(40, 45, 60)] * 252, np.float32)
HALF = 70     # tile cells either side of the site centre
PANEL = 280   # output pixels per panel side


def tile_of(q, r):
    return q * 2, int(round((r + (q % 2) * .5) * 2))


def render(L, q, r):
    cx, cy = tile_of(q, r)
    x0, x1 = max(0, cx - HALF), min(cw.TILE_W, cx + HALF)
    y0, y1 = max(0, cy - HALF), min(cw.TILE_H, cy + HALF)
    h = L['heights'][y0 * 4:y1 * 4, x0 * 4:x1 * 4].astype(np.float32)
    bl = L['blend'][y0 * 4:y1 * 4, x0 * 4:x1 * 4].astype(np.float32)
    cl = np.repeat(np.repeat(L['climate'][y0:y1, x0:x1], 4, 0), 4, 1)[:h.shape[0], :h.shape[1]]
    gy, gx = np.gradient(h)
    shade = np.clip(.75 + (gy - gx) * 6, .35, 1.25)[..., None]
    ground = CLIMATE_RGB[cl]
    sand = np.clip(bl[..., 0] / 30, 0, 1)[..., None]
    rock = np.clip(bl[..., 1] / 8, 0, 1)[..., None]
    ground = ground * (1 - sand) + np.array([205, 185, 140]) * sand
    ground = ground * (1 - rock) + np.array([125, 120, 115]) * rock
    p1 = np.clip(ground * shade, 0, 255)
    p2 = np.repeat(np.clip(170 * shade, 0, 255), 3, axis=2)
    contour = np.abs(np.diff(np.floor(h / .5), axis=0, prepend=np.floor(h[:1] / .5))) + np.abs(np.diff(np.floor(h / .5), axis=1, prepend=np.floor(h[:, :1] / .5)))
    p2[contour > 0] = (60, 90, 160)
    p3 = np.dstack([np.clip(bl[..., 0] * 6, 0, 255), np.clip(bl[..., 1] * 24, 0, 255), np.zeros(h.shape)])
    # Native border in canvas pixels, drawn on panels 2 and 3.
    bx0, bx1 = cw.NATIVE_TILE_X0 * 4 - x0 * 4, cw.NATIVE_TILE_X1 * 4 - x0 * 4
    by0, by1 = cw.NATIVE_TILE_Y0 * 4 - y0 * 4, cw.NATIVE_TILE_Y1 * 4 - y0 * 4
    for p in (p2, p3):
        for v in (bx0, bx1):
            if 0 <= v < p.shape[1]:
                p[:, v] = (220, 40, 40)
        for v in (by0, by1):
            if 0 <= v < p.shape[0]:
                p[v, :] = (220, 40, 40)
    panels = [Image.fromarray(p[::-1].astype(np.uint8)).resize((PANEL, PANEL), Image.Resampling.BILINEAR) for p in (p1, p2, p3)]
    out = Image.new('RGB', (PANEL * 3 + 8, PANEL), (0, 0, 0))
    for i, p in enumerate(panels):
        out.paste(p, (i * (PANEL + 4), 0))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', type=Path)
    ap.add_argument('--pack', type=Path)
    ap.add_argument('--variant', default='3k_dlc07_main_map')
    ap.add_argument('--edge-audit', type=Path, default=cw.WORKSPACE / 'analysis/native-edge-audit.json')
    ap.add_argument('--out', type=Path, default=cw.WORKSPACE / 'analysis/terrain-preview')
    ap.add_argument('--max-edge-sites', type=int, default=12)
    args = ap.parse_args()
    source, vanilla = open_source(args.stage, args.pack)
    L = load_layers(source, vanilla, args.variant)
    sites = dict(SITES)
    for name, (q, r) in cw.load_authored_coordinates()['check_points'].items():
        sites.setdefault(name, (int(q), int(r)))
    if args.edge_audit.exists():
        audit = json.loads(args.edge_audit.read_text(encoding='utf-8'))
        failing = [(e, s) for e, v in audit['edges'].items() for s in v['failing']]
        for e, s in failing[:args.max_edge_sites]:
            sites[f"edge_{e}_{s['hex'][0]}_{s['hex'][1]}_{'+'.join(s['fails'])}"] = tuple(s['hex'])
    args.out.mkdir(parents=True, exist_ok=True)
    tiles = []
    for name, (q, r) in sites.items():
        img = render(L, q, r)
        img.save(args.out / f'{name}.png')
        label = Image.new('RGB', (img.width, 22), (20, 20, 20))
        ImageDraw.Draw(label).text((6, 4), f'{name}  [{q},{r}]', fill=(230, 230, 230))
        tiles.append((label, img))
    sheet = Image.new('RGB', (tiles[0][1].width, sum(l.height + i.height + 6 for l, i in tiles)), (0, 0, 0))
    y = 0
    for label, img in tiles:
        sheet.paste(label, (0, y)); sheet.paste(img, (0, y + label.height)); y += label.height + img.height + 6
    sheet.save(args.out / 'contact-sheet.png')
    print(f'{len(tiles)} sites ->', args.out / 'contact-sheet.png')


if __name__ == '__main__':
    main()
