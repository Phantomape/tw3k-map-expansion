"""Snow mask crops for the offline art review (analysis/art-review-gate.md; cameras with review 'snow-mask-crop').

The offline renderer does not draw the campaign snow post-process (the game lays it over the ground at run time by
season), so the review looks at the mask the snow bake shipped: per camera, the packed ground (LF hillshade) and the
same window with the snow the game would show in winter (A) and summer (G, the peaks): snow_emulate's model of the
vanilla shader (sRGB mask, .8 threshold, noise, slope cut on the LF), white = shown, yellow = mask the shader drops
(analysis/snow-v22b-findings.md). Read-only: the mask
and LF come from a built pack or from a stage copy; nothing in the workspace is written.

  python snow_review.py --run RUN --pack PACK            crops from a candidate pack
  python snow_review.py --run RUN --stage COPY_OF_RELEASE_INPUT
Writes output/offline-view/art-review/<RUN>/snow/<camera>.png and snow-crops.json (shown / dropped shares per camera).
"""
import argparse, json, struct, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import campaign_world as cw
import snow_bake as sb
import snow_emulate as se

ROOT = Path(__file__).resolve().parents[1]
LF = f'terrain/campaigns/{cw.TERRAIN_VARIANTS[0]}/lf_height_map.dds'
CAMERAS = ROOT / 'analysis/art-review-cameras.json'
SCALE = 4          # output pixels per mask pixel
SPAN = 1.0         # crop half-width in world units per unit of camera height


class Source:
    """read(key) / read_rows(key, r0, r1) from a pack (uncompressed entries) or a stage directory."""
    def __init__(self, pack=None, stage=None):
        self.stage = Path(stage) if stage else None; self.idx = {}
        if pack:
            from analyze_reference import inventory
            self.pack = Path(pack)
            self.idx = {e['path'].replace('\\', '/').lower(): e for e in inventory(self.pack, False)['entries']}

    def _span(self, key, off, n):
        if self.stage:
            with open(self.stage / key, 'rb') as f:
                f.seek(off); return f.read(n)
        e = self.idx[key.lower()]
        assert not e['compressed'], key
        with open(self.pack, 'rb') as f:
            f.seek(e['offset'] + off); return f.read(n)

    def read(self, key):
        if self.stage:
            return (self.stage / key).read_bytes()
        return self._span(key, 0, self.idx[key.lower()]['size'])

    def lf_shape(self):
        return struct.unpack('<2I', self._span(LF, 12, 8))

    def lf_rows(self, r0, r1):
        """LF rows [r0, r1) counted from the top (north) of the file, uint16."""
        h, w = self.lf_shape()
        r0, r1 = max(0, r0), min(h, r1)
        return np.frombuffer(self._span(LF, 128 + r0 * w * 2, (r1 - r0) * w * 2), '<u2').reshape(r1 - r0, w), (h, w), r0


def cameras(path):
    d = json.loads(Path(path).read_text(encoding='utf-8'))
    return [c for k in ('fixed', 'changes') for c in d.get(k, []) if c.get('review') == 'snow-mask-crop']


def crop(src, mask, cam, noise):
    W, H = sb.world_size(); mh, mw = mask.shape[:2]
    x, z, height = cam['cam'][:3]; half = height * SPAN
    # mask window (north up)
    c0, c1 = int((x - half) / W * mw), int((x + half) / W * mw)
    r0, r1 = int((1 - (z + half) / H) * mh), int((1 - (z - half) / H) * mh)
    c0, r0 = max(c0, 0), max(r0, 0); c1, r1 = min(c1, mw), min(r1, mh)
    m = mask[r0:r1, c0:c1].astype(np.float32) / 255
    # LF window over the same world box, resampled to the output size
    lh = src.lf_shape()[0]
    lf, (lh, lw), _ = src.lf_rows(int((1 - (z + half) / H) * lh), int((1 - (z - half) / H) * lh) + 1)
    lc0, lc1 = int((x - half) / W * lw), int((x + half) / W * lw)
    g = lf[:, max(lc0, 0):min(lc1, lw)].astype(np.float32)
    out = ((c1 - c0) * SCALE, (r1 - r0) * SCALE)
    g = np.asarray(Image.fromarray(g, 'F').resize(out, Image.Resampling.BILINEAR))
    gy, gx = np.gradient(g)
    shade = np.clip(.55 + (gx - gy) / (np.percentile(np.abs(gx - gy), 99) + 1e-6) * .35, 0, 1)
    base = np.dstack([shade * 200 + 30] * 3)
    # What the game shows (snow_emulate: the vanilla shader on the sRGB mask, its noise and the LF slope), not the raw
    # bytes: a shade under the threshold is no snow in game (v22b). White: snow; yellow: mask > 0 the shader drops.
    hgt = cw.LEVEL + (g / 65535 - cw.NORM) * cw.GAIN
    hy, hx = np.gradient(hgt, H / mh / SCALE, W / mw / SCALE)
    ny = 1 / np.sqrt(1 + hx * hx + hy * hy)
    jj, ii = np.mgrid[:out[1], :out[0]]
    u, v = (c0 + (ii + .5) / SCALE) / mw, (r0 + (jj + .5) / SCALE) / mh
    panels, st = [base], {'window_px': [int(c1 - c0), int(r1 - r0)]}
    for season in ('winter', 'summer'):
        s = se.snow(mask, noise, u, v, ny, se.SEASON_I[season])
        shown, dropped = s['snow'] >= .5, (s['S'] > 1e-3) & (s['amount'] < .5)
        p = base.copy()
        p[dropped] = p[dropped] * .5 + np.array([255, 220, 0]) * .5
        p[shown] = p[shown] * (1 - s['snow'][shown, None]) + np.array([245, 248, 255]) * s['snow'][shown, None]
        panels.append(p)
        st[season] = {'shown': round(float(shown.mean()), 4), 'dropped': round(float(dropped.mean()), 4),
                      'slope_cut': round(float(((s['amount'] >= .5) & (s['aff'] < .5)).mean()), 4)}
    img = Image.new('RGB', (out[0] * 3 + 8, out[1] + 18), (30, 30, 30)); d = ImageDraw.Draw(img)
    for i, (p, lab) in enumerate(zip(panels, ('ground (LF)', 'winter as shown (A)', 'summer as shown (G, peaks)'))):
        img.paste(Image.fromarray(np.clip(p, 0, 255).astype(np.uint8)), (i * (out[0] + 4), 18))
        d.text((i * (out[0] + 4) + 4, 3), f"{cam['name']}  {lab}", fill=(255, 255, 255))
    st.update({'a_mean': round(float(m[..., 3].mean()), 3), 'g_px': int((m[..., 1] >= se.BYTE_NO_NOISE).sum())})
    return img, st


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', required=True); g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--pack', type=Path); g.add_argument('--stage', type=Path, help='a copy of work/release-input')
    ap.add_argument('--cameras', type=Path, default=CAMERAS)
    ap.add_argument('--out', type=Path, help='output folder (default output/offline-view/art-review/<RUN>/snow)')
    o = ap.parse_args()
    src = Source(o.pack, o.stage)
    mask, _ = sb.read_mask(src.read(sb.MASK_KEY))
    from campaign_ground_textures import VanillaPacks
    noise = se.noise_texture(VanillaPacks().read(se.NOISE_KEY))
    out = o.out or ROOT / 'output/offline-view/art-review' / o.run / 'snow'; out.mkdir(parents=True, exist_ok=True)
    rep = {}
    for cam in cameras(o.cameras):
        img, st = crop(src, mask, cam, noise); img.save(out / f"{cam['name']}.png"); rep[cam['name']] = st
    (out / 'snow-crops.json').write_text(json.dumps({'source': str(o.pack or o.stage), 'cameras': rep}, indent=1), encoding='utf-8')
    print(json.dumps(rep))


if __name__ == '__main__':
    main()
