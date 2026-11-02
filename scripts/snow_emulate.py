"""The campaign snow post-process as the game computes it (analysis/snow-v22b-findings.md §3), for offline checks.

Vanilla shaders/campaign_snow.xml (data.pack node graph), with the material's parameters (tw3k_exp_campaign_snow is
the vanilla material with only the mask path changed):
  S      = the season's mask channel, sRGB-decoded: R i < .1, G .1 < i < .5, B .5 < i < .7, A i > .7 (i = snowing_intensity)
  alpha  = saturate(S + S * noise.r(uv * T) * noise.b(uv * T * .1))        T = uv_tiling (150)
  amount = smoothstep(.80, .85, alpha)                                      snowing_amount_target / blend_range
  aff    = smoothstep(.585, .885, saturate(n.y))                            directional_affinity target / range
  snow   = amount * aff
uv = (x / bounds.x, 1 - z / bounds.y) over g_world_bounds; snow is a hard switch: a mask byte below 170 never shows,
below 232 only where the noise helps (BYTE_BEST_NOISE, BYTE_NO_NOISE).

  python snow_emulate.py --out DIR [--pack PACK] [--shots DIR] [--mask DDS]
      the live-check cameras (SITES x VIEWS x seasons): LF ray march of the live camera (render_live_view), the
      prediction over the screenshot of the same camera: white = snow, magenta = mask says snow but the slope removes
      it, yellow = mask > 0 but under the threshold. Read-only: the pack and screenshots are only read.
"""
import tw3k_paths
import argparse, io, json, os, struct, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

UV_TILING = 150.
AMOUNT = (.80, .85)
AFFINITY = (.585, .885)
NOISE_KEY = 'terrain/textures/campaign/default/snow/snow_noise.dds'
SEASON_I = {'winter': .8, 'summer': .2, 'harvest': .4, 'autumn': .6, 'spring': .6}


def srgb_lin(b):
    """sRGB bytes (any shape) -> linear 0..1."""
    c = np.asarray(b, np.float64) / 255.
    return np.where(c <= .04045, c / 12.92, ((c + .055) / 1.055) ** 2.4)


BYTE_NO_NOISE = int(np.argmax(srgb_lin(np.arange(256)) >= AMOUNT[0]))         # 232
BYTE_BEST_NOISE = int(np.argmax(srgb_lin(np.arange(256)) >= AMOUNT[0] / 2))    # 170


def smooth(e0, e1, x):
    t = np.clip((np.asarray(x, np.float64) - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def channel(intensity):
    """Mask channel index (RGBA) the shader reads at this snowing_intensity."""
    return 3 if intensity > .7 else (2 if intensity > .5 else (1 if intensity > .1 else 0))


def bilinear(img, u, v, wrap=False):
    """img (H, W[, C]) at uv, texel centres at +.5; clamp, or wrap (tiled textures)."""
    img = np.asarray(img); H, W = img.shape[:2]
    if wrap:
        u, v = np.mod(u, 1.), np.mod(v, 1.)
    fx, fy = np.asarray(u) * W - .5, np.asarray(v) * H - .5
    x0, y0 = np.floor(fx).astype(np.intp), np.floor(fy).astype(np.intp)
    ax, ay = fx - x0, fy - y0
    if img.ndim == 3:
        ax, ay = ax[..., None], ay[..., None]
    def at(y, x):
        if wrap:
            return img[np.mod(y, H), np.mod(x, W)]
        return img[np.clip(y, 0, H - 1), np.clip(x, 0, W - 1)]
    return (at(y0, x0) * (1 - ax) + at(y0, x0 + 1) * ax) * (1 - ay) + (at(y0 + 1, x0) * (1 - ax) + at(y0 + 1, x0 + 1) * ax) * ay


def noise_texture(data):
    """snow_noise.dds (BC1 sRGB in vanilla) -> linear RGBA float (h, w, 4), top mip."""
    from PIL import Image
    fmt = struct.unpack_from('<I', data, 128)[0] if data[84:88] == b'DX10' else None
    srgb = fmt in (72, 78)
    if srgb:          # PIL has no _SRGB ids: decode as the UNORM twin, then linearise
        data = bytearray(data); struct.pack_into('<I', data, 128, fmt - 1); data = bytes(data)
    a = np.asarray(Image.open(io.BytesIO(data)).convert('RGBA'), np.float64)
    return srgb_lin(a) if srgb else a / 255.


def snow(mask_rgba_u8, noise, u, v, ny, intensity):
    """Per-sample snow terms: dict(S, amount, aff, snow). mask_rgba_u8: (H, W, 4) bytes as in the DDS (RGBA order);
    u, v: mask uv; ny: up component of the surface normal."""
    S = bilinear(srgb_lin(mask_rgba_u8[..., channel(intensity)]), u, v)    # sRGB texels decode before filtering
    nr = bilinear(noise[..., 0], u * UV_TILING, v * UV_TILING, wrap=True)
    nb = bilinear(noise[..., 2], u * UV_TILING * .1, v * UV_TILING * .1, wrap=True)
    amount = smooth(*AMOUNT, np.clip(S + S * nr * nb, 0, 1))
    aff = smooth(*AFFINITY, np.clip(ny, 0, 1))
    return {'S': S, 'amount': amount, 'aff': aff, 'snow': amount * aff}


def lf_normal_y(heights, wx, wz, eps=.04):
    """Up component of the height-field normal (heights(wx, wz) -> world y)."""
    hx = heights(wx + eps, wz) - heights(wx - eps, wz)
    hz = heights(wx, wz + eps) - heights(wx, wz - eps)
    return 2 * eps / np.sqrt(hx * hx + hz * hz + 4 * eps * eps)


# live check v22b (analysis/live-check-v22b-result.md): requested cameras (x, z) and views (dist, heading, height)
SITES = {'bogda': (456.5, 848.4), 'muztagh': (307.72, 748.25), 'qiangtang': (445.16, 655.03), 'gongga': (670.99, 565.28),
         'hulunbuir': (977.21, 929.68), 'liaodong': (1052.93, 799.48)}
VIEWS = {'medium': (120., 0., 60.), 'close': (40., 0., 35.)}
SIZE = (640, 360)


def live_cameras(pack, shots, out, mask=None):
    os.environ['TW3K_RENDER_CALIB_PACK'] = str(pack)
    import render_live_view as L
    import snow_bake as sb
    from PIL import Image
    g = L.PackGround(Path(pack))
    P = dict(L.LIVE_DEFAULTS)
    m8, _ = sb.read_mask(Path(mask).read_bytes() if mask else g.read(sb.MASK_KEY))
    noise = noise_texture(g.read(NOISE_KEY))
    ws = sb.world_size()
    out.mkdir(parents=True, exist_ok=True)
    stats = {}
    for site, (x, z) in SITES.items():
        for view, (dist, hd, hgt) in VIEWS.items():
            cam = (x, z, dist, hd, hgt)
            eye, d, _, _ = L.camera_rays(g, cam, SIZE, P)
            reach = dist * 6
            GX, GZ = np.meshgrid(np.linspace(x - reach, x + reach, 256), np.linspace(z - reach, z + reach, 256))
            hh = g.heights(GX, GZ)
            found, t = L.march(g.heights, eye, d, float(hh.min()) - .2, float(hh.max()) + .2)
            wx = np.where(found, eye[0] + d[..., 0] * t, x); wz = np.where(found, eye[2] + d[..., 2] * t, z)
            ny = lf_normal_y(g.heights, wx, wz)
            for season in ('winter', 'summer', 'autumn', 'spring'):
                s = snow(m8, noise, wx / ws[0], 1 - wz / ws[1], ny, SEASON_I[season])
                shot = Path(shots) / f'{season}-{site}-{view}.png'
                base = (np.asarray(Image.open(shot).convert('RGB').resize(SIZE, Image.Resampling.BILINEAR), np.float64)
                        if shot.is_file() else np.zeros(SIZE[::-1] + (3,)))
                faint = found & (s['S'] > 1e-3) & (s['amount'] < .5)
                killed = found & (s['amount'] >= .5) & (s['aff'] < .5)
                white = found & (s['snow'] >= .5)
                ov = base.copy()
                ov[faint] = ov[faint] * .5 + np.array([255, 220, 0]) * .5
                ov[killed] = ov[killed] * .4 + np.array([255, 0, 255]) * .6
                ov[white] = ov[white] * .3 + 255 * .7
                Image.fromarray(np.concatenate([base, ov], 1).astype(np.uint8)).save(out / f'{season}-{site}-{view}-emul.png')
                n = max(int(found.sum()), 1)
                stats[f'{season}-{site}-{view}'] = {
                    'channel': 'RGBA'[channel(SEASON_I[season])], 'mask_nonzero': round(float(((s['S'] > 1e-3) & found).sum() / n), 4),
                    'snow': round(float(white.sum() / n), 4), 'killed_by_slope': round(float(killed.sum() / n), 4),
                    'too_faint': round(float(faint.sum() / n), 4)}
    (out / 'emul-stats.json').write_text(json.dumps(stats, indent=1), encoding='utf-8')
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--pack', type=Path, default=tw3k_paths.MAIN_ROOT / 'output/tw3k_regions_and_battles_terrain_repair_candidate.pack')
    ap.add_argument('--shots', type=Path, default=tw3k_paths.MAIN_ROOT / 'output/live-v22b')
    ap.add_argument('--mask', type=Path, help='a snow mask DDS to use instead of the pack\'s (e.g. a snow_bake --preview)')
    o = ap.parse_args()
    print(json.dumps(live_cameras(o.pack, o.shots, o.out, o.mask), indent=1))


if __name__ == '__main__':
    main()
