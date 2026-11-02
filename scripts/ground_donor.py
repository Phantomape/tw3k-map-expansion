"""Donor patch choice of the polish blend (the one writer of donor_choice's per-biome rule; numpy only).

polish_campaign_terrain lays canvas ground from vanilla donor patches (biome_profiles CONSTS polish
DONOR_PATCHES: per tile climate, flattest to most rugged). A tile takes patch k of its climate where the
donor choice in [0, 1) passes the cumulative DONOR_SHARES (box-blurred over DONOR_BAND cells: soft patch edges).
The choice is the canvas rank of large-scale noise blended with the rank of local relief.

compress_choice() [§9.3 taiga understory flecks, TW3K_TAIGA_DONOR]: on the biomes of CONSTS polish
DONOR_COMPRESS ({biome: cap}) the choice is scaled into [0, cap), so they take only their climate's flattest
patches: on the West Siberian taiga the noise rank had laid the rugged cold patches (38,482) and (110,476)
(cold_3/4 .44 / .37) as several-hex light-green islands on the dark conifer floor. The biome share is
box-blurred twice over `band` cells (as the patch shares), so the change has no hard edge.

treeline_weight() / treeline_pick() [§9.3 treeline donor blend, TW3K_TREELINE_BLEND]: on the tundra side of a
tundra/cold-forest class line the donor climate is dithered per pixel between the forests' cold climate and
the tundra's arid one, the cold share smoothstep(1 - d / band): forest-floor patches thin out into the tundra
instead of switching on the class line (art review dc7a9361: a crisp cut on the tile-grid class map).
"""
import numpy as np


def compress_choice(choice, eco_bio, bnames, band, compress, box_blur):
    """choice (TH, TW float32 in [0, 1)) scaled toward [0, cap) by the blurred share of each DONOR_COMPRESS biome
    in eco_bio (TH, TW class codes of bnames): choice * (1 - t (1 - cap)), t the biome share blurred twice over
    band cells. Biomes absent from bnames are skipped. Returns float32."""
    out = np.asarray(choice, np.float32)
    for name, cap in compress.items():
        if name not in bnames:
            continue
        t = box_blur(box_blur((eco_bio == bnames.index(name)).astype(np.float32), band), band)
        out = (out * (1 - t * (1 - np.float32(cap)))).astype(np.float32)
    return out


def treeline_weight(eco_bio, bnames, rule):
    """(p, zone) on the tile grid for the CONSTS polish TREELINE_DONOR rule: zone = rule['from'] tiles within
    rule['band'] tiles (euclidean) of a rule['forest'] tile, p = smoothstep(1 - d / band) there (0 elsewhere): the
    share of rule['climates'][1] in the per-pixel donor climate dither (treeline_pick). None when either class is absent."""
    from scipy import ndimage
    src = np.isin(eco_bio, [bnames.index(n) for n in rule['from'] if n in bnames])
    fo = np.isin(eco_bio, [bnames.index(n) for n in rule['forest'] if n in bnames])
    if not src.any() or not fo.any():
        return None
    d = ndimage.distance_transform_edt(~fo).astype(np.float32)
    zone = src & (d < float(rule['band']))
    t = np.clip(1 - d / np.float32(rule['band']), 0, 1)
    return np.where(zone, t * t * (3 - 2 * t), 0).astype(np.float32), zone


def treeline_pick(picked, p, u, zone, climates):
    """Donor climate per pixel in the treeline zone: climates[1] where u < p else climates[0], on zone pixels whose
    argmax pick is one of the two (others keep theirs). picked: int climate codes; p, u float; zone bool (same shape)."""
    a, b = climates
    on = zone & ((picked == a) | (picked == b))
    return np.where(on, np.where(u < p, b, a), picked).astype(picked.dtype)

