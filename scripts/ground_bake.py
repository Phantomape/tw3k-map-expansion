"""Single selection of final blend materials from read-only source proposals.

The terrain bake prepares all fields first. Providers never receive the output
array: they contribute a material, a weight and a climate policy. This module
filters candidates, computes one common tone target, and selects (source, R, G)
once. A caller commits each disjoint output block once. No post-paint repair.
"""
from dataclasses import dataclass
import threading
import numpy as np


SOURCES = {0: 'preserved', 1: 'inherited', 2: 'donor', 3: 'forest',
           4: 'open_ground', 5: 'landform', 6: 'farm', 7: 'mud',
           8: 'town', 9: 'authored_farm', 10: 'wetland', 11: 'tone_bridge',
           12: 'authored_forest'}

# These identities are final (source, G) draws, not provider purpose numbers.
# In particular, provider cold_hit also uses5; climate ownership uses101.
FINAL_NATURAL_PURPOSES = frozenset(source + 16*band
                                 for band in range(4) for source in range(1, 6))


def single_ground_workers():
    """Opt-in row concurrency, bounded even when other pool overrides are set.

    The default stays one. Two workers require at least 3 GiB per worker plus
    3 GiB free-commit reserve *after* the shared bake fields have been allocated.
    Unknown commit availability keeps the serial path. This changes execution
    only: all candidates and world-coordinate draws are unchanged.
    """
    import os
    import memory_budget
    raw = os.environ.get('TW3K_SINGLE_GROUND_THREADS', '1')
    try:
        requested = int(raw)
    except ValueError as error:
        raise ValueError('TW3K_SINGLE_GROUND_THREADS must be 1 or 2') from error
    if requested not in (1, 2):
        raise ValueError('TW3K_SINGLE_GROUND_THREADS must be 1 or 2')
    if requested == 1:
        return 1
    free = memory_budget.free_commit_gb()
    if free is None or not np.isfinite(free):
        return 1
    # pool_size's env argument deliberately bypasses budgeting. Use its same
    # calculation on this single known commit snapshot, without an env override
    # or a second measurement that could be unavailable.
    return min(requested, max(1, int((free - 3.) / 3.)))


@dataclass
class Candidate:
    source: int
    slot: object
    weight: object
    band: object


def smooth(a, radius):
    """Separable box mean; callers include a halo on block boundaries."""
    a = np.asarray(a, np.float32)
    if radius <= 0:
        return a.copy()
    for axis in (0, 1):
        pad = [(0, 0)] * a.ndim
        pad[axis] = (radius, radius)
        p = np.pad(a, pad, mode='edge')
        z = np.zeros_like(np.take(p, [0], axis=axis))
        cs = np.concatenate((z, np.cumsum(p, axis=axis, dtype=np.float64)), axis=axis)
        lo = [slice(None)] * a.ndim
        hi = lo.copy()
        lo[axis] = slice(0, a.shape[axis])
        hi[axis] = slice(2 * radius + 1, 2 * radius + 1 + a.shape[axis])
        a = ((cs[tuple(hi)] - cs[tuple(lo)]) / (2 * radius + 1)).astype(np.float32)
    return a


def sample_rows(field, y0, y1, width, height, nearest=False):
    """South-up tile field -> north-up blend row block (four pixels/tile)."""
    import ground_materials as gm
    tx = (np.arange(width, dtype=np.float32) + .5) / 4 - .5
    ty = (height - 1 - np.arange(y0, y1, dtype=np.float32) + .5) / 4 - .5
    if nearest:
        ix = np.clip(np.arange(width) // 4, 0, field.shape[-1] - 1)
        iy = np.clip((height - 1 - np.arange(y0, y1)) // 4, 0, field.shape[-2] - 1)
        return field[..., iy[:, None], ix[None, :]]
    a, i0, i1, fy = gm._bilinear_rows(field, tx, ty)
    return a[..., i0, :] * (1 - fy[:, None]) + a[..., i1, :] * fy[:, None]


def frayed_uniform_cdf(x, amplitude):
    """CDF of U0 + amplitude*(Ufray-.5), independent unit uniforms.

    This is the marginal model of the old farm/fray threshold, not a replay
    of its spatial noise. Piecewise tails avoid subtracting squared float32
    terms. The rule's amplitude is .10; zero keeps the unfrayed limit.
    """
    amplitude = float(amplitude)
    if not np.isfinite(amplitude) or not 0 <= amplitude <= 1:
        raise ValueError('Farm fray amplitude must be finite and between0 and1')
    x = np.asarray(x, np.float64)
    if amplitude == 0:
        return np.clip(x, 0, 1)
    half = amplitude / 2
    lower = np.clip(x + half, 0, amplitude)
    upper = np.clip(1 + half - x, 0, amplitude)
    return np.where(x < half, lower*lower/(2*amplitude),
                    np.where(x > 1-half, 1-upper*upper/(2*amplitude), x))


def farm_inner_fraction(farm_weight, rim, fray_amplitude):
    """Conditional margin>rim within an already reserved farm source.

    The legacy writer required u<F and F-u>rim, followed by its unchanged
    INNER[k] dab share. Keep F/K/M and farm ownership mass intact; only the
    conditional inner/rim recipe uses CDF(F-rim)/CDF(F). Real correlated
    quantile fields and tone selection need actual captured-ROI validation.
    """
    rim = float(rim)
    if not np.isfinite(rim) or rim < 0:
        raise ValueError('Farm rim must be finite and nonnegative')
    f = np.asarray(farm_weight, np.float64)
    denominator = frayed_uniform_cdf(f, fray_amplitude)
    fraction = np.zeros_like(f)
    np.divide(frayed_uniform_cdf(f-rim, fray_amplitude), denominator,
              out=fraction, where=(f > 0) & (denominator > 0))
    return np.clip(fraction, 0, 1).astype(np.float32)


def density_ground_palette(candidates, density, picked, keep, forbidden, tones,
                           open_palette, *, enabled, sources, wood_slots,
                           clearance=None):
    """Pure display-palette response on already reserved candidate masses.

    This changes generated woodland MATERIAL coverage, not shared forest D or
    tree density. Farming eligibility and ownership reservations have already
    been compiled: brighter forest-floor slots do not grant new farming support.
    Keep each source/G's mass, exact w*D then w-dark arithmetic, and original
    palette order for equal Lab distances. No legal open slot means keep wood.
    Off returns the original candidate set without touching inputs or draws.
    """
    if not enabled:
        return candidates
    density = np.asarray(density, np.float32)
    picked = np.asarray(picked)
    shape = density.shape
    if density.ndim != 2 or picked.shape != shape or not np.isfinite(density).all():
        raise ValueError('Ground palette density/picked frame mismatch')
    if np.any((density < 0) | (density > 1)):
        raise ValueError('Ground palette density outside[0,1]')
    keep = np.broadcast_to(np.asarray(keep, bool), shape)
    clearance = np.zeros(shape, bool) if clearance is None else np.broadcast_to(np.asarray(clearance, bool), shape)
    sources, wood_slots = frozenset(sources), tuple(wood_slots)
    tones64 = np.asarray(tones, np.float64)
    # Only 32x32 tone arithmetic; never a three-channel full-row temporary.
    distances = ((tones64[:, None, :] - tones64[None, :, :]) ** 2).sum(-1)
    destinations, legal_open = {}, {}

    def destination(wood):
        if wood not in destinations:
            out = np.full(shape, 255, np.uint8)
            for climate, options in open_palette.items():
                climate_mask = picked == int(climate)
                if not climate_mask.any():
                    continue
                # Python's stable sort preserves palette order on exact ties.
                for material in sorted(options, key=lambda m: distances[wood, m]):
                    if material not in legal_open:
                        legal = ~np.broadcast_to(forbidden(np.uint8(material)), shape)
                        legal_open[material] = legal & (~clearance if material == 31 else True)
                    use = climate_mask & (out == 255) & legal_open[material]
                    out[use] = material
            destinations[wood] = out
        return destinations[wood]

    result, residuals = [], []
    for candidate in candidates:
        if candidate.source not in sources:
            result.append(candidate)
            continue
        slot = np.broadcast_to(np.asarray(candidate.slot, np.uint8), shape)
        weight = np.broadcast_to(np.asarray(candidate.weight, np.float32), shape)
        wanted = np.isin(slot, wood_slots) & (weight > 0) & ~keep & ~forbidden(slot)
        if not wanted.any():
            result.append(candidate)
            continue
        chosen = np.full(shape, 255, np.uint8)
        for wood in wood_slots:
            use = wanted & (slot == wood)
            if use.any():
                chosen[use] = destination(wood)[use]
        valid = wanted & (chosen < 32)
        dark = weight.copy()
        dark[valid] = weight[valid] * density[valid]
        residual = np.zeros(shape, np.float32)
        residual[valid] = weight[valid] - dark[valid]
        result.append(Candidate(candidate.source, candidate.slot, dark, candidate.band))
        for material in np.unique(chosen[residual > 0]):
            open_weight = np.where(chosen == material, residual, 0).astype(np.float32)
            residuals.append(Candidate(candidate.source, int(material), open_weight, candidate.band))
    return result + residuals


def _missing_details(candidates, missing, forbidden, y0, context):
    """Bounded diagnostics on failure only; never retain a second candidate set."""
    shape = missing.shape
    samples = []
    for y, x in np.argwhere(missing)[:4]:
        proposals, zero = [], 0
        for c in candidates:
            weight = float(np.broadcast_to(np.asarray(c.weight), shape)[y, x])
            if weight <= 0:
                zero += 1
                continue
            slot = np.broadcast_to(np.asarray(c.slot, np.uint8), shape)
            band = np.broadcast_to(np.asarray(c.band, np.uint8), shape)
            proposals.append({'source': int(c.source), 'R': int(slot[y,x]),
                              'G': int(band[y,x]), 'weight': weight,
                              'bad': bool(forbidden(slot)[y,x])})
        sample = {'x': int(x), 'y': int(y0+y), 'proposals': proposals[:12],
                  'proposals_omitted': max(0,len(proposals)-12), 'zero_weight_proposals': zero}
        for key, value in (context or {}).items():
            a = np.asarray(value)
            sample[key] = a.item() if a.ndim == 0 else a[y,x].item()
        samples.append(sample)
    return samples


def select(candidates, preserved, keep, forbidden, tones, thresholds, draw,
           *, tone_radius=2, tone_strength=1., y0=0, context=None):
    """Return final RG, exact source owner and diagnostics, without writing input.

    forbidden(slot) returns a bool pixel mask using the generator's exact ban
    field. draw(source, slot, mask) gives one deterministic uniform per selected
    coordinate, independent of traversal/chunking. No winning value is re-drawn.
    """
    shape = preserved.shape[:2]
    keep = np.broadcast_to(np.asarray(keep, bool), shape)
    total = np.zeros(shape, np.float32)
    target = np.zeros(shape + (3,), np.float32)
    records, masses = [], []
    pixels = int(np.prod(shape))
    rejected = 0
    for c in candidates:
        if c.source not in SOURCES or c.source == 0:
            raise ValueError('Unknown/non-generative candidate source')
        slot = np.broadcast_to(np.asarray(c.slot, np.uint8), shape)
        band = np.broadcast_to(np.asarray(c.band, np.uint8), shape)
        if np.any(slot > 31) or np.any(band > 3):
            raise ValueError('Candidate outside R/G slot range')
        w = np.broadcast_to(np.asarray(c.weight, np.float32), shape).copy()
        if not np.isfinite(w).all() or np.any(w < 0):
            raise ValueError('Invalid candidate weight')
        bad = forbidden(slot)
        rejected += int(((w > 0) & bad & ~keep).sum())
        w[bad | keep] = 0
        if not w.any():
            continue
        total += w
        target += w[..., None] * tones[slot]
        idx = np.flatnonzero(w)
        code = c.source * 128 + slot.ravel()[idx].astype(np.int64) * 4 + band.ravel()[idx]
        records.append(code * pixels + idx)
        masses.append(w.ravel()[idx])
    missing = (total <= 0) & ~keep
    if missing.any():
        raise ValueError(f'No legal candidate at {int(missing.sum())} editable pixels; '
                         f'samples={_missing_details(candidates,missing,forbidden,y0,context)}')
    target /= np.maximum(total, 1e-20)[..., None]
    target[keep] = tones[preserved[..., 0][keep]]
    target = smooth(target, tone_radius)
    best = np.full(pixels, -np.inf, np.float64)
    best_code = np.full(pixels, 65535, np.uint16)
    owner = np.zeros(shape, np.uint8)
    result = preserved.copy()
    if records:
        keys = np.concatenate(records)
        vals = np.concatenate(masses)
        order = np.argsort(keys, kind='stable')
        keys = keys[order]
        vals = vals[order].astype(np.float64)
        start = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1]) + 1]
        weights = np.add.reduceat(vals, start)
        keys = keys[start]
        codes, indices = np.divmod(keys, pixels)
        cuts = np.r_[0, np.flatnonzero(codes[1:] != codes[:-1]) + 1, len(codes)]
        del records, masses, vals, order, keys
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            code = int(codes[lo]); source, pair = divmod(code,128); slot, band = divmod(pair,4)
            ix = indices[lo:hi]
            de2 = ((tones[slot] - target.reshape(-1,3)[ix]) ** 2).sum(axis=-1)
            w = weights[lo:hi] * np.exp(-tone_strength*de2/max(float(thresholds[band])**2,1e-6))
            mask = np.zeros(pixels,bool); mask[ix] = True; mask = mask.reshape(shape)
            slot_grid = np.broadcast_to(np.uint8(slot),shape)
            # The climate band is part of candidate identity as well as its
            # purpose key. Identical source/R/G contributions have been summed.
            u = np.clip(draw(source+16*band,slot_grid,mask),1e-9,1-1e-9)
            score = np.log(u)/np.maximum(w,1e-300)
            better = (score > best[ix]) | ((score == best[ix]) & (code < best_code[ix]))
            dest = ix[better]
            best[dest] = score[better]; best_code[dest] = code
            owner.ravel()[dest] = source
            result.reshape(-1,2)[dest,0] = slot
            result.reshape(-1,2)[dest,1] = band
    if np.any((owner == 0) & ~keep):
        raise AssertionError('An editable pixel has no selected owner')
    if not np.array_equal(result[keep], preserved[keep]):
        raise AssertionError('Protected original bytes changed')
    return result, owner, {'illegal_candidates_removed': rejected,
                           'protected_pixels': int(keep.sum()),
                           'selected_pixels': int((~keep).sum())}


class SparseCandidates:
    """Small authored patches, read by the final selector; never a blend image."""
    def __init__(self, source):
        self.source = source
        self.patches = []

    def add(self, box, slots, weights):
        y0, y1, x0, x1 = box
        if np.asarray(weights).shape != (y1-y0, x1-x0):
            raise ValueError('Candidate patch has wrong shape')
        if np.any(weights):
            self.patches.append((box, np.asarray(slots, np.uint8).copy(),
                                 np.asarray(weights, np.float32).copy()))

    def rows(self, y0, y1, width, band):
        # Patches may overlap. Combining identical source/slot candidates before
        # selection keeps provider enumeration from affecting their probability.
        weights = {}
        for (a, b, x0, x1), slots, w in self.patches:
            lo, hi = max(a, y0), min(b, y1)
            if hi <= lo:
                continue
            sub = slots[lo-a:hi-a]
            sw = w[lo-a:hi-a]
            for slot in np.unique(sub[sw > 0]):
                if int(slot) not in weights:
                    weights[int(slot)] = np.zeros((y1-y0, width), np.float32)
                dst = weights[int(slot)][lo-y0:hi-y0, x0:x1]
                np.maximum(dst, sw * (sub == slot), out=dst)
        return [Candidate(self.source, m, w, band) for m, w in sorted(weights.items())]


class OrganicDraw:
    def __init__(self, origin, grain=2.5, *, farm_grain=None, inner_grain=None,
                 natural_grain=None):
        import ground_materials as gm
        self.q = gm.Quantiles(origin, {m: (grain, 3, 450 + m) for m in range(32)})
        # Only the geographical farm source opts into its larger spatial field.
        # Keep q unchanged: other sources and the separate climate purpose101
        # must retain their previous fields and quantile calibration.
        self.farm_q = None
        if farm_grain is not None or inner_grain is not None:
            scale, octaves = (grain, 3) if farm_grain is None else (farm_grain, 4)
            specs = {m: (scale, octaves, 450 + m)
                     for m in sorted({m for pair in gm.FARM_KINDS for m in pair})}
            if inner_grain is not None:
                # Tea has one material for inner and rim. Only the distinct
                # grain/rice inner slots opt into fine texture-scale draws.
                for inner, rim in gm.FARM_KINDS:
                    if inner != rim:
                        specs[inner] = (inner_grain, 2, 450 + inner)
            self.farm_q = gm.Quantiles(origin, specs)
        # Only final_rows may use this field. Provider rows and the old q stay
        # unchanged, including cold_hit and the separate climate purpose101.
        self.natural_q = None
        if natural_grain is not None:
            self.natural_q = gm.Quantiles(
                origin, {m: (natural_grain, 3, 450 + m) for m in range(32)},
                calibration_size=2048, calibration_step=.125)

    def rows(self, y0, height):
        def draw(source, slots, mask):
            import gpu_noise
            # select encodes its source and G as source+16*G. Old captured draw
            # instances created with __new__ only carry q; their legacy path
            # remains valid for exact replay of the original producer.
            farm_q = getattr(self, 'farm_q', None)
            q = farm_q if source % 16 == 6 and farm_q is not None else self.q
            iy, ix = np.nonzero(mask)
            vals = np.empty(len(ix), np.float32)
            material = slots[mask]
            for m in np.unique(material):
                part = material == m
                # Purpose-separated world translation retains spatial coherence.
                if gpu_noise.active():
                    cm = mask & (slots == m)
                    sc, oc, seed = q.specs[int(m)]
                    nz = gpu_noise.fbm_at(np.arange(mask.shape[1])/8. + source*131,
                            (height-1-y0-np.arange(mask.shape[0]))/8. + source*113,
                            cm,q.origin,sc,oc,seeds=(seed,))[0]
                    vals[part] = q.uniform(int(m), nz)
                else:
                    vals[part] = q.u(int(m), ix[part] / 8. + source * 131,
                                         (height - 1 - y0 - iy[part]) / 8. + source * 113)
            return vals
        return draw

    def final_rows(self, y0, height):
        """Final selection only; provider draws must continue to call rows."""
        base = self.rows(y0, height)
        natural_q = getattr(self, 'natural_q', None)
        if natural_q is None:
            return base
        # Reuse the exact world-coordinate CPU/GPU draw implementation without
        # changing its default path or recalibrating either old or farm fields.
        natural = OrganicDraw.__new__(OrganicDraw)
        natural.q = natural_q
        fine = natural.rows(y0, height)
        def draw(purpose, slots, mask):
            return (fine if purpose in FINAL_NATURAL_PURPOSES else base)(purpose, slots, mask)
        return draw


class GroundCandidates:
    """Compile existing geographical recipes into one row-block candidate set."""
    def __init__(self, fields, domains, forest, origin, shape, *, town=None,
                 farms=None, wetland=None, town_clear=None, authored_forest=None):
        import ground_materials as gm
        import biome_profiles as bp
        import ground_tone
        self.fields, self.domains, self.forest = fields, domains, forest
        self.height, self.width = shape
        self.town, self.farms, self.wetland = town, farms, wetland
        # WetlandCandidates may call a WaterSupport -> NativeWater reader with
        # a mutable two-item asset LRU. Serialize the whole provider call, not
        # independent pixel candidates or the source/noise selection.
        self._wetland_lock = threading.Lock()
        self.town_clear, self.authored_forest = town_clear, authored_forest
        self.draw = OrganicDraw(origin, farm_grain=gm.FARM_GRAIN,
                                inner_grain=gm.FARM_DAB, natural_grain=1.)
        self.tones = ground_tone.tone_table()
        if self.tones is None:
            raise ValueError('Single material selection requires measured tones')
        self.thresholds = np.array([ground_tone.max_dE(g) for g in range(4)], np.float32)
        self.wood = tuple(bp.WOODLAND_SLOTS)
        self.dw = bp.CONSTS['ground_materials']['DONOR_WOODLAND']
        self.gm = gm

    def rows(self, y0, y1, inherited, donor, share, picked, preserved, base_band=None):
        gm = self.gm
        h, w = inherited.shape[:2]
        at = lambda a, nearest=False: sample_rows(a, y0, y1, w, self.height, nearest)
        keep = at(self.domains['native_keep'], True)
        takeover = np.clip(at(self.domains['material_takeover']), 0, 1)
        fd = np.clip(at(self.forest), 0, 1)
        clearance = np.zeros((h,w), bool) if self.town_clear is None else at(self.town_clear, True)
        tx = (np.arange(w, dtype=np.float32) + .5) / 4 - .5
        ty = (self.height - 1 - np.arange(y0, y1, dtype=np.float32) + .5) / 4 - .5
        lb, al = gm._ban_rows_fields(self.fields, tx, ty)
        forbidden = lambda slot: gm._banned(lb, al, slot)
        W, Wd, Wo = (np.clip(at(self.fields[k]) / np.maximum(takeover, 1e-6), 0, 1)
                     for k in ('W', 'Wd', 'Wo'))
        cold = np.clip(at(self.fields['cold']), 0, 1)
        draw = self.draw.rows(y0, self.height)
        cold_hit = np.zeros((h, w), bool)
        cm = cold > 0
        if cm.any():
            cold_hit[cm] = draw(5, np.zeros((h, w), np.uint8), cm) < cold[cm]
        # Resolve existing climate policy once. Farm proposals carry this band;
        # wetland proposals alone carry their map-climate policy.
        inherited_g = np.where(cold_hit, 0, inherited[..., 1]).astype(np.uint8)
        donor_g = np.where(cold_hit, 0, donor[..., 1]).astype(np.uint8)
        support_band = (np.where(share >= .5, donor_g, inherited_g) if base_band is None else
                        np.where(cold_hit, 0, base_band)).astype(np.uint8)
        F = np.clip(at(self.fields['F']), 0, 1) if 'F' in self.fields else np.zeros((h, w), np.float32)
        M = np.clip(at(self.fields['M']), 0, 1) if 'M' in self.fields else np.zeros_like(F)
        natural = []
        recipe_mass = np.zeros((h, w), np.float32)
        k = np.zeros(16, np.float32)
        for c, value in self.dw['k'].items():
            k[int(c)] = value
        for source, raw, mass, band in ((1, inherited[..., 0], 1-share, inherited_g),
                                        (2, donor[..., 0], share, donor_g)):
            wood = np.isin(raw, self.wood)
            forest_share = np.clip(fd*k[picked], 0, 1)
            # Imported farm/wetland detail has no independent authority on fully
            # geographical terrain. Its semantic support is supplied by the
            # farm/wetland rules, while reach=0 preserves the original source.
            imported = np.isin(raw, tuple(range(8, 13)) + (31,))
            replacement = (wood | imported).astype(np.float32)
            retained = mass * (1-takeover)
            natural.append(Candidate(source, raw, retained, band))
            geographic_mass = mass * takeover
            raw_open = geographic_mass * (1-forest_share) * (1-replacement)
            take = np.where(gm.REPLACEABLE[raw], W, 0).astype(np.float32)
            take = np.where(gm.DESERT_EXTRA[raw], np.maximum(take, Wd), take)
            take = np.where(gm.OPEN_EXTRA[raw], np.maximum(take, Wo), take)
            take = np.where(forbidden(raw), 1., take)
            recipe_mass += raw_open * take
            natural.append(Candidate(source, raw, raw_open * (1-take), band))
            # Use the existing preferred woodland slot for newly forested ground
            # and the thinning edge. Slot 5 is darker than slot 4 in the measured
            # tone table: inventing it as density falls made Funan's edge harsher.
            # An actual source woodland family member can survive in the dense
            # interior; both shares still consume exactly the one forest target.
            dense_slot = np.where(wood, raw, self.wood[0]).astype(np.uint8)
            natural.append(Candidate(3, dense_slot, geographic_mass * forest_share * fd, band))
            natural.append(Candidate(3, self.wood[0], geographic_mass * forest_share * (1-fd), band))
            for climate, slots in self.dw['open'].items():
                cmask = picked == int(climate)
                for slot in slots:
                    rw = geographic_mass * (1-forest_share) * replacement * cmask / len(slots)
                    rt = np.where(gm.REPLACEABLE[slot], W, 0)
                    if gm.DESERT_EXTRA[slot]:
                        rt = np.maximum(rt, Wd)
                    if gm.OPEN_EXTRA[slot]:
                        rt = np.maximum(rt, Wo)
                    rt = np.where(forbidden(np.full((h,w), slot, np.uint8)), 1., rt)
                    recipe_mass += rw * rt
                    natural.append(Candidate(4, slot, rw * (1-rt), band))
        # Reject a natural proposal before it reserves farm support or source
        # coverage. A reflected source may carry 31 where the static reference
        # did not: clearance is therefore checked on the actual candidate.
        for c in natural:
            slot = np.broadcast_to(np.asarray(c.slot,np.uint8),(h,w))
            bad = forbidden(slot) | ((slot == 31) & clearance)
            recipe_mass += c.weight * bad
            c.weight = c.weight * ~bad
        P = at(self.fields['P'])
        # Filter/renormalize the geological recipe before it competes. Existing
        # own-landform fallback lives here, never after selecting final pixels.
        P = gm._clear_banned(P.reshape(len(gm.MATERIALS), -1), lb.ravel(),
                            None if al is None else al.ravel()).reshape(P.shape)
        Psum = P.sum(axis=0)
        P /= np.maximum(Psum, 1e-20)[None]
        # Compile a legal conditional landform recipe first. Geological wood
        # relinquishes its FD=0 share to the existing climate open recipe. If
        # an open slot is itself banned, return that probability to the local
        # legal NON-wood landform recipe, never recursively back into wood.
        nonwood_total = sum((P[i] for i,slot in enumerate(gm.MATERIALS)
                             if slot not in self.wood),np.zeros((h,w),np.float32))
        rejected_open = np.zeros((h,w),np.float32)
        recipe = []
        # With no landform domain, P=0 is normal (for example retained native
        # dry clearance at T=0). Only when an actual proposal was rejected,
        # use its existing climate's open family as the conditional dry recipe.
        # A positive L must instead have its own legal landform recipe.
        empty_climate = (Psum <= 0) & (lb == 0) & (recipe_mass > 0)
        climate_total = np.zeros((h,w),np.float32)
        if empty_climate.any():
            climate_recipe = []
            for climate,slots in self.dw['open'].items():
                for slot in slots:
                    cw = (empty_climate & (picked == int(climate))).astype(np.float32) / len(slots)
                    bad = forbidden(np.full((h,w),slot,np.uint8)) | ((slot == 31) & clearance)
                    cw = cw * ~bad
                    climate_total += cw
                    climate_recipe.append(Candidate(4,slot,cw,support_band))
            for c in climate_recipe:
                c.weight = c.weight * recipe_mass / np.maximum(climate_total,1e-20)
            recipe.extend(climate_recipe)
        for i, slot in enumerate(gm.MATERIALS):
            # Keep the original multiplication order on legal branches.
            rw = recipe_mass * P[i]
            if slot in self.wood:
                # Geological recipes must not independently invent forest.
                removed = rw * (1-fd)
                rw = rw * fd
                for climate, slots in self.dw['open'].items():
                    for om in slots:
                        ow = removed * (picked == int(climate)) / len(slots)
                        bad = forbidden(np.full((h,w),om,np.uint8))
                        rejected_open += ow * bad
                        recipe.append(Candidate(4,om,ow * ~bad,support_band))
            recipe.append(Candidate(5,slot,rw,support_band))
        if rejected_open.any():
            for i,slot in enumerate(gm.MATERIALS):
                if slot not in self.wood:
                    recipe.append(Candidate(5,slot,rejected_open * P[i] /
                                            np.maximum(nonwood_total,1e-20),support_band))
        recipe_total = sum((c.weight for c in recipe),np.zeros((h,w),np.float32))
        natural.extend(recipe)
        town = [] if self.town is None else self.town.rows(y0, y1, w, support_band)
        farm = [] if self.farms is None else self.farms.rows(y0, y1, w, support_band)
        for c in town + farm:
            c.weight = c.weight * ~forbidden(np.broadcast_to(np.asarray(c.slot,np.uint8),(h,w)))
        def patch_coverage(candidates):
            # Overlapping authored patches describe one coverage field. Their
            # materials share that mass; an overlap cannot amplify coverage.
            coverage = np.zeros_like(F)
            total = np.zeros_like(F)
            for c in candidates:
                coverage = np.maximum(coverage, c.weight)
                total += c.weight
            scale = coverage / np.maximum(total, 1e-20)
            for c in candidates:
                c.weight = c.weight * scale
            return coverage
        town_mass = patch_coverage(town)
        authored_farm = patch_coverage(farm)
        farm_eligible = np.zeros_like(F)
        for c in natural:
            farm_eligible += c.weight * gm.FARM_SOURCE[np.asarray(c.slot,dtype=np.uint8)]
        farm_eligible = np.clip(farm_eligible,0,1)
        F *= farm_eligible * (~clearance & (support_band != 0))
        M *= farm_eligible * ~clearance
        M *= ~forbidden(np.full((h,w),14,np.uint8))
        farm_proportions=[]
        farm_total=np.zeros_like(F)
        if 'K' in self.fields:
            K=at(self.fields['K'])
            interior=farm_inner_fraction(F, gm.FARM_RIM, gm.FARM_FRAY_AMP)
            for j,(inner,rim) in enumerate(gm.FARM_KINDS):
                a=interior*float(gm.FARM_INNER[j])
                for slot,p in ((inner,K[j]*a),(rim,K[j]*(1-a))):
                    p=p*~forbidden(np.full((h,w),slot,np.uint8))
                    farm_proportions.append((slot,p));farm_total+=p
        F *= farm_total>0
        wet = []
        wet_mass = np.zeros_like(F)
        if self.wetland is not None:
            with self._wetland_lock:
                wet_mass, wet_band, valid = self.wetland.rows(y0, y1, w)
            # The provider describes geography, not ownership strength. Apply
            # the shared takeover once here; protected bytes always win.
            wet_mass = wet_mass * takeover * ~keep * valid * ~clearance * ~forbidden(np.full((h,w),31,np.uint8))
            wet = [Candidate(10, 31, wet_mass*(1-town_mass), wet_band)]
        available = (1-town_mass)*(1-wet_mass)
        farm_mass = available*(1-authored_farm)*F
        mud_mass = available*(1-authored_farm)*(1-F)*M
        natural_mass = available*(1-authored_farm)*(1-F)*(1-M)
        # A missing legal nonforest recipe is a rule/input error, not a slot-0
        # default. Only demand it where natural ownership actually needs it.
        no_recipe = ((recipe_mass > 0) & (recipe_total <= 0) &
                     (natural_mass > 0) & ~keep)
        no_nonwood = ((rejected_open > 0) & (nonwood_total <= 0) &
                     (recipe_mass > 0) & (natural_mass > 0) & ~keep)
        failure = no_recipe | no_nonwood
        if failure.any():
            context = {'takeover':takeover,'fd':fd,'F':F,'M':M,'clearance':clearance,
                       'L':lb,'picked':picked,'recipe_mass':recipe_mass,
                       'recipe_total':recipe_total,'nonwood_P':nonwood_total,
                       'rejected_open':rejected_open,'climate_open_total':climate_total}
            raise ValueError('No legal local nonforest landform/climate recipe for rejected natural mass; '
                             f'samples={_missing_details(natural,failure,forbidden,y0,context)}')
        for c in natural:
            c.weight = c.weight * natural_mass
        farm_recipe = []
        for slot,p in farm_proportions:
            farm_recipe.append(Candidate(6,slot,farm_mass*p/np.maximum(farm_total,1e-20),support_band))
        for c in farm:
            c.weight = c.weight * (1-town_mass) * (1-wet_mass)
        extra = []
        if self.authored_forest is not None:
            af = np.clip(at(self.authored_forest), 0, 1) * fd * ~clearance
            af *= ~forbidden(np.full((h,w),self.wood[0],np.uint8))
            for c in natural:
                c.weight *= 1-af
            extra.append(Candidate(12, self.wood[0], af*natural_mass, support_band))
        candidates = natural + farm_recipe + [Candidate(7, 14, mud_mass, support_band)] + farm + town + wet + extra
        palette_rule = self.dw.get('ground_palette_response', {})
        candidates = density_ground_palette(candidates, fd, picked, keep, forbidden,
                    self.tones, self.dw['open'], enabled=palette_rule.get('enabled', False),
                    sources=palette_rule.get('sources', ()), wood_slots=self.wood,
                    clearance=clearance)
        result, owner, report = select(candidates, preserved, keep, forbidden,
                                       self.tones, self.thresholds, self.draw.final_rows(y0, self.height),y0=y0,
                                       context={'takeover':takeover,'fd':fd,'F':F,'M':M,
                                                'clearance':clearance,'L':lb,'picked':picked})
        return result, owner, report


def town_candidates(plan, biome, forest, domains, original, reference,
                    town_clear, settings):
    """Authored town templates classified from shared forest intent, not a
    partially painted blend. Dry clearance is a candidate constraint."""
    import campaign_world as cw
    from terrain_noise import fbm
    oh, ow = reference.shape[:2]
    hh, ww = original.shape[:2]
    TH, TW = biome.shape
    r_in, r_out = np.asarray(settings['TOWN_GROUND_R']) * 8
    span = int(r_out) + 1
    oy, ox = np.mgrid[-span:span+1, -span:span+1]
    falloff = np.clip((r_out-np.hypot(ox,oy)) / (r_out-r_in), 0, 1)
    grain = .5 + .5 * fbm(ox/8., oy/8., (0,0), 1.2, 3, seed=95)
    coverage = np.clip((falloff-grain+.15)/.3, 0, 1)
    out = SparseCandidates(8)
    for town in plan['regions']:
        tq, tr = town['town']
        cx, cy = int(tq*8), oh-1-int(round((tr+tq%2*.5)*8))
        if not (span <= cx < ow-span and span <= cy < oh-span):
            continue
        yy, xx = ((oh-1-(cy+oy))//4).clip(0,TH-1), ((cx+ox)//4).clip(0,TW-1)
        fd = forest[yy,xx]
        normal = float(fd[span,span]) > .1 and float(fd.mean()) < float(settings['FOREST_TOWN'][2])
        clear = town_clear[yy,xx]
        cl = int(biome[min(int((tr+tq%2*.5)*2),TH-1),min(tq*2,TW-1)])
        dq, dr = settings['TOWN_GROUND'][cl]
        dx, dy = int(dq*8), hh-1-int(round((dr+dq%2*.5)*8))
        if not (span <= dx < ww-span and span <= dy < hh-span):
            raise ValueError('Town template is outside original texture')
        slots = original[dy-span:dy+span+1,dx-span:dx+span+1,0]
        ref = reference[cy-span:cy+span+1,cx-span:cx+span+1,0]
        dry = clear & (ref == 31)
        weight = np.maximum(coverage * normal, dry.astype(np.float32))
        weight *= ~domains['native_keep'][yy,xx]
        if np.any((weight > 0) & clear & (slots == 31)):
            raise ValueError(f"Wet town template in dry clearance: {town['region']}")
        out.add((cy-span,cy+span+1,cx-span,cx+span+1),slots,weight)
    return out


def prepare(*, fields, lon, lat, domains, forest, origin, cells, names, plan,
            biome, surface, alpha, physical, protected, plots, oasis,
            real_bio, bnames, real_elev, real_river, q, r, inherited, donor,
            water_field, settings, authored_forest=None):
    """Prepare shared fields and authored proposals once, before any blend draw."""
    import os
    import campaign_world as cw
    import ground_materials as gm
    import farm_fields
    import apply_climate
    import strategic_ink
    import map_hex
    ids = map_hex.region_ids(cells)
    ty, tx = np.mgrid[:biome.shape[0], :biome.shape[1]]
    hq, hr = cw.tile_hex(tx, ty)
    ground = apply_climate.ground(cells)
    river = np.isin(ground[hr,hq], (10,11)) | np.isin(ids[hr,hq],
                 [i for i,k in enumerate(names) if k.startswith('tw3k_riv_sea_')])
    lake = (ids[hr,hq] == names.index('3k_main_sea_lake')) if '3k_main_sea_lake' in names else np.zeros_like(physical)
    if os.environ.get('TW3K_FARM_MUD','1') != '0':
        towns = [(int(t['town'][0])*2, int((t['town'][1]+t['town'][0]%2*.5)*2),
                  farm_fields.kind_of(t.get('battle_layout'))) for t in plan['regions']]
        fields.update(gm.farm_mud_fields(real_elev,real_bio,bnames,lon,lat,
                      domains['material_takeover']*physical,biome,surface,
                      river,lake,real_river>0,towns,q,r,origin,oasis=oasis,cold=fields['cold']))
    clear = apply_climate.town_clear_hex(cells)[hr,hq]
    towns = town_candidates(plan,biome,forest,domains,donor,inherited,clear,settings)
    farms = SparseCandidates(9)
    rb, rn = strategic_ink.load()[:2]
    farm_report = farm_fields.paint(inherited,donor,plan,biome,surface,alpha,physical,
                    protected,plots,oasis,rb,rn,edited=~domains['native_keep'],ignore=None,
                    filler=domains['material_takeover']*domains['native_filler'],
                    candidate_sink=farms.add,domains=domains)
    del rb,rn,ty,tx,hq,hr,river,lake
    wet = None
    if os.environ.get('TW3K_CLIMATE_WETLANDS','1') != '0':
        # Shared ownership replaces the legacy native/filler zone, including
        # native edit exceptions. Keep the provider's other hard exclusions.
        wet = apply_climate.WetlandCandidates.from_inputs(cells,names,protected=domains['native_keep'],
                filler=True,water_field=water_field,domain=(cells[...,0] & 3) != 1)
    bake = GroundCandidates(fields,domains,forest,origin,inherited.shape[:2],town=towns,
                farms=farms,wetland=wet,town_clear=clear,authored_forest=authored_forest)
    bake.farm_report = farm_report
    bake.water_provenance = None if wet is None else wet.provenance
    return bake
