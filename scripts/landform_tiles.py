"""The persisted landform field (bake design step 2): analysis/landform-tiles.npz.

polish's GROUND MATERIALS block writes it once per run (3k_dlc07_main_map) from the same tile_fields call
that paints the ground, so every reader judges a tile with the generator's own inputs instead of
re-classifying: audit_terrain_realism (--profiles, --gate), terrain_regression (desert_banned_share_new),
the offline replica (analysis/relief-ground/relief_proto.py) and the step invariants.

Fields (canvas tile grid, south-up, like west-real-terrain.npz):
  dom      uint8    1 + index (biome_profiles.LANDFORMS order) of the dominant landform under the ban reach
  domw     float16  its weight (class x gain x reach); the ban applies where domw > BAN_MIN
                    (float32 exact in the visual-domains schema 3)
  reach    float16  real ground x donor share x physical x (1 - 4 oasis) x (1 - 4 keep): the ban reach
                    (float32 exact in schema 3)
  L        uint8    schema 3 only: exact enforced landform index (0 outside the producer's ban mask);
                    absent in schema 2, where enforced_mask derives it from dom/domw as before
  allowed  uint32   per-slot whitelist bits on top (biome_profiles.allowed_slots)
  eco_bio  uint8    the ecotone-edited RESOLVE class polish painted with (real_bio_eco)
  ground_w, desert_w, oasis, keep, river   float16  the tile_fields inputs (offline replica)
  forest_density float16  biome_profiles.forest_density (the ground's woodland and the props bake's trees)
  forest_fill    float16  biome_profiles.forest_fill (the fill fade inside forest_density: trees = fill_mask & u < density / fill)
                 (both float32 in schema 3)
  schema 3 only (visual domains passed, TW3K_SINGLE_GROUND=1):
  native_keep, native_filler, canvas_land bool, exact visual-domain masks
  material_takeover, vegetation_takeover float32, exact shared visual ownership weights
Meta biome_names: the class names of eco_bio.
  w_alpine float16  raw classify weight alpine_meadow + alpine_steppe, before CLASS_GAIN and the real-ground
                    weight (props bake: biome_profiles.alpine_treeline_tile compares TREELINE_CLASS_MIN with it)
Meta: landforms, ban_enforced, ban_min, map_sha256, design_sha256, canvas, schema.
numpy only.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA = 2   # 2: w_alpine (legacy path: no visual domains passed; byte-identical to the pre-domain writer)
SCHEMA_DOMAINS = 3   # 3: exact L/domw/reach float32 and exact shared visual domains (written only with visual_domains)
NAME = 'analysis/landform-tiles.npz'


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


def meta(A):
    """Provenance of a workspace: map.hex and design sha256, canvas stamp."""
    import campaign_world as cw
    design = Path(A) / 'work/current-region-design.json'
    design = cw.DESIGN_PATH if Path(A).resolve() == cw.WORKSPACE.resolve() or not design.is_file() else design
    return {'map_sha256': _sha(Path(A) / 'work/final-map/map.hex'), 'design_sha256': _sha(design),
            'canvas': cw.canvas_stamp()}


def save(A, fields, eco_bio, inputs):
    """Write <A>/analysis/landform-tiles.npz (through a temporary file). fields: tile_fields' result;
    inputs: {'ground_w', 'desert_w', 'oasis', 'keep', 'river'} arrays (None -> zeros),
    optional visual_domains (visual_domains.build result), forest_density and forest_fill.
    Exact ownership/protection masks and the ban domain are never reconstructed from float16."""
    import biome_profiles as bp
    path = Path(A) / NAME
    shape = fields['dom'].shape
    f16 = lambda a: (np.zeros(shape, np.float16) if a is None else np.asarray(a, np.float32).astype(np.float16))
    domains = inputs.get('visual_domains')
    if domains is None:
        # Legacy path (TW3K_SINGLE_GROUND / TW3K_VISUAL_DOMAINS off): exactly the schema-2 writer, same keys,
        # dtypes and order, so landform-tiles.npz stays byte-identical for the same inputs.
        m = dict(meta(A), schema=SCHEMA, landforms=list(bp.LANDFORMS), ban_enforced=list(bp.BAN_ENFORCED), ban_min=bp.BAN_MIN)
        if inputs.get('biome_names') is not None:
            m['biome_names'] = [str(n) for n in inputs['biome_names']]   # eco_bio's class names (the props bake's species read eco_bio too)
        from map_layers import savez_atomic   # byte-deterministic (np.savez_compressed stamps the time)
        savez_atomic(path, dom=fields['dom'], domw=fields['domw'].astype(np.float16), reach=f16(fields['reach']),
                     allowed=fields.get('allowed', bp.allowed_slots(shape)), eco_bio=np.asarray(eco_bio, np.uint8),
                     w_alpine=f16(fields['w_alpine']),
                     forest_density=f16(inputs.get('forest_density')),
                     forest_fill=f16(inputs.get('forest_fill')),
                     **{k: f16(inputs.get(k)) for k in ('ground_w', 'desert_w', 'oasis', 'keep', 'river')},
                     meta=np.array(json.dumps(m)))
        return path
    m = dict(meta(A), schema=SCHEMA_DOMAINS, landforms=list(bp.LANDFORMS), ban_enforced=list(bp.BAN_ENFORCED), ban_min=bp.BAN_MIN)
    if inputs.get('biome_names') is not None:
        m['biome_names'] = [str(n) for n in inputs['biome_names']]   # eco_bio's class names (the props bake's species read eco_bio too)
    import visual_domains as vd
    vd.validate(domains, shape, check_producer=True)
    extra = {key: np.asarray(domains[key]) for key in vd.FIELDS}
    forest = {}
    for key in ('forest_density', 'forest_fill'):
        if inputs.get(key) is None:
            raise ValueError(f'visual-domain landform field requires {key}')
        forest[key] = np.asarray(inputs[key], np.float32)
        if forest[key].shape != shape or not np.isfinite(forest[key]).all():
            raise ValueError(f'landform field: invalid {key}')
        if (forest[key][domains['native_keep']] != 0).any():
            raise ValueError(f'landform field: {key} is nonzero on native_keep')
    m['visual_domains'] = dict(domains['meta'])
    m['forest_producer_sha256'] = _forest_producer_sha()
    m['forest_rules_sha256'] = _forest_rules_sha()
    m['forest_fields_sha256'] = vd._json_sha({key: vd._array_sha(a) for key, a in forest.items()})
    L = fields.get('L')
    if L is None:
        # Older callers may lack L; derive once from the exact float32 producer values.
        ids = [list(bp.LANDFORMS).index(n) + 1 for n in bp.BAN_ENFORCED]
        L = np.where((np.asarray(fields['domw'], np.float32) > bp.BAN_MIN) & np.isin(fields['dom'], ids),
                     fields['dom'], 0).astype(np.uint8)
    from map_layers import savez_atomic   # byte-deterministic (np.savez_compressed stamps the time)
    savez_atomic(path, dom=fields['dom'], domw=np.asarray(fields['domw'], np.float32),
                 reach=np.asarray(fields['reach'], np.float32), L=np.asarray(L, np.uint8),
                 allowed=fields.get('allowed', bp.allowed_slots(shape)), eco_bio=np.asarray(eco_bio, np.uint8),
                 w_alpine=f16(fields['w_alpine']),
                 **forest, **extra,
                 **{k: f16(inputs.get(k)) for k in ('ground_w', 'desert_w', 'oasis', 'keep', 'river')},
                 meta=np.array(json.dumps(m)))
    return path


def _forest_producer_sha():
    """Hash the code that creates the shared target, not just its visual-domain producer."""
    code_sha = lambda name: hashlib.sha256(Path(__file__).with_name(name).read_text(
        encoding='utf-8').encode('utf-8')).hexdigest()
    return hashlib.sha256((code_sha('biome_profiles.py') + code_sha('frontier_forest.py')).encode('ascii')).hexdigest()


def _forest_rules_sha():
    import biome_profiles as bp
    # fill_weight reads constants in frontier_forest; its complete source is hashed above.
    return hashlib.sha256(json.dumps(bp.CONSTS['ground_materials']['FOREST_DENSITY'],
                                    sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()


def hex_view(lt, key, shape=None):
    """A tile field read at every hex's block-centre tile (2q + 1, 2r + q%2 + 1), as (rows r, cols q); shape:
    the hex grid (default: the tile grid halved). Hexes whose centre tile is off the grid read 0."""
    a = lt[key]
    H, W = shape if shape is not None else (a.shape[0] // 2, a.shape[1] // 2)
    rr, qq = np.mgrid[:H, :W]
    cy, cx = 2 * rr + qq % 2 + 1, 2 * qq + 1
    ok = (cy < a.shape[0]) & (cx < a.shape[1])
    out = np.zeros((H, W), a.dtype)
    out[ok] = a[cy[ok], cx[ok]]
    return out


def load(A, check=True):
    """The field of workspace A as a dict (meta parsed), or None when it is missing. check: refuse a field
    written for another map.hex, design or canvas (RuntimeError)."""
    path = Path(A) / NAME
    if not path.is_file():
        return None
    with np.load(path) as z:
        out = {k: z[k] for k in z.files if k != 'meta'}
        out['meta'] = json.loads(str(z['meta'].reshape(-1)[0]))
    if check:
        cur = meta(A)
        bad = [k for k in cur if out['meta'].get(k) != cur[k]]
        if bad:
            raise RuntimeError(f'{path} was written for another {", ".join(bad)}')
        if 'visual_domains' in out['meta']:
            import visual_domains as vd
            try:
                vd.validate({key: out[key] for key in vd.FIELDS} | {'meta': out['meta']['visual_domains']},
                            out['dom'].shape, check_producer=True)
            except (ValueError, KeyError) as e:
                raise RuntimeError(f'{path}: {e}') from e
            if out['meta'].get('forest_producer_sha256') != _forest_producer_sha():
                raise RuntimeError(f'{path}: stale forest producer; rerun ground bake')
            if out['meta'].get('forest_rules_sha256') != _forest_rules_sha():
                raise RuntimeError(f'{path}: stale forest rules; rerun ground bake')
            if out['meta'].get('forest_fields_sha256') != vd._json_sha(
                    {key: vd._array_sha(out[key]) for key in ('forest_density', 'forest_fill')}):
                raise RuntimeError(f'{path}: forest field hash mismatch')
    return out


def enforced_mask(lt, landforms=None):
    """bool per tile: the ban applies (dominant landform in `landforms`, default the field's ban_enforced, and
    its weight above ban_min)."""
    names = lt['meta']['landforms']
    want = lt['meta']['ban_enforced'] if landforms is None else landforms
    ks = [names.index(n) + 1 for n in want if n in names]
    if 'L' in lt and set(want) <= set(lt['meta']['ban_enforced']):
        return np.isin(lt['L'], ks)
    return np.isin(lt['dom'], ks) & (lt['domw'].astype(np.float32) > lt['meta']['ban_min'])


def banned_share(mat, lt, sel=None, landforms=None):
    """{landform: {'tiles', 'banned', 'share'}}: tile-centre materials (mat, the field's grid) holding a slot
    banned for their dominant landform (biome_profiles.AUDIT, the slots ground_materials redraws), minus the
    per-slot whitelist, on enforced tiles (enforced_mask) inside sel. The one definition the gates count."""
    import biome_profiles as bp
    names = lt['meta']['landforms']
    want = lt['meta']['ban_enforced'] if landforms is None else landforms
    on = enforced_mask(lt, want)
    if sel is not None:
        on &= sel
    out = {}
    for n in want:
        if n not in names:
            continue
        t = on & (lt['dom'] == names.index(n) + 1)
        banned = np.zeros(256, bool); banned[list(bp.AUDIT[n]['banned'] or ())] = True
        m = mat[t]
        hit = banned[m] & ~((lt['allowed'][t] >> m.astype(np.uint32)) & 1).astype(bool)
        out[n] = {'tiles': int(t.sum()), 'banned': int(hit.sum()), 'share': round(float(hit.sum()) / max(1, int(t.sum())), 5)}
    return out
