"""Model instances of a campaign prop bundle (global_props.bin) in world coordinates, one reader for every check
and fix that compares props with the ground (stage_invariants prop_mesh_offset / prop_scene_offset, the resource
scene placement).

Leaves are world-space (translate_world_space_leaves / world_space_leaves.py): an instance's matrix translation
(m[9], m[10], m[11]) is its world x, y, z; the root/parent prefab matrices are not applied again (the offline view
reads them the same way, render_live_objects._props). Every leaf asset with a model table is read; parents (child
prefab lists) and model-less variant switches have none and are skipped.

  instances(bundle) -> list of Instance(model, x, y, z, scale_y, mode, region, prefab, key)

region / prefab come from the asset key terrain/campaigns/<v>/bmd_objects.<region>.<prefab>[.<n>].bin: one scene
(root) is one (region, prefab) group. mode is the record's BHM_* height mode: BHM_IGNORE y is an absolute height,
BHM_CUSTOM_VERTEX_OFFSETTING y is an offset over the terrain (vanilla cold mountains), not a height.
"""
import re
import struct
from collections import namedtuple

from props_bundle import embedded

# translate_campaign_lake_instances.MODEL (copied: that module imports campaign_world, which needs a workspace)
MODEL = re.compile(rb'(?:rigidmodels|terrain)/[A-Za-z0-9_./-]+\.(?:wsmodel|rigid_model_v2)', re.I)

Instance = namedtuple('Instance', 'model x y z scale_y mode region prefab key')
KEY = re.compile(r'bmd_objects\.([^.]+)\.(\d+)\.')
PARKED_Y = -100.   # disabled instances sit at y = -200 (reconcile_terrain_prefabs); the offline view skips y < -100
# The vanilla resource-scene templates complete_campaign_resource_scenes.py clones and seats on town terraces (its
# `templates` dict: anding 2412 / 2476, daijun 1912 / 1913, youzhou 1852, yanmen 1972 / 1973). Every placed scene is one
# bmd_objects.<region>.<num> group with num in this set (one root per region and template).
RESOURCE_TEMPLATES = (1852, 1912, 1913, 1972, 1973, 2412, 2476)
SCENE_SKIP_FAMILIES = ('vegetation', 'decals')   # scene pieces that do not stand on the ground as the scene does


def leaf_records(b):
    """[(model, 12-float matrix, height mode)] of one leaf asset (translate_campaign_lake_instances.records plus the
    BHM mode string). Raises AssertionError / struct.error on assets that are not leaves."""
    starts = []
    for m in MODEL.finditer(b):
        s = m.start()
        if (s >= 8 and struct.unpack_from('<H', b, s - 8)[0] == 2 and struct.unpack_from('<H', b, s - 2)[0] == len(m.group())
                and 0 < struct.unpack_from('<I', b, s - 6)[0] < 10000):
            starts.append(s)
    assert len(starts) == 1, ('model table', len(starts))
    s = starts[0]; n = struct.unpack_from('<I', b, s - 6)[0]; pos = s - 2; models = []
    for _ in range(n):
        le = struct.unpack_from('<H', b, pos)[0]; models.append(b[pos + 2:pos + 2 + le].decode()); pos += 2 + le
    count = struct.unpack_from('<I', b, pos)[0]; p = pos + 4; out = []
    for _ in range(count):
        version, idx = struct.unpack_from('<HI', b, p)
        assert version == 20 and idx < n, (p, version, idx)
        mode = p + 107; le = struct.unpack_from('<H', b, mode - 2)[0]
        assert 4 < le < 100 and b[mode:mode + 4] == b'BHM_'
        out.append((models[idx], struct.unpack_from('<12f', b, p + 24), b[mode:mode + le].decode()))
        p += 131 + le
    assert p <= len(b)
    return out


def scene_of(key):
    """(region, prefab number) of a bmd_objects asset key, else ('?', -1)."""
    m = KEY.search(key)
    return (m.group(1), int(m.group(2))) if m else ('?', -1)


def instances(bundle):
    """Every model instance of a global_props.bin bundle (bytes), world coordinates (module docstring)."""
    assets = embedded(bundle)[0]
    out = []
    for k, b in assets.items():
        if not k.endswith('.bin') or k.endswith('/bmd_objects.bin') or b'rigidmodels/' not in b:
            continue
        try:
            recs = leaf_records(b)
        except (AssertionError, struct.error, UnicodeDecodeError):
            continue
        region, prefab = scene_of(k)
        for model, m, mode in recs:
            out.append(Instance(model.lower(), m[9], m[10], m[11], m[4], mode, region, prefab, k))
    return out


def family(model):
    """'mountains', 'rocks', 'settlements', 'vegetation', ... (rigidmodels/campaign/<family>/...), else '?'."""
    p = model.split('/')
    return p[2] if len(p) > 2 and p[0] == 'rigidmodels' else '?'


def scene_piece_mask(fam, mode, y, prefab):
    """bool per instance (arrays): a piece of a placed resource scene that stands on the ground with it: prefab number
    in RESOURCE_TEMPLATES, BHM_IGNORE (absolute y), not parked, family not in SCENE_SKIP_FAMILIES. Group the pieces by
    (region, prefab) for one entry per placed scene."""
    import numpy as np
    fam, mode, prefab = np.asarray(fam), np.asarray(mode), np.asarray(prefab)
    return (np.isin(prefab, RESOURCE_TEMPLATES) & (mode == 'BHM_IGNORE') & (np.asarray(y, np.float64) > PARKED_Y)
            & ~np.isin(fam, SCENE_SKIP_FAMILIES))
