"""The Hehuang / Korea patch caches' one writer: bake_map step 10, and bake_map --diff after it writes the map (P6,
coordinator / integrator 11-01). It writes TW3K_WORKSPACE/analysis: a design candidate runs with TW3K_WORKSPACE set to
its scratch workspace (candidate_workspace.py), never the live one. ensure(map_cells, map_names)
rebuilds a cache (analysis/<name>-patch.npz; korea-patch-forests.npz with TW3K_KOREA_FORESTS=1) that is missing or stale
against the map the bake is building, and keeps a current one; the climate step (11) then loads them against the same
map. Stale = hehuang_terrain.design_mismatch, source_mismatch (inputs and build rules; a cache without those keys) or,
for korea, the forest rules_sha. A patch switched off (TW3K_<NAME>=0) is skipped. After a rebuild the cache is checked
again; RuntimeError if it still does not match. The pipeline's map checkpoint (patch_design) only checks.

  python ensure_patch_caches.py        # standalone: the same against the workspace's final map.hex
"""
import importlib
import json
import os
import sys

import numpy as np

import campaign_world as cw
import hehuang_terrain as ht


def stale(name, path, plan, map_cells=None, map_names=None):
    """The reason the cache at path cannot be used with this map (map_cells / map_names; default the final map.hex),
    or None."""
    if not path.is_file():
        return 'missing'
    with np.load(path) as z:
        why = ht.design_mismatch(z, plan) or ht.source_mismatch(name, z, None, map_cells, map_names)
        if not why and name == 'korea':
            try:
                ht.check_rules(name, z, path)
            except RuntimeError as e:
                why = str(e)
    return why


def ensure(names=ht.PATCHES, build=None, map_cells=None, map_names=None):
    """{name: 'current' | 'rebuilt (<reason>)' | 'off'}; raises RuntimeError when a rebuilt cache still does not match.
    map_cells / map_names: the map to build from and check against (bake_map step 10: the map it is building).
    build(name): rebuild one patch cache at its default path (default: <name>_patch.build(map_cells=, map_names=))."""
    build = build or (lambda n: importlib.import_module(f'{n}_patch').build(map_cells=map_cells, map_names=map_names))
    plan = json.loads(cw.DESIGN_PATH.read_text(encoding='utf-8'))
    out = {}
    for name in names:
        if os.environ.get('TW3K_' + name.upper(), '1') == '0':
            out[name] = 'off'
            continue
        path = ht.patch_path(name)
        why = stale(name, path, plan, map_cells, map_names)
        if why is None:
            out[name] = 'current'
            continue
        build(name)
        again = stale(name, path, plan, map_cells, map_names)
        if again is not None:
            raise RuntimeError(f'{name}: rebuilt {path} still does not match ({again})')
        out[name] = f'rebuilt ({why})'
    return out


def main():
    try:
        for name, state in ensure().items():
            print(f'patch cache {name}: {state}', flush=True)
    except RuntimeError as e:
        print('ERROR', e, flush=True)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
