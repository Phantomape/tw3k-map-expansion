"""Write root translations into every leaf instance (see world_space_leaves.py).

Runs after the river-root step and before the lake step (lake leaf files also
hold world-space instances) on the staged global_props.bin, which
reconcile_terrain_prefabs.py rebuilds from the pre-polish base every run.
Changed leaves also ship loose, like the lake leaves. Any other loose copy of a
bundled leaf left in the stage by an earlier build (another canvas shift) is
reset to the bundled bytes, so loose and embedded leaves never disagree.
Relocated resource scenes carry their own leaf copies
(complete_campaign_resource_scenes.py); bake() refuses a leaf shared by roots.

  python translate_world_space_leaves.py
"""
import json
from concurrent.futures import ThreadPoolExecutor

import campaign_world as _frame
from campaign_world import TERRAIN_VARIANTS
from terrain_checkpoint import StepSource, inputs
from repair_campaign_river_roots import embedded
from world_space_leaves import bake

A = _frame.WORKSPACE
S = A / 'work/release-input'


def main():
    step = StepSource('world-space-leaves', inputs())
    report = []
    for variant in TERRAIN_VARIANTS:
        key = f'terrain/campaigns/{variant}/global_props.bin'
        data = step.read(S / key)
        if data is None:
            print(variant, 'world-space leaves already current', flush=True)
            continue
        out, changed, moved = bake(data)
        if not moved:
            raise RuntimeError('No world-space leaf instance moved; the leaf format or root translations changed')
        (S / key).write_bytes(out)
        # The base rebuilds the same leaves every run: rewrite a loose leaf only when its
        # bytes differ (stamp and content caches key on size+mtime, never on a newer mtime).
        # Compared on threads: a cold open (Defender scan) costs ~7 ms per file here.
        with ThreadPoolExecutor(16) as pool:
            same = list(pool.map(lambda kb: (S / kb[0]).is_file() and (S / kb[0]).stat().st_size == len(kb[1]) and (S / kb[0]).read_bytes() == kb[1], changed.items()))
        for (k, b), unchanged in zip(changed.items(), same):
            if unchanged:
                continue
            p = S / k
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b)
        reset = 0
        for k, b in embedded(out)[0].items():
            p = S / k
            if k not in changed and k != key.replace('global_props.bin', 'bmd_objects.bin') and p.exists() and p.read_bytes() != b:
                p.write_bytes(b)
                reset += 1
        report.append({'variant': variant, 'leaf_files': len(changed), 'instances': moved, 'stale_loose_reset': reset})
        print(variant, 'world-space leaf instances moved', moved, 'in', len(changed), 'leaf files; stale loose leaves reset', reset, flush=True)
    step.finish()
    if report:
        (A / 'analysis/world-space-leaves.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
