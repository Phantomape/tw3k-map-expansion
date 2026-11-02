"""Check the SDK map export's settlement anchors against vanilla and the design.

Baseline is the vanilla game's own campaign_maps/3k_dlc07_main_map/map_data.esf:
every vanilla settlement must keep its vanilla anchors (shifted by the native
offset), every authored region must have valid anchors at its design town, and
nothing else may appear. (Formerly compared with an earlier expanded export and
a hard-coded 420 settlements.)
"""
import json, subprocess, tempfile
from pathlib import Path

from analyze_reference import ROOT, digest
import campaign_world as _frame
from repair_campaign_river_roots import Assets

A = _frame.WORKSPACE
TOOL = ROOT / 'tools/caime-1.1.0/lib/app/Tw3kSettlementAudit.exe'
NONE = 65535


def rows(path):
    result = subprocess.run([str(TOOL), str(path)], check=True, capture_output=True, text=True)
    return {s.split('\t')[0]: list(map(int, s.split('\t')[1:])) for s in result.stdout.splitlines()}


def shifted(v):
    """Native anchors -> canvas. Fields 0-5 are primary/approach/port (x, y) pairs:
    x + PAD_LEFT, y + PAD_BOTTOM; fields 6-7 are slot counts and stay."""
    return [n if (n == NONE or i >= 6) else n + (_frame.PAD_LEFT if i % 2 == 0 else _frame.PAD_BOTTOM) for i, n in enumerate(v)]


def main():
    p = A / 'work/standalone_kit/working_data/campaign_maps/3k_dlc07_main_map/map_data.esf'
    current = rows(p)
    with tempfile.TemporaryDirectory() as tmp:
        vp = Path(tmp) / 'map_data.esf'
        vp.write_bytes(Assets(original_only=True).read('campaign_maps/3k_dlc07_main_map/map_data.esf'))
        vanilla = rows(vp)
    plan = json.loads(_frame.DESIGN_PATH.read_text(encoding='utf-8'))
    auth = {r['region']: r for r in plan['regions']}
    expected = set(vanilla) | set(auth)
    if set(current) != expected:
        raise AssertionError({'missing': sorted(expected - set(current))[:10], 'unexpected': sorted(set(current) - expected)[:10]})
    for key, v in current.items():
        assert len(v) == 8 and all(n != NONE for n in v[:4]), (key, 'invalid primary or approach anchor', v)
        if key in auth:
            assert max(abs(v[i] - auth[key]['town'][i]) for i in (0, 1)) <= 1, (key, 'exported town offset', v, auth[key]['town'])
        else:
            assert v == shifted(vanilla[key]), (key, 'vanilla settlement anchor changed', v, shifted(vanilla[key]))
    ports = [k for k, v in current.items() if v[4] != NONE]
    report = {'status': 'pass', 'baseline': 'vanilla map_data.esf (shifted by the native offset)', 'settlements': len(current),
              'authored_anchors_verified': len(auth), 'vanilla_anchors_identical': len(vanilla), 'invalid_primary_anchors': 0,
              'port_anchors': ports, 'source_map_sha256': digest(A / 'work/final-map/map.hex'), 'map_data_sha256': digest(p)}
    (A / 'analysis/exported-settlement-acceptance.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f'Settlement anchors PASS: {len(auth)} authored at design towns, {len(vanilla)} vanilla identical to vanilla', flush=True)


if __name__ == '__main__':
    main()
