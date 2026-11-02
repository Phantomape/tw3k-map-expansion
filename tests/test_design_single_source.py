"""design_single_source (2026-11-01): region records live only in the top-level design['regions']; provinces[].regions
hold bare {'region': key} entries matching each region's province_key (the 10-30 design had 139 stale nested copies and
the c4 generator built its region list from them). The DB step never writes the design: it stops when the design is not
normalized (one design writer)."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import campaign_world as cw
import stage_invariants as si


def design():
    a = {'region': 'a', 'town': [1, 2], 'name': 'A', 'province_key': 'p'}
    b = {'region': 'b', 'town': [3, 4], 'name': 'B', 'province_key': 'p'}
    return {'regions': [a, b], 'provinces': [{'province': {'key': 'p'}, 'regions': [{'region': 'a'}, {'region': 'b'}]}]}


class DesignSingleSourceTest(unittest.TestCase):
    def test_keys_only_passes(self):
        self.assertEqual(cw.design_nested_mismatch(design()), [])

    def test_nested_body_fails(self):
        d = design(); d['provinces'][0]['regions'][1] = dict(d['regions'][1])   # a full copy, even an identical one
        self.assertEqual(cw.design_nested_mismatch(d), ['b: body'])

    def test_grouping_follows_province_key(self):
        d = design(); d['regions'][0]['province_key'] = 'q'
        self.assertEqual(cw.design_nested_mismatch(d), ['a: province_key'])

    def test_missing_extra_twice(self):
        d = design(); d['provinces'][0]['regions'] = [{'region': 'a'}, {'region': 'a'}, {'region': 'z'}]
        self.assertEqual(cw.design_nested_mismatch(d), ['a: twice', 'nested only z', 'top only b'])

    def test_keys_only_conversion(self):
        d = design(); d['provinces'][0]['regions'] = [dict(r) for r in d['regions']]
        k = cw.design_keys_only(d)
        self.assertEqual(cw.design_nested_mismatch(k), [])
        self.assertEqual(k['regions'], d['regions'])

    def test_registered_at_pipeline_start(self):
        self.assertEqual(si.INVARIANTS['design_single_source'][2], ('map',))

    def test_database_step_checks_and_never_writes(self):
        build = (ROOT / 'scripts/all_directions_build.py').read_text(encoding='utf-8')
        self.assertIn('design_nested_mismatch', build[build.index('def database('):build.index('\ndef native(')])
        pp = (ROOT / 'scripts/prepare_province.py').read_text(encoding='utf-8')
        body = pp[pp.index('def main('):]
        write = body.index("write_text(json.dumps(want")
        self.assertIn("TW3K_NORMALIZE_DESIGN')!='1'", body[:write])   # the only design write sits behind the switch
        self.assertEqual(body.count('current-region-design.json').__ge__(1), True)
        self.assertNotIn("write_text(json.dumps(plan,ensure_ascii=False,indent=2)", body)


if __name__ == '__main__':
    unittest.main()
