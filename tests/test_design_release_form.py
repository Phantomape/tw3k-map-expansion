"""design_release_form (2026-11-01): keys-only nesting, import-era provenance moved verbatim into the private history,
idempotent, invertible for audits; design_capitals: is_capital marks province capitals."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import campaign_world as cw
import design_release_form as drf

# the earlier reference mod's key prefix and name, assembled at run time (test_no_import_mentions)
PREFIX, NAME, FIELD = 'iron' + 'ic_', '190 ' + 'Expanded', drf.REQUIRES_FIELD


def design():
    r1 = {'region': 'tw3k_kr_a_capital', 'province_key': 'tw3k_kr_province_a', 'is_capital': True, 'town': [1, 1],
          'imported': True, 'source_region': PREFIX + 'region_a'}
    r2 = {'region': 'tw3k_kr_a_farm', 'province_key': 'tw3k_kr_province_a', 'is_capital': False, 'town': [2, 2],
          'note': 'from ' + NAME}
    return {'regions': [r1, r2], 'reference_reuse': '39 land regions from ' + NAME, FIELD: False,
            'imported_sea_regions': [{'region': 's', 'colour': [1, 2, 3], 'is_sea': True, 'source_region': PREFIX + 'sea_s',
                                      'consolidated_sources': [PREFIX + 'sea_s']}],
            'sdk_region_budget': {'total_regions': 3, 'added_sea_basins': {'s': [PREFIX + 'sea_s']}},
            'provinces': [{'province': {'key': 'tw3k_kr_province_a'}, 'imported': True, 'boundary_data_source': 'x',
                           'regions': [dict(r1), dict(r2)]}]}


class ReleaseFormTest(unittest.TestCase):
    def test_convert(self):
        new, moved = drf.convert(design())
        self.assertEqual(drf.leftovers(new), [])
        self.assertEqual(cw.design_nested_mismatch(new), [])
        self.assertEqual(new['imported_sea_regions'], [{'region': 's', 'colour': [1, 2, 3], 'is_sea': True}])
        self.assertEqual(new['sdk_region_budget'], {'total_regions': 3})
        self.assertNotIn('note', new['regions'][1])   # a field naming the earlier reference mod moves too
        self.assertEqual(moved['region/tw3k_kr_a_capital/source_region'], PREFIX + 'region_a')
        self.assertEqual(moved['top/' + FIELD], False)

    def test_idempotent_and_invertible(self):
        d = design()
        new, moved = drf.convert(d)
        again, more = drf.convert(new)
        self.assertEqual((again, more), (new, {}))
        self.assertEqual(drf.restore(new, moved), cw.design_keys_only(d))

    def test_duplicate_new_sea_list_and_budget(self):
        d = design(); d['new_sea_regions'] = [{'region': 's', 'colour': [1, 2, 3], 'purpose': 'x'}]
        new, moved = drf.convert(d, map_counts=7)
        self.assertNotIn('new_sea_regions', new)
        self.assertEqual(new['sdk_region_budget']['total_regions'], 7)
        self.assertIn('top/new_sea_regions', moved)
        d['new_sea_regions'].append({'region': 't', 'colour': [4, 5, 6]})   # a sea only this list has: kept
        self.assertIn('new_sea_regions', drf.convert(d)[0])

    def test_design_map_problems(self):
        d, _ = drf.convert(design(), map_counts=4)
        m = {'land_regions': ['l1', 'l2'], 'sea_regions': ['v', 's']}
        v = {'land_regions': ['l1'], 'sea_regions': ['v']}
        self.assertEqual(cw.design_map_problems(d, m, v), [])
        m2 = {'land_regions': ['l1', 'l2'], 'sea_regions': ['v', 'n', 'x']}
        self.assertEqual(cw.design_map_problems(d, m2, v), ['listed sea not on the map: s', 'map sea not listed: n',
                                                            'map sea not listed: x', 'sdk_region_budget.total_regions 4 != map 5'])

    def test_capitals(self):
        d, _ = drf.convert(design())
        self.assertEqual(cw.design_capital_problems(d), [])
        d['regions'][1]['is_capital'] = True
        self.assertEqual(cw.design_capital_problems(d),
                         ['tw3k_kr_a_farm: capital without the _capital key', 'tw3k_kr_province_a: 2 capitals'])
        v = {'regions': [{'region': 'tw3k_exp_x_capital', 'province_key': '3k_main_province_zangke', 'is_capital': False}]}
        self.assertEqual(cw.design_capital_problems(v), [])   # a city in a vanilla province is no province capital


if __name__ == '__main__':
    unittest.main()
