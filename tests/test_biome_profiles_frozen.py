"""stage_invariants biome_profiles_frozen (2026-11-01): the pipeline start fails when the biome profile table differs from
analysis/biome-profiles-frozen.json or its review JSON is stale (P6 korea_patch merged without --refreeze)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import biome_profiles as bp
import check_biome_profiles as cbp
import stage_invariants as si


class BiomeProfilesFrozenTest(unittest.TestCase):
    def setUp(self):
        self.old = cbp.FROZEN, cbp.REVIEW
        self.d = tempfile.TemporaryDirectory()
        d = Path(self.d.name)
        cbp.FROZEN, cbp.REVIEW = d / 'frozen.json', d / 'review.json'
        cbp.FROZEN.write_text(json.dumps({'snapshot': bp.snapshot()}), encoding='utf-8')
        bp.dump_json(cbp.REVIEW)

    def tearDown(self):
        cbp.FROZEN, cbp.REVIEW = self.old
        self.d.cleanup()

    def test_registered_at_pipeline_start(self):
        self.assertEqual(si.INVARIANTS['biome_profiles_frozen'][2], ('map',))

    def test_current_table_passes(self):
        self.assertEqual(si.biome_profiles_frozen(None)['value'], 0)

    def test_unfrozen_region_entry_fails(self):
        snap = bp.snapshot(); snap.pop('regions.korea_patch')   # the P6 miss: entry in the table, not in the snapshot
        cbp.FROZEN.write_text(json.dumps({'snapshot': snap}), encoding='utf-8')
        r = si.biome_profiles_frozen(None)
        self.assertEqual(r['value'], 1)
        self.assertEqual(r['detail']['differ'], ['regions.korea_patch'])

    def test_stale_review_json_fails(self):
        cbp.REVIEW.write_text('{}\n', encoding='utf-8')
        self.assertEqual(si.biome_profiles_frozen(None)['value'], 1)


if __name__ == '__main__':
    unittest.main()
