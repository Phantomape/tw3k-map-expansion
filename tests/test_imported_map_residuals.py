"""Synthetic checks for coordinate alignment and false-positive controls."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_imported_map_residuals import (  # noqa: E402
    audit_maps, compare_masks, compare_roads, merged_sea_sources,
)
import map_hex  # noqa: E402


def map_fixture(ids, land, seas):
    ids = np.asarray(ids, dtype=np.int32)
    cells = np.zeros((*ids.shape, 16), dtype=np.uint8)
    map_hex.set_region_ids(cells, ids)
    return {"cells": cells, "land_regions": land, "sea_regions": seas}


class ImportedResidualTests(unittest.TestCase):
    def test_offset_and_full_current_union(self):
        source = np.array([[1, 0, 0], [0, 1, 0]], dtype=bool)
        current = np.zeros((4, 6), dtype=bool)
        current[1:3, 2:5] = source
        self.assertTrue(compare_masks(source, current, (2, 1))["complete_mask_equal"])
        current[3, 0] = True  # Outside the source frame, still part of this region.
        measured = compare_masks(source, current, (2, 1))
        self.assertEqual(measured["intersection_hexes"], 2)
        self.assertAlmostEqual(measured["iou"], 2 / 3)
        self.assertFalse(measured["complete_mask_equal"])

    def test_consolidated_seas_resolve_aliases_and_compare_the_union(self):
        sea = {"region": "new_sea", "source_region": "old_a",
               "consolidated_sources": ["mapped_a", "old_b", "new_canvas_sea"]}
        aliases = {"old_a": "mapped_a", "old_b": "old_b"}
        sources, unresolved = merged_sea_sources(sea, aliases, ["old_a", "old_b"])
        self.assertEqual(sources, ["old_a", "old_b"])
        self.assertEqual(unresolved, ["new_canvas_sea"])
        reference = map_fixture([[0, 1, 1], [0, 0, 1]], [], ["old_a", "old_b"])
        current = map_fixture([[0, 0, 0], [0, 0, 0]], [], ["new_sea"])
        design = {"regions": [], "imported_region_map": aliases,
                  "imported_sea_regions": [sea]}
        report = audit_maps(reference, current, reference, design, {"regions": []}, (0, 0))
        merged = report["consolidated_sea_regions"][0]
        self.assertEqual(merged["source_count"], 2)
        self.assertEqual(merged["source_hexes"], 6)
        self.assertTrue(merged["complete_mask_equal"])

    def test_roads_exclude_matches_already_shared_with_vanilla(self):
        reference = np.array([[0, 3, 4, 9]])
        vanilla = np.array([[0, 3, 0, 0]])
        current = np.zeros((2, 6), dtype=int)
        current[1, 1:5] = [0, 3, 4, 0]
        measured = compare_roads(reference, vanilla, current,
                                 np.ones_like(reference, dtype=bool), (1, 1))
        # Shared absent roads and a shared road (0 and 3) are not evidence.
        self.assertEqual(measured["reference_different_from_vanilla_hexes"], 2)
        self.assertEqual(measured["same_reference_road_mask_in_current"], 1)
        self.assertEqual(measured["vanilla_road_mask_now"], 1)
        self.assertEqual(measured["other_road_mask_now"], 0)
        self.assertTrue(measured["counts_partition_candidates"])

    def test_empty_masks_do_not_claim_an_exact_geometry_match(self):
        measured = compare_masks(np.zeros((2, 2)), np.zeros((4, 4)), (1, 1))
        self.assertIsNone(measured["iou"])
        self.assertFalse(measured["complete_mask_equal"])


if __name__ == "__main__":
    unittest.main()
