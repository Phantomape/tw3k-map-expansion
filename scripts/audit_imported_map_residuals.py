"""Read-only, coordinate-aligned diagnostics for documented imported map regions.

This tool does not remove content, change provenance, apply a license, or determine
copyright status. Its only write is the JSON path explicitly passed with --out.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import map_hex
import map_layers


ROOT = Path(__file__).resolve().parents[1]


def aligned_slices(source_shape, current_shape, offset):
    """Slices for source (q,r) -> current (q+dq,r+dr), including clipped frames."""
    sh, sw = source_shape
    ch, cw = current_shape
    dq, dr = (int(v) for v in offset)
    x0, x1 = max(0, -dq), min(sw, cw - dq)
    y0, y1 = max(0, -dr), min(sh, ch - dr)
    if x0 >= x1 or y0 >= y1:
        return None
    return ((slice(y0, y1), slice(x0, x1)),
            (slice(y0 + dr, y1 + dr), slice(x0 + dq, x1 + dq)))


def compare_masks(source_mask, current_mask, offset):
    """Full-map IoU: current territory outside the source frame remains in union."""
    source_mask = np.asarray(source_mask, dtype=bool)
    current_mask = np.asarray(current_mask, dtype=bool)
    if source_mask.ndim != 2 or current_mask.ndim != 2:
        raise ValueError("Region masks must be two dimensional")
    slices = aligned_slices(source_mask.shape, current_mask.shape, offset)
    intersection = 0
    if slices is not None:
        ss, cs = slices
        intersection = int(np.count_nonzero(source_mask[ss] & current_mask[cs]))
    source_count = int(np.count_nonzero(source_mask))
    current_count = int(np.count_nonzero(current_mask))
    union = source_count + current_count - intersection
    return {
        "source_hexes": source_count,
        "current_hexes": current_count,
        "intersection_hexes": intersection,
        "union_hexes": union,
        "iou": intersection / union if union else None,
        "complete_mask_equal": bool(source_count and
                                    source_count == current_count == intersection),
    }


def merged_sea_sources(sea, imported_region_map, reference_sea_names):
    """Resolve current consolidated-source aliases into original reference keys."""
    reference_sea_names = set(reference_sea_names)
    reverse = {}
    for source, destination in imported_region_map.items():
        reverse.setdefault(destination, []).append(source)
    candidates = list(sea.get("consolidated_sources") or [])
    if sea.get("source_region"):
        candidates.append(sea["source_region"])
    resolved = set()
    unresolved = set()
    for name in candidates:
        if name in reference_sea_names:
            resolved.add(name)
        else:
            found = set(reverse.get(name, [])) & reference_sea_names
            resolved.update(found)
            if not found:
                unresolved.add(name)
    return sorted(resolved), sorted(unresolved)


def compare_roads(reference_road, vanilla_road, current_road, source_land, offset):
    """Count matches only where reference roads differ from vanilla, not shared zeros."""
    reference_road = np.asarray(reference_road)
    vanilla_road = np.asarray(vanilla_road)
    current_road = np.asarray(current_road)
    source_land = np.asarray(source_land, dtype=bool)
    if reference_road.shape != vanilla_road.shape or source_land.shape != reference_road.shape:
        raise ValueError("Reference, vanilla and imported-land mask need the same frame")
    candidates = source_land & (reference_road != vanilla_road)
    total = int(np.count_nonzero(candidates))
    compared = same_reference = same_vanilla = other = 0
    slices = aligned_slices(reference_road.shape, current_road.shape, offset)
    if slices is not None:
        ss, cs = slices
        eligible = candidates[ss]
        ref = reference_road[ss]
        vanilla = vanilla_road[ss]
        current = current_road[cs]
        compared = int(np.count_nonzero(eligible))
        same_reference = int(np.count_nonzero(eligible & (current == ref)))
        same_vanilla = int(np.count_nonzero(eligible & (current == vanilla)))
        other = int(np.count_nonzero(eligible & (current != ref) & (current != vanilla)))
    return {
        "selection": "documented active imported land in reference AND reference road != vanilla road",
        "reference_different_from_vanilla_hexes": total,
        "compared_hexes": compared,
        "same_reference_road_mask_in_current": same_reference,
        "vanilla_road_mask_now": same_vanilla,
        "other_road_mask_now": other,
        "outside_current_frame": total - compared,
        "counts_partition_candidates": same_reference + same_vanilla + other == compared,
    }


def audit_maps(reference, current, vanilla, design, early_plan, offset):
    """Audit in-memory maps and plans; never mutate input arrays or dictionaries."""
    reference_ids = map_hex.region_ids(reference["cells"])
    current_ids = map_hex.region_ids(current["cells"])
    if reference["cells"].shape != vanilla["cells"].shape:
        raise ValueError("Reference and vanilla must share their unshifted map frame")
    rn = map_layers.region_names(reference)
    cn = map_layers.region_names(current)
    ri = {name: i for i, name in enumerate(rn)}
    ci = {name: i for i, name in enumerate(cn)}
    imported = [row for row in design["regions"] if row.get("imported")]
    early = {row["region"]: row for row in early_plan["regions"]}
    land_rows = []
    unchanged_towns = []
    source_land_ids = []
    missing = []
    dq, dr = offset
    for row in imported:
        source, destination = row.get("source_region"), row["region"]
        if source not in ri or destination not in ci:
            missing.append({"source_region": source, "current_region": destination,
                            "missing_in": [kind for kind, absent in
                                           (("reference", source not in ri),
                                            ("current", destination not in ci)) if absent]})
            continue
        if source not in reference["land_regions"]:
            raise ValueError(f"Documented imported land is not reference land: {source}")
        source_land_ids.append(ri[source])
        old_town = early.get(destination, {}).get("town")
        shifted_town = [old_town[0] + dq, old_town[1] + dr] if old_town else None
        same_town = shifted_town is not None and list(row["town"]) == shifted_town
        if same_town:
            unchanged_towns.append(destination)
        land_rows.append({
            "source_region": source,
            "current_region": destination,
            "current_name": row.get("name"),
            "provenance_flag_in_design": row.get("imported"),
            "province_key": row.get("province_key"),
            **compare_masks(reference_ids == ri[source], current_ids == ci[destination], offset),
            "current_town": list(row["town"]),
            "early_plan_town_unshifted": old_town,
            "early_plan_town_in_current_frame": shifted_town,
            "unchanged_from_early_project_plan": same_town,
        })

    sea_rows = []
    for sea in design.get("imported_sea_regions", []):
        sources, unresolved = merged_sea_sources(
            sea, design.get("imported_region_map", {}), reference["sea_regions"])
        if not sources:
            continue
        destination = sea["region"]
        if destination not in ci:
            missing.append({"current_region": destination, "missing_in": ["current"]})
            continue
        sea_rows.append({
            "current_region": destination,
            "reference_source_regions": sources,
            "source_count": len(sources),
            "additional_consolidated_sources_not_in_reference": unresolved,
            **compare_masks(np.isin(reference_ids, [ri[name] for name in sources]),
                            current_ids == ci[destination], offset),
        })

    road = compare_roads(
        map_layers.get(reference["cells"], "road"),
        map_layers.get(vanilla["cells"], "road"),
        map_layers.get(current["cells"], "road"),
        np.isin(reference_ids, source_land_ids),
        offset,
    )
    return {
        "diagnostic_only": True,
        "changes_input_content": False,
        "grants_license": False,
        "warnings": [
            "Similarity metrics are diagnostics, not a legal conclusion about authorization or copyright.",
            "Non-identical geometry does not establish independent authorship.",
            "Matching geography can arise from shared facts; provenance and independent inputs need separate review.",
            "Town comparisons use the project's early imported plan, not an inferred original-mod town centroid.",
            "Road comparison excludes every position where reference and vanilla already share the same road mask.",
            "No provenance labels are deleted or rewritten by this audit.",
        ],
        "coordinate_frame": {
            "units": "map.hex hex columns q and south-up rows r",
            "reference_to_current_offset_qr": list(offset),
            "vanilla_and_reference_frame": "unshifted source map",
            "early_plan_frame_assumption": "towns in the project's early unshifted authoring map",
            "reference_grid_wh": list(reference_ids.shape[::-1]),
            "current_grid_wh": list(current_ids.shape[::-1]),
        },
        "summary": {
            "historical_imported_region_mapping_count": len(design.get("imported_region_map", {})),
            "active_imported_land_flags": len(imported),
            "compared_land_regions": len(land_rows),
            "land_complete_mask_equal": sum(row["complete_mask_equal"] for row in land_rows),
            "unchanged_from_early_plan_town_count": len(unchanged_towns),
            "compared_consolidated_sea_regions": len(sea_rows),
            "sea_complete_mask_equal": sum(row["complete_mask_equal"] for row in sea_rows),
            "missing_comparison_inputs": len(missing),
        },
        "land_regions": land_rows,
        "unchanged_from_early_plan_region_ids": unchanged_towns,
        "consolidated_sea_regions": sea_rows,
        "roads": road,
        "missing_comparison_inputs": missing,
    }


def file_digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-map", type=Path, default=ROOT / "work/all-directions-30/work/final-map/map.hex")
    parser.add_argument("--reference-map", type=Path, required=True,
                        help="the reference map.hex to measure against (no default: bring your own copy)")
    parser.add_argument("--vanilla-map", type=Path,
                        default=ROOT / "tools/caime-1.1.0/lib/app/Templates/3k_dlc07_main_map/map.hex")
    parser.add_argument("--design", type=Path, default=ROOT / "work/all-directions-30/work/current-region-design.json")
    parser.add_argument("--early-plan", type=Path, default=ROOT / "analysis/standalone-prototype.json")
    parser.add_argument("--offset", type=int, nargs=2, metavar=("DQ", "DR"),
                        help="Reference to current offset; otherwise use design padding[0:2]")
    parser.add_argument("--out", type=Path, required=True, help="Only output written by this tool")
    parser.add_argument("--provenance", type=Path,
                        help="provenance-history.json (design_release_form.py): the import-era fields of one entry "
                             "(--provenance-entry) are put back into the design before the audit")
    parser.add_argument("--provenance-entry", help="history entry label")
    args = parser.parse_args(argv)
    inputs = {
        "current_map": args.current_map.resolve(),
        "reference_map": args.reference_map.resolve(),
        "vanilla_map": args.vanilla_map.resolve(),
        "current_design": args.design.resolve(),
        "early_project_plan": args.early_plan.resolve(),
    }
    out = args.out.resolve()
    if out in inputs.values():
        parser.error("--out cannot overwrite an input")
    before = {key: file_digest(path) for key, path in inputs.items()}
    design = json.loads(inputs["current_design"].read_text(encoding="utf-8-sig"))
    if args.provenance:
        # the active design holds no import-era fields (design_release_form.py); the private history keeps them
        import design_release_form
        history = json.loads(args.provenance.read_text(encoding="utf-8"))["entries"]
        design = design_release_form.restore(design, history[args.provenance_entry])
    early_plan = json.loads(inputs["early_project_plan"].read_text(encoding="utf-8-sig"))
    offset = args.offset if args.offset is not None else design.get("padding", [])[:2]
    if len(offset) != 2:
        parser.error("Need --offset DQ DR or current design padding")
    report = audit_maps(
        map_hex.read(inputs["reference_map"]),
        map_hex.read(inputs["current_map"]),
        map_hex.read(inputs["vanilla_map"]),
        design, early_plan, offset,
    )
    after = {key: file_digest(path) for key, path in inputs.items()}
    if after != before:
        raise SystemExit("An input changed during the audit; no report was written")
    report["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["tool"] = "scripts/audit_imported_map_residuals.py"
    report["inputs"] = {key: {"path": str(path), "sha256": before[key]}
                        for key, path in inputs.items()}
    report["input_hashes_unchanged_during_audit"] = True
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(out), **report["summary"], "roads": report["roads"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
