"""Export vanilla-derived Assembly Kit XML to schema-matched RPFM TSV.

The build-input pack needs the start_pos source tables; the release pack will
keep only runtime table changes and the newly generated ESFs.
"""
from __future__ import annotations
import tw3k_paths
import csv
import argparse
import json
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from analyze_reference import ROOT
from sdk_empty_tables import from_official_export

KIT = ROOT / "work/standalone_kit"
SCHEMA = tw3k_paths.RPFM_SCHEMA
OUT = ROOT / "work/build-input"
import buddhist_temple_db
RUNTIME_TABLES = {"regions", "campaign_map_regions", "campaign_map_settlements", "region_to_province_junctions", "campaign_settlement_display_settlement_rotations", "campaign_settlement_display_settlement_layouts", "campaigns", "provinces", "season_province_effect_bundles", "campaign_camera_map_bounds", "campaign_map_areas_of_interest_label_positions", "campaign_areas_of_interest", "campaign_map_areas_of_interest", "region_effect_bundles", "cai_region_groups", "cai_region_groups_to_regions_junctions", "campaign_region_additional_adjacency_rules"} | (set(buddhist_temple_db.RUNTIME_TABLES) if buddhist_temple_db.ENABLED else set())   # temple copy off (stop-loss 2026-10-26): its tables are not shipped
TOOL_ONLY_TABLES = {"start_pos_family_relationship_types"}


def definitions():
    text = SCHEMA.read_text(encoding="utf-8")
    tables = list(re.finditer(r'^        "([^"]+)": \[', text, re.M))
    result = {}
    for i, match in enumerate(tables):
        block = text[match.end():tables[i + 1].start() if i + 1 < len(tables) else len(text)]
        versions = list(re.finditer(r'^                version: (-?\d+),', block, re.M))
        if not versions:
            continue
        best = max(range(len(versions)), key=lambda j: int(versions[j].group(1)))
        version = int(versions[best].group(1))
        definition = block[versions[best].end():versions[best + 1].start() if best + 1 < len(versions) else len(block)]
        names = list(re.finditer(r'^                        name: ("(?:[^"\\]|\\.)*"),', definition, re.M))
        fields = []
        for j, name in enumerate(names):
            field = definition[name.end():names[j + 1].start() if j + 1 < len(names) else len(definition)]
            kind = re.search(r'^                        field_type: ([^,\n]+)', field, re.M).group(1)
            default = re.search(r'^                        default_value: Some\(("(?:[^"\\]|\\.)*")\)', field, re.M)
            colour = re.search(r'^                        is_part_of_colour: Some\((\d+)\)', field, re.M)
            fields.append(dict(name=json.loads(name.group(1)), kind=kind,
                               default=json.loads(default.group(1)) if default else None,
                               colour=int(colour.group(1)) if colour else None))
        result[match.group(1)] = dict(version=version, fields=fields)
    return result


def value(row, field):
    cell = row.find(field["name"])
    if cell is None:
        cell = next((c for c in row if c.tag.casefold() == field["name"].casefold()), None)
    # An existing empty nullable XML field means null, not the schema's default
    # string "0". In particular, unowned regions must not reference faction 0.
    if cell is not None and cell.text is None and "String" in field["kind"]:
        raw = ""
    else:
        raw = cell.text if cell is not None and cell.text is not None else field["default"]
    if field["kind"] == "Boolean":
        return "true" if str(raw).lower() in ("true", "1") else "false"
    if raw is None:
        return "" if "String" in field["kind"] else "[]" if "Sequence" in field["kind"] else "0"
    return str(raw)


def main():
    global KIT, OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--vanilla-control", action="store_true")
    parser.add_argument("--campaign-split", action="store_true")
    # Resolve this inside main: all_directions_build redirects ROOT to its
    # isolated workspace before invoking this exporter.
    parser.add_argument("--official-empty-db", type=Path, default=ROOT / "work/native-db",
                        help="Directory of official SDK zero-row DB exports; missing formats fail closed")
    args = parser.parse_args()
    if args.vanilla_control:
        KIT = tw3k_paths.ASSEMBLY_KIT
        OUT = ROOT / "work/control-input"
    schema = definitions()
    selection = {}
    if args.campaign_split:
        for source in (KIT / "raw_data/db").glob("start_pos_*.xml"):
            table = source.stem
            records = ET.parse(source).getroot().findall(table)
            selection[table] = [r for r in records if r.find("campaign") is None or r.findtext("campaign") == "3k_main_campaign_map"]
        constraints = {}
        for table in selection:
            doc = ET.parse(KIT / "raw_data/db" / ("TWaD_" + table + ".xml")).getroot()
            constraints[table] = [(f.findtext("name"), f.findtext("column_source_table"), f.findtext("column_source_column"))
                                  for f in doc.findall("field") if f.findtext("column_source_table", "").startswith("start_pos_")]
        changed = True
        while changed:
            changed = False
            # Diplomacy parent rows have no campaign field. Their campaign is
            # derived from a concrete simple/complex deal and its participants.
            # Keep only parents represented by this campaign's concrete deals;
            # otherwise the game sees null deal objects in the ordering table.
            concrete_deals = {
                r.findtext("id", "")
                for name in ("start_pos_diplomacy_simple_deals", "start_pos_diplomacy_complex_deals")
                for r in selection.get(name, [])
            }
            parents = selection.get("start_pos_diplomacy_deals", [])
            active_parents = [r for r in parents if r.findtext("id", "") in concrete_deals]
            if len(active_parents) != len(parents):
                selection["start_pos_diplomacy_deals"] = active_parents
                changed = True
            for table, records in selection.items():
                keep = records
                for field, target, column in constraints[table]:
                    if target not in selection or target == table:
                        continue
                    valid = {r.findtext(column, "") for r in selection[target]}
                    keep = [r for r in keep if r.findtext(field, "") in ("", "0") or r.findtext(field, "") in valid]
                if len(keep) != len(records):
                    selection[table] = keep
                    changed = True
    report = {}
    prototype = json.loads((ROOT / "analysis/standalone-prototype.json").read_text(encoding="utf-8"))
    changes = {} if args.vanilla_control else prototype["changed_rows"]
    removed = {} if args.vanilla_control else prototype.get("removed_rows", {})
    # Validate every missing-schema input before writing any export. These are
    # internal, empty startpos containers that RPFM does not describe; the
    # official converter supplies their format, not a workshop/reference mod.
    empty_containers = {}
    for path in sorted((KIT / "raw_data/db").glob("*.xml")):
        table = path.stem
        if (not table.startswith("start_pos_") and table not in RUNTIME_TABLES
                or table in TOOL_ONLY_TABLES or table + "_tables" in schema):
            continue
        records = ET.parse(path).getroot().findall(table)
        if args.campaign_split:
            records = selection.get(table, records)
        if records:
            raise ValueError(f"No schema for nonempty table {table}")
        empty_containers[table] = from_official_export(table, args.official_empty_db)
    # Temple copy off (buddhist_temple_db.ENABLED, stop-loss 2026-10-26): its tables are no runtime tables, so the
    # loop below skips them; retire the fragments an earlier export generated for them (release and build input).
    if not buddhist_temple_db.ENABLED:
        for table in sorted(set(buddhist_temple_db.RUNTIME_TABLES) - RUNTIME_TABLES):
            for base in (ROOT / "work/release-input", OUT):
                frag = base / "db" / (table + "_tables") / "!tw3k_exp.tsv"
                if frag.exists():
                    frag.unlink()
            # and its native split copies (the SDK/compiler inputs, audit_map_asset_coverage reads the XML)
            split_xml = ROOT / "work/native-split-xml" / (table + ".xml")
            if split_xml.exists():
                split_xml.unlink()
            split_db = ROOT / "work/native-split-db" / (table + "_tables")
            if split_db.exists():
                shutil.rmtree(split_db)
    for path in sorted((KIT / "raw_data/db").glob("*.xml")):
        table = path.stem
        if not table.startswith("start_pos_") and table not in RUNTIME_TABLES:
            continue
        records = ET.parse(path).getroot().findall(table)
        if args.campaign_split:
            records = selection.get(table, records)
            if table == "campaigns":
                records = [r for r in records if r.findtext("campaign_name") == "3k_main_campaign_map"]
            # Preserve the XML schema, UUIDs and nullable cells for the official
            # SDK converter; do not round-trip these compiler inputs via TSV.
            root = ET.parse(path).getroot()
            for old in list(root.findall(table)):
                root.remove(old)
            for selected in records:
                root.append(selected)
            split_target = ROOT / "work/native-split-xml" / path.name
            split_target.parent.mkdir(parents=True, exist_ok=True)
            ET.ElementTree(root).write(split_target, encoding="utf-8", xml_declaration=True)
        if table in TOOL_ONLY_TABLES:
            report[table] = dict(rows=len(records), omitted="Assembly Kit enum-definition table; no runtime pack table")
            continue
        definition = schema.get(table + "_tables")
        if definition is None:
            header, empty_report = empty_containers[table]
            target = OUT / "db" / (table + "_tables") / "!tw3k_exp"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(header)
            report[table] = empty_report
            continue
        processed = []
        seen_colours = set()
        for field in definition["fields"]:
            if field["colour"] is not None:
                group = field["colour"]
                if group in seen_colours:
                    continue
                seen_colours.add(group)
                group_fields = [f for f in definition["fields"] if f["colour"] == group]
                processed.append(("unnamed colour group_" + str(group), group_fields))
            else:
                processed.append((field["name"], [field]))
        targets = [(OUT, records, "!tw3k_exp")]
        # Release table fragments contain only this project's runtime rows.
        if table in RUNTIME_TABLES and table in changes:
            own_rows = []
            for change in changes[table]:
                row = ET.Element(table)
                for key, raw in change.items():
                    ET.SubElement(row, key).text = raw
                own_rows.append(row)
            targets.append((ROOT / "work/release-input", own_rows, "!tw3k_exp"))
            # Rows removed from a vanilla table (retired regions) cannot be expressed by an additive
            # fragment: ship the vanilla table under its own name (data__), minus the removed rows.
            if removed.get(table):
                own = {tuple(sorted(c.items())) for c in changes[table]}
                vanilla_rows = [r for r in records if tuple(sorted((c.tag, c.text) for c in r)) not in own]
                targets.append((ROOT / "work/release-input", vanilla_rows, "data__"))
        stale = ROOT / "work/release-input" / "db" / (table + "_tables") / "data__.tsv"
        if not removed.get(table) and stale.exists():
            stale.unlink()                                   # no retired rows any more: vanilla table again
        for base, selected, name in targets:
            target = base / "db" / (table + "_tables") / (name + ".tsv")
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
                writer.writerow([name for name, _ in processed])
                writer.writerow([f"#{table}_tables;{definition['version']};db/{table}_tables/{name}"] + [""] * (len(processed) - 1))
                for row in selected:
                    values = []
                    for name, fields in processed:
                        if name.startswith("unnamed colour group_"):
                            if len(fields) != 3:
                                raise ValueError("Unexpected RGB group")
                            values.append("".join(f"{int(value(row, f)):02X}" for f in fields))
                        else:
                            values.append(value(row, fields[0]))
                    writer.writerow(values)
        report[table] = dict(rows=len(records), version=definition["version"], columns=[name for name, _ in processed])
    for base in (() if args.vanilla_control else (OUT, ROOT / "work/release-input")):
        loc = base / "text/db/tw3k_exp.loc.tsv"
        loc.parent.mkdir(parents=True, exist_ok=True)
        design_path = ROOT / "work/current-region-design.json"
        design = json.loads(design_path.read_text(encoding="utf-8")) if design_path.exists() else {}
        counties = design.get("regions", [dict(region=design.get("region", "tw3k_exp_wuwei_tools"),first_id=design.get("first_id", 1900000001),name=design.get("name", "武威工具作坊"))])
        with loc.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
            writer.writerow(["key", "text", "tooltip"])
            writer.writerow(["#Loc;1;text/db/tw3k_exp.loc", "", ""])
            for county in counties:
                key=county['region'];region_id=county['first_id']
                for entry in ("regions_onscreen_" + key, "regions_battle_name_" + key,
                              "start_pos_settlements_onscreen_name_settlement:" + key + str(region_id)):
                    writer.writerow([entry, county['name'], "false"])
            if 'regions' in design:
                for group in design.get('provinces',[design]):
                    if group['province'].get('emit_localization') is False:
                        continue
                    writer.writerow(['provinces_onscreen_'+group['province']['key'],group['province']['name'],'false'])
            original_region_ids={r.findtext('region'):r.findtext('id') for r in ET.parse(KIT/'raw_data/db/start_pos_regions.xml').getroot().findall('start_pos_regions') if r.findtext('campaign')=='3k_main_campaign_map'}
            for change in design.get('vanilla_administrative_overrides',[]):
                if not change.get('name'):continue
                key=change['region'];rid=original_region_ids[key]
                for entry in ('regions_onscreen_'+key,'regions_battle_name_'+key,'start_pos_settlements_onscreen_name_settlement:'+key+rid):
                    writer.writerow([entry,change['name'],'false'])
            # Landmarks of the new regions: area-of-interest titles and tooltips (landmarks_db.py).
            import landmarks_db
            for entry in landmarks_db.loc_rows():
                writer.writerow(list(entry))
            # Jibin's Buddhist monastery (buddhist_temple_db.py): its chain, level and text names.
            for entry in buddhist_temple_db.loc_rows():
                writer.writerow(list(entry))
    objectives = KIT / "raw_data/EmpireDesignData/campaigns/3k_main_campaign_map/victory_objectives.txt"
    (OUT / "db").mkdir(exist_ok=True)
    shutil.copyfile(objectives, OUT / "db/victory_objectives.txt")
    filename = "control-input-table-export.json" if args.vanilla_control else "build-input-table-export.json"
    (ROOT / "analysis" / filename).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Exported {sum('version' in info for info in report.values())} nonempty schema-matched tables")


if __name__ == "__main__":
    main()
