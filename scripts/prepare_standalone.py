"""Build the first standalone region's XML from vanilla records only.

Map creation is performed by Tw3kCreateRegion first. No 190 database, faction,
character, CEO or gameplay rows are imported here.
"""
from __future__ import annotations
import tw3k_paths
import copy
import json
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from analyze_reference import ROOT, read_hex

SOURCE = tw3k_paths.ASSEMBLY_KIT
KIT = ROOT / "work/standalone_kit"
REGION = "tw3k_exp_wuwei_tools"
CAMPAIGN = "3k_main_campaign_map"
SETTLEMENT = "settlement:" + REGION
DONOR = "3k_main_anding_resource_2"
TREES = {}
CHANGES = {}
IDS = {"region": 1900000001, "settlement": 1900000002, "primary_slot": 1900000003, "secondary_slot": 1900000004}
# The early 0.3 single-region design (repository-root work/current-region-design.json) is no longer kept; the
# constants below are its defaults. The current build uses this module as a kit library only (prepare_province).
DESIGN = {}
NAME = DESIGN.get("name", "武威工具作坊")
PROVINCE = DESIGN.get("province", "3k_main_province_wuwei")
OWNER_REGION = DESIGN.get("owner_region", "3k_main_wuwei_capital")
SECONDARY_DONOR = DESIGN.get("secondary_donor", "3k_main_wuwei_resource_1")
if DESIGN and "regions" not in DESIGN:
    REGION = DESIGN["region"]
    SETTLEMENT = "settlement:" + REGION
    DONOR = DESIGN["donor"]
    IDS = dict(zip(IDS, range(DESIGN["first_id"], DESIGN["first_id"] + 4)))


def tree(table):
    if table not in TREES:
        TREES[table] = ET.parse(SOURCE / "raw_data/db" / f"{table}.xml")
    return TREES[table]


def rows(table):
    return tree(table).getroot().findall(table)


def find(table, **values):
    return next(r for r in rows(table) if all(r.findtext(k) == v for k, v in values.items()))


def set_value(row, key, value):
    cell = row.find(key)
    if cell is None:
        cell = ET.SubElement(row, key)
    cell.text = str(value)


def add(table, donor, **values):
    row = copy.deepcopy(donor)
    for key, value in values.items():
        set_value(row, key, value)
    schema = ET.parse(SOURCE / "raw_data/db" / f"TWaD_{table}.xml").getroot()
    keys = [r.findtext("name") for r in schema.findall("field") if r.findtext("primary_key") == "1"]
    row.set("record_key", "".join(row.findtext(k, "") for k in keys))
    identity=[(k,row.findtext(k,"")) for k in keys]
    row.set("record_uuid", "{"+str(uuid.uuid5(uuid.NAMESPACE_URL,"tw3k-authored/"+table+"/"+json.dumps(identity,ensure_ascii=False)))+"}")
    tree(table).getroot().append(row)
    CHANGES.setdefault(table, []).append({child.tag: child.text for child in row})
    return row


def main():
    if not KIT.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("Prototype must stay inside this workspace")
    map_path = KIT / "raw_data/EmpireDesignData/campaign_maps/3k_dlc07_main_map/map.hex"
    info = read_hex(map_path)
    if len(info["land_regions"]) != 203 or len(info["sea_regions"]) != 77 or REGION not in info["land_regions"]:
        raise ValueError("Create the standalone map prototype first")
    if any("ironic" in key for key in info["land_regions"] + info["sea_regions"]):
        raise ValueError("Map contains reference mod region keys")
    used_ids = {r.findtext("id") for table in ("start_pos_regions", "start_pos_settlements", "start_pos_region_slot_templates") for r in rows(table)}
    if any(str(value) in used_ids for value in IDS.values()):
        raise ValueError("Prototype IDs collide with vanilla")
    colour = (37, 196, 129)
    if colour in {(int(r.findtext("r")), int(r.findtext("g")), int(r.findtext("b"))) for r in rows("regions")}:
        raise ValueError("Prototype lookup colour collides with vanilla")
    add("regions", find("regions", key=DONOR), key=REGION, r=colour[0], g=colour[1], b=colour[2], is_sea=0,
        onscreen=NAME, battle_name=NAME)
    add("campaign_map_regions", find("campaign_map_regions", campaign_map=info["campaign_map"], region=DONOR),
        campaign_map=info["campaign_map"], region=REGION)
    add("region_to_province_junctions", find("region_to_province_junctions", region=SECONDARY_DONOR),
        region=REGION, province=PROVINCE, is_capital=0)
    add("campaign_map_settlements", find("campaign_map_settlements", settlement_id="settlement:" + DONOR),
        settlement_id=SETTLEMENT)
    add("campaign_settlement_display_settlement_rotations", find("campaign_settlement_display_settlement_rotations", settlement="settlement:" + DONOR),
        settlement=SETTLEMENT)
    donor_region = find("start_pos_regions", campaign=CAMPAIGN, region=DONOR)
    donor_id = donor_region.findtext("id")
    owner = find("start_pos_regions", campaign=CAMPAIGN, region=OWNER_REGION).findtext("owning_faction")
    add("start_pos_regions", donor_region, region=REGION, campaign=CAMPAIGN, id=IDS["region"],
        owning_faction=owner, faction_capital=0, slot_cap=2)
    add("start_pos_settlements", find("start_pos_settlements", region=donor_id, settlement_id="settlement:" + DONOR),
        settlement_id=SETTLEMENT, region=IDS["region"], id=IDS["settlement"], onscreen_name=NAME)
    add("start_pos_region_slot_templates", find("start_pos_region_slot_templates", campaign=CAMPAIGN, region=DONOR, slot_type="primary"),
        campaign=CAMPAIGN, region=REGION, id=IDS["primary_slot"])
    add("start_pos_region_slot_templates", find("start_pos_region_slot_templates", campaign=CAMPAIGN, region=SECONDARY_DONOR, slot_type="secondary"),
        campaign=CAMPAIGN, region=REGION, id=IDS["secondary_slot"])
    for table in ("start_pos_region_pooled_resources", "start_pos_region_religions"):
        donors = [r for r in rows(table) if r.findtext("region") == donor_id]
        for donor in donors:
            add(table, donor, region=IDS["region"])
    campaign_row = find("campaigns", campaign_name=CAMPAIGN)
    set_value(campaign_row, "map_name", info["campaign_map"])
    set_value(campaign_row, "display_location", info["campaign_map"])
    CHANGES["campaigns"] = [{child.tag: child.text for child in campaign_row}]
    # Match the map's region IDs, including the shifted sea-region IDs.
    order = {key: i for i, key in enumerate(info["land_regions"] + info["sea_regions"])}
    root = tree("regions").getroot()
    records = list(root.findall("regions"))
    for row in records:
        root.remove(row)
    for row in sorted(records, key=lambda r: (order.get(r.findtext("key"), len(order)), r.findtext("key"))):
        root.append(row)
    root = tree("campaigns").getroot()
    records = list(root.findall("campaigns"))
    for row in records:
        root.remove(row)
    for row in sorted(records, key=lambda r: r.findtext("campaign_name") != CAMPAIGN):
        root.append(row)
    for table, document in TREES.items():
        if table not in CHANGES:
            continue
        ET.indent(document, space="  ")
        document.write(KIT / "raw_data/db" / f"{table}.xml", encoding="utf-8", xml_declaration=True)
    report = dict(region=REGION, requires_other_mods=False, game_base="vanilla", campaign=CAMPAIGN,
                  status="authoring_and_map_compilation; playable_pack_not_yet_created",
                  reference_reuse=DESIGN.get("reference_reuse", "2,087 hexes of the wuwei_resource_3 layout; existing vanilla regions preserved"),
                  name=NAME, province=PROVINCE, donor=DONOR, town=DESIGN.get("town", [179,603]),
                  independent_region_ids=IDS, initial_owner_vanilla_id=owner,
                  geometry=info["region_geometry"][REGION], changed_rows=CHANGES,
                  land_regions=len(info["land_regions"]), sea_regions=len(info["sea_regions"]),
                  remaining_steps=["generate startpos_romance.esf and startpos_historical.esf", "process lookup into game formats", "verify battle tile assignment", "in-game campaign validation"])
    # The 0.3 standalone report; the workspace build report analysis/standalone-prototype.json has one writer
    # (all_directions_build.database).
    (ROOT / "analysis/standalone-0.3-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "changed_rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
