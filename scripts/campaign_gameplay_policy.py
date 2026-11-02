"""Native-compatible authoring policy for campaign data; importing never writes."""
from __future__ import annotations
import copy,json,uuid
from collections import Counter,defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET

CAMPAIGN = "3k_main_campaign_map"
ROLES = {
 "3k_city":"邑落", "3k_dlc06_city_nanman":"邑落", "3k_resource_fire_northern_horses":"马场",
 "3k_resource_fire_iron":"矿场", "3k_resource_wood_lumber_pine":"林场",
 "3k_resource_wood_farms_grain":"粮田", "3k_resource_wood_farms_rice":"稻田",
 "3k_resource_wood_livestock":"牧场", "3k_resource_water_salt":"盐场",
 "3k_resource_metal_tools":"工具作坊", "3k_resource_wood_fish":"渔港",
 "3k_resource_water_trading_port":"商港",
 # Donor types introduced by design-changeset-2026-10-18.
 "3k_resource_metal_jade":"玉矿", "3k_resource_metal_copper":"铜矿",
 "3k_resource_wood_lumber_bamboo":"竹林", "3k_resource_water_spice":"香料市",
 # Reviewed historical resources 2026-10-25 (region_gameplay_db.RESOURCE_OVERRIDES).
 "3k_resource_earth_temple":"佛寺", "3k_resource_water_silk":"蚕桑"
}
# 牧地 (grazing land, the relayout_western_regions 'pasture' suffix) means livestock (牧场);
# a region with a horse donor is renamed ·马场 (the buildings differ: cavalry costs vs farming).
ALIASES = {"邑落":{"邑落"}, "马场":{"马场"}, "矿场":{"矿场","铁矿"},
 "林场":{"林场"}, "粮田":{"粮田","农田","屯田","田地","农场"},
 "稻田":{"稻田","农田","屯田","田地","农场"}, "牧场":{"牧场","牧地"},
 "盐场":{"盐场"}, "工具作坊":{"工具作坊"}, "渔港":{"渔港","渔场"},
 "商港":{"商港","港口"},
 "玉矿":{"玉矿"}, "铜矿":{"铜矿"}, "竹林":{"竹林"}, "香料市":{"香料市","香市"},
 "佛寺":{"佛寺","寺院"}, "蚕桑":{"蚕桑","丝坊"}}
RESOURCE_SUFFIXES=set().union(*ALIASES.values())
# Explicit current geography, not retired Hainan IDs that now denote western tribes.
TROPICAL_DONORS = {
 "tw3k_hist_province_rinan":"3k_main_province_jiaozhi",
 "tw3k_hist_province_linyi":"3k_main_province_jiaozhi",
 "tw3k_hist_province_funan":"3k_main_province_jiaozhi",
 "tw3k_hist_province_shan_country":"3k_main_province_cangwu"
}
PROFILE_FIELDS=("base_population","base_max_population","population","base_gdp",
 "town_wealth","rebel_faction","alternative_rebel_faction","cultural_originator",
 "settler_rebellions_enabled","base_fertility")

def native_catalogue(directory):
 directory=Path(directory)
 def rows(table):return ET.parse(directory/(table+".xml")).getroot().findall(table)
 return {"directory":directory,
  "regions":{r.findtext("region"):r for r in rows("start_pos_regions") if r.findtext("campaign")==CAMPAIGN},
  "slots":{r.findtext("region"):r.findtext("slot_template") for r in rows("start_pos_region_slot_templates") if r.findtext("campaign")==CAMPAIGN and r.findtext("slot_type")=="primary"},
  "provinces":{r.findtext("key"):r for r in rows("provinces")},
  "seasons":rows("season_province_effect_bundles"),
  "cultures":{r.findtext("key") for r in rows("cultures")},
  "factions":{r.findtext("key") for r in rows("factions")}}

def normalize_design(plan,native):
 plan=copy.deepcopy(plan);changes=[];regions=plan["regions"]
 before={(r["region"]):list(r["town"]) for r in regions}
 for r in regions:
  # The primary chain (role) is the building donor's (region_gameplay_db.assign_settlements),
  # the design donor's when there is none.
  donor=r["donor"];slot_donor=r.get("building_donor",donor)
  assert donor in native["regions"] and slot_donor in native["slots"],(donor,slot_donor)
  role=ROLES[native["slots"][slot_donor]]
  name=r["name"]
  if "·" in name:
   base,suffix=name.rsplit("·",1)
   if suffix in RESOURCE_SUFFIXES and suffix not in ALIASES[role]:
    r["name"]=base+"·"+role
    changes.append({"region":r["region"],"before":name,"after":r["name"],"primary_slot":native["slots"][slot_donor]})
  r["gameplay_template"]={"donor":donor,"building_donor":slot_donor,"source":"vanilla 190 start_pos_regions",
   "primary_slot":native["slots"][slot_donor],"resource_label":role,
   "population_is_historical":r.get("population_pool") is not None,"culture_is_historical":False}
 duplicates=defaultdict(list)
 for r in regions:duplicates[(r["province_key"],r["name"])].append(r)
 for (_,name),members in duplicates.items():
  if len(members)<2:continue
  for r in members:
   old=r["name"];role=ROLES[native["slots"][r.get("building_donor",r["donor"])]]
   r["name"]=old.split("·",1)[0]+"·"+role
   if r["name"]!=old:changes.append({"region":r["region"],"before":old,"after":r["name"],"reason":"same-commandery duplicate, native primary slot"})
 check=Counter((r["province_key"],r["name"]) for r in regions)
 assert max(check.values(),default=0)==1,"Duplicate resource roles require explicit location review"
 season_changes=[]
 for group in plan["provinces"]:
  p=group["province"]
  if p["key"] in native["provinces"]:
   # Sharing an existing commandery does not authorize renaming it.
   p["emit_localization"]=False
  elif p["key"] in TROPICAL_DONORS:
   wanted=TROPICAL_DONORS[p["key"]]
   assert wanted in native["provinces"]
   if p.get("season_donor",p["donor"])!=wanted:
    season_changes.append({"province":p["key"],"before":p.get("season_donor",p["donor"]),"after":wanted})
   p["season_donor"]=wanted
  # provinces[].regions hold region keys only (design_single_source, 2026-11-01): the records are the top level
 plan["gameplay_data_policy"]={
  "population":"start_pos_regions profile fields keep native 190 defaults (all 201 vanilla regions also have 100000). The starting population pool comes from the donor unless a region sets population_pool, which is a tiered estimate from Han-period records relative to vanilla frontier commanderies, not a census.",
  "culture":"Use validated culture IDs from the existing 190 donor. Foreign-polity culture authoring remains pending.",
  "rebels":"Retain the donor's existing compatible faction IDs; not a historical statement about foreign rebellions.",
  "season":"Tropical authored groups inherit vanilla Jiaozhi/Cangwu; existing vanilla commanderies retain their native seasons.",
  "pending":["Historical population estimates","Dedicated Korean/Southeast Asian culture content","Historically appropriate foreign rebel factions"]}
 assert before=={r["region"]:r["town"] for r in regions},"Gameplay policy must not move towns"
 return plan,{"name_changes":changes,"season_donor_changes":season_changes}

def set_cell(row,key,value):
 cell=row.find(key)
 if cell is None:cell=ET.SubElement(row,key)
 cell.text=value

def row_uuid(table,row,native):
 schema=ET.parse(native["directory"]/("TWaD_"+table+".xml")).getroot()
 keys=[r.findtext("name") for r in schema.findall("field") if r.findtext("primary_key")=="1"]
 identity=[(k,row.findtext(k,"")) for k in keys]
 row.set("record_key","".join(v for k,v in identity))
 row.set("record_uuid","{"+str(uuid.uuid5(uuid.NAMESPACE_URL,
  "tw3k-authored/"+table+"/"+json.dumps(identity,ensure_ascii=False)))+"}")

def apply_xml_policy(plan,kit,native):
 kit=Path(kit);auth={r["region"]:r for r in plan["regions"]};changed={};summary={}
 for table in ("regions","start_pos_regions","start_pos_settlements","season_province_effect_bundles"):
  path=kit/"raw_data/db"/(table+".xml");doc=ET.parse(path);root=doc.getroot()
  if table=="season_province_effect_bundles":
   custom={g["province"]["key"]:g["province"] for g in plan["provinces"] if g["province"]["key"] not in native["provinces"]}
   for row in list(root.findall(table)):
    if row.findtext("province") in custom:root.remove(row)
   for key,p in sorted(custom.items()):
    donor=p.get("season_donor",p["donor"])
    source=[r for r in native["seasons"] if r.findtext("province")==donor]
    assert source,"No native season records for "+donor
    for original in source:
     row=copy.deepcopy(original);set_cell(row,"province",key);row_uuid(table,row,native);root.append(row)
   selected=[r for r in root.findall(table) if r.findtext("province") in custom]
   ids=[r.get("record_uuid") for r in root.findall(table)]
   assert len(set(ids))==len(ids),"Season record UUIDs must be unique"
   summary["season_records"]=len(selected);summary["season_uuids_unique"]=True
  else:
   selected=[]
   for row in root.findall(table):
    key=(row.findtext("key") if table=="regions" else row.findtext("settlement_id","").removeprefix("settlement:") if table=="start_pos_settlements" else row.findtext("region"))
    if key not in auth:continue
    r=auth[key]
    if table=="start_pos_regions":
     if row.findtext("campaign")!=CAMPAIGN:continue
     original=native["regions"][r["donor"]]
     for field in PROFILE_FIELDS:
      if original.find(field) is not None:set_cell(row,field,original.findtext(field))
     assert row.findtext("cultural_originator") in native["cultures"]
     assert row.findtext("rebel_faction") in native["factions"]
     assert row.findtext("alternative_rebel_faction") in native["factions"]
    elif table=="regions":
     for field in ("onscreen","battle_name"):set_cell(row,field,r["name"])
    else:set_cell(row,"onscreen_name",r["name"])
    selected.append(row)
  ET.indent(doc,space="  ");doc.write(path,encoding="utf-8",xml_declaration=True)
  changed[table]=[{c.tag:c.text for c in row} for row in selected]
 return changed,summary

