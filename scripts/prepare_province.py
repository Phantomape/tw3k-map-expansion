"""Author one independent commandery with four counties from vanilla records."""
import copy,json,shutil
import xml.etree.ElementTree as ET
from pathlib import Path
import prepare_standalone as base
from analyze_reference import ROOT,read_hex
from campaign_gameplay_policy import native_catalogue,normalize_design
import numpy as np,map_hex
import region_gameplay_db
def land_port_hexes(m):
 """Land port hexes (byte 2 high nibble 2) per land region of a map_hex.read map."""
 c=m['cells'];ids=map_hex.region_ids(c);n=len(m['land_regions'])
 counts=np.bincount(ids[((c[...,2]>>4)==2)&(ids>=0)&(ids<n)],minlength=n)
 return {k:int(counts[i]) for i,k in enumerate(m['land_regions']) if counts[i]}
LANDLOCKED_SEAS={'3k_main_sea_lake','3k_main_sea_non_playable'}
def landlocked_ports(m):
 """Land port hexes whose adjacent water reaches no navigable sea region (an inland lake):
 a port needs water connected to the sea. Searches each port's water until it meets a
 navigable sea hex, so a coastal port stops at once and a lake port walks only its lake."""
 import collections
 c=m['cells'];h,w=c.shape[:2];ids=map_hex.region_ids(c);names=list(m['land_regions'])+list(m['sea_regions'])
 water=(c[...,0]&3)==1;navigable=water&np.isin(ids,[i for i,k in enumerate(names) if i>=len(m['land_regions']) and k not in LANDLOCKED_SEAS])
 def nbrs(q,r):
  for a,b in ((0,-1),(0,1),(-1,0),(1,0))+(((-1,1),(1,1)) if q%2 else ((-1,-1),(1,-1))):
   if 0<=q+a<w and 0<=r+b<h:yield q+a,r+b
 out=[]
 for r,q in zip(*np.nonzero(((c[...,2]>>4)==2)&~water)):
  seen=set();todo=collections.deque((a,b) for a,b in nbrs(q,r) if water[b,a]);found=False
  while todo and not found:
   a,b=todo.popleft()
   if (a,b) in seen:continue
   seen.add((a,b));found=bool(navigable[b,a]);todo.extend(n for n in nbrs(a,b) if water[n[1],n[0]] and n not in seen)
  if not found:out.append((names[ids[r,q]],int(q),int(r)))
 return out
def county_slots(county,donor_slots,port):
 """Donor slot rows with the port slot following the map, as in vanilla (a 'port' template exactly
 when the region owns a land port hex): drop the donor's port row without a port hex; with one and
 no donor port row, add the vanilla row (3k_minor_port, or 3k_province_capital_port for a capital)."""
 slots=[r for r in donor_slots if port or r.findtext('slot_type')!='port']
 if port and not any(r.findtext('slot_type')=='port' for r in slots):
  slots.append(base.find('start_pos_region_slot_templates',campaign=base.CAMPAIGN,slot_type='port',slot_template='3k_province_capital_port' if county['is_capital'] else '3k_minor_port'))
 return sorted(slots,key=lambda r:r.findtext('slot_type'))
def normalized_design(plan,kit,ports=None,rows=None,native=None):
 """(design, settlements): the design with region_gameplay_db.assign_settlements (building donor, battle layout,
 primary building / slot) and campaign_gameplay_policy.normalize_design (names, gameplay templates, province season
 donors) applied, and provinces[].regions as bare keys (campaign_world.design_keys_only). A pure function of the design,
 the kit map (map_hex.read) and the vanilla tables: the DB step compares the design with it and the generator writes it."""
 import campaign_world as _cw
 p=copy.deepcopy(plan)
 settlements=region_gameplay_db.assign_settlements(p,land_port_hexes(kit) if ports is None else ports,rows or base.rows,region_gameplay_db.coastal_towns(kit,p))
 p,_=normalize_design(p,native or native_catalogue(base.SOURCE/'raw_data/db'))
 return _cw.design_keys_only(p),settlements
# Vanilla DB tables a source kit may hold edited (earlier standalone stages wrote them): always restored from the
# official kit first, with the tables the previous report lists.
RESTORE_TABLES=('provinces','season_province_effect_bundles','regions','campaign_map_regions','region_to_province_junctions',
                'campaign_map_settlements','campaign_settlement_display_settlement_rotations',
                'campaign_settlement_display_settlement_layouts','start_pos_regions','start_pos_settlements',
                'start_pos_region_slot_templates','start_pos_region_pooled_resources','start_pos_region_religions',
                'campaign_camera_map_bounds','campaigns')
def restore_tables(previous):
 """RESTORE_TABLES, then the tables of the previous report (read only; its one writer is all_directions_build.database)."""
 tables=list(RESTORE_TABLES)
 if previous.exists():tables+=[t for t in json.loads(previous.read_text(encoding='utf-8')).get('changed_rows',{}) if t not in tables]
 return tables
def main():
 """Region, settlement and start-position tables into the kit; returns the report (the caller writes it)."""
 plan=json.loads((ROOT/'work/current-region-design.json').read_text(encoding='utf-8'))
 # Restore previously authored tables so switching experiments cannot retain stale rows.
 for table in restore_tables(ROOT/'analysis/standalone-prototype.json'):
  source=base.SOURCE/'raw_data/db'/f'{table}.xml'
  if source.exists():shutil.copyfile(source,base.KIT/'raw_data/db'/source.name)
 import kit_map;kit_map.sync_from_final()   # the kit copy must match final-map before it is read
 kit_hex=base.KIT/'raw_data/EmpireDesignData/campaign_maps/3k_dlc07_main_map/map.hex'
 info=read_hex(kit_hex);kit=map_hex.read(kit_hex);ports=land_port_hexes(kit)
 if 'provinces' in plan:
  # Building donor and battle layout per region (region_gameplay_db tables), before the policy
  # reads the primary chain. The design has one writer (2026-11-01): this step only checks that the design already
  # holds these values (normalized_design); TW3K_NORMALIZE_DESIGN=1 (all_directions_build.py normalize-design, and
  # the rebuild generator through the same function) writes them once.
  want,settlements=normalized_design(plan,kit,ports)
  if want!=plan:
   import os
   if os.environ.get('TW3K_NORMALIZE_DESIGN')!='1':
    diff=sorted(r['region'] for r,w in zip(plan['regions'],want['regions']) if r!=w) if len(plan['regions'])==len(want['regions']) else ['(region list)']
    raise RuntimeError(f'design not normalized: {len(diff)} regions differ from normalized_design (e.g. {diff[:5]}); run python scripts/all_directions_build.py normalize-design, review and commit the design')
   (ROOT/'work/current-region-design.json').write_text(json.dumps(want,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
  plan=want
  (ROOT/'analysis/settlement-donors.json').write_text(json.dumps(settlements,ensure_ascii=False,indent=1,sort_keys=True)+'\n',encoding='utf-8')
 # No ports on inland lakes: a port hex must touch water that reaches the sea.
 if (landlocked:=landlocked_ports(kit)):raise RuntimeError(f'Port hexes on water with no sea connection (inland lakes): {landlocked}')
 import campaign_world as _cw,map_hex as _mh   # vanilla land-region count from the vanilla map, not a literal
 assert len(info['land_regions'])==len(_mh.read(_cw.VANILLA_MAP)['land_regions'])+len(plan['regions'])+len(plan.get('sdk_internal_land_regions',[]))
 groups=plan.get('provinces',[plan]);province=groups[0]['province']
 for group in groups:
  p=group['province']
  if not any(r.findtext('key')==p['key'] for r in base.rows('provinces')):
   base.add('provinces',base.find('provinces',key=p['donor']),key=p['key'],onscreen=p['name'])
   for donor in [r for r in base.rows('season_province_effect_bundles') if r.findtext('province')==p.get('season_donor',p['donor'])]:
    base.add('season_province_effect_bundles',donor,province=p['key'])
 faction_ids={r.findtext('faction'):r.findtext('ID') for r in base.rows('start_pos_factions') if r.findtext('campaign')==base.CAMPAIGN}
 # Rebel keys a new region may name, all in the 190 startpos: is_rebel factions, and the Yellow
 # Turban armies vanilla 190 regions already name as rebel_faction (is_rebel 0 there too).
 vanilla_rebels={r.findtext(f) for r in base.rows('start_pos_regions') if r.findtext('campaign')==base.CAMPAIGN for f in ('rebel_faction','alternative_rebel_faction')}
 is_rebel={r.findtext('key') for r in base.rows('factions') if r.findtext('is_rebel')=='1'}
 rebel_ids={k for k in faction_ids if k in vanilla_rebels or k in is_rebel}
 owner=faction_ids[province['owner_faction']]
 group_owners={g['province']['key']:faction_ids[g['province']['owner_faction']] for g in groups}
 used_ids={r.findtext('id') for table in ('start_pos_regions','start_pos_settlements','start_pos_region_slot_templates') for r in base.rows(table)}
 colours={(int(r.findtext('r')),int(r.findtext('g')),int(r.findtext('b'))) for r in base.rows('regions')}
 for county in plan['regions']:
  key=county['region'];donor=county['donor'];name=county['name'];settlement='settlement:'+key
  first=county['first_id'];ids=list(range(first,first+4))
  assert not any(str(value) in used_ids for value in ids), 'ID collision'
  colour=tuple(county['colour']);assert colour not in colours,'Lookup colour collision';colours.add(colour)
  assert info['region_geometry'][key]['hex_count']>=40   # vanilla passes go down to 60 (jiameng); small islands (Tsushima) need ~50
  base.add('regions',base.find('regions',key=donor),key=key,r=colour[0],g=colour[1],b=colour[2],is_sea=0,onscreen=name,battle_name=name)
  base.add('campaign_map_regions',base.find('campaign_map_regions',campaign_map=info['campaign_map'],region=donor),campaign_map=info['campaign_map'],region=key)
  base.add('region_to_province_junctions',base.find('region_to_province_junctions',region=donor),region=key,province=county.get('province_key',province['key']),is_capital=int(county['is_capital']))
  base.add('campaign_map_settlements',base.find('campaign_map_settlements',settlement_id='settlement:'+donor),settlement_id=settlement)
  base.add('campaign_settlement_display_settlement_rotations',base.find('campaign_settlement_display_settlement_rotations',settlement='settlement:'+donor),settlement=settlement)
  # A city (its gameplay donor has a layout) takes the layout of a vanilla city chosen by
  # province and geography, a port layout exactly when it has a port slot (region_gameplay_db).
  for layout in [r for r in base.rows('campaign_settlement_display_settlement_layouts') if r.findtext('settlement')=='settlement:'+donor]:
   shown=region_gameplay_db.layout_donor(county,ports.get(key,0)>0,base.rows)
   if shown is None:print('Layout: no class for province',county.get('province_key'),'- keeping',donor,'for',key,flush=True)
   else:layout=base.find('campaign_settlement_display_settlement_layouts',settlement='settlement:'+shown)
   base.add('campaign_settlement_display_settlement_layouts',layout,settlement=settlement)
  source_region=base.find('start_pos_regions',campaign=base.CAMPAIGN,region=donor);source_id=source_region.findtext('id')
  county_owner=faction_ids[county['owner_faction']] if county.get('owner_faction') else group_owners[county.get('province_key',province['key'])]
  # Buildings (settlement row, slot templates, slot_cap) come from the building donor
  # (region_gameplay_db.building_donor); the region profile stays the design donor's.
  building=county.get('building_donor',donor)
  building_region=base.find('start_pos_regions',campaign=base.CAMPAIGN,region=building)
  slots=[r for r in base.rows('start_pos_region_slot_templates') if r.findtext('campaign')==base.CAMPAIGN and r.findtext('region')==building]
  assert 1<=len(slots)<=8,'Unexpected vanilla slot count for '+building
  own_slots=county_slots(county,slots,ports.get(key,0))   # vanilla slot_cap counts the port slot too
  if county.get('primary_slot_template'):   # region_gameplay_db.PRIMARY_SLOT_OVERRIDES (Jibin's Buddhist temple copy)
   own_slots=[copy.deepcopy(r) for r in own_slots]
   for r in own_slots:
    if r.findtext('slot_type')=='primary':base.set_value(r,'slot_template',county['primary_slot_template'])
  rebel,alternative,_=region_gameplay_db.rebel_factions(county,base.rows)   # by class / vanilla province
  assert rebel in rebel_ids and alternative in rebel_ids,('rebel faction not in the 190 startpos',key,rebel,alternative)
  base.add('start_pos_regions',source_region,campaign=base.CAMPAIGN,region=key,id=ids[0],owning_faction=county_owner,faction_capital=0,slot_cap=int(building_region.findtext('slot_cap'))+len(own_slots)-len(slots),rebel_faction=rebel,alternative_rebel_faction=alternative)
  # A reviewed start building replaces the donor's primary building (region_gameplay_db: Jibin's
  # Buddhist monastery, RESOURCE_OVERRIDES; Dunxun's city level, CITY_START_OVERRIDES).
  start={'primary_building':county['primary_building']} if county.get('primary_building') else {}
  import buddhist_temple_db   # the mod's own levels count as existing
  if start:assert start['primary_building'] in buddhist_temple_db.LEVELS or any(r.findtext('level_name')==start['primary_building'] for r in base.rows('building_levels')),start
  base.add('start_pos_settlements',base.find('start_pos_settlements',region=building_region.findtext('id'),settlement_id='settlement:'+building),settlement_id=settlement,region=ids[0],id=ids[1],onscreen_name=name,**start)
  for offset,row in enumerate(own_slots):
   slot_id=first+2+offset
   assert str(slot_id) not in used_ids,'Slot ID collision'
   base.add('start_pos_region_slot_templates',row,campaign=base.CAMPAIGN,region=key,id=slot_id)
  # An authored population_pool replaces the donor's starting population;
  # without it the donor's value is kept.
  pool=county.get('population_pool')
  for table in ('start_pos_region_pooled_resources','start_pos_region_religions'):
   for row in [r for r in base.rows(table) if r.findtext('region')==source_id]:
    extra={'amount':str(int(pool))} if pool is not None and table=='start_pos_region_pooled_resources' and row.findtext('pooled_resource')=='3k_main_pooled_resource_population' else {}
    base.add(table,row,region=ids[0],**extra)
 # Explicit reviewed original-region administration edits preserve settlement
 # IDs, building templates, coordinates and faction ownership. Include modified
 # rows in the compiler/runtime table exporter rather than changing XML alone.
 for change in plan.get('vanilla_administrative_overrides',[]):
  key=change['region']
  junction=base.find('region_to_province_junctions',region=key)
  assert junction.findtext('is_capital')=='0', 'Cannot remove an existing original province capital'
  base.set_value(junction,'province',change['province'])
  base.CHANGES.setdefault('region_to_province_junctions',[]).append({c.tag:c.text for c in junction})
  if change.get('name'):
   row=base.find('regions',key=key)
   for field in ('onscreen','battle_name'):base.set_value(row,field,change['name'])
   base.CHANGES.setdefault('regions',[]).append({c.tag:c.text for c in row})
   row=base.find('start_pos_settlements',settlement_id='settlement:'+key)
   base.set_value(row,'onscreen_name',change['name'])
   base.CHANGES.setdefault('start_pos_settlements',[]).append({c.tag:c.text for c in row})
 for internal in plan.get('sdk_internal_land_regions',[]):
  key=internal['region'];colour=internal['colour']
  base.add('regions',base.find('regions',key='3k_main_reg_non_playable'),key=key,r=colour[0],g=colour[1],b=colour[2],is_sea=0)
  base.add('campaign_map_regions',base.find('campaign_map_regions',campaign_map=info['campaign_map'],region='3k_main_reg_non_playable'),campaign_map=info['campaign_map'],region=key)
 for sea in plan.get('imported_sea_regions',[]):
  key=sea['region'];colour=sea['colour']
  base.add('regions',base.find('regions',key='3k_main_sea_non_playable'),key=key,r=colour[0],g=colour[1],b=colour[2],is_sea=1,onscreen=key,battle_name=key)
  base.add('campaign_map_regions',base.find('campaign_map_regions',campaign_map=info['campaign_map'],region='3k_main_sea_non_playable'),campaign_map=info['campaign_map'],region=key)
 # Retired regions (seas merged into neighbours, SDK filler folded into the vanilla filler) leave this
 # map: drop their campaign_map_regions rows for it. Shared regions rows stay (other maps use them).
 # export_build_tables ships the filtered vanilla table as a full replacement, since runtime
 # fragments can only add rows.
 retired=set(plan.get('retired_regions',{}))
 if retired:
  root=base.tree('campaign_map_regions').getroot()
  gone=[r for r in root.findall('campaign_map_regions') if r.findtext('campaign_map')==info['campaign_map'] and r.findtext('region') in retired]
  for r in gone:root.remove(r)
  base.CHANGES.setdefault('campaign_map_regions',[])
  base.REMOVED={'campaign_map_regions':[{c.tag:c.text for c in r} for r in gone]}
 campaign=base.find('campaigns',campaign_name=base.CAMPAIGN)
 camera=base.find('campaign_camera_map_bounds',campaign=base.CAMPAIGN)
 # One source for the camera limits (vanilla margins on the expanded world).
 import campaign_world as _cw
 for k,v in _cw.CAMERA_BOUNDS.items():base.set_value(camera,k,v)
 base.CHANGES['campaign_camera_map_bounds']=[{c.tag:c.text for c in camera}]
 base.set_value(campaign,'map_name',info['campaign_map']);base.set_value(campaign,'display_location',info['campaign_map'])
 base.CHANGES['campaigns']=[{c.tag:c.text for c in campaign}]
 order={key:i for i,key in enumerate(info['land_regions']+info['sea_regions'])}
 root=base.tree('regions').getroot();records=list(root.findall('regions'))
 for row in records:root.remove(row)
 for row in sorted(records,key=lambda r:(order.get(r.findtext('key'),len(order)),r.findtext('key'))):root.append(row)
 root=base.tree('campaigns').getroot();records=list(root.findall('campaigns'))
 for row in records:root.remove(row)
 for row in sorted(records,key=lambda r:r.findtext('campaign_name')!=base.CAMPAIGN):root.append(row)
 for table,doc in base.TREES.items():
  if table in base.CHANGES:ET.indent(doc,space='  ');doc.write(base.KIT/'raw_data/db'/f'{table}.xml',encoding='utf-8',xml_declaration=True)
 report={**plan,'region':plan['regions'][0]['region'],'name':province['name'],'province':province['key'],'town':plan['regions'][0]['town'],'initial_owner_vanilla_id':owner,'requires_other_mods':False,'changed_rows':base.CHANGES,'removed_rows':getattr(base,'REMOVED',{}),'land_regions':len(info['land_regions']),'sea_regions':len(info['sea_regions']),'geometry':{r['region']:info['region_geometry'][r['region']] for r in plan['regions']}}
 if plan.get('version') in ('0.2','0.3'):
  report_name='historical-expansion-source.json' if plan['version']=='0.2' else 'seven-commanderies-source.json'
  (ROOT/'analysis'/report_name).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 else:
  (ROOT/'analysis/saibei-province-source.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
  if len(groups)>1:(ROOT/'analysis/four-frontiers-source.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print('Independent province:',province['key'],'counties:',len(plan['regions']),'capital:',plan['regions'][0]['region'],'owner:',owner,'tables:',len(base.CHANGES))
 return report
if __name__=='__main__':main()

