"""Gameplay data of the new regions (tw3k_*) that vanilla carries per region.

Settlement display layouts (campaign_settlement_display_settlement_layouts)
-------------------------------------------------------------------------
Vanilla gives a layout only to the 75 city settlements and 9 passes (84 rows, 13 layouts:
settlement_land_layout_a..g, settlement_port_layout_1/2, settlement_gate_layout_a and the
dlc06 Nanman land a/c and port a); resource towns have none. Vanilla's port rule is strict: the
14 cities with a port slot template use a port layout and no city without one does (ye_capital,
outside the 190 start_pos, is the only exception). prepare_province copied every new city's
layout from its gameplay donor, 3k_main_youzhou_capital, so all 97 new cities showed Youzhou.

Here a new city takes the layout of a vanilla donor city chosen per region:
  1. a city in a vanilla province copies that province's vanilla capital (Ailao -> Yongchang's
     Nanman layout, Nanjiang -> Zangke's), when its port-ness matches;
  2. otherwise its province's class (PROVINCE_LAYOUT_CLASS, a table by geography: oasis kingdom,
     Han frontier commandery, steppe, high mountain/plateau, north-east forest, Korean peninsula,
     islands, Indochina coast, southern inland, Nanman upland) picks the donor in LAYOUT_DONORS;
  3. a city with a port slot (a land port hex) takes the class's port donor, one without takes the
     land donor, so port layouts appear exactly where the port slot is.
A province missing from the table keeps the gameplay donor's layout (the old behaviour) and is
reported, so new designs never fail on it.

Settlement buildings and battle maps: assign_settlements() (called by prepare_province before the
gameplay policy) records per region a 'building_donor' (the vanilla region whose start_pos building
rows it copies: city level, starting districts, primary / secondary / port slot templates) and its
'battle_layout' in the design; the tables and rules are in the 'settlement buildings' section.

The other tables (region_effect_bundles, ...) are written by apply_kit(), which
all_directions_build.database() calls after the landmark rows, as landmarks_db does: rows go to the
kit XML (earlier tw3k_ rows replaced) and to standalone-prototype.json changed_rows, so
export_build_tables ships them as '!tw3k_exp' fragments over the vanilla tables (RUNTIME_TABLES).
Rules per table are documented above each section.

  python region_gameplay_db.py      # dry run on the current workspace: print the rows, write nothing
  python region_gameplay_db.py --check-battles   # also: battle templates exist in the vanilla packs
"""
import tw3k_paths
import json
import os
import sys
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

# class -> (land-layout donor city, port-layout donor city); the comment gives the donor's layout.
LAYOUT_DONORS = {
    'oasis':        ('3k_main_wuwei_capital', '3k_main_jincheng_capital'),      # land c (Guzang, Hexi oasis) / port 2
    'han_frontier': ('3k_main_anding_capital', '3k_main_shoufang_capital'),     # land d (Liang frontier) / port 2
    'steppe':       ('3k_main_daijun_capital', '3k_main_shoufang_capital'),     # land g (Dai, Xianbei frontier) / port 2
    'mountain':     ('3k_main_wudu_capital', '3k_main_badong_capital'),         # land g (Wudu, Di/Qiang hills) / port 2
    'northeast':    ('3k_dlc06_liaodong_capital', '3k_main_donglai_capital'),   # land a (Xiangping, Liaodong) / port 2
    'peninsula':    ('3k_main_beihai_capital', '3k_main_donglai_capital'),      # land f (Shandong peninsula) / port 2
    'island':       ('3k_main_yizhou_island_capital', '3k_main_dongou_capital'),  # land a (Yizhou island) / port 1
    'south_coast':  ('3k_dlc06_jiuzhen_capital', '3k_main_hepu_capital'),       # land a (Jiuzhen, next to Rinan) / port 1
    'south_inland': ('3k_main_gaoliang_capital', '3k_main_cangwu_capital'),     # land f (Gaoliang, tropical hills) / port 1
    'nanman':       ('3k_main_jianning_capital', '3k_main_jiangyang_capital'),  # dlc06 Nanman land c / Nanman port a
}

# Per-design tables (province classes, rice / walled provinces, reviewed resources) live in the design's
# region-traits.json (campaign_world.region_traits, beside current-region-design.json), so a rebuilt design (the
# Korea rebuild, P5) brings its own provinces without code edits.
import campaign_world as _cw
_TRAITS = _cw.region_traits()
# class -> province stems (tw3k_<any>_province_<stem>: tw3k_hist / tw3k_exp, and the rebuild's tw3k_kr / tw3k_rb)
_CLASSES = {cls: tuple(names) for cls, names in _TRAITS['province_classes'].items()}
_STEM_CLASS = {}
for _cls, _names in sorted(_CLASSES.items()):
    for _n in _names:
        assert _n not in _STEM_CLASS, _n
        _STEM_CLASS[_n] = _cls


class _ProvinceClasses(dict):
    """province key -> class for every authored province (tw3k_<prefix>_province_<stem>, the stem in region-traits
    province_classes). Was a table of tw3k_hist_ / tw3k_exp_ keys only, so the Korea rebuild's tw3k_kr_ / tw3k_rb_
    provinces fell back to the gameplay donor's layout and buildings (c3 dry run, 2026-11-01)."""
    def get(self, province, default=None):
        if province and province.startswith('tw3k_') and '_province_' in province:
            return _STEM_CLASS.get(province.split('_province_', 1)[1], default)
        return default


PROVINCE_LAYOUT_CLASS = _ProvinceClasses()

LAYOUT_TABLE = 'campaign_settlement_display_settlement_layouts'


def is_port_layout(layout):
    return 'port_layout' in layout


def layout_donor(county, port, rows):
    """Vanilla city whose layout row the new city copies, or None to keep the gameplay donor's.
    county: design region; port: the region has a port slot; rows: table -> vanilla XML rows."""
    layouts = {r.findtext('settlement'): r.findtext('layout') for r in rows(LAYOUT_TABLE)}
    province = county.get('province_key')

    def fits(region):
        layout = layouts.get('settlement:' + region)
        return layout is not None and is_port_layout(layout) == bool(port) and 'gate' not in layout

    capitals = sorted(r.findtext('region') for r in rows('region_to_province_junctions')
                      if r.findtext('province') == province and r.findtext('is_capital') == '1'
                      and r.findtext('region', '').startswith('3k_'))
    for region in capitals:
        if fits(region):
            return region
    cls = PROVINCE_LAYOUT_CLASS.get(province)
    if cls is None:
        return None
    donor = LAYOUT_DONORS[cls][1 if port else 0]
    if not fits(donor):
        raise ValueError(f'{donor} has no {"port" if port else "land"} layout for {county["region"]}')
    return donor


# ---------------------------------------------------------------- settlement buildings (building donor)
# A new region copies its start_pos building rows from one vanilla 190 region, its building donor:
# the start_pos_settlements row (primary_building, building1..5, port_building, wealth), the
# start_pos_region_slot_templates rows (primary chain, secondary chain, port; county_slots then
# makes the port slot follow the map) and slot_cap. Every other row (regions, campaign map,
# start_pos_regions profile: culture, rebels, development, population) still comes from the
# design's 'donor'. Vanilla's primary chain is 3k_city (66 cities), 3k_dlc06_city_nanman (7) or a
# resource chain; the secondary is 3k_districts(_port) for a city and 3k_resource_bandit for a
# resource town; a port slot exists exactly where the region has a land port hex.
#
# Cities: every new city copied Youzhou (3k_city_2 + landlord estates). Now the city copies a
# vanilla city chosen like its display layout: first its vanilla province's capital (Ailao ->
# Yongchang, Nanjiang -> Zangke, both Nanman cities), else its class's donor below, a port donor
# exactly when it has a port slot. The donors were picked by the starting districts (the city's
# economy) and a modest starting level, never Luoyang-class cities.
CITY_BUILDING_DONORS = {
    # class: (land donor, port donor)    # land donor's start: city level + districts
    'oasis':        ('3k_main_wuwei_capital', '3k_main_nanhai_capital'),      # city_1 + market inn: Guzang, caravan town / trade-port district
    'han_frontier': ('3k_main_anding_capital', '3k_main_shoufang_capital'),   # city_3 + rural administration, landlords: tuntian farming
    'steppe':       ('3k_main_xihe_capital', '3k_main_shoufang_capital'),     # city_0, no districts: Southern Xiongnu court
    'mountain':     ('3k_main_shangyong_capital', '3k_main_badong_capital'),  # city_2 + administration, state workshop: hill commandery
    'northeast':    ('3k_dlc06_liaodong_capital', '3k_main_dongou_capital'),  # city_1 + state workshop: Xiangping
    'peninsula':    ('3k_main_beihai_capital', '3k_main_hepu_capital'),       # city_2 + landlord estates: farming / fishing harbour
    'island':       ('3k_main_yizhou_island_capital', '3k_main_dongou_capital'),  # city_0: island chiefdom / fishing harbour
    'south_coast':  ('3k_dlc06_jiuzhen_capital', '3k_main_nanhai_capital'),   # city_1 + landlords, logistics / trade-port district
    'south_inland': ('3k_main_changsha_capital', '3k_main_cangwu_capital'),   # city_2 + market inn: rice-plain market town
    'nanman':       ('3k_main_zangke_capital', '3k_main_jiangyang_capital'),  # dlc06 Nanman city_1 / Nanman port city
}
# Han frontier commanderies that were military first (duwei garrisons on the Juyan and Ordos
# lines) copy a garrison city instead: 3k_city_2 + conscription + border market (Yulin).
HAN_FRONTIER_GARRISONS = {'juyan', 'xi', 'wuyuan'}
GARRISON_BUILDING_DONOR = '3k_main_yulin_capital'
# Vanilla provinces holding new regions: their class for the resource rules below (their cities
# copy the province capital first).
VANILLA_PROVINCE_CLASS = {'3k_dlc06_province_yongchang': 'nanman', '3k_main_province_zangke': 'nanman',
                          '3k_main_province_wuwei': 'han_frontier', '3k_main_province_xihe': 'han_frontier'}

# Resource towns keep their design donor unless its chain contradicts the region's climate, in
# which case the chain of the same role that fits is used (role and name stay consistent: a name
# suffix fixes the role through campaign_gameplay_policy.ALIASES, so only chains the suffix allows
# are chosen; a bare place name allows any chain of the role):
#   * fields: rice in the monsoon and paddy lands (islands, south coast, south inland, Nanman,
#     southern Korea: Sanguozhi, Han zhuan: Byeonjin 'yi zhong wu gu ji dao'), dry grain elsewhere;
#   * timber: bamboo only in the south (south coast, south inland, Nanman), pine elsewhere.
RICE_CLASSES = {'island', 'south_coast', 'south_inland', 'nanman'}
RICE_PROVINCES = set(_TRAITS['rice_provinces'])   # peninsula stems with paddy fields (old design: Mahan, Jinhan, Byeonhan)
BAMBOO_CLASSES = {'south_coast', 'south_inland', 'nanman'}
SWAPS = {'3k_resource_wood_farms_grain': ('3k_resource_wood_farms_rice', '3k_main_jiaozhi_resource_1'),
         '3k_resource_wood_farms_rice': ('3k_resource_wood_farms_grain', '3k_main_anding_resource_1'),
         '3k_resource_wood_lumber_pine': ('3k_resource_wood_lumber_bamboo', '3k_main_cangwu_resource_2'),
         '3k_resource_wood_lumber_bamboo': ('3k_resource_wood_lumber_pine', '3k_main_yanmen_resource_1')}


# Reviewed historical resources (analysis/region-diversity-2026-10-25.md, approved 2026-10-25): the
# region's building donor is set here, whatever its design donor; the name suffix follows the new
# primary chain (campaign_gameplay_policy.normalize_design) and so does the battle map (below).
# region: (building donor, needs a port slot, start building replacing the donor's or None, why), from
# region-traits.json resource_overrides. A trading port without a port slot on the map keeps its design donor
# (reported) until add_ports.py has given it one. Jibin (Gandhara, Buddhist centre under the Kushans) builds the
# mod's Buddhist copy of Pengcheng's temple chain (buddhist_temple_db.py: slot template in PRIMARY_SLOT_OVERRIDES).
RESOURCE_OVERRIDES = {k: (v['donor'], v['port'], v['start_building'], v['why'])
                      for k, v in _TRAITS['resource_overrides'].items()}


# Primary slot template replaced in the copied donor rows: region -> (donor template, new template).
PRIMARY_SLOT_OVERRIDES = {'tw3k_hist_jibin_farm': ('3k_resource_earth_temple', 'tw3k_resource_earth_buddhist_temple')}
# Starting city level of a city whose building donor's level is too high for it (region -> (that
# donor, level, why)); the donor's districts and port building stay. Dunxun (Mergui) copies the south-coast port
# donor Nanhai (3k_city_3, Panyu, the Han South Sea capital); approved 2026-10-25: one level lower.
CITY_START_OVERRIDES = {'tw3k_hist_dunxun_capital': ('3k_main_nanhai_capital', '3k_city_2', 'Dunxun: one level below its donor Nanhai (Panyu)')}
# Stop-loss 2026-10-26 (video package, romance startpos compile crashed): the Buddhist temple copy and Dunxun's
# lower start level are off unless TW3K_BUDDHIST_TEMPLE=1 / TW3K_DUNXUN_LEVEL=1; Jibin keeps the vanilla temple
# chain and start building of its donor (as before 5d44ed8f), Dunxun its donor's level.
import os as _os
if _os.environ.get('TW3K_BUDDHIST_TEMPLE', '0') != '1':
    PRIMARY_SLOT_OVERRIDES = {}
    _j = RESOURCE_OVERRIDES['tw3k_hist_jibin_farm']
    RESOURCE_OVERRIDES['tw3k_hist_jibin_farm'] = (_j[0], _j[1], None, _j[3])
if _os.environ.get('TW3K_DUNXUN_LEVEL', '0') != '1':
    CITY_START_OVERRIDES = {}


def region_class(county):
    """Geographic class of a design region (PROVINCE_LAYOUT_CLASS, then VANILLA_PROVINCE_CLASS) or None."""
    province = county.get('province_key')
    return PROVINCE_LAYOUT_CLASS.get(province) or VANILLA_PROVINCE_CLASS.get(province)


def province_stem(province):
    return province.split('_province_', 1)[-1]


def _vanilla_slots(rows):
    slots = {}
    for r in rows('start_pos_region_slot_templates'):
        if r.findtext('campaign') == '3k_main_campaign_map':
            slots.setdefault(r.findtext('region'), {})[r.findtext('slot_type')] = r.findtext('slot_template')
    return slots


def _wanted_chain(county, chain):
    """The chain of the same role that fits the region's climate (SWAPS), else chain."""
    cls = region_class(county)
    stem = province_stem(county.get('province_key', ''))
    if chain in ('3k_resource_wood_farms_grain', '3k_resource_wood_farms_rice'):
        rice = cls in RICE_CLASSES or (cls == 'peninsula' and stem in RICE_PROVINCES)
        return '3k_resource_wood_farms_rice' if rice else '3k_resource_wood_farms_grain'
    if chain in ('3k_resource_wood_lumber_pine', '3k_resource_wood_lumber_bamboo'):
        return '3k_resource_wood_lumber_bamboo' if cls in BAMBOO_CLASSES else '3k_resource_wood_lumber_pine'
    return chain


def name_allows(name, chain):
    """True when the region name's role suffix (if any) allows the chain's role."""
    from campaign_gameplay_policy import ALIASES, RESOURCE_SUFFIXES, ROLES
    suffix = name.rsplit('·', 1)[1] if '·' in name else None
    return suffix not in RESOURCE_SUFFIXES or suffix in ALIASES[ROLES[chain]]


def building_donor(county, port, rows):
    """(vanilla region whose start_pos building rows the new region copies, why).
    county: design region (its 'donor' is the gameplay donor); port: it has a port slot;
    rows: table -> vanilla XML rows."""
    slots = _vanilla_slots(rows)
    donor = county['donor']
    override = RESOURCE_OVERRIDES.get(county['region'])
    if override is not None and (port or not override[1]):
        chosen = override[0]
        if ('port' in slots[chosen]) != bool(override[1]):
            raise ValueError(f'{chosen} port slot does not match the override for {county["region"]}')
        return chosen, f'reviewed resource: {override[3]}'
    chain = slots[donor]['primary']
    if chain in ('3k_city', '3k_dlc06_city_nanman'):
        province = county.get('province_key')
        capitals = sorted(r.findtext('region') for r in rows('region_to_province_junctions')
                          if r.findtext('province') == province and r.findtext('is_capital') == '1'
                          and r.findtext('region', '').startswith('3k_'))
        for region in capitals:
            if region in slots and slots[region]['primary'] in ('3k_city', '3k_dlc06_city_nanman') and ('port' in slots[region]) == bool(port):
                return region, 'capital of its vanilla province'
        cls = PROVINCE_LAYOUT_CLASS.get(province)
        if cls is None:
            return donor, 'no class: design donor'
        if cls == 'han_frontier' and province_stem(province) in HAN_FRONTIER_GARRISONS and not port:
            chosen, why = GARRISON_BUILDING_DONOR, 'han_frontier garrison city'
        else:
            chosen, why = CITY_BUILDING_DONORS[cls][1 if port else 0], f'{cls} {"port" if port else "land"} city'
        if ('port' in slots[chosen]) != bool(port) or slots[chosen]['primary'] not in ('3k_city', '3k_dlc06_city_nanman'):
            raise ValueError(f'{chosen} is not a {"port" if port else "land"} city for {county["region"]}')
        return chosen, why
    wanted = _wanted_chain(county, chain)
    if wanted == chain or not name_allows(county['name'], wanted):
        return donor, 'design donor'
    chosen = SWAPS[chain][1]
    assert slots[chosen]['primary'] == wanted, chosen
    return chosen, f'{chain} -> {wanted} ({region_class(county)})'


# A resource town's battle map follows its primary chain: the vanilla resource_han_* map of that
# chain (size kept where vanilla has that size, else small), or a port settlement map for a water
# chain that already has one (Hanseong's trading port keeps settlement_port_han_b_small).
RESOURCE_BATTLE = {
    '3k_resource_wood_farms_grain': ('resource_han_farm_grain_a',),
    '3k_resource_wood_farms_rice': ('resource_han_farm_rice_a',),
    '3k_resource_wood_lumber_pine': ('resource_han_lumber_a',),
    '3k_resource_wood_lumber_bamboo': ('resource_han_lumber_a',),
    '3k_resource_earth_temple': ('resource_han_temple_a',),
    '3k_resource_water_silk': ('resource_han_market_a_silk',),
    '3k_resource_water_trading_port': ('resource_han_port_a_trade', 'settlement_port_'),
}
# The vanilla resource battle templates (battles.xml; checked against the packs by --check-battles).
VANILLA_RESOURCE_BATTLES = frozenset(
    [f'resource_han_{k}_{s}' for k in ('farm_grain_a', 'farm_rice_a', 'market_a_silk', 'market_a_spice', 'mine_a_copper',
                                       'mine_a_iron', 'mine_a_jade', 'port_a_fishing', 'port_a_trade', 'tools_a_regular')
     for s in ('small', 'large')]
    + ['resource_han_livestock_a_horses_small', 'resource_han_livestock_a_regular_small', 'resource_han_lumber_a_small',
       'resource_han_salt_a_small', 'resource_han_tea_a_small', 'resource_han_temple_a_small',
       'resource_han_tools_a_master_craftsmen_small'])


def resource_battle_layout(chain, layout):
    """Battle layout for a resource town of this chain, keeping the current one when it fits."""
    stems = RESOURCE_BATTLE.get(chain)
    if stems is None or any(layout.startswith(s) for s in stems):
        return layout
    size = 'large' if layout.endswith('_large') else 'small'
    out = f'{stems[0]}_{size}'
    return out if out in VANILLA_RESOURCE_BATTLES else f'{stems[0]}_small'


# ---------------------------------------------------------------- rebels (start_pos_regions)
# start_pos_regions.rebel_faction / alternative_rebel_faction were copied from the design donor:
# Yellow Turban armies (taishan, anding, 'rebels') in Korea, the steppe or Funan. Vanilla 190 gives
# every playable region alternative 3k_main_faction_rebels and a regional Yellow Turban
# rebel_faction; even its Nanman regions (Yongchang, Jianning, Zangke) keep Yellow Turbans. New:
#   * a region in a vanilla province copies its province capital's pair (Ailao -> Yongchang,
#     Nanjiang -> Zangke, Wuwei, Xihe), as vanilla does within a province;
#   * Nanman, south inland and south coast classes: 3k_dlc06_faction_nanman_rebels for both (an
#     is_rebel Nanman-subculture faction already in the 190 startpos, Nanman units);
#   * Han frontier commanderies: the Liang / Bing frontier pair of vanilla (Wuwei, Xihe, Shuofang,
#     Anding, Jincheng: rebel 3k_main_faction_yellow_turban_anding, alternative 3k_main_faction_rebels);
#   * every other new region: 3k_main_faction_rebels for both (the generic is_rebel faction).
# No faction is created; prepare_province checks every key against the 190 start_pos_factions.
NANMAN_REBELS = '3k_dlc06_faction_nanman_rebels'
GENERIC_REBELS = '3k_main_faction_rebels'
NANMAN_REBEL_CLASSES = {'nanman', 'south_inland', 'south_coast'}
FRONTIER_REBELS = ('3k_main_faction_yellow_turban_anding', GENERIC_REBELS)   # vanilla Wuwei / Xihe / Shuofang


def rebel_factions(county, rows):
    """(rebel_faction, alternative_rebel_faction, why) for a new region (rules above)."""
    province = county.get('province_key')
    capitals = sorted(r.findtext('region') for r in rows('region_to_province_junctions')
                      if r.findtext('province') == province and r.findtext('is_capital') == '1'
                      and r.findtext('region', '').startswith('3k_'))
    for region in capitals:
        for r in rows('start_pos_regions'):
            if r.findtext('campaign') == '3k_main_campaign_map' and r.findtext('region') == region:
                return r.findtext('rebel_faction'), r.findtext('alternative_rebel_faction'), 'vanilla province capital ' + region
    cls = PROVINCE_LAYOUT_CLASS.get(province)
    if cls in NANMAN_REBEL_CLASSES:
        return NANMAN_REBELS, NANMAN_REBELS, cls
    if cls == 'han_frontier':
        return FRONTIER_REBELS + ('han_frontier (vanilla Liang / Bing frontier)',)
    return GENERIC_REBELS, GENERIC_REBELS, cls or 'no class'


def kind_of_layout(layout):
    """farm_fields.kind_of: terrain steps read the battle layout's kind (farm / city / other)."""
    layout = str(layout or '')
    return 'farm' if 'farm' in layout or 'tea' in layout else 'city' if layout.startswith('settlement_') else 'other'


# ---------------------------------------------------------------- settlement battle maps (cities)
# A new region's battle entrances (build_expanded_battle_entrances.py) name the design's
# battle_layout, a vanilla terrain/battles/<template> folder. 82 of the 97 new cities used
# settlement_city_han_a_small. The vanilla settlement templates differ by town plan (han a..h,
# Nanman a / c, port han a / b, port Nanman a), size (small, small_walled, medium, large) and by
# sea (port templates draw their own harbour); their ground is flat, the relief comes from the
# campaign map, so a mountain fort is a walled template on mountain ground. A city takes:
#   * style: Nanman (stilt and palisade villages) for the Nanman, south inland and south coast
#     classes, except Rinan (a Han commandery); Han elsewhere;
#   * walls: walled for the oasis city-states (walled towns: Kucha, Khotan, Loulan), the Han
#     frontier commanderies, the mountain forts, and the walled polities the Sanguozhi describes
#     (Buyeo 'zuo cheng zha', Goguryeo, Xuantu, Liaodong, Lelang, Jinhan 'you cheng zha', Byeonhan
#     'you cheng guo', Yamatai 'cheng zha yan she'); open for the steppe (camps), Mahan ('wu cheng
#     guo'), Ye, Okjeo, Yilou, Suoli and the other Wa states;
#   * size: medium (the larger walled plan) for a city whose population pool is >= MEDIUM_POPULATION
#     (Shu and Guanghan dependencies, Lelang, Buyeo, Goguryeo, Hanyang, Wusun), else small;
#   * sea: a port template when the town is within COAST_HEXES of navigable sea or has a port slot;
#   * plan letter: per province from its class's letters (crc32 of the province key), so the
#     towns of one province look alike and neighbouring provinces differ.
# Only vanilla templates (VANILLA_SETTLEMENT_BATTLES, checked against the packs by
# `python region_gameplay_db.py --check-battles`) can be chosen.
VANILLA_SETTLEMENT_BATTLES = frozenset(
    [f'settlement_city_han_{x}_{s}' for x in 'abcdefgh' for s in ('small', 'small_walled', 'medium', 'large')]
    + [f'settlement_city_nanman_{x}{s}' for x in 'ac' for s in ('', '_small', '_medium')] + ['settlement_city_nanman_b']
    + [f'settlement_port_han_{x}_{s}' for x in 'ab' for s in ('small', 'small_walled', 'medium', 'large')]
    + ['settlement_port_nanman_a_small', 'settlement_port_nanman_a_medium'])
NANMAN_STYLE_CLASSES = {'nanman', 'south_inland', 'south_coast'}
HAN_STYLE_PROVINCES = {'rinan'}
WALLED_CLASSES = {'oasis', 'han_frontier', 'mountain'}
WALLED_PROVINCES = set(_TRAITS['walled_provinces'])
HAN_LETTERS = {'oasis': 'cde', 'han_frontier': 'ha', 'steppe': 'bg', 'mountain': 'fg', 'northeast': 'db',
               'peninsula': 'efb', 'island': 'ac', 'south_coast': 'h'}
NANMAN_LETTERS = 'ac'
PORT_HAN_LETTERS = 'ab'
MEDIUM_POPULATION = 40
COAST_HEXES = 3


def city_battle_layout(county, coastal):
    """Vanilla settlement battle template for a new city (rules above)."""
    import zlib
    cls = region_class(county) or 'han_frontier'
    stem = province_stem(county.get('province_key', ''))
    pick = lambda letters: letters[zlib.crc32(county.get('province_key', '').encode()) % len(letters)]
    medium = (county.get('population_pool') or 0) >= MEDIUM_POPULATION
    if cls in NANMAN_STYLE_CLASSES and stem not in HAN_STYLE_PROVINCES:
        size = 'medium' if medium else 'small'
        layout = f'settlement_port_nanman_a_{size}' if coastal else f'settlement_city_nanman_{pick(NANMAN_LETTERS)}_{size}'
    else:
        walled = cls in WALLED_CLASSES or stem in WALLED_PROVINCES
        size = 'medium' if medium else 'small_walled' if walled else 'small'
        layout = (f'settlement_port_han_{pick(PORT_HAN_LETTERS)}_{size}' if coastal
                  else f'settlement_city_han_{pick(HAN_LETTERS.get(cls, "a"))}_{size}')
    assert layout in VANILLA_SETTLEMENT_BATTLES, layout
    return layout


def coastal_towns(m, plan, radius=COAST_HEXES):
    """{region: True} for design towns within radius hexes of navigable sea (not a lake or river
    sea region) on a map_hex.read map."""
    import map_hex
    from hex_components import hex_dilate
    c = m['cells']
    ids = map_hex.region_ids(c)
    names = list(m['land_regions']) + list(m['sea_regions'])
    n = len(m['land_regions'])
    navigable = [i for i, k in enumerate(names) if i >= n and k not in ('3k_main_sea_lake', '3k_main_sea_non_playable')
                 and RIVER_SEA_TAG not in k and 'lake' not in k]
    near = ((c[..., 0] & 3) == 1) & np.isin(ids, navigable)
    for _ in range(radius):
        near = near | hex_dilate(near)
    return {t['region']: bool(near[t['town'][1], t['town'][0]]) for t in plan['regions']}


def assign_settlements(plan, ports, rows, coastal=None):
    """Set 'building_donor' and 'battle_layout' on every design region (plan['regions']; provinces[].regions hold
    keys only, design_single_source); returns {region: decision}. ports: {region: land port hexes};
    coastal: {region: town near navigable sea} (coastal_towns; None leaves city battle maps as they
    are); rows: table -> vanilla XML rows. A pure function of the design's donor, name, province,
    population and the map, so a second run changes nothing. The battle layout keeps its terrain
    kind (farm / city / other), which farm_fields and polish_campaign_terrain read."""
    slots = _vanilla_slots(rows)
    decisions = {}
    for county in plan['regions']:
        key = county['region']
        port = ports.get(key, 0) > 0
        bd, why = building_donor(county, port, rows)
        layout = county.get('battle_layout', '')
        if slots[bd]['primary'] in ('3k_city', '3k_dlc06_city_nanman') and coastal is not None:
            new_layout = city_battle_layout(county, port or coastal.get(key, False))
        else:
            new_layout = resource_battle_layout(slots[bd]['primary'], layout)
        # A reviewed resource change may change the terrain kind (mine -> rice fields, grain -> temple):
        # the terrain steps that read it (farm_fields, polish) must run again after the database.
        reviewed = bd == RESOURCE_OVERRIDES.get(key, (None,))[0]
        if kind_of_layout(new_layout) != kind_of_layout(layout) and not reviewed:
            raise ValueError(f'{key}: battle layout {layout} -> {new_layout} changes its terrain kind')
        decisions[key] = {'building_donor': bd, 'why': why, 'primary': slots[bd]['primary'],
                          'battle_layout': new_layout, 'port_slot': port}
        if kind_of_layout(new_layout) != kind_of_layout(layout):
            decisions[key]['terrain_kind_changed'] = [kind_of_layout(layout), kind_of_layout(new_layout)]
        if key in RESOURCE_OVERRIDES and RESOURCE_OVERRIDES[key][1] and not port:
            decisions[key]['why'] += ' (reviewed trading port waits for a port slot: add_ports.py)'
        start = RESOURCE_OVERRIDES[key][2] if reviewed else None
        if CITY_START_OVERRIDES.get(key, (None,))[0] == bd:     # only with the donor it lowers
            start = CITY_START_OVERRIDES[key][1]
        if start:
            county['primary_building'] = decisions[key]['primary_building'] = start
        else:
            county.pop('primary_building', None)
        if reviewed and key in PRIMARY_SLOT_OVERRIDES:
            assert slots[bd]['primary'] == PRIMARY_SLOT_OVERRIDES[key][0], key
            county['primary_slot_template'] = decisions[key]['primary_slot_template'] = PRIMARY_SLOT_OVERRIDES[key][1]
        else:
            county.pop('primary_slot_template', None)
        county['building_donor'] = bd
        county['battle_layout'] = new_layout
    return decisions


# ---------------------------------------------------------------- shared helpers
VANILLA_DB = tw3k_paths.ASSEMBLY_KIT / 'raw_data' / 'db'
PREFIX = 'tw3k_'
_VANILLA = {}


def vanilla_rows(table):
    """Vanilla assembly-kit rows of a table as dicts (cached)."""
    if table not in _VANILLA:
        _VANILLA[table] = [{c.tag: c.text or '' for c in r}
                           for r in ET.parse(VANILLA_DB / f'{table}.xml').getroot().findall(table)]
    return _VANILLA[table]


def primary_keys(table):
    schema = ET.parse(VANILLA_DB / f'TWaD_{table}.xml').getroot()
    return [f.findtext('name') for f in schema.findall('field') if f.findtext('primary_key') == '1']


def vanilla_playable():
    """Vanilla 190 playable land regions: start_pos regions of the 190 campaign, no filler or sea."""
    return sorted({r['region'] for r in vanilla_rows('start_pos_regions') if r['campaign'] == '3k_main_campaign_map'
                   and r['region'].startswith('3k_') and 'non_playable' not in r['region'] and 'sea' not in r['region']})


def apply_rows(kit, report, table, new_rows, ours):
    """Replace this module's rows of a kit table (ours(row dict) is True) by new_rows (ordered dicts,
    already sorted) and record them in report['changed_rows'] (export_build_tables ships them as the
    table's '!tw3k_exp' runtime fragment). Writes through a temporary file (atomic replace)."""
    path = Path(kit) / f'raw_data/db/{table}.xml'
    doc = ET.parse(path)
    root = doc.getroot()
    for r in [r for r in root.findall(table) if ours({c.tag: c.text or '' for c in r})]:
        root.remove(r)
    keys = primary_keys(table)
    for row in new_rows:
        el = ET.SubElement(root, table)
        identity = [(k, row[k]) for k in keys]
        el.set('record_uuid', '{' + str(uuid.uuid5(uuid.NAMESPACE_URL, 'tw3k-authored/' + table + '/' + json.dumps(identity, ensure_ascii=False))) + '}')
        el.set('record_key', ''.join(row[k] for k in keys))
        for k, v in row.items():
            ET.SubElement(el, k).text = v
    tmp = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    try:
        doc.write(tmp, encoding='utf-8', xml_declaration=True)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    changed = report.setdefault('changed_rows', {})
    changed[table] = [r for r in changed.get(table, []) if not ours(r)] + [dict(r) for r in new_rows]
    return len(new_rows)


def region_features(design, map_path):
    """Per new region: land hexes, road hexes, mean canvas q, subtropical climate share, tropical
    biome share and mean real elevation (strategic_ink cache: RESOLVE biome, ETOPO metres per hex)."""
    import map_hex
    import strategic_ink
    m = map_hex.read(Path(map_path))
    c = m['cells']
    ids = map_hex.region_ids(c)
    n = len(m['land_regions'])
    biome, names, _salt, elev = strategic_ink.load()
    if biome.shape != ids.shape:
        raise ValueError(f'strategic ink cache {biome.shape} does not match the map {ids.shape}')
    land = ((c[..., 0] & 3) != 1) & (ids >= 0) & (ids < n)
    road = ((c[..., 3] >> 1) & 63) != 0
    climate = (c[..., 7] >> 1).astype(np.int32) - 1
    sub = m['climates'].index('subtropical')
    tropical = names.index('tropical')
    index = {k: i for i, k in enumerate(m['land_regions'])}
    lid = ids[land]
    count = np.bincount(lid, minlength=n).astype(np.float64)

    def mean(values):
        return np.bincount(lid, weights=values[land].astype(np.float64), minlength=n) / np.maximum(count, 1)
    q = np.broadcast_to(np.arange(ids.shape[1])[None, :], ids.shape)
    mean_q, mean_e = mean(q), mean(elev)
    roads = np.bincount(lid, weights=road[land].astype(np.float64), minlength=n)
    subtropical, trop = mean(climate == sub), mean(biome == tropical)
    out = {}
    for county in design['regions']:
        i = index[county['region']]
        out[county['region']] = {'land_hexes': int(count[i]), 'road_hexes': int(roads[i]), 'mean_q': round(float(mean_q[i]), 1),
                                 'subtropical': round(float(subtropical[i]), 3), 'tropical': round(float(trop[i]), 3),
                                 'elevation_m': int(round(float(mean_e[i])))}
    return out


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    try:
        tmp.write_text(json.dumps(value, ensure_ascii=False, indent=1, sort_keys=True) + '\n', encoding='utf-8')
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------- region_effect_bundles
# Vanilla (329 rows, 94% of the 201 playable regions) uses five kinds of region bundle:
#   * province bundles (3k_*_effect_bundle_province_<zhou>_<commandery>, group
#     3k_dlc05_bandit_network_region_effect_bundle_empty): every region of a province carries its
#     province's one bundle (bandit-network cost reduction for that area); the new provinces have no
#     bundle, so only new regions in a vanilla province get one;
#   * road grades (group 3k_main_campaign_roads): _02 'Road' (+10 road level, 34 regions) and _03
#     'Imperial Highway' (+40, 51 regions); 116 regions have none. The grade is the best road tile
#     class drawn in the region (vanilla tile_list.bin, by the piece's origin hex): any roads_imperial
#     piece -> _03, else any roads_paved -> _02, tracks only or no road -> none. On the 201 vanilla
#     regions this is right for 180 (166 of the 177 3k_main regions; most misses are dlc06 passes and
#     regions dlc06 added without bundles): _03 precision 46/53, recall 46/51; _02 30/42, 30/34; none
#     104/106, 104/116; 104 of the 106 tracks-only regions have no grade. Nothing else predicts it:
#     road-hex count (best threshold 118/201 vs the all-none baseline 116), road density (123),
#     distance to Luoyang (139), elevation (132), capital vs resource (39 of the 116 ungraded regions
#     are capitals); culture and development are the same in all 201 (region_gameplay_db.py
#     --vanilla-road-check reproduces the 180). New regions: the same rule on the tiles road_tiles.py
#     lays on their road hexes (road_tiles.SET, roads_tracks: frontier tracks), so they get none, like
#     vanilla Wuwei, Liaodong, Jiaozhi and Yongchang (tracks only). Paved / imperial pieces on a route
#     would raise its regions' grade through the same table;
#   * dense jungle (3k_dlc06_campaign_dense_jungle_dummy, 16 regions): the tropical uplands of the
#     south-west. Learned on vanilla: tropical forest (RESOLVE) >= JUNGLE_TROPICAL_MIN, subtropical
#     climate >= JUNGLE_SUBTROPICAL_MIN, mean elevation >= JUNGLE_ELEVATION_MIN and west of the Wuling /
#     Three Gorges line (mean canvas q < PAD_LEFT + JUNGLE_WEST_OF_VANILLA_Q) selects exactly the 16
#     vanilla jungle regions plus jiangyang_resource_3 (Nanman lands, no jungle in vanilla);
#   * Nanman lands (3k_dlc06_campaign_group_nanman_lands, 21 regions): every jungle region plus the rest
#     of Jiangyang, Wuling capital and Jiaozhi 2. New regions: jungle regions, and regions of a vanilla
#     province whose vanilla regions are all Nanman lands;
#   * 3k_main_ancient_capital (Chang'an, Luoyang, Yingchuan, Weijun, Pengcheng: former capitals of the
#     central dynasties; Nanyue's Panyu, Nanhai capital, has none) and ep_region_payload_capital
#     (Luoyang, the Emperor's seat). No new region was a capital of a central dynasty: none.
ROAD_GROUP = '3k_main_campaign_roads'
ROAD_GRADE_BY_TILE_CLASS = (('roads_imperial', '3k_main_campaign_roads_03'), ('roads_paved', '3k_main_campaign_roads_02'))
JUNGLE_GROUP, JUNGLE_BUNDLE = '3k_dlc06_campaign_dense_jungle_dummy', '3k_dlc06_effect_bundle_region_jungle_dummy'
NANMAN_GROUP, NANMAN_BUNDLE = '3k_dlc06_campaign_group_nanman_lands', '3k_dlc06_effect_bundle_region_nanman_lands'
PROVINCE_GROUP = '3k_dlc05_bandit_network_region_effect_bundle_empty'
JUNGLE_TROPICAL_MIN = 0.89      # vanilla jungle minimum 0.896 (Yunnan 2)
JUNGLE_SUBTROPICAL_MIN = 0.30   # vanilla jungle minimum 0.32 (Fuling capital)
JUNGLE_ELEVATION_MIN = 700      # metres; vanilla jungle minimum 756 (Zangke 2); Wuling capital (689) has none
JUNGLE_WEST_OF_VANILLA_Q = 292  # vanilla-frame q: jungle regions <= 279 (Zangke capital); Badong 1 (317) has none


def road_grade(tile_classes):
    """Road-grade bundle for the road tile classes drawn in a region, or None."""
    for tile_class, bundle in ROAD_GRADE_BY_TILE_CLASS:
        if tile_class in tile_classes:
            return bundle
    return None


def new_road_tile_classes(road_hexes):
    """Road tile classes on a new region's road hexes: road_tiles.py lays one class on all of them."""
    import road_tiles
    return {road_tiles.SET} if road_hexes else set()


def vanilla_province_bundles():
    """{vanilla province: its province bundle}, from the bundles its vanilla regions carry."""
    province = {r['region']: r['province'] for r in vanilla_rows('region_to_province_junctions')}
    found = {}
    for r in vanilla_rows('region_effect_bundles'):
        if r['group'] == PROVINCE_GROUP and r['region'] in province:
            found.setdefault(province[r['region']], set()).add(r['effect_bundle'])
    return {p: sorted(b)[0] for p, b in sorted(found.items()) if len(b) == 1}


def vanilla_nanman_provinces():
    """Vanilla provinces whose playable vanilla regions all carry the Nanman lands bundle."""
    playable = set(vanilla_playable())
    nanman = {r['region'] for r in vanilla_rows('region_effect_bundles') if r['effect_bundle'] == NANMAN_BUNDLE}
    members = {}
    for r in vanilla_rows('region_to_province_junctions'):
        if r['region'] in playable:
            members.setdefault(r['province'], set()).add(r['region'])
    return sorted(p for p, regions in members.items() if regions <= nanman)


def vanilla_road_check():
    """The road-grade rule on vanilla: (right, total, confusion {(vanilla grade, rule grade): n})."""
    import collections
    import campaign_world as cw
    import map_hex
    from campaign_ground_textures import VanillaPacks
    from read_tile_list import parse
    m = map_hex.read(cw.VANILLA_MAP)
    c = m['cells']
    ids = map_hex.region_ids(c)
    h, w = ids.shape
    land = ((c[..., 0] & 3) != 1) & (ids >= 0) & (ids < len(m['land_regions']))
    paths, _, _, _, recs, _ = parse(VanillaPacks().read('terrain/campaigns/3k_dlc07_main_map/global_map/tile_list.bin'))
    sep = chr(92)
    classes = collections.defaultdict(set)
    for r in recs:
        path = paths[r[1]]
        if sep + 'roads_' not in path:
            continue
        q = int(round(r[3] / 2)); rr = int(round(r[4] / 2 - (q % 2) * .5))
        if 0 <= q < w and 0 <= rr < h and land[rr, q]:
            classes[m['land_regions'][ids[rr, q]]].add(path.split(sep)[3])
    graded = {r['region']: r['effect_bundle'] for r in vanilla_rows('region_effect_bundles') if r['group'] == ROAD_GROUP}
    confusion = collections.Counter((graded.get(k), road_grade(classes.get(k, set()))) for k in vanilla_playable())
    return sum(n for (a, b), n in confusion.items() if a == b), sum(confusion.values()), dict(sorted(confusion.items(), key=str))


def effect_bundle_rows(design, features, pad_left):
    """(rows, decisions): region_effect_bundles rows of the new regions (sorted) and why."""
    bundles = vanilla_province_bundles()
    nanman_provinces = set(vanilla_nanman_provinces())
    rows, decisions = [], {}
    for county in sorted(design['regions'], key=lambda c: c['region']):
        key, province, f = county['region'], county.get('province_key', ''), features[county['region']]
        why = []
        if province in bundles:
            rows.append((key, PROVINCE_GROUP, bundles[province])); why.append('province bundle of ' + province)
        tiles = new_road_tile_classes(f['road_hexes'])
        grade = road_grade(tiles)
        if grade:
            rows.append((key, ROAD_GROUP, grade))
        why.append(f"road grade {grade or 'none'}: {f['road_hexes']} road hexes, {', '.join(sorted(tiles)) or 'no road'}")
        jungle = (f['tropical'] >= JUNGLE_TROPICAL_MIN and f['subtropical'] >= JUNGLE_SUBTROPICAL_MIN
                  and f['elevation_m'] >= JUNGLE_ELEVATION_MIN and f['mean_q'] < pad_left + JUNGLE_WEST_OF_VANILLA_Q)
        if jungle:
            rows.append((key, JUNGLE_GROUP, JUNGLE_BUNDLE)); why.append('dense jungle: tropical south-western upland')
        if jungle or province in nanman_provinces:
            rows.append((key, NANMAN_GROUP, NANMAN_BUNDLE)); why.append('Nanman lands' + ('' if jungle else ': Nanman province'))
        decisions[key] = {'province': province, 'features': f, 'bundles': why}
    out = [{'effect_bundle': b, 'group': g, 'region': r} for r, g, b in sorted(rows)]
    return out, decisions


# ---------------------------------------------------------------- cai_region_groups(_to_regions_junctions)
# Vanilla (1246 junction rows, 91% of the playable regions): a region group is one faction's set of
# target regions, read by cai_region_hint_groups (hint CAPTURE_VICTORY_CONDITIONS) through a hint
# profile; the 190 groups are 3k_main_region_group_<faction> (12) and 3k_dlc06_region_group_nanman.
# They are built from whole provinces (4 of 13 exactly; the others differ by one or two regions of a
# province, mostly regions dlc06 added later),
# and the Nanman group holds every Nanman-lands region. So a new region joins:
#   * every 190 group holding all vanilla regions of its vanilla province (keeps the groups
#     province-complete: Wuwei -> Ma Teng, Xihe -> Zheng Jiang, Zangke -> Liu Biao and Nanman,
#     Yongchang -> Nanman);
#   * the Nanman group when it carries the Nanman lands bundle;
#   * in a new province, that province's own new group tw3k_*_region_group_<province> (a new
#     cai_region_groups key, one group per commandery / kingdom). No hint group or profile uses
#     these yet (that is faction AI data), so they only take effect once a faction's profile does.
CAI_GROUPS_190 = '3k_main_region_group_'
CAI_NANMAN = '3k_dlc06_region_group_nanman'


def new_group_key(province):
    return province.replace('_province_', '_region_group_', 1)


def cai_group_rows(design, bundle_rows):
    """(cai_region_groups rows, junction rows), sorted."""
    playable = set(vanilla_playable())
    members = {}
    for r in vanilla_rows('cai_region_groups_to_regions_junctions'):
        if r['region_group_key'].startswith(CAI_GROUPS_190) or r['region_group_key'] == CAI_NANMAN:
            members.setdefault(r['region_group_key'], set()).add(r['region_key'])
    province_regions = {}
    for r in vanilla_rows('region_to_province_junctions'):
        if r['region'] in playable:
            province_regions.setdefault(r['province'], set()).add(r['region'])
    nanman = {r['region'] for r in bundle_rows if r['effect_bundle'] == NANMAN_BUNDLE}
    groups, junctions = set(), set()
    for county in design['regions']:
        key, province = county['region'], county.get('province_key', '')
        if province in province_regions:
            for group, regions in members.items():
                if province_regions[province] <= regions:
                    junctions.add((group, key))
        else:
            groups.add(new_group_key(province))
            junctions.add((new_group_key(province), key))
        if key in nanman:
            junctions.add((CAI_NANMAN, key))
    return ([{'key': g} for g in sorted(groups)],
            [{'region_group_key': g, 'region_key': r} for g, r in sorted(junctions)])


# ---------------------------------------------------------------- campaign_region_additional_adjacency_rules
# Vanilla (55 rows, 30% of the playable regions) makes two regions adjacent where a river separates
# them: 47 rules join regions facing each other across one or two hexes of a river sea region
# ('*_riv_sea_*', Yangtze, Yellow River, Xiang), the other 8 are hand-placed strategic links across
# the Yellow River and Hedong hills (Hedong - Luoyang / Chang'an). Measured on the vanilla map: of
# the region pairs (no pass regions) that face each other across <= RIVER_GAP_MAX river hexes with
# >= RIVER_FRONTAGE_MIN facing hexes on both banks, 45 have a rule and 7 do not (87%), and 45 of the 47
# river rules are found. Vanilla links no sea strait: not Zhuya (Hainan, 6 hexes off Hepu) nor Yizhou
# island (32 hexes); so no new link crosses the sea either (Jeju is 20 hexes from Korea, Kyushu 57+;
# there is no Tsushima / Iki region). New rules: the same river test on every pair with a new region.
RIVER_GAP_MAX = 2
RIVER_FRONTAGE_MIN = 4
RIVER_SEA_TAG = '_riv_sea_'
ADJACENCY_TABLE = 'campaign_region_additional_adjacency_rules'


def river_crossings(map_path, regions):
    """{(a, b): facing hexes} for region pairs (a < b, at least one in regions) that touch nowhere but
    face each other across <= RIVER_GAP_MAX river hexes with >= RIVER_FRONTAGE_MIN hexes on each bank."""
    import map_hex
    from hex_components import hex_dilate
    m = map_hex.read(Path(map_path))
    c = m['cells']
    ids = map_hex.region_ids(c)
    n = len(m['land_regions'])
    names = list(m['land_regions']) + list(m['sea_regions'])
    h, w = ids.shape
    water = (c[..., 0] & 3) == 1
    land = (ids >= 0) & (ids < n) & ~water
    river = water & np.isin(ids, [i for i, k in enumerate(names) if RIVER_SEA_TAG in k])
    playable = {i for i, k in enumerate(m['land_regions'])
                if (k.startswith(PREFIX) or (k.startswith('3k_') and 'non_playable' not in k)) and not k.endswith('_pass')}
    memo = {}

    def facing(i):
        """{b: (facing hexes of b, river share of the water before them)} seen from region i."""
        if i in memo:
            return memo[i]
        r, q = np.nonzero(land & (ids == i))
        r0, r1 = max(0, r.min() - RIVER_GAP_MAX - 3), min(h, r.max() + RIVER_GAP_MAX + 4)
        q0, q1 = max(0, q.min() - RIVER_GAP_MAX - 3), min(w, q.max() + RIVER_GAP_MAX + 4)
        q0 -= q0 % 2                                     # keep the column parity of hex_dilate
        L, I, W, RV = land[r0:r1, q0:q1], ids[r0:r1, q0:q1], water[r0:r1, q0:q1], river[r0:r1, q0:q1]
        mine = L & (I == i)
        direct = set(np.unique(I[hex_dilate(mine) & L]).tolist())
        reach = mine.copy()
        for _ in range(RIVER_GAP_MAX):
            reach |= hex_dilate(reach) & W
        wet = reach & ~mine
        hit = hex_dilate(wet) & L & ~mine
        out = {}
        for b in sorted(set(np.unique(I[hit]).tolist()) - direct):
            if b not in playable:
                continue
            front = hit & (I == b)
            before = wet & hex_dilate(front)
            out[b] = (int(front.sum()), float(RV[before].mean()) if before.any() else 0.0)
        memo[i] = out
        return out

    want = sorted(i for i, k in enumerate(m['land_regions']) if k in regions)
    pairs = {}
    for i in want:
        for b, (fa, ra) in facing(i).items():
            fb, rb = facing(b).get(i, (0, 0.0))
            if min(fa, fb) >= RIVER_FRONTAGE_MIN and min(ra, rb) > 0.5:
                a_key, b_key = sorted((names[i], names[b]))
                pairs[(a_key, b_key)] = min(fa, fb)
    return dict(sorted(pairs.items()))


def adjacency_rows(design, map_path):
    new = {c['region'] for c in design['regions']}
    pairs = river_crossings(map_path, new)
    return [{'first_region': a, 'second_region': b} for a, b in pairs], pairs


def apply_kit(kit, report, workspace, design):
    """All region gameplay rows of this module into the kit XML and report['changed_rows'], with a
    decision report in <workspace>/analysis/region-gameplay-db.json. Called by
    all_directions_build.database() before export_build_tables."""
    import campaign_world as cw
    features = region_features(design, Path(workspace) / 'work/final-map/map.hex')
    ours = lambda r: r.get('region', '').startswith(PREFIX)
    rows, decisions = effect_bundle_rows(design, features, cw.PAD_LEFT)
    counts = {'region_effect_bundles': apply_rows(kit, report, 'region_effect_bundles', rows, ours)}
    groups, junctions = cai_group_rows(design, rows)
    counts['cai_region_groups'] = apply_rows(kit, report, 'cai_region_groups', groups, lambda r: r.get('key', '').startswith(PREFIX))
    counts['cai_region_groups_to_regions_junctions'] = apply_rows(
        kit, report, 'cai_region_groups_to_regions_junctions', junctions, lambda r: r.get('region_key', '').startswith(PREFIX))
    for county in design['regions']:
        decisions[county['region']]['cai_region_groups'] = sorted(g['region_group_key'] for g in junctions if g['region_key'] == county['region'])
    adjacency, pairs = adjacency_rows(design, Path(workspace) / 'work/final-map/map.hex')
    counts[ADJACENCY_TABLE] = apply_rows(
        kit, report, ADJACENCY_TABLE, adjacency,
        lambda r: r.get('first_region', '').startswith(PREFIX) or r.get('second_region', '').startswith(PREFIX))
    for (a, b), front in pairs.items():
        for k, other in ((a, b), (b, a)):
            if k in decisions:
                decisions[k].setdefault('river_adjacency', []).append({'region': other, 'facing_hexes': front})
    write_json(Path(workspace) / 'analysis/region-gameplay-db.json', {'counts': counts, 'regions': decisions})
    return counts


def dry_run():
    import campaign_world as cw
    design = json.loads(cw.DESIGN_PATH.read_text(encoding='utf-8'))
    features = region_features(design, cw.WORKSPACE / 'work/final-map/map.hex')
    rows, decisions = effect_bundle_rows(design, features, cw.PAD_LEFT)
    print(json.dumps(rows, ensure_ascii=False, indent=1))
    print(len(rows), 'region_effect_bundles rows;', len({r['region'] for r in rows}), 'regions with a bundle')
    groups, junctions = cai_group_rows(design, rows)
    print(json.dumps([j for j in junctions if not j['region_group_key'].startswith(PREFIX)], indent=1))
    print(len(groups), 'new cai_region_groups;', len(junctions), 'junction rows')
    adjacency, pairs = adjacency_rows(design, cw.WORKSPACE / 'work/final-map/map.hex')
    print(pairs)
    if '--check-battles' in sys.argv:   # every selectable template is a vanilla terrain/battles folder
        from campaign_ground_textures import VanillaPacks
        packs = VanillaPacks()
        found = {k.split('/')[2] for k in packs.keys('terrain/battles/') if k.endswith('/tile_list.bin')}
        layouts = {r.get('battle_layout') for r in design['regions']}
        print('selectable templates missing from the vanilla packs:', sorted((VANILLA_SETTLEMENT_BATTLES | VANILLA_RESOURCE_BATTLES) - found) or 'none')
        print('design battle layouts missing from the vanilla packs:', sorted(layouts - found) or 'none')
    if '--vanilla-road-check' in sys.argv:
        right, total, confusion = vanilla_road_check()
        print('vanilla road grades: rule right for', right, 'of', total, confusion)
    if '--vanilla-check' in sys.argv:   # the river test against vanilla's own rules on the vanilla map
        found = set(river_crossings(cw.VANILLA_MAP, set(vanilla_playable())))
        rules = {tuple(sorted((r['first_region'], r['second_region']))) for r in vanilla_rows(ADJACENCY_TABLE)}
        print('vanilla: found', len(found), 'with a rule', len(found & rules), 'rules', len(rules))


if __name__ == '__main__':
    dry_run()
