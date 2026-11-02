"""Buddhist monastery: a mod copy of the vanilla temple chain for Jibin (Gandhara), with its own names.

Vanilla has one temple chain in use, 3k_resource_earth_temple ('Grand Temple City of Confucius',
孔子本庙), plus Taoist, Yellow Turban and bandit variants in the same superchain. Its loc keys are
shared with the vanilla Pengcheng temple, so Jibin cannot be renamed through them. This module
clones the generic chain under new keys and gives it its own text:

  superchain / chain / slot template  tw3k_resource_earth_buddhist_temple
  levels                              tw3k_resource_earth_buddhist_temple_0 (deserted site)
                                      tw3k_resource_earth_buddhist_temple_1 (built)
  long / short description keys       tw3k_resource_long_description_buddhist_monastery
                                      tw3k_resource_short_description_buddhist_monastery

What is copied (every vanilla row naming the generic temple superchain, chain, slot template or
levels, the keys renamed, nothing else changed; costs, upkeep, prefab and UI positions as vanilla):
building_superchains, building_chains, building_levels, building_culture_variants (the vanilla rows
have no culture, subculture or faction: one variant for every culture, 3k_main_chinese included,
like 508 of vanilla's 635 variant rows), building_upgrades_junction and
building_downgrade_junctions (rows whose two ends are both generic temple levels),
building_level_to_campaign_map_prefab_types, building_level_ui_positions,
building_levels_campaign_bmd_layer_group_junctions, campaign_battle_building_level_tile_upgrades,
siege_obstacles, building_set_to_building_junctions (the copy joins vanilla's sets, the temple set
3k_resource_earth_temple included, so the sets' effects and cost modifiers apply), building_ui_groups_to_superchains, cai_construction_system_building_values,
slot_templates, slot_template_to_building_superchain_junctions, campaign_settlement_display_building_ids
(new keys, vanilla temple models: prefabs/campaign/resource_temple.bmd).
Effects and garrisons: building_effects_junction is keyed by campaign groups; a campaign group
member names one building level (campaign_group_member_criteria_building_levels has the member as
its only key), so every member naming a generic temple level is cloned as 'tw3k_bt_' + member, in
the same group with all its criteria rows (campaign_group_members and every
campaign_group_member_criteria_* table): the groups' effects, garrisons and road objects then apply
to the copy as to the vanilla temple.
Availability: building_chain_availability_sets lists the copy in every set that lists a temple
chain (vanilla's 15 generic sets plus the bandit and Yellow Turban sets), so every owner can use it.
Not copied: the cdir council missions that name the vanilla chain, and the unused Taoist chain.

Text: the DB fields carry the English text, the shipped loc (export_build_tables ->
text/db/tw3k_exp.loc.tsv, which carries the mod's Chinese names) the Chinese.

apply_kit(kit, report) is called by all_directions_build.database() after region_gameplay_db; rows
go to the kit XML (earlier copies replaced) and to report['changed_rows'], and export_build_tables
ships them as '!tw3k_exp' fragments (TABLES are runtime tables there).

  python buddhist_temple_db.py     # print the rows and loc, write nothing
"""
import json

import region_gameplay_db as g

import os
# Stop-loss 2026-10-26 (video package): the romance startpos compile crashed; the temple copy is off unless
# TW3K_BUDDHIST_TEMPLE=1 (Jibin then keeps the vanilla temple chain of its donor). Off, apply_kit removes any
# copy rows an earlier build left in the kit and loc_rows is empty.
ENABLED = os.environ.get('TW3K_BUDDHIST_TEMPLE', '0') == '1'
CHAIN = 'tw3k_resource_earth_buddhist_temple'
LEVELS = (CHAIN + '_0', CHAIN + '_1')
LONG = 'tw3k_resource_long_description_buddhist_monastery'
SHORT = 'tw3k_resource_short_description_buddhist_monastery'
MEMBER_PREFIX = 'tw3k_bt_'
SLOT_JUNCTION_ID = '2140000101'      # above every vanilla id (max 2129784646)
VANILLA = '3k_resource_earth_temple'
KEYMAP = {VANILLA: CHAIN, VANILLA + '_0': LEVELS[0], VANILLA + '_1': LEVELS[1]}
TEMPLE_CHAINS = (VANILLA, '3k_resource_earth_temple_bandit', '3k_ytr_resource_earth_temple_yellow_turban')
# The vanilla key string names a superchain, a chain, a slot template, a building set and a display
# key at once: only these fields are renamed (the building set and the display models stay vanilla's).
FIELDS = {
    'building_superchains': ('key',), 'building_chains': ('key', 'building_superchain'),
    'building_levels': ('level_name', 'chain'), 'building_culture_variants': ('building',),
    'building_upgrades_junction': ('from', 'to'), 'building_downgrade_junctions': ('from', 'to'),
    'building_level_to_campaign_map_prefab_types': ('building_level',), 'building_level_ui_positions': ('building_level',),
    'building_levels_campaign_bmd_layer_group_junctions': ('building_level',),
    'campaign_battle_building_level_tile_upgrades': ('building_level',), 'siege_obstacles': ('building_level',),
    'building_set_to_building_junctions': ('building_chain', 'building_level'),
    'building_ui_groups_to_superchains': ('building_superchain',),
    'cai_construction_system_building_values': ('building_chain', 'building_super_chain', 'building_instance',
                                                'building_or_building_range_start_inclusive', 'building_range_end_inclusive'),
    'slot_templates': ('key',), 'campaign_settlement_display_building_ids': ('key', 'building_level_key'),
    'campaign_group_member_criteria_building_levels': ('building_level',),
}

TEXT = {   # (English in the DB rows, Chinese in the shipped loc)
    'chain': ('Buddhist Monastery', '佛寺'),
    'override_description': ('All conditioned things are impermanent; all things are without self; nirvana is peace.',
                             '诸行无常，诸法无我，涅槃寂静。'),
    LEVELS[0]: ('Deserted Buddhist Monastery Site', '佛寺遗址'),
    LEVELS[1]: ('Grand Sangharama', '大伽蓝'),
    SHORT: ('Monks gather in Jibin, and sutras and images travel east.', '罽宾僧众云集，经像由此东传。'),
    LONG: ('In Jibin and Gandhara the Dharma flourished under the Kushan kings; Kanishka raised great stupas '
           'and gathered the sangha. In the last years of the Han, monks from Parthia and the Yuezhi carried '
           'the sutras east to Luoyang and translated them, and the monastery found its way into the Central Plains.',
           '罽宾、犍陀罗佛法大兴于贵霜诸王之世，迦腻色伽王起大塔、集僧伽。汉末，安息、月氏沙门携经东至洛阳，'
           '译经传法，伽蓝由此渐入中土。'),
}

SIMPLE = ('building_levels', 'building_upgrades_junction', 'building_downgrade_junctions',
          'building_level_to_campaign_map_prefab_types', 'building_level_ui_positions',
          'building_levels_campaign_bmd_layer_group_junctions', 'campaign_battle_building_level_tile_upgrades',
          'siege_obstacles', 'building_set_to_building_junctions', 'building_ui_groups_to_superchains',
          'cai_construction_system_building_values', 'slot_templates', 'campaign_settlement_display_building_ids')


def _criteria_tables():
    return sorted(p.stem for p in g.VANILLA_DB.glob('campaign_group_member_criteria_*.xml'))


def _rename(table, row):
    return {k: (KEYMAP.get(v, v) if k in FIELDS.get(table, ()) else v) for k, v in row.items()}


def _names_temple(table, row):
    return any(row.get(k) in KEYMAP for k in FIELDS[table])


def rows():
    """{table: [row dicts]} of the copy, sorted by primary key."""
    out = {}
    sup = [r for r in g.vanilla_rows('building_superchains') if r['key'] == VANILLA]
    out['building_superchains'] = [dict(_rename('building_superchains', sup[0]), name='Buddhist Monastery')]
    ch = [r for r in g.vanilla_rows('building_chains') if r['key'] == VANILLA]
    out['building_chains'] = [dict(_rename('building_chains', ch[0]), chain_tooltip=TEXT['chain'][0],
                                   encyclopedia_name=TEXT['chain'][0], override_description=TEXT['override_description'][0])]
    for table in SIMPLE:
        src = [r for r in g.vanilla_rows(table) if _names_temple(table, r)]
        if table in ('building_upgrades_junction', 'building_downgrade_junctions'):
            src = [r for r in src if r['from'] in KEYMAP and r['to'] in KEYMAP]
        out[table] = [_rename(table, r) for r in src]
    variants = []
    for r in g.vanilla_rows('building_culture_variants'):
        if r['building'] in KEYMAP:
            new = _rename('building_culture_variants', r)
            new['name'] = TEXT[new['building']][0]
            if new['building'] == LEVELS[1]:
                new['description'], new['short_description'] = LONG, SHORT
            variants.append(new)
    out['building_culture_variants'] = variants
    out['building_description_texts'] = [{'key': LONG, 'long_description': TEXT[LONG][0]}]
    out['building_short_description_texts'] = [{'key': SHORT, 'short_description': TEXT[SHORT][0]}]
    sets = sorted({r['id'] for r in g.vanilla_rows('building_chain_availability_sets') if r['building_chain'] in TEMPLE_CHAINS})
    out['building_chain_availability_sets'] = [{'building_chain': CHAIN, 'id': s} for s in sets]
    out['slot_template_to_building_superchain_junctions'] = [
        {'building_superchain': CHAIN, 'id': SLOT_JUNCTION_ID, 'slot_template': CHAIN}]
    # Campaign group members naming a generic temple level: cloned with all their criteria.
    members = sorted({r['member'] for r in g.vanilla_rows('campaign_group_member_criteria_building_levels')
                      if r['building_level'] in KEYMAP})
    mset = set(members)
    out['campaign_group_members'] = [dict(r, id=MEMBER_PREFIX + r['id'])
                                     for r in g.vanilla_rows('campaign_group_members') if r['id'] in mset]
    for table in _criteria_tables():
        src = [r for r in g.vanilla_rows(table) if r.get('member') in mset]
        if src:
            out[table] = [dict(_rename(table, r), member=MEMBER_PREFIX + r['member']) for r in src]
    for table, rs in out.items():
        keys = g.primary_keys(table)
        rs.sort(key=lambda r: tuple(r.get(k, '') for k in keys))
        assert len({tuple(r.get(k, '') for k in keys) for r in rs}) == len(rs), table
        vanilla_keys = {tuple(r.get(k, '') for k in keys) for r in g.vanilla_rows(table)}
        assert not vanilla_keys & {tuple(r.get(k, '') for k in keys) for r in rs}, ('copy collides with vanilla', table)
        assert all(set(r) == set(next(iter(g.vanilla_rows(table)), r)) for r in rs), table
    assert {r['building'] for r in out['building_culture_variants']} == set(LEVELS)
    assert len(out['campaign_group_members']) == len(members)
    return dict(sorted(out.items()))


def ours(table):
    """Predicate: a kit row (dict) written by this module."""
    def f(r):
        return any(str(v).startswith((CHAIN, MEMBER_PREFIX)) or v in (LONG, SHORT) for v in r.values()) \
            or (table == 'slot_template_to_building_superchain_junctions' and r.get('id') == SLOT_JUNCTION_ID)
    return f


TABLES = ('building_superchains', 'building_chains', 'building_culture_variants',
          'building_description_texts', 'building_short_description_texts', 'building_chain_availability_sets',
          'slot_template_to_building_superchain_junctions', 'campaign_group_members') + SIMPLE \
    + tuple(t for t in ('campaign_group_member_criteria_booleans', 'campaign_group_member_criteria_building_levels',
                        'campaign_group_member_criteria_campaigns', 'campaign_group_member_criteria_factions',
                        'campaign_group_member_criteria_subcultures'))


# The two text tables exist in the Assembly Kit only (no runtime pack table, no RPFM schema): their
# rows stay in the kit; the game reads the text through the loc keys below.
TEXT_TABLES = ('building_description_texts', 'building_short_description_texts')
RUNTIME_TABLES = tuple(t for t in TABLES if t not in TEXT_TABLES)
assert len(set(TABLES)) == len(TABLES)


def loc_rows():
    """(key, Chinese text, tooltip flag) rows for text/db/tw3k_exp.loc.tsv."""
    if not ENABLED:
        return []
    return [('building_chains_chain_tooltip_' + CHAIN, TEXT['chain'][1], 'false'),
            ('building_chains_encyclopedia_name_' + CHAIN, TEXT['chain'][1], 'false'),
            ('building_chains_override_description_' + CHAIN, TEXT['override_description'][1], 'false'),
            ('building_culture_variants_name_' + LEVELS[0], TEXT[LEVELS[0]][1], 'false'),
            ('building_culture_variants_name_' + LEVELS[1], TEXT[LEVELS[1]][1], 'false'),
            ('building_description_texts_long_description_' + LONG, TEXT[LONG][1], 'false'),
            ('building_short_description_texts_short_description_' + SHORT, TEXT[SHORT][1], 'false')]


def apply_kit(kit, report):
    if not ENABLED:   # stop-loss: take earlier copies out of the kit, add none
        return {t: g.apply_rows(kit, report, t, [], ours(t)) for t in TABLES}
    new = rows()
    assert set(new) == set(TABLES), sorted(set(new) ^ set(TABLES))
    return {t: g.apply_rows(kit, report, t, new[t], ours(t)) for t in TABLES}


if __name__ == '__main__':
    r = rows()
    print(json.dumps(r, ensure_ascii=False, indent=1))
    print({t: len(v) for t, v in r.items()})
    for e in loc_rows():
        print('\t'.join(e))
