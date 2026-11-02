"""Landmark DB rows, label positions and loc (landmarks_spec.py).

Called by all_directions_build.database() right after the dlc07 label rows are moved
with the native map (762d7b7): it appends the landmark rows to the kit XML
(campaign_areas_of_interest, campaign_map_areas_of_interest,
campaign_map_areas_of_interest_label_positions; earlier landmark rows are replaced)
and adds them to standalone-prototype.json changed_rows, so export_build_tables ships
them as keyed '!tw3k_exp' fragments over the vanilla tables (all three are in its
RUNTIME_TABLES). Label positions are canvas-frame logical hexes already: they are not
shifted again. export_build_tables writes loc_rows() into text/db/tw3k_exp.loc.tsv
(keys campaign_areas_of_interest_{title,description}_<key>, as vanilla's loc).

Reads:  landmarks_spec; kit raw_data/db XML of the three tables
Writes: the same kit XML; the report's changed_rows (caller dumps it)

  python landmarks_db.py      # print the rows (dry run, writes nothing)
"""
import json
import xml.etree.ElementTree as ET

import landmarks_spec as spec

TABLES = ('campaign_areas_of_interest', 'campaign_map_areas_of_interest', 'campaign_map_areas_of_interest_label_positions')


def rows():
    """{table: [ordered dict rows]} for every landmark AOI key."""
    aoi, mapped, labels = [], [], []
    n = 0
    for lm in spec.LANDMARKS:
        for key in spec.aoi_keys(lm):
            aoi.append({'area_of_interest': key, 'description': lm['description'], 'group': spec.GROUP,
                        'icon': spec.ICON.format(lm['icon']), 'icon_height_offset': '0', 'key': key,
                        'spline_colour': '', 'spline_material': '', 'title': lm['title']})
            mapped.append({'campaign_map': spec.CAMPAIGN_MAP, 'key': key})
        for key, (q, r) in spec.label_positions(lm):
            labels.append({'campaign_map_area_of_interest': key, 'key': str(spec.LABEL_KEY_BASE + n),
                           'logical_position_x': str(q), 'logical_position_y': str(r)})
            n += 1
    return {TABLES[0]: aoi, TABLES[1]: mapped, TABLES[2]: labels}


def loc_rows():
    out = []
    for lm in spec.LANDMARKS:
        for key in spec.aoi_keys(lm):
            out.append(('campaign_areas_of_interest_title_' + key, lm['title'], 'false'))
            out.append(('campaign_areas_of_interest_description_' + key, lm['description'], 'false'))
    return out


def _ours(row, table):
    field = 'campaign_map_area_of_interest' if table == TABLES[2] else 'key'
    return (row.findtext(field) or '').startswith(spec.PREFIX)


def apply_kit(kit, report):
    """Append the rows to the kit XML (replacing earlier landmark rows) and record them in
    report['changed_rows']. The label table's changed rows keep the moved vanilla dlc07 rows."""
    new = rows()
    for table in TABLES:
        path = kit / f'raw_data/db/{table}.xml'
        doc = ET.parse(path)
        root = doc.getroot()
        for r in [r for r in root.findall(table) if _ours(r, table)]:
            root.remove(r)
        for row in new[table]:
            el = ET.SubElement(root, table)
            for k, v in row.items():
                ET.SubElement(el, k).text = v
        doc.write(path, encoding='utf-8', xml_declaration=True)
        changed = report.setdefault('changed_rows', {})
        kept = [r for r in changed.get(table, []) if not (r.get('campaign_map_area_of_interest' if table == TABLES[2] else 'key') or '').startswith(spec.PREFIX)]
        changed[table] = kept + new[table]
    return {t: len(v) for t, v in new.items()}


if __name__ == '__main__':
    r = rows()
    print(json.dumps({t: len(v) for t, v in r.items()}))
    print(json.dumps(r, ensure_ascii=False, indent=1))
    print(len(loc_rows()), 'loc rows')
