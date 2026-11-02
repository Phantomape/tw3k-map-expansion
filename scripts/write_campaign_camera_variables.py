"""Scale the campaign camera's zoom-out limits with the expanded world.

campaign_variables holds the camera's maximum height, maximum pull-back distance
and far clip plane, tuned for the vanilla 595x541 world. On the expanded world the
farthest zoom no longer shows the whole map. This writes a release table fragment
(release-input/db/campaign_variables_tables/!tw3k_exp.tsv) with only those rows,
scaled by the world-size ratio; every other campaign variable stays vanilla.
"""
import xml.etree.ElementTree as ET

import campaign_world as cw

VANILLA_WORLD = (595.0999755859375, 541.2066650390625)   # native g_world_bounds
KEYS = ('camera_angle_dependent_on_zoom_maximum_height',
        'camera_angle_dependent_on_zoom_maximum_distance',
        'camera_angle_dependent_on_zoom_far_z')
A = cw.WORKSPACE
KIT_TABLE = A / 'work/standalone_kit/raw_data/db/campaign_variables.xml'
OUT = A / 'work/release-input/db/campaign_variables_tables/!tw3k_exp.tsv'


def main():
    scale = max(cw.WORLD_WIDTH / VANILLA_WORLD[0], cw.WORLD_HEIGHT / VANILLA_WORLD[1])
    rows = {r.findtext('variable_key'): float(r.findtext('value'))
            for r in ET.parse(KIT_TABLE).getroot().findall('campaign_variables')}
    missing = [k for k in KEYS if k not in rows]
    if missing:
        raise RuntimeError(f'campaign_variables lacks {missing}')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    lines = ['variable_key\tvalue', '#campaign_variables_tables;0;db/campaign_variables_tables/!tw3k_exp\t']
    lines += [f'{k}\t{rows[k] * scale:.4f}' for k in KEYS]
    OUT.write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
    print(f'campaign camera limits scaled x{scale:.3f}:', {k.rsplit("_zoom_", 1)[1]: round(rows[k] * scale, 2) for k in KEYS})


if __name__ == '__main__':
    main()
