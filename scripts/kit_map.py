"""The SDK kit's copy of map.hex, kept identical to work/final-map/map.hex.

Every edit (layout, rebase, region deletion, canvas migration) lands in
final-map/map.hex, but the SDK and the province/database build read the kit
copy. Sync it before either reads it; keep the replaced copy for rollback.
"""
import hashlib, shutil

import campaign_world as _frame

A = _frame.WORKSPACE
KIT_MAP = A / 'work/standalone_kit/raw_data/EmpireDesignData/campaign_maps/3k_dlc07_main_map/map.hex'
FINAL_MAP = A / 'work/final-map/map.hex'


def sync_from_final():
    if KIT_MAP.read_bytes() == FINAL_MAP.read_bytes():
        return False
    back = A / 'terrain-repair/kit-map-before-sync' / hashlib.sha256(KIT_MAP.read_bytes()).hexdigest() / 'map.hex'
    back.parent.mkdir(parents=True, exist_ok=True)
    if not back.exists():
        shutil.copy2(KIT_MAP, back)
    shutil.copy2(FINAL_MAP, KIT_MAP)
    print('Kit map.hex synchronized from final-map', flush=True)
    return True
