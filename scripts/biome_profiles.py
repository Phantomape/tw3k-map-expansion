"""Biome profile table: the landform, biome and tuning constants of the real-terrain ground (polish blend),
frontier trees, logical forest ground and gameplay climate, in one place (design and inventory:
analysis/biome-profile-table-design.md). Data only, no numpy at import: every consumer keeps its old
module-level names as reads from here (ground_materials.MIX = mixes(), ...), so values, types and dict
order are exactly the literals they replaced. Accessors return deep copies (no consumer can mutate the
table for another). dump_json() writes analysis/biome-profiles.json for review; the pipeline never reads it.

polish's SETUP MEMO key hashes scripts/*.py, so an edit here invalidates the memo by itself.
"""
import copy
import json

# Shared semantic group: the existing polish woodland floors, cold_1 and cold_2.
# The cold material family (4-7) also contains lighter open/buffer ground (6, 7).
WOODLAND_SLOTS = (4, 5)

SCHEMA = 1
# build_west_real_terrain.BIOME_CODES (the RESOLVE class order of west-real-terrain.npz biome_names)
BIOME_CODES = ('desert', 'steppe', 'grass', 'alpine', 'tundra', 'snow', 'forest', 'conifer', 'taiga',
               'temperate', 'tropical', 'tropical_dry', 'savanna', 'wetland', 'paddy', 'mangrove', 'other')

# ------------------------------------------------------------------ landforms (ground_materials classes)
# Insertion order = tile_fields' evaluation order (its class dict and CLASS_GAIN; P is a float32 sum in
# this order): never re-sort. 'design' names the landform of the design table (§2.2).
# 'river_oasis' (owner review item 3, unused yet): oasis growth along river hexes after the next river
# round (Konqi, Tarim, Shule, Dang, Hei): spread distance and belt width from river hexes, per landform.
# Mixes (material slot -> share), in the stored order (tile_fields adds them in MIX[c].items() order).
# Vanilla Qilian > 3000 m on slopes: 4:81 5:8 6:7 7:4; a little steppe_4 breaks the single dark sheet
# the cold donor laid (offline 2048bc).
# [REALISM 2026-10-24] (analysis/terrain-realism/README.md): steppe_4 is olive green (mean 134,126,45)
# and read as grass flecks on bare desert (vanilla real desert: steppe_4 .3 %, ours 14.2 %; mud .3 vs
# 5.3 %). Desert mixes take vanilla's flat Hexi gobi (arid_2 47, arid_4 19, arid_3 16, arid_1 13 %)
# and hardly any mud; plateau and tundra carry less mud; steppe gravel keeps the steppe greens.
# [GROUND MOTTLE step 1, user decision 2026-11-01] recipe v24 (the default): each mix 2-4 materials of close tone (pairwise
# dE76 <= 20 at harvest, expected pair contrast no higher than v23's), no mud (14) in the landform mixes (mud stays in
# the MUD pass on river banks and lake shores), own valley / slope / crest relief for plateau, alpine_meadow and
# drymtn, arid towns' own fields ARID_OWN_OASIS 6 -> 2. Tundra keeps its recipe (its own switches TW3K_TUNDRA_SCREE /
# TW3K_TUNDRA_BAN). The v23 values: GROUND_RECIPE_V23, TW3K_GROUND_RECIPE=v23.
# [r5, user decision 2026-11-01] material blobs by landform ('grain', hexes; the rest GRAIN[1] = 3, ground_materials joins
# the grains by a Gaussian copula, seamless at boundaries): tundra 3 (its frozen A2 gates: a global 6 joined its pale
# materials into sheets), plateau / high / gobi / yardang 7, erg / salt_crust 6, the meadows, drymtn, sgravel, alpine 5.
# High ground olive and brown-grass, dark only on crests (the user: 'still deep dark spots' near Yalong): high and alpine
# without cold_1, high's own valley / slope / crest relief (crest cold_3 / cold_4); 'redraw': the dark cold donors
# (cold_1-3) on plateau, alpine_meadow and high are taken by the landform's mix with its class weight (graded edge; the
# step-6 ban drew tile blocks there). Replica (<scratch>/mottle-20261101): see results.md.
LANDFORMS = {
    'high':    {'design': 'cold_high', 'mix': {7: .50, 22: .35, 21: .15}, 'gain': 1., 'grain': 7., 'redraw': (4, 5, 6),
                'relief': {'blend': .7, 'mix': {'valley': {22: .45, 21: .35, 7: .20}, 'slope': {7: .50, 22: .30, 21: .20},
                                                'crest': {6: .50, 7: .50}}},
                'river_oasis': {'spread_hex': None, 'width_hex': None}},
    'plateau': {'design': 'cold_plateau', 'mix': {2: .45, 3: .25, 20: .20, 21: .10}, 'gain': .92, 'grain': 7., 'redraw': (4, 5, 6),
                'relief': {'blend': .7, 'mix': {'valley': {21: .60, 20: .25, 27: .15}, 'slope': {20: .50, 2: .30, 21: .20},
                                                'crest': {2: .60, 3: .40}}},
                'river_oasis': {'spread_hex': None, 'width_hex': None}},
    # [§9.3 tundra] own relief: the global RELIEF parts painted tundra valleys temperate_2 (valley_humid), banned
    # .456 in the eighteenth pack (sixteenth .034); valleys dark olive (steppe_4/3, cold_4), slopes brown-grey, crests
    # scree, no temperate slot (replica 2026-10-30: temperate share on tundra-enforced pixels .40-.64 -> 0-.017)
    'tundra':  {'design': 'tundra', 'mix': {14: .10, 7: .30, 22: .20, 21: .18, 2: .14, 5: .08}, 'gain': .9, 'grain': 3.,
                'relief': {'blend': .5, 'mix': {'valley': {22: .45, 21: .30, 7: .25}, 'slope': {21: .40, 7: .35, 14: .25},
                                                'crest': {2: .45, 3: .30, 7: .25}}},
                'river_oasis': {'spread_hex': None, 'width_hex': None}},
    # [BEAUTIFY G] basin floors light gravel plain, slopes and crests dark hamada, graded by the relief and a
    # ~40-hex low-frequency noise (no regular shapes at one height)
    'gobi':    {'design': 'gobi', 'mix': {1: .55, 3: .25, 2: .20}, 'gain': .88, 'grain': 7.,
                'relief': {'blend': .5, 'noise': (40., .35, 213),
                           'mix': {'valley': {1: .45, 0: .25, 2: .20, 3: .10}, 'slope': {3: .45, 2: .40, 1: .15},
                                   'crest': {3: .55, 2: .45}}},
                'river_oasis': {'spread_hex': None, 'width_hex': None}},
    'drymtn':  {'design': 'drymtn (+ alpine_dry)', 'mix': {2: .42, 3: .38, 1: .20}, 'gain': .9, 'grain': 5.,
                'relief': {'blend': .5, 'mix': {'valley': {1: .60, 19: .40}, 'slope': {2: .50, 3: .50}, 'crest': {3: .60, 2: .40}}},
                'river_oasis': {'spread_hex': None, 'width_hex': None}},
    'sgravel': {'design': 'steppe_gravel', 'mix': {21: .45, 22: .35, 20: .20}, 'gain': .45, 'grain': 5.,
                'river_oasis': {'spread_hex': None, 'width_hex': None}},
    # [MATERIALS ECOTONE] montane meadow below the cold line: vanilla's 'alpine' ground is
    # cold_1 47, cold_2 8, steppe_3 8, cold_4 6, steppe_4 5, cold_3 5 % (not sand); lighter
    # here (mid luminance ~110-130) so it bridges conifer (cold_1, L74) and steppe.
    'alpine':  {'design': 'alpine_wet', 'mix': {7: .45, 22: .35, 21: .20}, 'gain': .85, 'grain': 5.,
                'river_oasis': {'spread_hex': None, 'width_hex': None}},
    # sand seas: [BEAUTIFY G] the dune sand takes the erg mix too (Taklamakan arid_1 was .73 against vanilla's
    # 90-95 %), sunlit crests brightest, interdune hollows a little darker
    'erg':     {'design': 'erg', 'mix': {0: .75, 1: .15, 2: .10}, 'gain': 1., 'grain': 6.,
                'relief': {'blend': .3, 'noise': (30., .3, 214),
                           'mix': {'valley': {1: .6, 0: .3, 2: .1}, 'slope': {0: .85, 1: .15}, 'crest': {0: 1.}}},
                'river_oasis': {'spread_hex': None, 'width_hex': None}},
    # [BEAUTIFY H] Tibet: humid alpine meadow (south-east, brown-green) and dry alpine steppe (Changtang,
    # brown-yellow); both before plateau/alpine/the dry cold band, after high (classify)
    'alpine_meadow': {'design': 'alpine_meadow', 'mix': {21: .50, 22: .25, 20: .25}, 'gain': .9, 'grain': 5., 'redraw': (4, 5, 6),
                      'relief': {'blend': .5, 'mix': {'valley': {27: .40, 21: .40, 28: .20}, 'slope': {21: .55, 22: .30, 20: .15},
                                                      'crest': {22: .40, 7: .30, 21: .30}}},
                      'river_oasis': {'spread_hex': None, 'width_hex': None}},
    # Changtang green target < .10 (replica r1: .35, steppe_3 and the donors' cold_4/steppe_4 left)
    'alpine_steppe': {'design': 'alpine_steppe', 'mix': {19: .40, 2: .35, 20: .25}, 'gain': .92, 'grain': 5.,
                      'relief': {'blend': .4, 'mix': {'valley': {19: .50, 20: .30, 21: .20}, 'slope': {19: .50, 2: .30, 20: .20},
                                                      'crest': {2: .50, 3: .50}}},
                      'river_oasis': {'spread_hex': None, 'width_hex': None}},
    # [BEAUTIFY G] salt crusts (REGIONS salt_crusts) and yardang fields (REGIONS yardang); salt first
    'salt_crust': {'design': 'salt_crust', 'mix': {0: .75, 1: .25}, 'gain': 1., 'grain': 6.,
                   'river_oasis': {'spread_hex': None, 'width_hex': None}},
    # brown sandstone bands without mud blobs (mud read as dark specks in the replica)
    'yardang': {'design': 'yardang', 'mix': {3: .50, 2: .35, 1: .15}, 'gain': .95, 'grain': 7.,
                'relief': {'blend': .5, 'mix': {'valley': {1: .7, 2: .3}, 'slope': {3: .50, 2: .35, 1: .15}, 'crest': {3: .7, 2: .3}}},
                'river_oasis': {'spread_hex': None, 'width_hex': None}},
}
MIX_ORDER = ('high', 'plateau', 'tundra', 'gobi', 'drymtn', 'sgravel', 'erg', 'alpine', 'alpine_meadow', 'alpine_steppe', 'salt_crust', 'yardang')   # ground_materials.MIX key order (erg before alpine)

# ------------------------------------------------------------------ audit fields; 'banned' enforced (step 6)
# Material families (blend slots) and, per landform, share bounds and banned slots for the regression
# report (design §4). Bounds stay audit only. 'banned' is enforced since step 6 (2026-10-25):
# ground_materials.apply_rows redraws from the landform's mix every pixel whose material is banned for the
# landform that dominates its tile (class weight x real ground x not-oasis > BAN_MIN; dominant(), ban_table()).
# TW3K_ENFORCE_BANNED=0 turns it off for comparisons. Bounds are (lo, hi) shares of tile-centre materials on the
# landform's real ground (weight > .5); None = no measured reference yet. Sources: the vanilla surveys
# quoted in ground_materials / analysis/terrain-realism/README.md.
FAMILIES = {'arid': (0, 1, 2, 3), 'cold': (4, 5, 6, 7), 'farm': (8, 9, 10, 11, 12), 'road': (13,), 'mud': (14,),
            'steppe_light': (19, 20), 'steppe_dark': (21, 22), 'subtropical': (23, 24, 25, 26), 'temperate': (27, 28, 29, 30),
            'wetland': (31,), 'green': (4, 5, 6, 7, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30)}
AUDIT = {
    # Qilian > 3000 m (vanilla): cold_1 76, cold_2 9, cold_3 8, cold_4 6 %
    'high':    {'bounds': {'cold': (.85, 1.), 'subtropical': (0, .01)}, 'banned': (23, 24, 25, 26, 27, 28, 29, 30)},
    'plateau': {'bounds': {'mud': (0, .12), 'subtropical': (0, .01)}, 'banned': (23, 24, 25, 26)},
    # tundra: the audit's cold_tropic rule (no subtropical/temperate on tundra)
    'tundra':  {'bounds': {'mud': (0, .12)}, 'banned': (23, 24, 25, 26, 27, 28, 29, 30)},
    # flat Hexi gobi (vanilla): arid_2 47, arid_4 19, arid_3 16, arid_1 13 %; real desert steppe_4 .3 %, mud .3 %;
    # fields outside oases (0, .03), mud (0, .005) of land (vanilla .35 %)
    # [STEP 6] desert landforms: every green slot is banned outside oases (the take dither left donor
    # greens as specks: r16 gobi 939, drymtn 664 green speck components); drymtn keeps its own steppe_3.
    'gobi':    {'bounds': {'arid': (.85, 1.), 'steppe_dark': (0, .01), 'green': (0, .01), 'mud': (0, .08), 'farm': (0, .03)},
                'banned': (4, 5, 6, 7, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30)},
    'drymtn':  {'bounds': {'green': (0, .15), 'mud': (0, .05)}, 'banned': (4, 5, 6, 7, 22, 23, 24, 25, 26, 27, 28, 29, 30)},
    'sgravel': {'bounds': None, 'banned': None},
    # vanilla 'alpine' ground: cold_1 47, cold_2 8, steppe_3 8, cold_4 6, steppe_4 5, cold_3 5 %
    'alpine':  {'bounds': {'cold': (.5, .8), 'steppe_dark': (.05, .25), 'arid': (0, .1)}, 'banned': None},
    # sand seas (Badain Jaran, Alxa, vanilla): arid_1 90-95 %
    'erg':     {'bounds': {'arid': (.9, 1.), 'green': (0, .01), 'mud': (0, .02), 'farm': (0, .01)},
                'banned': (4, 5, 6, 7, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30)},
    'alpine_meadow': {'bounds': {'cold': (0, .30), 'steppe_dark': (.25, .60)}, 'banned': (4, 5, 23, 24, 25, 26)},
    'alpine_steppe': {'bounds': {'green': (0, .10), 'arid': (.3, .7)}, 'banned': (4, 5, 6, 7, 22, 23, 24, 25, 26, 27, 28, 29, 30)},
    'salt_crust': {'bounds': {'arid': (.9, 1.)}, 'banned': tuple(range(4, 8)) + tuple(range(19, 31))},
    'yardang': {'bounds': None, 'banned': tuple(range(4, 8)) + tuple(range(21, 31))},
}
# Landforms of the design table not in LANDFORMS yet (step 7): paddy_plain bans 0-3 (sand on paddy plains).
AUDIT_PENDING = {'paddy_plain': {'banned': (0, 1, 2, 3)}}
BAN_MIN = .5      # dominant landform weight above which its banned slots are redrawn (step 6)

# ------------------------------------------------------------------ biomes (RESOLVE classes)
# One record per class of BIOME_CODES. 'climate': polish tile climate (_clim: 0 subtropical, 1 cold,
# 2 temperate, 3 arid); 'gameplay': apply_climate's map.hex climate in the Western Regions box
# (POLISH_CLIMATE); 'tree_fill': frontier_forest FOREST (trees per hex, MIX key); 'sets': {flag: rank},
# one flag per consumer group (SETS), rank = the member's position in that consumer's own tuple/dict
# (order matters: argmax ties in polish _edom and ecotone _dominant, noise seeds 150 + k in alpine_dither).
BIOMES = {
    'desert':       {'climate': 3, 'gameplay': 'arid',
                     'sets': {'polish_climate': 0, 'alpine_peer': 2, 'gm_dry_high': 0, 'farm_dry': 0, 'farm_admit': 8, 'farm_hum_dry': 0,
                              'ff_dry': 2, 'shrub': 0, 'gameplay_polish': 0, 'gc_arid': 0, 'mirror_open': 0, 'real_desert': 0}},
    'steppe':       {'climate': 3, 'gameplay': 'arid',
                     'sets': {'polish_climate': 1, 'ecotone_open': 0, 'polish_steppe_donor': 0, 'alpine_peer': 0, 'gm_dry_high': 4, 'farm_dry': 1,
                              'farm_admit': 9, 'farm_hum_dry': 1, 'ff_steppe': 0, 'ff_dry': 3, 'fol_sparse': 0, 'shrub': 1, 'gameplay_polish': 1,
                              'mirror_open': 1, 'real_desert': 1}},
    'grass':        {'climate': 3, 'gameplay': 'arid',
                     'sets': {'polish_climate': 2, 'ecotone_open': 1, 'polish_steppe_donor': 1, 'alpine_peer': 3, 'gm_dry_high': 5, 'farm_admit': 2,
                              'ff_steppe': 1, 'shrub': 4, 'gameplay_polish': 2, 'gc_sub31': 0, 'mirror_open': 2}},
    'alpine':       {'climate': 3, 'gameplay': 'arid',
                     'sets': {'polish_climate': 3, 'gm_dry_high': 1, 'arid_alpine': 0, 'fol_rock_high': 2, 'fol_sparse': 1, 'shrub': 2,
                              'gameplay_polish': 3, 'gc_arid': 1}},
    'tundra':       {'climate': 3, 'gameplay': 'arid',
                     'sets': {'polish_climate': 4, 'gm_dry_high': 3, 'fol_rock_high': 1, 'fol_sparse': 2, 'shrub': 3,
                              'gameplay_polish': 4, 'gc_cold': 1, 'no_oasis': 0, 'no_alpine_belt': 0}},
    'snow':         {'climate': 3, 'gameplay': 'arid',
                     'sets': {'polish_climate': 5, 'alpine_peer': 1, 'gm_dry_high': 2, 'arid_alpine': 1, 'fol_rock_high': 0, 'gameplay_polish': 5,
                              'gc_cold': 0, 'no_oasis': 1}},
    'forest':       {'climate': 2, 'gameplay': 'temperate', 'tree_fill': (.82, 'temperate'),
                     'sets': {'polish_climate': 8, 'ecotone_forest': 2, 'treeline_forest': 2, 'sub_forest': 2, 'farm_admit': 0, 'tree_fill': 2,
                              'forest_ground': 2, 'gameplay_polish': 8, 'gc_sub31': 4}},
    'conifer':      {'climate': 1, 'gameplay': 'cold', 'tree_fill': (.85, 'cold_mixed'),
                     'sets': {'polish_climate': 6, 'ecotone_forest': 1, 'treeline_forest': 1, 'sub_forest': 4, 'farm_admit_half': 0, 'fol_fir': 1,
                              'tree_fill': 1, 'forest_ground': 1, 'gameplay_polish': 6, 'fill_floor': 1}},
    'taiga':        {'climate': 1, 'gameplay': 'cold', 'tree_fill': (.85, 'cold'),
                     'sets': {'polish_climate': 7, 'ecotone_forest': 0, 'treeline_forest': 0, 'sub_forest': 5, 'fol_fir': 0, 'tree_fill': 0,
                              'forest_ground': 0, 'gameplay_polish': 7, 'gc_cold': 2, 'fill_floor': 0}},
    'temperate':    {'climate': 2, 'gameplay': 'temperate',
                     'sets': {'polish_climate': 9, 'ecotone_forest': 3, 'treeline_forest': 3, 'sub_forest': 3, 'farm_admit': 1, 'gameplay_polish': 9,
                              'gc_sub31': 2}},
    'tropical':     {'climate': 0, 'gameplay': 'subtropical', 'tree_fill': (.82, 'subtropical'),
                     'sets': {'polish_climate': 11, 'ecotone_forest': 4, 'sub_forest': 0, 'farm_admit': 3, 'fol_subtropical': 0, 'tree_fill': 3,
                              'forest_ground': 3, 'gameplay_polish': 11, 'gc_tropical': 0}},
    'tropical_dry': {'climate': 0, 'gameplay': 'subtropical', 'tree_fill': (.50, 'dry'),
                     'sets': {'polish_climate': 12, 'ecotone_forest': 5, 'sub_forest': 1, 'farm_admit': 4, 'farm_hum_dry': 2, 'ff_dry': 0, 'fol_dry': 0,
                              'tree_fill': 5, 'gameplay_polish': 12, 'gc_tropical': 1}},
    'savanna':      {'climate': 0, 'gameplay': 'subtropical',
                     'sets': {'polish_climate': 14, 'ecotone_open': 2, 'farm_admit': 5, 'farm_hum_dry': 3, 'ff_dry': 1, 'fol_dry': 1, 'shrub': 5,
                              'gameplay_polish': 14, 'gc_tropical': 2}},
    'wetland':      {'climate': 3, 'gameplay': 'arid',
                     'sets': {'polish_climate': 10, 'polish_wet': 0, 'farm_admit': 7, 'gameplay_polish': 10, 'gc_sub31': 3}},
    'paddy':        {'climate': 0, 'gameplay': 'subtropical',
                     'sets': {'polish_climate': 15, 'sub_paddy': 0, 'farm_admit': 6, 'gameplay_polish': 15, 'gc_tropical': 3}},
    'mangrove':     {'climate': 0, 'gameplay': 'subtropical', 'tree_fill': (.60, 'subtropical'),
                     'sets': {'polish_climate': 13, 'sub_paddy': 1, 'fol_subtropical': 1, 'tree_fill': 4, 'forest_ground': 4, 'gameplay_polish': 13,
                              'gc_tropical': 4}},
    # no tile climate (keeps the authored band in polish) and no POLISH_CLIMATE entry (OTHER_DEFAULT)
    'other':        {'climate': None, 'gameplay': None, 'sets': {'gc_sub31': 1}},
}
# The flags: consumer, what the group decides, and where today's members disagree with a sibling
# group. The disagreements are kept exactly during the migration (owner review 2026-10-24: each later
# unification is its own commit with a before/after comparison).
SETS = {
    'polish_climate': "polish _clim: real-ground tile climate (the record's climate)",
    'ecotone_open': 'polish USR3 ECOTONE _eopen: open side of the open/forest redraw',
    'ecotone_forest': 'polish USR3 ECOTONE _efor: forest side (with tropical/tropical_dry, no mangrove; treeline_forest has no tropics)',
    'polish_steppe_donor': 'polish real_steppe: steppe donor (9) where the tile climate is arid',
    'polish_wet': 'polish real_wet: wetland belts (oasis .9, alpine .6)',
    'fill_floor': 'biome_profiles.fill_floor: FOREST_FILL_FLOOR of the far forest fill (TW3K_FOREST_FILL_FLOOR; ground and tree fill)',
    'no_oasis': 'polish oasis belts (river, wetland, lake ring) never on these classes, nor on grass north of TUNDRA_GRASS_LAT (TW3K_TUNDRA_OASIS_OFF)',
    'no_alpine_belt': 'polish alpine (keep) belts never on these classes, nor on grass north of TUNDRA_GRASS_LAT (TW3K_TUNDRA_OASIS_OFF)',
    'treeline_forest': 'ecotone_bands FOREST: forest side of the treeline (taiga/conifer/forest/temperate only)',
    'alpine_peer': 'ecotone_bands ALPINE_PEERS: classes dithered against alpine (seed 150 + rank)',
    'gm_dry_high': 'ground_materials tile_fields dry_high: weight of the high/plateau (cold band) classes',
    'farm_dry': 'ground_materials farm_mud_fields dry: watered-only farming (desert, steppe; farm_hum_dry adds tropical_dry/savanna)',
    'farm_admit': 'ground_materials farm_mud_fields admit 1 (alpine/tundra/snow/taiga/other admit 0)',
    'farm_admit_half': 'ground_materials farm_mud_fields admit .5 (conifer)',
    'farm_hum_dry': 'ground_materials farm_mud_fields hum: 1 - blurred share of these (tea humidity); same members as ff_dry',
    'sub_paddy': 'oasis_ground SUB_PADDY: subtropical paddy-lowland donor (10)',
    'sub_forest': 'oasis_ground SUB_FOREST: subtropical forest share keeps donor list 0 (ecotone_forest plus conifer/taiga)',
    'arid_alpine': 'oasis_ground arid_share: alpine/snow counted arid times dryness',
    'ff_steppe': 'farm_fields steppe_ids: pastoral steppe towns (STEPPE_SCALE)',
    'ff_dry': 'farm_fields dry_ids: no tea field donors on these',
    'fol_rock_high': 'foliage_biomes choose: rock above surface 3.5 (else 4.8)',
    'fol_fir': 'foliage_biomes choose: dense firs',
    'fol_subtropical': 'foliage_biomes choose: bamboo/castanopsis forest',
    'fol_dry': 'foliage_biomes choose: sparse castanopsis',
    'fol_sparse': 'foliage_biomes choose: .008 (fir on alpine/tundra, katsura on steppe)',
    'tree_fill': "frontier_forest FOREST: one tree per hex at the record's tree_fill (no temperate, unlike treeline_forest)",
    'shrub': 'frontier_forest OPEN_BIOMES: rock shrubs',
    'forest_ground': 'apply_forest_ground FOREST_BIOMES: light/dense forest ground (tree_fill without tropical_dry)',
    'gameplay_polish': "apply_climate POLISH_CLIMATE: Western Regions map.hex climate (the record's gameplay; other -> OTHER_DEFAULT)",
    'gc_tropical': 'apply_climate classify tropical: subtropical; cold at >= 27 N and >= 2000 m',
    'gc_sub31': 'apply_climate classify: subtropical south of 31 N',
    'gc_arid': 'apply_climate classify: arid (vanilla desert 75 %, montane grassland 81 %)',
    'gc_cold': 'apply_climate classify: always cold',
    'mirror_open': 'polish mirror band: real open ground where mirrored vanilla woodland (WOODLAND_MATERIALS) takes the canvas donor',
    'real_desert': 'polish real climate: desert and steppe override the carried native climate from REAL_DESERT_FADE (not REAL_GROUND_FADE); steppe takes the steppe donor there',
}
# 'other' outside a consumer's map: polish keeps the authored band climate (no _clim entry); apply_climate's
# POLISH_CLIMATE.get(n, 'arid') makes it arid. Owner intention: 'other' keeps its band climate (not arid).
OTHER_DEFAULT = {'polish_climate': None, 'gameplay_polish': 'arid'}
# Cold lines (three forms today; owner intention: the latitude TREELINE plus SNOWLINE, the flat COLD_ELEV
# for gameplay climate only).
COLD_LINES = {
    # ground_materials cold band (m): 3000 m south of 47 N (apply_climate COLD_ELEV, vanilla >= 3000 m:
    # 99 % G cold), lower to the north (Altai ~2400, Sayan ~2000). Edge fuzz +-SNOW_FUZZ m (fbm).
    'SNOWLINE': ((47., 50., 54., 58.), (3000., 2500., 2000., 1700.)),
    # ecotone_bands treeline (m) by latitude: Himalaya ~3900, Pamir/Kunlun 3600-3400, Tian Shan ~2900,
    # Altai 2300-2100, Sayan ~1900, Stanovoy/Verkhoyansk 1300-1000.
    'TREELINE': ((28., 34., 38., 42., 46., 50., 54., 58., 62.), (3900., 3700., 3400., 3000., 2600., 2200., 1800., 1400., 1000.)),
    # snow_bake perennial snowline (m, year-round white peaks; snow-bake-design §3.2): the regional glacier
    # equilibrium-line order by latitude on the dry side (Himalaya north slope / inner Tibet 5,500-6,000, Kunlun ~5,000-5,400,
    # Pamir 4,600-5,200, Tian Shan ~4,000, Altai ~3,100-3,200, Sayan ~2,800, Verkhoyansk ~2,000); the monsoon side is
    # lower (snow_bake MONSOON_DROP: Hengduan ~4,800). First cut for the offline previews; WorldCover class 70 checks it.
    # Peak area (ETOPO 60", fuzz 0; second cut after the snow preview 2026-10-30) vs real glaciers km2: Pamir/Karakoram W
    # 15,600 / ~20,000, Kunlun 11,800 / ~12,000, Himalaya N of 27 N 18,000, inner Tibet 2,800 / ~5,000, Hengduan 1,400 /
    # ~1,500, Qilian 1,600 / ~1,900, Tian Shan 9,300 / ~9,000, Altai 1,700 / 1,200-2,000.
    'PERENNIAL_SNOW': ((26., 28., 30., 32., 35., 36.5, 38., 39., 42., 44., 47., 50., 54., 60., 66.),
                       (5200., 5400., 5700., 5650., 5800., 5350., 4950., 4700., 4100., 3800., 3400., 3100., 2700., 2200., 1900.)),
    'COLD_ELEV': 3000,        # apply_climate (m): vanilla land at or above it is 86% cold (gameplay climate only)
    'NORTH_COLD': 47,         # apply_climate (deg): north of vanilla's land (42.6 N); continental Dfb/Dfc and taiga
}

# ------------------------------------------------------------------ constants by consumer
# Names are the consumer's module-level names (read back as NAME = consts(module)['NAME']); slot lists
# stand for the numpy masks/tables the consumer builds from them.
CONSTS = {
    # snow bake (snow-bake-design §3; snow_bake.py reads them through perennial_snow / winter_snow / shoulder_snow):
    # fuzz with the same fbm field as the ground (classify seed 211, -1..1) so no edge is a straight line or a square
    'snow_bake': {
        'PERENNIAL_FUZZ': 120.,   # m: fbm shift of the perennial line
        'PERENNIAL_RAMP': 150.,   # m: 0 -> 1 above the (fuzzed) line (the peak set: weight >= .5)
        # monsoon side (SE Tibet, Hengduan): line lowered by MONSOON_DROP x smooth((lon - lon0) / dlon) x smooth((lat1 - lat) / dlat)
        'MONSOON_DROP': 450., 'MONSOON_LON': (90., 6.), 'MONSOON_LAT': (34.5, 2.5),
        # dry interior (2026-11-01, snow review round 1: Qiangtang winter .257/.276 > .25, ridges near 5,700 m read as
        # perennial): the arid plateau interior has the highest snowline on Earth, firn lines above 6,000 m in the
        # Qiangtang (Wang, Guo & Wu, AGU 2014, > 3,000 glaciers; Purogangri ELA ~5,900-6,000 m, MODIS). The line
        # rises by DRY_RAISE x smoothstep((regional elevation - DRY_REGION[0]) / DRY_REGION[1]) (snow_reference_elev
        # 100 km mean): Qiangtang (regional ~5,080 m) 5,750 -> ~6,010 m; the plateau rim keeps its line (Himalaya
        # crest region ~3,550, Muztagh Ata ~3,590, K2 ~4,600, Kunlun W ~4,810, Amne Machin ~4,210, Bogda ~1,310,
        # Gongga ~3,340). Only with e_region (TW3K_SNOW_COVERAGE callers); without it the line is unchanged.
        'DRY_RAISE': 400., 'DRY_REGION': (4900., 300.),
        # winter (channel A): the cold ground line (COLD_LINES SNOWLINE, classify's cold), the treeline band, and
        # continental winters by latitude (deg, intensity): ~.7 at the native border (vanilla's north-China winter, snow
        # preview 10-30), full from 50 N; on dry ground every winter term keeps 1 - WINTER_DRY
        'WINTER_LAT': ((38., 45., 50.), (0., .7, 1.)),
        # plateau dryness (plateau_dryness: m from elev[0] over elev[1]; deg south of lat[0] over lat[1])
        'PLATEAU_DRY': {'elev': (3000., 1000.), 'lat': (40., 2.)}, 'WINTER_DRY': .6, 'TREE_FUZZ': 150., 'TREE_RAMP': 300.,
        # winter mountain snow (winter_snow): from WINTER_MOUNTAIN[0] m below the treeline, full WINTER_MOUNTAIN[1] m higher
        # (Tian Shan forest belt 2,000-2,700 m, Bayanbulak 2,500 m: deep winter snow)
        'WINTER_MOUNTAIN': (1000., 800.),
        # [TW3K_SNOW_COVERAGE] winter_snow as the COVERAGE share (snow_bake writes a binary mask, the vanilla snow shader
        # is a hard switch at sRGB byte ~232: snow-v22b-findings section 3). Coverage from MODIS literature: Tibetan Plateau
        # winter snow-cover frequency ~21 % (Chu et al. 2023, MOD10A2), the interior incl. Qiangtang below 10 % (annual),
        # Himalaya / eastern Tibet above 40 % (Basang et al. 2017), rising with elevation (annual < 4 % below 3,000 m,
        # 77 % above 6,000 m: Chu et al. 2023). winter_dry: the flat dry plateau keeps 1 - winter_dry of the cold ground
        # (.15: winter is the snowiest MODIS season on the plateau, mean ~21 %, February ~33 % (Pu et al. 2007), and the
        # central Qiangtang is among its lowest winter SCF areas (Basang et al. 2017), so the interior sits below .21;
        # .12 left the snow review near view at .088 < .10; Qiangtang box .174 on the real terrain); the mountain term is graded from 1,000 m below the treeline up to
        # the perennial line (no full band in the meadow belt); on the dry plateau only ranges standing rel m above the
        # local mean (local_km box) take it; plateau dryness reads the regional elevation (region_km box), so it is
        # constant along a slope and coverage only grows upward. <scratch>/snow-coverage-20261101.
        # stable_snow: the Junggar basin (northern Xinjiang, one of China's three stable seasonal snow regions with the
        # northeast and the plateau; the Gurbantunggut desert lies under snow all winter): the desert dryness does not
        # thin the latitude term there. An ellipse in degrees (centre, radii) with a soft rim (share of the radius), so
        # no polygon edge; Tarim (south of the Tian Shan) and the Gobi (east of 92 E) stay outside. Snow review B7
        # (2026-11-01): basin floor < 1,000 m was .23 against >= .40 with the plain dryness.
        'COVERAGE': {'winter_dry': .85, 'rel': (150., 600.), 'local_km': 26., 'region_km': 100.,
                     'stable_snow': {'centre': (85.5, 45.4), 'radius': (6.0, 2.2), 'rim': .3}},
        # spring/autumn (channel B): vanilla has it on northern high ground only: above the treeline, and north of SHOULDER_LAT
        'SHOULDER_LAT': (50., 54.),
    },
    # props bake (step 3) rule constants, read through the predicates below (flat_arid_hex,
    # alpine_treeline_tile); moved verbatim from foliage_biomes (ROCK RELIEF, TREELINE ALPINE)
    'props': {
        'ROCK_RELIEF': 400.,          # m: 7x7-hex real ETOPO range below which arid ground is flat (no rocks)
        'ROCK_RELIEF_K': 3,           # window half width (hexes)
        'ROCK_ARID': ('desert', 'steppe', 'grass', 'savanna'),   # arid RESOLVE classes of the rock rule
        'TREELINE_CLASS_MIN': .5,     # alpine_meadow + alpine_steppe classify weight of a treeline tile
        # Oasis town trees (user decision 2026-10-30, decisions 6b0d5e97 step 3 A; replaces foliage_biomes.TOWN_OASIS
        # (12, .22) as the data source; planter and the oasis_town_trees invariant: the props bake). A band-walk hex
        # (polish_campaign_foliage) in 'ring_hex' (town distance; the planter clears < 8) takes the poplar chance
        # 'chance' when its eco_bio class is in 'classes' (desert: unchanged) or in 'oasis_classes' on an oasis hex
        # (oasis >= oasis_min; eco_bio calls the Yanqi / Qiuci / Gumo oasis margins steppe where strategic_ink said
        # forest, and they lost both; steppe off the oases keeps its sparse rule). The floor is real geography: the
        # oasis hexes of a town = walk hexes in 'floor_ring_hex' of their nearest town, class in 'classes' or
        # 'oasis_classes', landform-
        # tiles 'oasis' (polish oasis belts: oasis_seeds from the Hanshu areas, the Turpan Han oases, river and lake
        # belts) >= 'oasis_min' at the block-centre tile; a town keeps at least ceil(min_hit_share x its oasis
        # hexes) poplar hits (trees_per_hit trees each), drawn deterministically where the chance draws gave fewer.
        # min_hit_share: tree cover of the irrigated oasis. No measured oasis canopy share was found (search
        # 2026-10-30); .15 is the order of the Three-North shelterbelt forest-cover target (5 -> 15 %), an
        # approximation to replace with a measured Tarim-rim canopy share. Floors on main inputs (oasis-floor copy,
        # 08:56): 33 towns, e.g. Yanqi farm 27 oasis hexes -> 5 hits, Qiuci lumber 19 -> 3, Gumo capital 5 -> 1;
        # Jushi-front farm 0 (its Turpan oasis hexes lie inside the town clearance or nearer another town).
        'TOWN_OASIS': {'classes': ('desert',), 'oasis_classes': ('steppe',), 'ring_hex': (8., 12.), 'chance': .22,
                       'floor_ring_hex': (8., 16.), 'oasis_min': .5, 'min_hit_share': .15, 'trees_per_hit': 3},
        # Frontier mountain props without a family cap (stateless draws, coordinator 2026-10-31): the warn-only
        # invariant frontier_mountain_family_count warns when a family's placed instances deviate from its
        # vanilla-rate target (frontier_mountain_props report) by more than this share.
        'MOUNTAIN_FAMILY_COUNT_TOL': .10,
        # Frontier mountain family targets (coordinator 2026-10-31, options a + ii): target = rate x scale x the
        # family's canvas candidate hexes (frontier_mountain_props' own eligibility test: impassable land within REACH
        # of playable land, clear of water / roads / towns, relief >= MIN_RELIEF, family climate mask).
        # rate = vanilla's own: instances of vanilla's cold / arid mountain models (every leaf of the vanilla dlc07
        # global_props.bin, moved by the native shift) on native-rectangle hexes that pass the same test, per such
        # hex (frontier_mountain_props.py --measure-vanilla-rate, v21 inputs 2026-10-31): cold 31 / 18,827 (30
        # clusters), arid 2 / 4,714. scale: baseline = the v21 placement (cold 157, arid 195 on 57,327 / 66,204
        # candidates: 157 / 94.4 -> 1.66, 195 / 28.1 -> 6.94). The frontier is wide uninhabited mountain land, and
        # vanilla's mountains mostly sit near towns, roads and water, which the candidate test refuses (309 of 340
        # cold), so the per-site rate is not directly comparable; arid rests on 2 vanilla samples. Density is judged
        # by the art review, which changes the scale; frontier_mountain_family_count warns on drift from it
        # (> MOUNTAIN_FAMILY_COUNT_TOL). Before: target = the slope / TPI / real-ridge seed density summed over the
        # candidates (346 cold, never reached; 199 arid, the old cap).
        'MOUNTAIN_TARGET': {'rate': {'cold': 31 / 18827, 'arid': 2 / 4714}, 'scale': {'cold': 1.66, 'arid': 6.94}},
    },
    'ground_materials': {
        # Native ring (tile climate rule, user decision 2026-10-26): canvas tiles less than reach + meander x
        # clip(.5 + .7 fbm) tile cells outside the native rectangle take the carried native climate (polish's
        # tile climate: inherited, extended outward), the rest the blend's G band. Vanilla's own blend G and tile
        # climate disagree on 8 border stretches (west 575, east 1025/1050, north 775-825/1325), so one G cannot
        # meet both the blend 'cut' and the 'climate_cut' gate there. grain hexes and seed of the meander noise.
        'NATIVE_RING': {'reach': 12., 'meander': 20.},
        # The native edge geometry, one definition (details review 2026-10-27): the euclidean tile distance to the
        # native rectangle and the meander field clip(.5 + .7 fbm(q, r, grain, 3 octaves, seed)). Read by the tile
        # climate ring (NATIVE_RING reach/meander) and the mirror band (polish MIRROR_EDGE), each with its own reach.
        'NATIVE_EDGE': {'grain': 14., 'seed': 95},
        # Shared visual ownership only (no map/height edits): smoothstep distance in hexes from
        # explicitly protected native playable land. The native rectangle is not a fade boundary.
        'VISUAL_DOMAINS': {'material_ramp_hex': (0., 4.), 'vegetation_ramp_hex': (0., 4.)},
        # Mirror band where its native source's climate differs from the canvas climate: the donor takes over by
        # smoothstep(dn / R), R = reach + meander x m on native_edge (tile cells; inside NATIVE_RING: R <= ring R).
        'MIRROR_EDGE': {'reach': 6., 'meander': 14.},
        # Native-edge tone CUT (ground contrast fix design §10, user 10-30; audit_native_edges 'tone_cut'): a border
        # line fails in a season when its tone jump > min and > ratio x max(its interior lines' jump, floor). min = p99
        # of vanilla's own interior reference-line jumps over the native-border segments, floor = min / ratio (10.1), by
        # build_material_tones.py (analysis/material-tones.json TONE_CUT): vanilla 3k_dlc07 global_blend sha256
        # 08fa46b3..., terrain.pack, 2026-10-30, 242 segments x 4 lines. Re-measure from vanilla only; never tune to pass.
        'TONE_CUT': {'ratio': 3.0,
                     'by_season': {'spring': {'min': 5.757603178024315, 'floor': 1.9192010593414384},
                                   'summer': {'min': 5.6349348449707115, 'floor': 1.8783116149902372},
                                   'harvest': {'min': 5.6349348449707115, 'floor': 1.8783116149902372},
                                   'autumn': {'min': 5.55613847732544, 'floor': 1.85204615910848},
                                   'winter': {'min': 5.865808463096624, 'floor': 1.9552694876988745}},
                     'provenance': {'blend_sha256': '08fa46b37cee492f36f434ed3e46b37521d4ae66e9d8fe143b0db1b235f5b9bb',
                                    'date': '2026-10-30', 'segments': 242, 'json': 'analysis/material-tones.json'}},
        # Ground tone table (ground contrast fix design §1, user decision 2026-10-27: the canvas no harsher than
        # vanilla). MATERIAL_TONE: Lab(mean sRGB) proxy of each texture-array layer (0-31), five seasons,
        # measured by build_material_tones.py. Layer=R and seasonal fallback remain renderer assumptions,
        # not captured engine cbuffer values. CONTRAST: vanilla native land's p99 of differing adjacent
        # pixels, calculated from these exact float32 tones, per G band. Provenance: analysis/material-tones.json.
        # Shared with the patchiness metric. Runtime reads only this data, never the original texture assets.
        'MATERIAL_TONE': {'spring': [[73.76, 2.89, 34.26], [69.44, 2.8, 36.0], [65.79, 1.3, 31.13], [67.86, 0.83, 32.92],
                    [32.82, -7.42, 31.4], [27.7, -10.83, 20.43], [39.86, -14.21, 35.12], [47.53, -9.06, 42.35],
                    [45.47, -8.24, 38.31], [55.82, -10.44, 46.53], [52.63, -18.82, 42.48], [52.0, -12.64, 37.45],
                    [50.78, -20.37, 45.62], [60.98, -1.59, 27.77], [48.78, 0.95, 34.14], [56.2, -25.17, 5.26],
                    [48.54, -8.56, -0.4], [49.54, -24.12, 5.9], [47.67, -14.13, 0.79], [67.23, 1.83, 37.27],
                    [62.39, 1.58, 39.78], [56.42, -1.81, 43.42], [51.58, -8.06, 44.16], [63.72, -6.84, 50.55],
                    [55.88, -11.75, 50.21], [50.41, -10.43, 44.62], [48.24, -7.35, 40.8], [67.07, -5.82, 47.44],
                    [61.56, -12.45, 46.65], [59.23, -11.49, 49.13], [57.82, -12.98, 50.0], [50.7, -8.66, 36.39]],
         'summer': [[73.76, 2.89, 34.26], [69.44, 2.8, 36.0], [65.79, 1.3, 31.13], [67.86, 0.83, 32.92],
                    [32.82, -7.42, 31.4], [27.7, -10.83, 20.43], [39.86, -14.21, 35.12], [47.53, -9.06, 42.35],
                    [65.41, -12.81, 57.81], [62.22, -9.32, 51.53], [55.91, -21.79, 52.75], [55.94, -18.16, 51.92],
                    [50.78, -20.37, 45.62], [60.98, -1.59, 27.77], [48.78, 0.95, 34.14], [56.2, -25.17, 5.26],
                    [48.54, -8.56, -0.4], [49.54, -24.12, 5.9], [47.67, -14.13, 0.79], [67.23, 1.83, 37.27],
                    [62.39, 1.58, 39.78], [56.42, -1.81, 43.42], [51.58, -8.06, 44.16], [63.72, -6.84, 50.55],
                    [55.88, -11.75, 50.21], [45.29, -11.81, 39.69], [42.79, -7.63, 37.01], [64.4, -2.39, 43.32],
                    [59.87, -9.76, 49.59], [56.29, -12.13, 50.47], [50.05, -11.82, 45.45], [49.33, -8.64, 36.49]],
         'harvest': [[73.76, 2.89, 34.26], [69.44, 2.8, 36.0], [65.79, 1.3, 31.13], [67.86, 0.83, 32.92],
                     [32.82, -7.42, 31.4], [27.7, -10.83, 20.43], [39.86, -14.21, 35.12], [47.53, -9.06, 42.35],
                     [75.69, 1.54, 58.45], [63.5, -6.87, 51.78], [67.99, -12.25, 41.32], [61.58, -13.65, 47.73],
                     [50.78, -20.37, 45.62], [60.98, -1.59, 27.77], [48.78, 0.95, 34.14], [56.2, -25.17, 5.26],
                     [48.54, -8.56, -0.4], [49.54, -24.12, 5.9], [47.67, -14.13, 0.79], [67.23, 1.83, 37.27],
                     [62.39, 1.58, 39.78], [56.42, -1.81, 43.42], [51.58, -8.06, 44.16], [63.72, -6.84, 50.55],
                     [55.88, -11.75, 50.21], [45.29, -11.81, 39.69], [42.79, -7.63, 37.01], [64.4, -2.39, 43.32],
                     [59.87, -9.76, 49.59], [56.29, -12.13, 50.47], [50.05, -11.82, 45.45], [49.33, -8.64, 36.49]],
         'autumn': [[73.76, 2.89, 34.26], [69.44, 2.8, 36.0], [65.79, 1.3, 31.13], [67.86, 0.83, 32.92],
                    [32.82, -7.42, 31.4], [27.7, -10.83, 20.43], [39.86, -14.21, 35.12], [47.53, -9.06, 42.35],
                    [51.27, 7.72, 43.03], [55.16, -1.12, 40.78], [42.33, 2.87, 32.91], [49.03, -3.34, 40.59],
                    [50.78, -20.37, 45.62], [60.98, -1.59, 27.77], [48.78, 0.95, 34.14], [56.2, -25.17, 5.26],
                    [48.54, -8.56, -0.4], [49.54, -24.12, 5.9], [47.67, -14.13, 0.79], [67.23, 1.83, 37.27],
                    [62.39, 1.58, 39.78], [56.42, -1.81, 43.42], [51.58, -8.06, 44.16], [63.72, -6.84, 50.55],
                    [55.88, -11.75, 50.21], [45.38, -9.98, 41.16], [43.67, 1.67, 39.54], [63.49, 2.12, 42.42],
                    [59.05, -6.83, 40.11], [54.31, -7.03, 43.76], [49.57, -5.7, 43.16], [48.8, -4.81, 29.94]],
         'winter': [[73.76, 2.89, 34.26], [69.44, 2.8, 36.0], [65.79, 1.3, 31.13], [67.86, 0.83, 32.92],
                    [32.82, -7.42, 31.4], [27.7, -10.83, 20.43], [39.86, -14.21, 35.12], [47.53, -9.06, 42.35],
                    [53.01, 0.58, 16.68], [58.18, -0.61, 14.55], [39.52, 3.72, 25.73], [45.18, -1.88, 34.75],
                    [50.78, -20.37, 45.62], [60.98, -1.59, 27.77], [48.78, 0.95, 34.14], [56.2, -25.17, 5.26],
                    [48.54, -8.56, -0.4], [49.54, -24.12, 5.9], [47.67, -14.13, 0.79], [67.23, 1.83, 37.27],
                    [62.39, 1.58, 39.78], [56.42, -1.81, 43.42], [51.58, -8.06, 44.16], [63.72, -6.84, 50.55],
                    [55.88, -11.75, 50.21], [73.58, -1.09, 2.79], [54.42, -4.43, 14.38], [55.52, -2.37, 17.1],
                    [53.31, -2.76, 13.08], [54.35, -3.79, 10.69], [56.17, -2.8, 9.73], [48.25, -2.18, 20.68]]},
        'CONTRAST': {'pairs': 53613816,
         'differing_share': 0.03749,
         'percentiles_of_differing': {'50': 11.441902160644531,
                                      '75': 14.927227020263672,
                                      '90': 22.663524627685547,
                                      '95': 30.072057723999023,
                                      '99': 36.86005783081055,
                                      '99.9': 43.4322395324707},
         'max_dE': 36.86005783081055,
         'share_above_max': 0.000332,
         'share_above': {'5': 0.034558,
                         '10': 0.024355,
                         '15': 0.009136,
                         '20': 0.004767,
                         '25': 0.003212,
                         '30': 0.002252,
                         '40': 0.000327},
         'by_band': {'0': {'pairs': 12543856,
                           'differing_share': 0.03109,
                           'percentiles_of_differing': {'50': 11.441902160644531,
                                                        '75': 12.811882019042969,
                                                        '90': 19.367137908935547,
                                                        '95': 28.44580841064453,
                                                        '99': 34.127464294433594,
                                                        '99.9': 42.31500244140625},
                           'max_dE': 34.127464294433594,
                           'share_above_max': 0.000131},
                     '1': {'pairs': 8452192,
                           'differing_share': 0.02872,
                           'percentiles_of_differing': {'50': 7.93950080871582,
                                                        '75': 11.73133659362793,
                                                        '90': 22.69724464416504,
                                                        '95': 27.07235336303711,
                                                        '99': 36.030189514160156,
                                                        '99.9': 42.31500244140625},
                           'max_dE': 36.030189514160156,
                           'share_above_max': 0.000231},
                     '2': {'pairs': 12758052,
                           'differing_share': 0.04027,
                           'percentiles_of_differing': {'50': 10.684134483337402,
                                                        '75': 11.895264625549316,
                                                        '90': 21.826467514038086,
                                                        '95': 30.06267738342285,
                                                        '99': 34.127464294433594,
                                                        '99.9': 42.31500244140625},
                           'max_dE': 34.127464294433594,
                           'share_above_max': 0.000195},
                     '3': {'pairs': 19859716,
                           'differing_share': 0.04348,
                           'percentiles_of_differing': {'50': 12.577096939086914,
                                                        '75': 16.147363662719727,
                                                        '90': 26.208261489868164,
                                                        '95': 30.605390548706055,
                                                        '99': 41.490718841552734,
                                                        '99.9': 43.4322395324707},
                           'max_dE': 41.490718841552734,
                           'share_above_max': 0.000373}}},
        # Tone bridge (polish blend): the natural slots a bridge may take (arid, cold, steppe, subtropical, temperate;
        # not farm, road, mud, wetland), and the bridge share of the dither near its threshold.
        'BRIDGE_SLOTS': (0, 1, 2, 3, 4, 5, 6, 7, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30),
        'BRIDGE_W': .25,
        # Forest density (ground contrast fix §3, shared with the props bake): 0-1 per tile, how dense a forest the
        # ground carries. 'base': per RESOLVE class, the props bake's choose() high probabilities normalised by the
        # largest (subtropical .45: fir .36, forest .32, dry .16, open .10, sparse .008); box-blurred over 'blur'
        # tile cells; times 1 - amp + amp x clip(.5 + .7 fbm(grain, seed)). No jitter, no town / road / shore /
        # slope exclusions: those stay with the props bake's placement.
        'FOREST_DENSITY': {'base': {'taiga': .8, 'conifer': .8, 'forest': .71, 'tropical': 1., 'mangrove': 1.,
                                    'tropical_dry': .36, 'savanna': .36, 'steppe': .02, 'alpine': .02, 'tundra': .02,
                                    'grass': .22, 'temperate': .22, 'wetland': .22, 'paddy': .22, 'other': .22,
                                    'desert': 0., 'snow': 0.},
                           'blur': 3, 'amp': .4, 'grain': 6., 'seed': 131,
                           # props bake: a map forest-ground hex keeps at least this tree probability (gameplay forest has trees)
                           'map_forest_floor': .08},
        # Donor woodland floor (polish blend, ground contrast fix §3): the woodland slots follow FOREST_DENSITY x k per
        # donor climate (0 subtropical, 1 cold, 2 temperate, 3 arid, 9 steppe, 10 paddy, 11 open subtropical); a
        # woodland pixel the density does not want takes one of that climate's 'open' candidates (noise-picked).
        'DONOR_WOODLAND': {'slots': WOODLAND_SLOTS, 'grain': 3., 'seed': 133,
                           'k': {0: 1., 1: .9, 2: .9, 3: .3, 9: .3, 10: .7, 11: .5},
                           # Ground display palette only, AFTER farming/ownership reservations.
                           # Generated wood keeps w*D dark; w-dark takes its nearest legal
                           # existing climate-open material, retaining source/G. Shared D and
                           # all tree/authoring rules stay intact. False restores the old palette.
                           'ground_palette_response': {'enabled': True, 'sources': (3, 5)},
                           'open': {0: (26, 25, 24), 1: (6, 7), 2: (28, 27, 29), 3: (1, 2, 0), 9: (19, 20, 21),
                                    10: (25, 26, 24), 11: (26, 25, 24)}},
        'SNOW_FUZZ': 180.,      # m: fbm fuzz of COLD_LINES SNOWLINE
        'SNOW_RAMP': 260.,      # m: 0 -> 1 above the (fuzzed) line
        'STEEP': (110., 300.),  # m per tile cell (~2 km) of real elevation: ridge weight 0 -> 1
        'TUNDRA_GRASS_LAT': 62.,
        # Only these source materials change: bright sand and bright steppe (the arid and steppe donors'
        # light members). Woodland 4/5, cold_3/4, steppe_3/4, temperate 27-30 (oasis), subtropical, farms
        # 8-12, road 13, mud 14, wetlands 31 keep theirs.
        'REPLACEABLE': [0, 1, 2, 3, 19, 20],
        # [REALISM] On real desert below the cold line and outside oases (fields['Wd']) the donors' greens
        # also change: steppe_3/4, woodland cold_1/2 and temperate 27-30 left there read as green flecks
        # (vanilla real desert: green in 47 large sheets, ours in 10,614 specks).
        'DESERT_EXTRA': [4, 5, 21, 22, 27, 28, 29, 30],
        # [EGOBI] the desert pass weight is the real-ground weight ramped over this many tiles from the native
        # border, not real_gw x donor_share (both near 0 next to the border: the East Gobi greens stayed)
        'DESERT_RAMP': 16,
        # [RELIEF GROUND, beautify F 2026-10-25] open ground (steppe, grass, humid alpine, tundra, savanna) takes
        # its materials from the real relief as vanilla's does: green valley floors, brown slopes, stony crests,
        # graded, instead of the donor windows' green/ochre patches that ignore valleys and ridges. Per tile:
        # valley = (local mean - elevation) / RH_REF over a VALLEY_R box, plus the river share; crest = the
        # opposite times the slope ramp; slope = the rest. Humidity (1 - oasis_ground.dryness, grass/temperate
        # count humid) picks the valley mix. The donors' greens and browns there are redrawn (OPEN_EXTRA) with
        # the open weight; desert, high, plateau and the oases keep their own rules.
        # [USER 2026-10-25 'greener and coarser'] steppe counts humid (meadow steppe: Tuoba, Hulunbuir), damped by
        # oasis_ground.dryness; valley/crest parts sharpened (contrast) over a wider relief box with a humid
        # bias, each part almost one material, so the relief reads in large sheets instead of pepper specks
        'RELIEF': {'biomes': ('steppe', 'grass', 'alpine', 'tundra', 'savanna'), 'humid': ('steppe', 'grass', 'alpine', 'tundra', 'savanna'),
                   'blur': 3, 'valley_r': 24, 'rh_ref': 150., 'river_gain': 3., 'slope_ref': (40., 160.), 'gain': .95,
                   'contrast': 2.5, 'humid_bias': .25,
                   'mix': {'valley_humid': {28: .80, 27: .20},
                           'valley_dry': {21: .85, 22: .15},
                           'slope': {19: .80, 20: .20},
                           'crest': {2: .70, 3: .30}},
                   # landforms whose own mix is graded by the same relief (share 'classes_blend' of their mix
                   # goes to the relief parts; their donor greens/browns are redrawn with that share)
                   'classes': ('plateau', 'alpine', 'sgravel', 'tundra'), 'classes_blend': .7},
        # woodland 4/5 stays (real forest patches inside open ground: Bogda north slope)
        'OPEN_EXTRA': [19, 20, 21, 22, 27, 28, 29, 30],
        'GRAIN': (4., 3., 4.),   # hexes: take-dither, material blobs (default; landforms: LANDFORMS 'grain'), G-band dither
        # farms and mud (ground_materials farm_mud_fields; survey and rules in its comment block)
        'FARM_KINDS': ((8, 9), (10, 11), (12, 12)),      # grain, rice, tea: (inner, rim) material
        'FARM_GRAIN': 4.5, 'FARM_FRAY': .5, 'FARM_DAB': 1.0, 'FARM_KIND_GRAIN': 3.5,   # hexes (FARM_GRAIN: 4 octaves, field complexes)
        'FARM_FRAY_AMP': .10,        # weight units of the fine edge noise
        'FARM_RIM': .07,             # weight margin under which a field pixel is rim (_2) material
        'FARM_INNER': (.45, .55, 1.),   # share of the inner part that takes the _1 material (grain, rice, tea)
        'FARM_RIVER': (1., 3., 9., 16.),     # hexes from river hexes: 0 -> 1 -> 1 -> 0
        'FARM_REAL_RIVER': (.5, 2., 6., 11.),   # hexes from Natural Earth rivers (no river hexes)
        'FARM_LAKE': (1., 3., 6., 10.),
        'FARM_TOWN_CLEAR': 10., 'FARM_TOWN': (10., 13., 18., 26.),   # hexes from authored towns
        'FARM_TOWN_KIND': {'farm': 1., 'city': .7, 'other': .35},
        # Valley, lake and plain weights halved after the r16 review (Funan/Mekong plains densely tiled: 60-hex
        # window median 5.1%, p90 10% on farmable land against vanilla's p75 2.5%, p90 8%); towns and oases kept.
        'FARM_WEIGHT': {'river': .15, 'real_river': .11, 'lake': .09, 'plain': .06, 'town': .26, 'oasis': .40},
        'FARM_ELEV': (2000., 2600.),   # m: no fields above
        'FARM_SLOPE': ((0, .05, .1, .2, .3), (1, .9, .45, .12, 0)),   # world units per hex (vanilla: farm_fields.SLOPE_KEEP)
        'TEA_SLOPE': ((.05, .1, .35, .5), (0, 1, 1, 0)),
        'TEA_ELEV': (200., 400., 1800., 2200.),
        'RICE_ELEV': (900., 1300.),
        'RICE_LAT': (33., 35.),
        'PLAIN_ELEV': ((300., 700.), (700., 1200.)),   # m: rice / grain plains fade out
        'VALLEY_REL': (60., 260.),   # m above the 10-cell local mean: valley floor 1 -> 0
        'MUD_RIVER': (.3, 1., 2.5, 5.),
        'MUD_LAKE': (.3, 1., 3., 5.5),
        'MUD_WEIGHT': {'river': .40, 'lake': .38},
        'MUD_ELEV': 3200.,
        'MUD_GRAIN': 2.5,
        'OASIS_FARM': .9,            # oasis setting weight before FARM_WEIGHT['oasis']
        # [ARID FARM] oasis_discs that do not water fields (farm_mud_fields' oasis term only; the discs' other readers -
        # apply_ground_types oasis hexes, oasis_ground seed centres - keep them): Turpan, whose basin farms are the Han
        # oases of REGIONS turpan_han_oasis ([TURPAN3]); its 30 km disc drew fields south-east of Turpan city with no
        # Han source (C2 review 2026-11-01). v23: none (GROUND_RECIPE_V23).
        'FARM_OASES_SKIP': ('Turpan',),
        # [ARID FARM] an arid town's own fields (its own water: wells, karez mouth): rise from 1.2 to 1.6 hexes, full to
        # farm_fields ARID_OWN_OASIS (the one constant both farm writers read; None here), gone .5 hex beyond. Further
        # out arid fields need real water (valley, lake shore, oasis belts). v23: (1.2, 2.5, 4.5, 9.) - a 9-hex disc that
        # watered itself drew fields on the gobi round Jushi / Weixu (C2 review 2026-11-01, criteria 8529d9cc).
        'ARID_TOWN': (1.2, 1.6, None, .5),
        'TOWN_WARP': .35,          # fbm warp of the town distance (no circular rings)
        'FARM_SOURCE': list(range(0, 8)) + list(range(19, 31)),
    },
    'oasis_ground': {
        # Dry high ground (survey on alpine cells in oasis_ground): desert share box-blurred twice over
        # DRY_R tile cells (~35 km reach), DRY maps that share to 0 -> 1 (smoothstep).
        'DRY_R': 16,
        'DRY': (.12, .30),
        'ARID_R': 8,   # tile cells: box radius (twice) of the arid share (polish's tonal buffer)
        'SEED_CORE': .45, 'SEED_EDGE': 1.6,   # x R = sqrt(km^2 / pi): weight 1 inside SEED_CORE R, 0 from SEED_EDGE R (.5 at R)
        'SEED_WARP': .35, 'SEED_GRAIN': 4.,    # fbm warp of the distance (no circles); hexes
        'SEED_RIVER': 1.,                    # reach x (1 + this x river share): oases run along their rivers
        'SEED_RISE': (100., 350.),           # m above the seed ground: full -> none (the water runs downhill)
        'SEED_SLOPE': (60., 160.),           # m per tile cell of real elevation (box 1): full -> none
        # [TURPAN2 2026-10-25] Aspect: the desert share times (1 + gain x south-facing share) before DRY, so the
        # sunny flank of a range beside a desert is drier than its shaded flank (Bogda: dry south slope above
        # Turpan, meadows on the north slope). South-facing share: northward rise of the real elevation (box
        # radius [2] twice) per tile cell / [1] m, clipped to 0..1; shaded slopes and humid ranges (desert share
        # 0: Ili, Yulduz, Altai) are unchanged. (gain, m per tile cell, box radius)
        'DRY_ASPECT': (1.5, 80., 2),
        # [TURPAN2] Lower forest line of the dry ranges: RESOLVE's Tian Shan conifer polygon reaches the Bogda
        # north piedmont at 600-1300 m (Jimsar/Fukang; one dark cold-donor sheet, live 2048p), where the real
        # spruce belt starts at ~1500-1700 m (Tianchi 1900 m) above desert steppe. Forest classes (treeline_forest)
        # below CONIFER_FLOOR m (+- CONIFER_FLOOR_FUZZ fbm) take the dominant open class around them
        # (CONIFER_FLOOR_OPEN, box radius DRY_R) where dryness is above CONIFER_FLOOR_DRY (ramp; humid
        # forests - Altai, Ili, Sayan - have dryness 0 and keep theirs).
        'CONIFER_FLOOR': (1350., 1750.),
        'CONIFER_FLOOR_DRY': (.15, .5),
        'CONIFER_FLOOR_FUZZ': 150.,
        'CONIFER_FLOOR_OPEN': ('steppe', 'grass', 'alpine'),
        'SUB_R': 3,            # tile cells: box radius of the shares
        'SUB_KEYS': (10, 11),  # polish pseudo-climates: paddy lowland, open subtropical
    },
    'ecotone_bands': {
        'TREE_LOCAL': .5,        # share of the local boundary elevation in the treeline
        'TREE_FUZZ': 160.,       # m of fbm on the treeline
        'TREE_SIGMA': 90.,       # m: soft step around the treeline
        'DITHER_AMP': .35,       # fbm of ECOTONE_GRAIN hexes: meander of the boundary
        'DITHER_FINE': .25,      # fbm of a third of that: interleaved tongues and islands
        # [§9.3 NE tundra band] tundra <-> forest (ecotone_bands.tundra_forest; tundra left the USR3 open set, whose
        # 'forest on ridges' rule is backwards at a cold treeline). RESOLVE drew the arctic treeline as long straight
        # vector edges (canvas tile y 3330-3650, 25% of the tundra/forest boundary windows one straight line); the real
        # forest-tundra is a wide mosaic with larch tongues along the rivers and tundra on the interfluves and uplands.
        #   WARP     tiles per unit fbm (fbm std ~.27: ~40 tiles) of the domain warp of the outline, WGRAIN hexes;
        #            only on flat ground (local relief std below the middle of REL m: mountains keep the elevation rule)
        #   BAND     tile cells (box twice) of the share; NOISE (GRAIN hexes) and FINE (GRAIN/4) fbm on the share
        #   RELIEF   x local elevation above its 24-cell mean / 150 m (uplands tundra); RIVER x river corridor
        #            (RIVER_R cells, x6) while the tundra share is below .85 (the forest tongues; not in the deltas)
        #   STRAIGHT stage_invariants tundra_forest_straight (ecotone_bands.straight_share): window, smooth, tol, min px
        'TUNDRA_FOREST': {'WARP': 150., 'WGRAIN': 30., 'REL': (60., 180.), 'BAND': 24, 'GRAIN': 24., 'NOISE': .37,
                          'FINE': .55, 'RELIEF': .2, 'RIVER': .3, 'RIVER_R': 6, 'STRAIGHT': (128, 8, 6., 60)},
    },
    'polish': {
        # Settlement ground copied from a same-climate vanilla town (native hex), by tile climate
        # 0 subtropical, 1 cold, 2 temperate, 3 arid; radii in hexes.
        'TOWN_GROUND': {0: (249, 160), 1: (550, 601), 2: (430, 481), 3: (135, 563)},   # Jiaozhi, Youbeiping, Luoyang, Wuwei
        'TOWN_GROUND_R': (2.5, 5),
        'WOODLAND_MATERIALS': set(WOODLAND_SLOTS),   # blend materials of vanilla woodland floors (never under vanilla towns)
        'FOREST_TOWN': (3, 8, .3),      # ring (hexes) and woodland share that keeps a town's forest floor
        'MIRROR_SKIP': [26],            # the mirrored native band does not copy paddy stripes (np.uint8 array)
        # [EGOBI 2026-10-25] the mirror band copied vanilla's north-rim cold_1 forest onto the real East Gobi
        # (hex 1239,1103 / 1181,1111): mirrored woodland (WOODLAND_MATERIALS) on real open ground (set
        # 'mirror_open') takes the canvas donor, ramped in over this many tiles from the native border.
        'MIRROR_OPEN_RAMP': 8,
        # Real desert (set 'real_desert') overrides the carried native climate over this tile fade instead of
        # REAL_GROUND_FADE (24-72): the carried cold climate took the cold donor on the East Gobi (hex 1311,1092).
        'REAL_DESERT_FADE': (8, 16),
        # [NONPLAY GROUND 2026-10-25] vanilla non-playable filler that is now playable mod land takes canvas ground,
        # weighted 1 - 3 x the box-blurred (this radius, hexes) share of vanilla playable land: 0 next to it.
        'NONPLAY_RAMP': 4,
        'DONOR_WARP': 64,      # pixels: warp of the donor patch tiling
        'DONOR_SCALE': 36,     # hexes: largest feature of the noise choosing among a climate's donor patches
        'DONOR_RELIEF': .4,    # weight of local relief (vs noise) in that choice
        'DONOR_BAND': 10,
        'DONOR_GRAIN': 3,
        'ECOTONE': 20,         # USR3 ECOTONE: half-width (tile cells, box-blurred twice)
        'ECOTONE_GRAIN': 4,    # and noise grain (hexes)
        'ECO_DETAIL': .85, 'ECO_DETAIL_R': 2,   # MATERIALS ECOTONE: fine climate share back into the donor score
        'LAKE_MOUTH': (12, 16.),   # USR3 BOSTEN: reed ground only near river mouths
        'LAKE_OASIS': 10,          # tile cells: riparian ring of inland lakes
        # OASIS_REMAP: oasis belts draw no paddy or cold material: (from, to) slot spans of the identity table
        'OASIS_REMAP': (((23, 27), (27, 31)), ((4, 8), (27, 31))),
        # FIX-BATCH TONAL BUFFER (curve, dense-forest test) and FIX2 BUFFER SHAPE (grain, octaves, meander, mass)
        'TONAL_BUFFER': ((0, 10, 20, 32, 48, 72), (.05, .1, .45, .9, 1., 0.)),
        'BUFFER_DENSE': (3, .9),
        'BUFFER_GRAIN': 3.5,   # FIX2 BUFFER SHAPE (the fix-batch 2. was overwritten before any use)
        'BUFFER_OCTAVES': 5,
        'BUFFER_MEANDER': .35,
        'BUFFER_MASS': (64, .01, .03),
        'BUFFER_REMAP': {0: {27: 30, 28: 29, 0: 7, 1: 7, 19: 7, 20: 30, 23: 30},
                         1: {0: 22, 1: 21, 2: 21, 3: 22, 19: 21, 20: 22, 23: 22, 27: 22},
                         2: {27: 29, 28: 30, 23: 26, 19: 29, 20: 30, 0: 30, 1: 29},
                         3: {23: 26}},   # subtropical: only subtropical_1
        # Share of each patch of a climate (default equal): the flattest subtropical and cold patches are
        # 90-99% cold_1 and at a full share laid one dark sheet over every plain (Funan, Lin Yi).
        'DONOR_SHARES': {0: (.1, .3, .3, .3), 1: (.15, .3, .3, .25)},
        # [section 9.3 taiga understory flecks] TW3K_TAIGA_DONOR=1 (ground_donor.compress_choice): on these eco_bio
        # biomes the donor choice is scaled into [0, cap) (blurred share, DONOR_BAND): taiga takes only the two flat
        # cold patches, .15 : .15 of the cold shares (equal; (100,422) cold_1/2 .996, (32,572) .80 with cold_3/4 .19),
        # not the rugged (38,482) / (110,476) whose cold_3/4 (.44 / .37) lay light-green islands on the conifer floor.
        'DONOR_COMPRESS': {'taiga': .30},
        # [section 9.3 treeline donor blend] TW3K_TREELINE_BLEND=1 (ground_donor.treeline_weight / treeline_pick): the tile climate of
        # tundra is arid (3) and of the forests cold (1), so the donor climate switched on the tundra/forest class line
        # (ECO_DETAIL follows the radius-2 share inside the redrawn ecotone): v22 siberia window dark floor .45 at -1
        # tile, .16 at +1, .03 at +2, against the tundra ground weight's 7-tile ramp (art review dc7a9361 section 1).
        # On 'from' tiles within 'band' tiles (euclidean) of a cold 'forest' class the donor climate is dithered
        # per pixel: climates[1] (cold, the forest's) where a quantile-uniform fbm ('grain' hexes, 'seed') is below
        # p = smoothstep(1 - d / band), else climates[0] (arid, the tundra's own): forest-floor patches thin out
        # into the tundra over 'band' tiles. A smooth share only moved the argmax contour (band 12 A/B: the same
        # crisp line, 6 tiles further in). The forest side is left as it is (no sand under the trees); donor
        # climate only, never the engine tile climate.
        'TREELINE_DONOR': {'from': ('tundra',), 'forest': ('taiga', 'conifer'), 'band': 8.,
                           'climates': (3, 1), 'grain': 1.5, 'seed': 141},
        'DONOR_PHASE': (307, 451),   # pixels: tiling phase step from one patch of a climate to the next
        'DONOR_JITTER': 3,           # pixels: per-pixel jitter of the donor lookup
        # Donor patches (native hex centres, 48x48 hexes) per tile climate, flattest to most rugged; 9 steppe,
        # 10 paddy lowland, 11 open subtropical (OASIS GROUND); see the comments at polish 'patches'.
        'DONOR_PATCHES': {0: [(64, 202), (194, 242), (338, 320), (278, 278)],
                          1: [(100, 422), (32, 572), (38, 482), (110, 476)],
                          2: [(526, 436), (416, 440), (458, 614)],
                          3: [(130, 670), (26, 644)],
                          9: [(262, 538), (290, 542), (266, 584)],
                          10: [(308, 188), (540, 304)],
                          11: [(194, 242), (338, 320), (278, 278)]},
        'OASIS_ELEV': (2200, 2800),   # m: oasis belts full below, none above (alpine belts there)
    },
    'farm_fields': {
        # Vanilla farm towns (region key: dlc07 map) used as donors, by field kind.
        'DONORS': {
            'grain_temperate': ('3k_main_anping_resource_1', '3k_main_chenjun_resource_2', '3k_main_yangzhou_resource_2',
                                '3k_main_weijun_resource_1', '3k_dlc06_xiapi_resource_1', '3k_main_jiangxia_resource_1',
                                '3k_main_henei_resource_1'),
            'grain_cold': ('3k_main_bajun_resource_1', '3k_main_chengdu_resource_2'),
            # Arid: vanilla's own arid fields are few (Anding 46 hexes, Wuwei none); the oasis rule
            # below cuts the larger temperate complexes down to irrigated ground.
            'grain_arid': ('3k_main_anping_resource_1', '3k_main_weijun_resource_1',
                           '3k_main_henei_resource_1', '3k_main_jiangxia_resource_1'),
            'rice': ('3k_main_cangwu_resource_1', '3k_main_kuaiji_resource_1', '3k_main_nanhai_resource_1',
                     '3k_main_yuzhang_resource_1', '3k_main_jianan_resource_1', '3k_main_jiaozhi_resource_1'),
            'tea': ('3k_main_changsha_resource_3', '3k_main_zangke_resource_1', '3k_main_jiaozhi_resource_2'),
        },
        'KIND_SCALE': {'farm': 1.0, 'city': .10, 'other': 0.},
        'CLIMATE_SCALE': {0: 1.0, 1: .35, 2: 1.0, 3: .5},
        'CITY_CLIMATE': {0: 1.0, 1: 0., 2: 1.0, 3: .6},     # cities: no cold fields
        'STEPPE_SCALE': {'farm': .3, 'city': 0., 'other': 0.},   # pastoral steppe towns
        'ARID_OWN_OASIS': 2,    # hexes around an arid town that count as irrigated (v23: 6, GROUND_RECIPE_V23)
        'SLOPE_KEEP': ((0, .05, .1, .2, .3, .5), (1, .9, .42, .2, .05, 0)),          # world units per hex
        'HEIGHT_KEEP': ((-1.5, -.5, -.1, .5, 1., 2.), (0, .12, 1, 1, .5, 0)),        # above the town
        'TEA_SLOPE': (.08, .45), 'TEA_HEIGHT': (.1, 1.2),
    },
    'frontier_forest': {
        'MIX': {   # vanilla tree-list shares by climate (dlc07)
            'cold': {'cold_tree_fir_large_1': .45, 'cold_tree_fir_medium_1': .45, 'cold_tree_fir_small_1': .03,
                     'cold_tree_fir_small_2': .03, 'cold_tree_fir_medium_2': .04},
            'cold_mixed': {'cold_tree_fir_large_1': .38, 'cold_tree_fir_medium_1': .38, 'cold_tree_fir_medium_2': .04,
                           'temperate_tree_katsura_medium_1': .07, 'temperate_tree_katsura_large_1': .07, 'temperate_tree_katsura_small_1': .06},
            'temperate': {'temperate_tree_katsura_summer_blossom_small_1': .11, 'temperate_tree_katsura_medium_1a': .11,
                          'temperate_tree_katsura_small_1': .11, 'temperate_tree_katsura_medium_1': .11, 'temperate_tree_katsura_medium_2': .10,
                          'temperate_tree_katsura_large_2': .10, 'temperate_tree_katsura_large_1': .10, 'cold_tree_fir_medium_1': .10,
                          'cold_tree_fir_large_1': .10, 'temperate_tree_metasequoia_large_1': .03, 'temperate_tree_metasequoia_medium_1': .03},
            'subtropical': {'subtropical_castanopsis_medium_1': .13, 'subtropical_castanopsis_large_1': .13, 'subtropical_castanopsis_large_2': .13,
                            'subtropical_castanopsis_small_2': .12, 'subtropical_castanopsis_small_1': .12, 'bamboo_1': .09, 'bamboo_2': .09,
                            'bamboo_3': .09, 'bamboo_small_1': .025, 'bamboo_small_2': .025, 'bamboo_small_3': .025},
            'dry': {'subtropical_castanopsis_medium_1': .25, 'subtropical_castanopsis_small_1': .25, 'subtropical_castanopsis_small_2': .25,
                    'subtropical_castanopsis_large_1': .25},
        },
        'STEEP_FACTOR': .75,       # vanilla mountain 0.62 vs light forest 0.82
        'TOWN_RAMP': ((0, 4, 7, 10), (.26, .64, .87, 1.)),   # vanilla thinning beyond the cleared ring
        'ROAD_HEX_FACTOR': .25,
        # Rock entries per impassable hex by slope (world units per hex): vanilla general rocks at about 2/3
        'ROCK_DENSITY': ((0, .1, .2, .4, .7, 1.2), (.001, .002, .003, .006, .02, .02)),
        'ROCK_CLUSTER_P': .7,      # geometric growth: median cluster of 3
        'SHRUB_DENSITY': .010,     # per open arid/steppe/alpine hex (vanilla ~13 per 1000 flat land hexes)
    },
    'apply_forest_ground': {
        'TOWN_CLEAR': 7,
        'CHANGE': {False: {'grassland': 'light_forest', 'plains': 'light_forest',
                           'hilly_grassland': 'hilly_light_forest', 'hilly_plains': 'hilly_light_forest'},
                   True: {'grassland': 'dense_forest', 'plains': 'dense_forest', 'desert': 'dense_forest'}},
    },
    'apply_ground_types': {
        # Learned table key: RESOLVE biome x ETOPO hex-mean metres band x relief band (std of the hex-mean
        # metres over the hex and its six neighbours); coarser keys where a key has < LEARN_MIN vanilla hexes.
        'ELEV_BINS': (200, 500, 1000, 1500, 2000, 3000, 4000),
        'ROUGH_BINS': (10, 20, 35, 55, 80, 120, 180, 260),
        'LEARN_MIN': 60,
        # Vanilla passable playable ground -> family (wetland, river crossing and passable mountain: not learned)
        'FAMILY': {'grassland': 'flat', 'plains': 'flat', 'light_forest': 'flat', 'dense_forest': 'flat',
                   'hilly_grassland': 'hilly', 'hilly_plains': 'hilly', 'hilly_light_forest': 'hilly',
                   'hilly_dense_forest': 'hilly', 'desert': 'desert'},
        # Family -> ground, on open ground (OPEN_FROM) and on forest ground (FOREST_FROM; apply_forest_ground
        # CHANGE source in FOREST_BASE, so its record stays a valid revert)
        'OPEN_FROM': ('plains', 'grassland', 'hilly_plains', 'hilly_grassland', 'desert', 'tundra', 'mountain'),
        'OPEN_TO': {'flat': 'grassland', 'hilly': 'hilly_grassland', 'desert': 'desert', 'tundra': 'tundra'},
        'FOREST_FROM': ('light_forest', 'hilly_light_forest'),
        'FOREST_TO': {'flat': 'light_forest', 'hilly': 'hilly_light_forest', 'desert': 'light_forest', 'tundra': 'light_forest'},
        'FOREST_BASE': {'light_forest': 'grassland', 'hilly_light_forest': 'hilly_grassland'},
        # Tundra (no vanilla example): RESOLVE tundra/snow, or at/above TUNDRA_ELEV m (Changtang); hilly hexes
        # there stay hilly_grassland. Open ground north of 62 N that RESOLVE calls grass/shrub stays grassland.
        'TUNDRA_BIOMES': ('tundra', 'snow'),
        'TUNDRA_ELEV': 4500,
        # Supply (user review 2026-10-25: desert/tundra cost -10 supplies, the new regions averaged -4.4 against
        # vanilla -2.75): desert only on the real desert core, RESOLVE DESERT_BIOMES at least DESERT_WATER_CLEAR
        # hexes from water (sea, lakes, map rivers, Natural Earth rivers) and DESERT_OASIS_CLEAR hexes from the
        # oases (oasis_discs x OASIS_DISC_REACH, oasis_seeds reach, oasis_areas); no desert or tundra within
        # SUPPLY_TOWN_CLEAR hexes of a town. Elsewhere the desert share goes to flat / hilly in the learned ratio.
        'DESERT_BIOMES': ('desert',),
        'DESERT_EDGE_CLEAR': 8,      # hexes from any other biome (the desert-steppe transition is not core)
        'DESERT_WATER_CLEAR': 6,
        'DESERT_OASIS_CLEAR': 4,
        'OASIS_DISC_REACH': 1.6,
        'SUPPLY_TOWN_CLEAR': 4,
        'DITHER': (4., 3, 431),    # fbm grain (hexes), octaves, seed of the uniform dither field
        'TOWN_RING': 1,            # hexes round a settlement slot / authored town kept as they are
    },
    'apply_climate': {
        # climate_map.cm / tile climate order (polish) and global blend G band, by map.hex climate name.
        'TILE_CLIMATE': {'subtropical': 0, 'cold': 1, 'temperate': 2, 'arid': 3},
        'BLEND_BAND': {'cold': 0, 'arid': 1, 'temperate': 2, 'subtropical': 3},   # measured on vanilla: 90-99% per climate
        'LAT_BINS': (15, 20, 24, 27, 30, 33, 36, 39, 42, 45),
        'ELEV_BINS': (200, 500, 1000, 1500, 2000, 3000),
        'LEARN_MIN': 100, 'LEARN_SHARE': .6,
        'WR_TOWN_KEEP': 2,
        # Wetland ecoregions (Ecoregions2017 ECO_ID): full = every flat low hex, near_water = only hexes
        # within WATER_REACH of water (the broad meadow plains are mostly dry grassland).
        'WETLAND_ECO_FULL': {
            319: 'Indochina mangroves', 321: 'Myanmar Coast mangroves', 322: 'Sunda Shelf mangroves',
            323: 'Sundarbans mangroves', 320: 'Indus River Delta-Arabian Sea mangroves',
            234: 'Irrawaddy freshwater swamp forests', 285: 'Tonle Sap freshwater swamp forests',
            286: 'Tonle Sap-Mekong peat swamp forests', 266: 'Red River freshwater swamp forests',
            224: 'Chao Phraya freshwater swamp forests', 282: 'Sundarbans freshwater swamp forests',
            264: 'Peninsular Malaysian peat swamp forests', 746: 'Suiphun-Khanka meadows and forest meadows',
            742: 'Bohai Sea saline meadow', 748: 'Yellow Sea saline meadow', 747: 'Tigris-Euphrates alluvial salt marsh'},
        'WETLAND_ECO_NEAR_WATER': {741: 'Amur meadow steppe', 743: 'Nenjiang River grassland'},
        'NEAR_WATER_ECO_MAX_ELEV': 180,   # m: the Zeya-Bureya and Songnen plains above it are dry meadow steppe
        'FULL_MAX_ELEV': 40,      # m: swamp/mangrove ecoregion hexes above it are not marsh
        'NEAR_WATER_MAX_ELEV': 250,
        'WATER_REACH': 1,         # hexes
        'FLAT_STD': 25,           # m: elevation std over the 3x3 hex neighbourhood
        'WETLAND_MATERIAL': 31,
        'BLEND_CUT': .88,       # mean 0.58 of a wetland hex painted (vanilla 0.56)
        'WET_WATER': ((0, 1, 2, 3, 5), (1., 1., .88, .68, .5)),   # FIX2 WETLAND PATCHES: hexes from water -> paint share
        'WET_NOISE': (5., 4, 1., .81),
        # Wetland pixel mask (ground contrast fix §4): the marsh paint takes a smooth weight of the wetland hexes
        # (box blur radius px; .5 on the hex edge) instead of the hard hex mask, so it may reach soft_floor of the way
        # into a neighbouring land hex: no hex-shaped outlines. (radius px, soft_floor)
        'WET_SOFT': (8, .15),
    },
}


# ------------------------------------------------------------------ regions and places
# Hard-coded region data, read back by its consumers through region(name). The geometry code
# (sand_mask, oasis_mask, oasis_seeds, the boxes' masks) stays where it is. 'consumers' names the readers;
# entries without readers are data for later steps (karez: owner review 2026-10-24, not wired yet).
REGIONS = {
    # ground_materials SAND_SEAS: erg outlines (lon, lat; real dune fields), edges blurred and fbm-warped
    'sand_seas': {'consumers': ('ground_materials.SAND_SEAS', 'audit_terrain_realism (sand_mask)'), 'polygons': {
        'Taklamakan': [(77.6, 38.6), (78.4, 39.8), (80.3, 40.5), (83.4, 40.9), (86.4, 40.7), (88.2, 40.0),
                       (88.6, 38.9), (87.2, 38.3), (84.6, 37.4), (81.6, 37.1), (79.2, 37.5)],
        'Kumtag': [(89.6, 40.0), (91.0, 40.5), (92.8, 40.4), (93.4, 39.9), (92.0, 39.4), (90.1, 39.3)],
        'Gurbantunggut': [(84.8, 44.7), (86.5, 45.7), (88.8, 46.0), (90.1, 45.4), (89.6, 44.5), (87.0, 44.4)],
        'Badain Jaran': [(100.0, 39.6), (100.8, 41.2), (102.8, 41.6), (104.0, 40.5), (103.0, 39.4), (101.0, 39.2)],
        'Tengger': [(102.8, 38.0), (103.2, 39.6), (104.8, 39.9), (105.6, 38.6), (104.3, 37.5)],
        'Otindag': [(114.0, 42.5), (116.4, 43.5), (117.6, 43.0), (116.1, 42.2)],
        'Horqin': [(119.4, 42.5), (122.4, 43.8), (123.6, 43.0), (121.0, 42.1)],
        'Khongoryn Els': [(101.4, 43.55), (104.2, 43.95), (104.3, 43.7), (101.5, 43.35)],
        'Karakum': [(54.0, 37.8), (53.5, 40.5), (57.0, 41.5), (61.0, 41.5), (63.5, 39.5), (62.0, 37.5), (58.0, 37.6)],
        'Kyzylkum': [(61.5, 41.2), (62.0, 43.5), (64.5, 44.0), (67.5, 43.2), (67.5, 41.0), (65.0, 40.0), (63.0, 40.3)],
        'Moyynkum': [(68.5, 44.0), (70.0, 45.2), (73.5, 45.1), (73.8, 44.0), (71.0, 43.5)],
        'Saryesik-Atyrau': [(74.5, 45.3), (75.0, 46.4), (78.5, 46.3), (79.5, 45.3), (77.0, 44.8)],
        'Thar': [(69.5, 24.5), (70.0, 29.0), (73.5, 30.0), (75.0, 27.5), (73.0, 24.5)],
        'Rub al Khali': [(47.0, 19.0), (48.0, 22.0), (52.0, 23.0), (55.5, 22.5), (55.0, 19.0), (51.0, 17.5)],
        'Lut': [(57.6, 29.0), (58.0, 32.0), (59.5, 31.5), (59.5, 29.0)],
    }},
    # ground_materials OASES / OASIS_AREAS: irrigated oases without a Natural Earth river in the real-terrain
    # layer (lon, lat, radius km; Ferghana an outline), farm-only today (farm_mud_fields oasis setting)
    'oasis_discs': {'consumers': ('ground_materials.OASES', 'oasis_ground.OASIS_SEEDS (shared centres)'), 'discs': {
        'Dunhuang': (94.66, 40.14, 28), 'Anxi': (95.9, 40.5, 18), 'Hami': (93.5, 42.82, 22), 'Turpan': (89.19, 42.95, 30),
        'Korla': (86.15, 41.73, 22), 'Kucha': (82.96, 41.72, 28), 'Aksu': (80.26, 41.17, 32), 'Kashgar': (75.99, 39.47, 38),
        'Yarkand': (77.25, 38.42, 38), 'Khotan': (79.92, 37.11, 32), 'Niya': (82.7, 37.07, 14), 'Charklik': (88.17, 39.03, 14),
        'Jiuquan': (98.5, 39.74, 26), 'Zhangye': (100.45, 38.93, 30), 'Wuwei': (102.64, 37.93, 30),
        'Samarkand': (66.96, 39.65, 40), 'Bukhara': (64.42, 39.77, 40), 'Merv': (61.83, 37.6, 34), 'Balkh': (66.9, 36.76, 30),
        'Khwarazm': (60.6, 41.5, 45), 'Tashkent': (69.24, 41.3, 30), 'Herat': (62.2, 34.35, 26)}},
    'oasis_areas': {'consumers': ('ground_materials.OASIS_AREAS',), 'polygons': {
        'Ferghana': [(70.55, 40.25), (71.4, 40.1), (72.4, 40.35), (73.15, 40.55), (73.35, 40.95), (72.6, 41.2),
                     (71.6, 41.15), (70.9, 41.05), (70.45, 40.65)]}},
    # oasis_ground OASIS_SEEDS (polish oasis belts): {'at': OASES disc whose centre the seed takes} or
    # {'lonlat': (lon, lat)}, 'km2' the oasis area (Hanshu 96 population x 2 ha per person, floor 50 km^2;
    # Turpan basin, Yiwu and Dunhuang at garrison-field scale; see oasis_ground)
    'oasis_seeds': {'consumers': ('oasis_ground.OASIS_SEEDS',), 'seeds': {
        # Turpan basin: REGIONS['turpan_han_oasis'] (Han water sources and the spring line), not seeds
        # Yiwu, Dunhuang commandery
        'Hami':     {'at': 'Hami', 'km2': 200, 'source': 'Yiwulu: Han garrison fields on the Hami oasis'},
        # [TURPAN2] Dunhuang commandery (Hanshu 28: 38,335 people) x 2 ha = ~770 km^2 over its six counties:
        # ~450 on the Dang fan (Dunhuang, Longle, Xiaogu), ~300 on the Shule (Guazhou: Yuanquan, Guangzhi, Mingan)
        'Dunhuang': {'at': 'Dunhuang', 'km2': 450, 'source': 'Shazhou on the Dang river fan'},
        'Guazhou':  {'at': 'Anxi', 'km2': 300, 'source': 'Guazhou/Anxi on the Shule river'},
        # Tarim rim (Hanshu states)
        'Kucha':    {'at': 'Kucha', 'km2': 1600},
        'Aksu':     {'at': 'Aksu', 'km2': 490},
        'Kashgar':  {'at': 'Kashgar', 'km2': 370},
        'Yarkand':  {'at': 'Yarkand', 'km2': 330},
        'Khotan':   {'at': 'Khotan', 'km2': 390},
        'Niya':     {'lonlat': (82.72, 37.97), 'km2': 70, 'source': 'Niya ruins (Jingjue), Stein site N, 37 58 N 82 43 E'},
        'Cherchen': {'lonlat': (85.53, 38.14), 'km2': 50, 'source': 'Qiemo county seat 38 08 N 85 32 E'},
        'Charklik': {'at': 'Charklik', 'km2': 200, 'source': 'Yuni, the Shanshan capital'},
        'Loulan':   {'lonlat': (89.92, 40.52), 'km2': 80, 'source': 'Loulan LA site 40 31 N 89 55 E'},
        'Korla':    {'at': 'Korla', 'km2': 190},
        'Yanqi':    {'lonlat': (86.57, 42.06), 'km2': 640, 'source': 'Yanqi county seat 42 04 N 86 34 E (Yuanqu)'},
    }},
    # Karez (owner review 2026-10-24, item 2): segment ends (lon, lat) of the karez-fed oasis bands, checked
    # against ETOPO 2026-10-25 (Turpan SE 94 .. -75 m, Gaochang -30 .. -44, Toksun 94 .. -8, Piqan fan 360-520).
    # [TURPAN3] not an oasis source any more: the karez are mostly post-Han (their wide spread is Qing); data kept.
    'karez': {'consumers': (), 'core_hex': .2, 'radius_hex': 1., 'lines': {
        'Turpan, south-east of the city': ((89.10, 42.98), (89.30, 42.85)),
        'Gaochang / Shengjin':            ((89.45, 42.93), (89.60, 42.85)),
        'Shanshan / Lukqun':              ((90.10, 42.95), (90.30, 42.78)),
        # [TURPAN2] Liuzhong (Lukqun) proper: the Tuyuk-gorge outflow south of the Flaming Mountains to the town
        # (ETOPO -60 .. -20 m); the line above lies on the Piqan fan (360-520 m), 40 km east of Lukqun
        'Liuzhong (Lukqun)':              ((89.64, 42.84), (89.84, 42.72)),
        'Toksun':                         ((88.55, 42.85), (88.70, 42.75)),
        'Hami':                           ((93.40, 42.85), (93.60, 42.75)),
    }},
    # [TURPAN3 2026-10-25] Turpan basin oases of 190: Han farmland ~120-200 km^2 (Jushi front 6,050 people x 2 ha,
    # the tuntian of Gaochang bi and Liuzhong), drawn as the oasis landscape around the Han water sources. Solid
    # patches (lonlat, km2: weight 1 within core x R, smoothstep to 0 at edge x R, .5 at R = sqrt(km2/pi); fbm
    # warp of the outline only) on the gorge outflows, and the spring line along the Flaming Mountains' south foot
    # as a chain of discs (radius_hex at weight .5, centres step_hex apart; stretches under 1 hex wide left out).
    # Positions checked on ETOPO 1' 2026-10-25: Jiaohe terrace 82 m between the Yarnaz branches; the Murtuk /
    # Sengim outflow leaves the range at 89.56 E 42.90 N (-12 m; crest 186-396 m north of it), Gaochang ruins -45 m;
    # Tuyuk mouth 89.69 E 42.85 N (27-136 m, crest 420-670 m; patch on its fan); Lukqun -27 m. Spring-line points 1.5-3 km south of
    # the foot (where the plain meets the range: 42.92 N at 89.2-89.45 E, 42.90 at 89.55, 42.85 at 89.65, 42.81
    # at 89.8). clean_islands: support (> support) components without a core of island_min_hex hexes or a source
    # are removed in box (lon0, lon1, lat0, lat1).
    'turpan_han_oasis': {'consumers': ('oasis_ground.han_oases', 'oasis_ground.clean_islands'),
                         'core': .7, 'edge': 1.3, 'warp': .15, 'grain_hex': 3., 'support': .1, 'island_min_hex': 4,
                         'box': (88.85, 90.05, 42.6, 43.1),
                         'patches': {
        'Jiaohe':   {'lonlat': (89.075, 42.94), 'km2': 25, 'source': 'Jiaohe ruins 42 57 N 89 04 E: terrace between the two Yarnaz branches (Jushi front capital)'},
        'Gaochang': {'lonlat': (89.545, 42.875), 'km2': 90, 'source': 'Murtuk/Sengim gorge outflow to Gaochang bi (Han tuntian), ruins 42 51 N 89 32 E'},
        'Tuyuk':    {'lonlat': (89.69, 42.83), 'km2': 30, 'source': 'Tuyuk gorge mouth 42 51 N 89 41 E; centre on its fan 2 km south (27 .. -6 m)'},
        'Liuzhong': {'lonlat': (89.76, 42.75), 'km2': 80, 'source': 'Liuzhong/Lukchun (Later Han Wuji colonel), 42.744 N 89.758 E'}},
                         'spring_lines': {
        # [TURPAN3b] Jiaohe, the Jushi front capital: its 19-hex slot cluster (88.96-89.20 E, 42.86-43.02 N) reads as an
        # oasis town (green >= .45 at the town hex, >= .15 within 1 hex of every slot) through three thin lines on the
        # Yarnaz water, not one big disc (basin union <= 500 km^2). ETOPO: branches 133 m (gorge mouth) .. 44 (terrace)
        # .. -63 m; fan 13 (west branch) .. 44 .. 45 m (Turpan side); south fan -70 .. -63 .. -51 m.
        # the Yarnaz branches on either side of the Jiaohe terrace, from the gorge mouth to the fan (fields along both)
        'Yarnaz branches': {'radius_hex': .5, 'step_hex': .25, 'points': ((89.07, 42.99), (89.075, 42.94), (89.075, 42.886))},
        # the Yarnaz water spread across its fan, west branch to the Turpan side (ETOPO 0-90 m)
        'Yarnaz fan': {'radius_hex': .5, 'step_hex': .25, 'points': ((88.966, 42.938), (89.075, 42.94), (89.185, 42.956))},
        # the spread below the terrace, where the branches rejoin on the fan
        'Yarnaz south fan': {'radius_hex': .5, 'step_hex': .25, 'points': ((89.02, 42.884), (89.075, 42.886), (89.175, 42.895))},
        'Flaming Mountains south foot': {'radius_hex': .55, 'step_hex': .25, 'points': (
            (89.41, 42.913), (89.45, 42.915), (89.53, 42.895), (89.60, 42.845), (89.66, 42.81),
            (89.72, 42.795), (89.80, 42.775), (89.86, 42.77))}}},
    # apply_climate WR box (west of Dunhuang, north of the Tibetan plateau: the plan_western_regions
    # projection): polish table climate there, cold crests >= COLD_ELEV away from towns
    'western_regions': {'consumers': ('apply_climate.WR_LON_MAX/WR_LAT_MIN/WR_LAT_MAX',), 'lon_max': 94.66, 'lat_min': 34.5, 'lat_max': 47.5,
                        'note': 'DUNHUANG_REAL lon; Kunlun to the Irtysh/Balkhash'},
    # frontier_forest / apply_forest_ground: the Western Regions hex band (west of the native map at its
    # latitudes: deserts and steppe, left sparse on purpose): q < PAD_LEFT, PAD_BOTTOM <= r < PAD_BOTTOM + NATIVE_H
    'wr_west_band': {'consumers': ('frontier_forest.plant (forest fill)', 'apply_forest_ground (forest ground)'),
                     'q_lt': 'PAD_LEFT', 'r_ge': 'PAD_BOTTOM', 'r_lt': ('PAD_BOTTOM', 'NATIVE_H')},
    # native_campaign_climate band_climate: the authored straight climate bands (values in
    # canvas-authored-coordinates.json biome_bands, migrated with the canvas: never copied here) and their wobble
    # (amplitude, frequencies of q/r) - rule #1; polish_campaign_foliage's band rule (#46) reads the same keys
    'authored_bands': {'consumers': ('native_campaign_climate.band_climate', 'polish_campaign_foliage band rule'),
                       'source': 'work/canvas-authored-coordinates.json:biome_bands',
                       'keys': ('arid_max_q', 'arid_min_r', 'tropical_max_r', 'cold_min_r', 'cold_min_q', 'foliage_arid_max_q'),
                       'wobble': {'q': ((7, .083), (4, .21, .05)), 'r': ((7, .071), (4, .19, .06))}},
    # ground_materials FARM_LAT: degrees N, grain fades out (Han-era farming frontier)
    # snow_bake perennial_snow: ETOPO 60" flattens sharp summits by 500-1,300 m (Siguniang 5,049 against 6,250), so peaks
    # fell under the perennial line. Each point: (name, lon, lat, real height m, ETOPO 60" max within 0.12 deg m), the
    # position snapped to that ETOPO maximum (<scratch>/tundra-relief-20261030/summits.py, 2026-10-30); a cone of
    # 'slope' m per km around the summit raises the elevation the perennial term reads (only near the top, no range widens).
    # SCOPE: the cones feed the snow predicates only (perennial_snow); never the LF / height bake or the landform
    # classification (that would change the mountain shapes: a separate review).
    # RULE (check_biome_profiles enforces it): a summit is listed when its real height is >= the local perennial line
    # + 'min_above_line' m and ETOPO underestimates it by >= 'min_flattening' m; candidates are the named peaks that
    # previews, reviews or the glacier atlases bring up (not chosen by fame), added by the same rule.
    'perennial_summits': {'consumers': ('biome_profiles.perennial_snow',), 'slope': 450., 'min_above_line': 150.,
                          'min_flattening': 500., 'points': (
        ('Gongga', 101.888, 29.588, 7556, 6273), ('Siguniang', 102.900, 31.118, 6250, 5049), ("Que'ershan", 98.817, 32.036, 6168, 5232),
        ('Xuebaoding', 103.847, 32.679, 5588, 4851), ('Yulong', 100.185, 27.108, 5596, 4800), ('Haba', 100.102, 27.331, 5396, 4636),
        ('Kawagebo (Meili)', 98.679, 28.437, 6740, 5612), ("Ge'nyen", 99.610, 29.796, 6204, 5342), ('Baima', 98.976, 28.302, 5430, 4929),
        ('Amne Machin', 99.463, 34.802, 6282, 5773), ('Nyainqentanglha', 90.573, 30.376, 7162, 6540), ('Namcha Barwa', 95.058, 29.638, 7782, 6772),
        ('Gyala Peri', 94.960, 29.818, 7294, 6113), ('Kangrinboqe', 81.253, 31.142, 6638, 5817),
        ('Gangshiqia (Lenglongling)', 101.675, 37.605, 5254, 4474))},
    'han_farming_frontier': {'consumers': ('ground_materials.FARM_LAT',), 'lat': (45., 49.)},
    # apply_climate WETLANDS_EXTRA: named marshes without a RESOLVE polygon (name, lon0, lon1, lat0, lat1, max metres, near water only)
    'wetlands_extra': {'consumers': ('apply_climate.WETLANDS_EXTRA',), 'boxes': (
        ('Vasyugan mire (West Siberian plain)', 75.0, 83.5, 56.0, 58.8, 150, False),
        ('Baraba lowland marshes', 76.0, 80.5, 54.6, 56.0, 150, True),
        ('Ob-Irtysh floodplain marshes', 68.0, 74.5, 58.5, 61.5, 80, True),
        ('Selenga delta (Baikal)', 105.8, 106.9, 52.0, 52.6, 520, False),
        ('Sanjiang plain (Amur-Ussuri-Songhua)', 130.8, 134.8, 46.5, 48.4, 80, False),
        ('Lower Amur lakes (Udyl, Evoron)', 136.0, 140.0, 50.5, 53.0, 60, True),
        ('Mekong delta (Plain of Reeds, U Minh)', 104.5, 106.8, 8.6, 11.2, 4, False),
        ('Irrawaddy delta', 94.2, 96.3, 15.5, 17.8, 5, False))},
    # apply_real_lakes KEEP_BOXES (lon0, lon1, lat0, lat1): lakes kept as they are
    'lake_keep': {'consumers': ('apply_real_lakes.KEEP_BOXES',), 'boxes': {
        'Lop Nur': (89.3, 91.6, 39.6, 41.3),      # 蒲昌海, a Han-era lake that is a playa today
        'Zaysan': (83.0, 85.0, 47.6, 48.6),       # 斋桑泊: natural, though Natural Earth files it as the Bukhtarma reservoir
    }},
    # polish OASIS_VALLEYS: named oasis belts along authored valleys. Dead path (runs only without
    # analysis/west-real-terrain.npz); marked for removal (owner review item 4), kept until then.
    # [BEAUTIFY G] salt crusts and yardang fields (lon, lat outlines; first draft, beautify-GH-entries.md)
    'salt_crusts': {'consumers': ('biome_profiles.classify salt_crust',), 'polygons': {
        'Lop Nur': [(89.6, 39.9), (90.2, 40.9), (91.4, 41.0), (91.6, 40.3), (90.8, 39.8)],
        'Halaqi (Shule terminal)': [(93.4, 40.3), (93.7, 40.6), (94.6, 40.6), (94.8, 40.3), (94.0, 40.15)],
        'Qarhan (Qaidam)': [(94.4, 36.7), (94.7, 37.2), (96.4, 37.2), (96.6, 36.8), (95.5, 36.6)]}},
    'yardang': {'consumers': ('biome_profiles.classify yardang',), 'polygons': {
        'Lop Nur NE (Long Cheng / Bailongdui)': [(90.6, 40.2), (91.0, 40.8), (92.2, 40.8), (92.4, 40.3), (91.4, 40.0)],
        'Yumen NW (Dunhuang Yadan)': [(92.6, 40.35), (92.8, 40.6), (93.35, 40.6), (93.4, 40.35)]}},
    # [BEAUTIFY G/H] saline lakes of Tibet and Qaidam drawn turquoise: paint_salt_pan_sea_colour paints sea_colour.dds (the water
    # colour vanilla's lake material samples; vanilla's own high lake, Xihai, is pale green-grey there) over
    # every 3k_main_sea_lake hex piece that holds a point below (lon, lat; the piece nearest within snap_deg),
    # grown by grow_tiles tile cells onto the shore; colour from rgb_shore at the shore to rgb_deep at deep_hex
    # hexes inside. Fresh lakes (Puma Yumco, Yamzho Yumco, Mapam Yumco with Langa Co, Bosten, Zaysan, Sasykkol,
    # Har Us / Har Nuur) are left out, and so are lakes outside Tibet / Qaidam (Lop Nur takes salt_crusts); Qarhan has no
    # lake hexes on the canvas (its crust takes salt_crusts).
    'salt_lakes': {'consumers': ('paint_salt_pan_sea_colour (salt-lake water colour)',),
                   'rgb_shore': (88, 170, 156), 'rgb_deep': (34, 122, 130), 'deep_hex': 5., 'grow_tiles': 2, 'snap_deg': .3,
                   'lakes': {
        'Qinghai Lake (Qinghai Hu)': (100.2, 36.9), 'Nam Co': (90.6, 30.75), 'Siling Co': (89.0, 31.8),
        'Zhari Namco': (85.62, 30.94), 'Tangra Yumco': (86.59, 31.05), 'Dagze Co': (87.52, 31.91),
        'Ngangla Ringco': (83.09, 31.55), 'Dorsoidong Co': (90.07, 33.45), 'Bangong Co': (78.69, 33.8),
        'Elsen Nur (Hoh Xil)': (92.13, 35.23), 'Hoh Xil Hu': (91.15, 35.6), 'Ayakkum Hu': (89.4, 37.55)}},
    'oasis_valleys': {'consumers': ('polish OASIS_VALLEYS',), 'status': 'dead, marked for removal, kept',
                      'weights': {'塔里木河': 1.1, '伊犁河谷': .9, '锡尔河': .8, '阿姆河': .8}},
    # polish TOWN_CLIMATE: authored towns whose vanilla climate contradicts their geography ({'region key':
    # climate}; empty: Longle was moved to the Gobi instead, move_town.py), noisy disc radii (inner, outer)
    'town_climate': {'consumers': ('polish TOWN_CLIMATE/TOWN_CLIMATE_R',), 'towns': {}, 'radius_hex': (6, 10)},
    # korea_patch forest zones (analysis/korea-patch.npz, its ground classes 11 broadleaf / 12 mixed / 13 conifer and
    # their foliage species). cold_line(lat) = clip(base - per_lat (lat - lat0), lo, hi) m: conifer above it, mixed
    # within mixed_band m below it (elevation dithered by fbm). 'off' = the values the patch was built with
    # (2026-10-24: ~400 m on the Yalu, the Jian uplands one fir carpet: window conifer .51, broadleaf .02).
    # [§9.3 Goguryeo monotony] 'on' (TW3K_KOREA_FORESTS=1, korea_forests()): the real zonation - Korean pine /
    # broadleaf mixed forest below ~1000-1100 m on the Yalu and Changbai, spruce-fir above (~1130 m at 41 N).
    # korea_patch window (P6, 2026-11-01): the land hexes of the design regions whose province is in region-traits
    # province_classes[province_class] (campaign_world.region_traits; one data source with region_gameplay_db) set the
    # west edge and the window's west / south ends, at these hex offsets from the class mask's west / south extent
    # (calibrated: the 2026-11-01 design's mask q 1571-1745, r 825-998 gives the old 1578 / 1560 / 812). The north / east
    # edges and the window's north / east ends are the native frame's (rim offsets: 1048 / 1767, 1077 / 1780).
    # Guard: every class town on the real peninsula: its tile's (lon, lat) inside `peninsula` (korea_patch.on_peninsula):
    # south of the Yalu / Tumen line (river mouth to mouth through Sinuiju, Ji'an, Hyesan, Paektu, Musan, Hoeryong, the
    # Onsong bend), closed over the sea (Jeju inside). 2026-11-01 class towns 34.9-40.35 N all inside; Xuantu (Liaodong,
    # 42.0-42.9 N) and the old North Okjeo towns (beyond the Tumen) outside. Not a fixed latitude: the Tumen bank runs to
    # 42.3-42.9 N (coordinator 11-01). Tested peninsula_margin degrees south of the town: the lower Yalu is ~5 km wide
    # between Dandong and Sinuiju, so the margin stays small (Lelang resource 1, 124.85 E 40.35 N, is inside by .03).
    'korea_patch': {'consumers': ('korea_patch.window', 'coast_tiles.korea_aligned_elevation'),
                    'province_class': 'peninsula', 'west_edge': 7, 'win_west': -11, 'win_south': -13,
                    'north_edge_below_top': 29, 'east_edge_inside': 13,
                    'peninsula_margin': .05,   # deg: a town this far north of the line still passes (bank towns); .2 let Dandong (124.39 E 40.13 N, Liaodong) through
                    'peninsula': ((124.33, 39.85), (124.40, 40.10), (124.97, 40.46), (126.19, 41.10), (128.18, 41.40), (128.07, 42.00),
                                  (129.21, 42.23), (129.75, 42.45), (130.00, 42.95), (130.70, 42.30),   # Yalu / Tumen, lon lat
                                  (131.50, 41.00), (131.50, 33.00), (124.00, 33.00), (124.00, 39.60))},   # closed over the sea
    'korea_forests': {'consumers': ('korea_patch.build (cold_line, ground classes, SPECIES)',),
                      'off': {'cold_line': (1250., 170., 36., 350., 1400.), 'mixed_band': 350.},
                      'on': {'cold_line': (1400., 90., 38., 900., 1500.), 'mixed_band': 450.},
                      'species': {   # class: (species, dense, sparse, trees per hit, jitter hexes) (foliage replanting)
                          10: (['temperate_tree_katsura_small_1', 'temperate_tree_katsura_single_1', 'temperate_tree_katsura_small_blossom'], .12, .03, 3, .9),
                          11: (['temperate_tree_katsura_medium_1', 'temperate_tree_katsura_medium_2', 'temperate_tree_katsura_large_1', 'temperate_tree_katsura_small_1'], .9, .45, 8, 1.1),
                          12: (['cold_tree_fir_medium_1', 'temperate_tree_katsura_medium_1', 'temperate_tree_metasequoia_medium_1', 'temperate_tree_katsura_large_2'], .9, .45, 8, 1.1),
                          13: (['cold_tree_fir_medium_1', 'cold_tree_fir_large_1', 'cold_tree_fir_small_1', 'cold_tree_fir_medium_2'], .95, .5, 8, 1.1),
                          14: (['general_rock_medium_1', 'general_rock_small_1', 'rock_shrub_small_1'], .2, .1, 1, .7)}},
}


def korea_forests_on():
    import os
    return os.environ.get('TW3K_KOREA_FORESTS', '0') == '1'


def korea_forests():
    """The active korea_patch forest rule: {'cold_line', 'mixed_band', 'species', 'switch'} (REGIONS korea_forests,
    'on' with TW3K_KOREA_FORESTS=1, else 'off')."""
    k = REGIONS['korea_forests']; on = korea_forests_on()
    return copy.deepcopy({**k['on' if on else 'off'], 'species': k['species'], 'switch': on})


def korea_forests_sha():
    """Fingerprint of the active korea_patch forest rule (korea-patch npz 'rules_sha'; hehuang_terrain checks it)."""
    import hashlib
    return hashlib.sha256(repr(sorted(korea_forests().items())).encode()).hexdigest()


def region(name):
    """One REGIONS entry (deep copy)."""
    return copy.deepcopy(REGIONS[name])


def oasis_seeds():
    """oasis_ground.OASIS_SEEDS: {name: (lon, lat, km^2)}; 'at' seeds take their oasis_discs centre."""
    discs = REGIONS['oasis_discs']['discs']
    return {n: (discs[s['at']][:2] if 'at' in s else s['lonlat']) + (s['km2'],) for n, s in REGIONS['oasis_seeds']['seeds'].items()}


def west_band(qq, rr, cw):
    """wr_west_band mask on hex grids qq, rr (campaign_world cw for the canvas frame)."""
    b = REGIONS['wr_west_band']
    return (qq < getattr(cw, b['q_lt'])) & (rr >= getattr(cw, b['r_ge'])) & (rr < getattr(cw, b['r_lt'][0]) + getattr(cw, b['r_lt'][1]))


# ------------------------------------------------------------------ classification rules
# Parameters of the per-consumer classifiers below (classify, dryness, arid_share, classify_gameplay);
# the types are those of the literals they replaced (ints stay ints). Box radii of the class shares and
# noise seeds stay in the code (algorithm, not tuning).
RULES = {
    # ground_materials tile_fields: steppe gravel band (Mongolian plateau, Kazakh uplands) and the erg mask
    'sgravel': {'elev': (700, 2500), 'elev_ramp': 300, 'lat_max': 50.5, 'lat_ramp': 1.5},
    'erg': {'blur': 6, 'warp': .3, 'contrast': 2.2},
    # [BEAUTIFY H] Tibet split (classify): elevation bands (m, ramp), latitude bands (deg, ramp .5), humid by
    # dryness (humid weight 1 below dry[0], 0 above dry[1])
    # RESOLVE calls the whole plateau 'alpine' (dryness 0 on Changtang), so the humid/dry split also runs on
    # longitude: humid east of humid_lon[1], dry west of humid_lon[0] (the 92 E meadow/steppe line)
    'tibet': {'meadow_elev': (3200., 5100.), 'meadow_lat': (28., 38.5), 'steppe_elev': (4000., 5400.), 'steppe_lat': (30., 37.),
              'elev_ramp': 200., 'lat_ramp': .5, 'dry': (.3, .5), 'humid_lon': (90., 92.5)},
    # [BEAUTIFY G] salt crusts below this elevation (terminal basins), polygon edges blurred over 'blur' tiles
    'salt_crust': {'max_elev': 3200., 'blur': 3}, 'yardang': {'blur': 3},
    # apply_climate classify: vanilla's own tendencies carried beyond its coverage (lat 17-43 N, < 4000 m)
    'gameplay': {'sub_lat': 31,                 # grass/other/temperate/wetland/forest south of it: subtropical
                 'forest_sub_lat': 27,          # forest south of it: subtropical
                 'tropical_cold': (27, 2000),   # tropical at/above lat and m: cold (vanilla 27-30 N above 2000 m 75-100 %)
                 'steppe_arid_elev': 1500,      # steppe at/above: arid (vanilla 1500-2000 m arid, below temperate)
                 'conifer_lat': (34, 44),       # conifer outside: cold (vanilla 27-33 N 100 %); inside at/above conifer_arid_elev: arid (Qilian 82-100 %)
                 'conifer_arid_elev': 1500,
                 'forest_cold': (27, 36, 1500),   # forest in [lat0, lat1) at/above m: cold
                 'forest_cold_lat': 45},        # forest at/above: cold (Manchurian mixed forest)
}


def classify(real_elev, real_bio, bnames, lon, lat, q, r, origin):
    """ground_materials tile_fields classes: (soft class weights per tile in evaluation order, the shared
    layers {'e', 'des', 'nc', 'cold', 'dry_high', 'tundra_c'}). Real elevation (box 1, slope -> ridge with
    STEEP), the SNOWLINE cold line (fbm fuzz), tundra (and grass north of TUNDRA_GRASS_LAT), desert in or out
    of the SAND_SEAS ergs, steppe gravel (RULES sgravel), alpine split by dryness (humid -> alpine, dry ->
    drymtn). Weights are soft (every consumer blends them); ground_w and oasis are applied by the caller."""
    import numpy as np
    import ground_materials as gm
    _box, _smooth = gm._box, gm._smooth
    sg_, er_ = RULES['sgravel'], RULES['erg']
    ix = lambda *ns: [bnames.index(n) for n in ns if n in bnames]
    e = _box(real_elev, 1)
    gy, gx = np.gradient(e); slope = np.hypot(gx, gy); del gy, gx
    ridge = _smooth((_box(slope, 2) - gm.STEEP[0]) / (gm.STEEP[1] - gm.STEEP[0]))
    fz = gm.fbm(q, r, origin, 9, 3, seed=211)
    cold = _smooth((e - gm.snowline(lat) + gm.SNOW_FUZZ * fz) / gm.SNOW_RAMP)
    tundra_c = np.isin(real_bio, ix('tundra')) | (np.isin(real_bio, ix('grass')) & (lat >= gm.TUNDRA_GRASS_LAT))
    tundra = _box(tundra_c, 3) * (1 - cold)
    des = _box(np.isin(real_bio, ix('desert')), 3)
    dry_high = _box(np.isin(real_bio, ix(*gm.DRY_HIGH)), 3)
    sand = _smooth((_box(gm.sand_mask(lon, lat), er_['blur']) - .5 + er_['warp'] * gm.fbm(q, r, origin, 10, 3, seed=212)) * er_['contrast'] + .5)
    stp = _box(np.isin(real_bio, ix('steppe')), 3)
    sg = stp * _smooth((e - sg_['elev'][0]) / sg_['elev_ramp']) * _smooth((sg_['elev'][1] - e) / sg_['elev_ramp']) * _smooth((sg_['lat_max'] - lat) / sg_['lat_ramp'])
    nc = 1 - cold
    # [OASIS GROUND] the alpine meadow mix only where the alpine is humid (dryness 0: Ili, Yulduz, north
    # Tian Shan, Altai, eastern Qilian); dry alpine (Bogda south, Kucha/Aksu north, Kunlun, western Qilian)
    # takes the dry-mountain mix.
    alp = _box(np.isin(real_bio, ix('alpine')), 3) * nc; dry = dryness(real_bio, bnames, real_elev)
    # [TURPAN2 2026-10-25] the cold band splits by dryness too: dry high ground (Bogda's sunny crest, the Altun
    # and Danghe Nanshan above Dunhuang, the Kunlun: bare rock and scree) takes the dry-mountain mix, the humid
    # (Tian Shan north side, Qilian east, Altai) keeps the high/plateau mixes; the G cold band is unchanged.
    # [BEAUTIFY H] Tibet: alpine meadow (humid) and alpine steppe (dry, flat) take over plateau, alpine and the
    # dry cold band in their elevation/latitude bands; high (steep humid cold) keeps its ground
    tb = RULES['tibet']
    band = lambda v, lo, hi, ramp: _smooth((v - lo) / ramp) * _smooth((hi - v) / ramp)
    hm = np.minimum(1 - _smooth((dry - tb['dry'][0]) / (tb['dry'][1] - tb['dry'][0])),
                    _smooth((lon - tb['humid_lon'][0]) / (tb['humid_lon'][1] - tb['humid_lon'][0])))
    high = cold * ridge * dry_high * (1 - dry)
    am = (_box(np.isin(real_bio, ix('alpine')), 3) * band(e, *tb['meadow_elev'], tb['elev_ramp'])
          * band(lat, *tb['meadow_lat'], tb['lat_ramp']) * hm * (1 - high))
    ast = (_box(np.isin(real_bio, ix('alpine', 'desert')), 3) * band(e, *tb['steppe_elev'], tb['elev_ramp'])
           * band(lat, *tb['steppe_lat'], tb['lat_ramp']) * (1 - hm) * (1 - ridge) * (1 - high))
    tib = np.clip(am + ast, 0, 1)
    # [BEAUTIFY G] salt crusts (terminal basins), then yardang fields, take over erg/gobi/drymtn
    sc_, yd_ = RULES['salt_crust'], RULES['yardang']
    salt = _box(poly_mask(lon, lat, REGIONS['salt_crusts']['polygons']), sc_['blur']) * _smooth((sc_['max_elev'] - e) / 200) * nc
    yard = _box(poly_mask(lon, lat, REGIONS['yardang']['polygons']), yd_['blur']) * (1 - salt) * nc
    xs = np.clip(1 - salt - yard, 0, 1)
    w = {'high': high, 'plateau': cold * (1 - ridge) * dry_high * (1 - dry) * (1 - tib), 'tundra': tundra,
         'gobi': des * (1 - sand) * (1 - ridge) * nc * xs,
         'drymtn': des * ridge * nc * xs + (alp * dry + cold * dry_high * dry) * (1 - tib), 'sgravel': sg * (1 - des) * nc,
         'alpine': alp * (1 - dry) * (1 - tib), 'erg': des * sand * nc * xs,
         'alpine_meadow': am, 'alpine_steppe': ast, 'salt_crust': salt, 'yardang': yard}
    assert tuple(w) == tuple(LANDFORMS)
    return w, {'e': e, 'des': des, 'nc': nc, 'cold': cold, 'dry_high': dry_high, 'tundra_c': tundra_c}


def poly_mask(lon, lat, polygons):
    """lon/lat outlines (REGIONS polygons) -> float32 mask (ground_materials._in_poly)."""
    import numpy as np
    import ground_materials as gm
    m = np.zeros(lon.shape, bool)
    for poly in polygons.values():
        p = np.asarray(poly); b = (lon >= p[:, 0].min()) & (lon <= p[:, 0].max()) & (lat >= p[:, 1].min()) & (lat <= p[:, 1].max())
        ii, jj = np.nonzero(b)
        if len(ii):
            m[ii, jj] |= gm._in_poly(lon[ii, jj], lat[ii, jj], poly)
    return m.astype(np.float32)


def relief_specs():
    """{landform: {'blend', 'mix' (None = the global RELIEF parts), 'noise' (scale hexes, amp, seed) or None}}:
    landforms whose own mix is graded by the real relief (ground_materials.tile_fields)."""
    out = {c: copy.deepcopy(v['relief']) for c, v in LANDFORMS.items() if v.get('relief')}
    for c in CONSTS['ground_materials']['RELIEF']['classes']:
        out.setdefault(c, {'blend': CONSTS['ground_materials']['RELIEF']['classes_blend'], 'mix': None})
    return out


# ================================================================== RULE PREDICATES (bake design step 2)
# One function per rule; generators (ground_materials, foliage, props), filters and audit gates call these
# and read the same persisted inputs (landform_tiles.py: analysis/landform-tiles.npz), so a rule's definition
# cannot fork between who paints and who checks. numpy is imported inside (the table stays import-light).
# The constants they read keep their names (COLD_LINES, CONSTS ground_materials, AUDIT, RELIEF, REGIONS).

# Landforms whose AUDIT 'banned' slots ground_materials redraws (the stop-loss of 2026-10-26: desert only;
# the other landforms' bans stay audit-only until the ground-blend convergence, step 4). Gates count these.
BAN_ENFORCED = ('gobi', 'erg', 'drymtn', 'salt_crust', 'yardang')
# [§9.3 tundra light flecks] the take dither (tundra gain .9) keeps ~10% of the donor base on real tundra, mostly
# arid_1/arid_2 sand (v20 = v21 pixel for pixel; canvas tundra tiles .071 arid_1/2, temperate 27-30 .012): the
# pale 'camouflage' blobs of art review tundra-mid. TW3K_TUNDRA_BAN=1 enforces the tundra ban (AUDIT temperate +
# TUNDRA_BAN_EXTRA): those pixels are redrawn from the tundra mix like the desert bans. Default off (byte-identical).
# 2026-11-01: + arid_3 (2) and cold_2 (5). Without them in the mix (TUNDRA_SCREE) the donor base the take dither keeps
# still left arid_3 .004-.0085 and cold_2 .0014-.0027 on tundra tiles (window replica on v22b, tundra-live-20261101);
# the invariants tundra_arid3 / tundra_cold2_treeless want 0. Ban-enforced tundra tiles carry no forest density.
TUNDRA_BAN_EXTRA = (0, 1, 2, 5)


def tundra_ban_on():
    import os
    return os.environ.get('TW3K_TUNDRA_BAN', '0') == '1'


if tundra_ban_on():
    BAN_ENFORCED = BAN_ENFORCED + ('tundra',)
    AUDIT['tundra']['banned'] = tuple(sorted(set(AUDIT['tundra']['banned']) | set(TUNDRA_BAN_EXTRA)))

# [§9.3 tundra light flecks, part 2] the tundra mix's own arid_3 (mix .14, relief crest arid_3 .45 + arid_4 .30) draws
# pale sand blobs of the material grain (pale share 0-3/19/20 on tundra tiles: v21 .15, with TW3K_TUNDRA_BAN .077).
# TW3K_TUNDRA_SCREE=1: arid_3 down to .04 in the mix, crests mostly cold_4/mud/steppe_3 (lichen, frost-shattered
# brown). Default off (byte-identical).
# 2026-11-01: no arid_3 at all (mix .04 -> cold_4, crest .20 -> cold_4). The v22b live check (b838596c) still showed
# the camouflage: aligned to the pack's R, the live pale-beige pixels sit on arid_3 (65 % of its pixels pale against
# 1-11 % on cold_4/mud/steppe_3/4); the .024 left on tundra tiles came as hard-edged 1-4 world-unit blobs, contrast
# ~130 (sum RGB) to the grass (<scratch>/tundra-live-20261101/README.md). Invariant tundra_arid3 (= 0).
# Same day, no cold_2 (.08 -> cold_4): the woodland floor (WOODLAND_SLOTS) drew hard-edged dark-green spots on open
# tundra (2 % of the texels, live v22b: the user asked what they were). Ban-enforced tundra tiles carry no forest
# density (fd >= .2 on 0 of 1.06 M tiles), so the tundra mix never sits under a forest; forest edges keep the
# taiga donor's cold_1/cold_2. Invariant tundra_cold2_treeless (= 0).
# Round 3 (coordinator, 11-01): the freed shares (mix .12, crest .20) all on cold_4 clumped into dark blobs (wide
# windows 3.7-4.4 dark blobs per 100 world^2, vanilla median 3.59). They are now split by the vanilla 190 share of
# the remaining materials on G=cold ground (global_blend, every 2nd texel): cold_4 .644, steppe_3 .252, mud .079,
# steppe_4 .025 (<scratch>/tundra-live-20261101/round3).
TUNDRA_SCREE = {'mix': {14: .130, 7: .397, 22: .223, 21: .250}, 'crest': {7: .479, 14: .266, 21: .250, 22: .005}}


def tundra_scree_on():
    import os
    return os.environ.get('TW3K_TUNDRA_SCREE', '0') == '1'


if tundra_scree_on():
    LANDFORMS['tundra']['mix'] = dict(TUNDRA_SCREE['mix'])
    LANDFORMS['tundra']['relief']['mix']['crest'] = dict(TUNDRA_SCREE['crest'])



# [GROUND MOTTLE step 1] the v23 ground recipe as a named version (the rollback): TW3K_GROUND_RECIPE=v23 puts these values
# back (snapshot byte-identical to the frozen v23 copy, analysis/biome-profiles-frozen-v23.json; check_biome_profiles
# checks that copy under the switch). Landform entries: mix and own relief (None = none) as they were; consts: the
# values of CONSTS[module][key]. Polish's setup memo keys the recipe name (ground_recipe()).
GROUND_RECIPE_DEFAULT = 'v24'
GROUND_RECIPES = ('v23', 'v24')
GROUND_RECIPE_V23 = {
    'landforms': {
        'high': {'mix': {4: .58, 5: .12, 6: .10, 7: .12, 22: .08}, 'relief': None},
        'plateau': {'mix': {2: .32, 14: .10, 21: .18, 3: .20, 22: .06, 7: .14}, 'relief': None},
        'gobi': {'mix': {1: .47, 3: .20, 2: .18, 0: .14, 14: .01},
                 'relief': {'blend': .5, 'noise': (40., .35, 213),
                            'mix': {'valley': {1: .45, 0: .25, 2: .20, 3: .10}, 'slope': {3: .45, 2: .40, 1: .15},
                                    'crest': {3: .50, 2: .45, 14: .05}}}},
        'drymtn': {'mix': {2: .36, 3: .32, 1: .16, 21: .12, 14: .04}, 'relief': None},
        'sgravel': {'mix': {21: .40, 22: .35, 2: .15, 3: .10}, 'relief': None},
        'alpine': {'mix': {4: .30, 22: .20, 21: .16, 7: .14, 6: .08, 5: .06, 20: .06}, 'relief': None},
        'alpine_meadow': {'mix': {21: .30, 22: .15, 7: .15, 19: .15, 14: .10, 6: .10, 27: .05}, 'relief': {'blend': .5, 'mix': None}},
        'alpine_steppe': {'mix': {19: .35, 2: .30, 3: .20, 20: .10, 21: .05},
                          'relief': {'blend': .4, 'mix': {'valley': {19: .50, 20: .30, 21: .20}, 'slope': {19: .50, 2: .30, 20: .20},
                                                          'crest': {2: .50, 3: .50}}}},
    },
    'consts': {('ground_materials', 'GRAIN'): (4., 3., 4.), ('farm_fields', 'ARID_OWN_OASIS'): 6,
               ('ground_materials', 'ARID_TOWN'): (1.2, 2.5, 4.5, 9.)},
}


def ground_recipe():
    """The ground recipe: TW3K_GROUND_RECIPE (default GROUND_RECIPE_DEFAULT; 'v23' = the recipe before ground mottle step 1)."""
    import os
    v = os.environ.get('TW3K_GROUND_RECIPE', GROUND_RECIPE_DEFAULT)
    if v not in GROUND_RECIPES:
        raise ValueError(f'TW3K_GROUND_RECIPE={v!r}: one of {GROUND_RECIPES}')
    return v


if ground_recipe() == 'v23':
    for _c in LANDFORMS:
        LANDFORMS[_c].pop('grain', None)
        LANDFORMS[_c].pop('redraw', None)
    for _c, _v in GROUND_RECIPE_V23['landforms'].items():
        LANDFORMS[_c]['mix'] = copy.deepcopy(_v['mix'])
        if _v['relief'] is None:
            LANDFORMS[_c].pop('relief', None)
        else:
            LANDFORMS[_c]['relief'] = copy.deepcopy(_v['relief'])
    for (_m, _k), _v in GROUND_RECIPE_V23['consts'].items():
        CONSTS[_m][_k] = copy.deepcopy(_v)
    CONSTS['ground_materials'].pop('FARM_OASES_SKIP')   # [ARID FARM] every disc waters fields again



def tundra_oasis_off_on():
    """TW3K_TUNDRA_OASIS_OFF=1: polish's oasis belts (river x tile climate 3 x low valley, wetland, lake ring) skip the
    'no_oasis' classes and its alpine (keep) belts the 'no_alpine_belt' classes, grass north of TUNDRA_GRASS_LAT
    counting as tundra (classify tundra_c). Tundra's polish climate is 3 (arid) and arctic lowlands are low, so arctic
    rivers took desert oasis belts: the oasis zeroes the tundra rule (tile_fields gw (1 - 4 oasis)) and the pixels kept
    the arid/temperate donor, the pale river corridor of tundra-mid (v22b live; 63,720 tundra oasis tiles). North of NO_OASIS_LINE only. Default off (byte-identical)."""
    import os
    return os.environ.get('TW3K_TUNDRA_OASIS_OFF', '0') == '1'


# North of the cold line only (coordinator 11-01): the RESOLVE 'tundra' of the Indus valley (Ladakh/Zanskar, 30-34 N,
# 1.2-2.9 km) holds real oases (Leh) and keeps them. The line is COLD_LINES NORTH_COLD (apply_climate's continental
# cold north of 47 N), fuzzed by fbm in degrees so the gate edge meanders (no straight latitude cut); Sayan/Altai
# (50-60 N) and the arctic fall north of it. ladakh_box: invariant tundra_oasis checks the gate never reaches it.
NO_OASIS_LINE = {'line': 'NORTH_COLD', 'fuzz_deg': 1.5, 'scale': 30., 'octaves': 3, 'seed': 143,
                 'ladakh_box': ((74., 80.), (30., 36.))}


def no_oasis_masks(eco, bnames, lat, q, r, origin):
    """(no-oasis, no-alpine-belt) bool masks per tile for TW3K_TUNDRA_OASIS_OFF: the classes (and grass north of
    TUNDRA_GRASS_LAT) north of the fuzzed NO_OASIS_LINE. q, r: tile coordinates / 2 (polish), origin: noise origin."""
    import numpy as np
    from terrain_noise import fbm
    s = NO_OASIS_LINE
    north = lat + s['fuzz_deg'] * fbm(q, r, origin, s['scale'], s['octaves'], seed=s['seed']) >= COLD_LINES[s['line']]
    ix = lambda flag: [bnames.index(n) for n in biome_set(flag) if n in bnames]
    arctic_grass = (eco == bnames.index('grass')) & (lat >= CONSTS['ground_materials']['TUNDRA_GRASS_LAT']) if 'grass' in bnames else False
    return (np.isin(eco, ix('no_oasis')) | arctic_grass) & north, (np.isin(eco, ix('no_alpine_belt')) | arctic_grass) & north


def treeline_blend_on():
    """TW3K_TREELINE_BLEND=1: polish blends the donor climate shares across the tundra/forest class line (CONSTS
    polish TREELINE_DONOR, ground_donor.treeline_weight / treeline_pick, section 9.3). Default off: the blend byte-identical."""
    import os
    return os.environ.get('TW3K_TREELINE_BLEND', '0') == '1'


def taiga_donor_on():
    """TW3K_TAIGA_DONOR=1: polish compresses the donor choice on CONSTS polish DONOR_COMPRESS biomes
    (ground_donor.compress_choice, section 9.3). Default off: donor_choice and the blend byte-identical."""
    import os
    return os.environ.get('TW3K_TAIGA_DONOR', '0') == '1'


def snowline_m(lat):
    """Cold line (m) by latitude, COLD_LINES SNOWLINE (ground_materials cold band; no fuzz)."""
    import numpy as np
    return np.interp(lat, *COLD_LINES['SNOWLINE']).astype(np.float32)


def treeline_m(lat):
    """Treeline (m) by latitude, COLD_LINES TREELINE (ecotone_bands, foliage TREELINE ALPINE)."""
    import numpy as np
    return np.interp(lat, *COLD_LINES['TREELINE']).astype(np.float32)


def cold_weight(e, lat, fuzz):
    """0..1 above the snow line (classify's cold): smoothstep of (e - snowline + SNOW_FUZZ * fuzz) / SNOW_RAMP.
    e: box-1 real elevation (m), fuzz: the fbm field (classify seed 211) or 0."""
    import ground_materials as gm
    k = CONSTS['ground_materials']
    return gm._smooth((e - snowline_m(lat) + k['SNOW_FUZZ'] * fuzz) / k['SNOW_RAMP'])


def _smoothstep(x):
    import numpy as np
    x = np.clip(x, 0, 1)
    return (x * x * (3 - 2 * x)).astype(np.float32)


def perennial_snowline_m(lat, lon, e_region=None):
    """Perennial snowline (m): COLD_LINES PERENNIAL_SNOW by latitude, lowered on the monsoon side (CONSTS snow_bake
    MONSOON_*: a smooth longitude x latitude ramp, no polygon edge); with e_region (snow_reference_elev's regional mean)
    raised over the dry plateau interior (DRY_RAISE / DRY_REGION)."""
    import numpy as np
    k = CONSTS['snow_bake']
    (l0, dl), (a1, da) = k['MONSOON_LON'], k['MONSOON_LAT']
    wet = _smoothstep((np.asarray(lon, np.float32) - l0) / dl) * _smoothstep((a1 - np.asarray(lat, np.float32)) / da)
    line = np.interp(lat, *COLD_LINES['PERENNIAL_SNOW']) - k['MONSOON_DROP'] * wet
    if e_region is not None:
        line = line + k['DRY_RAISE'] * _smoothstep((np.asarray(e_region, np.float32) - k['DRY_REGION'][0]) / k['DRY_REGION'][1])
    return line.astype(np.float32)


def summit_elevation(lat, lon):
    """Elevation (m) of the named-summit cones (REGIONS perennial_summits: real height - slope x distance km), -inf away
    from them; perennial_snow reads max(e, this)."""
    import numpy as np
    rg = REGIONS['perennial_summits']
    lat = np.asarray(lat, np.float32); lon = np.asarray(lon, np.float32)
    out = np.full(lat.shape, -np.inf, np.float32)
    for _, lo, la, h, _etopo in rg['points']:
        r = (h - 2000.) / rg['slope'] / 111.2 / max(np.cos(np.radians(la)), .2)   # degrees of longitude reach
        m = (np.abs(lon - lo) < r) & (np.abs(lat - la) < r)
        if m.any():
            d = 111.2 * np.hypot((lon[m] - lo) * np.cos(np.radians(la)), lat[m] - la)
            out[m] = np.maximum(out[m], h - rg['slope'] * d)
    return out


def perennial_snow(e, lat, lon, fuzz=0., e_region=None):
    """0..1 year-round snow (snow bake: all four mask channels): smoothstep of (e - perennial line + PERENNIAL_FUZZ x
    fuzz) / PERENNIAL_RAMP. e: real elevation (m), raised to the named-summit cones (summit_elevation: ETOPO 60"
    flattens the sharp peaks); fuzz: the ground's fbm field (-1..1) or 0."""
    import numpy as np
    k = CONSTS['snow_bake']
    e = np.maximum(e, summit_elevation(lat, lon))
    return _smoothstep((e - perennial_snowline_m(lat, lon, e_region) + k['PERENNIAL_FUZZ'] * fuzz) / k['PERENNIAL_RAMP'])


def _treeline_w(e, lat, fuzz):
    k = CONSTS['snow_bake']
    return _smoothstep((e - treeline_m(lat) + k['TREE_FUZZ'] * fuzz) / k['TREE_RAMP'])


def shoulder_snow(e, lat, lon, fuzz=0.):
    """0..1 spring/autumn snow (channel B), vanilla's northern high ground: above the treeline, north of
    SHOULDER_LAT, and the perennial snow."""
    import numpy as np
    k = CONSTS['snow_bake']
    la = _smoothstep((np.asarray(lat, np.float32) - k['SHOULDER_LAT'][0]) / (k['SHOULDER_LAT'][1] - k['SHOULDER_LAT'][0]))
    return np.maximum.reduce([_treeline_w(e, lat, fuzz), la, perennial_snow(e, lat, lon, fuzz)]).astype(np.float32)


def plateau_dryness(e, lat, lon):
    """0..1 winter dryness of the high plateau the desert dryness misses (inner Tibet, Kunlun, Pamir are alpine steppe,
    not desert): elevation ramp PLATEAU_DRY 'elev' x south of 'lat' (Tian Shan and the north keep their winter). The
    monsoon side is no exception: its rain falls in summer, winter is the dry season on the whole plateau and in
    Hengduan (thin snow and bare ground; the peaks stay white through perennial_snow). lon: unused, kept for callers."""
    import numpy as np
    pd = CONSTS['snow_bake']['PLATEAU_DRY']
    lat = np.asarray(lat, np.float32)
    return (_smoothstep((np.asarray(e, np.float32) - pd['elev'][0]) / pd['elev'][1]) *
            _smoothstep((pd['lat'][0] - lat) / pd['lat'][1])).astype(np.float32)


def snow_coverage_on():
    """TW3K_SNOW_COVERAGE=1: winter_snow returns the coverage share (CONSTS snow_bake COVERAGE; snow_bake
    quantizes it to a binary mask). Default off (the old intensity predicate, byte-identical)."""
    import os
    return os.environ.get('TW3K_SNOW_COVERAGE', '0') == '1'


def snow_reference_elev(e, cell_km):
    """(local mean, regional mean) of the real elevation (m) for winter_snow's coverage terms: two box blurs over
    COVERAGE local_km and region_km (cell_km: km per grid cell of e; ~2.2 for the tile grid)."""
    import numpy as np
    from scipy import ndimage
    c = CONSTS['snow_bake']['COVERAGE']; e = np.asarray(e, np.float32); out = []
    for km in (c['local_km'], c['region_km']):
        n = 2 * max(1, int(round(km / cell_km))) + 1
        out.append(ndimage.uniform_filter(ndimage.uniform_filter(e, n, mode='nearest'), n, mode='nearest').astype(np.float32))
    return tuple(out)


def winter_snow_coverage(e, lat, lon, fuzz=0., dry=None, e_local=None, e_region=None):
    """0..1 share of the ground white in winter (TW3K_SNOW_COVERAGE): the largest of the cold ground x (1 - winter_dry
    x plateau dryness of the region), the mountain band graded up to the perennial line (on the dry plateau only where
    the ground stands above its local mean) and the latitude curve x (1 - winter_dry x dry); then the perennial snow.
    e_local / e_region: snow_reference_elev (None: e itself, i.e. no relief term and the point's own dryness)."""
    import numpy as np
    k = CONSTS['snow_bake']; c = k['COVERAGE']
    e = np.asarray(e, np.float32)
    el = e if e_local is None else e_local; er = e if e_region is None else e_region
    la = np.interp(lat, *k['WINTER_LAT']).astype(np.float32)
    start = treeline_m(lat) - k['WINTER_MOUNTAIN'][0] - k['TREE_FUZZ'] * fuzz
    mw = _smoothstep((e - start) / np.maximum(perennial_snowline_m(lat, lon, e_region) - start, 300.))
    pd = plateau_dryness(er, lat, lon)
    relw = _smoothstep((e - el - c['rel'][0]) / c['rel'][1])
    ss = c['stable_snow']
    rr = np.hypot((np.asarray(lon, np.float32) - ss['centre'][0]) / ss['radius'][0], (np.asarray(lat, np.float32) - ss['centre'][1]) / ss['radius'][1])
    stable = _smoothstep((1 - rr) / ss['rim'])   # 1 inside the Junggar ellipse, 0 outside its rim
    ld = pd if dry is None else np.maximum(pd, np.clip(dry, 0, 1) * (1 - stable))
    w = c['winter_dry']
    a = np.maximum.reduce([cold_weight(e, lat, fuzz) * (1 - w * pd), mw * (1 - pd + pd * relw), la * (1 - w * ld)])
    return np.maximum(a, perennial_snow(e, lat, lon, fuzz, e_region)).astype(np.float32)


def winter_snow(e, lat, lon, fuzz=0., dry=None):
    """0..1 winter snow (channel A) on canvas land: the largest of the cold ground (cold_weight: COLD_LINES SNOWLINE),
    the mountain band (WINTER_MOUNTAIN: from 1,000 m below the treeline) and the continental-winter latitude curve (WINTER_LAT: (lats, intensities), vanilla's
    north-China level near the native border, full in Siberia), all three times 1 - WINTER_DRY x dry on dry ground
    (dry 0..1, e.g. oasis_ground.dryness, which sees the desert share only, maxed with plateau_dryness: the dry
    plateau and basins a light dusting, the monsoon-side mountains full); then the perennial snow
    (A contains the peaks). shoulder_snow (B) is not part of it (user 10-30: snow only in winter besides the peaks)."""
    import numpy as np
    k = CONSTS['snow_bake']
    la = np.interp(lat, *k['WINTER_LAT']).astype(np.float32)
    mw = _smoothstep((e - treeline_m(lat) + k['WINTER_MOUNTAIN'][0] + k['TREE_FUZZ'] * fuzz) / k['WINTER_MOUNTAIN'][1])
    # [10-30 Bogda] the desert dryness (blurred desert share of the basins around) thins the lowland latitude term only:
    # mountain snow is orographic (Bogda's north slope under the Turpan / Junggar blur had .59); the mountains' own
    # dryness is plateau_dryness, on every term
    pd = plateau_dryness(e, lat, lon)
    ld = pd if dry is None else np.maximum(pd, np.clip(dry, 0, 1))
    w = k['WINTER_DRY']
    a = np.maximum.reduce([cold_weight(e, lat, fuzz) * (1 - w * pd), mw * (1 - w * pd), la * (1 - w * ld)])
    return np.maximum(a, perennial_snow(e, lat, lon, fuzz)).astype(np.float32)


def above_treeline(elev, lat):
    """bool: real elevation (m) above the latitude treeline."""
    return elev > treeline_m(lat)


def in_set(real_bio, bnames, flag):
    """bool per tile: the RESOLVE class is a member of biome set `flag` (BIOMES flags; e.g. the forest,
    desert, steppe-donor sets)."""
    import numpy as np
    return np.isin(real_bio, [bnames.index(n) for n in biome_set(flag) if n in bnames])


def region_mask(name, lon, lat):
    """float32 mask of REGIONS[name]['polygons'] (lon/lat outlines)."""
    return poly_mask(lon, lat, REGIONS[name]['polygons'])


def ban_table(enforced=None):
    """(len(LANDFORMS) + 1, 256) bool: row k + 1 = the slots banned and redrawn on landform k (AUDIT
    'banned', landforms in `enforced`, default BAN_ENFORCED); row 0 (no dominant landform) bans nothing.
    TW3K_ENFORCE_BANNED=0 returns it empty (pre-step-6 output)."""
    import os
    import numpy as np
    enforced = BAN_ENFORCED if enforced is None else enforced
    ban = np.zeros((len(LANDFORMS) + 1, 256), bool)
    if os.environ.get('TW3K_ENFORCE_BANNED', '1') != '0':
        for i, n in enumerate(LANDFORMS):
            if n in enforced and AUDIT[n]['banned']:
                ban[i + 1, list(AUDIT[n]['banned'])] = True
    return ban


def dominant(w):
    """(index, weight): per tile, 1 + index (LANDFORMS order) of the landform with the largest weight in w
    ({landform: weight}, LANDFORMS order) and that weight; ban and gates use index where weight > BAN_MIN."""
    import numpy as np
    names = list(w)
    assert names == list(LANDFORMS), names
    Wst = np.stack([w[n] for n in names])
    k = Wst.argmax(0); top = np.take_along_axis(Wst, k[None], 0)[0]; del Wst
    return (k + 1).astype(np.uint8), top.astype(np.float32)


def relief_parts(e, real_bio, bnames, river, raw):
    """Relief rule (RELIEF): (open weight, {valley_humid, valley_dry, slope, crest: share}) per tile. Open biomes
    (blurred share) left over by the landform classes; valley = .5 + the local depression below the valley_r
    mean (rh_ref metres) plus the river share, crest = the opposite on steep ground, slope = the rest; the
    valley splits humid/dry by the humid-biome share times (1 - dryness)."""
    import numpy as np
    import ground_materials as gm
    _box, _smooth = gm._box, gm._smooth
    R = CONSTS['ground_materials']['RELIEF']
    ix = lambda ns: [bnames.index(n) for n in ns if n in bnames]
    open_ = _box(np.isin(real_bio, ix(R['biomes'])), R['blur'])
    humid = _box(np.isin(real_bio, ix(R['humid'])), R['blur'])
    taken = np.clip(sum(raw.values()), 0, 1)
    wo = (open_ * (1 - taken) * R['gain']).astype(np.float32)
    mean = _box(_box(e, R['valley_r']), R['valley_r'])
    rh = (mean - e) / R['rh_ref']
    if river is not None:
        rh = rh + R['river_gain'] * _box(np.asarray(river, np.float32), 2)
    gy, gx = np.gradient(e)
    sl = _smooth((np.hypot(gx, gy) - R['slope_ref'][0]) / (R['slope_ref'][1] - R['slope_ref'][0])); del gy, gx
    # flat ground (rh 0) is half valley, more where humid: plains read by their humidity (green meadow, dry
    # steppe), not as slope; contrast sharpens the parts into sheets (the edges stay graded)
    h = np.clip(humid / np.maximum(open_, 1e-6), 0, 1) * (1 - dryness(real_bio, bnames))
    k = R['contrast']; hb = R['humid_bias'] * h
    valley = _smooth(.5 + k * (rh + hb)); crest = _smooth(.5 - k * (rh + hb)) * sl
    slope = np.clip(1 - valley - crest, 0, 1)
    tot = np.maximum(valley + crest + slope, 1e-6)
    parts = {'valley_humid': valley * h / tot, 'valley_dry': valley * (1 - h) / tot, 'slope': slope / tot, 'crest': crest / tot}
    return wo, {k: v.astype(np.float32) for k, v in parts.items()}


def allowed_slots(shape, riparian=None):
    """uint32 per tile: bit s set = slot s is whitelisted there even where the dominant landform bans it.
    The oasis belts and the alpine belts are whitelisted for every slot through the ban reach (tile_fields:
    reach x (1 - 4 oasis) x (1 - 4 keep) drops their dominant landform below BAN_MIN); this field carries the
    per-slot exceptions on top (none in this package; the riparian belts' steppe_3/4, Dunhuang item 4, come here)."""
    import numpy as np
    out = np.zeros(shape, np.uint32)
    if riparian is not None:
        out[np.asarray(riparian) > 0] |= np.uint32((1 << 21) | (1 << 22))
    return out


# [FILL FLOOR] (BIOMES set 'fill_floor': taiga, conifer) the frontier fill fades forest_density to 0 beyond 30-50 hexes of playable land, so far taiga had no
# density: bare ground under the woodland rule and trees only from the tree list's own spread (0.18 per hex, 128 k on
# 699 k far-taiga hexes; woodland_off_forest .268 -> .341 with TAIGA_DONOR). TW3K_FOREST_FILL_FLOOR=1 keeps the fill
# at least FOREST_FILL_FLOOR (frontier_forest.fill_weight, the one fade both forest_fill (ground and the shared
# density) and fill_mask (tree fill) read), so the ground (donor woodland) and the props bake read one weight there.
# f = .15 (user tier, 2026-11-01: about +90 k trees). Native rectangle / native_keep / the legacy west band stay 0
# (forest_fill and the tree bake exclude them after the fade). Default off (byte-identical).
FOREST_FILL_FLOOR = .15


def forest_fill_floor_on():
    import os
    return os.environ.get('TW3K_FOREST_FILL_FLOOR', '0') == '1'


def fill_floor(eco, bnames):
    """float32 per cell: FOREST_FILL_FLOOR on the 'fill_floor' classes (taiga, conifer) with TW3K_FOREST_FILL_FLOOR,
    else None (no floor). eco: RESOLVE class indices (hex grid for the tree fill, tile grid for forest_fill)."""
    import numpy as np
    if not forest_fill_floor_on():
        return None
    ix = [bnames.index(n) for n in biome_set('fill_floor') if n in bnames]
    return np.where(np.isin(eco, ix), np.float32(FOREST_FILL_FLOOR), np.float32(0)).astype(np.float32)


def forest_fill(shape, cells=None, names=None, domains=None, real_bio=None, bnames=None):
    """float32 0-1 per tile (rows south-up): the frontier fill fade of forest_density, i.e. frontier_forest.
    fill_weight of the playable land (FILL FADE 30-50 hexes) at each tile's hex. Legacy calls exclude the Western
    Regions band (west_band) and the native rectangle; shared domains use the ecology field continuously across
    the asset rectangle and protect native_keep. real_bio/bnames (tile grid): the [FILL FLOOR] classes keep
    FOREST_FILL_FLOOR on land before the exclusions (fill_floor; the tree fill reads the same floor per hex). Target fill is not multiplied
    by vegetation_takeover: the consuming bake applies that ownership fade once. The no-domains legacy path
    retains the original arithmetic and exclusion. The fill is a target, not an edit or height alpha."""
    import numpy as np
    import campaign_world as cw
    TH, TW = shape
    out = np.ones(shape, np.float32)
    ty, tx = np.mgrid[:TH, :TW]
    hq, hr = cw.tile_hex(tx, ty)
    if cells is not None:
        import map_hex
        from frontier_forest import fill_weight
        ids = map_hex.region_ids(cells)
        npl = np.array(['non_playable' in n for n in names] + [True])[np.where((ids >= 0) & (ids < len(names)), ids, len(names))]
        land = (cells[..., 0] & 3) != 1
        out = fill_weight(land & ~npl)[np.clip(hr, 0, cells.shape[0] - 1), np.clip(hq, 0, cells.shape[1] - 1)].astype(np.float32)
        fl = fill_floor(real_bio, bnames) if real_bio is not None else None
        if fl is not None:   # [FILL FLOOR] on land tiles (the exclusions below still apply)
            out = np.where(land[np.clip(hr, 0, cells.shape[0] - 1), np.clip(hq, 0, cells.shape[1] - 1)], np.maximum(out, fl), out).astype(np.float32)
        if domains is None:
            out[west_band(hq, hr, cw)] = 0
    if domains is None:
        out[cw.NATIVE_TILE_Y0:cw.NATIVE_TILE_Y1, cw.NATIVE_TILE_X0:cw.NATIVE_TILE_X1] = 0
    else:
        if np.asarray(domains['native_keep']).shape != tuple(shape):
            raise ValueError('forest_fill: visual domain shape mismatch')
        out[domains['native_keep'] | (domains['vegetation_takeover'] <= 0)] = 0
    return out


def forest_density(real_bio, bnames, q, r, origin, cells=None, names=None, fill=None, domains=None):
    """float32 0-1 per tile (tile grid of real_bio, rows south-up): the forest density the ground and the props
    bake share (CONSTS ground_materials FOREST_DENSITY): class base x patch noise x forest_fill (the frontier fill
    fade; legacy excludes west band/native rectangle, shared domains protect native_keep; pass fill when
    already computed). real_bio: polish's real_bio_eco (the
    ecotone-edited RESOLVE classes, persisted as landform-tiles eco_bio), q, r, origin: tile-grid hex coordinates
    and noise origin."""
    import numpy as np
    import oasis_ground as og
    from terrain_noise import fbm
    c = CONSTS['ground_materials']['FOREST_DENSITY']
    lut = np.array([c['base'].get(n, 0.) for n in bnames] + [0.], np.float32)
    b = lut[np.clip(np.asarray(real_bio, np.int64), 0, len(bnames))]
    b = og._box(b, c['blur']).astype(np.float32)
    patch = np.clip(.5 + .7 * fbm(q, r, origin, c['grain'], 3, seed=c['seed']), 0, 1)
    if fill is None:
        fill = forest_fill(b.shape, cells, names, domains=domains, real_bio=real_bio, bnames=bnames)
    out = np.clip(b * (1 - c['amp'] + c['amp'] * patch) * fill, 0, 1).astype(np.float32)
    if domains is not None:
        if np.asarray(domains['native_keep']).shape != out.shape:
            raise ValueError('forest_density: visual domain shape mismatch')
        out[domains['native_keep'] | (domains['vegetation_takeover'] <= 0)] = 0
    return out


def native_edge(shape, q, r, origin):
    """(dn, m) on the tile grid (TILE_H, TILE_W): dn the euclidean tile distance to the native rectangle (0 inside),
    m the meander field in [0, 1] (CONSTS ground_materials NATIVE_EDGE). An edge of reach a + b x m meanders the same
    way wherever it is drawn: the tile climate ring and the mirror band read this one geometry."""
    import numpy as np
    import campaign_world as cw
    from terrain_noise import fbm
    c = CONSTS['ground_materials']['NATIVE_EDGE']
    TH, TW = shape
    x0, x1, y0, y1 = cw.NATIVE_TILE_X0, cw.NATIVE_TILE_X1, cw.NATIVE_TILE_Y0, cw.NATIVE_TILE_Y1
    y, x = np.mgrid[:TH, :TW]
    dn = np.hypot(np.maximum(np.maximum(x0 - x, x - x1 + 1), 0), np.maximum(np.maximum(y0 - y, y - y1 + 1), 0))
    m = np.clip(.5 + .7 * fbm(q, r, origin, c['grain'], 3, seed=c['seed']), 0, 1)
    return dn, m


def band_to_tile_climate(band):
    """Tile-list climate index (TILE_CLIMATE: subtropical 0, cold 1, temperate 2, arid 3) of a global blend G
    band (BLEND_BAND: cold 0, arid 1, temperate 2, subtropical 3): 0->1, 1->3, 2->2, 3->0. The one table
    for polish (blend-climate-tiles) and the tile bake. Not map.hex's climate order (apply_climate.climate_of)."""
    import numpy as np
    K = CONSTS['apply_climate']
    lut = np.zeros(len(K['BLEND_BAND']), np.uint8)
    for name, b in K['BLEND_BAND'].items():
        lut[b] = K['TILE_CLIMATE'][name]
    return lut[np.asarray(band)]


def wetland_candidate_hex(eco, elev, lon, lat, water):
    """(bool per hex, {source: mask}): where the wetland rule (apply_climate) lays marsh, before the ground,
    town and component tests: flat (3x3 elevation std < FLAT_STD) and a RESOLVE wetland ecoregion (FULL below
    FULL_MAX_ELEV; NEAR_WATER below NEAR_WATER_ECO_MAX_ELEV within WATER_REACH of water) or a named box of
    REGIONS wetlands_extra. eco: ecoregion id per hex; elev, lon, lat: hex_geo; water: water ground or river.
    Read by apply_climate.wetland_hexes and apply_ground_types (no hills drawn on a candidate), so the map
    bake's ground-types-first order keeps the marsh (Mekong delta, 2026-10-26)."""
    import numpy as np
    from hex_components import hex_dilate
    from terrain_noise import fbm
    K = CONSTS['apply_climate']
    H, W = elev.shape
    pad = np.pad(elev, 1, mode='edge')
    win = np.stack([pad[1 + dy:H + 1 + dy, 1 + dx:W + 1 + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
    flat = win.std(0) < K['FLAT_STD']
    near_water = np.asarray(water, bool)
    for _ in range(K['WATER_REACH']):
        near_water = hex_dilate(near_water)
    full = np.isin(eco, list(K['WETLAND_ECO_FULL'])) & (elev < K['FULL_MAX_ELEV'])
    near = np.isin(eco, list(K['WETLAND_ECO_NEAR_WATER'])) & (elev < K['NEAR_WATER_ECO_MAX_ELEV']) & near_water
    cand = full | near
    sources = {nm: eco == k for k, nm in {**K['WETLAND_ECO_FULL'], **K['WETLAND_ECO_NEAR_WATER']}.items()}
    rq, rr_ = np.mgrid[0:H, 0:W][::-1]
    for i, (nm, lon0, lon1, lat0, lat1, emax, nw) in enumerate(region('wetlands_extra')['boxes']):
        # Ellipse inscribed in the box, edge frayed by noise, and only the lower part of
        # the ground in it: a box of flat lowland drew a rectangle (West Siberian plain).
        ex = (lon - (lon0 + lon1) / 2) / ((lon1 - lon0) / 2); ey = (lat - (lat0 + lat1) / 2) / ((lat1 - lat0) / 2)
        rad = np.hypot(ex, ey) + .35 * fbm(rq, rr_, (0, 0), 6, 3, seed=400 + i)
        box = (rad < 1) & (elev < emax)
        if box.any():
            box &= elev <= np.percentile(elev[box], 70) + 1
        if nw:
            box &= near_water
        sources[nm] = box
        cand |= box
    return cand & flat, sources


# ------------------------------------------------------------------ props bake predicates (step 3)
# Species families and where they may not stand (accept and the post-filters read the same mask). 'bare':
# the hex sets that ban the family ('desert': RESOLVE desert; 'cold': map.hex cold climate); 'except_set':
# a biome set (SETS) where the ban does not apply. Later: stone_forest, karst (now mountain_tiles.PROP_HABITAT).
SPECIES_RULES = {
    'rock_shrub_': {'bare': ('desert', 'cold'), 'except_set': None},                    # AUTUMN SHRUBS
    'temperate_tree_katsura_': {'bare': ('desert', 'cold'), 'except_set': 'tree_fill'},  # deciduous forest keeps its broadleaves
}


def window_range(a, k):
    """max - min over the (2k+1)^2 window of every cell, clipped at the edges (frontier_mountain_props.window_extreme)."""
    import numpy as np
    out = []
    for op, fill in ((np.maximum, -np.inf), (np.minimum, np.inf)):
        acc = a
        for ax in (0, 1):
            n = acc.shape[ax]
            p = np.pad(acc, [(k, k) if i == ax else (0, 0) for i in (0, 1)], constant_values=fill)
            cur = acc.copy()
            for d in range(2 * k + 1):
                cur = op(cur, p[d:d + n] if ax == 0 else p[:, d:d + n])
            acc = cur
        out.append(acc)
    return out[0] - out[1]


def flat_arid_hex(biome, bnames, elev):
    """bool per hex: an arid class (props ROCK_ARID) whose real relief (ETOPO range over the 7x7-hex window)
    stays below ROCK_RELIEF (ROCK RELIEF; beautify B's gobi rocks read the same predicate). biome, elev: the
    strategic_ink hex grids."""
    import numpy as np
    c = CONSTS['props']; bnames = list(bnames)
    arid = np.isin(biome, [bnames.index(n) for n in c['ROCK_ARID'] if n in bnames])
    return arid & (window_range(np.asarray(elev, np.float32), c['ROCK_RELIEF_K']) < c['ROCK_RELIEF'])


def alpine_treeline_tile(w_alpine, elev, lat, fir_tile):
    """bool per tile: above the treeline on the alpine meadow / steppe (TREELINE ALPINE): classify weight
    alpine_meadow + alpine_steppe >= TREELINE_CLASS_MIN, real elevation above treeline_m(lat), not RESOLVE fir."""
    import numpy as np
    return (np.asarray(w_alpine) >= CONSTS['props']['TREELINE_CLASS_MIN']) & (elev > treeline_m(lat)) & ~np.asarray(fir_tile, bool)


def species_bans(family, biome, bnames, cold):
    """bool per hex where a species of family (a SPECIES_RULES key, or a species name starting with one) may
    not stand; None when no rule covers it. biome: strategic_ink hex classes; cold: map.hex cold climate."""
    import numpy as np
    key = next((k for k in SPECIES_RULES if family.startswith(k)), None)
    if key is None:
        return None
    rule = SPECIES_RULES[key]; bnames = list(bnames)
    out = np.zeros(np.shape(biome), bool)
    for b in rule['bare']:
        out |= np.asarray(cold, bool) if b == 'cold' else (biome == bnames.index(b))
    if rule['except_set']:
        out &= ~in_set(biome, bnames, rule['except_set'])
    return out


def dryness(real_bio, bnames, real_elev=None):
    """oasis_ground.dryness: 0 (humid) .. 1 (dry) per tile, smoothstep DRY of the RESOLVE desert share
    box-blurred twice over DRY_R tile cells; with real_elev, times 1 + DRY_ASPECT gain x south-facing share."""
    import numpy as np
    import oasis_ground as og
    d = og._box(og._box(np.isin(real_bio, og._ix(bnames, 'desert')), og.DRY_R), og.DRY_R)
    if real_elev is not None:
        gain, per_cell, rad = og.DRY_ASPECT
        south = np.clip(np.gradient(og._box(og._box(real_elev, rad), rad), axis=0) / per_cell, 0, 1)
        d = d * (1 + gain * south)
    return og._smooth((d - og.DRY[0]) / (og.DRY[1] - og.DRY[0])).astype(np.float32)


def arid_share(real_bio, bnames, dry=None, real_elev=None):
    """oasis_ground.arid_share: blurred share of arid ground per tile, desert and ARID_ALPINE times dryness,
    box radius ARID_R twice (polish's tonal buffer x (1 - arid_share)); real_elev adds the aspect term."""
    import numpy as np
    import oasis_ground as og
    if dry is None:
        dry = dryness(real_bio, bnames, real_elev)
    a = np.isin(real_bio, og._ix(bnames, 'desert')).astype(np.float32)
    a = np.maximum(a, np.isin(real_bio, og._ix(bnames, *og.ARID_ALPINE)) * dry)
    return np.clip(og._box(og._box(a, og.ARID_R), og.ARID_R), 0, 1).astype(np.float32)


def classify_gameplay(geo, table, names):
    """apply_climate.classify: gameplay climate index (into names) per hex from real geography (RULES
    gameplay, the gc_* groups, COLD_LINES), the learned vanilla table where it has a key; -1 where no
    geography. Returns (climate, learned-table hit)."""
    import numpy as np
    import apply_climate as ac
    g_ = RULES['gameplay']
    C = {n: names.index(n) for n in ('arid', 'cold', 'subtropical', 'temperate')}
    bn = geo['names']
    b, e, la = geo['biome'], geo['elev'], geo['lat_n']
    isb = lambda *ns: np.isin(b, [bn.index(n) for n in ns if n in bn])
    out = np.full(b.shape, -1, np.int32)
    tropical = isb(*ac.GC_TROPICAL)
    rule = np.full(b.shape, C['temperate'], np.int32)
    rule[(la < g_['sub_lat']) & isb(*ac.GC_SUB31)] = C['subtropical']
    rule[(la < g_['forest_sub_lat']) & isb('forest')] = C['subtropical']
    rule[tropical] = C['subtropical']
    rule[tropical & (la >= g_['tropical_cold'][0]) & (e >= g_['tropical_cold'][1])] = C['cold']
    rule[isb(*ac.GC_ARID)] = C['arid']
    rule[isb('steppe') & (e >= g_['steppe_arid_elev'])] = C['arid']
    rule[isb('conifer') & ((la < g_['conifer_lat'][0]) | (la >= g_['conifer_lat'][1]))] = C['cold']
    rule[isb('conifer') & (la >= g_['conifer_lat'][0]) & (la < g_['conifer_lat'][1]) & (e >= g_['conifer_arid_elev'])] = C['arid']
    rule[isb('forest') & (la >= g_['forest_cold'][0]) & (la < g_['forest_cold'][1]) & (e >= g_['forest_cold'][2])] = C['cold']
    rule[(la >= ac.NORTH_COLD) & ~isb('desert')] = C['cold']           # Siberia, Amur, northern Mongolia
    rule[isb('forest') & (la >= g_['forest_cold_lat'])] = C['cold']
    # Learned table where vanilla has enough hexes.
    key = (b.astype(np.int64) * 16 + ac._bin(la, ac.LAT_BINS)) * 16 + ac._bin(e, ac.ELEV_BINS)
    lut_keys = np.array(sorted(table), np.int64)
    hit = np.isin(key, lut_keys)
    learned = np.full(b.shape, -1, np.int32)
    if len(lut_keys):
        learned[hit] = np.array([table[int(k)] for k in lut_keys])[np.searchsorted(lut_keys, key[hit])]
    out = np.where(hit, learned, rule)
    # Always cold: snow, tundra, taiga and high ground (vanilla >= 3000 m: 86% cold).
    out[isb(*ac.GC_COLD) | (e >= ac.COLD_ELEV)] = C['cold']
    out[~geo['ok']] = -1
    return out, hit


def consts(module):
    """The constants of one consumer module (deep copy, stored order)."""
    return copy.deepcopy(CONSTS[module])


def biome_set(flag):
    """Tuple of the RESOLVE classes carrying flag, in rank order (the consumer's own order)."""
    assert flag in SETS, flag
    return tuple(sorted((b for b in BIOMES if flag in BIOMES[b]['sets']), key=lambda b: BIOMES[b]['sets'][flag]))


def biome_map(flag, field):
    """{class: record[field]} for the classes carrying flag, in rank order."""
    return {b: copy.deepcopy(BIOMES[b][field]) for b in biome_set(flag)}


def cold_line(name):
    return copy.deepcopy(COLD_LINES[name])


def mixes():
    """ground_materials.MIX: {class: {slot: share}} in MIX_ORDER."""
    return {c: copy.deepcopy(LANDFORMS[c]['mix']) for c in MIX_ORDER}


def gains():
    """ground_materials.CLASS_GAIN: {class: gain} in landform (evaluation) order."""
    return {c: LANDFORMS[c]['gain'] for c in LANDFORMS}


def redraws():
    """ground_materials.REDRAW: {class: donor slots its mix redraws with the class weight} (LANDFORMS 'redraw'; one slot
    set for all, in landform order)."""
    out = {c: tuple(LANDFORMS[c]['redraw']) for c in LANDFORMS if LANDFORMS[c].get('redraw')}
    assert len(set(out.values())) <= 1, out
    return out


def grains():
    """ground_materials.GRAINS: {class: material-blob grain (hexes)} of the landforms with their own 'grain' (the rest
    take CONSTS ground_materials GRAIN[1]), in landform order."""
    return {c: LANDFORMS[c]['grain'] for c in LANDFORMS if 'grain' in LANDFORMS[c]}


def _jsonable(v):
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (set, frozenset)):
        return sorted(_jsonable(x) for x in v)
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


def _fmt(v, ind=0):
    """JSON text: containers of scalars (or of flat scalar lists) on one line, the rest indented one space a level."""
    flat = lambda x: not isinstance(x, (dict, list)) or (isinstance(x, list) and all(not isinstance(y, (dict, list)) for y in x))
    if not isinstance(v, (dict, list)) or (isinstance(v, list) and all(flat(x) for x in v)) or (isinstance(v, dict) and all(not isinstance(x, (dict, list)) for x in v.values())):
        return json.dumps(v, ensure_ascii=False)
    pad = ' ' * (ind + 1)
    if isinstance(v, list):
        return '[\n' + ',\n'.join(pad + _fmt(x, ind + 1) for x in v) + '\n' + ' ' * ind + ']'
    return '{\n' + ',\n'.join(pad + json.dumps(k, ensure_ascii=False) + ': ' + _fmt(x, ind + 1) for k, x in v.items()) + '\n' + ' ' * ind + '}'


def table():
    """The whole table as one ordered dict (data only)."""
    return {'schema': SCHEMA, 'biome_codes': BIOME_CODES, 'landforms': LANDFORMS, 'mix_order': MIX_ORDER, 'biomes': BIOMES,
            'sets': {f: {'members': biome_set(f), 'note': n} for f, n in SETS.items()}, 'other_default': OTHER_DEFAULT,
            'cold_lines': COLD_LINES, 'regions': REGIONS, 'oasis_seeds': oasis_seeds(), 'rules': RULES, 'consts': CONSTS,
            'audit_only': {'families': FAMILIES, 'landforms': AUDIT, 'pending': AUDIT_PENDING}}


def snapshot():
    """{'key': repr} of every value a pipeline consumer reads from the table (values, types and order;
    repr keeps tuple/list, int/float and dict order apart). Audit-only fields (AUDIT, FAMILIES) and
    entries no consumer reads (REGIONS with no consumers: karez; LANDFORMS river_oasis) are left out:
    editing them changes no output. check_biome_profiles.py compares this with the frozen copy."""
    out = {}
    for mod in CONSTS:
        for k, v in consts(mod).items():
            out[f'consts.{mod}.{k}'] = repr(v)
    out['landforms.mixes'] = repr(mixes()); out['landforms.gains'] = repr(gains())
    if grains():   # only when a landform has its own grain (the v23 recipe has none: its snapshot keeps the old keys)
        out['landforms.grains'] = repr(grains())
    if redraws():   # [DARK REDRAW] (the v23 recipe has none)
        out['landforms.redraw'] = repr(redraws())
    out['landforms.relief'] = repr(relief_specs())   # own relief grading (ground_materials RELIEF_SPECS)
    for k in COLD_LINES:
        out[f'cold_lines.{k}'] = repr(cold_line(k))
    for f in SETS:
        out[f'sets.{f}'] = repr(biome_set(f))
    for f, field in (('polish_climate', 'climate'), ('gameplay_polish', 'gameplay'), ('tree_fill', 'tree_fill')):
        out[f'maps.{f}.{field}'] = repr(biome_map(f, field))
    out['other_default'] = repr(OTHER_DEFAULT)
    for k in RULES:
        out[f'rules.{k}'] = repr(RULES[k])
    for k, v in REGIONS.items():
        if v['consumers']:
            out[f'regions.{k}'] = repr({a: b for a, b in v.items() if a != 'consumers'})
    out['oasis_seeds'] = repr(oasis_seeds())
    for k, v in SPECIES_RULES.items():
        out[f'species_rules.{k}'] = repr(v)
    return out


def dump_json(path=None):
    """analysis/biome-profiles.json (review copy; int keys as strings, tuples as lists). Returns the text."""
    from pathlib import Path
    text = _fmt(_jsonable(table())) + '\n'
    assert json.loads(text) == _jsonable(table())
    path = Path(path) if path else Path(__file__).resolve().parent.parent / 'analysis/biome-profiles.json'
    path.write_text(text, encoding='utf-8', newline='\n')
    return text


if __name__ == '__main__':
    import sys
    print(len(dump_json(sys.argv[1] if len(sys.argv) > 1 else None)), 'characters written')
