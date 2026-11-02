# Building TW3K Map Expansion

> **Draft (2026-11-01).** Order checked against the scripts (packaging review); two known gaps are listed in
> section 4. Not yet verified from a clean environment.

This repository holds the source code, the design data and the layout data of the map expansion. It does **not**
hold the game, the official Assembly Kit, the third-party tools or the geographic source data: you bring your own
copies and the build reads them. There is no one-click build; a full build takes a Windows machine with the game
installed and around an hour.

The region layout is generated from the vanilla map, independent geographic data and researched historical sites
(see `THIRD_PARTY_NOTICES.md` for the data sources and their terms).

```powershell
git clone https://github.com/Phantomape/tw3k-map-expansion.git
cd tw3k-map-expansion
```

## 1. What you need

| Item | Version | Notes |
|---|---|---|
| Windows 10/11, x64 | | the SDK tools and CAIME are Windows programs |
| Total War: THREE KINGDOMS | current Steam release | your own legal copy, with the **Assembly Kit** installed (Steam: Tools) |
| CAIME (Campaign Map Toolkit) | 1.1.0 | https://github.com/TW-Campaign-Map-Modding-Team/CampaignMapToolkit; unpack to `tools/caime-1.1.0/` |
| RPFM | 4.2.3 | https://github.com/Frodo45127/rpfm; run it once to download its schemas (`schema_3k.ron`) |
| .NET Framework 4.x C# compiler (`csc.exe`) | | ships with Windows; compiles `scripts/Tw3k*.cs` against CAIME |
| Python | 3.14 (conda-forge) | exact environment: `env/create_tw3k_env.ps1` (NumPy, SciPy, Pillow, h5py, CuPy) |
| NVIDIA GPU with CUDA | optional | CuPy speeds up polish and previews; the CPU path gives the same result |
| Disk | about 40 GB free | workspace, SDK copy, caches and the built pack |
| Memory | 32 GB recommended | the polish step needs about 22 GB of free commit |

## 2. Configure the paths

Every machine-specific path is read from one place (`scripts/tw3k_paths.py`; PowerShell: `Get-Tw3kPath` in
`scripts/tw3k_env.ps1`). Copy `tw3k.local.example.json` to `tw3k.local.json` in the repository root and edit it,
or set the environment variables, which take precedence:

| Key in tw3k.local.json | Environment variable | Meaning | Default |
|---|---|---|---|
| `game_dir` | `TW3K_GAME_DIR` | game folder (holds `data/` and `assembly_kit/`) | none: required |
| `rpfm_cli` | `TW3K_RPFM` | `rpfm_cli.exe` | none: required |
| `rpfm_schema` | `TW3K_RPFM_SCHEMA` | RPFM's `schema_3k.ron` | `%APPDATA%\FrodoWazEre\rpfm\config\schemas\schema_3k.ron` |
| `ca_appdata` | `TW3K_CA_APPDATA` | the game's settings folder | `%APPDATA%\The Creative Assembly\ThreeKingdoms` |
| `main_root` | `TW3K_MAIN_ROOT` | main checkout (locks, caches) | this checkout (for a git worktree: its main checkout) |
| `scratch` | `TW3K_SCRATCH` | scratch folder for copies and caches | `<main_root>-scratch` next to it |
| `fonts` | `TW3K_FONTS` | fonts for preview labels | `%WINDIR%\Fonts` |

Example `tw3k.local.json`:

```json
{
 "game_dir": "<Steam library>/steamapps/common/Total War THREE KINGDOMS",
 "rpfm_cli": "<RPFM folder>/rpfm_cli.exe"
}
```

The Python interpreter is named in `env/python-path.txt` (one line, the full path of the environment's
`python.exe`); `TW3K_PYTHON` overrides it. Set `OPENBLAS_NUM_THREADS=1` before the geographic steps.

## 3. Python environment

```powershell
.\env\create_tw3k_env.ps1 -Prefix <conda root>\envs\tw3k -Conda <conda root>\Scripts\conda.exe
Set-Content env\python-path.txt '<conda root>\envs\tw3k\python.exe'
```

## 4. Build order

All steps run from the repository root with the tw3k Python. The release is built with every switch (`TW3K_*`
environment variables named in the scripts) at its default, so none needs to be set (section 5). The order matters: the
patches and the SDK workspace read the logical map, so it comes before them.

> **Known gaps of this release.** Two inputs must be prepared by hand for a first build; scripts that make them
> will follow in a later release. The build has not yet been run end to end from a clean machine.
>
> 1. **Source SDK kit** `work/standalone_kit/`. Copy these four folders from your
>    `<game>/assembly_kit/` into `work/standalone_kit/`, keeping their relative paths: `raw_data/db`,
>    `binaries`, `raw_data/EmpireDesignData/campaigns/3k_main_campaign_map` and
>    `raw_data/EmpireDesignData/campaign_maps/3k_dlc07_main_map`. Step 6 (`all_directions_build.py setup`)
>    copies them into the build workspace, replaces the map with the expanded one and restores the region and
>    start-position tables from the official kit.
> 2. **First vanilla terrain in the stage** `work/all-directions-30/work/release-input/`. The terrain build
>    polishes the vanilla campaign terrain that it finds there, moved onto the canvas; the geometry export of
>    step 7 does not write terrain. Put there, from your game's `terrain.pack` and `fast.pack`, the vanilla
>    `3k_dlc07_main_map` terrain placed at the native offset (888, 375 hexes) of the 2048x2047 canvas, with the
>    border repeating the outermost vanilla pixels: under `terrain/campaigns/3k_dlc07_main_map/` the
>    `lf_height_map` and `lf_sea_height_map` (`.dds` and `.compressed_map`), `global_map/global_blend.dds`,
>    `climate_map.cm`, both `tile_list.bin` (header set to the canvas size, tiles shifted), the
>    `global_meshes/` models moved by the native shift, the props and water, and `environment_collection.xml`
>    (lighting spheres moved); under `campaign_maps/3k_dlc07_main_map/` the `camera_heightmap.png` (padded) and
>    `display/trees/trees.campaign_tree_list` (moved).
>
> The logical map is not a gap: step 4 builds `final-map/map.hex` from your vanilla map and the layout diff, and
> checks that this reproduces the released map byte for byte.

1. **Geographic source data** (about 690 MB; checked against recorded SHA-256):

   ```powershell
   python scripts\fetch_geodata.py            # ETOPO 2022, RESOLVE Ecoregions 2017, Natural Earth (rivers, lakes, playas)
   python scripts\fetch_geodata.py --check    # verify only
   ```

   A file whose hash differs from `data/geodata-sources.json` is not installed (the publisher may have replaced it).

2. **Geographic caches that need only the geodata and the vanilla map** (the geodata hashes were checked in
   step 1):

   ```powershell
   python scripts\strategic_ink.py --build
   python scripts\build_west_real_terrain.py
   ```

3. **Vanilla reference files**: the battle-location map, extracted from your game's `fast.pack` (only from that
   vanilla pack, never from a mod pack in `data/`) and checked against the hash recorded for the release's game
   version:

   ```powershell
   python scripts\extract_vanilla_reference.py            # -> reference\vanilla-terrain\
   ```

4. **The logical map** `work/all-directions-30/work/final-map/map.hex`, built from your vanilla map and the
   layout diff in `data/map-layout-diff/`. The vanilla logical map is
   `tools/caime-1.1.0/lib/app/Templates/3k_dlc07_main_map/map.hex` (CAIME's export of the shipped campaign map).
   The diff stores only the bytes that differ from vanilla:

   | File | Content |
   |---|---|
   | `layout-diff.npz` | `q`, `r` of every hex that differs from vanilla placed at the native offset on the empty 2048x2047 canvas; `mask` (bit i: byte i differs); `bytes` (the differing bytes, 0 where the byte is vanilla's) |
   | `manifest.json` | format, SHA-256 of the vanilla map and of the resulting map, grid, native offset, the map header (region, ground, climate lists) |
   | `river-channels.json` | the river channel list written beside the map |

   ```powershell
   $env:TW3K_WORKSPACE = "$PWD\work\all-directions-30"
   python scripts\bake_map.py --diff --out work\all-directions-30\work\final-map\map.baked.hex
   Copy-Item work\all-directions-30\work\final-map\map.baked.hex work\all-directions-30\work\final-map\map.hex
   Copy-Item work\all-directions-30\work\final-map\map.baked.hex.river-channels.json work\all-directions-30\work\final-map\river-channels.json
   ```

   The bake stops when your vanilla map is not the recorded one (for example after a game update: the diff would
   land on a different base) and when the result is not byte for byte the released map. It writes beside its
   output, never into `map.hex` directly; the two copies put the checked map and its river channels where the
   later steps read them. If `map.hex` already holds the same bytes, skip the copy; if a copy fails because the
   file is briefly in use (an antivirus scan or an indexer mapping it), retry after a few seconds.
   `scripts/make_release_snapshot.py` checks that the diff holds no vanilla byte.

5. **Patches that read the map** (after step 4):

   ```powershell
   python scripts\hehuang_patch.py build
   python scripts\korea_patch.py build
   ```

6. **SDK workspace and database** (after step 4; known gap 1 for the source kit). The SDK DLLs are loaded in
   process; nothing in the game folder is written:

   ```powershell
   python scripts\all_directions_build.py setup              # once per workspace: copies the source kit, puts map.hex in it
   python scripts\all_directions_build.py official-headers   # official empty tables into work\all-directions-30\work\native-db (must run before database)
   python scripts\all_directions_build.py database           # region, settlement and start-position tables; build tables
   python scripts\build_combined_mod.py exports
   ```

   With `TW3K_WORKSPACE` set as in step 4. `official-headers` runs `native_db_export --all` (no `--split`) on the
   workspace kit's `raw_data/db` (the copy `setup` made) and writes the official tables into the workspace's
   `work/native-db`. A new workspace needs it: without it 21 empty tables are missing there and `database` (its
   build-table export) stops. The tables that hold region data are written again by `database`.

7. **Terrain** (about 30 minutes): on the first build, with `TW3K_SEED_BASE_FROM_STAGE=1`:

   ```powershell
   .\scripts\build_polished_campaign_terrain.ps1 -RebuildGeometry
   ```

   `-RebuildGeometry` runs the CAIME helpers (`Tw3kBuild` map data, pathfinding, borders, trade routes, lookup;
   `Tw3kWorldExtent`, `Tw3kLookup`, `Tw3kTheatreShader`), which read the game packs only. After the export,
   `terrain_checkpoint.py capture geometry` copies the stage's terrain into the vanilla snapshot
   `work/all-directions-30/terrain-before-polish/` and stamps it with the canvas; later builds refuse a snapshot
   from another canvas, map or design. The terrain it copies must already be in the stage (known gap 2). Later
   builds run without `-RebuildGeometry` as long as the map and design are unchanged. Stage invariants run after
   every step and stop the build on a failure.

   > **No layout cleanup data in the public source.** The release build regenerates the relief of ground that
   > earlier, private layouts of this project changed (polish's layout cleanup, `scripts/layout_cleanup.py`). Its
   > input is derived from those old layouts and is left out of this repository for source compliance. Without it
   > (and with the `PUBLIC_SOURCE` file at the repository root, which marks this tree as the public release; do not
   > delete it) polish prints a `WARN: ... layout cleanup skipped` line and the cleanup weight is 0 everywhere; the build
   > completes, but outside the vanilla map area the terrain differs slightly from the released pack. After this
   > release the cleanup will be replaced by a rule that reads public inputs only.

8. **Compile and package** (about 15 minutes; **the game must be closed**):

   ```powershell
   .\scripts\rebuild_section14_candidate.ps1
   ```

   It writes a compiler pack, `user.script.txt`, the mod list and preferences into the game folder and
   `%APPDATA%\The Creative Assembly\ThreeKingdoms`, starts `Three_Kingdoms.exe` hidden to compile the start
   positions of both 190 modes, and restores all of it when it ends (start positions already up to date are not
   compiled again). The pack is `output\tw3k_regions_and_battles_terrain_repair_candidate.pack`; packaging needs
   `rpfm_cli` and RPFM's schema.

9. **Offline review** (pauses while the game runs):

   ```powershell
   Copy-Item work\all-directions-30\work\final-map\map.hex output\review-map.hex   # renders read a copy, never the live map
   Copy-Item work\all-directions-30\work\current-region-design.json output\review-design.json   # the design of this pack
   python scripts\art_review.py run --pack output\tw3k_regions_and_battles_terrain_repair_candidate.pack --map output\review-map.hex `
       --design output\review-design.json --base <baseline pack> --base-map <its map.hex copy> --base-design <its design copy>
   ```

   It renders the review cameras and compares them with the baseline pack (`--base`; by default the last pack
   whose review passed). Each side draws the towns of its own region design, so keep a copy of the map and the
   design next to every pack you may compare with later; `--base-design` is required. A first build has no
   baseline yet (a pack is never compared with itself): keep its pack, map and design copies, and review the next
   build against them.

## 5. Switches

The release is built with every switch at its default; none needs to be set. The scripts read three kinds of
switches:

- **Layers that are on by default** (`=0` turns one off; for diagnosing a stage, never for the release):
  `TW3K_CLIMATE_WETLANDS`, `TW3K_FARM_MUD`, `TW3K_EDGE_CLEANUP`, `TW3K_GROUND_MATERIALS`,
  `TW3K_MATERIALS_ECOTONE`, `TW3K_ENFORCE_BANNED`, `TW3K_TILE_CLIMATE_BAND`, `TW3K_KOREA`, `TW3K_HEHUANG` (the
  two terrain patches of step 5) and `TW3K_GPU` (`=0` runs the GPU kernels on the CPU).
- **Experiments that are off by default** (for example `TW3K_SNOW_BAKE`, `TW3K_ARCTIC_TREELINE`,
  `TW3K_LAKE_LEVEL`, `TW3K_TUNDRA_BAN`): candidate changes under review, not part of the release.
- **The one fallback the release documents**:

| Variable | Values (default first) | What it changes | Rerun from |
|---|---|---|---|
| `TW3K_GROUND_RECIPE` | `v24` (the release), `v23` | `v23` restores the previous ground recipe: the material mix and relief of eight landforms (high, plateau, gobi, drymtn, sgravel, alpine, alpine_meadow, alpine_steppe), material patch grain 3 hexes instead of 6, and the arid town farm-oasis weight 6 instead of 2. Only the material channel (R) of `global_blend` changes (ground materials and arid farm fields); climate bands, heights, trees and the map stay as they are. Any other value stops at import. | the terrain build (step 7); the map and database steps need not be rerun. polish's setup memo is keyed on the recipe, so it is not reused |

With `v23` the frozen biome-profile checks compare against the v23 snapshot (`analysis/biome-profiles-frozen-v23.json`).
The `ground_fragmentation` invariant checks each landform's fragmentation against its baseline at polish and
sync_final in either case.

## 6. Notes

- Only the 190 main campaign (romance and records modes) is supported; start a new campaign.
- Only step 8 writes into the game folder and the game's settings, and it restores them when it ends.
- `scripts/make_release_snapshot.py` makes the source snapshot this repository was created from.
