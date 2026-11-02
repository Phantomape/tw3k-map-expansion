# Third-party notices

This repository contains only the source code, data and documentation the TW3K Map Expansion project authored
(see LICENSE). It does **not** contain or redistribute any of the software, game content or datasets listed
below. The build reads them from copies you obtain yourself (see BUILD.md). They remain under their own terms,
which are summarised or reproduced here.

## 1. Total War: THREE KINGDOMS and its Assembly Kit

Total War: THREE KINGDOMS, its game data (`data/*.pack`, shaders, terrain, databases, the vanilla campaign map)
and the official Assembly Kit (including its DLLs and `raw_data`) are the property of The Creative Assembly /
SEGA. They are used under the game's End User Licence Agreement and the Assembly Kit terms. **Nothing from the
game or the Assembly Kit is included in this repository.** Build outputs that embed game content (`.pack`
files, extracted terrain, the vanilla logical map) may be shared only as those terms allow.
You need your own legal copy of the game with the Assembly Kit installed to build. This project is not
affiliated with or endorsed by The Creative Assembly or SEGA.

The source references Total War: Three Kingdoms identifiers (region, province and table keys) solely for
interoperability; no game assets or map data are included.

源码中出现的《全面战争：三国》标识符（地区、省份和数据表的键名）仅用于互操作；本仓库不含任何游戏素材或地图数据。

## 2. Campaign Map Toolkit (CAIME) 1.1.0

The build drives CAIME (Campaign Map Toolkit, by the TW Campaign Map Modding Team) for map-data, pathfinding,
border, trade-route and lookup exports; the helper programs in `scripts/Tw3k*.cs` are compiled against
`CAIME.exe` and its ESF library. CAIME is **not** included; obtain it from its authors
(https://github.com/TW-Campaign-Map-Modding-Team/CampaignMapToolkit). This project adopts CAIME's licence terms
for its own code (see LICENSE). Use of the CAIME application is also governed by its End User License
Agreement (EULA.txt in the CAIME distribution). CAIME's notice and licence, reproduced as its clause 3 requires:

```
CampaignMapToolkit Non-Commercial Open Source License
Version 1.0

Copyright (c) 2026 Campaign Map Toolkit Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to use,
copy, modify, merge, and distribute the Software, subject to the following
conditions:

1. NON-COMMERCIAL USE ONLY
   The Software, and any derivative works, may not be used for commercial
   purposes. "Commercial purposes" means any use intended to generate revenue
   or financial gain, including but not limited to: selling the Software or
   derivative works, incorporating the Software into a commercial product or
   service, or using the Software to provide paid services to third parties.

2. SHARE-ALIKE (COPYLEFT)
   Any derivative works or projects that incorporate the Software, in whole
   or in part, must be distributed under the same terms as this License.
   The complete source code of such derivative works must be made publicly
   available under this License or a compatible license that preserves all
   three conditions stated herein.

3. ATTRIBUTION
   All copies or substantial portions of the Software must retain this
   copyright notice and license text.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 3. RPFM (Rusted PackFile Manager) 4.2.3

The build calls `rpfm_cli` (https://github.com/Frodo45127/rpfm) to read and write `.pack` files, with RPFM's
Three Kingdoms schema (`schema_3k.ron`). RPFM is **not** included. Its licence:

```
MIT License

Copyright (c) 2017-2023 Ismael Gutiérrez González

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

The schema files come from RPFM's schema repository (https://github.com/Frodo45127/rpfm-schemas) and keep that
repository's terms.

## 4. Geographic datasets

None of these files are included. `scripts/fetch_geodata.py` downloads them from their publishers and checks
the SHA-256 recorded in `data/geodata-sources.json`. Terrain, rivers, lakes, ground and climate in the
expanded areas are derived from them; the derived caches are rebuilt locally and not distributed.

### ETOPO 2022 Global Relief Model (60 arc-second, surface elevation)

Source: NOAA National Centers for Environmental Information (NCEI). A work of the U.S. Government, not subject to
copyright in the United States. Citation:

> NOAA National Centers for Environmental Information. 2022: ETOPO 2022 15 Arc-Second Global Relief Model.
> NOAA National Centers for Environmental Information. https://doi.org/10.25921/fd45-gt74

The 60 arc-second surface grid used here is part of the same ETOPO 2022 release. NOAA does not endorse this
project.

### RESOLVE Ecoregions 2017

Licensed under Creative Commons Attribution 4.0 International (CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/). Attribution:

> Dinerstein, E., Olson, D., Joshi, A., Vynne, C., Burgess, N. D., Wikramanayake, E., et al. (2017).
> An Ecoregion-Based Approach to Protecting Half the Terrestrial Realm. *BioScience* 67(6): 534-545.
> https://doi.org/10.1093/biosci/bix014. Data: RESOLVE Ecoregions 2017, https://ecoregions.appspot.com/

Changes made: the ecoregion polygons are rasterised onto the campaign-map grid (`scripts/frontier_terrain.py`,
cache `work/geodata/eco-grid-v2.npz`) and used to choose ground, vegetation and climate classes.

### Natural Earth

`ne_10m_rivers_lake_centerlines`, `ne_10m_lakes`, `ne_10m_playas` (version 5.0.0), and Natural Earth I 50m
shaded relief (used only by a comparison preview). Natural Earth data is in the public domain
(https://www.naturalearthdata.com/about/terms-of-use/). Made with Natural Earth.

## 5. Python and .NET dependencies

The build uses Python 3.14 with NumPy, SciPy, Pillow, h5py and CuPy (pinned in `env/tw3k-requirements.txt` and
`env/tw3k-conda-explicit-win64.txt`), and the .NET Framework C# compiler on Windows. They are installed from their
own distributions, are not included here, and keep their own licences.

## 6. Earlier development versions

Early development versions used part of the region layout of the Workshop mod 190 Expanded; before this release
those regions were fully rebuilt from independent sources, and this repository contains none of its data.

早期开发版本曾使用 190 Expanded 的部分地区布局；发布前这些地区已按独立资料全部重建，本仓库不含其数据。
