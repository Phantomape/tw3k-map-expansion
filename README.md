# TW3K Map Expansion

An expanded campaign map for **Total War: THREE KINGDOMS** (the 190 campaign): the vanilla map is kept as it
is, and a much larger canvas around it (2048×2047 hexes, vanilla placed at offset 888, 375) adds new regions,
coasts, rivers, terrain and vegetation from the Western Regions to the Korean peninsula and the northern steppe.
The region layout is generated from the vanilla map, independent geographic data and researched historical
sites.

《全面战争：三国》190 剧本的扩展战役地图。原版地图保持不变，周围扩展出 2048×2047 格的大画布（原版位于偏移
888, 375），新增从西域到朝鲜半岛、北方草原的地区、海岸、河流、地形和植被。地区布局按原版地图、独立地理数据和
考证的历史地点生成。

## Status / 状态

- Supports only the **190 main campaign** (romance and records modes), in a **new** campaign. Other start dates,
  DLC campaigns and old saves are not supported.
  只支持 **190 主战役**（演义和史实），需要开新档；不支持其他剧本、DLC 战役和旧存档。
- This repository is the **source** of the map build: scripts, design and layout data, tests and documentation.
  It contains no game files and no built `.pack`.
  本仓库是地图构建的**源码**：脚本、设计和布局数据、测试和文档；不含游戏文件，也不含打好的 `.pack`。

## Building / 构建

See [BUILD.md](BUILD.md). You need your own copy of the game with the official Assembly Kit, the CAIME and RPFM
tools, and a Python environment; the geographic source data is downloaded and checked by
`scripts/fetch_geodata.py`. A full build runs on Windows and takes about an hour.

构建步骤见 [BUILD.md](BUILD.md)：需要自备正版游戏和官方 Assembly Kit、CAIME 和 RPFM 工具，以及 Python 环境；地理
源数据由 `scripts/fetch_geodata.py` 下载并校验。完整构建在 Windows 上进行，约一小时。

```
git clone https://github.com/Phantomape/tw3k-map-expansion.git
```

## Repository layout / 目录

| Path | Content |
|---|---|
| `scripts/` | build, bake, polish, packaging and audit tools (Python, PowerShell, C# sources of the CAIME helpers) |
| `tests/` | unit tests (`python -m unittest discover -s tests`) |
| `data/map-layout-diff/` | layout data: what the expanded map changes relative to vanilla (step 4 of `BUILD.md` bakes the map from it) |
| `data/oss-rebuild/` | provenance of the rebuilt regions (rebuild parameters, open-filler list read by a build check) |
| `data/geodata-sources.json` | geographic source datasets with download URLs and SHA-256 |
| `work/all-directions-30/work/` | the region design and authored coordinates |
| `env/` | the exact Python environment |

## Licence / 许可

The project's own code and data are released under the terms of the **CampaignMapToolkit Non-Commercial Open
Source License v1.0** (see [LICENSE](LICENSE)): source-available for non-commercial use, share-alike, with
attribution. This is not an OSI-approved open-source licence. Third-party tools, data and the game keep their own
terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

本项目自有的代码和数据采用 **CampaignMapToolkit Non-Commercial Open Source License v1.0** 的条款（见
[LICENSE](LICENSE)）：源码公开，仅限非商业使用，相同方式共享，需署名；这不是 OSI 认可的开源许可。第三方工具、数据
和游戏本身按各自条款，见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

Total War: THREE KINGDOMS is a trademark of The Creative Assembly / SEGA. This project is not affiliated with or
endorsed by them.
