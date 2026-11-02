$ErrorActionPreference='Stop'
$combinedRoot=Split-Path -Parent $PSScriptRoot;$combinedA=if($env:TW3K_WORKSPACE){$env:TW3K_WORKSPACE}else{Join-Path $combinedRoot 'work\all-directions-30'};$combinedKit=Join-Path $combinedA 'work\standalone_kit';$combinedApp=Join-Path $combinedRoot 'tools\caime-1.1.0\lib\app';$combinedMap=Join-Path $combinedKit 'raw_data\EmpireDesignData\campaign_maps\3k_dlc07_main_map\map.hex';$combinedStage=Join-Path $combinedA 'work\release-input'
. (Join-Path $combinedRoot 'scripts\tw3k_env.ps1');$combinedBasePython=Get-Tw3kBasePython
& $combinedBasePython -B (Join-Path $combinedRoot 'scripts\sdk_memory_export.py')
if($LASTEXITCODE -ne 0){throw 'SDK region/area preflight or map-data export failed'}
& $combinedBasePython -B (Join-Path $combinedRoot 'scripts\normalize_campaign_world_frame.py')
if($LASTEXITCODE -ne 0){throw 'Native map-data and retained terrain coordinate frames differ'}
foreach($combinedAction in @('pathfinding','borders','trade-routes','lookup')){
 & (Join-Path $combinedApp 'Tw3kBuild.exe') $combinedKit $combinedMap $combinedAction | Out-File -LiteralPath (Join-Path $combinedA ('analysis\fixed-'+$combinedAction+'.log')) -Encoding utf8
 if($LASTEXITCODE -ne 0){throw "Native export failed: $combinedAction"}
}
$combinedDisplay=Join-Path $combinedStage 'campaign_maps\3k_dlc07_main_map\display\regions'
& (Join-Path $combinedApp 'Tw3kLookup.exe') (Join-Path $combinedKit 'working_data\campaign_maps\3k_dlc07_main_map\3k_main_campaign_map_lookup.bmp') $combinedDisplay
if($LASTEXITCODE -ne 0){throw 'Lookup encoding failed'}
New-Item -ItemType Directory -Path (Join-Path $combinedStage 'fxc') -Force | Out-Null
# The UI overlay shader samples the region lookup by its pixel size; read it from the SDK export.
Add-Type -AssemblyName System.Drawing
$combinedLookupImage=[System.Drawing.Image]::FromFile((Join-Path $combinedKit 'working_data\campaign_maps\3k_dlc07_main_map\3k_main_campaign_map_lookup.bmp'))
$combinedLookupW=$combinedLookupImage.Width;$combinedLookupH=$combinedLookupImage.Height;$combinedLookupImage.Dispose()
$combinedShaders=@('dx11_shaders__ui_overlay.xml.hlsl_sm_5_0__ps_ps_ldr_post_process.fxc','dx11_shaders__ui_overlay.xml.hlsl_enable_shadows_sm_5_0__ps_ps_ldr_post_process.fxc')
foreach($combinedShader in $combinedShaders){
 $combinedOriginal=Join-Path $combinedA ('shader-source\'+$combinedShader)
 & $combinedBasePython -B -c "import sys;sys.path.insert(0,'scripts');from analyze_reference import inventory;from pathlib import Path;import tw3k_paths;p=tw3k_paths.GAME_DATA/'shaders.pack';e=next(e for e in inventory(p,False)['entries'] if e['path']=='fxc/$combinedShader');f=p.open('rb');f.seek(e['offset']);d=Path(r'$combinedOriginal');d.parent.mkdir(exist_ok=True);d.write_bytes(f.read(e['size']))"
 if($LASTEXITCODE -ne 0){throw 'Shader extraction failed'}
 & (Join-Path $combinedApp 'Tw3kTheatreShader.exe') $combinedOriginal (Join-Path $combinedStage ('fxc\'+$combinedShader)) $combinedLookupW $combinedLookupH
 if($LASTEXITCODE -ne 0){throw 'UI shader dimension update failed'}
}
'Native map, routes, borders, lookup and UI shader dimensions rebuilt'
