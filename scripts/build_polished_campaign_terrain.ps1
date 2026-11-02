param([switch]$RebuildGeometry,[switch]$ResumeAfterPolish)
$ErrorActionPreference='Stop'
$tw3kRoot=Split-Path -Parent $PSScriptRoot
$tw3kArea=if($env:TW3K_WORKSPACE){$env:TW3K_WORKSPACE}else{Join-Path $tw3kRoot 'work/all-directions-30'}
. (Join-Path $PSScriptRoot 'tw3k_env.ps1');$tw3kPython=Get-Tw3kPython
Initialize-Tw3kWorktreeLinks $tw3kRoot   # tools, reference, geodata, SDK binaries in a worktree (tw3k_env.ps1)
$tw3kVariants=@(& $tw3kPython -B -c "import sys;sys.path.insert(0,sys.argv[1]);import campaign_world as c;print(*c.TERRAIN_VARIANTS,sep=chr(10))" $PSScriptRoot | Where-Object {$_})
if($LASTEXITCODE -ne 0 -or -not $tw3kVariants){throw 'Could not read campaign_world.TERRAIN_VARIANTS'}
$tw3kTotal=[Diagnostics.Stopwatch]::StartNew();$tw3kTotalStart=Get-Date
$tw3kTimingLog=Join-Path $tw3kArea 'analysis/pipeline-timing.log'
function Write-Tw3kTiming([string]$Step,$Timer){   # a Stopwatch or a TimeSpan
 $elapsed=if($Timer -is [TimeSpan]){$Timer}else{$Timer.Elapsed}
 $line=("{0:yyyy-MM-dd HH:mm:ss}`t{1}`t{2:n1}s" -f (Get-Date),$Step,$elapsed.TotalSeconds)
 ('[timing] '+$line);Add-Content -LiteralPath $tw3kTimingLog -Value $line -Encoding utf8
}
# --- STAGE INVARIANTS (stage_invariants.py) --- read-only checks after a step (a few seconds); exit 1 = an
# invariant this step broke (named in the log and analysis/stage-invariants.jsonl), any other non-zero exit is
# the checker's own problem and only warns. TW3K_INVARIANTS=warn reports failures without stopping, =off skips.
# river_centre_wet only warns in its first rollout; TW3K_INVARIANT_RIVER_CENTRE_WET=hard makes it stop the run.
$env:TW3K_INVARIANT_RUN=$tw3kTotalStart.ToString('yyyyMMdd-HHmmss')
function Invoke-Tw3kInvariants([string]$Step){
 $timer=[Diagnostics.Stopwatch]::StartNew()
 & $tw3kPython -B (Join-Path $PSScriptRoot 'stage_invariants.py') --after $Step
 $code=$LASTEXITCODE
 Write-Tw3kTiming ('stage_invariants.py --after '+$Step) $timer
 if($code -eq 1){throw ('Stage invariant failed after '+$Step+' (see above and '+(Join-Path $tw3kArea 'analysis/stage-invariants.jsonl')+')')}
 if($code -ne 0){'WARNING: stage_invariants.py --after '+$Step+' could not run (exit '+$code+'); continuing'}
}
# --- end STAGE INVARIANTS
# Run independent read-mostly steps concurrently (TW3K_AUDIT_JOBS, default 4:
# each 2048-canvas step holds several GB). Steps start in the given order, so
# list the slowest first. Each step logs to analysis/parallel-logs/; the logs are
# printed in order afterwards. Returns @{step=exitcode}; the caller decides
# which failures abort. -Started: steps already started with Start-Tw3kStep
# (a side lane running beside a foreground step), waited for here as well.
function Start-Tw3kStep([object[]]$s){
 $logDir=Join-Path $tw3kArea 'analysis/parallel-logs';$null=New-Item -ItemType Directory -Force -Path $logDir
 # One log per step: name it after the script and its arguments (paths by leaf).
 $name=$s[0];$tag=(@($s|ForEach-Object{Split-Path -Leaf "$_"}) -join '_') -replace '[^A-Za-z0-9_.-]','_'
 $log=Join-Path $logDir ($tag+'.log')
 $argList=@('-B',('"'+(Join-Path $PSScriptRoot $name)+'"'))+@($s|Select-Object -Skip 1|ForEach-Object{if("$_" -match '\s'){'"'+$_+'"'}else{"$_"}})
 $p=Start-Process -FilePath $tw3kPython -ArgumentList $argList -NoNewWindow -PassThru -RedirectStandardOutput $log -RedirectStandardError ($log+'.err')
 $null=$p.Handle   # keep the handle so ExitCode is readable after exit (PowerShell 5.1)
 return ,@($s,$p,[Diagnostics.Stopwatch]::StartNew(),$log)
}
function Invoke-Tw3kParallel([object[]]$Steps,[object[]]$Started=@()){
 $jobs=if($env:TW3K_AUDIT_JOBS){[int]$env:TW3K_AUDIT_JOBS}else{4}
 $queue=[Collections.Generic.Queue[object]]::new();foreach($s in $Steps){$queue.Enqueue($s)}
 $running=@();foreach($r in $Started){$running+=,$r};$done=[ordered]@{}
 while($queue.Count -or $running.Count){
  while($queue.Count -and $running.Count -lt $jobs){$running+=,(Start-Tw3kStep $queue.Dequeue())}
  Start-Sleep -Milliseconds 300
  $still=@()
  foreach($r in $running){
   if($r[1].HasExited){
    $r[1].WaitForExit();$label=($r[0] -join ' ');$done[$label]=@($r[1].ExitCode,$r[3])
    # Process run time, not time until noticed (a side lane is polled only after its foreground step).
    Write-Tw3kTiming ($label+' (parallel)') ($r[1].ExitTime-$r[1].StartTime)|Out-Host   # keep text out of the return value
   }else{$still+=,$r}
  }
  $running=$still
 }
 foreach($label in $done.Keys){
  $code,$log=$done[$label]
  Get-Content $log|Out-Host
  if($code -ne 0){Get-Content ($log+'.err') -Tail 30|Out-Host}
 }
 $codes=@{};foreach($label in $done.Keys){$codes[$label]=$done[$label][0]};return $codes
}
# Run a dependency graph of steps: nodes @(name,@(script,args...),@(names it waits for)).
# A node starts once all it waits for exited 0, at most TW3K_AUDIT_JOBS at once; of
# the ready nodes the earliest listed starts first (list them in the serial order).
# Each log is printed when its step exits. On a failure the running nodes finish,
# nothing new starts, and the failed steps are named.
function Invoke-Tw3kGraph([object[]]$Nodes){
 $jobs=if($env:TW3K_AUDIT_JOBS){[int]$env:TW3K_AUDIT_JOBS}else{4}
 $pending=[Collections.Generic.List[object]]::new();foreach($n in $Nodes){$pending.Add($n)}
 $running=@();$ok=@{};$failed=@()
 while($running.Count -or ($pending.Count -and -not $failed)){
  if(-not $failed){
   foreach($n in @($pending)){
    if($running.Count -ge $jobs){break}
    if(-not @($n[2]|Where-Object{-not $ok[$_]}).Count){$running+=,@($n[0],(Start-Tw3kStep $n[1]));$null=$pending.Remove($n)}
   }
   if(-not $running.Count){throw ('Step graph cannot proceed (unknown or cyclic dependency): '+(@($pending|ForEach-Object{$_[0]}) -join ', '))}
  }
  Start-Sleep -Milliseconds 300
  $still=@()
  foreach($r in $running){
   $p=$r[1][1]
   if($p.HasExited){
    $p.WaitForExit();$label=($r[1][0] -join ' ')
    Write-Tw3kTiming ($label+' (parallel)') ($p.ExitTime-$p.StartTime)|Out-Host
    Get-Content $r[1][3]|Out-Host
    if($p.ExitCode -eq 0){$ok[$r[0]]=$true}else{Get-Content ($r[1][3]+'.err') -Tail 30|Out-Host;$failed+=($label+' (exit '+$p.ExitCode+')')}
   }else{$still+=,$r}
  }
  $running=$still
 }
 $skipped=if($pending.Count){'; not started: '+(@($pending|ForEach-Object{$_[0]}) -join ', ')}else{''}
 if($failed){throw ('Terrain stage failed: '+($failed -join ', ')+$skipped+' (logs in '+(Join-Path $tw3kArea 'analysis/parallel-logs')+')')}
}
# STAGE INVARIANTS: guard on the input map before any terrain work (land regions split by rivers: v17 load crash)
Invoke-Tw3kInvariants 'map'
if($RebuildGeometry){
 $tw3kTimer=[Diagnostics.Stopwatch]::StartNew()
 & $tw3kPython -B (Join-Path $PSScriptRoot 'terrain_checkpoint.py') mark geometry
 if($LASTEXITCODE -ne 0){throw 'Pre-export asset checkpoint failed'}
 Write-Tw3kTiming 'terrain_checkpoint.py mark geometry' $tw3kTimer
 & (Join-Path $PSScriptRoot 'build_combined_geometry.ps1')
 if($LASTEXITCODE -ne 0){throw 'Native map export failed'}
 $tw3kTimer=[Diagnostics.Stopwatch]::StartNew()
 & $tw3kPython -B (Join-Path $PSScriptRoot 'repair_campaign_world_extent.py')
 if($LASTEXITCODE -ne 0){throw 'Theatre coordinate repair failed'}
 Write-Tw3kTiming 'repair_campaign_world_extent.py' $tw3kTimer
 $tw3kTimer=[Diagnostics.Stopwatch]::StartNew()
 & $tw3kPython -B (Join-Path $PSScriptRoot 'terrain_checkpoint.py') capture geometry
 if($LASTEXITCODE -ne 0){throw 'Geometry baseline checkpoint failed'}
 Write-Tw3kTiming 'terrain_checkpoint.py capture geometry' $tw3kTimer
 Invoke-Tw3kInvariants 'geometry'   # STAGE INVARIANTS: the refreshed base (no overflow land meshes)
}else{
 # Without a new export, polish reads terrain-before-polish: refuse it up front when it was
 # exported from another map.hex or design (its generation.json records them) instead of
 # polishing the old geometry under the new map.
 & $tw3kPython -B (Join-Path $PSScriptRoot 'terrain_checkpoint.py') check geometry
 if($LASTEXITCODE -ne 0){throw 'Geometry base belongs to another map/design; rerun with -RebuildGeometry'}
}
if($ResumeAfterPolish){
 $tw3kTimer=[Diagnostics.Stopwatch]::StartNew()
 & $tw3kPython -B (Join-Path $PSScriptRoot 'terrain_checkpoint.py') restore polish
 if($LASTEXITCODE -ne 0){throw 'No matching verified polish checkpoint; run the full terrain build'}
 Write-Tw3kTiming 'terrain_checkpoint.py restore polish' $tw3kTimer
 $tw3kTimer=[Diagnostics.Stopwatch]::StartNew()
 & $tw3kPython -B (Join-Path $PSScriptRoot 'polish_campaign_terrain.py') --cache-only
 if($LASTEXITCODE -ne 0){throw 'Current terrain display-cache rebuild failed'}
 Write-Tw3kTiming 'polish_campaign_terrain.py --cache-only' $tw3kTimer
}
else{
 $tw3kTimer=[Diagnostics.Stopwatch]::StartNew()
 & $tw3kPython -B (Join-Path $PSScriptRoot 'restore_native_height_baseline.py')
 if($LASTEXITCODE -ne 0){throw 'Native height baseline restoration failed'}
 Write-Tw3kTiming 'restore_native_height_baseline.py' $tw3kTimer
}
$tw3kSideLaneSteps=@('sync_campaign_sea_height_texture.py','expand_campaign_water_colour.py','translate_campaign_environment_spheres.py','extend_campaign_ocean_background.py');$tw3kSideLane=@()
foreach($tw3kStep in @('restore_native_water_meshes.py','polish_campaign_terrain.py','sync_final_terrain_surface.py','sync_campaign_sea_height_texture.py','expand_campaign_water_colour.py','translate_campaign_environment_spheres.py','extend_campaign_ocean_background.py')){
 # audit_campaign_polish/audit_city_ground_coverage/audit_terrain_continuity (and the
 # edited-coverage and town-elevation audits below) used to run here and again after
 # the seam repair, which rewrites their inputs and outputs; only the post-repair run counts.
 if($ResumeAfterPolish -and $tw3kStep -eq 'polish_campaign_terrain.py'){continue}
 if($tw3kStep -eq 'sync_final_terrain_surface.py'){
  # Side lane beside sync_final (~30 s of its ~160 s): the sea height texture, water
  # colour, environment spheres and ocean background read neither the height map nor
  # anything sync_final or the camera sync write (the sea height sync reads polish's
  # analysis/river-water-level.npy, written before), and write files no step before
  # their own audits reads. sync_final rewrites lf_height_map.dds in place (smooth_variant), so no LF reader may
  # run here: stage_invariants side_lane_lf checks this list at the pipeline start. snow_bake.py reads the LF
  # (visible peaks) and runs as graph node 'snow' below, after the LF's last writer.
  $tw3kSideLane=@();foreach($tw3kSide in $tw3kSideLaneSteps){$tw3kSideLane+=,(Start-Tw3kStep @($tw3kSide))}
 }
 # Until Invoke-Tw3kParallel collects the side lane, a failing foreground step must
 # not orphan its processes: the finally waits for them (they finish in seconds)
 # before the error propagates. $tw3kSideLane is emptied once collected.
 try{
  $tw3kArgs=@('-B',(Join-Path $PSScriptRoot $tw3kStep))
  $tw3kTimer=[Diagnostics.Stopwatch]::StartNew()
  if($tw3kStep -eq 'polish_campaign_terrain.py' -and $tw3kVariants.Count -eq 1){
   # A single variant (the 190 campaign): one serial run, so the global setup is
   # computed once instead of again in a --merge process.
   $tw3kPolishLog=Join-Path $tw3kArea ('analysis/polish-'+$tw3kVariants[0]+'.log')
   # --- MEM-BUDGET --- polish is the pipeline's largest step: wait (TW3K_MEMORY_WAIT_MIN, default 30 min)
   # until free commit plus page-file growth room covers its measured need (memory_budget.py), else stop
   # here with the reason. Its pools size themselves from free commit; no artifact depends on this.
   & $tw3kPython -B (Join-Path $PSScriptRoot 'memory_budget.py') gate polish --wait-min $(if($env:TW3K_MEMORY_WAIT_MIN){$env:TW3K_MEMORY_WAIT_MIN}else{30})
   if($LASTEXITCODE -ne 0){throw 'Not enough free commit for polish (memory_budget.py gate polish)'}
   # --- end MEM-BUDGET
   # Start-Process keeps stderr out of PowerShell's error stream (5.1 aborts on
   # the first native stderr line under ErrorActionPreference Stop).
   $tw3kSerial=Start-Process -FilePath $tw3kPython -ArgumentList @('-B',('"'+(Join-Path $PSScriptRoot $tw3kStep)+'"'),'--skip-cache') -NoNewWindow -PassThru -Wait -RedirectStandardOutput $tw3kPolishLog -RedirectStandardError ($tw3kPolishLog+'.err')
   if($tw3kSerial.ExitCode -ne 0){Get-Content ($tw3kPolishLog+'.err') -Tail 30;throw ('Polish failed (see '+$tw3kPolishLog+'.err), exit '+$tw3kSerial.ExitCode)}
   Get-Content $tw3kPolishLog
   Write-Tw3kTiming 'polish_campaign_terrain.py (serial)' $tw3kTimer
  }elseif($tw3kStep -eq 'polish_campaign_terrain.py'){
   # One process per terrain variant (campaign_world.TERRAIN_VARIANTS; only dlc07 since
   # the 190 campaign renders nothing else) for the heavy per-variant work, then a merge
   # run builds the tile cache. Each process logs to analysis/polish-<variant>.log.
   $tw3kPolish=@()
   foreach($tw3kPolishVariant in $tw3kVariants){
    $env:TW3K_POLISH_VARIANTS=$tw3kPolishVariant
    $tw3kPolishLog=Join-Path $tw3kArea ('analysis/polish-'+$tw3kPolishVariant+'.log')
    $tw3kProcess=Start-Process -FilePath $tw3kPython -ArgumentList @('-B',('"'+(Join-Path $PSScriptRoot $tw3kStep)+'"'),'--loop-only') -NoNewWindow -PassThru -RedirectStandardOutput $tw3kPolishLog -RedirectStandardError ($tw3kPolishLog+'.err')
    $null=$tw3kProcess.Handle;$tw3kPolish+=$tw3kProcess   # keep the handle so ExitCode is readable after exit (PowerShell 5.1)
   }
   Remove-Item Env:TW3K_POLISH_VARIANTS
   $tw3kPolish|Wait-Process
   foreach($tw3kProcess in $tw3kPolish){$tw3kProcess.WaitForExit();if($tw3kProcess.ExitCode -ne 0){throw ('Parallel polish failed (see analysis/polish-*.log.err), exit '+$tw3kProcess.ExitCode)}}
   foreach($tw3kPolishVariant in $tw3kVariants){Get-Content (Join-Path $tw3kArea ('analysis/polish-'+$tw3kPolishVariant+'.log'))}
   Write-Tw3kTiming 'polish_campaign_terrain.py --loop-only (parallel)' $tw3kTimer
   $tw3kTimer=[Diagnostics.Stopwatch]::StartNew()
   & $tw3kPython -B (Join-Path $PSScriptRoot $tw3kStep) --merge
   if($LASTEXITCODE -ne 0){throw 'Polish merge failed'}
   Write-Tw3kTiming 'polish_campaign_terrain.py --merge' $tw3kTimer
  }elseif($tw3kStep -notin $tw3kSideLaneSteps){
  & $tw3kPython @tw3kArgs
  if($LASTEXITCODE -ne 0){throw ('Terrain stage failed: '+$tw3kStep)}
  Write-Tw3kTiming $tw3kStep $tw3kTimer
  }
  if($tw3kStep -eq 'sync_final_terrain_surface.py'){
   $tw3kTimer=[Diagnostics.Stopwatch]::StartNew()
   & $tw3kPython -B (Join-Path $PSScriptRoot 'sync_authored_camera_heights.py')
   if($LASTEXITCODE -ne 0){throw 'Authored camera/CPU town height synchronization failed'}
   Write-Tw3kTiming 'sync_authored_camera_heights.py' $tw3kTimer
   $tw3kSideCodes=Invoke-Tw3kParallel @() $tw3kSideLane;$tw3kSideLane=@()
  }
 }finally{
  foreach($tw3kSide in @($tw3kSideLane)){
   if(-not $tw3kSide[1].HasExited){'Waiting for side-lane step '+$tw3kSide[0][0]+' before failing'|Out-Host}
   $tw3kSide[1].WaitForExit()
  }
  $tw3kSideLane=@()
 }
 if($tw3kStep -eq 'sync_final_terrain_surface.py'){
  $tw3kSideFailed=@($tw3kSideCodes.Keys|Where-Object{$tw3kSideCodes[$_] -ne 0})
  if($tw3kSideFailed){throw ('Terrain stage failed: '+($tw3kSideFailed -join ', ')+' (see analysis/parallel-logs)')}
  Invoke-Tw3kInvariants 'sync_final'   # STAGE INVARIANTS: the side lane is collected, nothing runs
  # Salt-pan pond floors take their colour from sea_colour.dds: painted after the water-colour
  # side lane (writes both textures) and sync_final (its polish --cache-only writes
  # salt-pan-cover.npy). Reads that cover and both sea_colour.dds; writes the two textures
  # and analysis/salt-pan-sea-colour.json. No step of the graph below reads them.
  # SALT LAKES: the same step first paints the saline lakes turquoise (biome_profiles REGIONS
  # salt_lakes; also reads work/final-map/map.hex and analysis/west-real-terrain.npz).
  $tw3kTimer=[Diagnostics.Stopwatch]::StartNew()
  & $tw3kPython -B (Join-Path $PSScriptRoot 'paint_salt_pan_sea_colour.py')
  if($LASTEXITCODE -ne 0){throw 'Salt-pan sea colour painting failed'}
  Write-Tw3kTiming 'paint_salt_pan_sea_colour.py' $tw3kTimer
  # The steps between the final surface and the ocean background, as a graph of the
  # files each one reads and writes (audit-hooked runs, 2026-10-22). Prefab chain:
  # global_props.bin and the loose bmd_objects leaves pass reconcile, resource scenes,
  # river roots, world-space leaves and lake leaves (world-space leaves first: lake
  # leaf files also hold them, and the lake step records each file's pre-translation
  # state for the final pack audit). Tile chain: the tile bake inside sync_final's polish
  # --cache-only (tile_bake.py: runtime floor placements and the obsolete-overlay cull, the
  # former placements/overlays steps) leaves tile_list.bin to the overlay audit and the
  # calibrated ranges; then the cache height sync rewrites tile_map.tiles, which the cache
  # audit reads. The gap fill reads the prefab reconciliation report and the tile bake's
  # cleared-overlay mask; the island patches edit the land meshes it rebuilds. The
  # foliage step (tree list) shares no file with any of them. hehuang-lake, river-water, mountain-props,
  # coast-props, landmark-props and ambient all rewrite global_props.bin, so they run in that order.
  Invoke-Tw3kGraph @(
   # SNOW: snow_bake.py (mask, material, lighting groups, analysis/snow-bake.json) reads map.hex, west-real-terrain
   # and the final LF. The LF's writers are restore_native_height_baseline, polish and sync_final (smooth_variant),
   # all done before this graph; no graph node writes it. No node reads the snow products (environment_collection,
   # written in the side lane, only names the lighting directory); the snow invariants run at the graph join.
   @('snow',@('snow_bake.py'),@()),
   @('foliage',@('polish_campaign_foliage.py'),@()),
   @('prefabs',@('reconcile_terrain_prefabs.py','--apply'),@()),
   @('resource-scenes',@('complete_campaign_resource_scenes.py'),@('prefabs')),
   # STAGE INVARIANTS checkpoint resource_scenes (prop_lf_offset / prop_scene_lf_offset: props vs the final LF), run
   # once the leaves are in world space (scene-check below). --node: a checker failure only warns (exit 0); an
   # invariant failure (exit 1) fails the graph like any node, naming this checkpoint.
   @('river-roots',@('repair_campaign_river_roots.py'),@('resource-scenes')),
   @('world-space-leaves',@('translate_world_space_leaves.py'),@('river-roots')),
   # scene-check after world-space-leaves: before it the leaves still hold their donor scene's coordinates (the root
   # translation is written into them there), so every clone of a template read as its donor (v21b run 034627: 92 off)
   @('scene-check',@('stage_invariants.py','--after','resource_scenes','--node'),@('world-space-leaves')),
   @('lake-leaves',@('translate_campaign_lake_instances.py','--apply'),@('world-space-leaves')),
   @('hehuang-lake',@('hehuang_lake_prop.py','--apply'),@('lake-leaves')),   # HEHUANG PATCH: Qinghai lake prop leaf
   # RIVER WATER: frontier river/lake water surfaces (river_water_props.py) as vanilla river-model scenes,
   # prefab numbers 9000-9399. Reads the final LF, polish's analysis/river-water-level.npy (+ stamp),
   # work/final-map/map.hex, global_props.bin; writes global_props.bin, loose bmd_objects.bin + its
   # scenes, models/tw3k_water_*.wsmodel(.rigid_model_v2), analysis/river-water-props.json.
   @('river-water',@('river_water_props.py','--apply'),@('hehuang-lake')),
   @('mountain-props',@('frontier_mountain_props.py','--apply'),@('river-water')),   # FARMS-ROCKS PATCH: vanilla mountain clusters on frontier mountains
   @('coast-props',@('coast_props.py','--apply'),@('mountain-props')),   # COAST PATCH: new-port docks, coastal-wave sounds
   # Landmark props (landmarks_props.py): Great Wall pieces/towers for the Han walls and passes,
   # stone-forest pillars for Bailongdui, as new world-space scenes in global_props.bin.
   # Reads global_props.bin, analysis/polished-surface-<map>.npy, work/final-map/map.hex; writes
   # global_props.bin, loose bmd_objects.bin + its scenes, analysis/landmark-props.json.
   @('landmark-props',@('landmarks_props.py','--apply'),@('coast-props')),
   # AMBIENT: vanilla emitters/fauna/workers on the new regions, last on the prefab chain
   # (global_props.bin, bmd_objects.bin, new loose leaves); reads the foliage step's tree list.
   @('ambient',@('place_ambient.py','--apply'),@('landmark-props','foliage')),
   @('overlay-audit',@('audit_native_overlay_preservation.py'),@()),
   @('calibrated-ranges',@('repair_calibrated_native_tile_ranges.py','--apply'),@('overlay-audit')),
   @('cache-heights',@('sync_native_cache_height_ranges.py','--apply','--output',(Join-Path $tw3kArea 'analysis/native-calibrated-cache-ranges.json')),@('calibrated-ranges')),
   @('cache-audit',@('audit_native_cache_preservation.py'),@('cache-heights')),
   @('gap-fill',@('fill_polished_mesh_gaps.py'),@('prefabs')),
   @('islands',@('add_island_ground_patches.py','--refresh-base'),@('gap-fill')),
   # RIVER BEDS: river_bed_mesh.py re-triangulates the coarse land-mesh faces that disagree with the LF
   # about the frontier river/lake water (river-water's surfaces) onto the LF lattice, in existing
   # land_mesh_* files only. Last coarse-mesh edit before the seam scan/repair below; reads the final
   # LF, the river level + stamp, map.hex; writes global_meshes/land_mesh_*, analysis/river-bed-mesh.json.
   @('river-beds',@('river_bed_mesh.py','--apply'),@('islands','river-water')))
  Invoke-Tw3kInvariants 'graph'   # STAGE INVARIANTS: at the join (no node running); failures name the writing nodes
 }
 if($tw3kStep -eq 'extend_campaign_ocean_background.py'){
  $tw3kTimer=[Diagnostics.Stopwatch]::StartNew()
  & $tw3kPython -B (Join-Path $PSScriptRoot 'restore_native_water_meshes.py') --audit
  if($LASTEXITCODE -ne 0){throw 'Native river/ocean water geometry preservation failed'}
  Write-Tw3kTiming 'restore_native_water_meshes.py --audit' $tw3kTimer
 }
 if($tw3kStep -eq 'polish_campaign_terrain.py'){
  $tw3kTimer=[Diagnostics.Stopwatch]::StartNew()
  & $tw3kPython -B (Join-Path $PSScriptRoot 'terrain_checkpoint.py') capture polish
  if($LASTEXITCODE -ne 0){throw 'Polish generation checkpoint failed'}
  Write-Tw3kTiming 'terrain_checkpoint.py capture polish' $tw3kTimer
  # Fail fast: the native border gates (walls, material/climate cuts) come from
  # polish's own height and blend output and cost seconds; the full run checks
  # them again at the end, ~30 minutes later. --no-tiles: the tile list is still
  # the previous run's here (the cache is rebuilt after polish), so its climate is checked only then.
  if(-not $env:TW3K_SKIP_EARLY_EDGE){
   $tw3kTimer=[Diagnostics.Stopwatch]::StartNew()
   & $tw3kPython -B (Join-Path $PSScriptRoot 'audit_native_edges.py') --strict --no-tiles
   $tw3kEarlyEdge=$LASTEXITCODE
   Write-Tw3kTiming 'audit_native_edges.py --strict (after polish)' $tw3kTimer
   if($tw3kEarlyEdge -ne 0){throw ('Native border seam audit failed right after polish; see '+(Join-Path $tw3kArea 'analysis/native-edge-audit.json')+' (set TW3K_SKIP_EARLY_EDGE=1 to defer it to the end)')}
  }
  Invoke-Tw3kInvariants 'polish'   # STAGE INVARIANTS
 }
}
$tw3kTimer=[Diagnostics.Stopwatch]::StartNew()
& $tw3kPython -B (Join-Path $PSScriptRoot 'audit_water_colour_mapping.py')
if($LASTEXITCODE -ne 0){throw 'Water colour world mapping regression failed'}
Write-Tw3kTiming 'audit_water_colour_mapping.py' $tw3kTimer
foreach($tw3kVariant in $tw3kVariants){
 $tw3kPassed=$false
 for($tw3kPass=0;$tw3kPass -lt 5;$tw3kPass++){
  $tw3kScan=Join-Path $tw3kArea ('analysis/ground-seam-pipeline-'+$tw3kVariant+'-'+$tw3kPass)
  $tw3kTimer=[Diagnostics.Stopwatch]::StartNew()
  & $tw3kPython -B (Join-Path $PSScriptRoot 'audit_ground_seams.py') --stage (Join-Path $tw3kArea 'work/release-input') --variant $tw3kVariant --out $tw3kScan --max-steep-ratio 0.0045
  if($LASTEXITCODE -ne 0){throw 'Full source seam scan failed'}
  Write-Tw3kTiming ('audit_ground_seams.py '+$tw3kVariant) $tw3kTimer
  if($tw3kPass -eq 0){Invoke-Tw3kInvariants 'seam_pass0'}   # STAGE INVARIANTS: classes the seam loop cannot repair
  $tw3kScanResult=Get-Content -LiteralPath (Join-Path $tw3kScan 'ground-seam-scan.json') -Raw | ConvertFrom-Json
  if($tw3kScanResult.first_five_passed){$tw3kPassed=$true;break}
  # A coarse defect that did not shrink and only moved next door is walking (each repair
  # clips the next triangle out): more passes cannot close it, stop with the meshes named.
  if($tw3kPass -gt 0){
   & $tw3kPython -B (Join-Path $PSScriptRoot 'seam_walk_check.py') --variant $tw3kVariant --previous $tw3kPreviousScan --current $tw3kScan
   if($LASTEXITCODE -eq 3){throw ('Seam scan defect is walking instead of closing for '+$tw3kVariant+' (pass '+$tw3kPass+'); see the meshes above and '+(Join-Path $tw3kScan 'ground-seam-scan.json'))}
   if($LASTEXITCODE -ne 0){throw 'Seam walk check failed'}
  }
  $tw3kPreviousScan=$tw3kScan
  if($tw3kPass -eq 4){break}
  $tw3kTimer=[Diagnostics.Stopwatch]::StartNew()
  & $tw3kPython -B (Join-Path $PSScriptRoot 'repair_local_coarse_seams.py') --variant $tw3kVariant --scan $tw3kScan
  if($LASTEXITCODE -ne 0){throw 'Exact coarse partition repair failed'}
  Write-Tw3kTiming ('repair_local_coarse_seams.py '+$tw3kVariant) $tw3kTimer
  $tw3kTimer=[Diagnostics.Stopwatch]::StartNew()
  & $tw3kPython -B (Join-Path $PSScriptRoot 'normalize_coarse_partition_heights.py')
  if($LASTEXITCODE -ne 0){throw 'Coarse partition height synchronization failed'}
  Write-Tw3kTiming 'normalize_coarse_partition_heights.py' $tw3kTimer
 }
 if(-not $tw3kPassed){throw ('Seam scan still fails for '+$tw3kVariant)}
}
# The seam repair removes and re-partitions triangles; close any sliver of
# edited land it left uncovered (exact partition, no re-tessellation).
$tw3kTimer=[Diagnostics.Stopwatch]::StartNew()
& $tw3kPython -B (Join-Path $PSScriptRoot 'patch_coverage_holes.py')
if($LASTEXITCODE -ne 0){throw 'Post-repair coverage hole patch failed'}
Write-Tw3kTiming 'patch_coverage_holes.py' $tw3kTimer
Invoke-Tw3kInvariants 'mesh_final'   # STAGE INVARIANTS: the land meshes as packed (last mesh writer above)
# Post-repair gates and reports. None reads another's output (the strategic-map
# rebuild writes lookup art no audit reads), so they run concurrently, slowest
# first. audit_native_edges: visible-seam gate along the old native border;
# its failure is reported after the previews below, which draw its segments.
# audit_coarse_skirts: flattened 1.0-deep crack skirts show the water plane as
# thin blue lines. audit_native_identity: native area vs vanilla, layer by layer.
# Order: run time of R2e 2026-11-01 (pipeline-timing.log), longest first, so the
# four slots finish together (about 67 s instead of 106 s).
$tw3kPostRepair=Invoke-Tw3kParallel @(
 @('audit_terrain_realism.py','--profiles','--gate','--blend',(Join-Path $tw3kArea 'work/release-input/terrain/campaigns/3k_dlc07_main_map/global_map/global_blend.dds')),   # 59 s; biome step 6 hard gate: desert banned share <= GATE_BANNED (exit 3 fails the build; user rule 2026-10-25)
 @('audit_terrain_continuity.py','--output',(Join-Path $tw3kArea 'terrain-repair/after-continuity.json')),   # 46 s
 @('rebuild_campaign_strategic_maps.py'),                 # 27 s
 @('audit_edited_ground_coverage.py','--strict'),         # 25 s
 @('audit_coarse_skirts.py'),                             # 23 s
 @('audit_native_edges.py','--strict'),                   # 22 s
 @('audit_native_identity.py'),                           # 22 s
 @('audit_local_town_elevations.py'),                     # 14 s
 @('audit_campaign_polish.py'),                           # 8 s
 @('audit_city_ground_coverage.py'),                      # 7 s
 @('prepare_terrain_repair_checks.py'))                   # 1 s
$tw3kEdgeExit=$tw3kPostRepair['audit_native_edges.py --strict']
$tw3kFailed=@($tw3kPostRepair.Keys|Where-Object{$_ -ne 'audit_native_edges.py --strict' -and $tw3kPostRepair[$_] -ne 0})
if($tw3kFailed){throw ('Post-repair step failed: '+($tw3kFailed -join ', ')+' (logs in '+(Join-Path $tw3kArea 'analysis/parallel-logs')+')')}
$tw3kTimer=[Diagnostics.Stopwatch]::StartNew()
& $tw3kPython -B (Join-Path $PSScriptRoot 'preview_terrain_sites.py')
if($LASTEXITCODE -ne 0){throw 'Terrain site previews failed'}
Write-Tw3kTiming 'preview_terrain_sites.py' $tw3kTimer
$tw3kTimer=[Diagnostics.Stopwatch]::StartNew()
if($env:TW3K_KEEP_BAKE_EVIDENCE -eq '1'){
 & $tw3kPython -B (Join-Path $PSScriptRoot 'prune_workspace_backups.py')
}else{
 & $tw3kPython -B (Join-Path $PSScriptRoot 'prune_workspace_backups.py') --apply
}
if($LASTEXITCODE -ne 0){throw 'Workspace backup pruning failed'}
Write-Tw3kTiming 'prune_workspace_backups.py' $tw3kTimer
# Terrain regression (terrain_regression.py, analysis/terrain-regression.md): snapshot the known-problem
# windows and compare them with the previous run's snapshot (the first run: the 2048o baseline). Warn only:
# compare --warn-only always exits 0 and a missing input is a note, so this never fails the pipeline.
$tw3kTimer=[Diagnostics.Stopwatch]::StartNew()
$tw3kRegDir=Join-Path (Split-Path $PSScriptRoot) 'analysis/terrain-regression'
$tw3kRegBase=Get-ChildItem $tw3kRegDir -Directory -Filter 'run-*' -ErrorAction SilentlyContinue|Sort-Object Name|Select-Object -Last 1
$tw3kRegBase=if($tw3kRegBase){$tw3kRegBase.FullName}else{Join-Path $tw3kRegDir '2048o'}
$tw3kRegNew=Join-Path $tw3kRegDir ('run-'+(Get-Date -Format 'yyyyMMdd-HHmmss'))
# This run's start and time so far (the snapshot runs before this run's TOTAL line is written).
$env:TW3K_PIPELINE_STARTED=$tw3kTotalStart.ToString('yyyy-MM-dd HH:mm:ss');$env:TW3K_PIPELINE_ELAPSED_S=[string][Math]::Round($tw3kTotal.Elapsed.TotalSeconds,1)
& $tw3kPython -B (Join-Path $PSScriptRoot 'terrain_regression.py') snapshot --out $tw3kRegNew
if($LASTEXITCODE -eq 0 -and (Test-Path (Join-Path $tw3kRegBase 'metrics.json'))){
 & $tw3kPython -B (Join-Path $PSScriptRoot 'terrain_regression.py') compare $tw3kRegBase $tw3kRegNew --warn-only
}else{'Terrain regression: snapshot failed or no baseline in '+$tw3kRegBase+' (warning only)'}
Write-Tw3kTiming 'terrain_regression.py' $tw3kTimer
if($tw3kEdgeExit -ne 0){throw ('Native border seam audit failed; see '+(Join-Path $tw3kArea 'analysis/native-edge-audit.json')+' and analysis/terrain-preview/contact-sheet.png')}
Write-Tw3kTiming 'TOTAL' $tw3kTotal
'Terrain source rebuilt and structurally audited; native startpos compilation and live QA remain separate.'
