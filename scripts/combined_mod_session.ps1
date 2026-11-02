param([ValidateSet('romance','historical','capture-romance','capture-historical','runtime','clean','vanilla','restore','status')][string]$Mode='status',[string]$RuntimePackPath='',[string]$ValidationPackPath='',[switch]$DeduplicateNativeBackups)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'game_session.ps1')
$combinedRoot=Split-Path -Parent $PSScriptRoot;$combinedA=if($env:TW3K_WORKSPACE){$env:TW3K_WORKSPACE}else{Join-Path $combinedRoot 'work\all-directions-30'};. (Join-Path $PSScriptRoot 'tw3k_env.ps1');$combinedGame=Get-Tw3kPath 'game_dir';$combinedCfg=Join-Path (Get-Tw3kPath 'ca_appdata') 'scripts';$combinedLedger=Join-Path $combinedA 'combined-session.json';$combinedOwner='regions-and-battles';$combinedUtf8=[Text.UTF8Encoding]::new($false)
if($Mode -eq 'status'){if(Test-Path -LiteralPath $combinedLedger){Get-Content -LiteralPath $combinedLedger};exit 0}
$combinedOperation=Enter-GameOperation $combinedRoot;$combinedState=$null;$combinedPreparing=$false
try{
 if(Get-Process Three_Kingdoms -ErrorAction SilentlyContinue){throw 'Game is running; session files will not be changed'}
 $combinedPaths=@((Join-Path $combinedGame 'data\tw3k_exp_build_input.pack'),(Join-Path $combinedCfg 'user.script.txt'),(Join-Path $combinedGame 'used_mods.txt'),(Join-Path $combinedGame 'data\3k_main_campaign_map\victory_objectives.txt'),(Join-Path $combinedCfg 'preferences.script.txt'),(Join-Path $combinedGame 'data\campaigns\3k_main_campaign_map\startpos_romance.esf'),(Join-Path $combinedGame 'data\campaigns\3k_main_campaign_map\startpos_historical.esf'),(Join-Path $combinedGame 'data\campaign_maps\3k_dlc07_main_map\hlp_data.esf'),(Join-Path $combinedGame 'data\campaign_maps\3k_dlc07_main_map\spd_data.esf'),(Join-Path $combinedGame 'data\tw3k_regions_and_battles.pack'),(Join-Path $combinedGame 'data\tw3k_combined_validation.pack'))
 if($Mode -eq 'restore' -or $Mode.StartsWith('capture-')){
  if(-not(Test-Path -LiteralPath $combinedLedger)){throw 'No active combined session'}
  $combinedState=Get-Content -LiteralPath $combinedLedger -Raw|ConvertFrom-Json
  if($combinedState.schema -ne 2 -or $combinedState.restored){throw 'No active version-2 session; legacy recovery must be explicit'}
  Assert-GameSessionLock $combinedRoot $combinedOwner $combinedState
  if($Mode -eq 'restore'){Restore-GameSession $combinedRoot $combinedOwner $combinedLedger $combinedState $combinedPaths;'Original game files and settings restored';exit 0}
  $captureMode=$Mode.Substring(8)
  if($combinedState.lastMode -ne $captureMode){throw 'Capture mode does not match last compiler launch'}
  Assert-SessionFiles $combinedState $combinedPaths -AllowGenerated
  $combinedGen=Join-Path $combinedA 'generated';New-Item -ItemType Directory -Path $combinedGen -Force|Out-Null
  $started=([DateTimeOffset]$combinedState.lastStartedUtc).UtcDateTime
  $capturePaths=@((Join-Path $combinedGame ('data\campaigns\3k_main_campaign_map\startpos_'+$captureMode+'.esf')),$combinedPaths[7],$combinedPaths[8])
  foreach($path in $capturePaths){$file=Get-Item -LiteralPath $path;if($file.Length -lt 1024 -or $file.LastWriteTimeUtc -lt $started){throw 'Missing or stale generated file'}}
  foreach($path in $capturePaths){$file=Get-Item -LiteralPath $path;Copy-Item -LiteralPath $path -Destination (Join-Path $combinedGen $file.Name) -Force;Update-SessionFileHash $combinedLedger $combinedState $path}
  & $(. (Join-Path $PSScriptRoot 'tw3k_env.ps1');Get-Tw3kBasePython) -B (Join-Path $PSScriptRoot 'record_native_generated_state.py') $captureMode --capture-ledger $combinedLedger
  if($LASTEXITCODE -ne 0){throw 'Captured files lack matching launch/compiler/map provenance'}
  'Captured '+$captureMode+' startpos and fresh HLP/SPD';exit 0
 }
 $combinedState=Open-GameSession $combinedRoot $combinedOwner $combinedLedger $combinedA $combinedPaths @($combinedPaths[5],$combinedPaths[6],$combinedPaths[7],$combinedPaths[8]);$combinedPreparing=$true
 if($DeduplicateNativeBackups){
  & $(. (Join-Path $PSScriptRoot 'tw3k_env.ps1');Get-Tw3kPython) -B (Join-Path $PSScriptRoot 'deduplicate_native_session_originals.py')
  if($LASTEXITCODE -ne 0){throw 'Immutable native backup identity/deduplication failed'}
 }
 if($Mode -in @('romance','historical')){
  $combinedState.compilerSources=Get-CompilerLaunchSources $combinedA
  Set-GameSessionFile $combinedLedger $combinedState $combinedPaths[0] -Source (Join-Path $combinedA 'compiler.pack')
  Set-GameSessionFile $combinedLedger $combinedState $combinedPaths[3] -Source (Join-Path $combinedA 'work\build-input\db\victory_objectives.txt')
  $combinedScript="mod tw3k_exp_build_input.pack;`nprocess_campaign_startpos 3k_main_campaign_map $Mode;`nprocess_campaign_ai_map_data;`nquit_after_campaign_processing;`n-no_crashlink;`n"
 }else{
  Clear-GameSessionFiles $combinedLedger $combinedState $combinedPaths @(3,5,6,7,8)
  if($Mode -eq 'vanilla'){
   Set-GameSessionFile $combinedLedger $combinedState $combinedPaths[10] -Source (Join-Path $combinedA 'validation-vanilla.pack')
   $combinedScript="mod tw3k_combined_validation.pack;`n-no_crashlink;`n"
  }else{
   $runtime=if($RuntimePackPath){[IO.Path]::GetFullPath($RuntimePackPath)}else{Join-Path $combinedRoot 'output\tw3k_regions_and_battles.pack'}
   if(-not $runtime.StartsWith(($combinedRoot+'\output\'),[StringComparison]::OrdinalIgnoreCase) -or -not(Test-Path -LiteralPath $runtime -PathType Leaf)){throw 'Runtime pack must be existing workspace output'}
   # Only published (gated) packs: an .ungated or .previous-* copy never installs.
   if([IO.Path]::GetFileName($runtime) -match '\.(ungated|previous-\d+)\.pack$'){throw 'Runtime pack is an ungated or superseded copy: '+$runtime}
   Set-GameSessionFile $combinedLedger $combinedState $combinedPaths[9] -Source $runtime
   $combinedScript="mod tw3k_regions_and_battles.pack;`n-no_crashlink;`n"
   if($Mode -eq 'runtime'){
    $fixture=if($ValidationPackPath){[IO.Path]::GetFullPath($ValidationPackPath)}else{Join-Path $combinedA 'validation.pack'}
    if(-not $fixture.StartsWith(($combinedA+'\'),[StringComparison]::OrdinalIgnoreCase) -or -not(Test-Path -LiteralPath $fixture -PathType Leaf)){throw 'Validation fixture must be existing task workspace pack'}
    if(-not [Text.Encoding]::UTF8.GetString([IO.File]::ReadAllBytes($fixture)).Contains('TW3K_ACTION_V1')){throw 'Validation fixture uses an old action protocol; regenerate it before runtime QA'}
    Set-GameSessionFile $combinedLedger $combinedState $combinedPaths[10] -Source $fixture
    $combinedScript="mod tw3k_regions_and_battles.pack;`nmod tw3k_combined_validation.pack;`n-no_crashlink;`n"
   }
  }
 }
 Set-GameSessionFile $combinedLedger $combinedState $combinedPaths[1] -Text $combinedScript -UseText
 Set-GameSessionFile $combinedLedger $combinedState $combinedPaths[2] -Text '' -UseText
 if(Test-Path -LiteralPath $combinedPaths[4]){
  $prefs=[IO.File]::ReadAllText($combinedPaths[4])
  foreach($setting in @(@('gfx_fullscreen','false'),@('x_res','1280'),@('y_res','720'),@('ui_scale','1'),@('write_preferences_at_exit','false'))){$prefs=[regex]::Replace($prefs,('(?m)^'+$setting[0]+' [^;]+;'),($setting[0]+' '+$setting[1]+';'))}
  Set-GameSessionFile $combinedLedger $combinedState $combinedPaths[4] -Text $prefs -UseText
 }
 $combinedState.actionSession=[Guid]::NewGuid().ToString('N')
 [IO.File]::WriteAllText((Join-Path $combinedA 'action.txt'),'',$combinedUtf8)
 [IO.File]::WriteAllText((Join-Path $combinedA 'action-session.txt'),$combinedState.actionSession,$combinedUtf8)
 $combinedState.lastMode=$Mode;$combinedState.lastStartedUtc=[DateTime]::UtcNow.ToString('o');Save-GameSession $combinedLedger $combinedState
 $style=if($Mode -in @('romance','historical')){'Hidden'}else{'Normal'}
 $process=Start-Process -FilePath (Join-Path $combinedGame 'Three_Kingdoms.exe') -ArgumentList 'temp_file.txt;' -WorkingDirectory $combinedGame -WindowStyle $style -PassThru
 $combinedPreparing=$false
 [pscustomobject]@{mode=$Mode;pid=$process.Id;startedUtc=$combinedState.lastStartedUtc}|ConvertTo-Json|Set-Content -LiteralPath (Join-Path $combinedA 'process.json') -Encoding utf8
 'Combined '+$Mode+' started, PID '+$process.Id
 $combinedLaunched=$true
}catch{
 $failure=$_
 if($combinedPreparing -and -not(Get-Process Three_Kingdoms -ErrorAction SilentlyContinue)){Undo-FailedGamePreparation $combinedRoot $combinedOwner $combinedLedger $combinedState $combinedPaths}
 throw $failure
}finally{$combinedOperation.Dispose()}
# Explicit success code: callers check $LASTEXITCODE after '& combined_mod_session.ps1'.
if($combinedLaunched){exit 0}
