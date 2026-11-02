# -Modes romance: test candidate with only the romance start compiled (about 3 minutes
# faster); the historical startpos is left out, so historical must not be started.
# A mode whose captured startpos and AI map already match the current map, design,
# map_data and compiler pack (native_compilation_state.verify_generated) is not
# compiled again; -ForceCompile compiles every requested mode regardless.
param([switch]$KeepSession,[switch]$RetireArchivedCompilerInputs,[switch]$ForceCompile,[ValidateSet('romance','historical')][string[]]$Modes=@('romance','historical'))
$ErrorActionPreference='Stop'
$section14Root=Split-Path -Parent $PSScriptRoot
$section14Area=if($env:TW3K_WORKSPACE){$env:TW3K_WORKSPACE}else{Join-Path $section14Root 'work/all-directions-30'}
. (Join-Path $PSScriptRoot 'tw3k_env.ps1');$section14Python=Get-Tw3kPython
Initialize-Tw3kWorktreeLinks $section14Root   # SDK binaries + battle-scenes pack in a worktree (tw3k_env.ps1)
if(Get-Process Three_Kingdoms -ErrorAction SilentlyContinue){throw 'Close game before native compilation'}
$section14T0=Get-Date
function Write-Section14Timing([string]$label){Write-Host ('[package-timing] {0,7:N1}s {1}' -f ((Get-Date)-$section14T0).TotalSeconds,$label)}
# Start positions: shift vanilla characters and move any whose hex an authored
# region took (impassable / foreign) before the startpos compiler reads the kit.
& $section14Python -B (Join-Path $PSScriptRoot 'repair_character_placeholder_positions.py') --apply
if($LASTEXITCODE -ne 0){throw 'Character start position repair failed'}
Write-Section14Timing 'character positions'
& $section14Python -B (Join-Path $PSScriptRoot 'build_combined_mod.py') compiler
if($LASTEXITCODE -ne 0){throw 'Compiler package build failed'}
Write-Section14Timing 'compiler pack'
if($RetireArchivedCompilerInputs){
 & $section14Python -B (Join-Path $PSScriptRoot 'retire_archived_compiler_inputs.py')
 if($LASTEXITCODE -ne 0){throw 'Obsolete compiler archive verification failed'}
}
$section14Compile=@()
foreach($section14Mode in $Modes){
 $section14Current=$false
 if(-not $ForceCompile){
  # A mismatch prints a traceback on stderr; under 'Stop' PowerShell 5.1 would turn
  # that into a terminating error, so only the exit code is read here.
  $section14Preference=$ErrorActionPreference;$ErrorActionPreference='Continue'
  & $section14Python -B -c "import sys;sys.path.insert(0,sys.argv[1]);import campaign_world as c,native_compilation_state as n;n.verify_generated(c.WORKSPACE,(sys.argv[2],))" $PSScriptRoot $section14Mode 2>$null
  $section14Current=($LASTEXITCODE -eq 0);$ErrorActionPreference=$section14Preference
 }
 if($section14Current){Write-Host ('Native '+$section14Mode+' startpos already matches the current sources; not recompiled')}else{$section14Compile+=$section14Mode}
}
Write-Section14Timing 'native provenance check'
$section14Succeeded=$false
try {
 foreach($section14Mode in $section14Compile){
  $global:LASTEXITCODE=0;& (Join-Path $PSScriptRoot 'combined_mod_session.ps1') -Mode $section14Mode
  if($LASTEXITCODE -ne 0){throw ('Native launch failed: '+$section14Mode)}
  & $section14Python -B (Join-Path $PSScriptRoot 'deduplicate_native_session_originals.py')
  if($LASTEXITCODE -ne 0){throw 'Native backup identity check failed'}
  # Wait on the game by name for the whole budget: the launched PID can hand off
  # to a relaunched process (Steam), so a PID wait alone ends too early.
  $section14Started=Get-Date
  do {
   Start-Sleep -Seconds 2
   if((Get-Date)-$section14Started -gt [TimeSpan]::FromMinutes(20)){throw 'Native compilation did not exit within 20 minutes'}
   $section14Running=Get-Process Three_Kingdoms -ErrorAction SilentlyContinue
   if(-not $section14Running){Start-Sleep -Seconds 1;$section14Running=Get-Process Three_Kingdoms -ErrorAction SilentlyContinue}
  } while($section14Running)
  $global:LASTEXITCODE=0;& (Join-Path $PSScriptRoot 'combined_mod_session.ps1') -Mode ('capture-'+$section14Mode)
  if($LASTEXITCODE -ne 0){throw ('Native capture failed: '+$section14Mode)}
  Write-Section14Timing ('native compile '+$section14Mode)
 }
 if($section14Compile){   # no game session was opened when every mode was reused
  . (Join-Path $PSScriptRoot 'game_session.ps1')
  $section14Operation=Enter-GameOperation $section14Root
  try {
   $section14Ledger=Join-Path $section14Area 'combined-session.json'
   $section14State=Get-Content -LiteralPath $section14Ledger -Raw|ConvertFrom-Json
   Assert-GameSessionLock $section14Root 'regions-and-battles' $section14State
   Clear-GameSessionFiles $section14Ledger $section14State @($section14State.files.path) @(0)
  } finally {$section14Operation.Dispose()}
  & $section14Python -B (Join-Path $PSScriptRoot 'deduplicate_compiler_session_input.py')
  if($LASTEXITCODE -ne 0){throw 'Retired compiler input identity check failed'}
 }
 & $section14Python -B (Join-Path $PSScriptRoot 'build_combined_mod.py') runtime --terrain-repair-candidate ('--modes='+($Modes -join ','))
 if($LASTEXITCODE -ne 0){throw 'Candidate packaging or final pack gates failed'}
 Write-Section14Timing 'runtime pack and gates'
 $section14Succeeded=$true
} catch {
 $section14Failure=$_
} finally {
 $section14Ledger=Join-Path $section14Area 'combined-session.json'
 if((-not $KeepSession -or -not $section14Succeeded) -and (Test-Path -LiteralPath $section14Ledger)){
  # A failure can leave the compiler running with compile commands installed;
  # give it a moment, then stop it so the original game files are always restored.
  $section14Deadline=(Get-Date).AddMinutes(2)
  while((Get-Process Three_Kingdoms -ErrorAction SilentlyContinue) -and (Get-Date) -lt $section14Deadline){Start-Sleep -Seconds 2}
  Get-Process Three_Kingdoms -ErrorAction SilentlyContinue|Stop-Process -Force -ErrorAction SilentlyContinue
  Start-Sleep -Seconds 2
  try {
   $global:LASTEXITCODE=0;& (Join-Path $PSScriptRoot 'combined_mod_session.ps1') -Mode restore
   if($LASTEXITCODE -ne 0){throw 'Native compilation session restoration failed'}
  } catch {
   # Report both, but never let a restore error hide the original failure.
   if($section14Failure){Write-Warning ('Restore after failure also failed: '+$_)}else{$section14Failure=$_}
  }
 }elseif($KeepSession -and (Test-Path -LiteralPath $section14Ledger)){
  Write-Warning 'Session kept: user.script.txt still holds compile commands; run combined_mod_session.ps1 -Mode restore before a normal game launch.'
 }
}
if($section14Failure){throw $section14Failure}
'Candidate includes world-space river pivots, lake leaves, verified water colours and local town plateaus; live visual acceptance remains separate.'
