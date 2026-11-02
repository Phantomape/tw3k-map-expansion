param([Parameter(Mandatory=$true)][string]$Action,[int]$TimeoutSeconds=20)
$ErrorActionPreference='Stop'
function Read-SharedActionLog([string]$Path){
 $stream=[IO.File]::Open($Path,[IO.FileMode]::Open,[IO.FileAccess]::Read,([IO.FileShare]::ReadWrite -bor [IO.FileShare]::Delete))
 try{$reader=[IO.StreamReader]::new($stream);try{return $reader.ReadToEnd()}finally{$reader.Dispose()}}finally{$stream.Dispose()}
}
. (Join-Path $PSScriptRoot 'game_session.ps1')
$root=Split-Path -Parent $PSScriptRoot;$area=if($env:TW3K_WORKSPACE){$env:TW3K_WORKSPACE}else{Join-Path $root 'work\all-directions-30'};$ledger=Join-Path $area 'combined-session.json'
if([string]::IsNullOrWhiteSpace($Action) -or $Action -match '[\r\n\x00]'){throw 'Action must be one nonempty line'}
$operation=Enter-GameOperation $root
try{
 $state=Get-Content -LiteralPath $ledger -Raw|ConvertFrom-Json
 if($state.schema -ne 2 -or $state.restored -or $state.lastMode -notin @('runtime','vanilla')){throw 'No active nonce-enabled combined validation session'}
 Assert-GameSessionLock $root 'regions-and-battles' $state
 $session=[IO.File]::ReadAllText((Join-Path $area 'action-session.txt'))
 if($session -notmatch '^[a-f0-9]{32}$' -or $session -ne $state.actionSession){throw 'Validation action session identity changed'}
 $nonce=[Guid]::NewGuid().ToString('N');$log=Join-Path $area 'events.log';$before=Read-SharedActionLog $log
 $actionPath=Join-Path $area 'action.txt';$temporary=$actionPath+'.'+$nonce+'.tmp'
 [IO.File]::WriteAllText($temporary,"TW3K_ACTION_V1`t$session`t$nonce`n$Action",[Text.UTF8Encoding]::new($false))
 if(Test-Path -LiteralPath $actionPath){[IO.File]::Replace($temporary,$actionPath,($temporary+'.old'));Remove-Item -LiteralPath ($temporary+'.old')}else{[IO.File]::Move($temporary,$actionPath)}
 $prefix='CAMPAIGN_ACTION_V1 '+$session+' '+$nonce+' RESULT ';$deadline=[DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
 do{
  Start-Sleep -Milliseconds 200;$after=Read-SharedActionLog $log
  if($after.Length -lt $before.Length){throw 'Validation log replaced; action outcome unknown'}
  $new=$after.Substring($before.Length);$line=@($new -split '\r?\n'|Where-Object {$_.StartsWith($prefix)})|Select-Object -First 1
  if($line){if(-not $line.StartsWith($prefix+'true ')){throw ('Campaign action failed: '+$line)};Write-Output $line;exit 0}
 }while([DateTime]::UtcNow -lt $deadline)
 throw 'No acknowledgement for this session and nonce; outcome unknown. Inspect game/log before retrying.'
}finally{$operation.Dispose()}
