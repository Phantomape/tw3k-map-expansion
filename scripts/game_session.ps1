function Write-SessionJson($Path,$Value) {
 $tmp=$Path+'.'+[Guid]::NewGuid().ToString('N')+'.tmp'
 [IO.File]::WriteAllText($tmp,($Value|ConvertTo-Json -Depth 12),[Text.UTF8Encoding]::new($false))
 if(Test-Path -LiteralPath $Path){[IO.File]::Replace($tmp,$Path,($tmp+'.old'));Remove-Item -LiteralPath ($tmp+'.old')}else{[IO.File]::Move($tmp,$Path)}
}
# Session checks hash every managed file (multi-GB packs included) many times per
# run. Cache each SHA-256 by full path, size and last-write time, as git's stat
# cache does: any write changes the timestamp and forces a fresh hash.
$script:Tw3kHashCachePath=Join-Path (Split-Path -Parent $PSScriptRoot) 'work\session-hash-cache.json'
$script:Tw3kHashCache=$null
function Get-SessionHash($Path) {
 if(-not(Test-Path -LiteralPath $Path -PathType Leaf)){return $null}
 if($null -eq $script:Tw3kHashCache){
  $script:Tw3kHashCache=@{}
  if(Test-Path -LiteralPath $script:Tw3kHashCachePath){try{(Get-Content -LiteralPath $script:Tw3kHashCachePath -Raw|ConvertFrom-Json).PSObject.Properties|ForEach-Object{$script:Tw3kHashCache[$_.Name]=$_.Value}}catch{$script:Tw3kHashCache=@{}}}
 }
 $item=Get-Item -LiteralPath $Path;$key=$item.FullName.ToLowerInvariant();$stamp=('{0}|{1}' -f $item.Length,$item.LastWriteTimeUtc.Ticks)
 $hit=$script:Tw3kHashCache[$key]
 if($hit -and $hit.stamp -eq $stamp){return $hit.sha256}
 $hash=(Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash
 $script:Tw3kHashCache[$key]=[pscustomobject]@{stamp=$stamp;sha256=$hash}
 try{Write-SessionJson $script:Tw3kHashCachePath ([pscustomobject]$script:Tw3kHashCache)}catch{}
 return $hash
}
# Install a source file at a managed game path. Same volume: replace the target
# with a hard link (no multi-GB copy); never write into an existing file, which
# would change every other link to it. Other volumes: copy.
function Install-SessionFile($Source,$Path){
 if(Test-Path -LiteralPath $Path){Remove-Item -LiteralPath $Path -Force}
 if([IO.Path]::GetPathRoot([IO.Path]::GetFullPath($Source)) -eq [IO.Path]::GetPathRoot([IO.Path]::GetFullPath($Path))){
  try{New-Item -ItemType HardLink -Path $Path -Target $Source|Out-Null;return}catch{}
 }
 Copy-Item -LiteralPath $Source -Destination $Path
}
function Enter-GameOperation($Root) {
 $p=Join-Path $Root 'work\game-validation.operation-lock'
 New-Item -ItemType Directory -Path (Split-Path -Parent $p) -Force|Out-Null
 try{return [IO.File]::Open($p,[IO.FileMode]::OpenOrCreate,[IO.FileAccess]::ReadWrite,[IO.FileShare]::None)}catch{throw 'Another game-session operation is in progress'}
}
function Assert-GameSessionLock($Root,$Owner,$State) {
 $p=Join-Path $Root 'work\game-validation.lock'
 if(-not(Test-Path -LiteralPath $p)){throw 'Active session lost its shared lock; preserve files and reconcile ledger'}
 $lock=Get-Content -LiteralPath $p -Raw|ConvertFrom-Json
 if($lock.owner -ne $Owner -or $lock.sessionId -ne $State.sessionId -or $lock.session -ne $State.session){throw 'Another session owns the shared game lock'}
}
function Assert-SessionFiles($State,$Paths,[switch]$AllowGenerated) {
 foreach($entry in $State.files){
  if($entry.path -notin $Paths){throw 'Unexpected managed session path'}
  if(-not([IO.Path]::GetFullPath($entry.backup).StartsWith(($State.session+'\'),[StringComparison]::OrdinalIgnoreCase))){throw 'Backup outside recorded session'}
  if($entry.existed -and (Get-SessionHash $entry.backup) -ne $entry.sha256){throw ('Original backup missing/changed: '+$entry.path)}
  if((Get-SessionHash $entry.path) -ne $entry.expectedSha256){
   $fresh=$false
   if($entry.PSObject.Properties['preparedPreferences'] -and $entry.preparedPreferences -and (Get-SessionHash $entry.preparedPreferences) -eq $entry.expectedSha256 -and (Test-Path -LiteralPath $entry.path)){
    $expectedPrefs=Read-GamePreferenceValues ([IO.File]::ReadAllText($entry.preparedPreferences))
    $actualPrefs=Read-GamePreferenceValues ([IO.File]::ReadAllText($entry.path))
    # The game records the temporary window's position even with exit writes
    # disabled. No other changed preference is accepted as a session output.
    $differences=@(@($expectedPrefs.Keys+$actualPrefs.Keys)|Select-Object -Unique|Where-Object {$expectedPrefs[$_] -ne $actualPrefs[$_]})
    $fresh=@($differences|Where-Object {$_ -notin @('x_pos','y_pos')}).Count -eq 0
   }
   if($AllowGenerated -and $entry.generated -and $State.lastMode -in @('romance','historical','ai-only') -and $State.lastStartedUtc -and (Test-Path -LiteralPath $entry.path -PathType Leaf)){
    $fresh=(Get-Item -LiteralPath $entry.path).LastWriteTimeUtc -ge ([DateTimeOffset]$State.lastStartedUtc).UtcDateTime
   }
   if(-not $fresh){throw ('Managed file changed outside recorded operation; preserved: '+$entry.path)}
  }
 }
}
function Open-GameSession($Root,$Owner,$Ledger,$Base,$Paths,$GeneratedPaths=@()) {
 $lockPath=Join-Path $Root 'work\game-validation.lock';$old=$null
 if(Test-Path -LiteralPath $Ledger){
  $old=Get-Content -LiteralPath $Ledger -Raw|ConvertFrom-Json
  if(-not $old.restored){
   if($old.schema -ne 2 -or $old.owner -ne $Owner){throw 'Legacy active ledger requires explicit recovery; refusing old backups'}
   Assert-GameSessionLock $Root $Owner $old;Assert-SessionFiles $old $Paths -AllowGenerated;return $old
  }
  if(Test-Path -LiteralPath $lockPath){throw 'Shared lock still exists; reconcile completed session'}
  $retired=Join-Path $Base ('retired-ledger-'+[Guid]::NewGuid().ToString('N')+'.json');Move-Item -LiteralPath $Ledger -Destination $retired
 }
 if(Test-Path -LiteralPath $lockPath){throw 'Another validation session owns the game'}
 $id=[Guid]::NewGuid().ToString('N');$session=Join-Path $Base ('game-session-'+$id)
 New-Item -ItemType Directory -Path $session -Force|Out-Null
 $state=[pscustomobject]@{schema=2;owner=$Owner;sessionId=$id;session=$session;restored=$false;restoredUtc=$null;files=@();lastMode=$null;lastStartedUtc=$null;compilerSources=$null;actionSession=$null}
 $stream=[IO.File]::Open($lockPath,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
 try{$bytes=[Text.UTF8Encoding]::new($false).GetBytes(([pscustomobject]@{schema=2;owner=$Owner;sessionId=$id;session=$session;ledger=$Ledger;startedUtc=[DateTime]::UtcNow.ToString('o')}|ConvertTo-Json));$stream.Write($bytes,0,$bytes.Length)}finally{$stream.Dispose()}
 try{
  foreach($path in $Paths){
   $path=[IO.Path]::GetFullPath($path);$backup=Join-Path $session ($state.files.Count.ToString()+'.original');$existed=Test-Path -LiteralPath $path -PathType Leaf;$hash=Get-SessionHash $path
   if($existed){Copy-Item -LiteralPath $path -Destination $backup;if((Get-SessionHash $backup) -ne $hash){throw 'Snapshot hash mismatch'}}
   $state.files+=[pscustomobject]@{index=$state.files.Count;path=$path;backup=$backup;existed=$existed;sha256=$hash;expectedSha256=$hash;generated=($path -in $GeneratedPaths)}
  }
  Write-SessionJson $Ledger $state;return $state
 }catch{Move-Item -LiteralPath $lockPath -Destination (Join-Path $session 'failed-snapshot.lock');throw}
}
function Save-GameSession($Ledger,$State){Write-SessionJson $Ledger $State}
function Update-SessionFileHash($Ledger,$State,$Path){
 $entry=$State.files|Where-Object path -EQ $Path;if(-not $entry){throw 'File not managed by session'}
 $entry.expectedSha256=Get-SessionHash $Path;Save-GameSession $Ledger $State
}
function Set-GameSessionFile($Ledger,$State,$Path,[string]$Source,[AllowEmptyString()][string]$Text,[switch]$UseText){
 $entry=$State.files|Where-Object path -EQ $Path;if(-not $entry){throw 'File not managed by session'}
 if((Get-SessionHash $Path) -ne $entry.expectedSha256){
  if((Split-Path -Leaf $Path) -eq 'preferences.script.txt'){
   $preferenceState=[pscustomobject]@{files=@($entry);session=$State.session;lastMode=$State.lastMode;lastStartedUtc=$State.lastStartedUtc}
   Assert-SessionFiles $preferenceState @($Path)
   Update-SessionFileHash $Ledger $State $Path
  }else{throw ('Managed file changed; preserved: '+$Path)}
 }
 New-Item -ItemType Directory -Path (Split-Path -Parent $Path) -Force|Out-Null
 if($UseText){
  # WriteAllText rewrites in place: drop a hard link first so its source is untouched.
  if((Test-Path -LiteralPath $Path) -and (Get-Item -LiteralPath $Path).LinkType -eq 'HardLink'){Remove-Item -LiteralPath $Path -Force}
  [IO.File]::WriteAllText($Path,$Text,[Text.UTF8Encoding]::new($false))
 }else{Install-SessionFile $Source $Path}
 if($UseText -and (Split-Path -Leaf $Path) -eq 'preferences.script.txt'){
  $prepared=Join-Path $State.session ('prepared-preferences-'+[Guid]::NewGuid().ToString('N')+'.txt')
  Copy-Item -LiteralPath $Path -Destination $prepared
  $entry|Add-Member -NotePropertyName preparedPreferences -NotePropertyValue $prepared -Force
 }
 Update-SessionFileHash $Ledger $State $Path
}
function Read-GamePreferenceValues($Text){
 $values=@{};foreach($match in [regex]::Matches($Text,'(?m)^([a-zA-Z0-9_]+)\s+([^;]*);')){$values[$match.Groups[1].Value]=$match.Groups[2].Value.Trim()};return $values
}
function Clear-GameSessionFiles($Ledger,$State,$Paths,$Indices){
 Assert-SessionFiles $State $Paths -AllowGenerated
 foreach($entry in $State.files|Where-Object {$_.index -in $Indices}){
  if(Test-Path -LiteralPath $entry.path -PathType Leaf){Move-Item -LiteralPath $entry.path -Destination (Join-Path $State.session ($entry.index.ToString()+'.'+[Guid]::NewGuid().ToString('N')+'.compiled'))}
  $entry.expectedSha256=$null;Save-GameSession $Ledger $State
 }
}
function Restore-GameSession($Root,$Owner,$Ledger,$State,$Paths){
 if(-not $State -or $State.schema -ne 2 -or $State.restored){throw 'No active hash-checked session; legacy recovery must be explicit'}
 Assert-GameSessionLock $Root $Owner $State;Assert-SessionFiles $State $Paths -AllowGenerated
 foreach($entry in $State.files){
  if((Get-SessionHash $entry.path) -ne $entry.sha256){
   # Stage the verified original beside the target first; the live file is moved
   # aside only once a correct replacement is ready, so a failed copy never
   # leaves the game without the file.
   $staged=$null
   if($entry.existed){
    $staged=$entry.path+'.restore-'+[Guid]::NewGuid().ToString('N')
    Copy-Item -LiteralPath $entry.backup -Destination $staged
    if((Get-SessionHash $staged) -ne $entry.sha256){Remove-Item -LiteralPath $staged -Force;throw ('Backup hash mismatch for '+$entry.path)}
   }
   if(Test-Path -LiteralPath $entry.path -PathType Leaf){Move-Item -LiteralPath $entry.path -Destination (Join-Path $State.session ($entry.index.ToString()+'.'+[Guid]::NewGuid().ToString('N')+'.session-output'))}
   if($staged){Move-Item -LiteralPath $staged -Destination $entry.path}
  }
  if((Get-SessionHash $entry.path) -ne $entry.sha256){throw 'Restore hash mismatch'}
  $entry.expectedSha256=$entry.sha256;Save-GameSession $Ledger $State
 }
 $State.restored=$true;$State.restoredUtc=[DateTime]::UtcNow.ToString('o');Save-GameSession $Ledger $State
 Move-Item -LiteralPath (Join-Path $Root 'work\game-validation.lock') -Destination (Join-Path $State.session 'completed-validation.lock')
 Move-Item -LiteralPath $Ledger -Destination (Join-Path $State.session 'ledger-restored.json')
}
function Undo-FailedGamePreparation($Root,$Owner,$Ledger,$State,$Paths){
 if($State -and -not $State.restored){try{Restore-GameSession $Root $Owner $Ledger $State $Paths;Write-Warning 'Preparation failed; originals restored, lock released'}catch{Write-Warning ('Safe recovery blocked; files/lock preserved: '+$_.Exception.Message)}}
}
function Get-CompilerLaunchSources($Area){
 # One definition of the compile inputs (native_compilation_state.current_sources: map,
 # design, map_data, compiler pack and the game's own packs/executable); the capture
 # compares it with the same function, so the two can never disagree on the key set.
 . (Join-Path $PSScriptRoot 'tw3k_env.ps1')
 $preference=$ErrorActionPreference;$ErrorActionPreference='Continue'
 $out=& (Get-Tw3kPython) -B -c "import sys,json;sys.path.insert(0,sys.argv[1]);from pathlib import Path;import native_compilation_state as n;print(json.dumps(n.current_sources(Path(sys.argv[2]))))" $PSScriptRoot $Area 2>&1
 $code=$LASTEXITCODE;$ErrorActionPreference=$preference
 if($code -ne 0){throw ('Compiler source changed: '+(($out|ForEach-Object{"$_"}) -join ' '))}
 return (@($out|Where-Object{$_ -is [string]})[-1]|ConvertFrom-Json)
}
