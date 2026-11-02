# Interpreter resolution for the pipeline. Dot-source this file, then call
# Get-Tw3kPython (numpy + Pillow, terrain scripts) or Get-Tw3kBasePython
# (SDK/database scripts). Set TW3K_PYTHON / TW3K_BASE_PYTHON to override.
# Get-Tw3kDefaultPython is the single tw3k environment (conda-forge Python 3.14,
# newest numpy/Pillow/scipy/h5py, called by path without activation) and the
# default for both (terrain outputs verified byte-identical to the old Codex Python).

# The path lives in env/python-path.txt, shared with build_combined_mod.py.
function Get-Tw3kDefaultPython { (Get-Content -LiteralPath (Join-Path $PSScriptRoot '..\env\python-path.txt') -TotalCount 1).Trim() }

function Test-Tw3kPython([string]$Path, [string]$Imports) {
    if (-not $Path -or -not (Test-Path -LiteralPath $Path)) { return $false }
    # Under $ErrorActionPreference='Stop' (PS 5.1) redirected native stderr is a
    # terminating error; probe with a local 'Continue' so a failed import just
    # moves on to the next candidate.
    $ErrorActionPreference = 'Continue'
    & $Path -c "import $Imports" 2>$null | Out-Null
    return ($LASTEXITCODE -eq 0)
}

function Get-Tw3kPython {
    $candidates = @($env:TW3K_PYTHON, (Get-Tw3kDefaultPython))
    foreach ($candidate in $candidates) {
        if (Test-Tw3kPython $candidate 'numpy,PIL') { return $candidate }
    }
    throw "No Python with numpy and Pillow found; create $(Get-Tw3kDefaultPython) or set TW3K_PYTHON"
}

# Machine paths: the keys, order and defaults of scripts/tw3k_paths.py (environment variable, then
# tw3k.local.json in this checkout / the main checkout / %APPDATA%\tw3k, then the default).
$Tw3kPathVars = @{game_dir='TW3K_GAME_DIR'; rpfm_cli='TW3K_RPFM'; rpfm_schema='TW3K_RPFM_SCHEMA'; ca_appdata='TW3K_CA_APPDATA'; main_root='TW3K_MAIN_ROOT'; scratch='TW3K_SCRATCH'; fonts='TW3K_FONTS'}
function Get-Tw3kMainCheckout([string]$Root) {
    $g = Join-Path $Root '.git'
    if (Test-Path -LiteralPath $g -PathType Leaf) {
        $line = (Get-Content -LiteralPath $g -TotalCount 1).Trim()
        if ($line -like 'gitdir:*') {
            $d = $line.Substring(7).Trim(); if (-not [IO.Path]::IsPathRooted($d)) { $d = Join-Path $Root $d }
            $p = Split-Path -Parent ([IO.Path]::GetFullPath($d))
            if ((Split-Path -Leaf $p) -eq 'worktrees' -and (Split-Path -Leaf (Split-Path -Parent $p)) -eq '.git') { return Split-Path -Parent (Split-Path -Parent $p) }
        }
    }
    return $Root
}
function Get-Tw3kPath([string]$Key) {
    $v = [Environment]::GetEnvironmentVariable($Tw3kPathVars[$Key]); if ($v) { return [IO.Path]::GetFullPath($v) }
    $root = Split-Path -Parent $PSScriptRoot
    foreach ($base in @($root, (Get-Tw3kMainCheckout $root), $(if ($env:APPDATA) { Join-Path $env:APPDATA 'tw3k' }))) {
        if (-not $base) { continue }
        $f = Join-Path $base 'tw3k.local.json'
        if (Test-Path -LiteralPath $f) { $c = Get-Content -LiteralPath $f -Raw | ConvertFrom-Json; if ($c.$Key) { return [IO.Path]::GetFullPath($c.$Key) }; break }
    }
    switch ($Key) {
        'main_root' { return [IO.Path]::GetFullPath((Get-Tw3kMainCheckout $root)) }
        'scratch' { $m = Get-Tw3kPath 'main_root'; return Join-Path (Split-Path -Parent $m) ((Split-Path -Leaf $m) + '-scratch') }
        'rpfm_schema' { if ($env:APPDATA) { return Join-Path $env:APPDATA 'FrodoWazEre\rpfm\config\schemas\schema_3k.ron' } }
        'ca_appdata' { if ($env:APPDATA) { return Join-Path $env:APPDATA 'The Creative Assembly\ThreeKingdoms' } }
        'fonts' { if ($env:WINDIR) { return Join-Path $env:WINDIR 'Fonts' } }
    }
    throw "$Key is not configured: set $($Tw3kPathVars[$Key]) or `"$Key`" in tw3k.local.json (see BUILD.md)"
}

# Untracked inputs a git worktree (e.g. the integrator worktree) lacks: junctions to the main checkout. Packaging needs
# the SDK binaries (native_db_export's DLL directory) and the battle-scenes pack; terrain steps read tools,
# reference and the geodata cache. A cleanup took them for build leftovers once (18th candidate, 2026-10-26),
# so the packaging and pipeline entry points create or verify them here instead of by hand. No-op in the main
# checkout; a real (non-link) directory is left alone.
$Tw3kMainRoot = Get-Tw3kPath 'main_root'
$Tw3kWorktreeLinks = @('tools', 'reference', 'work\geodata', 'work\standalone_kit\binaries', 'mods\terrain_scenes\output')
function Initialize-Tw3kWorktreeLinks([string]$Root) {
    $Root = (Resolve-Path -LiteralPath $Root).Path.TrimEnd('\')
    if ($Root -ieq $Tw3kMainRoot) { return }
    foreach ($rel in $Tw3kWorktreeLinks) {
        $link = Join-Path $Root $rel; $target = Join-Path $Tw3kMainRoot $rel
        if (-not (Test-Path -LiteralPath $target)) { throw "Worktree link target missing in the main checkout: $target" }
        if (Test-Path -LiteralPath $link) {
            $item = Get-Item -LiteralPath $link -Force
            if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint -and $item.Target -and ($item.Target | Select-Object -First 1) -ine $target) {
                throw "Worktree link $link points to $($item.Target), expected $target"
            }
            continue
        }
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $link) | Out-Null
        New-Item -ItemType Junction -Path $link -Target $target | Out-Null
        Write-Host "Worktree link created: $link -> $target"
    }
}

function Get-Tw3kBasePython {
    $candidates = @($env:TW3K_BASE_PYTHON, (Get-Tw3kDefaultPython), $env:TW3K_PYTHON)
    foreach ($candidate in $candidates) {
        if (Test-Tw3kPython $candidate 'hashlib,json') {
            $ErrorActionPreference = 'Continue'
            & $candidate -c "import hashlib,sys;sys.exit(0 if hasattr(hashlib,'file_digest') else 1)" 2>$null
            if ($LASTEXITCODE -eq 0) { return $candidate }
        }
    }
    throw "No Python 3.11+ found for SDK/database scripts; create $(Get-Tw3kDefaultPython) or set TW3K_BASE_PYTHON"
}
