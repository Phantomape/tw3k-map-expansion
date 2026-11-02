# Recreate the single tw3k Python environment exactly (conda-forge base + pinned pip layer).
# Always call the result by path (<Prefix>\python.exe, recorded in env/python-path.txt); no 'conda activate' needed.
# Example: .\env\create_tw3k_env.ps1 -Prefix <conda root>\envs\tw3k -Conda <conda root>\Scripts\conda.exe
param([Parameter(Mandatory)][string]$Prefix, [Parameter(Mandatory)][string]$Conda)
$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
if (Test-Path -LiteralPath $Prefix) { throw "$Prefix exists; remove it first or pass -Prefix" }
# Exact conda layer (explicit URLs + md5). The command that originally made it was:
#   conda create -p <Prefix> --override-channels -c conda-forge python=3.14 -y
& $Conda create -p $Prefix --file (Join-Path $here 'tw3k-conda-explicit-win64.txt') -y
if ($LASTEXITCODE -ne 0) { throw 'conda create failed' }
$py = Join-Path $Prefix 'python.exe'
& $py -m pip install --no-deps --only-binary=:all: -r (Join-Path $here 'tw3k-requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'pip install failed' }
& $py -m pip check
if ($LASTEXITCODE -ne 0) { throw 'pip check failed' }
# Smoke test: the imports every tw3k task needs, plus the Get-Tw3kBasePython requirement.
& $py -I -c "import hashlib,zlib,sys,numpy,PIL,scipy,h5py,lz4,requests,bs4;assert hasattr(hashlib,'file_digest');print(sys.version.split()[0],'zlib',zlib.ZLIB_VERSION,'numpy',numpy.__version__,'Pillow',PIL.__version__,'scipy',scipy.__version__,'h5py',h5py.__version__)"
if ($LASTEXITCODE -ne 0) { throw 'smoke test failed' }
