"""Machine-specific paths the build reads, in one place (open-source checklist C1).

Each path comes from, in order: its environment variable, tw3k.local.json (in the repository root, or in the main
checkout when this is a git worktree), or its default. require(key) raises for a path with no value, naming the
variable and the config key; the module attributes (GAME_DATA, ...) give UNSET-<VARIABLE> instead, so a script
that never opens the game still imports. Copy tw3k.local.example.json to tw3k.local.json and fill it in (BUILD.md).

  game_dir     TW3K_GAME_DIR     Total War: THREE KINGDOMS install (holds data/ and assembly_kit/); no default
  rpfm_cli     TW3K_RPFM         rpfm_cli.exe (RPFM 4.2.3); no default
  rpfm_schema  TW3K_RPFM_SCHEMA  RPFM's schema_3k.ron; default under %APPDATA%\\FrodoWazEre\\rpfm
  ca_appdata   TW3K_CA_APPDATA   the game's %APPDATA%\\The Creative Assembly\\ThreeKingdoms folder
  main_root    TW3K_MAIN_ROOT    the main checkout (shared locks, caches, untracked inputs); default: this
                                 checkout, or for a git worktree the checkout that owns it
  scratch      TW3K_SCRATCH      scratch folder for copies and caches; default <main_root>-scratch beside it
  fonts        TW3K_FONTS        TrueType fonts for preview labels; default %WINDIR%\\Fonts

Derived: GAME_DATA = game_dir/data, ASSEMBLY_KIT = game_dir/assembly_kit, LOCKS = main_root/work.
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = 'tw3k.local.json'
KEYS = {
    'game_dir': 'TW3K_GAME_DIR', 'rpfm_cli': 'TW3K_RPFM', 'rpfm_schema': 'TW3K_RPFM_SCHEMA',
    'ca_appdata': 'TW3K_CA_APPDATA', 'main_root': 'TW3K_MAIN_ROOT', 'scratch': 'TW3K_SCRATCH', 'fonts': 'TW3K_FONTS',
}


def main_checkout(root=ROOT):
    """root, or for a git worktree (root/.git is a 'gitdir: <main>/.git/worktrees/<name>' file) the main checkout."""
    g = Path(root) / '.git'
    if g.is_file():
        line = g.read_text(encoding='utf-8').strip()
        if line.startswith('gitdir:'):
            d = Path(line[7:].strip())
            d = d if d.is_absolute() else (Path(root) / d).resolve()
            if d.parent.name == 'worktrees' and d.parent.parent.name == '.git':
                return d.parent.parent.parent
    return Path(root)


def load_config(root=ROOT):
    """tw3k.local.json from root, else from the main checkout, else from %APPDATA%\\tw3k (the one a copy of the
    scripts outside any checkout still finds, so its locks stay the main checkout's); {} when none has one."""
    for base in (Path(root), main_checkout(root), _appdata('tw3k')):
        p = base / CONFIG if base else None
        if p and p.is_file():
            return json.loads(p.read_text(encoding='utf-8'))
    return {}


def _appdata(*parts):
    a = os.environ.get('APPDATA')
    return Path(a, *parts) if a else None


def resolve(key, root=ROOT, env=None, config=None):
    """One configured path: environment variable, then tw3k.local.json, then the default (None if there is none)."""
    env = os.environ if env is None else env
    config = load_config(root) if config is None else config
    if env.get(KEYS[key]):
        return Path(env[KEYS[key]])
    if config.get(key):
        return Path(config[key])
    if key == 'main_root':
        return main_checkout(root)
    if key == 'scratch':
        m = resolve('main_root', root, env, config)
        return m.with_name(m.name + '-scratch')
    if key == 'rpfm_schema':
        return _appdata('FrodoWazEre', 'rpfm', 'config', 'schemas', 'schema_3k.ron')
    if key == 'ca_appdata':
        return _appdata('The Creative Assembly', 'ThreeKingdoms')
    if key == 'fonts':
        w = env.get('WINDIR') or env.get('SystemRoot')
        return Path(w, 'Fonts') if w else None
    return None


def require(key, root=ROOT):
    p = resolve(key, root)
    if p is None:
        raise RuntimeError(f'{key} is not configured: set {KEYS[key]} or "{key}" in {CONFIG} '
                           f'(copy tw3k.local.example.json; see BUILD.md)')
    return p


def portable(path):
    """A path as recorded in tracked files: relative to the main checkout ('output/x.pack'), 'scratch:<rel>' under
    the scratch folder, else the file name only. Never a drive letter or a user folder (the public repository
    carries no machine path)."""
    p = Path(path).resolve()
    for prefix, base in (('', resolve('main_root')), ('scratch:', resolve('scratch'))):
        try:
            return prefix + p.relative_to(Path(base).resolve()).as_posix()
        except (ValueError, TypeError):
            continue
    return p.name


def font(name):
    """A TrueType font file from the fonts folder (preview labels only)."""
    return str(require('fonts') / name)


def _or_unset(key):
    """The configured path, or a path named after the missing variable: scripts read these at import time, and a
    tool that never opens the game must still import; one that does fails on a path that names what to set."""
    p = resolve(key)
    return p if p is not None else Path(f'UNSET-{KEYS[key]}')


_DERIVED = {
    'GAME_DIR': lambda: _or_unset('game_dir'),
    'GAME_DATA': lambda: _or_unset('game_dir') / 'data',
    'ASSEMBLY_KIT': lambda: _or_unset('game_dir') / 'assembly_kit',
    'RPFM': lambda: _or_unset('rpfm_cli'),
    'RPFM_SCHEMA': lambda: _or_unset('rpfm_schema'),
    'CA_APPDATA': lambda: _or_unset('ca_appdata'),
    'MAIN_ROOT': lambda: require('main_root'),
    'SCRATCH': lambda: require('scratch'),
    'LOCKS': lambda: require('main_root') / 'work',
}


def __getattr__(name):
    """Module attributes, resolved when read (environment and tw3k.local.json at that moment)."""
    if name in _DERIVED:
        return _DERIVED[name]()
    raise AttributeError(name)
