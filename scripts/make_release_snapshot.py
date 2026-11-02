"""Public source snapshot: the files of one commit that the build needs, copied into a new directory (the seed of
the fresh public repository) and checked before anything is written.

  python scripts/make_release_snapshot.py OUT [--rev REV] [--dry-run] [--allow-machine-paths]
                                              [--init-repo [--message MSG]]

--init-repo makes OUT a git repository with one commit whose author and committer are RELEASE_NAME and
RELEASE_EMAIL, with no system or user git configuration read (RELEASE-MANIFEST.json is left out of the commit).

Only files tracked at REV (default HEAD) are read, through git, so a dirty working tree never leaks into the
snapshot. The private repository stays as it is; nothing is deleted from it. What is taken:

  * the import closure of the build entry points (ROOTS): every scripts/ file a taken Python file imports or names
    ('x.py', 'x.ps1', 'x.json', Tw3kX.exe -> Tw3kX.cs), every one a taken PowerShell file names, transitively;
  * the tests whose scripts/ imports all lie in that closure;
  * the fixed non-script files in INCLUDE (licence, notices, docs, environment, layout data, design).
Everything else is left out; RELEASE-MANIFEST.json counts it per reason. Checks, each one failing the run:

  required   LICENSE, THIRD_PARTY_NOTICES.md, README.md, BUILD.md and the generated PUBLIC_SOURCE marker are present
  closure    the closure reaches no module of another workstream (FORBIDDEN: people, faction, unit, lore mods);
             such a dependency of the map build is a real problem to report, never a file to take
  binary     no file with a NUL byte or a denied extension (packs, executables, archives, images, game formats),
             except BINARY_OK (the project's own layout arrays)
  canvas     the design's padding places vanilla at 888, 375 and the authored coordinates carry the design's
             grid and padding (what campaign_world.require_canvas checks when the build reads them)
  layout     the layout diff (data/map-layout-diff) holds no vanilla content: layout_diff.layout_problems of
             the snapshot's own scripts against the vanilla map (every stored native byte differs from vanilla,
             nothing outside the mask); vanilla comes from the user's own copy at build time
  size       no file over MAX_BYTES
  imported   no key or id of the earlier reference mod anywhere (IRONIC, no exception: the snapshot carries none of
             its data); its name only in NAME_OK (statements that it is not used)
  secrets    no token or key (GitHub, AWS, Steam Web API, private key headers), no password/token assignment, no
             e-mail address outside EMAIL_OK, no user-profile path segment (<drive>\\Users\\<name>, /home/<name>);
             no exception, binary layout files included
  source     none of the project's old layouts (PRIVATE_LAYOUT: the layout cleanup data and its sources; user
             decision 2026-11-02, option c, source compliance): polish skips the cleanup without them
  machine    no absolute machine path (a drive letter and a path) in any file: MACHINE_OK is empty; tracked files
             record paths through tw3k_paths.portable;
             --allow-machine-paths lists them without failing
"""
import argparse
import fnmatch
import hashlib
import io
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Build entry points (BUILD.md): the steps of a build from vanilla with every switch at its default, its gates
# and audits, and the release tools.
ROOTS = [
    'scripts/fetch_geodata.py', 'scripts/strategic_ink.py', 'scripts/build_west_real_terrain.py',
    'scripts/extract_vanilla_reference.py', 'scripts/bake_map.py', 'scripts/hehuang_patch.py', 'scripts/korea_patch.py', 'scripts/wuyuan_patch.py',
    'scripts/all_directions_build.py',
    'scripts/build_polished_campaign_terrain.ps1', 'scripts/rebuild_section14_candidate.ps1',
    'scripts/combined_mod_session.ps1', 'scripts/combined_action.ps1', 'scripts/game_session.ps1',
    'scripts/build_combined_mod.py', 'scripts/export_build_tables.py', 'scripts/art_review.py',
    'scripts/stage_invariants.py', 'scripts/audit_imported_map_residuals.py', 'scripts/make_release_snapshot.py',
]
INCLUDE = [
    'LICENSE', 'THIRD_PARTY_NOTICES.md', 'BUILD.md', '.gitattributes', 'tw3k.local.example.json',
    'env/create_tw3k_env.ps1', 'env/tw3k-requirements.txt', 'env/tw3k-conda-explicit-win64.txt',
    # data/oss-rebuild: filler-open-known.json is read by a stage invariant; params.json records the parameters the
    # rebuilt regions were generated with (provenance; the generator is not published in this release)
    'data/map-layout-diff/*', 'data/lake-level-spec.json', 'data/geodata-sources.json', 'data/oss-rebuild/*.json',
    # tracked analysis/ files the build and its checks read (art_review cameras, frozen biome profiles, patchiness
    # and fragmentation baselines, road plans, town prop positions); the rest of analysis/ is private history
    'analysis/art-review-cameras.json', 'analysis/biome-profiles-frozen*.json', 'analysis/biome-profiles*.json',
    'analysis/ground-patchiness-baseline.json', 'analysis/ground-patchiness-vanilla.json',
    'analysis/ground-fragmentation-baseline*.json', 'analysis/roads/*.json', 'analysis/saibei-town-prop-positions.json',
    'work/all-directions-30/work/current-region-design.json',
    'work/all-directions-30/work/canvas-authored-coordinates.json',
    'work/all-directions-30/work/region-traits.json',        # campaign_world.region_traits: beside the design
]
# Published under another path: the public README replaces the private repository's working README.
PUBLISH = {'release/README.md': 'README.md'}
NEVER = ['scripts/archive/*']                  # history of the private repository: never followed
# The project's old layouts (pre-layout-v2 map, coast-revision-4 change mask) and the layout cleanup data made from
# them: never published (user 2026-11-02, option c, source compliance; polish's layout_cleanup.mask runs without
# them). The generator is named in comments, so it is also never followed.
PRIVATE_LAYOUT = ['data/map-layout/layout-cleanup.npz', 'scripts/extract_layout_cleanup.py', '*before-layout-v2*',
                  '*layout-v4*', '*changed-coast*', '*changed_coast*.np*']
NEVER += ['scripts/extract_layout_cleanup.py']
# Modules of other workstreams (people, faction, unit, lore mods): the map build must not depend on them.
FORBIDDEN = [
    'scripts/*faction*', 'scripts/*depth*', 'scripts/*people*', 'scripts/*units_*', 'scripts/*vendor_*',
    'scripts/*unit_workspace*', 'scripts/*character_assets*', 'scripts/*oasis_icons*', 'scripts/*completion*',
    'scripts/*lore_v04*', 'scripts/*oasis_buildings*', 'scripts/research_firearm_fx.py',
    'scripts/rename_jinling_yellow_turban.py', 'scripts/verify_gameplay_v05.py',
    'scripts/*integrated*', 'scripts/*balance_r4*', 'scripts/*_lore_*', 'scripts/assess_oasis_town_3d.py',
    'scripts/audit_external_compatibility.py',
]
# Published with a part left out: the review-camera file's 'candidates' (the shots its private generator picks from;
# art_review reads only 'fixed' and 'changes').
def _drop_candidates(data):
    d = json.loads(data.decode('utf-8'))
    d.pop('candidates', None)
    return (json.dumps(d, ensure_ascii=False, indent=1) + '\n').encode('utf-8')


TRANSFORM = {'analysis/art-review-cameras.json': _drop_candidates}
BINARY_OK = ['data/map-layout-diff/*.npz']     # the layout diff (numpy), not game or tool binaries
DENY_EXT = {'.pack', '.exe', '.dll', '.pdb', '.lib', '.so', '.zip', '.7z', '.rar', '.gz', '.nc', '.tif', '.tiff',
            '.png', '.jpg', '.jpeg', '.bmp', '.dds', '.tga', '.npy', '.bin', '.esf', '.hex', '.ppd', '.pbd', '.ptd',
            '.ccd', '.rigid_model_v2', '.wsmodel', '.variantmeshdefinition'}
MAX_BYTES = 2 << 20
IRONIC = re.compile('iron' + 'ic_|' + '32401' + '50507', re.I)   # the mod's key prefix and workshop id (split: not in this file)
NAME = re.compile(r'190[ _-]?expanded', re.I)
NAME_OK = {
    'THIRD_PARTY_NOTICES.md': 'disclosure sentence chosen by the user',
    'scripts/campaign_world.py': 'comment: why the vanilla map comes from the CAIME template',
    'scripts/repair_character_placeholder_positions.py': 'comment on the shipped assembly kit',
    'scripts/audit_imported_map_residuals.py': 'the audit that measures imported residue (reference map required)',
    'scripts/make_release_snapshot.py': 'this check',
    'tests/test_sdk_empty_tables.py': 'decoy path: proves the SDK patch never reads it',
    'tests/test_release_snapshot.py': 'tests of this check',
}
MACHINE = re.compile(r'(?<![A-Za-z0-9\\])[A-Za-z]:[\\/]{1,2}(?=[A-Za-z0-9_.$%{(-])')
# Files whose absolute paths are examples or provenance records, with the reason.
MACHINE_OK = {}   # none: no file of the public repository may hold a machine path (coordinator 2026-11-01)
# Secrets and personal data: no exception except the authorship addresses in EMAIL_OK (filled in when the
# user has chosen the commit author address).
SECRETS = [
    ('GitHub token', re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{22,})')),
    ('AWS access key', re.compile(r'\b(?:AKIA|ASIA)[0-9A-Z]{16}\b')),
    ('private key', re.compile(r'-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----')),
    ('Steam Web API key', re.compile(r'(?i)(?:steam|web_?api|api[_-]?key)\W{0,20}\b[0-9A-F]{32}\b')),
    ('password/token assignment', re.compile(
        r'''(?i)\b(?:password|passwd|pwd|secret|token|api[_-]?key|access[_-]?key)\b["']?\s*[:=]\s*["'][^"'\s]{6,}["']''')),
    ('e-mail address', re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b')),
    ('user profile path', re.compile(r'(?i)(?:\b[A-Z]:)?[\\/]{1,2}(?:Users|Documents and Settings|home)[\\/]{1,2}'
                                     r'(?!Public\b|Default\b|<|%|\$|\{)[^\\/\s"\'<>:|*?]+')),
]
# The public repository's author and committer (user decision 2026-11-01: GitHub's no-reply address), fixed
# here and never read from the local git configuration.
RELEASE_NAME = 'Phantomape'
RELEASE_EMAIL = '8809221+Phantomape@users.noreply.github.com'
EMAIL_OK = {RELEASE_EMAIL.lower()}
# The public source marker (generated here, never tracked): layout_cleanup.mask skips the cleanup only when it
# exists, so a private build missing its npz still fails (integrator review of #49).
PUBLIC_SOURCE = 'PUBLIC_SOURCE'
PUBLIC_SOURCE_TEXT = ('This tree is the public source release of TW3K Map Expansion, made by '
                      'scripts/make_release_snapshot.py.\nData left out for source compliance is skipped by the build '
                      'with a WARN (see BUILD.md).\n')
REQUIRED = ['LICENSE', 'THIRD_PARTY_NOTICES.md', 'README.md', 'BUILD.md', 'work/all-directions-30/work/region-traits.json',
            PUBLIC_SOURCE]
# Every script BUILD.md tells the reader to run must be published (build_west_real_terrain was named only there).
BUILD_NAMED = re.compile(r'scripts[\\/]([A-Za-z0-9_]+\.(?:py|ps1))')
GITIGNORE = """# Local inputs and build products (bring your own game, SDK, tools and data; see BUILD.md)
/work/*
!/work/all-directions-30/
/work/all-directions-30/*
!/work/all-directions-30/work/
/work/all-directions-30/work/*
!/work/all-directions-30/work/current-region-design.json
!/work/all-directions-30/work/canvas-authored-coordinates.json
!/work/all-directions-30/work/region-traits.json
/output/
/tools/
/tw3k.local.json
/reference/
*.pack
__pycache__/
"""


def match(path, patterns):
    return any(fnmatch.fnmatchcase(path, p) for p in patterns)


def tracked(rev, repo=ROOT):
    """[(path, blob sha)] of regular files tracked at rev (symlinks and submodules left out)."""
    out = subprocess.run(['git', '-C', str(repo), 'ls-tree', '-r', '-z', '--full-tree', rev],
                         capture_output=True, check=True).stdout.decode('utf-8')
    files = []
    for entry in filter(None, out.split('\0')):
        meta, path = entry.split('\t', 1)
        mode, kind, sha = meta.split()
        if kind == 'blob' and mode in ('100644', '100755'):
            files.append((path, sha))
    return files


def blobs(shas, repo=ROOT):
    """Contents of the given blobs, through one git cat-file --batch (communicate: a pipe written while unread
    deadlocks on Windows once the 4 KB buffer fills)."""
    out = subprocess.run(['git', '-C', str(repo), 'cat-file', '--batch'], input=''.join(s + '\n' for s in shas).encode(),
                         capture_output=True, check=True).stdout
    data, at = {}, 0
    for s in shas:
        nl = out.index(b'\n', at)
        size = int(out[at:nl].split()[2])
        data[s] = out[nl + 1:nl + 1 + size]
        at = nl + 2 + size
    return data


IMPORT = re.compile(r'(?:^|;)[ \t]*(?:from[ \t]+([A-Za-z_]\w*)[ \t]+import|import[ \t]+([\w \t,.]+))', re.M)
NAMED = re.compile(r"['\"/\\]([\w.-]+\.(?:py|ps1|json))['\"]")
TOOL_EXE = re.compile(r'\b(Tw3k\w+)\.exe\b')


def deps(path, text):
    """scripts/ paths a file imports or names (names only; the caller keeps those tracked under scripts/)."""
    out = set() if path == 'scripts/make_release_snapshot.py' else (          # its own lists name files it judges
        {f'scripts/{n}' for n in NAMED.findall(text)} | {f'scripts/{n}.cs' for n in TOOL_EXE.findall(text)})
    if path.endswith('.py'):
        for frm, names in IMPORT.findall(text):
            for n in [frm] if frm else names.split(','):
                n = n.split()[0].split('.')[0] if n.split() else ''
                if n:
                    out.add(f'scripts/{n}.py')
    return out if path.endswith(('.py', '.ps1')) else set()


def closure(texts, roots=None):
    """(taken scripts {path: reached from}, forbidden reached [(path, from)]) from the entry points (ROOTS).
    texts: {tracked path: text}."""
    seen, bad, todo = {}, [], [(r, 'root') for r in (ROOTS if roots is None else roots) if r in texts]
    while todo:
        p, via = todo.pop()
        if p in seen or match(p, NEVER):
            continue
        if match(p, FORBIDDEN):
            bad.append((p, via))
            continue
        seen[p] = via
        todo += [(d, p) for d in sorted(deps(p, texts[p])) if d in texts and d != p]
    return seen, bad


def select(texts):
    """{path: reason} for every tracked path: 'take ...' or the reason it is left out; and the forbidden hits."""
    scripts, bad = closure(texts)
    why = {}
    for p in texts:
        if p in scripts:
            why[p] = 'take closure'
        elif p in PUBLISH:
            why[p] = 'take publish as ' + PUBLISH[p]
        elif match(p, INCLUDE):
            why[p] = 'take include'
        elif p.startswith('tests/') and p.endswith('.py'):
            d = {x for x in deps(p, texts[p]) if x in texts and x.startswith('scripts/')}
            why[p] = 'take test' if d and d <= set(scripts) else 'test of a module outside the closure'
        elif p.startswith('scripts/'):
            why[p] = 'archive' if match(p, NEVER) else 'not in the build closure'
        else:
            why[p] = 'not in whitelist'
    return why, bad


DESIGN = 'work/all-directions-30/work/current-region-design.json'
AUTHORED = 'work/all-directions-30/work/canvas-authored-coordinates.json'
RELEASE_OFFSET = [888, 375]                    # the released canvas: vanilla at this offset (coordinator 2026-11-01)


def canvas_problems(files):
    """The design's canvas is the released one and the authored coordinates were made on it: the condition
    campaign_world.require_canvas enforces when the build reads them (grid and padding equal)."""
    import json as _json
    if DESIGN not in files or AUTHORED not in files:
        return [f'missing {x}' for x in (DESIGN, AUTHORED) if x not in files]
    try:
        d = _json.loads(files[DESIGN])
        a = _json.loads(files[AUTHORED]).get('canvas', {})
    except ValueError as e:
        return [f'unreadable design or authored coordinates ({e})']
    out = []
    if list(d.get('padding', [])[:2]) != RELEASE_OFFSET:
        out.append(f'{DESIGN}: padding {d.get("padding")} does not place vanilla at {RELEASE_OFFSET}')
    if (a.get('grid'), a.get('padding')) != (d.get('grid'), d.get('padding')):
        out.append(f'{AUTHORED}: canvas {a} differs from the design (grid {d.get("grid")}, padding {d.get("padding")}); '
                   'campaign_world.require_canvas would stop the build')
    return out


DIFF_DIR = 'data/map-layout-diff'


def vanilla_map():
    """The user's vanilla logical map (CAIME's template) in the main checkout."""
    import tw3k_paths
    return tw3k_paths.MAIN_ROOT / 'tools/caime-1.1.0/lib/app/Templates/3k_dlc07_main_map/map.hex'


def layout_problems(files, vanilla=None):
    """No vanilla content in the released layout diff: layout_diff.layout_problems (the snapshot's own version) on
    the snapshot's data/map-layout-diff against the vanilla map. Missing data, module or vanilla map is a problem."""
    import importlib
    import tempfile
    diff = {p: d for p, d in files.items() if p.startswith(DIFF_DIR + '/')}
    if not diff:
        return [f'no {DIFF_DIR}/ in the snapshot (the map cannot be built from the source)']
    if 'scripts/layout_diff.py' not in files:
        return ['scripts/layout_diff.py is not in the snapshot']
    vanilla = Path(vanilla or vanilla_map())
    if not vanilla.is_file():
        return [f'vanilla map not found at {vanilla} (needed to check the diff)']
    with tempfile.TemporaryDirectory(prefix='tw3k-snapshot-layout-') as t:
        for path, data in files.items():
            if path.startswith(('scripts/', DIFF_DIR + '/')):
                q = Path(t) / path
                q.parent.mkdir(parents=True, exist_ok=True)
                q.write_bytes(data)
        saved = {k: sys.modules.pop(k) for k in list(sys.modules) if k in ('layout_diff', 'map_hex')}
        sys.path.insert(0, str(Path(t) / 'scripts'))
        try:
            mod = importlib.import_module('layout_diff')
            return [f'{DIFF_DIR}/{x}' for x in mod.layout_problems(Path(t) / DIFF_DIR, vanilla)]
        except Exception as e:                              # a broken diff or module fails the check, not the run
            return [f'{DIFF_DIR}: layout check could not run ({type(e).__name__}: {e})']
        finally:
            sys.path.remove(str(Path(t) / 'scripts'))
            for k in ('layout_diff', 'map_hex'):
                sys.modules.pop(k, None)
            sys.modules.update(saved)


def check(files, bad=(), vanilla=None):
    """files: {path: bytes}; bad: forbidden modules the closure reached. Returns {check: [problems]}."""
    res = {k: [] for k in ('required', 'closure', 'binary', 'canvas', 'layout', 'size', 'imported', 'source', 'secrets',
                           'machine')}
    res['source'] = sorted(p for p in files if match(p, PRIVATE_LAYOUT))
    res['required'] = [f'missing {r}' for r in REQUIRED if r not in files]
    build = files.get('BUILD.md', b'').decode('utf-8', errors='replace')
    res['required'] += sorted({f'BUILD.md names scripts/{n}, not published' for n in BUILD_NAMED.findall(build)
                               if f'scripts/{n}' not in files})
    res['closure'] = [f'{via} -> {p}' for p, via in bad]
    res['canvas'] = canvas_problems(files)
    res['layout'] = layout_problems(files, vanilla)
    for path, data in sorted(files.items()):
        ext = Path(path).suffix.lower()
        if not match(path, BINARY_OK) and (ext in DENY_EXT or b'\0' in data[:1 << 16]):
            res['binary'].append(path)
        if len(data) > MAX_BYTES:
            res['size'].append(f'{path} {len(data)} bytes')
        text = data.decode('latin-1') if match(path, BINARY_OK) else data.decode('utf-8', errors='replace')
        m = IRONIC.search(text)
        if m:
            res['imported'].append(f'{path}:{text.count(chr(10), 0, m.start()) + 1} key or id')
        m = NAME.search(text)
        if m and not match(path, NAME_OK):
            res['imported'].append(f'{path}:{text.count(chr(10), 0, m.start()) + 1} name')
        for what, pattern in SECRETS:
            for m in pattern.finditer(text):
                if what == 'e-mail address' and m.group(0).lower() in EMAIL_OK:
                    continue
                res['secrets'].append(f'{path}:{text.count(chr(10), 0, m.start()) + 1} {what}')
        if match(path, BINARY_OK):
            continue
        for m in (() if match(path, MACHINE_OK) else MACHINE.finditer(text)):
            res['machine'].append(f'{path}:{text.count(chr(10), 0, m.start()) + 1}')
    return res


def snapshot(rev='HEAD', repo=ROOT):
    """(files {path: bytes}, left out {reason: count}, commit sha, forbidden hits, per-path reasons)."""
    commit = subprocess.run(['git', '-C', str(repo), 'rev-parse', rev + '^{commit}'], capture_output=True,
                            text=True, check=True).stdout.strip()
    all_ = tracked(commit, repo)
    code = [(p, s) for p, s in all_ if p.startswith(('scripts/', 'tests/'))]
    data = blobs([s for _, s in code], repo)
    texts = {p: data[s].decode('utf-8', errors='replace') for p, s in code}
    texts.update({p: '' for p, _ in all_ if p not in texts})
    why, bad = select(texts)
    keep = [(p, s) for p, s in all_ if why[p].startswith('take')]
    data.update(blobs([s for p, s in keep if s not in data], repo))
    left = {}
    for p, w in why.items():
        if not w.startswith('take'):
            left[w] = left.get(w, 0) + 1
    files = {PUBLISH.get(p, p): TRANSFORM[p](data[s]) if p in TRANSFORM else data[s] for p, s in keep}
    return files, left, commit, bad, why


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('out', type=Path)
    ap.add_argument('--rev', default='HEAD')
    ap.add_argument('--repo', type=Path, default=ROOT)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--allow-machine-paths', action='store_true')
    ap.add_argument('--vanilla-map', type=Path, help='vanilla map.hex for the layout check (default: CAIME template '
                    'in the main checkout)')
    ap.add_argument('--init-repo', action='store_true',
                    help=f'make OUT a git repository with one commit by {RELEASE_NAME} <{RELEASE_EMAIL}>')
    ap.add_argument('--message', default='Initial public source release')
    a = ap.parse_args(argv)
    files, left, commit, bad, why = snapshot(a.rev, a.repo)
    files.setdefault('.gitignore', GITIGNORE.encode())
    if PUBLIC_SOURCE in why:
        raise SystemExit(f'{PUBLIC_SOURCE} is tracked at {commit[:12]}: it is generated for the public tree only')
    files[PUBLIC_SOURCE] = PUBLIC_SOURCE_TEXT.encode()
    res = check(files, bad, a.vanilla_map)
    failed = [k for k, v in res.items() if v and not (k == 'machine' and a.allow_machine_paths)]
    report = dict(rev=a.rev, commit=commit, files=len(files), bytes=sum(map(len, files.values())),
                  roots=ROOTS, left_out=dict(sorted(left.items())),
                  left_out_scripts=sorted(p for p, w in why.items() if p.startswith(('scripts/', 'tests/'))
                                          and not w.startswith('take') and w != 'archive'),
                  checks={k: dict(ok=not v, count=len(v), items=v[:200]) for k, v in res.items()},
                  failed=failed,
                  manifest=[dict(path=p, sha256=hashlib.sha256(d).hexdigest(), bytes=len(d),
                                 why=why.get(p, 'generated'))
                            for p, d in sorted(files.items())])
    for k, v in res.items():
        print(f'{k:9s} {"ok" if not v else ("LISTED" if k == "machine" and a.allow_machine_paths else "FAIL")}'
              f' {len(v)}' + ''.join(f'\n    {x}' for x in v[:20]) + ('\n    ...' if len(v) > 20 else ''))
    print(f'{len(files)} files, {report["bytes"]} bytes from {commit[:12]}; left out:',
          ', '.join(f'{k} {n}' for k, n in sorted(left.items())))
    if a.dry_run or failed:
        if failed:
            print('FAILED:', ', '.join(failed), '- nothing written')
        else:
            print(json.dumps({k: report[k] for k in ('commit', 'files', 'bytes', 'failed')}))
        return 1 if failed else 0
    if a.out.exists() and any(a.out.iterdir()):
        raise SystemExit(f'{a.out} exists and is not empty')
    for p, d in files.items():
        q = a.out / p
        q.parent.mkdir(parents=True, exist_ok=True)
        q.write_bytes(d)
    (a.out / 'RELEASE-MANIFEST.json').write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding='utf-8')
    print('written', a.out)
    if a.init_repo:
        print('repository:', init_repo(a.out, a.message))
    return 0


def git_env():
    """Environment for git in the new repository: the release identity for author and committer, and no system
    or user configuration (no local name, e-mail, signing key, hooks template or aliases leak in)."""
    env = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}
    env.update(GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=os.devnull,
               GIT_AUTHOR_NAME=RELEASE_NAME, GIT_AUTHOR_EMAIL=RELEASE_EMAIL,
               GIT_COMMITTER_NAME=RELEASE_NAME, GIT_COMMITTER_EMAIL=RELEASE_EMAIL)
    return env


def init_repo(out, message):
    """git init OUT and commit the snapshot as one commit by the release identity; RELEASE-MANIFEST.json (it
    names the private commit and the files left out) stays outside the commit. Returns the commit sha."""
    env = git_env()
    def git(*args):
        return subprocess.run(['git', '-C', str(out), '-c', f'user.name={RELEASE_NAME}',
                               '-c', f'user.email={RELEASE_EMAIL}', '-c', 'commit.gpgsign=false', *args],
                              env=env, capture_output=True, text=True, check=True).stdout.strip()
    subprocess.run(['git', 'init', '-q', '--template=', '-b', 'main', str(out)], env=env, capture_output=True,
                   check=True)
    (Path(out) / '.git/info').mkdir(parents=True, exist_ok=True)
    (Path(out) / '.git/info/exclude').write_text('/RELEASE-MANIFEST.json\n', encoding='utf-8')
    git('add', '-A')
    git('commit', '-q', '--no-verify', '-m', message)
    return git('rev-parse', 'HEAD')


if __name__ == '__main__':
    sys.exit(main())
