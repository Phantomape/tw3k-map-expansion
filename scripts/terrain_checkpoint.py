"""Content-verified checkpoints at explicit terrain pipeline boundaries.

Immutable dated checkpoints preserve evidence. A small pointer selects the
current generation; resuming restores that generation, never a first-run copy.
"""
import argparse, hashlib, json, os, shutil, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from analyze_reference import ROOT, digest
import campaign_world as _frame

A = _frame.WORKSPACE
S = A / 'work/release-input'
C = A / 'terrain-checkpoints'
# Polish writes the stage LF height map and sync_final then smooths it in place,
# so a polish checkpoint must carry it or a resume re-smooths a smoothed map.
POLISH_LF = tuple(f'terrain/campaigns/{v}/{n}' for v in _frame.TERRAIN_VARIANTS
                  for n in ('lf_height_map.dds', 'lf_height_map.compressed_map'))
# Every other stage file polish's variant loop writes besides global_meshes (the
# material blend, climate map, camera height map): a resume must not pair this
# generation's height map with another run's blend or climate.
POLISH_STAGE = tuple(f'terrain/campaigns/{v}/{n}' for v in _frame.TERRAIN_VARIANTS
                     for n in ('global_map/global_blend.dds', 'climate_map.cm')) + tuple(
    f'campaign_maps/{v}/camera_heightmap.png' for v in _frame.TERRAIN_VARIANTS)
# Likewise the edit alphas and sampled surface (workspace-relative): the boundary
# smoothing and sync_final raise/rewrite them in place after polish. The rest is
# what a polish run (serial --skip-cache or --merge) writes for the steps after it:
# the river water level and its stamp (sea height side lane), the salt-pan windows
# and tile climate (cache stage), farm-field reports and the polish report (read
# back by --cache-only).
POLISH_ANALYSIS = tuple(f'analysis/{n}-{v}.npy' for v in _frame.TERRAIN_VARIANTS
                        for n in ('polished-edit-alpha', 'polished-surface')) + ('analysis/polished-edit-alpha.npy',) + tuple(
    f'analysis/farm-fields-{v}.json' for v in _frame.TERRAIN_VARIANTS) + (
    'analysis/river-water-level.npy', 'analysis/river-water-level.json', 'analysis/salt-pan-windows.json',
    'analysis/polished-tile-climate.npy', 'analysis/geographic-terrain-polish.json')
if os.environ.get('TW3K_SINGLE_GROUND','0') == '1':
    # These exact shared fields and owners belong to this material generation;
    # a resume must not pair its blend with another run's tree acceptance field.
    POLISH_ANALYSIS += ('analysis/landform-tiles.npz','analysis/blend-writers.json',
                       'analysis/native-canvas-boundary-smoothing.json') + tuple(
        f'analysis/blend-owners-{v}.npy' for v in _frame.TERRAIN_VARIANTS)
JOBS = min(8, os.cpu_count() or 1)   # hash/copy threads: per-file open and hashing, not the NVMe, bound these steps

def inputs():
    return {rel: digest(A / rel) for rel in
            ('work/final-map/map.hex', 'work/current-region-design.json')}

def overflow_rels():
    """Stage paths of the overflow land meshes (fill_polished_mesh_gaps, mesh bake 3.2.1: land_mesh_N with N >=
    stage_invariants.BASE_LAND_MESHES, both files), in the stage or the base. The base set has one definition, the
    constant base_land_mesh_set checks (the gap fill numbers its overflow from 1 + the last base mesh), so a missing
    overflow manifest (a copied analysis) changes nothing. They are graph output, never geometry: a run that stopped
    after the gap fill left them in the stage, and the next geometry capture took them into the checkpoint and
    terrain-before-polish (release rehearsal 2026-11-01: 36 files, base_land_mesh_set)."""
    import re
    from stage_invariants import BASE_LAND_MESHES
    out = set()
    for v in _frame.TERRAIN_VARIANTS:
        for d in (S, A / 'terrain-before-polish'):
            for p in (d / f'terrain/campaigns/{v}/global_meshes').glob('land_mesh_*'):
                m = re.fullmatch(r'land_mesh_(\d+)\.(rigid_model_v2|compressed_map)', p.name)
                if m and int(m.group(1)) >= BASE_LAND_MESHES:
                    out.add(f'terrain/campaigns/{v}/global_meshes/{p.name}')
    return out

def files(boundary):
    skip = overflow_rels() if boundary == 'geometry' else set()
    for p in S.rglob('*'):
        if not p.is_file():
            continue
        rel = p.relative_to(S).as_posix()
        if not _frame.ships_terrain_path(rel):
            continue   # 3k_main_map stays vanilla; only TERRAIN_VARIANTS are processed
        if boundary == 'geometry' and rel.startswith(('terrain/campaigns/', 'campaign_maps/')):
            if rel in skip:
                continue   # graph output (overflow land meshes), not geometry
            yield p
        elif boundary == 'polish' and ('/global_meshes/' in rel and p.suffix in ('.rigid_model_v2', '.compressed_map') or rel in POLISH_LF or rel in POLISH_STAGE):
            yield p

def copy_hashed(src, dst):
    """shutil.copy2 that returns the sha256 of the copied bytes (one read, not copy + re-read)."""
    h = hashlib.sha256()
    with open(src, 'rb') as fi, open(dst, 'wb') as fo:
        while chunk := fi.read(16 << 20):
            h.update(chunk); fo.write(chunk)
    shutil.copystat(src, dst)
    return h.hexdigest()

def stat_key(p):
    st = p.stat()
    return [st.st_size, st.st_mtime_ns]

def pmap(fn, *items):
    """list(map(fn, *items)) on JOBS threads: hashlib and file I/O release the GIL."""
    with ThreadPoolExecutor(JOBS) as pool:
        return list(pool.map(fn, *items))

def capture(boundary):
    target = C / (boundary + '-' + str(time.time_ns()))
    before = None
    prev_dir, prev_files = None, {}
    if boundary == 'geometry':
        before_path = C / 'geometry-before-export.json'
        if not before_path.exists():
            raise RuntimeError('Record pre-export asset hashes before refreshing the geometry base')
        marked = json.loads(before_path.read_text())
        before, before_stats = marked['files'], marked.get('stats', {})
        try:   # the generation this one follows: pointer (replaced atomically) and its own manifest must agree
            prev = json.loads((C / 'geometry.json').read_text()); prev_dir = C / Path(prev['directory']).name
            if json.loads((prev_dir / 'manifest.json').read_text())['files'] == prev['files']: prev_files = prev['files']
        except (OSError, ValueError, KeyError, TypeError):
            pass   # no previous generation to share files with: copy everything

    def link_previous(source, rel, q):
        # Share the previous generation's file instead of copying when the source
        # holds exactly its bytes (hashed now: copy2 carries write and change times
        # between files, so no stat proves content) and its size, write time and
        # mode, i.e. the metadata copy_hashed would give the copy. Checkpoint files
        # are never written in place: restore only reads them, prune and
        # migrate_canvas delete or move whole generations.
        try:
            old = prev_dir / rel; a, b = old.stat(), source.stat()
            if rel not in prev_files or (a.st_size, a.st_mtime_ns, a.st_mode) != (b.st_size, b.st_mtime_ns, b.st_mode) or digest(source) != prev_files[rel]:
                return None
            os.link(old, q)   # OSError (e.g. NTFS 1023-link limit): copy instead
        except OSError:
            return None
        return prev_files[rel]

    def current_digest(p, rel):
        # Most files are untouched by the export: same size and write time as at
        # mark() means the hash mark() recorded (a git-style stat check).
        if rel in before and before_stats.get(rel) == stat_key(p):
            return before[rel]
        return digest(p)
    base_dir = A / 'terrain-before-polish'
    if boundary == 'geometry' and base_dir.exists():
        _frame.check_snapshot(base_dir)   # never restamp another canvas's base below
    paths = list(files(boundary)); rels = [p.relative_to(S).as_posix() for p in paths]
    currents = pmap(current_digest, paths, rels) if before is not None else [None] * len(paths)
    jobs, from_base = [], set()
    for p, rel, current in zip(paths, rels, currents):
        q = target / rel
        q.parent.mkdir(parents=True, exist_ok=True)
        source = p
        if before is not None and before.get(rel) == current and (base_dir / rel).exists():
            # Logical geometry export leaves renderer files untouched. Do not
            # promote already polished stage files into their own native base.
            source = base_dir / rel; from_base.add(rel)
        elif boundary == 'geometry':
            _frame.seed_from_stage_allowed(base_dir)
        jobs.append((source, rel, q))
    records = dict(zip(rels, pmap(lambda source, rel, q: (prev_files and link_previous(source, rel, q)) or copy_hashed(source, q), *zip(*jobs))))
    state = {'schema': 1, 'boundary': boundary, 'directory': str(target),
             'inputs': inputs(), 'files': records}
    if boundary == 'polish':
        workspace = state['workspace_files'] = {}
        for rel in POLISH_ANALYSIS:
            q = target / 'workspace' / rel; q.parent.mkdir(parents=True, exist_ok=True); workspace[rel] = copy_hashed(A / rel, q)
    target.mkdir(parents=True, exist_ok=True)
    (target / 'manifest.json').write_text(json.dumps(state, indent=2))
    C.mkdir(exist_ok=True)
    pointer = C / (boundary + '.json')
    temp = pointer.with_suffix('.tmp')
    temp.write_text(json.dumps(state, indent=2)); temp.replace(pointer)
    if boundary == 'geometry':
        # A geometry export is the only operation allowed to refresh this base.
        base = A / 'terrain-before-polish'
        def refresh(rel):
            # Temp + replace: an interrupted refresh leaves each base file whole
            # (old or new), never a truncated file a later capture would re-promote.
            q = base / rel; q.parent.mkdir(parents=True, exist_ok=True)
            temp = q.with_name(q.name + '.tmp')
            if rel in from_base:   # its checkpoint holds the bytes and write time just read from this very file
                temp.unlink(missing_ok=True); return
            shutil.copy2(target / rel, temp); os.replace(temp, q)
        pmap(refresh, records)
        # overflow land meshes an earlier capture took into the base go (graph output, never geometry)
        for rel in overflow_rels():
            (base / rel).unlink(missing_ok=True)
        (base / 'generation.json').write_text(json.dumps(state, indent=2))
        _frame.stamp_snapshot(base)
    print(f'Captured {boundary} checkpoint: {len(records)} verified assets', flush=True)

def mark(boundary):
    if boundary != 'geometry': raise ValueError('Only geometry export requires a pre-export mark')
    C.mkdir(exist_ok=True)
    hashes, stats = {}, {}
    paths = list(files(boundary))
    for p, (key, h) in zip(paths, pmap(lambda p: (stat_key(p), digest(p)), paths)):
        rel = p.relative_to(S).as_posix(); stats[rel] = key; hashes[rel] = h
    (C / 'geometry-before-export.json').write_text(json.dumps({'inputs': inputs(), 'files': hashes, 'stats': stats}, indent=2))
    print('Recorded pre-export geometry asset hashes', flush=True)

def check(boundary):
    """Fail unless the geometry base (terrain-before-polish) was exported from the current map/design.
    Every terrain build without -RebuildGeometry polishes that base; after a map or design edit it
    would lay the old export's geometry under the new map."""
    if boundary != 'geometry': raise ValueError('Only the geometry base records its export inputs')
    generation = A / 'terrain-before-polish' / 'generation.json'
    if not generation.exists():
        raise RuntimeError(f'{generation} is missing; run the terrain build with -RebuildGeometry')
    recorded = json.loads(generation.read_text()).get('inputs', {}); current = inputs()
    changed = sorted(rel for rel in current if recorded.get(rel) != current[rel])
    if changed:
        raise RuntimeError('Geometry base was exported from another ' + ', '.join(changed) + '; run the terrain build with -RebuildGeometry')
    print('Geometry base matches the current map/design', flush=True)

def restore(boundary):
    state = json.loads((C / (boundary + '.json')).read_text())
    if state['inputs'] != inputs():
        raise RuntimeError('Terrain checkpoint belongs to a different map/design; rebuild terrain')
    workspace = state.get('workspace_files', {})
    if boundary == 'polish' and not (set(POLISH_LF + POLISH_STAGE) <= set(state['files']) and set(POLISH_ANALYSIS) <= set(workspace)):
        raise RuntimeError('Polish checkpoint predates the snapshot of all polish outputs (LF height map, blend, climate, '
                           'edit alpha, river level, salt-pan windows); run the full terrain build')
    target = Path(state['directory'])
    for rel, expected in state['files'].items():
        if digest(target / rel) != expected:
            raise RuntimeError('Corrupt terrain checkpoint: ' + rel)
    for rel, expected in workspace.items():
        if digest(target / 'workspace' / rel) != expected:
            raise RuntimeError('Corrupt terrain checkpoint: ' + rel)
    for rel in state['files']:
        if _frame.ships_terrain_path(rel):   # older checkpoints also held 3k_main_map
            shutil.copy2(target / rel, S / rel)
    for rel in workspace:
        shutil.copy2(target / 'workspace' / rel, A / rel)
    print(f'Restored {boundary} checkpoint before downstream transforms', flush=True)

class StepSource:
    """Repeat a transform from its current input, or skip its verified output."""
    def __init__(self, name, context):
        self.path = C / (name + '-step.json')
        # Tie the record to the polish generation it consumed. A new polish
        # output (code or data change upstream) is a new input, not tampering;
        # resuming from a restored polish checkpoint keeps the same generation.
        pointer = C / 'polish.json'
        if pointer.exists():
            files = json.loads(pointer.read_text()).get('files', {})
            context = {**context, 'polish_generation': hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()}
        # Any pipeline code change is also a new input: an upstream step whose
        # code changed rewrites this step's input legitimately (not tampering).
        code = hashlib.sha256()
        for f in sorted(Path(__file__).parent.glob('*.py')) + sorted(Path(__file__).parent.glob('*.ps1')):
            code.update(f.name.encode() + b'|' + f.read_bytes())
        context = {**context, 'pipeline_code': code.hexdigest()}
        self.context = context
        try:
            self.previous = json.loads(self.path.read_text()) if self.path.exists() else {}
        except ValueError:
            # Never treat an unreadable record as empty: the step would then
            # re-transform its own output as if it were fresh input.
            raise RuntimeError(f'Unreadable step record {self.path}; restore the verified upstream checkpoint: delete '
                               'this file and rerun the full terrain build (it restores the step inputs first), never this step alone') from None
        self.records = {}

    def read(self, p):
        rel = p.relative_to(S).as_posix(); current = digest(p)
        prior = self.previous.get('files', {}).get(rel, {})
        if self.previous.get('context') == self.context and prior.get('output') == current:
            return None
        if self.previous.get('context') == self.context and prior and current != prior.get('input'):
            raise RuntimeError('Terrain asset changed after this step; restore the verified upstream checkpoint before rerunning: ' + rel)
        q = C / 'step-inputs' / (current + '-' + p.name)
        q.parent.mkdir(parents=True, exist_ok=True)
        if not q.exists(): shutil.copy2(p, q)
        self.records[rel] = {'input': current, 'source': str(q)}
        return p.read_bytes()

    def finish(self):
        for rel, record in self.records.items(): record['output'] = digest(S / rel)
        records = self.previous.get('files', {}) if self.previous.get('context') == self.context else {}
        records.update(self.records)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix('.tmp')
        temp.write_text(json.dumps({'schema': 1, 'context': self.context, 'files': records}, indent=2)); temp.replace(self.path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('capture', 'restore', 'mark', 'check'))
    parser.add_argument('boundary', choices=('geometry', 'polish'))
    args = parser.parse_args()
    {'capture': capture, 'restore': restore, 'mark': mark, 'check': check}[args.action](args.boundary)
