"""Translate vanilla campaign intro cameras and cutscene assets into the expanded world frame.

Vanilla intro cutscenes and literal start cameras are authored in the native
campaign world frame (x east, z north, metres of the 892x702 native map). The
expanded canvas moves the native map by NATIVE_SHIFT_WORLD, so every literal
world x/z in these assets must move by the same amount. Heights, angles, times,
FOV, scales and local (composite-scene) offsets stay byte-identical.

Formats and the fields this script moves (anything else in a matched file is
left byte-identical; an unrecognised structure raises):
  cameras/*.xml (WARSCAPE_PROPERTIES_FILE)  NODE[Camera]/VECTOR[position|target]/DOUBLE[X|Z]
  *.kfe (EFFECT_FILE), *.kfcs (COMPOSITE_SCENE_FILE), *.kfa (AUDIO_EVENT_FILE)
                                             root VECTOR[position|target]/DOUBLE[X|Z]
      moved: KEYFRAME data, KEYFRAME handle1_y/handle2_y (bezier handles are
      value-space; handle*_x are times), and DOUBLE default only when the
      channel has no keyframes (otherwise the default is inert and untouched).
  *.cindyscene (SCENE_FILE)  only the editor viewport attribute perspective_camera
      ("ex,ey,ez:tx,ty,tz") is world space; the root VECTOR is the scene's own
      local offset (0 / -0.85 in vanilla) and is never moved. Editor-only, so
      scenes are written only with --include-editor-only.
  *.lua under script/campaign/  numeric literal x/y (world x/z) arguments of the
      camera calls in CALLS; arguments computed from game objects (already in
      the expanded frame) are recorded and left alone.
Not translated (recorded in the report): .csc (binary composite scenes, local
space, placed by .kfcs), .ktr/.kfc (no positions), preferences/*.txt, replays
and *_temp editor files (never loaded).

Only files reachable from a cindy scene that campaign Lua actually plays, plus
the Lua files with literal world cameras, are written. Output mirrors pack
paths: <out>/<pack path>.
"""
import tw3k_paths
import argparse, json, math, re
from pathlib import Path

import campaign_world as cw
from audit_ground_seams import Packs

GAME = tw3k_paths.GAME_DATA
DEFAULT_OUT = cw.WORKSPACE / 'camera-translation/stage'
DEFAULT_REPORT = cw.WORKSPACE / 'camera-translation/camera-translation-report.json'
ROOTS = ('script/campaign/', 'script/benchmarks/campaign_benchmark/')
SKIP_PACK_PREFIX = ('tw3k_', 'local_', 'audio', 'movies', 'shaders')
DX, DZ = cw.NATIVE_SHIFT_WORLD
NATIVE_Z = cw.VANILLA_WORLD[1]
XML_EXT = {'.xml', '.kfe', '.kfcs', '.kfa', '.ktr', '.kfc', '.cindyscene', '.cindyscenemanager'}
IGNORED_EXT = {'.csc': 'binary composite scene in local space, placed by .kfcs',
               '.txt': 'cindy editor preferences', '.replay': 'editor capture replay',
               '.rpy': 'editor capture replay'}
NUM = r'-?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?'


def vanilla_packs():
    packs = [p for p in sorted(GAME.glob('*.pack')) if not p.name.startswith(SKIP_PACK_PREFIX)]
    seen = {}
    for p in packs:
        for k in Packs([p]).index:
            if k.startswith(ROOTS) or (k.startswith('script/_lib/lib_campaign') and k.endswith('.lua')):
                if k in seen:
                    raise RuntimeError(f'{k} is in both {seen[k]} and {p.name}; pack precedence would be a guess')
                seen[k] = p.name
    return Packs(packs), sorted(seen)


def fmt(v):
    return f'{v:.6f}'


def in_native(x, z):
    return -1e-6 <= x <= cw.NATIVE_WORLD_WIDTH + 1e-6 and -1e-6 <= z <= NATIVE_Z + 1e-6


def in_world(x, z):
    return 0 <= x <= cw.WORLD_WIDTH and 0 <= z <= cw.WORLD_HEIGHT


# ----------------------------------------------------------------------------- XML family
TAG = re.compile(r'<(/?)([A-Za-z_][A-Za-z_0-9]*)((?:\s+[A-Za-z_][A-Za-z_0-9]*="[^"]*")*)\s*(/?)>')
ATTR = re.compile(r'([A-Za-z_][A-Za-z_0-9]*)="([^"]*)"')
WORLD_ROOTS = {'WARSCAPE_PROPERTIES_FILE', 'EFFECT_FILE', 'COMPOSITE_SCENE_FILE', 'AUDIO_EVENT_FILE'}
PLAIN_ROOTS = {'TRIGGER_FILE', 'CUSTOM_FILE'}
SCENE_ROOTS = {'SCENE_FILE', 'SCENE_MANAGER_FILE'}


def elements(text):
    """Parse the cindy XML dialect into a tree of dicts holding attribute spans."""
    pos, stack, root = 0, [], None
    if text.startswith('\ufeff') or text.startswith('\xef\xbb\xbf'):
        raise RuntimeError('BOM present; unknown variant')
    for m in TAG.finditer(text):
        gap = text[pos:m.start()]
        if gap.strip():
            raise RuntimeError(f'Unexpected text {gap.strip()[:40]!r}')
        pos = m.end()
        close, tag, attrs, selfclose = m.group(1), m.group(2), m.group(3), m.group(4)
        if close:
            if not stack or stack[-1]['tag'] != tag:
                raise RuntimeError(f'Mismatched </{tag}>')
            stack.pop()
            continue
        base = m.start(3)
        e = {'tag': tag, 'attrs': {a.group(1): (a.group(2), base + a.start(2), base + a.end(2)) for a in ATTR.finditer(attrs)},
             'children': []}
        if stack:
            stack[-1]['children'].append(e)
        elif root is None:
            root = e
        else:
            raise RuntimeError('Multiple roots')
        if not selfclose:
            stack.append(e)
    if stack or root is None or text[pos:].strip():
        raise RuntimeError('Unbalanced or empty document')
    return root


def name(e):
    return e['attrs'].get('name', ('',))[0]


def world_vectors(root):
    """Yield (vector_name, {'X': DOUBLE, 'Y': ..., 'Z': ...}) for proven world-space vectors."""
    tag = root['tag']
    if tag == 'WARSCAPE_PROPERTIES_FILE':
        cams = [c for c in root['children'] if c['tag'] == 'NODE' and name(c) == 'Camera']
        if len(cams) != 1 or len(root['children']) != 1:
            raise RuntimeError('Expected exactly one NODE[Camera] in a warscape properties file')
        parent = cams[0]
    else:
        parent = root
    found = []
    for c in parent['children']:
        if c['tag'] == 'VECTOR':
            if name(c) not in ('position', 'target'):
                raise RuntimeError(f'Unknown VECTOR {name(c)!r} in {tag}')
            chans = {name(d): d for d in c['children']}
            if set(chans) != {'X', 'Y', 'Z'} or any(d['tag'] != 'DOUBLE' for d in c['children']):
                raise RuntimeError(f'VECTOR {name(c)} does not hold DOUBLE X/Y/Z')
            found.append((name(c), chans))
    # any VECTOR deeper than the recognised level is an unknown layout
    def deep(e, depth):
        for c in e['children']:
            if c['tag'] == 'VECTOR' and e is not parent:
                raise RuntimeError(f'VECTOR {name(c)!r} at an unrecognised depth in {tag}')
            deep(c, depth + 1)
    deep(root, 0)
    if tag == 'WARSCAPE_PROPERTIES_FILE' and {n for n, _ in found} != {'position', 'target'}:
        raise RuntimeError('Camera file without position and target vectors')
    if not found:
        raise RuntimeError(f'{tag} without a world position vector')
    return found


def keyframes(d):
    ks = [k for k in d['children'] if k['tag'] == 'KEYFRAME']
    if len(ks) != len(d['children']):
        raise RuntimeError('Unexpected child under a DOUBLE channel')
    return ks


def translate_xml(key, text):
    """Return (new_text, moved, info) for one cindy XML file."""
    root = elements(text)
    tag = root['tag']
    edits, moved, info = [], [], {'root': tag}

    def move(span_owner, attr, delta, label):
        val, a, b = span_owner['attrs'][attr]
        if not re.fullmatch(NUM, val):
            raise RuntimeError(f'Non-numeric {attr}={val!r}')
        new = fmt(float(val) + delta)
        edits.append((a, b, val, new))
        moved.append({'field': label, 'from': val, 'to': new})

    if tag in WORLD_ROOTS:
        effective = {}
        for vname, chans in world_vectors(root):
            for axis, d in chans.items():
                ks = keyframes(d)
                if axis == 'Y':
                    continue
                delta = DX if axis == 'X' else DZ
                for i, k in enumerate(ks):
                    if set(k['attrs']) - {'data', 'time', 'handle1_x', 'handle1_y', 'handle2_x', 'handle2_y',
                                          'handle1_type', 'handle2_type'}:
                        raise RuntimeError(f'Unknown KEYFRAME attributes {sorted(k["attrs"])}')
                    for attr in ('data', 'handle1_y', 'handle2_y'):
                        if attr in k['attrs']:
                            move(k, attr, delta, f'{vname}.{axis}[{i}].{attr}')
                if not ks:
                    move(d, 'default', delta, f'{vname}.{axis}.default(no keyframes)')
                vals = [float(k['attrs']['data'][0]) for k in ks] or [float(d['attrs']['default'][0])]
                effective[(vname, axis)] = vals
        for vname in {v for v, _ in effective}:
            xs, zs = effective[(vname, 'X')], effective[(vname, 'Z')]
            sentinel = all(abs(v + .85) < 1e-6 for v in xs + zs)
            if not sentinel and not (all(in_native(x, 0) for x in xs) and all(in_native(0, z) for z in zs)):
                raise RuntimeError(f'{vname} X {min(xs)}..{max(xs)} / Z {min(zs)}..{max(zs)} leaves the native world; '
                                   'frame unproven')
            info.setdefault('vectors', {})[vname] = {
                'native_last': [xs[-1], zs[-1]], 'canvas_last': [xs[-1] + DX, zs[-1] + DZ],
                'keyframes': [len(xs), len(zs)], 'editor_sentinel_-0.85': sentinel}
    elif tag in SCENE_ROOTS:
        for c in root['children']:
            if c['tag'] == 'VECTOR':
                for d in c['children']:
                    for k in d['children']:
                        if abs(float(k['attrs']['data'][0])) >= 1:
                            raise RuntimeError('Cindy scene root offset is not ~0; scene-level translation unknown')
        refs = []
        def walk(e):
            for a, (v, _, _) in e['attrs'].items():
                if a.endswith('_file') and v:
                    refs.append(v)
            for c in e['children']:
                walk(c)
        walk(root)
        info['refs'] = refs
        pc = root['attrs'].get('perspective_camera')
        if pc:
            val, a, b = pc
            parts = val.split(':')
            if len(parts) != 2 or any(len(p.split(',')) != 3 for p in parts):
                raise RuntimeError(f'Unknown perspective_camera {val!r}')
            pts = [[float(x) for x in p.split(',')] for p in parts]
            if all(in_native(p[0], p[2]) for p in pts):
                new = ':'.join(','.join((fmt(p[0] + DX), parts[i].split(',')[1], fmt(p[2] + DZ)))
                               for i, p in enumerate(pts))
                edits.append((a, b, val, new))
                moved.append({'field': 'perspective_camera(editor viewport)', 'from': val, 'to': new})
            else:
                info['perspective_camera_untouched'] = f'{val} lies outside the native world (editor default)'
    elif tag in PLAIN_ROOTS:
        def walk(e):
            if e['tag'] == 'VECTOR':
                raise RuntimeError(f'{tag} unexpectedly holds a VECTOR')
            for c in e['children']:
                walk(c)
        walk(root)
    else:
        raise RuntimeError(f'Unknown cindy root {tag}')
    return apply(text, edits), moved, info, edits


def apply(text, edits):
    out, pos = [], 0
    for a, b, old, new in sorted(edits):
        if a < pos or text[a:b] != old:
            raise RuntimeError('Overlapping or stale edit')
        out += [text[pos:a], new]
        pos = b
    return ''.join(out + [text[pos:]])


def unapply(new_text, edits):
    """Rebuild the original from the output; proves only the recorded spans changed."""
    out, pos, shift = [], 0, 0
    for a, b, old, new in sorted(edits):
        na = a + shift
        if new_text[na:na + len(new)] != new:
            raise RuntimeError('Output does not contain the recorded value')
        out += [new_text[pos:na], old]
        pos = na + len(new)
        shift += len(new) - (b - a)
    return ''.join(out + [new_text[pos:]])


# ----------------------------------------------------------------------------- Lua
# call name -> ('args', x index, y index) or ('tables', first table argument index)
CALLS = {
    'set_camera_position': ('args', 0, 1), 'SetCameraPosition': ('args', 0, 1),
    'show_benchmark_if_required': ('args', 3, 4),
    'set_skip_camera': ('args', 0, 1), 'set_restore_camera': ('args', 1, 2),
    'scroll_camera_for_intervention': ('args', 1, 2), 'scroll_camera_for_intervention_action': ('args', 1, 2),
    'scroll_camera_with_direction': ('tables', 2), 'ScrollCameraWithDirection': ('tables', 2),
    'scroll_camera_from_current': ('tables', 2), 'scroll_camera_with_cutscene': ('tables', 2),
    'cut_and_scroll_camera_with_cutscene': ('tables', 2),
}
CALL_RE = re.compile(r'\b(' + '|'.join(CALLS) + r')\s*\(')
CINDY_RE = re.compile(r'"(script/[^"]+\.cindyscene)"', re.I)
# Reviewed 5-number tables that are not camera waypoints (integer progression thresholds).
NON_CAMERA_TABLES = {'script/campaign/_shared/3k_campaign_progression_extended.lua': 4}
WAYPOINT_RE = re.compile(r'\{\s*' + r'\s*,\s*'.join([NUM] * 5) + r'\s*\}')


def lua_code_mask(text):
    """True for characters that are code (not comment, not string contents)."""
    mask = [True] * len(text)
    i, n = 0, len(text)
    def long_close(j):
        m = re.match(r'\[(=*)\[', text[j:])
        return (m.end(), ']' + m.group(1) + ']') if m else (None, None)
    while i < n:
        if text.startswith('--', i):
            ln, close = long_close(i + 2)
            end = text.find(close, i + 2 + ln) + len(close) if ln else text.find('\n', i)
            end = n if end < (len(close) if ln else 0) else end
            for j in range(i, end):
                mask[j] = False
            i = end
        elif text[i] in '"\'':
            q, j = text[i], i + 1
            while j < n and text[j] != q:
                j += 2 if text[j] == '\\' else 1
            for k in range(i + 1, j):
                mask[k] = False
            i = j + 1
        elif text[i] == '[' and long_close(i)[0]:
            ln, close = long_close(i)
            end = text.find(close, i + ln) + len(close)
            for k in range(i + 1, end - 1):
                mask[k] = False
            i = end
        else:
            i += 1
    return mask


def split_args(text, mask, open_paren):
    """Top-level argument spans of the call whose '(' is at open_paren."""
    depth, start, spans, i = 0, open_paren + 1, [], open_paren
    while True:
        i += 1
        if i >= len(text):
            raise RuntimeError('Unbalanced call')
        if not mask[i]:
            continue
        c = text[i]
        if c in '([{':
            depth += 1
        elif c in ')]}':
            if depth == 0:
                spans.append((start, i))
                return spans, i
            depth -= 1
        elif c == ',' and depth == 0:
            spans.append((start, i))
            start = i + 1


def code_of(text, mask, a, b):
    return ''.join(ch if mask[a + k] else ' ' for k, ch in enumerate(text[a:b])).strip()


def literal_span(text, mask, a, b):
    code = code_of(text, mask, a, b)
    if not re.fullmatch(NUM, code):
        return None
    for k in range(a, b):
        if mask[k] and not text[k].isspace():
            m = re.compile(NUM).match(text, k)
            return m.start(), m.end(), m.group(0)
    raise RuntimeError('literal vanished')


def classify_pair(text, mask, sa, sb, label, edits, moved, computed):
    lx, ly = literal_span(text, mask, *sa), literal_span(text, mask, *sb)
    cx, cy = code_of(text, mask, *sa), code_of(text, mask, *sb)
    if lx and ly:
        x, y = float(lx[2]), float(ly[2])
        if not in_native(x, y):
            raise RuntimeError(f'{label} literal ({x}, {y}) is outside the native world')
        nx, ny = fmt(x + DX), fmt(y + DZ)
        edits += [(lx[0], lx[1], lx[2], nx), (ly[0], ly[1], ly[2], ny)]
        moved.append({'call': label, 'from': [lx[2], ly[2]], 'to': [nx, ny]})
    elif lx or ly:
        raise RuntimeError(f'{label}: one literal and one computed coordinate ({cx!r}, {cy!r})')
    else:
        for c in (cx, cy):
            if re.search(r'(?<![\w.])\d{2,}', c):
                raise RuntimeError(f'{label}: computed coordinate {c!r} embeds a numeric literal')
        computed.append({'call': label, 'args': [cx, cy]})


def translate_lua(key, text):
    mask = lua_code_mask(text)
    edits, moved, computed, scenes, consumed = [], [], [], [], set()
    for m in CALL_RE.finditer(text):
        if not mask[m.start()]:
            continue
        line_start = text.rfind('\n', 0, m.start()) + 1
        if re.match(r'\s*function\b', text[line_start:m.start()]) or re.search(r'function\s+[\w.:]*$', text[line_start:m.start()]):
            continue
        fn = m.group(1)
        spans, close = split_args(text, mask, m.end() - 1)
        line = text.count('\n', 0, m.start()) + 1
        label = f'{fn}@{line}'
        spec = CALLS[fn]
        if spec[0] == 'args':
            if fn == 'set_skip_camera' and len(spans) == 1 and not code_of(text, mask, *spans[0]):
                continue
            if len(spans) == 1 and re.fullmatch(r'unpack\([\w.\[\]]+\)', code_of(text, mask, *spans[0])):
                computed.append({'call': label, 'args': [code_of(text, mask, *spans[0])]})
                continue
            if len(spans) <= spec[2]:
                raise RuntimeError(f'{key}:{label}: unrecognised call form with {len(spans)} arguments')
            classify_pair(text, mask, spans[spec[1]], spans[spec[2]], label, edits, moved, computed)
        else:
            for j, (a, b) in enumerate(spans[spec[1]:]):
                code = code_of(text, mask, a, b)
                if code.startswith('{'):
                    inner, iclose = split_args(text, mask, text.index('{', a))
                    consumed.add(text.index('{', a))
                    if len(inner) != 5:
                        raise RuntimeError(f'{key}:{label}: waypoint table with {len(inner)} entries')
                    classify_pair(text, mask, inner[0], inner[1], f'{label}.wp{j}', edits, moved, computed)
                elif re.search(r'\d', code) and not re.fullmatch(r'[\w.]+', code):
                    raise RuntimeError(f'{key}:{label}: unrecognised waypoint argument {code!r}')
                else:
                    computed.append({'call': f'{label}.wp{j}', 'args': [code]})
    stray = [m for m in WAYPOINT_RE.finditer(text) if mask[m.start()] and m.start() not in consumed]
    if stray and len(stray) != NON_CAMERA_TABLES.get(key):
        m = stray[0]
        raise RuntimeError(f'{key}: stray literal camera-like table {m.group(0)!r} outside a recognised call')
    for m in CINDY_RE.finditer(text):
        if mask[m.start()]:
            scenes.append(m.group(1))
    return apply(text, edits), moved, computed, scenes, edits


# ----------------------------------------------------------------------------- sanity: settlements
def town_centres(path):
    from hex_grid import grid
    meta, cell = grid(Path(path))
    pts = {}
    for y in range(meta['height']):
        for x in range(meta['width']):
            key, c = cell(x, y)
            if c[2] >> 4 == 1:
                pts.setdefault(key, []).append((x, y))
    out = {}
    for k, ps in pts.items():
        q = sum(p[0] for p in ps) / len(ps)
        r = sum(p[1] for p in ps) / len(ps)
        # Engine display spacing: world width / (columns - 1) of this map.
        dx = cw.NATIVE_DX if meta['width'] == cw.NATIVE_W else cw.HEX_DX
        out[k] = (q * dx, (r + (int(q) % 2) * .5) * dx * cw.ZS)   # the engine's odd-column rule (campaign_world)
    return out


def nearest(towns, x, z):
    k = min(towns, key=lambda k: math.hypot(towns[k][0] - x, towns[k][1] - z))
    return k, round(math.hypot(towns[k][0] - x, towns[k][1] - z), 3)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('out', nargs='?', default=str(DEFAULT_OUT), help='output root (mirrors pack paths)')
    ap.add_argument('--report', default=str(DEFAULT_REPORT))
    ap.add_argument('--check', action='store_true', help='verify the output directory, write nothing')
    ap.add_argument('--include-editor-only', action='store_true',
                    help='also write .cindyscene files whose only change is the editor perspective_camera')
    ap.add_argument('--native-map', default=str(tw3k_paths.ASSEMBLY_KIT / 'raw_data/EmpireDesignData/campaign_maps/3k_dlc07_main_map/map.hex'))
    ap.add_argument('--canvas-map', default=str(cw.WORKSPACE / 'work/final-map/map.hex'))
    args = ap.parse_args()
    out_root = Path(args.out)
    packs, keys = vanilla_packs()
    report = {'shift_world': [DX, DZ], 'world': [cw.WORLD_WIDTH, cw.WORLD_HEIGHT], 'files': {}, 'ignored': {},
              'unreferenced_with_world_positions': [], 'lua_computed_calls': {}}
    results = {}
    for key in keys:
        ext = ('.' + key.rsplit('/', 1)[-1].rsplit('.', 1)[-1]).lower()
        if ext.endswith('_temp'):
            report['ignored'][key] = 'editor temp file, never loaded'
            continue
        raw = packs.read(key)
        text = raw.decode('latin-1')
        if ext == '.lua':
            if not key.startswith('script/campaign/') and not key.startswith('script/_lib/'):
                continue
            new, moved, computed, scenes, edits = translate_lua(key, text)
            results[key] = dict(kind='lua', new=new, moved=moved, edits=edits, scenes=scenes, raw=raw)
            if computed:
                report['lua_computed_calls'][key] = computed
        elif ext in XML_EXT:
            try:
                new, moved, info, edits = translate_xml(key, text)
            except RuntimeError as e:
                raise RuntimeError(f'{key}: {e}') from None
            results[key] = dict(kind=ext, new=new, moved=moved, edits=edits, info=info, raw=raw)
        elif ext in IGNORED_EXT:
            report['ignored'][key] = IGNORED_EXT[ext]
        elif ext in ('.environment', '.xlsm'):
            report['ignored'][key] = 'no campaign positions'
        else:
            raise RuntimeError(f'{key}: unknown file type {ext}')

    # reachability: scenes played from Lua -> assets they reference
    lower = {k.lower(): k for k in results}
    reached, queue = set(), []
    for key, r in results.items():
        if r['kind'] == 'lua':
            for s in r.get('scenes', []):
                if s.lower() not in lower:
                    report.setdefault('missing_scenes_referenced_by_vanilla_lua', []).append([key, s])
                    continue
                queue.append(lower[s.lower()])
    for k in list(results):
        if results[k]['kind'] == '.cindyscenemanager':
            queue.append(k)
    while queue:
        k = queue.pop()
        if k in reached:
            continue
        reached.add(k)
        for ref in results[k].get('info', {}).get('refs', []):
            target = (Path(k).parent / ref).as_posix()
            parts = []
            for p in target.split('/'):
                if p == '..':
                    parts.pop()
                elif p != '.':
                    parts.append(p)
            t = '/'.join(parts).lower()
            if t not in lower:
                raise RuntimeError(f'{k} references missing {ref}')
            queue.append(lower[t])

    native_towns = town_centres(args.native_map)
    canvas_towns = town_centres(args.canvas_map)
    checks, failures, written = [], [], []
    for key, r in sorted(results.items()):
        if not r['moved']:
            continue
        if unapply(r['new'], r['edits']).encode('latin-1') != r['raw']:
            raise RuntimeError(f'{key}: byte identity outside moved fields failed')
        editor_only = r['kind'] == '.cindyscene' or r['kind'] == '.cindyscenemanager'
        ship = (r['kind'] == 'lua' or key in reached) and (not editor_only or args.include_editor_only)
        entry = {'kind': r['kind'], 'moved_values': len(r['moved']), 'shipped': ship,
                 'reachable_from_lua': r['kind'] == 'lua' or key in reached,
                 'unchanged_bytes_identical': True, 'moved': r['moved']}
        if editor_only:
            entry['note'] = 'only the editor viewport perspective_camera changes'
        if r['kind'] != 'lua' and key not in reached:
            report['unreferenced_with_world_positions'].append(key)
        # out-of-world and settlement checks on effective points
        pts = []
        if r['kind'] == 'lua':
            pts = [(m['call'], float(m['from'][0]), float(m['from'][1])) for m in r['moved']]
        for vname, v in r.get('info', {}).get('vectors', {}).items():
            if not v['editor_sentinel_-0.85']:
                pts.append((vname + '(last)', *v['native_last']))
        for label, x, z in pts:
            nx, nz = x + DX, z + DZ
            nk, nd = nearest(native_towns, x, z)
            ck, cd = nearest(canvas_towns, nx, nz)
            c = {'file': key, 'point': label, 'native': [round(x, 3), round(z, 3)], 'canvas': [round(nx, 3), round(nz, 3)],
                 'inside_world': in_world(nx, nz), 'native_nearest_town': [nk, nd], 'canvas_nearest_town': [ck, cd],
                 'same_town': nk in canvas_towns and abs(math.hypot(canvas_towns[nk][0] - nx, canvas_towns[nk][1] - nz) - nd) < 1.0}
            c['nearest_changed_to_new_town'] = ck != nk
            checks.append(c)
            if not c['inside_world']:
                failures.append(c)
        for m in r['moved']:
            for a, b in [m.get('to', [None, None])] if r['kind'] == 'lua' else []:
                if not in_world(float(a), float(b)):
                    failures.append({'file': key, 'value': m})
        report['files'][key] = entry
        if ship:
            target = out_root / key
            written.append(key)
            if args.check:
                if not target.is_file() or target.read_bytes() != r['new'].encode('latin-1'):
                    failures.append({'file': key, 'check': 'output differs or missing'})
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(r['new'].encode('latin-1'))
    if args.check and out_root.exists():
        extra = sorted(p.relative_to(out_root).as_posix() for p in out_root.rglob('*') if p.is_file()
                       and p.relative_to(out_root).as_posix() not in written)
        if extra:
            failures.append({'check': 'unexpected files in output', 'files': extra})
    report['settlement_checks'] = checks
    report['settlement_mismatches'] = [c for c in checks if not c['same_town']]
    report['shipped_pack_paths'] = written
    report['summary'] = {
        'shipped': len(written),
        'shipped_by_type': {t: sum(1 for k in written if (results[k]['kind'] if results[k]['kind'] != 'lua' else '.lua') == t)
                            for t in sorted({results[k]['kind'] if results[k]['kind'] != 'lua' else '.lua' for k in written})},
        'files_with_world_values': len(report['files']),
        'moved_values': sum(e['moved_values'] for e in report['files'].values()),
        'settlement_checks': len(checks), 'settlement_mismatches': len(report['settlement_mismatches']),
        'failures': len(failures)}
    report['failures'] = failures
    if not args.check:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps(report['summary'], indent=1))
    raise SystemExit(1 if failures else 0)


if __name__ == '__main__':
    main()
