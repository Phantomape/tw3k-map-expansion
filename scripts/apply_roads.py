"""Gameplay roads for the planned Western Regions and frontier routes: road edge masks in map.hex.

Reads the route plans (plan_western_roads.py -> analysis/roads/western-roads.json, plan_frontier_roads.py
-> analysis/roads/frontier-roads.json) and lays every land leg as a chain of road hexes in
work/final-map/map.hex. Sea legs (kind 'sea') get no road bits.

Cell format (CAIME MapHexFile.ReadHexData16): byte 3 bits 1-6 are the road edge mask, one bit per
direction (0 up, 1 up-right, 2 down-right, 3 down, 4 down-left, 5 up-left; flat-top hexes, odd columns
half a row up); a hex is a road when its mask is non-zero and masks are mutual between neighbours.
The cell holds no road level: campaign_map_roads (3k_dlc07_road_level_1/2/3, movement cost 100/75/50,
threshold 0/10/40, one turn to up/downgrade) is applied by the game at run time; map.hex and the .ppd
carry only the road network (CAIME reads the cheapest level, 50, as the pathfinding heuristic).
So the new roads, like every vanilla road, start as level-1 tracks and follow the regions they serve.

How a leg is laid (both ends are towns' main-slot hexes, plan waypoints, or the vanilla road network):
  * cheapest hex path that keeps to the plan (plan hexes cost 1, a detour hex OFF_PLAN) and avoids what
    CAIME's RoadsValidator rejects or the road tracer (TownRouteTracer) cuts: sea, beach and cliff
    hexes, other towns' sprawl and main slots, and sprawl entered and left again;
  * a leg that starts or ends at a vanilla town starts from (ends on) that town's vanilla road network
    and spends as few vanilla playable hexes as possible (VPLAY_COST each, VPLAY_OPEN_COST where the
    vanilla impassable rim has to be crossed):
    vanilla roads rarely reach the edge of vanilla land, so those few hexes are the join and are
    reported one by one (vanilla_playable_changes);
  * the chain is linked both ways; afterwards every route hex next to another road that is not
    already within JOIN_DEPTH steps of it is joined to it (no triangles), so routes meet each other
    and the roads already on the map;
  * every route hex is opened: impassable bit (byte 2 bit 3) cleared, inland 'cliff' (byte 0 = 3, away
    from water) made plain land; closed slivers under FRAGMENT_MAX hexes left beside a route are opened too.
Protected: vanilla playable land except the reported joins; nothing but byte 2 bit 3 and byte 3 bits
1-6 changes anywhere.

Checks: mutual masks, no road on sea/beach/cliff/impassable among the changed hexes, and a replica of
CAIME's TownRouteTracer (town-to-town routes exported into pathfinding.ppd): every route's towns must
be connected by traced routes; SDK areas per region (hex_components.region_areas) <= SDK_AREA_LIMIT.
The report and a preview go to output/offline-view/roads-gameplay
(the report also to <workspace>/analysis with --apply).

  python apply_roads.py            # dry run
  python apply_roads.py --apply    # back up and write work/final-map/map.hex

Afterwards: kit_map.sync_from_final() (or sdk_memory_export.py) and Tw3kBuild pathfinding / borders /
trade-routes (build_combined_geometry.ps1).
"""
import tw3k_paths
import argparse, collections, heapq, json, math, os, shutil, time
from pathlib import Path

import numpy as np
from scipy import ndimage
from PIL import Image, ImageDraw, ImageFont

import campaign_world as cw
import map_hex
from hex_components import region_areas, hex_dilate, SDK_AREA_LIMIT

A = cw.WORKSPACE
ROOT = cw._ROOT
FINAL = A / 'work/final-map/map.hex'
# TW3K_ROADS_DIR: the plans of a test copy (western-roads.json there; frontier-roads.json from there
# when present, else analysis/roads); TW3K_ROADS_OUT: its previews and dry-run report
_PLAN_DIR = Path(os.environ.get('TW3K_ROADS_DIR', ROOT / 'analysis/roads'))
PLANS = (_PLAN_DIR / 'western-roads.json',
         _PLAN_DIR / 'frontier-roads.json' if (_PLAN_DIR / 'frontier-roads.json').exists() else ROOT / 'analysis/roads/frontier-roads.json')
OUT = Path(os.environ.get('TW3K_ROADS_OUT', ROOT / 'output/offline-view/roads-gameplay'))
OFF_PLAN = 6.0          # per hex outside the plan corridor (detours only where the plan cannot carry a road)
IMPASSABLE_EXTRA = 0.5  # per impassable hex opened (keeps detours on passable ground when equal)
VPLAY_COST = 10.0       # per new road hex on vanilla playable land
VPLAY_OPEN_COST = 30.0  # per impassable vanilla playable hex opened (the rim round the vanilla map)
HUB_PENALTY = 0.5       # per town: start/end on a main-slot hex other than the one already used
MARGIN = 30             # hexes around a leg's box searched for its path
FRAGMENT_MAX = 40       # closed (impassable/inland cliff) components under this many hexes touching a route are opened
SPRAWL_REACH = 30       # sprawl blobs this close to a route (a road can split a hazard area far from the town) are rechecked
JOIN_DEPTH = 6        # a route hex is joined to a neighbouring road only if not already this close
VANILLA_LINKS = {'云中'}  # route ends off any town that join the nearest vanilla road (Yanmen)

# CAIME Hex.Directions_FlatTop: (dq, dr) per direction for even / odd columns
DIRS = (((0, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0)),
        ((0, 1), (1, 1), (1, 0), (0, -1), (-1, 0), (-1, 1)))


class Grid:
    def __init__(self, cells, vcells, names):
        self.cells = cells
        self.H, self.W = cells.shape[:2]
        self.names = names
        self.mask = ((cells[:, :, 3] >> 1) & 63).astype(np.uint8)
        self.kind = cells[:, :, 0] & 3                  # 0 land, 1 sea, 2 beach, 3 cliff
        self.imp = (cells[:, :, 2] & 8) != 0
        self.slot = cells[:, :, 2] >> 4                 # 1 main settlement slot, 2 port
        self.sprawl = (cells[:, :, 3] & 1) != 0
        self.ids = map_hex.region_ids(cells)
        vn = np.array(vcells['names'] + ['?'], dtype=object)
        self.vplay = np.zeros((self.H, self.W), bool)
        self.vplay[cw.NATIVE_HEX_SLICE] = ~np.vectorize(lambda s: 'non_playable' in s or 'sea' in s)(vn[map_hex.region_ids(vcells['cells'])])
        self.vroad = np.zeros((self.H, self.W), bool)
        self.vroad[cw.NATIVE_HEX_SLICE] = ((vcells['cells'][:, :, 3] >> 1) & 63) > 0

    def nb(self, q, r, d):
        dq, dr = DIRS[q & 1][d]
        q, r = q + dq, r + dr
        return (q, r) if 0 <= q < self.W and 0 <= r < self.H else None

    def neighbours(self, q, r):
        for d in range(6):
            n = self.nb(q, r, d)
            if n:
                yield d, n

    def direction(self, a, b):
        for d, n in self.neighbours(*a):
            if n == b:
                return d
        raise ValueError(f'{a} and {b} are not neighbours')

    def spr(self, q, r):
        return int(self.ids[r, q]) if self.sprawl[r, q] else None

    def road_component(self, start, allowed):
        """Hexes reachable from start over mutual road edges, staying inside allowed."""
        seen, todo = {start}, [start]
        while todo:
            q, r = todo.pop()
            m = self.mask[r, q]
            for d, (a, b) in self.neighbours(q, r):
                if m >> d & 1 and allowed[b, a] and self.mask[b, a] >> ((d + 3) % 6) & 1 and (a, b) not in seen:
                    seen.add((a, b)); todo.append((a, b))
        return seen


def write_text_atomic(path, text):
    """<name>.<pid>.tmp beside the target, then os.replace (as map_hex.write)."""
    tmp = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    try:
        tmp.write_text(text, encoding='utf-8')
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def split_legs(route, sites):
    """[(from, to, kind, hexes)] for a planned route; legs given by the plan, else cut at the stops."""
    hx = [tuple(h) for h in route['hexes']]
    if route.get('legs'):
        out, i = [], 0
        for leg in route['legs']:
            out.append((leg['from'], leg['to'], leg['kind'], hx[i:i + leg['hexes']]))
            i += leg['hexes'] - 1
        return out
    out, i = [], 0
    for a, b in zip(route['stops'], route['stops'][1:]):
        j = hx.index(tuple(sites[b]), i)
        out.append((a, b, 'land', hx[i:j + 1]))
        i = j
    return out


def trace_towns(g, mask):
    """Region pairs CAIME's TownRouteTracer registers: walks along road masks from every main-slot road
    hex, stopping at the first main-slot hex reached, or when leaving a foreign sprawl it entered."""
    pairs = collections.Counter()
    H, W = g.H, g.W
    starts = np.argwhere((mask > 0) & (g.slot == 1))
    for r0, q0 in starts:
        q0, r0 = int(q0), int(r0)
        origin = int(g.ids[r0, q0])
        seen = set()
        todo = [((q0, r0), d) for d in range(6) if mask[r0, q0] >> d & 1]
        while todo:
            (q, r), d = todo.pop()
            n = g.nb(q, r, d)
            if n is None:
                pairs['off-map'] += 1
                continue
            a, b = n
            dep, ent = g.spr(q, r), g.spr(a, b)
            if dep is not None and dep != origin and ent != dep:
                continue
            if g.slot[b, a] == 1:
                pairs[tuple(sorted((origin, int(g.ids[b, a]))))] += 1
                continue
            if n in seen:
                continue
            seen.add(n)
            back = (d + 3) % 6
            todo += [(n, e) for e in range(6) if mask[b, a] >> e & 1 and e != back]
    return pairs


def components(pairs):
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for p in pairs:
        if p == 'off-map':
            continue
        parent[find(p[0])] = find(p[1])
    return find


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--only-towns', nargs='+', metavar='REGION_KEY',
                    help='targeted mode after towns moved: re-lay only the legs that end at these towns or whose '
                         "laid road (previous report) now crosses another town's sprawl or a lake / a hex beside "
                         "one; no vanilla playable hex "
                         'is touched; the replaced roads are taken up (their edges not on a new or kept leg, then the '
                         'dead-end stubs left)')
    args = ap.parse_args()
    targeted = bool(args.only_towns)
    prev_legs = {}
    if targeted:
        prev = json.loads((A / 'analysis/roads-apply-report.json').read_text(encoding='utf-8'))
        prev_legs = {(l['route'], l['leg']): l for l in prev['legs']}
    F, V = map_hex.read(FINAL), map_hex.read(cw.VANILLA_MAP)
    names = F['land_regions'] + F['sea_regions']
    g = Grid(F['cells'], {'cells': V['cells'], 'names': V['land_regions'] + V['sea_regions']}, names)
    H, W = g.H, g.W
    region = {n: i for i, n in enumerate(names)}
    mask0 = g.mask.copy()
    report = {'workspace': str(A), 'legs': [], 'routes': {}, 'warnings': []}
    # Road hexes: plain land (kind 0) or the map's inland 'cliff' wilderness (kind 3 away from water,
    # which the pipeline uses with the impassable bit for closed mountains; opened below), never next
    # to water: CAIME counts land beside sea as cliff, and RoadsValidator rejects roads on cliff/beach.
    sea = g.kind == 1
    seaadj = np.zeros_like(sea)
    for par in (0, 1):
        for dq, dr in DIRS[par]:
            cols = np.arange(par, W, 2); nc = cols + dq
            ok = (nc >= 0) & (nc < W)
            cols, nc = cols[ok], nc[ok]
            if dr >= 0:
                seaadj[:H - dr, cols] |= sea[dr:, nc]
            else:
                seaadj[-dr:, cols] |= sea[:H + dr, nc]
    land_ok = ((g.kind == 0) | (g.kind == 3)) & ~seaadj
    valid_main = (g.slot == 1) & land_ok
    # inland lakes and the hexes beside them (targeted mode: a lake grown since, apply_real_lakes.py)
    lakeside = hex_dilate((g.kind == 1) & (g.ids == region['3k_main_sea_lake'])) if '3k_main_sea_lake' in region else np.zeros((H, W), bool)

    # vanilla road networks, by mutual edges among vanilla road hexes
    vcomp = {}
    for r, q in np.argwhere(g.vroad):
        if (q, r) not in vcomp:
            comp = frozenset(g.road_component((int(q), int(r)), g.vroad))
            for h in comp:
                vcomp[h] = comp

    def endpoint(stop, site, towns):
        """('vanilla', region, hexes) | ('town', region, hexes) | ('point', None, {hex})."""
        q, r = site
        reg = int(g.ids[r, q]) if g.slot[r, q] == 1 else region.get(towns.get(stop))
        if stop in VANILLA_LINKS:
            hexes = {h for h, c in vcomp.items() if not g.sprawl[h[1], h[0]]}
            return 'vanilla-any', None, hexes, (q, r)
        if reg is not None and names[reg].startswith('3k_') and g.vplay[r, q]:
            hexes = set()
            for rr, qq in np.argwhere((g.ids == reg) & (g.slot == 1) & g.vroad):
                hexes |= vcomp[(int(qq), int(rr))]
            hexes = {h for h in hexes if g.spr(*h) in (None, reg)}
            return 'vanilla', reg, hexes, (q, r)
        if reg is not None:
            rr, qq = np.nonzero((g.ids == reg) & valid_main)
            return 'town', reg, set(zip(qq.tolist(), rr.tolist())), (q, r)
        if not land_ok[r, q] or g.slot[r, q] == 1 or g.sprawl[r, q]:
            report['warnings'].append(f'waypoint {stop} {site} is not plain land; snapped')
            cand = np.argwhere(land_ok & ~g.sprawl & ~g.vplay)
            k = np.argmin((cand[:, 1] - q) ** 2 + (cand[:, 0] - r) ** 2)
            q, r = int(cand[k, 1]), int(cand[k, 0])
        return 'point', None, {(q, r)}, (q, r)

    # Town stops by region key, sited at the design's current town (towns may have moved since the plan)
    import plan_western_roads as pw
    design_town = {c['region']: tuple(c['town']) for c in json.loads(cw.DESIGN_PATH.read_text(encoding='utf-8'))['regions']}
    stop_keys = {n: f'tw3k_hist_{s}_capital' for n, s in pw.STEMS.items()}
    stop_keys.update(pw.TOWN_KEYS); stop_keys.update(pw.VANILLA)

    def current_site(stop, site, towns):
        key = towns.get(stop) or stop_keys.get(stop)
        if key in design_town:
            return design_town[key]
        if key in region and key.startswith('3k_'):
            rr, qq = np.nonzero((g.ids == region[key]) & (g.slot == 1))
            return int(round(qq.mean())), int(round(rr.mean()))
        return site
    missing = [k for k in (args.only_towns or []) if k not in region]
    if missing:
        raise SystemExit(f'Unknown region keys {missing}')
    only = {region[k] for k in (args.only_towns or [])}

    hub = {}                                    # town region -> main-slot hex the roads use there
    new_links = []                              # (hex, hex) pairs to link, in order
    route_hexes = {}                            # route name -> set of hexes laid for it
    route_info = {}

    def lay(rname, a_spec, b_spec, plan_seg, leg_name):
        ka, ra, sa, sitea = a_spec
        kb, rb, tb, siteb = b_spec
        if ka == 'vanilla' and kb == 'vanilla' and sa & tb:
            report['legs'].append({'route': rname, 'leg': leg_name, 'status': 'vanilla road already joins both towns', 'hexes': 0})
            return []
        # The plan's corridor: its hexes and their neighbours. plan_*_roads.py searched with box-relative
        # columns, so where the box began on an odd column the plan steps by the wrong column parity
        # and its consecutive hexes are not always neighbours; the corridor absorbs that.
        plan = set(plan_seg)
        pts = list(plan_seg) + [sitea, siteb]
        q0 = max(min(p[0] for p in pts) - MARGIN, 0); q1 = min(max(p[0] for p in pts) + MARGIN, W - 1)
        r0 = max(min(p[1] for p in pts) - MARGIN, 0); r1 = min(max(p[1] for p in pts) + MARGIN, H - 1)
        if ka == 'vanilla-any':                 # widen the box to the nearest vanilla roads
            near = sorted(sa, key=lambda h: (h[0] - sitea[0]) ** 2 + (h[1] - sitea[1]) ** 2)[:1]
            q0 = max(min(q0, near[0][0] - MARGIN), 0); q1 = min(max(q1, near[0][0] + MARGIN), W - 1)
            r0 = max(min(r0, near[0][1] - MARGIN), 0); r1 = min(max(r1, near[0][1] + MARGIN), H - 1)
        inbox = lambda h: q0 <= h[0] <= q1 and r0 <= h[1] <= r1
        own = {ra, rb} - {None}

        def phase_of(h, phase):
            """Tracer sprawl rule along the leg (phase 0: still in ra's sprawl, 1: open land, 2: in rb's
            sprawl): ra's sprawl only as a prefix, rb's only as a suffix, no other town's sprawl."""
            s = g.spr(*h)
            if s is None:
                return 1 if phase in (0, 1) else None
            if s not in own:
                return None
            if s == rb:
                return 2
            return 0 if phase == 0 else None

        def step_cost(h):
            q, r = h
            if not land_ok[r, q]:
                return None
            if g.vplay[r, q]:
                if g.mask[r, q] or ka == 'vanilla-any' or targeted:
                    return None
                return VPLAY_OPEN_COST if (g.imp[r, q] or g.kind[r, q]) else VPLAY_COST
            return (1.0 if h in plan else OFF_PLAN) + (IMPASSABLE_EXTRA if g.imp[r, q] or g.kind[r, q] else 0.0)

        # sources
        dist, prev, heap = {}, {}, []
        for h in sa:
            if not inbox(h):
                continue
            if ka == 'town':
                c0 = 0.0 if hub.get(ra, sitea) == h else HUB_PENALTY
            else:
                c0 = 0.0
            ph = 0 if (ra is not None and g.spr(*h) == ra) else 1
            key = (h, ph)
            if c0 < dist.get(key, math.inf):
                dist[key] = c0; heapq.heappush(heap, (c0, h, ph))
        targets = {h for h in tb if inbox(h)}
        if kb == 'town':
            want = hub.get(rb, siteb)
        gq, gr = siteb
        found = None
        while heap:
            d, h, ph = heapq.heappop(heap)
            if d > dist.get((h, ph), math.inf):
                continue
            if h in targets:
                found = (h, ph); break
            for _, n in g.neighbours(*h):
                if not inbox(n) or n in sa:
                    continue
                q, r = n
                if g.slot[r, q] == 1 and n not in targets:
                    continue
                if kb in ('vanilla', 'vanilla-any') and n in targets:
                    c = 0.0
                    nph = 1
                else:
                    c = step_cost(n)
                    if c is None:
                        continue
                    nph = phase_of(n, ph)
                    if nph is None:
                        continue
                    if kb == 'town' and n in targets and n != want:
                        c += HUB_PENALTY
                nd = d + c
                if nd < dist.get((n, nph), math.inf):
                    dist[(n, nph)] = nd; prev[(n, nph)] = (h, ph)
                    heapq.heappush(heap, (nd, n, nph))
        if found is None:
            raise SystemExit(f'No road path for {rname} {leg_name}')
        path = [found]
        while path[-1] in prev:
            path.append(prev[path[-1]])
        path = [h for h, _ in reversed(path)]
        if ka == 'town':
            hub.setdefault(ra, path[0])
        if kb == 'town':
            hub.setdefault(rb, path[-1])
        on_plan = sum(h in plan for h in path)   # within the corridor
        report['legs'].append({'route': rname, 'leg': leg_name, 'status': 'laid', 'hexes': len(path),
                               'new_road_hexes': sum(not g.mask[h[1], h[0]] for h in path),
                               'off_plan_hexes': len(path) - on_plan, 'plan_hexes': len(plan_seg),
                               'vanilla_playable_hexes': sum(bool(g.vplay[h[1], h[0]]) for h in path),
                               'from_hex': list(path[0]), 'to_hex': list(path[-1]), 'path': [list(h) for h in path]})
        return path

    replaced = {}                               # targeted mode: (route, leg) -> the old laid path
    for plan_path in PLANS:
        P = json.loads(plan_path.read_text(encoding='utf-8'))
        towns = P.get('towns', {})
        plan_sites = {k: tuple(v) for k, v in P['sites'].items()}
        sites = {k: tuple(current_site(k, v, towns)) for k, v in plan_sites.items()}
        for rname, route in P['routes'].items():
            legs = split_legs(route, plan_sites)
            hexes = set()
            sections = [[]]                       # towns per land section (sea legs split a route)
            first = legs[0][0]
            seq = []
            if first in VANILLA_LINKS:            # join the route's open end to the vanilla network
                seq.append((endpoint(first, sites[first], towns), endpoint(first + '#', sites[first], towns), [], f'vanilla->{first}'))

            def add_town(stop):
                e = endpoint(stop, sites[stop], towns) if stop not in VANILLA_LINKS else None
                if e and e[0] in ('town', 'vanilla') and (stop, e[1]) not in sections[-1]:
                    sections[-1].append((stop, e[1]))
            for a, b, kind, seg in legs:
                if kind != 'land':
                    report['legs'].append({'route': rname, 'leg': f'{a}->{b}', 'status': 'sea leg: no road', 'hexes': 0})
                    if sections[-1]:
                        sections.append([])
                    continue
                add_town(a); add_town(b)
                ea = endpoint(a + '#' if a in VANILLA_LINKS else a, sites[a], towns)
                seq.append((ea, endpoint(b, sites[b], towns), seg, f'{a}->{b}'))
            for a_spec, b_spec, seg, leg_name in seq:
                if targeted:
                    old = prev_legs.get((rname, leg_name), {}).get('path')
                    ends = {a_spec[1], b_spec[1]} - {None}
                    crosses = bool(old) and any(g.sprawl[r, q] and int(g.ids[r, q]) not in ends for q, r in old)
                    # a lake grown since (apply_real_lakes.py) flooded the old road or now borders it (river
                    # fords are crossed by design: apply_rivers.py)
                    flooded = bool(old) and any(lakeside[r, q] and not g.vplay[r, q] for q, r in old)
                    if not (ends & only or crosses or flooded):
                        continue
                    replaced[(rname, leg_name)] = [tuple(h) for h in old or []]
                path = lay(rname, a_spec, b_spec, seg, leg_name)
                hexes |= set(path)
                new_links += list(zip(path, path[1:]))
            route_hexes[rname] = hexes
            route_info[rname] = {'stops': route['stops'], 'sections': [x for x in sections if x],
                                 'plan': plan_path.stem.split('-')[0]}

    # ---- write the masks: chains first, then joins with existing roads
    mask = g.mask.copy()

    def link(a, b):
        d = g.direction(a, b)
        mask[a[1], a[0]] |= 1 << d
        mask[b[1], b[0]] |= 1 << ((d + 3) % 6)
    for a, b in new_links:
        link(a, b)
    all_route = set().union(*route_hexes.values())

    def within(a, b, depth):
        seen, frontier = {a}, [a]
        for _ in range(depth):
            nxt = []
            for q, r in frontier:
                for d, n in g.neighbours(q, r):
                    if mask[r, q] >> d & 1 and n not in seen:
                        if n == b:
                            return True
                        seen.add(n); nxt.append(n)
            frontier = nxt
        return False

    joins = []
    for rname, hexes in route_hexes.items():
        for p in sorted(hexes, key=lambda h: (h[1], h[0])):
            q, r = p
            if g.vplay[r, q]:
                continue
            for d, n in g.neighbours(q, r):
                a, b = n
                if not mask[b, a] or mask[r, q] >> d & 1 or n in hexes:
                    continue
                if g.vplay[b, a] or not land_ok[b, a] or ((g.imp[b, a] or g.kind[b, a]) and n not in all_route):
                    continue
                if within(p, n, JOIN_DEPTH):
                    continue
                link(p, n)
                joins.append({'route': rname, 'hex': list(p), 'joined': list(n),
                              'joined_kind': 'route' if n in all_route else ('vanilla road' if g.vroad[b, a] else 'existing non-vanilla road')})

    # targeted mode: the replaced roads are taken up. First every road edge of a replaced leg that is
    # not an edge of a new or kept leg or of this run's joins (an old leg still joined to the network at
    # both ends would otherwise stay as a through-road past the moved town); then the dead-end stubs
    # left (not on a new path, not a main slot, not vanilla playable, not on a kept leg's road), back
    # to the first junction.
    pruned = set()
    cleared_edges = []
    if targeted:
        def leg_edges(path):
            out = set()
            for a, b in zip(path, path[1:]):
                a, b = tuple(a), tuple(b)
                if a != b:
                    out.add((min(a, b), max(a, b)))
            return out
        kept_legs = [l for k, l in sorted(prev_legs.items()) if k not in replaced]
        keep_edges = set().union(*(leg_edges(l.get('path', [])) for l in kept_legs)) | \
            {(min(a, b), max(a, b)) for a, b in new_links} | \
            {(min(tuple(j['hex']), tuple(j['joined'])), max(tuple(j['hex']), tuple(j['joined']))) for j in joins}
        old_edges = set().union(*(leg_edges(p) for p in replaced.values())) if replaced else set()
        for a, b in sorted(old_edges - keep_edges):
            if g.vplay[a[1], a[0]] or g.vplay[b[1], b[0]]:
                continue
            try:
                d = g.direction(a, b)
            except ValueError:
                continue
            if mask[a[1], a[0]] >> d & 1 and mask[b[1], b[0]] >> ((d + 3) % 6) & 1:
                mask[a[1], a[0]] &= np.uint8(~(1 << d) & 0xFF)
                mask[b[1], b[0]] &= np.uint8(~(1 << ((d + 3) % 6)) & 0xFF)
                cleared_edges.append([list(a), list(b)])
        kept = {tuple(h) for l in kept_legs for h in l.get('path', [])}
        cand = {h for path in replaced.values() for h in path} - all_route - kept
        cand = {h for h in cand if g.slot[h[1], h[0]] != 1 and not g.vplay[h[1], h[0]]}
        pruned = {h for h in cand if mask0[h[1], h[0]] and not mask[h[1], h[0]]}   # left bare by the edge clearing
        changed_ = True
        while changed_:
            changed_ = False
            for h in sorted(cand - pruned):
                q, r = h
                nbrs = [(d, n) for d, n in g.neighbours(q, r) if mask[r, q] >> d & 1]
                if len(nbrs) > 1 or any(g.vplay[n[1], n[0]] for _, n in nbrs):
                    continue
                for d, n in nbrs:
                    mask[n[1], n[0]] &= np.uint8(~(1 << ((d + 3) % 6)) & 0xFF)
                mask[r, q] = 0
                pruned.add(h); changed_ = True
    report['pruned_stub_hexes'] = len(pruned)
    report['cleared_old_edges'] = cleared_edges

    cells = g.cells.copy()
    cells[:, :, 3] = (cells[:, :, 3] & 0x81) | (mask << 1)
    # open the route: impassable bit off, inland 'cliff' back to plain land
    opened = sorted(h for h in all_route if (g.imp[h[1], h[0]] or g.kind[h[1], h[0]] == 3))
    for q, r in opened:
        cells[r, q, 2] &= ~np.uint8(8)
        cells[r, q, 0] &= ~np.uint8(3)
    # Fragments: a road cut through an impassable band leaves slivers that CAIME's Impassable and Town
    # Sprawl validators reject (isolated impassable hexes; impassable areas near or bordering a sprawl).
    # Open every closed component (impassable, or inland cliff) under FRAGMENT_MAX hexes touching a route.
    closed = (((cells[:, :, 2] & 8) != 0) | (((cells[:, :, 0] & 3) == 3) & ~seaadj)) & ((cells[:, :, 0] & 3) != 1)
    seen, fragments = set(), []
    for p in sorted(all_route, key=lambda h: (h[1], h[0])):
        for _, s in g.neighbours(*p):
            if not closed[s[1], s[0]] or s in seen:
                continue
            comp, todo = {s}, [s]
            while todo and len(comp) < FRAGMENT_MAX:
                for _, n in g.neighbours(*todo.pop()):
                    if closed[n[1], n[0]] and n not in comp:
                        comp.add(n); todo.append(n)
            seen |= comp
            if len(comp) < FRAGMENT_MAX and not (targeted and any(g.vplay[h[1], h[0]] for h in comp)):
                fragments.append(sorted(comp))
    frag_hexes = sorted({h for f in fragments for h in f})
    for q, r in frag_hexes:
        cells[r, q, 2] &= ~np.uint8(8)
        if g.kind[r, q] == 3:
            cells[r, q, 0] &= ~np.uint8(3)
    report['fragments_opened'] = {'max_size': FRAGMENT_MAX, 'components': len(fragments), 'hexes': len(frag_hexes),
                                  'vanilla_playable_hexes': sum(bool(g.vplay[r, q]) for q, r in frag_hexes),
                                  'sizes': sorted(len(f) for f in fragments)}
    # Sprawl clearance (CAIME SprawlValidator): within 2 hexes of a sprawl blob every hazard area
    # (impassable, river, beach/cliff) must touch the blob, and at most one may touch it. A road cut
    # through a band breaks that; for blobs within SPRAWL_REACH of a route that passed before the roads,
    # open the offending areas' hexes inside that 2-hex outline (keeping the largest touching area).
    river = (cells[:, :, 4] & 63) != 0
    hard = river | ((cells[:, :, 0] & 3) == 2) | (((cells[:, :, 0] & 3) == 3) & seaadj)

    def hazard_labels(arr):
        """CAIME SprawlValidator hazard areas (impassable, river, beach, cliff) labelled by hex
        adjacency: each hex covers two rows of a doubled grid, where 4-connectivity is hex adjacency."""
        hz = ((arr[:, :, 2] & 8) != 0) | river | ((arr[:, :, 0] & 3) >= 2)
        cols = np.arange(W)
        rows = 2 * np.arange(H)[:, None] + (cols[None, :] % 2)
        grid = np.zeros((2 * H + 1, W), bool)
        grid[rows, cols[None, :]] = hz
        grid[rows + 1, cols[None, :]] = hz
        lab, _ = ndimage.label(grid)
        out = lab[rows, cols[None, :]]
        return out, np.bincount(out.ravel())

    def rings(start, depth):
        seen, frontier, out = set(start), list(start), []
        for _ in range(depth):
            nxt = []
            for h in frontier:
                for _, n in g.neighbours(*h):
                    if n not in seen:
                        seen.add(n); nxt.append(n)
            out.append(nxt); frontier = nxt
        return out
    near_route = set(all_route).union(*rings(all_route, SPRAWL_REACH))
    blobs, in_blob = [], set()
    for h in sorted(near_route):
        if g.sprawl[h[1], h[0]] and h not in in_blob:
            blob, todo = {h}, [h]
            while todo:
                for _, n in g.neighbours(*todo.pop()):
                    if g.sprawl[n[1], n[0]] and n not in blob:
                        blob.add(n); todo.append(n)
            in_blob |= blob; blobs.append(blob)
    cleared, unresolved = set(), []
    lab0, size0 = hazard_labels(g.cells)
    dirty = True
    for blob in blobs:
        r1, r2 = rings(blob, 2)
        outline = set(r1) | set(r2)

        def state(lab, sizes):
            comp_of = {h: int(lab[h[1], h[0]]) for h in outline if lab[h[1], h[0]]}
            return comp_of, sizes, {comp_of[h] for h in r1 if h in comp_of}, set(comp_of.values())
        _, _, t0, p0 = state(lab0, size0)
        if (p0 - t0) or len(t0) > 1:            # already failing before the roads: not ours
            continue
        for _ in range(6):
            if dirty:
                lab1, size1 = hazard_labels(cells); dirty = False
            comp_of, size, touching, present = state(lab1, size1)
            drop = present - touching
            if len(touching) > 1:
                drop |= touching - {max(touching, key=lambda c: size[c])}
            todo_open = [h for h in outline if comp_of.get(h) in drop]
            if not todo_open:
                break
            blocked_ = [h for h in todo_open if hard[h[1], h[0]] or g.vplay[h[1], h[0]]]
            for q, r in todo_open:
                if hard[r, q] or g.vplay[r, q]:
                    continue
                cells[r, q, 2] &= ~np.uint8(8)
                cells[r, q, 0] &= ~np.uint8(3)
                cleared.add((q, r)); dirty = True
            if blocked_ and len(blocked_) == len(todo_open):
                unresolved.append({'blob_near': list(min(blob)), 'hexes': [list(h) for h in blocked_[:5]]})
                break
    report['sprawl_clearance'] = {'reach': SPRAWL_REACH, 'blobs_checked': len(blobs), 'hexes_opened': len(cleared),
                                  'unresolved (river/coast/vanilla)': unresolved}
    opened = sorted(set(opened) | set(frag_hexes) | cleared)
    opened_imp = sum(bool(g.imp[r, q]) for q, r in opened)
    opened_cliff = sum(bool(g.kind[r, q] == 3) for q, r in opened)

    # ---- checks
    changed = np.argwhere((cells != g.cells).any(axis=2))
    bad_mutual = 0
    for r, q in np.argwhere(mask != mask0):          # every edge bit set here has its partner bit
        added = mask[r, q] & ~mask0[r, q]
        for d, n in g.neighbours(int(q), int(r)):
            if added >> d & 1 and not mask[n[1], n[0]] >> ((d + 3) % 6) & 1:
                bad_mutual += 1
        removed = mask0[r, q] & ~mask[r, q]
        for d, n in g.neighbours(int(q), int(r)):
            if removed >> d & 1 and mask[n[1], n[0]] >> ((d + 3) % 6) & 1:
                bad_mutual += 1                        # a bit removed on one side only
    road = mask > 0
    newroad = road & (mask0 == 0)
    touched = mask != mask0
    imp1 = (cells[:, :, 2] & 8) != 0
    vchg = [(int(q), int(r)) for r, q in changed if g.vplay[r, q]]
    op_mask = np.zeros((H, W), bool)
    for q, r in opened:
        op_mask[r, q] = True
    other_bytes = (np.delete(cells, [0, 2, 3], axis=2) != np.delete(g.cells, [0, 2, 3], axis=2)).any() or \
        ((cells[:, :, 0] ^ g.cells[:, :, 0]) & np.where(op_mask, 0xFC, 0xFF).astype(np.uint8)).any() or \
        ((cells[:, :, 2] ^ g.cells[:, :, 2]) & ~np.uint8(8)).any() or ((cells[:, :, 3] ^ g.cells[:, :, 3]) & 0x81).any()
    kind1 = cells[:, :, 0] & 3
    report['counts'] = {
        'road_hexes_before': int((mask0 > 0).sum()), 'road_hexes_after': int(road.sum()),
        'new_road_hexes': int(newroad.sum()), 'route_hexes_laid': len(all_route),
        'existing_road_hexes_reused': int(sum(bool(mask0[r, q]) for q, r in all_route)),
        'hexes_with_mask_changed': int(touched.sum()), 'hexes_opened': len(opened), 'impassable_bit_cleared': opened_imp,
        'inland_cliff_to_land': opened_cliff,
        'joins': len(joins), 'joins_by_kind': dict(collections.Counter(j['joined_kind'] for j in joins)),
        'hexes_changed': int(len(changed))}
    report['checks'] = {
        'non_mutual_edges_on_changed_hexes': bad_mutual,
        'changed_road_hexes_on_sea_beach_cliff': int((touched & road & ((kind1 != 0) | seaadj)).sum()),
        'changed_road_hexes_impassable': int((touched & road & imp1).sum()),
        'other_fields_changed': bool(other_bytes)}
    report['vanilla_playable_changes'] = [
        {'hex': [q, r], 'region': names[g.ids[r, q]], 'was_road': bool(mask0[r, q]),
         'mask_before': int(mask0[r, q]), 'mask_after': int(mask[r, q]),
         'opened': bool(g.imp[r, q] or g.kind[r, q] == 3)} for q, r in vchg]
    report['joins'] = joins
    # SDK area index (hex_components.region_areas): opening impassable / cliff hexes splits and adds
    # traversal-class areas; a region past SDK_AREA_LIMIT hangs MapDataBuilder (as in apply_rivers.py)
    areas0, areas1 = region_areas(g.ids, g.cells), region_areas(g.ids, cells)
    report['sdk_areas'] = {'limit': SDK_AREA_LIMIT,
                           'top': [(names[k] if k < len(names) else str(k), v) for k, v in areas1.most_common(8)],
                           'overflow': {names[k] if k < len(names) else str(k): [areas0.get(k, 0), v]
                                        for k, v in sorted(areas1.items()) if v > SDK_AREA_LIMIT}}

    # town connectivity as the pathfinding export traces it
    before, after = trace_towns(g, mask0), trace_towns(g, mask)
    find = components(after)
    vanilla_roots = {find(i) for i, n in enumerate(names) if n.startswith('3k_')}
    for rname, info in route_info.items():
        secs = []
        for sec in info['sections']:
            roots = {find(rg) for _, rg in sec}
            direct = [f'{a}-{b}: {tuple(sorted((ra, rb))) in after}' for (a, ra), (b, rb) in zip(sec, sec[1:])]
            secs.append({'towns': [s for s, _ in sec], 'connected': len(roots) == 1,
                         'joins_vanilla_network': bool(roots & vanilla_roots),
                         'consecutive_towns_directly_traced': direct})
        report['routes'][rname] = {'sections': secs, 'connected': all(x['connected'] for x in secs),
                                   'hexes': len(route_hexes[rname])}
    report['traced_town_pairs'] = {'before': len([p for p in before if p != 'off-map' and p[0] != p[1]]),
                                   'after': len([p for p in after if p != 'off-map' and p[0] != p[1]]),
                                   'off_map_masks': after.get('off-map', 0)}
    # the plan's connected networks: Western Regions and each frontier road group
    groups = collections.defaultdict(set)
    for rname, info in route_info.items():
        key = 'Western Regions' if info['plan'] == 'western' else rname.split('·')[0]
        for i, sec in enumerate(info['sections']):
            groups[key + (f' section {i + 1}' if len(info['sections']) > 1 else '')].update(rg for _, rg in sec)
    report['networks'] = {k: {'connected': len({find(x) for x in v}) == 1,
                              'joins_vanilla_network': bool({find(x) for x in v} & vanilla_roots)} for k, v in groups.items()}
    report['warnings'] = sorted(set(report['warnings']))

    failed = bad_mutual or report['checks']['changed_road_hexes_on_sea_beach_cliff'] or \
        report['checks']['changed_road_hexes_impassable'] or other_bytes or \
        not all(v['connected'] for v in report['routes'].values()) or after.get('off-map', 0) or (targeted and vchg) or \
        report['sdk_areas']['overflow']
    summary = {k: v for k, v in report.items() if k in ('counts', 'checks', 'sdk_areas', 'fragments_opened', 'sprawl_clearance', 'traced_town_pairs', 'networks', 'warnings')}
    summary['routes_connected'] = {k: v['connected'] for k, v in report['routes'].items()}
    summary['vanilla_playable_changes'] = len(vchg)
    if targeted:
        summary['relaid_legs'] = sorted(f'{a} {b}' for a, b in replaced)
        summary['pruned_stub_hexes'] = len(pruned)
        summary['cleared_old_edges'] = len(cleared_edges)
        # keep the full leg list for later runs: previous legs with the re-laid ones replaced
        new = {(l['route'], l['leg']): l for l in report['legs'] if l['status'] == 'laid'}
        report['legs'] = [new.get(k, l) for k, l in prev_legs.items()] + [l for k, l in new.items() if k not in prev_legs]   # new spurs too
        report['targeted_towns'] = args.only_towns
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    OUT.mkdir(parents=True, exist_ok=True)
    preview(g, cells, mask0, mask, route_hexes, opened, PLANS)
    write_text_atomic(OUT / 'roads-apply-report.json', json.dumps(report, ensure_ascii=False, indent=1))
    if failed:
        raise SystemExit(f'Road apply fails its checks; nothing written (see {OUT / "roads-apply-report.json"})')
    if not args.apply:
        print('Dry run only; add --apply to write map.hex.')
        return
    back = A / f'backups/roads-{time.strftime("%Y%m%d-%H%M%S")}'
    back.mkdir(parents=True)
    shutil.copy2(FINAL, back / FINAL.name)
    map_hex.write(dict(F, cells=cells), FINAL)
    check = map_hex.read(FINAL)['cells']
    assert np.array_equal(check, cells)
    report['backup'] = str(back)
    (A / 'analysis').mkdir(exist_ok=True)
    write_text_atomic(A / 'analysis/roads-apply-report.json', json.dumps(report, ensure_ascii=False, indent=1))
    print('Roads written; backup at', back)


def preview(g, cells, mask0, mask, route_hexes, opened, plans):
    """Hex maps of the road layer: water, playable land, impassable land (after the opening), vanilla
    roads, other roads already on the map, the new route roads; road edges drawn centre to centre."""
    water = (cells[:, :, 0] & 3) == 1
    imp = (cells[:, :, 2] & 8) != 0
    font = ImageFont.truetype(tw3k_paths.font('msyh.ttc'), 13)
    sites = {}
    for p in plans:
        sites.update(json.loads(p.read_text(encoding='utf-8'))['sites'])
    route_set = set().union(*route_hexes.values())
    opened = set(opened)
    ys = cw.HEX_DZ / cw.HEX_DX
    views = {'overview': (0, g.W, 0, g.H, 0.6),
             'western-regions': (300, 1080, 830, 1140, 1.6),
             'north-mobei': (700, 1400, 940, 1380, 1.8),
             'northeast-korea-wa': (1520, 1830, 760, 1160, 2.6),
             'south-shendu-jiaozhou': (700, 1230, 360, 660, 2.4)}
    for name, (q0, q1, r0, r1, s) in views.items():
        img = np.full((r1 - r0, q1 - q0, 3), (205, 195, 160), np.uint8)
        img[imp[r0:r1, q0:q1]] = (165, 155, 128)
        img[g.vplay[r0:r1, q0:q1] & ~water[r0:r1, q0:q1]] = (185, 200, 160)
        img[water[r0:r1, q0:q1]] = (90, 130, 170)
        op = np.zeros((g.H, g.W), bool)
        for q, r in opened:
            op[r, q] = True
        img[op[r0:r1, q0:q1]] = (230, 175, 175)
        im = Image.fromarray(img[::-1]).resize((int((q1 - q0) * s), int((r1 - r0) * s * ys)), Image.Resampling.NEAREST)
        d = ImageDraw.Draw(im)
        px = lambda q, r: ((q - q0 + .5) * s, ((r1 - 1 - r) + .5 - (q % 2) * .5) * s * ys)
        width = 1 if s < 1 else 2
        for layer in ('vanilla', 'other', 'new'):
            for r, q in np.argwhere(mask[r0:r1, q0:q1] > 0):
                q, r = int(q) + q0, int(r) + r0
                is_new = (q, r) in route_set
                kind = 'new' if is_new else ('vanilla' if g.vroad[r, q] else 'other')
                if kind != layer:
                    continue
                colour = {'vanilla': (60, 60, 60), 'other': (235, 130, 20), 'new': (200, 20, 30)}[kind]
                m = mask[r, q]
                drew = False
                for dd in range(6):
                    if m >> dd & 1:
                        n = g.nb(q, r, dd)
                        if n:
                            x0, y0 = px(q, r); x1, y1 = px(*n)
                            d.line([(x0, y0), ((x0 + x1) / 2, (y0 + y1) / 2)], fill=colour, width=width + (1 if kind == 'new' else 0))
                            drew = True
                if not drew or s < 1:
                    x, y = px(q, r)
                    d.point((x, y), fill=colour)
        if s >= 1:
            for nm, (q, r) in sites.items():
                if q0 <= q < q1 and r0 <= r < r1:
                    x, y = px(q, r)
                    d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(255, 255, 255), outline=(0, 0, 0))
                    d.text((x + 5, y - 9), nm, font=font, fill=(0, 0, 0), stroke_width=2, stroke_fill=(255, 255, 255))
        d.text((8, 6), 'red: new roads  orange: other non-vanilla roads  dark: vanilla roads  pink: impassable opened  green: vanilla playable land',
               font=font, fill=(0, 0, 0), stroke_width=2, stroke_fill=(255, 255, 255))
        im.save(OUT / f'roads-{name}.png')


if __name__ == '__main__':
    main()
