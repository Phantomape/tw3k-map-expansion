"""Window water support from the production water surfaces and quantized LF.

No map/river outlines are invented here. Frontier quads/heights come from
river_water_props.surface; native water uses the renderer's mesh/pivot/instance
conventions; the ocean plane comes from extend_campaign_ocean_background's rule.
All boxes are north-first LF/blend pixel (y0,y1,x0,x1), arrays/levels south-up.
This is water versus LF, not the full coarse/fine terrain renderer z-buffer.
"""
import ast
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

import numpy as np


class WaterCoverageError(RuntimeError):
    pass


def _code_sha(name):
    return hashlib.sha256(Path(__file__).with_name(name).read_text(encoding='utf-8').encode('utf-8')).hexdigest()


def _array_sha(a):
    h = hashlib.sha256(repr((a.shape, a.dtype.str)).encode('ascii'))
    for row in a:
        h.update(np.ascontiguousarray(row).tobytes())
    return h.hexdigest()


def _literal(name, source):
    tree = ast.parse(Path(__file__).with_name(source).read_text(encoding='utf-8'))
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets):
            return ast.literal_eval(n.value)
    raise ValueError(f'{name} not found in production {source}')


def _box(box, shape):
    y0, y1, x0, x1 = (int(v) for v in box)
    if not 0 <= y0 <= y1 <= shape[0] or not 0 <= x0 <= x1 <= shape[1]:
        raise ValueError(f'Water window {box} outside {shape}')
    return y0, y1, x0, x1


class ProductionBackground:
    """The same future ocean quad as extend_campaign_ocean_background, not a guessed plane."""
    def __init__(self, shape):
        import campaign_world as cw
        self.shape = tuple(shape)
        self.level = float(np.float32(_literal('OCEAN_Y', 'extend_campaign_ocean_background.py'))) + float(
            _literal('SEA_LIFT', 'campaign_water_meshes.py'))
        self.bounds = (-32., cw.WORLD_WIDTH + 32., -32. * cw.ZS, (cw.RAW_MESH_HEIGHT + 32.) * cw.ZS)
        self.pitch = (cw.DX / 8., cw.DZ / 8.)
        self.provenance = {'kind': 'production-ocean-background-rule', 'level': self.level,
                           'world_bounds': list(self.bounds),
                           'producer_sha256': _code_sha('extend_campaign_ocean_background.py'),
                           'decode_sha256': _code_sha('campaign_water_meshes.py')}

    def read_heights(self, box):
        y0, y1, x0, x1 = _box(box, self.shape)
        x = np.arange(x0, x1, dtype=np.float64)[None, :] * self.pitch[0]
        z = (self.shape[0] - 1 - np.arange(y0, y1, dtype=np.float64))[:, None] * self.pitch[1]
        a, b, c, d = self.bounds
        return np.where((x >= a) & (x <= b) & (z >= c) & (z <= d), self.level, -np.inf).astype(np.float32)


class NativeWater:
    """Stream native/source meshes, never all water+land triangles at once.

    read(key)->immutable bytes; keys(prefix,suffix)->keys. frame_for_key(key)
    must explicitly return 'canvas' (already translated BASE) or 'vanilla'
    (game asset). Stale tw3k_water meshes are intentionally excluded: the new
    frontier plan is their sole authority. At most two decoded assets are cached.
    planned_lakes is a sequence of production prop dictionaries (model/matrix,
    e.g. HH.prop), in world space; their encoded instances are deduplicated.
    """
    def __init__(self, read, keys, shape, variant, *, frame_for_key, planned_lakes=None):
        import campaign_world as cw
        from props_bundle import embedded
        from translate_campaign_lake_instances import records
        self.read, self.shape, self.frame_for_key = read, tuple(shape), frame_for_key
        self.pitch = (cw.DX / 8., cw.DZ / 8.)
        self.assets, self.hashes, self._cache = [], {}, OrderedDict()
        self.shift_raw, self.shift_world = cw.NATIVE_SHIFT_RAW, cw.NATIVE_SHIFT_WORLD
        self.zs = cw.ZS
        self.sea_lift = float(_literal('SEA_LIFT', 'campaign_water_meshes.py'))
        pre = f'terrain/campaigns/{variant}/'
        sea = sorted(keys(pre + 'global_meshes/sea_mesh_', '.rigid_model_v2'))
        if not sea:
            raise WaterCoverageError('Native water source contains no sea meshes')
        for key in sea:
            self._add(key, 'sea')
        rivers = set(keys(pre + 'models/river_', '.wsmodel'))
        props_key = pre + 'global_props.bin'
        data = read(props_key)
        self.hashes[props_key] = hashlib.sha256(data).hexdigest()
        props = embedded(data)[0]
        lake_rows = []
        for key, blob in props.items():
            if not key.endswith('.bin') or key.endswith('/bmd_objects.bin'):
                continue
            if pre.encode() + b'models/' not in blob and b'/lakes/' not in blob:
                continue
            for row in records(blob):
                model = row['model']
                if model.startswith(pre + 'models/') and '/tw3k_water_' not in model:
                    rivers.add(model)
                if '/props/lakes/' in model.lower():
                    matrix = np.asarray(row['matrix'], np.float64).copy()
                    if self._frame(props_key) == 'vanilla':
                        # translate_campaign_lake_instances writes each translated
                        # position back through struct('<f') before rendering.
                        matrix[9] = np.float32(matrix[9] + self.shift_world[0])
                        matrix[11] = np.float32(matrix[11] + self.shift_world[1])
                    lake_rows.append((model, matrix))
        for model in sorted(rivers):
            key = self._geometry(model)
            self._add(key, 'river')
        lake_instances = set()
        for model, matrix in lake_rows:
            key = self._geometry(model)
            self._add(key, 'lake', matrix)
            lake_instances.add((key.lower(), np.asarray(matrix, '<f4').tobytes()))
        planned = []
        for prop in planned_lakes or ():
            model = str(prop['model'])
            # hehuang_lake_prop.run writes exactly these twelve float32s to
            # the leaf; using Python/float64 matrix values directly would
            # predict a different shoreline from the actual encoded instance.
            matrix32 = np.asarray(prop['matrix'], '<f4')
            if matrix32.shape != (12,) or not np.isfinite(matrix32).all():
                raise ValueError('Planned lake matrix must be twelve finite float32 values')
            key = self._geometry(model)
            identity = (key.lower(), matrix32.tobytes())
            duplicate = identity in lake_instances
            if not duplicate:
                self._add(key, 'lake', matrix32.astype(np.float64))
                lake_instances.add(identity)
            planned.append({'model': model, 'geometry': key, 'matrix_float32': matrix32.tolist(),
                            'matrix_sha256': hashlib.sha256(matrix32.tobytes()).hexdigest(),
                            'planned_input_sha256': hashlib.sha256(json.dumps(prop, sort_keys=True,
                                separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest(),
                            'geometry_sha256': self.hashes[key], 'deduplicated': duplicate})
        self.provenance = {'kind': 'native-source-water-meshes', 'assets_sha256': self.hashes,
                           'instances': len(self.assets), 'decode_sha256': _code_sha('campaign_water_meshes.py'),
                           'native_shift_raw': list(self.shift_raw), 'native_shift_world': list(self.shift_world),
                           'raw_z_to_world': self.zs, 'sea_lift': self.sea_lift,
                           'planned_lakes': planned,
                           'planned_lake_instances_added': sum(not p['deduplicated'] for p in planned),
                           'planned_lake_instances_deduplicated': sum(p['deduplicated'] for p in planned),
                           'planned_lake_producer_sha256': _code_sha('hehuang_lake_prop.py'),
                           'frames': {a['key']: a['frame'] for a in self.assets}}

    def _frame(self, key):
        frame = self.frame_for_key(key)
        if frame not in ('canvas', 'vanilla'):
            raise ValueError(f'Unknown source coordinate frame for {key}: {frame}')
        return frame

    def _geometry(self, model):
        raw = self.read(model)
        self.hashes[model] = hashlib.sha256(raw).hexdigest()
        key = ET.fromstring(raw.decode('utf-8', 'replace')).findtext('geometry')
        if not key:
            raise WaterCoverageError(f'No water geometry in {model}')
        try:
            self.read(key)
        except KeyError:
            key = key.lower()
        return key

    def _decode(self, desc, raw):
        from campaign_water_meshes import _mesh
        mp = struct.unpack_from('<I', raw, 152)[0]
        typ, _, _, vo, vc, io, _ = struct.unpack_from('<HH5I', raw, mp)
        kind = desc['kind']
        _, v, t, _ = _mesh(raw, 'f4' if kind == 'sea' or (io-vo)//vc == 48 else 'f2')
        if kind == 'sea':
            if typ != 101:
                raise WaterCoverageError(f'Unexpected native sea mesh type {typ}')
            if desc['frame'] == 'vanilla':
                v[:, 0] += self.shift_raw[0]; v[:, 2] += self.shift_raw[1]
            v[:, 2] *= self.zs
            v[:, 1] += self.sea_lift
        else:
            v += struct.unpack_from('<3f', raw, mp+628)
            if kind == 'lake':
                matrix = desc['matrix']
                v = v @ matrix[:9].reshape(3, 3) + matrix[9:]
            elif desc['frame'] == 'vanilla':
                v[:, 0] += self.shift_world[0]; v[:, 2] += self.shift_world[1]
        return v, t

    def _add(self, key, kind, matrix=None):
        raw = self.read(key)
        self.hashes[key] = hashlib.sha256(raw).hexdigest()
        desc = {'key': key, 'kind': kind, 'frame': self._frame(key), 'matrix': matrix}
        v, _ = self._decode(desc, raw)
        desc['bounds'] = (v[:, 0].min(), v[:, 0].max(), v[:, 2].min(), v[:, 2].max())
        self.assets.append(desc)

    def read_heights(self, box):
        from campaign_water_meshes import raster
        y0, y1, x0, x1 = _box(box, self.shape)
        out = np.full((y1-y0, x1-x0), -np.inf, np.float32)
        if not out.size:
            return out
        sy0 = self.shape[0]-y1
        bounds = (x0*self.pitch[0], (x1-1)*self.pitch[0], sy0*self.pitch[1], (self.shape[0]-1-y0)*self.pitch[1])
        for i, desc in enumerate(self.assets):
            a,b,c,d = desc['bounds']
            if b < bounds[0] or a > bounds[1] or d < bounds[2] or c > bounds[3]:
                continue
            if i not in self._cache:
                raw = self.read(desc['key'])
                if hashlib.sha256(raw).hexdigest() != self.hashes[desc['key']]:
                    raise WaterCoverageError(f'Water source changed: {desc["key"]}')
                self._cache[i] = self._decode(desc, raw)
                while len(self._cache) > 2:
                    self._cache.popitem(last=False)
            self._cache.move_to_end(i)
            v, triangles = self._cache[i]
            for start in range(0, len(triangles), 8192):
                tris = v[triangles[start:start+8192]].copy()
                hit = ((tris[...,0].max(1) >= bounds[0]) & (tris[...,0].min(1) <= bounds[1]) &
                       (tris[...,2].max(1) >= bounds[2]) & (tris[...,2].min(1) <= bounds[3]))
                tris = tris[hit]
                if not len(tris):
                    continue
                tris[...,0] /= self.pitch[0]; tris[...,2] /= self.pitch[1]
                # Existing renderer raster samples cell centres. Shift its origin
                # by half a pixel to sample the actual LF lattice vertices.
                z = raster(tris, x0-.5, sy0-.5, 1., x1-x0, y1-y0)
                np.maximum(out, z[::-1], out=out)
        return out


class WaterSupport:
    def __init__(self, lf, plan, *, native_water, background_water, provenance=None):
        import campaign_world as cw
        self.shape = tuple(lf.shape)
        if lf.dtype != np.uint16 or len(self.shape) != 2:
            raise ValueError('Water support requires quantized south-up uint16 LF')
        self.lf = lf.view(); self.lf.flags.writeable = False
        self.H = np.asarray(plan['H'])
        if self.shape != tuple(4*v for v in self.H.shape):
            raise ValueError('Production water plan and LF frames differ')
        self.quads = np.asarray(plan['river_quad']) | plan['lake_quad'] | plan['mouth_quad']
        if self.quads.shape != (self.H.shape[0]-1, self.H.shape[1]-1):
            raise ValueError('Production water quad grid has wrong shape')
        self.providers = {'native': native_water, 'ocean_background': background_water}
        self.level, self.norm, self.gain = cw.LEVEL, cw.NORM, cw.GAIN
        self.unsupported_queries = self.unsupported_pixels = 0
        self.provenance = dict(provenance or {},
            kind='production-water-triangles-versus-quantized-LF', shape=list(self.shape),
            coverage_status={key: 'complete' if p is not None else 'unsupported' for key,p in self.providers.items()},
            providers={key: getattr(p, 'provenance', None) for key,p in self.providers.items()},
            unsupported_queries=0, unsupported_pixels=0,
            lf_sha256=_array_sha(lf), plan_height_sha256=_array_sha(self.H),
            plan_quads_sha256=_array_sha(self.quads), producer_sha256=_code_sha('ground_water_support.py'),
            limitations=['LF comparison; coarse/fine rendered land-mesh z-buffer is not evaluated',
                         'later LF/water-rule changes invalidate this precomputed support'])

    def frontier_heights(self, box):
        """Production quad triangles, including build_model's exact diagonal choice."""
        y0,y1,x0,x1 = _box(box, self.shape)
        x = np.arange(x0,x1,dtype=np.float64)[None,:]/4-.5
        y = (self.shape[0]-1-np.arange(y0,y1,dtype=np.float64))[:,None]/4-.5
        ix, iy = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
        out = np.full((y1-y0,x1-x0), -np.inf, np.float64)
        for dy in (0,-1):
            for dx in (0,-1):
                qx,qy = ix+dx,iy+dy
                fx,fy = x-qx,y-qy
                valid = (qx>=0)&(qy>=0)&(qx<self.quads.shape[1])&(qy<self.quads.shape[0])&(fx>=0)&(fx<=1)&(fy>=0)&(fy<=1)
                qx,qy = np.clip(qx,0,self.quads.shape[1]-1),np.clip(qy,0,self.quads.shape[0]-1)
                valid &= self.quads[qy,qx]
                a,b,c,d = self.H[qy,qx],self.H[qy,qx+1],self.H[qy+1,qx],self.H[qy+1,qx+1]
                ad = np.abs(a-d) < np.abs(b-c)-1e-9
                h_ad = np.where(fy>=fx,a+(d-c)*fx+(c-a)*fy,a+(b-a)*fx+(d-b)*fy)
                h_bc = np.where(fx+fy<=1,a+(b-a)*fx+(c-a)*fy,b+c-d+(d-c)*fx+(d-b)*fy)
                np.maximum(out,np.where(valid,np.where(ad,h_ad,h_bc),-np.inf),out=out)
        return out.astype(np.float32)

    def read(self, box):
        y0,y1,x0,x1 = _box(box,self.shape)
        missing = [key for key,p in self.providers.items() if p is None]
        if missing:
            self.unsupported_queries += 1
            self.unsupported_pixels += (y1-y0)*(x1-x0)
            self.provenance['unsupported_queries'] = self.unsupported_queries
            self.provenance['unsupported_pixels'] = self.unsupported_pixels
            raise WaterCoverageError(f'Incomplete water coverage for {box}: {", ".join(missing)}')
        out = np.zeros((y1-y0,x1-x0),bool)
        for a in range(y0,y1,64):
            b = min(a+64,y1); crop=(a,b,x0,x1)
            water = self.frontier_heights(crop)
            for provider in self.providers.values():
                height = np.asarray(provider.read_heights(crop))
                if height.shape != water.shape or np.isnan(height).any() or np.isposinf(height).any():
                    raise WaterCoverageError('Water provider returned invalid height/coverage')
                np.maximum(water,height,out=water)
            raw = self.lf[self.shape[0]-b:self.shape[0]-a,x0:x1][::-1].astype(np.float64)
            ground = self.level+(raw/65535-self.norm)*self.gain
            out[a-y0:b-y0] = np.isfinite(water)&(water>ground)
        return out


def prepare(lf_uint16_south_up, river_level, cells, names, *, native_water=None,
            background_water=None, prepared_plan=None, provenance=None):
    """Build production river/lake surface once, before blend selection. No file writes.

    Supply NativeWater(read, keys,...frame_for_key) and ProductionBackground(shape)
    for complete support. Missing providers are rejected on read(), never treated
    as known dry land. prepared_plan must be river_water_props.surface's output.
    """
    if prepared_plan is None:
        import river_water_props as production
        prepared_plan = production.surface(lf_uint16_south_up,river_level,
            {'cells': cells,'land_regions': list(names),'sea_regions': []})
    meta = dict(provenance or {}, water_mesh_producer_sha256=_code_sha('river_water_mesh.py'),
                water_props_producer_sha256=_code_sha('river_water_props.py'),
                river_level_sha256=_array_sha(np.asarray(river_level)))
    return WaterSupport(lf_uint16_south_up,prepared_plan,native_water=native_water,
                        background_water=background_water,provenance=meta)
