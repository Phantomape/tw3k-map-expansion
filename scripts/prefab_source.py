"""Verified pre-prefab inputs, independent of the immutable ground-bake BASE.

An optional TW3K_PREFAB_SOURCE_MANIFEST binds original geometry global_props and
its authoring camera to one geometry checkpoint and current map/design. Without
that option, callers use the historical BASE/height-anchor paths unchanged.
This module only reads inputs; it never repairs or refreshes a snapshot.
"""
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re


INPUT_KEYS=('work/final-map/map.hex','work/current-region-design.json')


def sha256(data):return hashlib.sha256(data).hexdigest()


def _json(data):return json.loads(data.decode('utf-8-sig'))


def _hash(value,what):
    if not isinstance(value,str) or not re.fullmatch('[0-9a-f]{64}',value):
        raise ValueError(f'Invalid SHA256 for {what}')
    return value


def _read(record,what):
    path=Path(record['path'])
    if not path.is_absolute():raise ValueError(f'{what} path must be absolute')
    path=path.resolve(strict=True);data=path.read_bytes()
    expected=_hash(record['sha256'],what)
    if sha256(data)!=expected:raise RuntimeError(f'{what} checksum differs: {path}')
    return path,data


class PrefabSource:
    def __init__(self,workspace,variants,canvas,camera_size,manifest_path=None):
        self.workspace=Path(workspace);self.base=self.workspace/'terrain-before-polish'
        self.explicit=False;self._assets={};self._provenance=None
        option=manifest_path if manifest_path is not None else os.environ.get('TW3K_PREFAB_SOURCE_MANIFEST')
        if not option:return
        path=Path(option)
        if not path.is_absolute():raise ValueError('TW3K_PREFAB_SOURCE_MANIFEST must be an absolute path')
        path=path.resolve(strict=True);raw=path.read_bytes();data=_json(raw)
        if data.get('schema')!=1 or data.get('kind')!='pre-prefab-geometry-inputs':
            raise ValueError('Unsupported prefab source manifest schema/kind')
        if data.get('canvas')!=canvas:raise RuntimeError('Prefab source belongs to another canvas/frame')
        current={key:sha256((self.workspace/key).read_bytes()) for key in INPUT_KEYS}
        if data.get('inputs')!=current:raise RuntimeError('Prefab source belongs to another map/design')
        gr=data['geometry_source'];gp=Path(gr['manifest_path'])
        if not gp.is_absolute():raise ValueError('Geometry source manifest path must be absolute')
        gp=gp.resolve(strict=True);graw=gp.read_bytes()
        if sha256(graw)!=_hash(gr['manifest_sha256'],'geometry manifest'):
            raise RuntimeError('Original geometry manifest checksum differs')
        geometry=_json(graw)
        if geometry.get('boundary')!='geometry' or geometry.get('inputs')!=current:
            raise RuntimeError('Original geometry checkpoint belongs to another map/design or phase')
        if set(data['variants'])!=set(variants):raise RuntimeError('Prefab source must cover exactly the processed terrain variants')
        provenance={'manifest_path':str(path),'manifest_sha256':sha256(raw),
            'geometry_manifest_path':str(gp),'geometry_manifest_sha256':sha256(graw),
            'inputs':current,'canvas':canvas,'variants':{}}
        from PIL import Image
        for variant in variants:
            item=data['variants'][variant];blobs={};records={}
            for name,key in (('global_props',f'terrain/campaigns/{variant}/global_props.bin'),
                             ('authoring_camera',f'campaign_maps/{variant}/camera_heightmap.png')):
                rec=item[name]
                if rec.get('geometry_key')!=key:raise RuntimeError(f'Wrong original geometry key for {variant}/{name}')
                if geometry.get('files',{}).get(key)!=rec.get('sha256'):
                    raise RuntimeError(f'{variant}/{name} is not the recorded original geometry asset')
                asset_path,body=_read(rec,f'{variant}/{name}')
                blobs[name]=body;records[name]={'path':str(asset_path),'sha256':sha256(body),'geometry_key':key}
            image=Image.open(io.BytesIO(blobs['authoring_camera']))
            if image.format!='PNG' or image.size!=tuple(camera_size):
                raise RuntimeError(f'{variant} authoring camera has the wrong format/canvas dimensions')
            try:scale=float(image.info['height_scale'])
            except (KeyError,TypeError,ValueError):raise RuntimeError('Authoring camera is missing its height scale') from None
            if not math.isfinite(scale) or scale<=0:raise RuntimeError('Authoring camera has invalid height scale')
            image.verify()
            records['authoring_camera_bound_props_sha256']=records['global_props']['sha256']
            self._assets[variant]=blobs;provenance['variants'][variant]=records
        # The entire set is verified before any consumer can obtain one input.
        self._provenance=provenance;self.explicit=True

    def props(self,variant):
        if self.explicit:return self._assets[variant]['global_props']
        return (self.base/f'terrain/campaigns/{variant}/global_props.bin').read_bytes()

    def camera(self,variant):
        if self.explicit:return io.BytesIO(self._assets[variant]['authoring_camera'])
        # Preserve the existing default anchor selection and checks, centrally.
        anchor=self.base/f'campaign_maps/{variant}/camera_heightmap.png'
        clean_report=self.workspace/'analysis/native-height-baseline-restoration.json'
        if clean_report.exists():
            clean=json.loads(clean_report.read_text());expected=clean.get('resource_scene_anchor_inputs',{}).get(variant)
            current=sha256(self.props(variant))
            if expected!=current:raise RuntimeError('Resource scene geometry changed: refresh its authoring-height anchors before relocation')
            anchor_record=clean.get('resource_scene_anchor_cameras',{}).get(variant)
            if anchor_record:
                anchor=Path(anchor_record['path'])
                if sha256(anchor.read_bytes())!=anchor_record['sha256']:raise RuntimeError('Resource scene camera anchor was modified')
            else:anchor=Path(clean['historical_backup'])/f'campaign_maps/{variant}/camera_heightmap.png'
        return anchor

    def context(self):
        return {} if not self.explicit else {'prefab_source_manifest_sha256':self._provenance['manifest_sha256']}

    def provenance(self):return self._provenance


def load(workspace,variants,canvas,camera_size,manifest_path=None):
    return PrefabSource(workspace,variants,canvas,camera_size,manifest_path)
