"""Canonical pre-runtime tile-list inputs, independent of the ground/NativeWater BASE.

TW3K_TILE_SOURCE_MANIFEST explicitly opts in. It reuses the verified schema1
prefab geometry manifest identity. Optional variants[variant].tile_lists records
freeze both lists elsewhere; old schema1 derives both from its verified geometry
directory. Explicit errors never fall back to BASE. The launcher must freeze the
two files and use explicit records for reproducible future runs.
"""
import os
from pathlib import Path

import prefab_source

OPTION='TW3K_TILE_SOURCE_MANIFEST'
SUFFIXES=('tile_list.bin','global_map/tile_list.bin')


class TileListSource:
    def __init__(self,workspace,variants,canvas,camera_size,manifest_path=None):
        self.base=Path(workspace)/'terrain-before-polish'
        self.explicit=False;self._assets={};self._provenance=None
        option=manifest_path if manifest_path is not None else os.environ.get(OPTION)
        if option is None:return
        if not str(option).strip():raise ValueError(OPTION+' must not be empty when set')
        # Reuse existing strict map/design/canvas/phase/manifest and props-camera
        # verification; an arbitrary file with a self-declared checksum is not a
        # substitute for the one authoritative geometry checkpoint identity.
        verified=prefab_source.load(workspace,variants,canvas,camera_size,manifest_path=option)
        provenance=verified.provenance()
        _,raw=prefab_source._read({'path':provenance['manifest_path'],'sha256':provenance['manifest_sha256']},'tile manifest')
        manifest=prefab_source._json(raw)
        _,raw=prefab_source._read({'path':provenance['geometry_manifest_path'],'sha256':provenance['geometry_manifest_sha256']},'tile geometry manifest')
        geometry=prefab_source._json(raw)
        assets={};records={}
        for variant in variants:
            declared=manifest['variants'][variant].get('tile_lists')
            if declared is not None and set(declared)!=set(SUFFIXES):
                raise RuntimeError('Explicit tile source must provide the exact root/global pair')
            if declared is None:
                directory=Path(geometry['directory'])
                if not directory.is_absolute():raise ValueError('Geometry directory must be absolute')
            blobs={};pair={}
            for suffix in SUFFIXES:
                key=f'terrain/campaigns/{variant}/{suffix}'
                expected=geometry.get('files',{}).get(key)
                prefab_source._hash(expected,key)
                record=declared[suffix] if declared is not None else {
                    'path':str(directory/key),'sha256':expected,'geometry_key':key}
                if record.get('geometry_key')!=key or record.get('sha256')!=expected:
                    raise RuntimeError('Tile source is not the recorded same-geometry asset: '+key)
                path,body=prefab_source._read(record,key)
                blobs[suffix]=body
                pair[suffix]={'path':str(path),'sha256':expected,'geometry_key':key,
                              'record_origin':'explicit frozen tile_lists' if declared is not None else 'verified geometry directory'}
            assets[variant]=blobs;records[variant]=pair
        # Publish neither half until every variant's paired inputs verifies.
        self._assets=assets;self.explicit=True
        self._provenance={k:provenance[k] for k in ('manifest_path','manifest_sha256','geometry_manifest_path',
                            'geometry_manifest_sha256','inputs','canvas')}
        self._provenance.update(option=OPTION,variants=records,
            scope='Tile-list consumers only; immutable ground and NativeWater BASE unchanged')

    def list_bytes(self,variant,suffix):
        if suffix not in SUFFIXES:raise ValueError('Unexpected tile-list suffix')
        if self.explicit:return self._assets[variant][suffix]
        return (self.base/f'terrain/campaigns/{variant}/{suffix}').read_bytes()

    def context(self):
        return {} if not self.explicit else {'tile_source_manifest_sha256':self._provenance['manifest_sha256']}

    def provenance(self):return self._provenance


def load(workspace,variants,canvas,camera_size,manifest_path=None):
    return TileListSource(workspace,variants,canvas,camera_size,manifest_path)
