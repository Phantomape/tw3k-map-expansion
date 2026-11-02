"""analyze_reference.read_hex with the per-cell Python loop replaced by numpy.

Returns the same dict (same keys, order, values and Python int types) and raises
the same errors."""
import io
import struct
import zlib

import numpy as np

from analyze_reference import digest, read_exact, u32


def read_hex(path):
    def string(stream):
        return read_exact(stream, u32(stream)).decode("ascii")

    def strings(stream):
        return [string(stream) for _ in range(u32(stream))]

    data = path.read_bytes()
    if len(data) < 20 or zlib.crc32(data[:-4]) != struct.unpack("<I", data[-4:])[0]:
        raise ValueError("Map CRC32 mismatch")
    stream = io.BytesIO(data)
    major, minor = u32(stream), u32(stream)
    if (major, minor) != (0, 19):
        raise ValueError("Expected the Three Kingdoms version 19 map")
    read_exact(stream, 8)
    game, campaign_map = string(stream), string(stream)
    if game != "three_kingdoms":
        raise ValueError("Unexpected map game")
    lists = {name: strings(stream) for name in ("land_regions", "sea_regions", "land_ground_types", "sea_ground_types", "climates", "attritions", "areas_of_interest")}
    for _ in range(2):
        read_exact(stream, u32(stream) * 4)
    width, height = u32(stream), u32(stream)
    cells = read_exact(stream, width * height * 16)
    if len(stream.read()) != 4:
        raise ValueError("Unexpected trailing data; expected CRC32")
    regions = lists["land_regions"] + lists["sea_regions"]
    c = np.frombuffer(cells, np.uint8).reshape(-1, 16)
    region_id = ((c[:, 1].astype(np.int64) << 5) | (c[:, 0] >> 3)) - 1
    assigned = region_id != -1
    bad = np.flatnonzero(assigned & ~((region_id >= 0) & (region_id < len(regions))))
    if len(bad):
        raise ValueError(f"Invalid region ID at cell {int(bad[0])}")
    unassigned = int((~assigned).sum())
    idx = np.flatnonzero(assigned); rid = region_id[idx]; n = len(regions)
    x, y = idx % width, idx // width
    hex_count = np.bincount(rid, minlength=n)
    passable = np.bincount(rid, weights=((c[idx, 0] & 3) != 1) & ((c[idx, 2] & 8) == 0), minlength=n)
    town = np.bincount(rid, weights=(c[idx, 2] >> 4) > 0, minlength=n)
    min_x = np.full(n, width, np.int64); max_x = np.full(n, -1, np.int64)
    min_y = np.full(n, height, np.int64); max_y = np.full(n, -1, np.int64)
    np.minimum.at(min_x, rid, x); np.maximum.at(max_x, rid, x)
    np.minimum.at(min_y, rid, y); np.maximum.at(max_y, rid, y)
    summaries = [dict(hex_count=int(hex_count[i]), passable_land_count=int(passable[i]), town_slot_hex_count=int(town[i]),
                      min_x=int(min_x[i]), max_x=int(max_x[i]), min_y=int(min_y[i]), max_y=int(max_y[i])) for i in range(n)]
    return dict(source=str(path), sha256=digest(path), version=minor, game=game, campaign_map=campaign_map,
                width=width, height=height, crc32_valid=True, unassigned_hexes=unassigned, **lists,
                region_geometry=dict(zip(regions, summaries)))
