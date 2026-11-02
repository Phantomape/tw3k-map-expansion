"""Readers for PFH5 packs (inventory) and logical maps (read_hex), and file digests.

Only reads files. Binary layout references: RPFM pack_versions/pfh5.rs and CAIME MapHexFile.cs.
Not a pack writer or a map compiler. (The early reference-mod analysis that lived here is in the private
repository's scripts/archive/analyze_190_reference.py.)
"""
from __future__ import annotations

import hashlib
import io
import struct
import zlib
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_exact(stream, count):
    data = stream.read(count)
    if len(data) != count:
        raise ValueError("Truncated file")
    return data


def u32(stream):
    return struct.unpack("<I", read_exact(stream, 4))[0]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inventory(path, include_hash=True):
    with path.open("rb") as stream:
        magic = read_exact(stream, 4)
        if magic[:3] == b"MFH":
            stream.seek(8)
            magic = read_exact(stream, 4)
        if magic != b"PFH5":
            raise ValueError(f"Only PFH5 supported; got {magic!r}")
        flags, dep_count, dep_size, count, index_size, timestamp = struct.unpack("<6I", read_exact(stream, 24))
        if flags & ~0x4F:
            raise ValueError("Encrypted/extended/unknown pack flags unsupported")
        deps = read_exact(stream, dep_size).split(b"\0")[:-1]
        if len(deps) != dep_count:
            raise ValueError("Dependency count mismatch")
        # Parse the index from one buffer: reading each name byte by byte through
        # a stream took seconds per vanilla pack in every script that opens them.
        index = read_exact(stream, index_size)
        head = 9 if flags & 0x40 else 5
        pos = 0
        offset = stream.tell()
        entries = []
        for _ in range(count):
            if pos + head > index_size:
                raise ValueError("Truncated file")
            size = struct.unpack_from("<I", index, pos)[0]
            compressed = index[pos + head - 1]
            if compressed not in (0, 1):
                raise ValueError("Invalid compression flag")
            end = index.find(b"\0", pos + head)
            if end < 0:
                raise ValueError("Truncated file")
            entries.append(dict(path=index[pos + head:end].decode("utf-8").replace("\\", "/"), size=size, offset=offset, compressed=bool(compressed)))
            pos = end + 1
            offset += size
        if pos != index_size or offset != path.stat().st_size:
            raise ValueError("Index/data bounds do not match pack size")
        if len({e["path"] for e in entries}) != count:
            raise ValueError("Duplicate entry paths")
    return dict(source=str(path), sha256=digest(path) if include_hash else None, format="PFH5", flags=flags, timestamp=timestamp,
                dependencies=[d.decode("utf-8") for d in deps], entry_count=count,
                groups=dict(Counter(e["path"].split("/")[0] for e in entries)), entries=entries)


def _read_hex_loop(path):
    """Reference per-cell implementation; read_hex is fast_hex.read_hex (identical result)."""
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
    summaries = [dict(hex_count=0, passable_land_count=0, town_slot_hex_count=0, min_x=width, max_x=-1, min_y=height, max_y=-1) for _ in regions]
    unassigned = 0
    for i in range(width * height):
        cell = cells[i * 16 : i * 16 + 16]
        region_id = ((cell[1] << 5) | (cell[0] >> 3)) - 1
        if region_id == -1:
            unassigned += 1
            continue
        if not 0 <= region_id < len(regions):
            raise ValueError(f"Invalid region ID at cell {i}")
        info = summaries[region_id]
        x, y = i % width, i // width
        info["hex_count"] += 1
        info["passable_land_count"] += int(cell[0] & 3 != 1 and cell[2] & 8 == 0)
        info["town_slot_hex_count"] += int(cell[2] >> 4 > 0)
        for axis, value in (("x", x), ("y", y)):
            info[f"min_{axis}"] = min(info[f"min_{axis}"], value)
            info[f"max_{axis}"] = max(info[f"max_{axis}"], value)
    return dict(source=str(path), sha256=digest(path), version=minor, game=game, campaign_map=campaign_map,
                width=width, height=height, crc32_valid=True, unassigned_hexes=unassigned, **lists,
                region_geometry=dict(zip(regions, summaries)))


# Vectorised read_hex where numpy exists (identical dict, ~9 s -> 0.5 s). Imported
# after them: fast_hex uses digest/read_exact/u32 from this module.
try:
    from fast_hex import read_hex  # noqa: E402
except ImportError:                # interpreters without numpy (SDK map-data export)
    read_hex = _read_hex_loop
