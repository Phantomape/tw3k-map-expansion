"""Read empty internal-table formats from official Assembly Kit exports only.

This module never searches workshop/reference resources. The caller supplies a
directory produced by ``all_directions_build.py official-headers`` (or an equivalent
official SDK export). Only a verified zero-row container is accepted; the table
format is retained and the identifier is generated under this project's key.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import re
import struct
import uuid


_MAGIC = b"\xfd\xfe\xfc\xff"
_VERSION = b"\xfc\xfd\xfe\xff"


def empty_table_format(packed: bytes) -> bytes:
    """Return the optional SDK version block; reject records and malformed data."""
    if len(packed) < 11 or packed[:4] != _MAGIC:
        raise ValueError("Unknown or truncated official DB header")
    guid_length = struct.unpack_from("<H", packed, 4)[0]
    offset = 6 + guid_length * 2
    if not 1 <= guid_length <= 64 or offset + 5 > len(packed):
        raise ValueError("Invalid official DB GUID length")
    try:
        uuid.UUID(packed[6:offset].decode("utf-16le"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError("Invalid official DB GUID") from exc
    version = b""
    if packed[offset:offset + 4] == _VERSION:
        if offset + 13 > len(packed):
            raise ValueError("Truncated official DB version block")
        version = packed[offset:offset + 8]
        offset += 8
    if packed[offset] != 1:
        raise ValueError("Unexpected official DB flag")
    rows = struct.unpack_from("<I", packed, offset + 1)[0]
    if rows:
        raise ValueError("Official empty-table source contains records")
    if offset + 5 != len(packed):
        raise ValueError("Official empty-table source has trailing payload")
    return version


def from_official_export(table: str, directory: Path) -> tuple[bytes, dict]:
    """Create an empty container from an explicit SDK-export directory.

    Missing exports fail closed. No table-specific version is guessed, and no
    workshop file is a fallback. The report records the actual input and hash.
    """
    if not re.fullmatch(r"[a-z][a-z0-9_]*", table):
        raise ValueError("Invalid SDK table name")
    source = Path(directory) / (table + "_tables") / "!tw3k_exp"
    try:
        packed = source.read_bytes()
    except FileNotFoundError as exc:
        raise ValueError(
            f"No official SDK zero-row export for {table}: {source}. "
            "Run all_directions_build.py official-headers after setup, or prepare "
            "official Assembly Kit exports and pass --official-empty-db; "
            "workshop/reference table headers are not accepted."
        ) from exc
    version = empty_table_format(packed)
    guid = str(uuid.uuid5(uuid.NAMESPACE_URL, "tw3k-exp-empty/" + table)).encode("utf-16le")
    header = _MAGIC + struct.pack("<H", len(guid) // 2) + guid + version + b"\x01" + struct.pack("<I", 0)
    report = {
        "rows": 0,
        "empty_container": "verified official SDK zero-row export",
        "official_sdk_source": str(source),
        "official_sdk_source_sha256": hashlib.sha256(packed).hexdigest(),
        "bytes": len(header),
    }
    return header, report
