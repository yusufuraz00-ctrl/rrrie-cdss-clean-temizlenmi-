"""Binary typed delta codec for websocket transport."""

from __future__ import annotations

import struct
from typing import Any


_MAGIC = b"RRD1"


def _encode_string(value: str) -> bytes:
    payload = str(value).encode("utf-8")
    return b"S" + struct.pack(">I", len(payload)) + payload


def _encode_value(value: Any) -> bytes:
    if value is None:
        return b"N"
    if isinstance(value, bool):
        return b"T" if value else b"F"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return b"D" + struct.pack(">d", float(value))
    if isinstance(value, str):
        return _encode_string(value)
    if isinstance(value, (list, tuple)):
        items = [_encode_value(item) for item in value]
        return b"A" + struct.pack(">I", len(items)) + b"".join(items)
    if isinstance(value, dict):
        items: list[bytes] = []
        entries = list(value.items())
        items.append(b"O" + struct.pack(">I", len(entries)))
        for key, item in entries:
            items.append(_encode_string(str(key)))
            items.append(_encode_value(item))
        return b"".join(items)
    return _encode_string(str(value))


def encode_delta(payload: dict[str, Any]) -> bytes:
    return _MAGIC + _encode_value(dict(payload or {}))


def _read_exact(buffer: memoryview, cursor: int, size: int) -> tuple[bytes, int]:
    end = cursor + size
    if end > len(buffer):
        raise ValueError("truncated typed delta payload")
    return bytes(buffer[cursor:end]), end


def _decode_value(buffer: memoryview, cursor: int = 0) -> tuple[Any, int]:
    tag_bytes, cursor = _read_exact(buffer, cursor, 1)
    tag = tag_bytes.decode("ascii")
    if tag == "N":
        return None, cursor
    if tag == "T":
        return True, cursor
    if tag == "F":
        return False, cursor
    if tag == "D":
        raw, cursor = _read_exact(buffer, cursor, 8)
        value = struct.unpack(">d", raw)[0]
        if value.is_integer():
            return int(value), cursor
        return value, cursor
    if tag == "S":
        raw_len, cursor = _read_exact(buffer, cursor, 4)
        length = struct.unpack(">I", raw_len)[0]
        raw, cursor = _read_exact(buffer, cursor, length)
        return raw.decode("utf-8"), cursor
    if tag == "A":
        raw_len, cursor = _read_exact(buffer, cursor, 4)
        count = struct.unpack(">I", raw_len)[0]
        values: list[Any] = []
        for _ in range(count):
            item, cursor = _decode_value(buffer, cursor)
            values.append(item)
        return values, cursor
    if tag == "O":
        raw_len, cursor = _read_exact(buffer, cursor, 4)
        count = struct.unpack(">I", raw_len)[0]
        values: dict[str, Any] = {}
        for _ in range(count):
            key, cursor = _decode_value(buffer, cursor)
            item, cursor = _decode_value(buffer, cursor)
            values[str(key)] = item
        return values, cursor
    raise ValueError(f"unknown typed delta tag: {tag}")


def decode_delta(payload: bytes | bytearray | memoryview) -> dict[str, Any]:
    buffer = memoryview(bytes(payload))
    if bytes(buffer[:4]) != _MAGIC:
        raise ValueError("invalid typed delta magic header")
    value, cursor = _decode_value(buffer, 4)
    if cursor != len(buffer):
        raise ValueError("unexpected trailing bytes in typed delta payload")
    if not isinstance(value, dict):
        raise ValueError("typed delta root payload must be an object")
    return value
