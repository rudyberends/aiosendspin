from __future__ import annotations

import struct

from aiosendspin.models import pack_binary_header_raw, unpack_binary_header
from aiosendspin.models.types import BinaryMessageType


def test_source_binary_header_roundtrip() -> None:
    timestamp = 123456789
    data = pack_binary_header_raw(BinaryMessageType.SOURCE_AUDIO_CHUNK.value, timestamp)
    assert len(data) == 9
    assert data == struct.pack(">Bq", BinaryMessageType.SOURCE_AUDIO_CHUNK.value, timestamp)

    header = unpack_binary_header(data)
    assert header.message_type == BinaryMessageType.SOURCE_AUDIO_CHUNK.value
    assert header.timestamp_us == timestamp
