"""Live source stream integration for server playback."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from .stream import AudioFormat, MediaStream


@dataclass
class SourceStreamSession:
    """Tracks a live source stream and its media pipeline."""

    source_id: str
    audio_format: AudioFormat
    queue: asyncio.Queue[bytes | None]
    media_stream: MediaStream
    end_event: asyncio.Event
    play_task: asyncio.Task[int] | None = None

    @property
    def frame_stride(self) -> int:
        """Bytes per PCM frame for the source format."""
        return self.audio_format.channels * (self.audio_format.bit_depth // 8)

    def enqueue(self, chunk: bytes) -> None:
        """Queue PCM bytes for playback."""
        self.queue.put_nowait(chunk)

    def close(self) -> None:
        """Signal end of stream to the generator."""
        self.queue.put_nowait(None)


def build_source_media_stream(
    queue: asyncio.Queue[bytes | None],
    audio_format: AudioFormat,
) -> MediaStream:
    """Create a MediaStream backed by a source queue."""

    async def _generator() -> asyncio.AsyncGenerator[bytes, None]:
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            yield chunk

    return MediaStream(main_channel_source=_generator(), main_channel_format=audio_format)
