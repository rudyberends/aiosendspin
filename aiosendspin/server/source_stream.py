"""Live source stream integration for server playback."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

from aiosendspin.models import AudioCodec

from .stream import AudioFormat, MediaStream, _convert_s32_to_s24, _get_av, _resolve_audio_format

if TYPE_CHECKING:
    import av


class SourceDecoder:
    """Decode compressed source frames into PCM bytes."""

    def __init__(
        self,
        *,
        codec: AudioCodec,
        audio_format: AudioFormat,
        codec_header: bytes | None,
    ) -> None:
        self._codec = codec
        self._audio_format = audio_format
        self._codec_header = codec_header
        self._decoder: "av.AudioCodecContext" | None = None
        self._resampler: "av.AudioResampler" | None = None
        self._av_frame_stride: int = 0
        self._needs_s32_to_s24 = audio_format.bit_depth == 24
        self._setup()

    def _setup(self) -> None:
        av = _get_av()
        codec_name = "opus" if self._codec == AudioCodec.OPUS else self._codec.value
        decoder: "av.AudioCodecContext" = av.AudioCodecContext.create(codec_name, "r")  # type: ignore[name-defined]
        if self._codec_header is not None:
            decoder.extradata = self._codec_header
        decoder.open()
        _, av_format, av_layout, av_bytes = _resolve_audio_format(self._audio_format)
        self._av_frame_stride = av_bytes * self._audio_format.channels
        resampler: "av.AudioResampler" = av.AudioResampler(
            format=av_format,
            layout=av_layout,
            rate=self._audio_format.sample_rate,
        )
        self._decoder = decoder
        self._resampler = resampler

    def decode(self, data: bytes) -> list[bytes]:
        """Decode a compressed packet into PCM chunks."""
        if self._decoder is None or self._resampler is None:
            return []
        av = _get_av()
        packet = av.Packet(data)
        output: list[bytes] = []
        for frame in self._decoder.decode(packet):
            for out_frame in self._resampler.resample(frame):
                expected = self._av_frame_stride * out_frame.samples
                pcm_bytes = bytes(out_frame.planes[0])[:expected]
                if self._needs_s32_to_s24:
                    pcm_bytes = _convert_s32_to_s24(pcm_bytes)
                if pcm_bytes:
                    output.append(pcm_bytes)
        return output


@dataclass
class SourceStreamSession:
    """Tracks a live source stream and its media pipeline."""

    source_id: str
    audio_format: AudioFormat
    queue: asyncio.Queue[bytes | None]
    media_stream: MediaStream
    end_event: asyncio.Event
    decoder: SourceDecoder | None = None
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
