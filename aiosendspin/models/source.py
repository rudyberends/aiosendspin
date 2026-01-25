"""
Source role messages for the Sendspin protocol.

This module contains messages specific to clients with the source role, which
provide audio input to the server.
"""

from __future__ import annotations

from dataclasses import dataclass

from mashumaro.config import BaseConfig
from mashumaro.mixins.orjson import DataClassORJSONMixin

from .types import AudioCodec, SourceClientCommand, SourceCommand, SourceSignalType, SourceStateType


@dataclass
class SourceFormat(DataClassORJSONMixin):
    """Audio format for a source stream."""

    codec: AudioCodec
    """Codec identifier."""
    channels: int
    """Number of channels (e.g., 1 = mono, 2 = stereo)."""
    sample_rate: int
    """Sample rate in Hz (e.g., 44100, 48000)."""
    bit_depth: int
    """Bit depth for this format (e.g., 16, 24)."""

    def __post_init__(self) -> None:
        """Validate field values."""
        if self.channels <= 0:
            raise ValueError(f"channels must be positive, got {self.channels}")
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")
        if self.bit_depth <= 0:
            raise ValueError(f"bit_depth must be positive, got {self.bit_depth}")


@dataclass
class SourceFormatHint(DataClassORJSONMixin):
    """Partial audio format hint for a source stream."""

    codec: AudioCodec | None = None
    """Codec identifier."""
    channels: int | None = None
    """Number of channels (e.g., 1 = mono, 2 = stereo)."""
    sample_rate: int | None = None
    """Sample rate in Hz (e.g., 44100, 48000)."""
    bit_depth: int | None = None
    """Bit depth for this format (e.g., 16, 24)."""

    class Config(BaseConfig):
        """Config for parsing json messages."""

        omit_none = True


@dataclass
class InputStreamStartSource(DataClassORJSONMixin):
    """Source object in input_stream/start message."""

    codec: AudioCodec
    """Codec identifier."""
    channels: int
    """Number of channels (e.g., 1 = mono, 2 = stereo)."""
    sample_rate: int
    """Sample rate in Hz (e.g., 44100, 48000)."""
    bit_depth: int
    """Bit depth for this format (e.g., 16, 24)."""
    codec_header: str | None = None
    """Base64 encoded codec header (if necessary; e.g., FLAC/Opus)."""

    def __post_init__(self) -> None:
        """Validate field values."""
        if self.channels <= 0:
            raise ValueError(f"channels must be positive, got {self.channels}")
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")
        if self.bit_depth <= 0:
            raise ValueError(f"bit_depth must be positive, got {self.bit_depth}")

    class Config(BaseConfig):
        """Config for parsing json messages."""

        omit_none = True


@dataclass
class InputStreamRequestFormatSource(DataClassORJSONMixin):
    """Source object in input_stream/request-format message."""

    codec: AudioCodec | None = None
    """Requested codec identifier."""
    channels: int | None = None
    """Requested number of channels."""
    sample_rate: int | None = None
    """Requested sample rate in Hz."""
    bit_depth: int | None = None
    """Requested bit depth."""

    class Config(BaseConfig):
        """Config for parsing json messages."""

        omit_none = True


@dataclass
class SourceFeatures(DataClassORJSONMixin):
    """Source feature hints."""

    level: bool | None = None
    """True if source reports level (0..1)."""
    line_sense: bool | None = None
    """True if source reports line sense (signal present/absent)."""

    class Config(BaseConfig):
        """Config for parsing json messages."""

        omit_none = True


# Client -> Server: client/hello source support object
@dataclass
class ClientHelloSourceSupport(DataClassORJSONMixin):
    """Source support configuration - only if source role is set."""

    supported_formats: list[SourceFormat]
    """List of supported formats in priority order (first is preferred)."""
    features: SourceFeatures | None = None
    """Optional feature hints."""

    def __post_init__(self) -> None:
        """Validate field values."""
        if not self.supported_formats:
            raise ValueError("supported_formats cannot be empty")

    class Config(BaseConfig):
        """Config for parsing json messages."""

        omit_none = True


# Client -> Server: client/state source object
@dataclass
class SourceStatePayload(DataClassORJSONMixin):
    """Source object in client/state message."""

    state: SourceStateType
    """Source state."""
    level: float | None = None
    """Signal level in range 0..1 (optional)."""
    signal: SourceSignalType | None = None
    """Signal presence (optional)."""

    def __post_init__(self) -> None:
        """Validate field values."""
        if self.level is not None and not 0.0 <= self.level <= 1.0:
            raise ValueError(f"level must be in range 0..1, got {self.level}")

    class Config(BaseConfig):
        """Config for parsing json messages."""

        omit_none = True


# VAD settings hint for sources (server -> client)
@dataclass
class SourceVadSettings(DataClassORJSONMixin):
    """Voice activity detection settings."""

    threshold_db: float | None = None
    """Signal threshold in dB."""
    hold_ms: int | None = None
    """Hold time in milliseconds."""

    class Config(BaseConfig):
        """Config for parsing json messages."""

        omit_none = True


# Server -> Client: server/command source object
@dataclass
class SourceCommandPayload(DataClassORJSONMixin):
    """Source object in server/command message."""

    command: SourceCommand | None = None
    """Source command (start/stop)."""
    vad: SourceVadSettings | None = None
    """Optional VAD settings hint for the source."""

    class Config(BaseConfig):
        """Config for parsing json messages."""

        omit_none = True

# Client -> Server: client/command source object
@dataclass
class SourceClientCommandPayload(DataClassORJSONMixin):
    """Source object in client/command message."""

    command: SourceClientCommand
    """Source client command (started/stopped)."""

    class Config(BaseConfig):
        """Config for parsing json messages."""

        omit_none = True


# Server -> Client: server/state controller.sources item
@dataclass
class ControllerSourceItem(DataClassORJSONMixin):
    """Controller-facing source listing entry."""

    id: str
    """Source client id."""
    name: str
    """Friendly name for the source."""
    state: SourceStateType
    """Current source state."""
    signal: SourceSignalType | None = None
    """Signal presence (optional)."""
    selected: bool | None = None
    """True if this source is currently selected (optional)."""
    last_event: SourceClientCommand | None = None
    """Last source client event (optional)."""
    last_event_ts_us: int | None = None
    """Server timestamp for last event in microseconds (optional)."""

    class Config(BaseConfig):
        """Config for parsing json messages."""

        omit_none = True
