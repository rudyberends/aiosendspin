"""Source implementation and helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from aiosendspin.models.core import ServerCommandMessage, ServerCommandPayload
from aiosendspin.models.source import (
    ClientHelloSourceSupport,
    ControllerSourceItem,
    SourceClientCommandPayload,
    SourceCommandPayload,
    SourceFormat,
    SourceFormatHint,
    SourceStatePayload,
    SourceVadSettings,
)
from aiosendspin.models.types import SourceClientCommand, SourceCommand, SourceSignalType, SourceStateType

if TYPE_CHECKING:
    from .client import SendspinClient


class SourceClient:
    """Source role wrapper for a client."""

    client: SendspinClient
    _state: SourceStateType = SourceStateType.IDLE
    _signal: SourceSignalType | None = None
    _level: float | None = None
    _last_frame_ts_us: int | None = None
    _frames_received: int = 0
    _last_event: SourceClientCommand | None = None
    _last_event_ts_us: int | None = None

    def __init__(self, client: SendspinClient) -> None:
        """Initialize source wrapper for a client."""
        self.client = client
        self._logger = client._logger.getChild("source")  # noqa: SLF001

    @property
    def support(self) -> ClientHelloSourceSupport | None:
        """Return source capabilities advertised in the hello payload."""
        return self.client.info.source_support

    @property
    def state(self) -> SourceStateType:
        """Current source state."""
        return self._state

    @property
    def signal(self) -> SourceSignalType | None:
        """Current signal presence."""
        return self._signal

    @property
    def level(self) -> float | None:
        """Current signal level."""
        return self._level

    @property
    def frames_received(self) -> int:
        """Total frames received from this source."""
        return self._frames_received

    def update_state(self, payload: SourceStatePayload) -> None:
        """Update source state from client report."""
        self._state = payload.state
        self._signal = payload.signal
        self._level = payload.level

    def handle_audio_chunk(self, timestamp_us: int, data: bytes) -> bool:
        """Handle an incoming audio chunk from this source."""
        if self._state != SourceStateType.STREAMING:
            self._logger.warning(
                "Rejecting source audio while state=%s", self._state.value
            )
            return False
        self._last_frame_ts_us = timestamp_us
        self._frames_received += 1
        self._logger.debug("Received source audio frame (%d bytes)", len(data))
        return True

    def send_command(
        self,
        command: SourceCommand,
        *,
        format: SourceFormat | SourceFormatHint | None = None,
        vad: SourceVadSettings | None = None,
    ) -> None:
        """Send a source command to the client."""
        self.client.send_message(
            ServerCommandMessage(
                payload=ServerCommandPayload(
                    source=SourceCommandPayload(command=command, format=format, vad=vad)
                )
            )
        )

    def handle_client_command(self, payload: SourceClientCommandPayload) -> None:
        """Handle source client command events."""
        timestamp_us = int(self.client._server.loop.time() * 1_000_000)  # noqa: SLF001
        if payload.command == SourceClientCommand.STARTED:
            self._last_event = SourceClientCommand.STARTED
            self._last_event_ts_us = timestamp_us
            self._logger.info("Source reported started")
        elif payload.command == SourceClientCommand.STOPPED:
            self._last_event = SourceClientCommand.STOPPED
            self._last_event_ts_us = timestamp_us
            self._logger.info("Source reported stopped")
        self.client._server._notify_controller_sources_changed()  # noqa: SLF001

    def build_controller_item(self, *, selected: bool) -> ControllerSourceItem:
        """Build controller-facing source entry."""
        support = self.support
        return ControllerSourceItem(
            id=self.client.client_id,
            name=self.client.name,
            state=self._state,
            signal=self._signal,
            selected=selected,
            last_event=self._last_event,
            last_event_ts_us=self._last_event_ts_us,
        )
