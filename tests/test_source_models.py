from __future__ import annotations

import orjson

from aiosendspin.models.controller import ControllerCommandPayload, ControllerStatePayload
from aiosendspin.models.core import (
    ClientCommandMessage,
    ClientCommandPayload,
    ClientHelloMessage,
    ClientHelloPayload,
    ClientStateMessage,
    ClientStatePayload,
    ServerCommandMessage,
    ServerCommandPayload,
    ServerStateMessage,
    ServerStatePayload,
)
from aiosendspin.models.source import (
    ClientHelloSourceSupport,
    ControllerSourceItem,
    SourceClientCommandPayload,
    SourceCommandPayload,
    SourceFeatures,
    SourceFormat,
    SourceStatePayload,
)
from aiosendspin.models.types import (
    AudioCodec,
    ClientMessage,
    MediaCommand,
    Roles,
    SourceClientCommand,
    ServerMessage,
    SourceCommand,
    SourceSignalType,
    SourceStateType,
)


def test_source_hello_roundtrip() -> None:
    payload = ClientHelloPayload(
        client_id="source-1",
        name="Source One",
        version=1,
        supported_roles=[Roles.SOURCE],
        source_support=ClientHelloSourceSupport(
            format=SourceFormat(
                codec=AudioCodec.PCM,
                channels=2,
                sample_rate=48000,
                bit_depth=16,
            ),
            features=SourceFeatures(level=True, line_sense=True),
        ),
    )
    message = ClientHelloMessage(payload=payload)
    parsed = ClientMessage.from_json(message.to_json())
    assert isinstance(parsed, ClientHelloMessage)
    assert parsed.payload.source_support is not None


def test_source_state_roundtrip() -> None:
    payload = ClientStatePayload(
        source=SourceStatePayload(
            state=SourceStateType.STREAMING,
            level=0.7,
            signal=SourceSignalType.PRESENT,
        )
    )
    message = ClientStateMessage(payload=payload)
    parsed = ClientMessage.from_json(message.to_json())
    assert isinstance(parsed, ClientStateMessage)
    assert parsed.payload.source is not None
    assert parsed.payload.source.state == SourceStateType.STREAMING


def test_source_command_roundtrip() -> None:
    payload = ServerCommandPayload(
        source=SourceCommandPayload(
            command=SourceCommand.START,
            format=SourceFormat(
                codec=AudioCodec.OPUS,
                channels=1,
                sample_rate=48000,
                bit_depth=16,
            ),
        )
    )
    message = ServerCommandMessage(payload=payload)
    parsed = ServerMessage.from_json(message.to_json())
    assert isinstance(parsed, ServerCommandMessage)
    assert parsed.payload.source is not None
    assert parsed.payload.source.command == SourceCommand.START


def test_source_client_command_roundtrip() -> None:
    payload = ClientCommandPayload(
        source=SourceClientCommandPayload(command=SourceClientCommand.STARTED)
    )
    message = ClientCommandMessage(payload=payload)
    parsed = ClientMessage.from_json(message.to_json())
    assert isinstance(parsed, ClientCommandMessage)
    assert parsed.payload.source is not None
    assert parsed.payload.source.command == SourceClientCommand.STARTED


def test_controller_select_source_roundtrip() -> None:
    payload = ControllerCommandPayload(command=MediaCommand.SELECT_SOURCE, source_id="source-1")
    message = ClientCommandMessage(payload=ClientCommandPayload(controller=payload))
    parsed = ClientMessage.from_json(message.to_json())
    assert isinstance(parsed, ClientCommandMessage)
    assert parsed.payload.controller is not None
    assert parsed.payload.controller.source_id == "source-1"


def test_controller_clear_source_roundtrip() -> None:
    payload = ControllerCommandPayload(command=MediaCommand.SELECT_SOURCE, source_id=None)
    message = ClientCommandMessage(payload=ClientCommandPayload(controller=payload))
    parsed = ClientMessage.from_json(message.to_json())
    assert isinstance(parsed, ClientCommandMessage)
    assert parsed.payload.controller is not None
    assert parsed.payload.controller.source_id is None


def test_controller_sources_roundtrip() -> None:
    controller_payload = ControllerStatePayload(
        supported_commands=[MediaCommand.SELECT_SOURCE],
        volume=10,
        muted=False,
        sources=[
            ControllerSourceItem(
                id="source-1",
                name="Source One",
                state=SourceStateType.IDLE,
                signal=SourceSignalType.UNKNOWN,
                selected=True,
                last_event=SourceClientCommand.STARTED,
                last_event_ts_us=123456,
            )
        ],
    )
    message = ServerStateMessage(payload=ServerStatePayload(controller=controller_payload))
    parsed = ServerMessage.from_json(message.to_json())
    assert isinstance(parsed, ServerStateMessage)
    assert parsed.payload.controller is not None
    assert parsed.payload.controller.sources is not None
    assert parsed.payload.controller.sources[0].id == "source-1"


def test_source_support_json_shape() -> None:
    payload = ClientHelloPayload(
        client_id="source-1",
        name="Source One",
        version=1,
        supported_roles=[Roles.SOURCE],
        source_support=ClientHelloSourceSupport(
            format=SourceFormat(
                codec=AudioCodec.PCM,
                channels=2,
                sample_rate=48000,
                bit_depth=16,
            ),
            features=SourceFeatures(level=True, line_sense=False),
        ),
    )
    message = ClientHelloMessage(payload=payload)
    data = orjson.loads(message.to_json())
    assert "source@v1_support" in data["payload"]
