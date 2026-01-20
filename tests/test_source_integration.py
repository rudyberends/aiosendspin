from __future__ import annotations

import asyncio
import socket

import pytest

from aiosendspin.client import SendspinClient
from aiosendspin.models.source import (
    ClientHelloSourceSupport,
    SourceFormat,
    SourceStatePayload,
    SourceVadSettings,
)
from aiosendspin.models.types import (
    AudioCodec,
    MediaCommand,
    Roles,
    SourceCommand,
    SourceSignalType,
    SourceStateType,
)
from aiosendspin.server.server import SendspinServer


def _get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.mark.asyncio
async def test_source_flow_select_start_stop() -> None:
    loop = asyncio.get_running_loop()
    server = SendspinServer(loop, server_id="server-1", server_name="Test Server")
    port = _get_free_port()
    await server.start_server(port=port, host="127.0.0.1", discover_clients=False)
    url = f"ws://127.0.0.1:{port}{SendspinServer.API_PATH}"

    source_support = ClientHelloSourceSupport(
        format=SourceFormat(
            codec=AudioCodec.PCM,
            channels=1,
            sample_rate=48000,
            bit_depth=16,
        ),
    )

    source_client = SendspinClient(
        client_id="source-1",
        client_name="Source One",
        roles=[Roles.SOURCE],
        source_support=source_support,
    )
    controller_client = SendspinClient(
        client_id="controller-1",
        client_name="Controller One",
        roles=[Roles.CONTROLLER],
    )

    sources_event = asyncio.Event()
    selected_event = asyncio.Event()
    command_event = asyncio.Event()
    stop_event = asyncio.Event()

    def _on_controller_state(payload) -> None:
        if payload.controller and payload.controller.sources is not None:
            for source in payload.controller.sources:
                if source.id == "source-1":
                    sources_event.set()
                    if source.selected:
                        selected_event.set()

    def _on_source_command(payload) -> None:
        if payload.command == SourceCommand.START:
            command_event.set()
        elif payload.command == SourceCommand.STOP:
            stop_event.set()

    controller_client.add_controller_state_listener(_on_controller_state)
    source_client.add_source_command_listener(_on_source_command)

    await source_client.connect(url)
    await source_client.send_source_state(
        state=SourceStatePayload(
            state=SourceStateType.IDLE,
            signal=SourceSignalType.ABSENT,
        )
    )
    await controller_client.connect(url)

    await asyncio.wait_for(sources_event.wait(), timeout=5)

    await controller_client.send_group_command(
        MediaCommand.SELECT_SOURCE, source_id="source-1"
    )
    await asyncio.wait_for(selected_event.wait(), timeout=5)
    await asyncio.wait_for(command_event.wait(), timeout=5)

    await source_client.send_source_state(
        state=SourceStatePayload(
            state=SourceStateType.STREAMING,
            signal=SourceSignalType.PRESENT,
        )
    )
    await asyncio.sleep(0.1)

    server_timestamp_us = int(loop.time() * 1_000_000) + 1000
    await source_client.send_source_audio_chunk(
        b"frame-one", server_timestamp_us=server_timestamp_us
    )
    await asyncio.sleep(0.1)
    frames_received = server._sources["source-1"].frames_received  # noqa: SLF001
    assert frames_received == 1

    await controller_client.send_group_command(MediaCommand.SELECT_SOURCE, source_id=None)
    await asyncio.wait_for(stop_event.wait(), timeout=5)

    await source_client.send_source_state(
        state=SourceStatePayload(
            state=SourceStateType.IDLE,
            signal=SourceSignalType.ABSENT,
        )
    )
    await asyncio.sleep(0.1)
    server_timestamp_us = int(loop.time() * 1_000_000) + 1000
    await source_client.send_source_audio_chunk(
        b"frame-two", server_timestamp_us=server_timestamp_us
    )
    await asyncio.sleep(0.1)
    assert server._sources["source-1"].frames_received == frames_received  # noqa: SLF001

    await source_client.disconnect()
    await controller_client.disconnect()
    await server.close()


@pytest.mark.asyncio
async def test_source_vad_hint_roundtrip() -> None:
    loop = asyncio.get_running_loop()
    server = SendspinServer(loop, server_id="server-1", server_name="Test Server")
    port = _get_free_port()
    await server.start_server(port=port, host="127.0.0.1", discover_clients=False)
    url = f"ws://127.0.0.1:{port}{SendspinServer.API_PATH}"

    source_support = ClientHelloSourceSupport(
        format=SourceFormat(
            codec=AudioCodec.PCM,
            channels=1,
            sample_rate=48000,
            bit_depth=16,
        ),
    )

    source_client = SendspinClient(
        client_id="source-1",
        client_name="Source One",
        roles=[Roles.SOURCE],
        source_support=source_support,
    )

    vad_event = asyncio.Event()
    received = {}

    def _on_source_command(payload) -> None:
        if payload.vad is None:
            return
        received["threshold_db"] = payload.vad.threshold_db
        received["hold_ms"] = payload.vad.hold_ms
        vad_event.set()

    source_client.add_source_command_listener(_on_source_command)

    await source_client.connect(url)

    vad = SourceVadSettings(threshold_db=-42.0, hold_ms=1500)
    server._sources["source-1"].send_command(  # noqa: SLF001
        SourceCommand.START, vad=vad
    )

    await asyncio.wait_for(vad_event.wait(), timeout=5)
    assert received["threshold_db"] == -42.0
    assert received["hold_ms"] == 1500

    await source_client.disconnect()
    await server.close()
