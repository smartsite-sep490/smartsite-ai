import asyncio
import copy
import hashlib
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import UUID

import httpx
import pytest
from pydantic import SecretStr
from test_browser_uplink_ingestion import sample_packet
from test_frame_evidence_transport import (
    GENERATION_SECRET,
    SERVICE_TOKEN,
    live_evidence_packet,
    make_jpeg_bytes,
    parse_multipart,
)
from test_tracked_outbox import frame_parts
from test_tracked_runtime_selection import document

from smartsite_ai.core.canonical_hash import compute_canonical_payload_hash
from smartsite_ai.domain.event_delivery_context import EventDeliveryContext
from smartsite_ai.domain.observations import (
    FrameDimensions,
    ObservationFramePartition,
    TechnicalObservationEvent,
)
from smartsite_ai.domain.tracked_frame_result import TrackedFrameReferences
from smartsite_ai.evidence.local_publisher import LocalEvidencePublisher
from smartsite_ai.ingestion.browser_uplink import BrowserUplinkFrameSource
from smartsite_ai.ingestion.envelope import FrameEnvelope
from smartsite_ai.integrations.backend_client import BackendClient
from smartsite_ai.integrations.backend_configuration_client import BackendConfigurationClient
from smartsite_ai.integrations.frame_evidence_upload import TrackedFrameEvidenceUploader
from smartsite_ai.integrations.outbox import OutboxDispatcher, SqliteEventOutbox
from smartsite_ai.runtime.assembly import (
    assemble_sessions,
    close_configuration_clients,
    preflight_manifest,
)
from smartsite_ai.runtime.manifest import CameraRuntimeError, bind_manifest, parse_manifest

NOW = datetime(2026, 11, 17, tzinfo=UTC)


def _write_frame_jpeg(evidence_dir, event: dict, jpeg: bytes) -> tuple[dict, object]:
    session = event["streamSessionId"].lower()
    seq = event["framePartition"]["sequenceNumber"]
    event_id = event["eventId"].lower()
    digest = hashlib.sha256(jpeg).hexdigest()
    path = evidence_dir / f"{session}_{seq}_{event_id}.jpg"
    path.write_bytes(jpeg)

    updated = copy.deepcopy(event)
    updated["evidence"] = [
        {
            "kind": "FRAME",
            "uri": f"local://evidence/{session}/{seq}/{event_id}.jpg",
            "sha256": digest,
        }
    ]
    return updated, path


def _live_two_part_frame(evidence_dir):
    parts, context, credential, refs = frame_parts()
    context = EventDeliveryContext.model_validate(
        {
            **context.model_dump(mode="json", by_alias=True),
            "sourceKind": "LIVE",
        }
    )
    credential = credential.model_copy(update={"secret": SecretStr(GENERATION_SECRET)})
    frame_jpeg = make_jpeg_bytes(color=(10, 40, 80))
    part0, path0 = _write_frame_jpeg(evidence_dir, parts[0], frame_jpeg)
    part1, path1 = _write_frame_jpeg(evidence_dir, parts[1], frame_jpeg)
    return (part0, part1), frame_jpeg, (path0, path1), context, credential, refs


def _single_live_part(evidence_dir):
    parts, frame_jpeg, _, context, credential, refs = _live_two_part_frame(evidence_dir)
    single_part = copy.deepcopy(parts[0])
    single_part["framePartition"]["partCount"] = 1
    single_part, _ = _write_frame_jpeg(evidence_dir, single_part, frame_jpeg)
    single_refs = TrackedFrameReferences.model_validate(
        {
            "scopeSha256": refs.scope_sha256,
            "sequenceNumber": single_part["framePartition"]["sequenceNumber"],
            "subjects": [
                {
                    "segmentId": single_part["continuity"]["subjects"][0]["segmentId"],
                    "eventId": single_part["eventId"].lower(),
                    "personObservationIndex": 0,
                }
            ],
            "retiredSegmentIds": [],
        }
    )
    return single_part, context, credential, single_refs


def test_manifest_private_evidence_upload_defaults_off_and_rejects_replay_or_untracked(tmp_path):
    value, _ = document(tmp_path)
    default_manifest = parse_manifest(value)
    assert default_manifest.cameras[0].private_evidence_upload is False

    replay_doc = copy.deepcopy(value)
    replay_doc["cameras"][0]["live"] = False
    replay_doc["cameras"][0]["privateEvidenceUpload"] = True
    with pytest.raises(CameraRuntimeError):
        parse_manifest(replay_doc)

    untracked_doc = copy.deepcopy(value)
    untracked_doc["cameras"][0].pop("trackedV17")
    untracked_doc["cameras"][0]["privateEvidenceUpload"] = True
    with pytest.raises(CameraRuntimeError):
        parse_manifest(untracked_doc)

    non_bool_doc = copy.deepcopy(value)
    non_bool_doc["cameras"][0]["privateEvidenceUpload"] = "true"
    with pytest.raises(CameraRuntimeError):
        parse_manifest(non_bool_doc)


@pytest.mark.anyio
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("browser", [False, True])
async def test_assembly_wires_evidence_uploader_only_when_manifest_opts_in(
    tmp_path, enabled, browser
):
    value, bundle = document(tmp_path)
    if enabled:
        value["cameras"][0]["privateEvidenceUpload"] = True
    bound = bind_manifest(parse_manifest(value), {})

    def config_resp(_):
        return httpx.Response(
            200,
            json=bundle.model_dump(mode="json", by_alias=True),
            headers={"ETag": f'"sha256:{bundle.bundle_sha256}"'},
        )

    failed, plans = await preflight_manifest(
        bound,
        configuration_client_factory=lambda: pytest.fail("must not use legacy config"),
        policy_v2_client_factory=lambda: BackendConfigurationClient(
            "https://backend.example",
            SERVICE_TOKEN,
            policy_v2=True,
            transport=httpx.MockTransport(config_resp),
        ),
    )
    assert failed is None
    backends: list[BackendClient] = []

    def backend_factory():
        client = BackendClient("https://backend.example", SERVICE_TOKEN)
        backends.append(client)
        return client

    def source_factory(config):
        if not browser:
            return SimpleNamespace()
        generation = plans[0].generation
        assert generation is not None
        lease = UUID(generation.stream_session_id)

        async def packets():
            yield sample_packet() | {
                "cameraExternalId": config.camera_external_id,
                "sessionId": str(lease),
            }
            await asyncio.Event().wait()

        return BrowserUplinkFrameSource(
            packets,
            stream_id=config.stream_id,
            camera_external_id=config.camera_external_id,
            session_id=lease,
        )

    try:
        sessions = assemble_sessions(
            plans,
            detector=SimpleNamespace(),
            source_factory=source_factory,
            backend_client_factory=backend_factory,
            evidence_max_bytes=1_048_576,
            delivery_interval_seconds=0.05,
        )
        dispatcher = sessions[0].worker._dispatcher
        if enabled:
            assert isinstance(dispatcher._tracked_evidence_upload, TrackedFrameEvidenceUploader)
            assert dispatcher._tracked_evidence_upload._root == tmp_path.resolve()
            assert dispatcher._tracked_evidence_upload._backend is backends[-1]
        else:
            assert dispatcher._tracked_evidence_upload is None
        if browser:
            stream = sessions[0].worker._stream
            await stream.start()
            try:
                frame = await asyncio.wait_for(stream.get_frame(), timeout=1)
                assert frame.sequence_number == 42
                assert str(frame.session_id) == plans[0].generation.stream_session_id
            finally:
                await stream.stop()
    finally:
        await close_configuration_clients(plans)
        for client in backends:
            await client.aclose()


@pytest.mark.anyio
async def test_uploader_validates_root_link_changed_file_uri_and_missing_frame(
    tmp_path, monkeypatch
):
    event, _, credential, _, wire, jpeg, _ = live_evidence_packet()
    session = event["streamSessionId"].lower()
    seq = event["framePartition"]["sequenceNumber"]
    event_id = event["eventId"].lower()
    target = tmp_path / f"{session}_{seq}_{event_id}.jpg"
    target.write_bytes(jpeg)

    with pytest.raises(ValueError):
        TrackedFrameEvidenceUploader(
            BackendClient("https://backend.example", SERVICE_TOKEN),
            tmp_path / "missing-dir",
        )
    with pytest.raises(ValueError, match="unavailable"):
        TrackedFrameEvidenceUploader(
            BackendClient("https://backend.example", SERVICE_TOKEN),
            target,
        )

    def fail_if_called(_: httpx.Request) -> httpx.Response:
        pytest.fail("Invalid local evidence or envelope must not reach the network")

    async with BackendClient(
        "https://backend.example",
        SERVICE_TOKEN,
        transport=httpx.MockTransport(fail_if_called),
    ) as client:
        uploader = TrackedFrameEvidenceUploader(client, tmp_path)

        # No FRAME item -> skips upload without network calls.
        _, _, _, _, no_frame_wire, _, _ = live_evidence_packet(with_frame=False)
        await uploader(credential, no_frame_wire)

        # REPLAY sourceKind is refused locally without touching disk or network.
        _, _, _, _, replay_wire, _, _ = live_evidence_packet(source_kind="REPLAY")
        with pytest.raises(ValueError, match="envelope is invalid"):
            await uploader(credential, replay_wire)

        # Wrong event URI.
        bad_uri = json.loads(wire)
        bad_uri["event"]["evidence"][0]["uri"] = f"local://evidence/{session}/{seq}/../escape.jpg"
        with pytest.raises(ValueError, match="reference is invalid"):
            await uploader(credential, json.dumps(bad_uri).encode("utf-8"))

        # Mutated JPEG bytes after envelope creation (sha256 mismatch).
        target.write_bytes(make_jpeg_bytes(color=(1, 2, 3)))
        with pytest.raises(ValueError, match="unavailable or inconsistent"):
            await uploader(credential, wire)

        # Missing frame file on disk.
        target.unlink()
        with pytest.raises(ValueError, match="unavailable or inconsistent"):
            await uploader(credential, wire)
        target.write_bytes(jpeg)

        # Link-like target file is refused.
        monkeypatch.setattr(
            "smartsite_ai.integrations.frame_evidence_upload.is_link_like",
            lambda p: p == target,
        )
        with pytest.raises(ValueError, match="unavailable or inconsistent"):
            await uploader(credential, wire)

        # Link-like evidence root is refused before touching any child.
        monkeypatch.setattr(
            "smartsite_ai.integrations.frame_evidence_upload.is_link_like",
            lambda p: p == uploader._root,
        )
        with pytest.raises(ValueError, match="unavailable or inconsistent"):
            await uploader(credential, wire)


@pytest.mark.anyio
async def test_outbox_uploads_evidence_before_each_raw_part_and_still_requires_seal(tmp_path):
    parts, _, _, context, credential, refs = _live_two_part_frame(tmp_path)
    outbox = SqliteEventOutbox(tmp_path / "outbox.sqlite", clock=lambda: NOW)
    await outbox.enqueue_tracked_parts(
        credential,
        context,
        parts,
        expected_seal=refs,
        expires_at=NOW + timedelta(seconds=30),
        max_pending=2,
        max_bytes=200_000,
    )
    ((frame_id, _),) = await outbox.ready_tracked_parts(credential, limit=2)

    call_log: list[tuple[str, str]] = []
    sealed = False

    def backend(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/integrations/ai/tracked-parts/evidence":
            fields = parse_multipart(request)
            event = json.loads(fields["event"][2].decode("utf-8"))
            digest = hashlib.sha256(fields["frame"][2]).hexdigest()
            payload_hash = compute_canonical_payload_hash(event)
            call_log.append(("EVIDENCE", event["eventId"].lower()))
            return httpx.Response(
                200,
                json={
                    "schemaVersion": "1.0.0",
                    "kind": "EVIDENCE_STORED",
                    "eventId": event["eventId"].lower(),
                    "payloadHash": payload_hash,
                    "bindingHash": context.binding_hash(payload_hash),
                    "sha256": digest,
                },
            )
        if request.url.path == "/api/v1/integrations/ai/tracked-parts":
            event = json.loads(request.content)["event"]
            digest = compute_canonical_payload_hash(event)
            call_log.append(("RAW", event["eventId"].lower()))
            return httpx.Response(
                200,
                json={
                    "kind": "RAW_STORED",
                    "eventId": event["eventId"].lower(),
                    "payloadHash": digest,
                    "bindingHash": context.binding_hash(digest),
                },
            )
        assert request.method == "GET"
        call_log.append(("SEAL_CHECK", frame_id))
        if not sealed:
            return httpx.Response(404)
        return httpx.Response(
            200, json={"kind": "ADMITTED_TRACKED_FRAME", **refs.model_dump(by_alias=True)}
        )

    async with BackendClient(
        "https://backend.example",
        SERVICE_TOKEN,
        transport=httpx.MockTransport(backend),
    ) as client:
        dispatcher = OutboxDispatcher(
            outbox,
            client,
            generation_credential=credential,
            tracked_batch_limit=2,
            tracked_evidence_upload=TrackedFrameEvidenceUploader(client, tmp_path),
        )
        await dispatcher.drain_once(limit=2)

        assert call_log == [
            ("EVIDENCE", parts[0]["eventId"].lower()),
            ("RAW", parts[0]["eventId"].lower()),
            ("EVIDENCE", parts[1]["eventId"].lower()),
            ("RAW", parts[1]["eventId"].lower()),
            ("SEAL_CHECK", frame_id),
        ]
        assert not await outbox.ready_tracked_parts(credential, limit=2)
        # RAW_STORED + EVIDENCE_STORED never advance predecessor without ADMITTED seal.
        assert (
            await outbox.read_tracked_predecessor(refs.scope_sha256, refs.sequence_number) is None
        )

        sealed = True
        await dispatcher.drain_once(limit=2)

    assert call_log[4:] == [("SEAL_CHECK", frame_id), ("SEAL_CHECK", frame_id)]
    head = await outbox.read_tracked_predecessor(refs.scope_sha256, refs.sequence_number)
    assert head is not None
    assert head.subject_references() == refs.subject_references()


@pytest.mark.anyio
async def test_failed_evidence_upload_blocks_raw_and_retries_exact_binding(tmp_path):
    single_part, context, credential, single_refs = _single_live_part(tmp_path)

    outbox = SqliteEventOutbox(tmp_path / "retry.sqlite", clock=lambda: NOW)
    await outbox.enqueue_tracked_parts(
        credential,
        context,
        (single_part,),
        expected_seal=single_refs,
        expires_at=NOW + timedelta(seconds=30),
        max_pending=2,
        max_bytes=200_000,
    )
    ((frame_id, wires),) = await outbox.ready_tracked_parts(credential, limit=2)

    raw_calls = 0
    evidence_bindings: list[tuple[str, str, str]] = []
    fail_with_503 = True

    def backend(request: httpx.Request) -> httpx.Response:
        nonlocal raw_calls, fail_with_503
        if request.url.path == "/api/v1/integrations/ai/tracked-parts/evidence":
            fields = parse_multipart(request)
            event = json.loads(fields["event"][2].decode("utf-8"))
            payload_hash = compute_canonical_payload_hash(event)
            binding_hash = context.binding_hash(payload_hash)
            img_sha = hashlib.sha256(fields["frame"][2]).hexdigest()
            evidence_bindings.append((event["eventId"].lower(), binding_hash, img_sha))
            if fail_with_503:
                fail_with_503 = False
                return httpx.Response(503)
            return httpx.Response(
                200,
                json={
                    "schemaVersion": "1.0.0",
                    "kind": "EVIDENCE_STORED",
                    "eventId": event["eventId"].lower(),
                    "payloadHash": payload_hash,
                    "bindingHash": binding_hash,
                    "sha256": img_sha,
                },
            )
        if request.url.path == "/api/v1/integrations/ai/tracked-parts":
            raw_calls += 1
            assert request.content == wires[0]
            digest = compute_canonical_payload_hash(json.loads(request.content)["event"])
            return httpx.Response(
                200,
                json={
                    "kind": "RAW_STORED",
                    "eventId": single_part["eventId"].lower(),
                    "payloadHash": digest,
                    "bindingHash": context.binding_hash(digest),
                },
            )
        return httpx.Response(404)

    async with BackendClient(
        "https://backend.example",
        SERVICE_TOKEN,
        transport=httpx.MockTransport(backend),
    ) as client:
        dispatcher = OutboxDispatcher(
            outbox,
            client,
            generation_credential=credential,
            tracked_batch_limit=2,
            tracked_evidence_upload=TrackedFrameEvidenceUploader(client, tmp_path),
        )
        # Transient 503 on evidence blocks RAW and leaves durable bytes unchanged for next tick.
        await dispatcher.drain_once(limit=2)
        assert raw_calls == 0
        assert await outbox.ready_tracked_parts(credential, limit=2) == ((frame_id, wires),)

        # Next tick retries exact binding & succeeds before RAW send.
        await dispatcher.drain_once(limit=2)
        assert raw_calls == 1
        assert len(evidence_bindings) == 2
        assert evidence_bindings[0] == evidence_bindings[1]
        assert not await outbox.ready_tracked_parts(credential, limit=2)


@pytest.mark.anyio
@pytest.mark.parametrize("status_code", [400, 401, 403, 409])
async def test_permanent_upload_failure_blocks_raw_and_never_confirms(tmp_path, status_code):
    single_part, context, credential, single_refs = _single_live_part(tmp_path)

    outbox = SqliteEventOutbox(tmp_path / f"perm-{status_code}.sqlite", clock=lambda: NOW)
    await outbox.enqueue_tracked_parts(
        credential,
        context,
        (single_part,),
        expected_seal=single_refs,
        expires_at=NOW + timedelta(seconds=30),
        max_pending=2,
        max_bytes=200_000,
    )

    raw_calls = 0

    def backend(request: httpx.Request) -> httpx.Response:
        nonlocal raw_calls
        if request.url.path == "/api/v1/integrations/ai/tracked-parts/evidence":
            return httpx.Response(status_code)
        if request.url.path == "/api/v1/integrations/ai/tracked-parts":
            raw_calls += 1
        return httpx.Response(404)

    async with BackendClient(
        "https://backend.example",
        SERVICE_TOKEN,
        transport=httpx.MockTransport(backend),
    ) as client:
        dispatcher = OutboxDispatcher(
            outbox,
            client,
            generation_credential=credential,
            tracked_batch_limit=2,
            tracked_evidence_upload=TrackedFrameEvidenceUploader(client, tmp_path),
        )
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await dispatcher.drain_once(limit=2)

    assert exc_info.value.response.status_code == status_code
    assert raw_calls == 0
    assert len(await outbox.ready_tracked_parts(credential, limit=2)) == 1
    assert await outbox.has_pending_tracked_frame(credential)


@pytest.mark.anyio
async def test_cancelled_evidence_upload_blocks_raw_and_preserves_outbox(tmp_path):
    single_part, context, credential, single_refs = _single_live_part(tmp_path)

    outbox = SqliteEventOutbox(tmp_path / "cancel.sqlite", clock=lambda: NOW)
    await outbox.enqueue_tracked_parts(
        credential,
        context,
        (single_part,),
        expected_seal=single_refs,
        expires_at=NOW + timedelta(seconds=30),
        max_pending=2,
        max_bytes=200_000,
    )

    raw_calls = 0

    def backend(request: httpx.Request) -> httpx.Response:
        nonlocal raw_calls
        if request.url.path == "/api/v1/integrations/ai/tracked-parts":
            raw_calls += 1
        return httpx.Response(404)

    async def cancel_upload(_cred, _wire):
        raise asyncio.CancelledError("operator shutdown during evidence upload")

    async with BackendClient(
        "https://backend.example",
        SERVICE_TOKEN,
        transport=httpx.MockTransport(backend),
    ) as client:
        dispatcher = OutboxDispatcher(
            outbox,
            client,
            generation_credential=credential,
            tracked_batch_limit=2,
            tracked_evidence_upload=cancel_upload,
        )
        with pytest.raises(asyncio.CancelledError):
            await dispatcher.drain_once(limit=2)

    assert raw_calls == 0
    assert len(await outbox.ready_tracked_parts(credential, limit=2)) == 1


@pytest.mark.anyio
async def test_no_frame_tracked_event_skips_image_upload_and_sends_raw_unchanged(tmp_path):
    _, context, credential, _, wire, _, _ = live_evidence_packet(with_frame=False)
    event = json.loads(wire)["event"]
    refs = TrackedFrameReferences.model_validate(
        {
            "scopeSha256": "a" * 64,
            "sequenceNumber": event["framePartition"]["sequenceNumber"],
            "subjects": [
                {
                    "segmentId": event["continuity"]["subjects"][0]["segmentId"],
                    "eventId": event["eventId"].lower(),
                    "personObservationIndex": 0,
                }
            ],
            "retiredSegmentIds": [],
        }
    )

    outbox = SqliteEventOutbox(tmp_path / "noframe.sqlite", clock=lambda: NOW)
    await outbox.enqueue_tracked_parts(
        credential,
        context,
        (event,),
        expected_seal=refs,
        expires_at=NOW + timedelta(seconds=30),
        max_pending=2,
        max_bytes=200_000,
    )
    ((_frame_id, wires),) = await outbox.ready_tracked_parts(credential, limit=2)

    evidence_calls = 0
    raw_calls = 0

    def backend(request: httpx.Request) -> httpx.Response:
        nonlocal evidence_calls, raw_calls
        if request.url.path == "/api/v1/integrations/ai/tracked-parts/evidence":
            evidence_calls += 1
            return httpx.Response(500)
        if request.url.path == "/api/v1/integrations/ai/tracked-parts":
            raw_calls += 1
            assert request.content == wires[0]
            digest = compute_canonical_payload_hash(json.loads(request.content)["event"])
            return httpx.Response(
                200,
                json={
                    "kind": "RAW_STORED",
                    "eventId": event["eventId"].lower(),
                    "payloadHash": digest,
                    "bindingHash": context.binding_hash(digest),
                },
            )
        return httpx.Response(404)

    async with BackendClient(
        "https://backend.example",
        SERVICE_TOKEN,
        transport=httpx.MockTransport(backend),
    ) as client:
        dispatcher = OutboxDispatcher(
            outbox,
            client,
            generation_credential=credential,
            tracked_batch_limit=2,
            tracked_evidence_upload=TrackedFrameEvidenceUploader(client, tmp_path),
        )
        await dispatcher.drain_once(limit=2)

    assert evidence_calls == 0
    assert raw_calls == 1
    assert not await outbox.ready_tracked_parts(credential, limit=2)


@pytest.mark.anyio
async def test_missing_local_frame_file_blocks_raw_in_outbox_dispatcher(tmp_path):
    single_part, context, credential, single_refs = _single_live_part(tmp_path)
    session = single_part["streamSessionId"].lower()
    seq = single_part["framePartition"]["sequenceNumber"]
    event_id = single_part["eventId"].lower()
    (tmp_path / f"{session}_{seq}_{event_id}.jpg").unlink()

    outbox = SqliteEventOutbox(tmp_path / "missing-file.sqlite", clock=lambda: NOW)
    await outbox.enqueue_tracked_parts(
        credential,
        context,
        (single_part,),
        expected_seal=single_refs,
        expires_at=NOW + timedelta(seconds=30),
        max_pending=2,
        max_bytes=200_000,
    )

    network_calls = 0

    def backend(_: httpx.Request) -> httpx.Response:
        nonlocal network_calls
        network_calls += 1
        return httpx.Response(500)

    async with BackendClient(
        "https://backend.example",
        SERVICE_TOKEN,
        transport=httpx.MockTransport(backend),
    ) as client:
        dispatcher = OutboxDispatcher(
            outbox,
            client,
            generation_credential=credential,
            tracked_batch_limit=2,
            tracked_evidence_upload=TrackedFrameEvidenceUploader(client, tmp_path),
        )
        with pytest.raises(ValueError, match="unavailable or inconsistent"):
            await dispatcher.drain_tracked_once(limit=2)

    assert network_calls == 0
    assert len(await outbox.ready_tracked_parts(credential, limit=2)) == 1


@pytest.mark.anyio
async def test_multipart_partial_evidence_failure_retries_all_parts_in_order(tmp_path):
    parts, _, _, context, credential, refs = _live_two_part_frame(tmp_path)
    outbox = SqliteEventOutbox(tmp_path / "partial-retry.sqlite", clock=lambda: NOW)
    await outbox.enqueue_tracked_parts(
        credential,
        context,
        parts,
        expected_seal=refs,
        expires_at=NOW + timedelta(seconds=30),
        max_pending=2,
        max_bytes=200_000,
    )
    ((frame_id, wires),) = await outbox.ready_tracked_parts(credential, limit=2)

    call_log: list[tuple[str, str]] = []
    evidence_attempts: list[tuple[str, str, str]] = []
    fail_part1_evidence = True

    def backend(request: httpx.Request) -> httpx.Response:
        nonlocal fail_part1_evidence
        if request.url.path == "/api/v1/integrations/ai/tracked-parts/evidence":
            fields = parse_multipart(request)
            event = json.loads(fields["event"][2].decode("utf-8"))
            eid = event["eventId"].lower()
            payload_hash = compute_canonical_payload_hash(event)
            binding_hash = context.binding_hash(payload_hash)
            img_sha = hashlib.sha256(fields["frame"][2]).hexdigest()
            call_log.append(("EVIDENCE", eid))
            evidence_attempts.append((eid, binding_hash, img_sha))
            if eid == parts[1]["eventId"].lower() and fail_part1_evidence:
                fail_part1_evidence = False
                return httpx.Response(503)
            return httpx.Response(
                200,
                json={
                    "schemaVersion": "1.0.0",
                    "kind": "EVIDENCE_STORED",
                    "eventId": eid,
                    "payloadHash": payload_hash,
                    "bindingHash": binding_hash,
                    "sha256": img_sha,
                },
            )
        if request.url.path == "/api/v1/integrations/ai/tracked-parts":
            event = json.loads(request.content)["event"]
            eid = event["eventId"].lower()
            digest = compute_canonical_payload_hash(event)
            call_log.append(("RAW", eid))
            return httpx.Response(
                200,
                json={
                    "kind": "RAW_STORED",
                    "eventId": eid,
                    "payloadHash": digest,
                    "bindingHash": context.binding_hash(digest),
                },
            )
        return httpx.Response(404)

    async with BackendClient(
        "https://backend.example",
        SERVICE_TOKEN,
        transport=httpx.MockTransport(backend),
    ) as client:
        dispatcher = OutboxDispatcher(
            outbox,
            client,
            generation_credential=credential,
            tracked_batch_limit=2,
            tracked_evidence_upload=TrackedFrameEvidenceUploader(client, tmp_path),
        )
        # Part 0 succeeds, Part 1 evidence returns 503 -> Part 1 RAW blocked, outbox not confirmed.
        await dispatcher.drain_once(limit=2)
        assert call_log == [
            ("EVIDENCE", parts[0]["eventId"].lower()),
            ("RAW", parts[0]["eventId"].lower()),
            ("EVIDENCE", parts[1]["eventId"].lower()),
        ]
        assert await outbox.ready_tracked_parts(credential, limit=2) == ((frame_id, wires),)

        # Next tick replays full sequence with identical bindings before RAW confirmation.
        await dispatcher.drain_once(limit=2)
        assert call_log == [
            ("EVIDENCE", parts[0]["eventId"].lower()),
            ("RAW", parts[0]["eventId"].lower()),
            ("EVIDENCE", parts[1]["eventId"].lower()),
            ("EVIDENCE", parts[0]["eventId"].lower()),
            ("RAW", parts[0]["eventId"].lower()),
            ("EVIDENCE", parts[1]["eventId"].lower()),
            ("RAW", parts[1]["eventId"].lower()),
        ]
        assert evidence_attempts[0] == evidence_attempts[2]
        assert evidence_attempts[1] == evidence_attempts[3]
        assert not await outbox.ready_tracked_parts(credential, limit=2)


@pytest.mark.anyio
async def test_local_evidence_publisher_artifact_uploads_end_to_end_via_uploader(tmp_path):
    single_part, context, credential, single_refs = _single_live_part(tmp_path)
    single_part["frameDimensions"] = {"width": 32, "height": 24}
    # Remove helper-written image so LocalEvidencePublisher must publish fresh artifact.
    session = single_part["streamSessionId"].lower()
    seq_int = int(single_part["framePartition"]["sequenceNumber"])
    event_id = single_part["eventId"].lower()
    published_path = tmp_path / f"{session}_{seq_int}_{event_id}.jpg"
    published_path.unlink()

    captured_at = datetime.fromisoformat(single_part["capturedAt"].replace("Z", "+00:00"))
    frame = FrameEnvelope(
        stream_id="synthetic",
        session_id=UUID(session),
        camera_external_id=single_part["cameraExternalId"],
        captured_at=captured_at,
        width=32,
        height=24,
        sequence_number=seq_int,
        payload=bytes([60, 120, 180] * (32 * 24)),
    )
    legacy_event = TechnicalObservationEvent.create(
        event_id=event_id,
        camera_external_id=single_part["cameraExternalId"],
        stream_session_id=session,
        captured_at=frame.captured_at_iso,
        frame_dimensions=FrameDimensions(width=32, height=24),
        observations=single_part["observations"][:1],
        evidence=[],
        frame_partition=ObservationFramePartition.model_validate(single_part["framePartition"]),
        schema_version="1.4.0",
    )
    publisher = LocalEvidencePublisher(root_dir=tmp_path, bind_content_hash=True)
    published_event = await publisher.publish(frame=frame, event=legacy_event)
    assert published_path.is_file()
    published_bytes = published_path.read_bytes()

    single_part["capturedAt"] = frame.captured_at_iso
    single_part["evidence"] = [
        item.model_dump(mode="json", by_alias=True, exclude_none=True)
        for item in published_event.evidence
    ]

    outbox = SqliteEventOutbox(tmp_path / "publisher-e2e.sqlite", clock=lambda: NOW)
    await outbox.enqueue_tracked_parts(
        credential,
        context,
        (single_part,),
        expected_seal=single_refs,
        expires_at=NOW + timedelta(seconds=30),
        max_pending=2,
        max_bytes=200_000,
    )

    uploaded_bodies: list[bytes] = []

    def backend(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/integrations/ai/tracked-parts/evidence":
            fields = parse_multipart(request)
            uploaded_bodies.append(fields["frame"][2])
            event = json.loads(fields["event"][2].decode("utf-8"))
            digest = hashlib.sha256(fields["frame"][2]).hexdigest()
            payload_hash = compute_canonical_payload_hash(event)
            return httpx.Response(
                200,
                json={
                    "schemaVersion": "1.0.0",
                    "kind": "EVIDENCE_STORED",
                    "eventId": event["eventId"].lower(),
                    "payloadHash": payload_hash,
                    "bindingHash": context.binding_hash(payload_hash),
                    "sha256": digest,
                },
            )
        if request.url.path == "/api/v1/integrations/ai/tracked-parts":
            event = json.loads(request.content)["event"]
            digest = compute_canonical_payload_hash(event)
            return httpx.Response(
                200,
                json={
                    "kind": "RAW_STORED",
                    "eventId": event["eventId"].lower(),
                    "payloadHash": digest,
                    "bindingHash": context.binding_hash(digest),
                },
            )
        return httpx.Response(404)

    async with BackendClient(
        "https://backend.example",
        SERVICE_TOKEN,
        transport=httpx.MockTransport(backend),
    ) as client:
        dispatcher = OutboxDispatcher(
            outbox,
            client,
            generation_credential=credential,
            tracked_batch_limit=2,
            tracked_evidence_upload=TrackedFrameEvidenceUploader(client, tmp_path),
        )
        await dispatcher.drain_once(limit=2)

    assert uploaded_bodies == [published_bytes]
    assert not await outbox.ready_tracked_parts(credential, limit=2)
