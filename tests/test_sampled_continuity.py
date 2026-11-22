"""Focused regressions for synchronized sampled camera continuity."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from test_tracked_encoder import frame as encoder_frame
from test_tracking import detection

from smartsite_ai.core.canonical_hash import compute_canonical_payload_hash
from smartsite_ai.core.region_configuration_store import RegionConfigurationStore
from smartsite_ai.domain.event_delivery_context import EventDeliveryContext
from smartsite_ai.domain.policy_configuration_bundle import CameraPolicyConfigurationBundle
from smartsite_ai.domain.regions import CameraRegionConfiguration
from smartsite_ai.domain.tracked_frame_result import TrackedFrameResult
from smartsite_ai.domain.tracked_observations import TrackedObservationEvent
from smartsite_ai.domain.tracked_raw_receipt import TrackedRawReceipt
from smartsite_ai.inference.models import DetectionBatch
from smartsite_ai.ingestion.config import StreamConfig
from smartsite_ai.ingestion.envelope import FrameEnvelope
from smartsite_ai.ingestion.queue import QueueClosedError
from smartsite_ai.ingestion.status import StreamMetrics, StreamState, StreamStatus
from smartsite_ai.integrations.configuration_poller import CameraConfigurationPoller
from smartsite_ai.integrations.outbox import OutboxDispatcher, SqliteEventOutbox
from smartsite_ai.integrations.tracked_capture import stage_tracked_capture
from smartsite_ai.integrations.worker_generation import WorkerGenerationCredential
from smartsite_ai.pipelines.mf05_mf06 import Mf05Mf06Pipeline
from smartsite_ai.pipelines.ppe import PpePipeline
from smartsite_ai.pipelines.tracked_frame_encoder import encode_complete_tracked_frame
from smartsite_ai.pipelines.zones import RestrictedZonePipeline
from smartsite_ai.processing_worker import HeadlessCameraProcessingWorker, TrackedCapturePolicy
from smartsite_ai.runtime.assembly import CameraPlan, assemble_sessions
from smartsite_ai.runtime.manifest import (
    BoundCamera,
    CameraManifestEntry,
    CameraRuntimeError,
    SampledContinuityManifest,
    bind_manifest,
    parse_manifest,
)
from smartsite_ai.tracking import IoUPersonTracker
from smartsite_ai.tracking.continuity import FrameContinuityPolicy
from smartsite_ai.tracking.segment_lifecycle import SegmentLifecycle

BASE_TIME = datetime(2026, 11, 22, 7, 16, 0, tzinfo=UTC)
SESSION = UUID("00000000-0000-4000-8000-000000000001")
CAMERA_ID = UUID("11111111-1111-4111-8111-111111111111")
GENERATION_ID = UUID("22222222-2222-4222-8222-222222222222")
RECEIPT_ID = UUID("33333333-3333-4333-8333-333333333333")
SITE_ID = "44444444-4444-4444-8444-444444444444"
PPE_REGION_ID = UUID("55555555-5555-4555-8555-555555555555")
ZONE_REGION_ID = "66666666-6666-4666-8666-666666666666"
ZONE_ID = "77777777-7777-4777-8777-777777777777"
SCOPE_SHA = "a" * 64
PROFILE_SHA = "b" * 64
EXPIRES_AT = "2099-01-01T00:00:00.000Z"
_SMALL_FRAME_BYTES = b"\x00" * (64 * 48 * 3)


def sampled_batch(
    sequence_number: int,
    *detections,
    offset_ms: int = 0,
    width: int = 640,
    height: int = 480,
) -> DetectionBatch:
    return DetectionBatch(
        stream_id="stream-01",
        session_id=SESSION,
        camera_external_id="camera-01",
        captured_at=BASE_TIME + timedelta(milliseconds=offset_ms),
        frame_width=width,
        frame_height=height,
        sequence_number=sequence_number,
        model_artifact_id="fake-detector",
        model_version="test",
        model_sha256="a" * 64,
        detections=detections,
    )


def person(left: float = 0.1, right: float = 0.4):
    return detection("person", (left, 0.1, right, 0.8))


def _sampled_policy() -> FrameContinuityPolicy:
    return FrameContinuityPolicy(
        max_source_sequence_gap=64, max_capture_gap=timedelta(milliseconds=2000)
    )


def _zone_region_config() -> CameraRegionConfiguration:
    return CameraRegionConfiguration.model_validate_json(
        json.dumps(
            {
                "schemaVersion": "1.0.0",
                "configurationVersion": 1,
                "cameraExternalId": "camera-01",
                "regions": [
                    {
                        "regionId": ZONE_REGION_ID,
                        "geometryVersion": 7,
                        "coordinateSpace": "NORMALIZED_0_1",
                        "polygon": {
                            "coordinates": [
                                [0.55, 0.15],
                                [0.95, 0.15],
                                [0.95, 0.90],
                                [0.55, 0.90],
                            ]
                        },
                    }
                ],
            }
        )
    )


def _policy_bundle(*, view_revision: int = 1) -> CameraPolicyConfigurationBundle:
    raw: dict[str, object] = {
        "schemaVersion": "2.0.0",
        "cameraId": str(CAMERA_ID),
        "cameraExternalId": "camera-01",
        "configurationVersion": 1,
        "viewRevision": view_revision,
        "issuedAt": "2026-11-22T07:15:00.000Z",
        "cameraPpePolicy": {"mode": "EXPLICIT", "requiredPpe": ["HARD_HAT"]},
        "regions": [
            {
                "regionId": ZONE_REGION_ID,
                "geometryVersion": 7,
                "coordinateSpace": "NORMALIZED_0_1",
                "polygon": {
                    "coordinates": [[0.55, 0.15], [0.95, 0.15], [0.95, 0.90], [0.55, 0.90]]
                },
                "zoneId": ZONE_ID,
                "policyRevision": 1,
                "ppePolicy": {"mode": "INHERIT"},
            }
        ],
    }
    raw["bundleSha256"] = compute_canonical_payload_hash(raw)
    return CameraPolicyConfigurationBundle.model_validate_json(json.dumps(raw))


def _credential() -> WorkerGenerationCredential:
    return WorkerGenerationCredential(
        schemaVersion="1.0.0",
        cameraId=str(CAMERA_ID),
        workerGenerationId=str(GENERATION_ID),
        streamSessionId=str(SESSION),
        secret="A" * 43,
        expiresAt=EXPIRES_AT,
    )


def _delivery_context(bundle: CameraPolicyConfigurationBundle) -> EventDeliveryContext:
    return EventDeliveryContext(
        schemaVersion="1.0.0",
        cameraId=str(CAMERA_ID),
        workerGenerationId=str(GENERATION_ID),
        streamSessionId=str(SESSION),
        configurationVersion=bundle.configuration_version,
        bundleSha256=bundle.bundle_sha256,
        viewRevision=bundle.view_revision,
        receiptCommandId=str(RECEIPT_ID),
        sourceKind="LIVE",
    )


class _FakeTrackedBackend:
    def __init__(self) -> None:
        self.raw_events: list[dict[str, object]] = []
        self.stored_frames: dict[str, list[dict[str, object]]] = {}

    async def post_tracked_part(
        self, credential: WorkerGenerationCredential, wire: bytes
    ) -> TrackedRawReceipt:
        assert credential.camera_id == str(CAMERA_ID)
        envelope = json.loads(wire)
        event = envelope["event"]
        TrackedObservationEvent.model_validate(event)
        ctx = EventDeliveryContext.model_validate(envelope["context"])
        payload_hash = compute_canonical_payload_hash(event)
        frame_id = event["framePartition"]["frameId"].lower()
        self.raw_events.append(event)
        self.stored_frames.setdefault(frame_id, []).append(event)
        return TrackedRawReceipt(
            kind="RAW_STORED",
            eventId=event["eventId"],
            payloadHash=payload_hash,
            bindingHash=ctx.binding_hash(payload_hash),
        )

    async def read_tracked_frame_result(
        self,
        credential: WorkerGenerationCredential,
        frame_id: str,
        *,
        scope_sha256: str,
        sequence_number: str,
        subjects: dict[str, tuple[str, int]],
        retired_segment_ids: set[str],
    ) -> TrackedFrameResult | None:
        assert credential.camera_id == str(CAMERA_ID)
        if frame_id not in self.stored_frames:
            return None
        return TrackedFrameResult.model_validate(
            {
                "kind": "ADMITTED_TRACKED_FRAME",
                "scopeSha256": scope_sha256,
                "sequenceNumber": sequence_number,
                "subjects": [
                    {"segmentId": seg_id, "eventId": event_id, "personObservationIndex": idx}
                    for seg_id, (event_id, idx) in subjects.items()
                ],
                "retiredSegmentIds": sorted(retired_segment_ids),
            }
        )


def test_continuity_policy_defaults_bounds_sequence_and_batch_checks() -> None:
    strict = FrameContinuityPolicy()
    assert strict.max_source_sequence_gap == 1
    assert strict.max_capture_gap is None
    assert strict.allows_sequence(0, 1) is True
    assert strict.allows_sequence(0, 2) is False

    sampled = _sampled_policy()
    assert sampled.allows_sequence(0, 50) is True
    assert sampled.allows_sequence(0, 64) is True
    assert sampled.allows_sequence(0, 65) is False
    assert sampled.allows_sequence(50, 50) is False

    p0 = sampled_batch(0, person(), offset_ms=0)
    p50 = sampled_batch(50, person(), offset_ms=1613)
    p56 = sampled_batch(56, person(), offset_ms=1863)
    assert sampled.allows(p0, p50) is True
    assert sampled.allows(p50, sampled_batch(56, person(), offset_ms=3613)) is True
    assert sampled.allows(p50, sampled_batch(56, person(), offset_ms=3614)) is False
    assert sampled.allows(p50, sampled_batch(56, person(), offset_ms=1613)) is False
    assert sampled.allows(p0, sampled_batch(65, person(), offset_ms=1613)) is False
    for field, value in (
        ("camera_external_id", "camera-02"),
        ("stream_id", "stream-02"),
        ("session_id", uuid4()),
        ("model_sha256", "c" * 64),
        ("frame_width", 800),
    ):
        assert sampled.allows(p0, p56.model_copy(update={field: value})) is False

    for invalid_kwargs in (
        {"max_source_sequence_gap": 0},
        {"max_source_sequence_gap": 65},
        {"max_source_sequence_gap": True},
        {"max_source_sequence_gap": 64, "max_capture_gap": None},
        {"max_source_sequence_gap": 2, "max_capture_gap": timedelta(0)},
        {"max_source_sequence_gap": 2, "max_capture_gap": timedelta(milliseconds=2001)},
        {"max_capture_gap": timedelta(milliseconds=2001)},
    ):
        with pytest.raises((TypeError, ValueError)):
            FrameContinuityPolicy(**invalid_kwargs)


def test_strict_default_retires_skipped_sequence_and_rejects_skipped_encoder() -> None:
    tracker = IoUPersonTracker()
    first = tracker.update(sampled_batch(0, person(), offset_ms=0))
    skipped = tracker.update(sampled_batch(50, person(), offset_ms=1613))
    assert skipped.subject_segments[0].segment_id != first.subject_segments[0].segment_id
    assert skipped.subject_segments[0].reason == "FRAME_DISCONTINUITY"
    assert skipped.retired_segments[0].reason == "FRAME_DISCONTINUITY"

    lifecycle = SegmentLifecycle()
    base = encoder_frame()
    parts0, refs0 = encode_complete_tracked_frame(
        base, scope_sha256=SCOPE_SHA, profile_sha256=PROFILE_SHA
    )
    seal0 = TrackedFrameResult(kind="ADMITTED_TRACKED_FRAME", **refs0.model_dump(by_alias=True))
    skipped_frame = replace(
        base,
        event_id="00000000-0000-4000-8000-000000000050",
        source_sequence_number=50,
        subject_segments=(
            replace(base.subject_segments[0], previous_sequence_number=0, reason=None),
        ),
    )
    with pytest.raises(ValueError, match="predecessor"):
        encode_complete_tracked_frame(
            skipped_frame, scope_sha256=SCOPE_SHA, profile_sha256=PROFILE_SHA, predecessor=seal0
        )
    assert lifecycle is not None and len(parts0) == 1


def test_explicit_sampled_policy_continues_same_person_across_observed_skips() -> None:
    policy = _sampled_policy()
    tracker = IoUPersonTracker(continuity_policy=policy)
    steps = ((0, 0), (50, 1613), (56, 1863), (65, 2113), (73, 2363))
    frames = [tracker.update(sampled_batch(seq, person(), offset_ms=ms)) for seq, ms in steps]

    segment_id = frames[0].subject_segments[0].segment_id
    track_id = frames[0].persons[0].track_id
    assert [f.batch.sequence_number for f in frames] == [0, 50, 56, 65, 73]
    for prev_seq, frame in zip((None, 0, 50, 56, 65), frames, strict=True):
        claim = frame.subject_segments[0]
        assert claim.segment_id == segment_id
        assert frame.persons[0].track_id == track_id
        assert claim.previous_sequence_number == prev_seq
        assert claim.reason == ("INITIAL" if prev_seq is None else None)
        assert frame.retired_segments == ()


def test_sampled_policy_keeps_two_people_separate_and_retires_loss_or_ambiguity() -> None:
    policy = _sampled_policy()
    tracker = IoUPersonTracker(continuity_policy=policy)
    left_box, right_box = person(0.05, 0.25), person(0.70, 0.90)

    f0 = tracker.update(sampled_batch(0, left_box, right_box, offset_ms=0))
    f50 = tracker.update(sampled_batch(50, left_box, right_box, offset_ms=1613))
    f56 = tracker.update(sampled_batch(56, left_box, right_box, offset_ms=1863))
    seg_left, seg_right = (s.segment_id for s in f0.subject_segments)
    assert seg_left != seg_right
    for frame, prev_seq in ((f50, 0), (f56, 50)):
        assert [s.segment_id for s in frame.subject_segments] == [seg_left, seg_right]
        assert all(s.previous_sequence_number == prev_seq for s in frame.subject_segments)

    # Person B disappears -> seg_right retires once while seg_left continues cleanly.
    f65 = tracker.update(sampled_batch(65, left_box, offset_ms=2113))
    assert [s.segment_id for s in f65.subject_segments] == [seg_left]
    assert [(r.segment_id, r.reason) for r in f65.retired_segments] == [(seg_right, "LOST_TRACK")]

    # Ambiguous merge between two overlapping tracks retires both unambiguously.
    ambiguous = IoUPersonTracker(continuity_policy=policy)
    a0 = ambiguous.update(sampled_batch(0, person(0.10, 0.40), person(0.30, 0.60), offset_ms=0))
    a50 = ambiguous.update(sampled_batch(50, person(0.20, 0.50), offset_ms=1613))
    assert {r.segment_id for r in a50.retired_segments} == {
        s.segment_id for s in a0.subject_segments
    }
    assert all(r.reason == "AMBIGUOUS_ASSOCIATION" for r in a50.retired_segments)
    assert a50.subject_segments[0].reason == "INITIAL"


@pytest.mark.parametrize(
    "case",
    ["gap65", "time2001ms", "rewind", "model", "dims", "session"],
)
def test_sampled_policy_breaks_continuity_fail_closed_at_bounds_and_scope(case: str) -> None:
    policy = _sampled_policy()
    tracker = IoUPersonTracker(continuity_policy=policy)
    base = sampled_batch(0, person(), offset_ms=0)
    first = tracker.update(base)
    first_seg = first.subject_segments[0].segment_id

    if case == "gap65":
        nxt = sampled_batch(65, person(), offset_ms=1613)
        expected = "FRAME_DISCONTINUITY"
    elif case == "time2001ms":
        nxt = base.model_copy(
            update={
                "sequence_number": 50,
                "captured_at": BASE_TIME + timedelta(microseconds=2_000_001),
            }
        )
        expected = "FRAME_DISCONTINUITY"
    elif case == "rewind":
        nxt = base.model_copy(
            update={"sequence_number": 50, "captured_at": BASE_TIME - timedelta(milliseconds=1)}
        )
        expected = "FRAME_DISCONTINUITY"
    elif case == "model":
        nxt = sampled_batch(50, person(), offset_ms=250).model_copy(
            update={"model_sha256": "c" * 64}
        )
        expected = "PROFILE_CHANGE"
    elif case == "dims":
        nxt = sampled_batch(50, person(), offset_ms=250).model_copy(update={"frame_width": 800})
        expected = "VIEW_CHANGE"
    else:
        nxt = sampled_batch(50, person(), offset_ms=250).model_copy(update={"session_id": uuid4()})
        expected = "RECONNECT"

    result = tracker.update(nxt)
    claim = result.subject_segments[0]
    assert claim.segment_id != first_seg
    assert claim.reason == expected
    assert claim.previous_sequence_number is None
    if expected != "RECONNECT":
        assert [(r.segment_id, r.reason) for r in result.retired_segments] == [
            (first_seg, expected)
        ]
    else:
        assert result.retired_segments == ()


def test_encoder_continues_across_skipped_raw_sequences_only_from_exact_admitted_prior() -> None:
    policy = _sampled_policy()
    base = encoder_frame()
    parts0, refs0 = encode_complete_tracked_frame(
        base, scope_sha256=SCOPE_SHA, profile_sha256=PROFILE_SHA, continuity_policy=policy
    )
    seal0 = TrackedFrameResult(kind="ADMITTED_TRACKED_FRAME", **refs0.model_dump(by_alias=True))

    frame50 = replace(
        base,
        event_id="00000000-0000-4000-8000-000000000050",
        source_sequence_number=50,
        subject_segments=(
            replace(base.subject_segments[0], previous_sequence_number=0, reason=None),
        ),
    )
    parts50, refs50 = encode_complete_tracked_frame(
        frame50,
        scope_sha256=SCOPE_SHA,
        profile_sha256=PROFILE_SHA,
        predecessor=seal0,
        continuity_policy=policy,
    )
    assert parts50[0]["framePartition"]["sequenceNumber"] == "50"
    assert parts50[0]["continuity"]["subjects"][0]["link"] == {
        "kind": "CONTINUE",
        "previousEventId": parts0[0]["eventId"],
        "previousPersonObservationIndex": 0,
    }
    TrackedObservationEvent.model_validate(parts50[0])
    seal50 = TrackedFrameResult(kind="ADMITTED_TRACKED_FRAME", **refs50.model_dump(by_alias=True))

    frame56 = replace(
        base,
        event_id="00000000-0000-4000-8000-000000000056",
        source_sequence_number=56,
        subject_segments=(
            replace(base.subject_segments[0], previous_sequence_number=50, reason=None),
        ),
    )
    with pytest.raises(ValueError, match="predecessor"):
        encode_complete_tracked_frame(
            frame56,
            scope_sha256=SCOPE_SHA,
            profile_sha256=PROFILE_SHA,
            predecessor=seal0,
            continuity_policy=policy,
        )

    parts56, _ = encode_complete_tracked_frame(
        frame56,
        scope_sha256=SCOPE_SHA,
        profile_sha256=PROFILE_SHA,
        predecessor=seal50,
        continuity_policy=policy,
    )
    assert (
        parts56[0]["continuity"]["subjects"][0]["link"]["previousEventId"] == parts50[0]["eventId"]
    )
    TrackedObservationEvent.model_validate(parts56[0])

    with pytest.raises(ValueError):
        encode_complete_tracked_frame(
            frame50,
            scope_sha256=SCOPE_SHA,
            profile_sha256=PROFILE_SHA,
            predecessor=None,
            continuity_policy=policy,
        )
    wrong_scope_seal = TrackedFrameResult(
        **{**seal0.model_dump(by_alias=True), "scopeSha256": "c" * 64}
    )
    with pytest.raises(ValueError):
        encode_complete_tracked_frame(
            frame50,
            scope_sha256=SCOPE_SHA,
            profile_sha256=PROFILE_SHA,
            predecessor=wrong_scope_seal,
            continuity_policy=policy,
        )
    frame65 = replace(
        frame50,
        event_id="00000000-0000-4000-8000-000000000065",
        source_sequence_number=65,
    )
    with pytest.raises(ValueError):
        encode_complete_tracked_frame(
            frame65,
            scope_sha256=SCOPE_SHA,
            profile_sha256=PROFILE_SHA,
            predecessor=seal0,
            continuity_policy=policy,
        )


def test_zone_pipeline_sampled_continuity_and_fail_closed_resets() -> None:
    cfg = _zone_region_config()
    outside = person(0.35, 0.70)
    inside = person(0.45, 0.80)
    policy = _sampled_policy()

    tracker = IoUPersonTracker(continuity_policy=policy)
    zones = RestrictedZonePipeline(continuity_policy=policy)
    results = [
        zones.process(
            tracker.update(
                sampled_batch(seq, inside if seq > 0 else outside, offset_ms=ms),
                view_revision=1,
            ),
            cfg,
            view_revision=1,
            include_entry_origin=True,
        )
        for seq, ms in ((0, 0), (50, 1613), (56, 1863), (65, 2113))
    ]
    assert results[:3] == [(), (), ()]
    assert len(results[3]) == 1
    assert results[3][0].entry_origin == "OUTSIDE_TO_INSIDE"

    strict_tracker = IoUPersonTracker()
    strict_zones = RestrictedZonePipeline()
    strict_results = [
        strict_zones.process(
            strict_tracker.update(
                sampled_batch(seq, inside if seq > 0 else outside, offset_ms=ms),
                view_revision=1,
            ),
            cfg,
            view_revision=1,
            include_entry_origin=True,
        )
        for seq, ms in ((0, 0), (50, 1613), (51, 1863), (52, 2113))
    ]
    assert strict_results[:3] == [(), (), ()]
    assert strict_results[3][0].entry_origin == "INSIDE_ENTRY_UNOBSERVED"


@pytest.mark.parametrize("reset_mode", ["gap65", "time2001ms", "view_revision"])
def test_zone_pipeline_resets_sampled_entry_state_on_bounds_or_view_change(
    reset_mode: str,
) -> None:
    cfg = _zone_region_config()
    policy = _sampled_policy()
    tracker = IoUPersonTracker(continuity_policy=policy)
    zones = RestrictedZonePipeline(continuity_policy=policy)

    for seq, ms, box in ((0, 0, person(0.35, 0.70)), (50, 250, person(0.45, 0.80))):
        assert (
            zones.process(
                tracker.update(sampled_batch(seq, box, offset_ms=ms), view_revision=1),
                cfg,
                view_revision=1,
                include_entry_origin=True,
            )
            == ()
        )

    if reset_mode == "gap65":
        break_seq, break_ms, view = 115, 500, 1
    elif reset_mode == "time2001ms":
        break_seq, break_ms, view = 60, 2251, 1
    else:
        break_seq, break_ms, view = 60, 500, 2

    # Reset clears pending inside streak and outside baseline on break frame.
    reset_obs = [
        zones.process(
            tracker.update(
                sampled_batch(seq, person(0.45, 0.80), offset_ms=ms), view_revision=view
            ),
            cfg,
            view_revision=view,
            include_entry_origin=True,
        )
        for seq, ms in (
            (break_seq, break_ms),
            (break_seq + 1, break_ms + 250),
            (break_seq + 2, break_ms + 500),
        )
    ]
    assert reset_obs[:2] == [(), ()]
    assert len(reset_obs[2]) == 1
    assert reset_obs[2][0].entry_origin == "INSIDE_ENTRY_UNOBSERVED"


@pytest.mark.anyio
async def test_stage_tracked_capture_queries_exact_sampled_predecessor_and_fails_closed(
    tmp_path: Path,
) -> None:
    policy = _sampled_policy()
    bundle = _policy_bundle()
    credential = _credential()
    context = _delivery_context(bundle)
    outbox = SqliteEventOutbox(tmp_path / "outbox.sqlite3")
    await outbox.initialize()
    backend = _FakeTrackedBackend()

    pipeline = Mf05Mf06Pipeline(
        tracker=IoUPersonTracker(continuity_policy=policy),
        ppe=PpePipeline(),
        zones=RestrictedZonePipeline(continuity_policy=policy),
        ppe_region_id=str(PPE_REGION_ID),
        tracked_v17=True,
    )
    frame0 = pipeline.observe_frame(
        sampled_batch(0, person(), offset_ms=0),
        region_configuration=bundle,
        event_id="00000000-0000-4000-8000-000000000000",
    )
    frame50 = pipeline.observe_frame(
        sampled_batch(50, person(), offset_ms=1613),
        region_configuration=bundle,
        event_id="00000000-0000-4000-8000-000000000050",
    )
    frame56 = pipeline.observe_frame(
        sampled_batch(56, person(), offset_ms=1863),
        region_configuration=bundle,
        event_id="00000000-0000-4000-8000-000000000056",
    )

    refs0 = await stage_tracked_capture(
        outbox,
        frame0,
        context,
        credential,
        site_id=SITE_ID,
        profile_sha256=PROFILE_SHA,
        confirmed_ppe_candidates=(),
        expires_at=datetime.now(UTC) + timedelta(seconds=30),
        max_pending=8,
        max_bytes=1_000_000,
        continuity_policy=policy,
    )
    dispatcher = OutboxDispatcher(
        outbox,
        backend,  # type: ignore[arg-type]
        generation_credential=credential,
        tracked_batch_limit=4,
    )
    assert await dispatcher.drain_tracked_once(limit=4) == 1
    assert await outbox.reconcile_tracked_frames(backend, credential, limit=4) == 1  # type: ignore[arg-type]

    queried_sequences: list[tuple[str, str]] = []
    real_read = outbox.read_tracked_predecessor

    async def spy_read(scope: str, expected_seq: str) -> TrackedFrameResult | None:
        queried_sequences.append((scope, expected_seq))
        return await real_read(scope, expected_seq)

    outbox.read_tracked_predecessor = spy_read  # type: ignore[method-assign]

    # Rejects same skipped continuation under strict default policy.
    with pytest.raises(ValueError, match="sequence bound"):
        await stage_tracked_capture(
            outbox,
            frame50,
            context,
            credential,
            site_id=SITE_ID,
            profile_sha256=PROFILE_SHA,
            confirmed_ppe_candidates=(),
            expires_at=datetime.now(UTC) + timedelta(seconds=30),
            max_pending=8,
            max_bytes=1_000_000,
        )

    # Accepts sampled seq 50 -> queries exact previous_sequence_number "0" (not "49").
    await stage_tracked_capture(
        outbox,
        frame50,
        context,
        credential,
        site_id=SITE_ID,
        profile_sha256=PROFILE_SHA,
        confirmed_ppe_candidates=(),
        expires_at=datetime.now(UTC) + timedelta(seconds=30),
        max_pending=8,
        max_bytes=1_000_000,
        continuity_policy=policy,
    )
    assert queried_sequences == [(refs0.scope_sha256, "0")]
    assert await dispatcher.drain_tracked_once(limit=4) == 1
    event0 = backend.raw_events[0]
    event50 = backend.raw_events[-1]
    assert event50["framePartition"]["sequenceNumber"] == "50"  # type: ignore[index]
    assert event50["continuity"]["subjects"][0]["link"] == {  # type: ignore[index]
        "kind": "CONTINUE",
        "previousEventId": event0["eventId"],
        "previousPersonObservationIndex": 0,
    }

    # Fails closed when claiming previous_sequence_number=50 while sealed head is still "0".
    with pytest.raises(ValueError, match="predecessor"):
        await stage_tracked_capture(
            outbox,
            frame56,
            context,
            credential,
            site_id=SITE_ID,
            profile_sha256=PROFILE_SHA,
            confirmed_ppe_candidates=(),
            expires_at=datetime.now(UTC) + timedelta(seconds=30),
            max_pending=8,
            max_bytes=1_000_000,
            continuity_policy=policy,
        )
    assert queried_sequences[-1] == (refs0.scope_sha256, "50")

    # Rejects gap 65 at stage boundary before touching outbox head.
    frame65 = replace(
        frame50,
        event_id="00000000-0000-4000-8000-000000000065",
        source_sequence_number=65,
    )
    with pytest.raises(ValueError, match="sequence bound"):
        await stage_tracked_capture(
            outbox,
            frame65,
            context,
            credential,
            site_id=SITE_ID,
            profile_sha256=PROFILE_SHA,
            confirmed_ppe_candidates=(),
            expires_at=datetime.now(UTC) + timedelta(seconds=30),
            max_pending=8,
            max_bytes=1_000_000,
            continuity_policy=policy,
        )


def test_manifest_sampled_continuity_opt_in_and_assembly_wiring(tmp_path: Path) -> None:
    model = tmp_path / "model.json"
    model.write_text("{}", encoding="utf-8")
    cred_file = tmp_path / "generation.json"
    cred_file.write_text("{}", encoding="utf-8")
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()

    def base_doc(sampled: object = ..., *, live: bool = True, tracked: bool = True) -> dict:
        cam: dict[str, object] = {
            "streamId": "gate-01",
            "cameraId": str(CAMERA_ID),
            "cameraExternalId": "camera-01",
            "ppeRegionId": str(PPE_REGION_ID),
            "source": {"kind": "env", "name": "SMARTSITE_AI_GATE_01"},
            "live": live,
            "outbox": str(tmp_path / "outbox.sqlite3"),
        }
        if tracked:
            cam["evidenceDir"] = str(evidence_dir)
            cam["generationCredentialFile"] = str(cred_file)
            cam["trackedV17"] = {
                "siteId": SITE_ID,
                "profileSha256": PROFILE_SHA,
                "pendingSeconds": 10.0,
                "predecessorWaitSeconds": 2.0,
                "maxPending": 8,
                "maxBytes": 1_000_000,
            }
        if sampled is not ...:
            cam["sampledContinuity"] = sampled
        return {"schemaVersion": "1", "modelSpec": str(model), "cameras": [cam]}

    # Default absent -> None on manifest and strict default in assembled session.
    default_manifest = parse_manifest(base_doc())
    assert default_manifest.cameras[0].sampled_continuity is None

    # Valid opt-in wires exact FrameContinuityPolicy across tracker, zones, and worker.
    sampled_manifest = parse_manifest(
        base_doc({"maxSourceSequenceGap": 64, "maxCaptureGapMs": 2000})
    )
    assert sampled_manifest.cameras[0].sampled_continuity is not None

    for label, manifest in (("default", default_manifest), ("sampled", sampled_manifest)):
        bound = bind_manifest(manifest, {"SMARTSITE_AI_GATE_01": "rtsp://example.local/live"})
        store = RegionConfigurationStore(allow_policy_v2=True)
        client = SimpleNamespace()
        poller = CameraConfigurationPoller(
            client=client,  # type: ignore[arg-type]
            store=store,
            camera_id=CAMERA_ID,
            camera_external_id="camera-01",
            require_bundle=True,
        )
        sessions = assemble_sessions(
            [
                CameraPlan(
                    bound=bound.cameras[0],
                    store=store,
                    poller=poller,
                    configuration_client=client,
                    generation=_credential(),
                )
            ],  # type: ignore[arg-type]
            SimpleNamespace(),  # type: ignore[arg-type]
            backend_client_factory=lambda: SimpleNamespace(),  # type: ignore[arg-type,return-value]
            source_factory=lambda cfg: SimpleNamespace(
                config=cfg, source_id="s", is_connected=False
            ),
            evidence_max_bytes=1024,
            delivery_interval_seconds=0.25,
        )
        worker = sessions[0].worker
        expected = _sampled_policy() if label == "sampled" else None
        assert worker._tracked_capture.continuity_policy == expected  # type: ignore[attr-defined]
        assert worker._pipeline._zones._continuity_policy == expected  # type: ignore[attr-defined]
        tracker_policy = worker._pipeline._tracker._segments._continuity_policy  # type: ignore[attr-defined]
        assert tracker_policy == (expected or FrameContinuityPolicy())

    # Reject non-live, non-tracked, invalid bounds/types, and extra/missing keys.
    bad_docs = [
        base_doc({"maxSourceSequenceGap": 64, "maxCaptureGapMs": 2000}, live=False, tracked=False),
        base_doc({"maxSourceSequenceGap": 64, "maxCaptureGapMs": 2000}, live=True, tracked=False),
        base_doc({"maxSourceSequenceGap": 0, "maxCaptureGapMs": 2000}),
        base_doc({"maxSourceSequenceGap": 65, "maxCaptureGapMs": 2000}),
        base_doc({"maxSourceSequenceGap": True, "maxCaptureGapMs": 2000}),
        base_doc({"maxSourceSequenceGap": 64.0, "maxCaptureGapMs": 2000}),
        base_doc({"maxSourceSequenceGap": 64}),
        base_doc({"maxSourceSequenceGap": 64, "maxCaptureGapMs": 0}),
        base_doc({"maxSourceSequenceGap": 64, "maxCaptureGapMs": 2001}),
        base_doc({"maxSourceSequenceGap": 64, "maxCaptureGapMs": False}),
        base_doc({"maxSourceSequenceGap": 64, "maxCaptureGapMs": 2000.0}),
        base_doc({"maxCaptureGapMs": 2000}),
        base_doc({"maxSourceSequenceGap": 64, "maxCaptureGapMs": 2000, "extra": 1}),
    ]
    for doc in bad_docs:
        with pytest.raises(CameraRuntimeError):
            parse_manifest(doc)


def test_continuity_policy_type_guards_and_assembly_model_construct_rejection(
    tmp_path: Path,
) -> None:
    bad = SimpleNamespace(max_source_sequence_gap=64, max_capture_gap=timedelta(seconds=2))
    with pytest.raises(ValueError, match="policy"):
        SegmentLifecycle(continuity_policy=bad)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="policy"):
        IoUPersonTracker(continuity_policy=bad)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="policy"):
        RestrictedZonePipeline(continuity_policy=bad)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="policy"):
        TrackedCapturePolicy(
            _credential(),
            SITE_ID,
            PROFILE_SHA,
            10.0,
            2.0,
            8,
            1_000_000,
            continuity_policy=bad,  # type: ignore[arg-type]
        )

    sampled = SampledContinuityManifest(maxSourceSequenceGap=64, maxCaptureGapMs=2000)
    for live in (False, True):
        entry = CameraManifestEntry.model_construct(
            stream_id="gate-01",
            camera_id=CAMERA_ID,
            camera_external_id="camera-01",
            ppe_region_id=PPE_REGION_ID,
            live=live,
            tracked_v17=None,
            sampled_continuity=sampled,
            outbox=str(tmp_path / "outbox.sqlite3"),
            evidence_dir=None,
            partitioned_frame_delivery=False,
            generation_bound_event_delivery=False,
            private_evidence_upload=False,
        )
        bound = BoundCamera(
            entry,
            source="rtsp://example.local/live",
            source_config=StreamConfig(
                stream_id="gate-01",
                camera_external_id="camera-01",
                source_url="rtsp://example.local/live",
                is_live=live,
            ),
            outbox=tmp_path / "outbox.sqlite3",
            evidence_dir=None,
        )
        store = RegionConfigurationStore()
        client = SimpleNamespace()
        poller = CameraConfigurationPoller(
            client=client,  # type: ignore[arg-type]
            store=store,
            camera_id=CAMERA_ID,
            camera_external_id="camera-01",
        )
        with pytest.raises(CameraRuntimeError, match="tracked LIVE camera"):
            assemble_sessions(
                [
                    CameraPlan(
                        bound=bound,
                        store=store,
                        poller=poller,
                        configuration_client=client,  # type: ignore[arg-type]
                    )
                ],
                SimpleNamespace(),  # type: ignore[arg-type]
                backend_client_factory=lambda: SimpleNamespace(),  # type: ignore[arg-type,return-value]
                source_factory=lambda cfg: cfg,
                evidence_max_bytes=1024,
                delivery_interval_seconds=0.25,
            )


def _worker_frame(seq: int, offset_ms: int) -> FrameEnvelope:
    return FrameEnvelope(
        stream_id="stream-01",
        session_id=SESSION,
        camera_external_id="camera-01",
        captured_at=BASE_TIME + timedelta(milliseconds=offset_ms),
        width=64,
        height=48,
        sequence_number=seq,
        payload=_SMALL_FRAME_BYTES,
    )


async def _run_worker_sequence(
    tmp_path: Path,
    bundle: CameraPolicyConfigurationBundle,
    policy: FrameContinuityPolicy,
    frames_with_detections: list[tuple[FrameEnvelope, tuple]],
) -> list[dict[str, object]]:
    store = RegionConfigurationStore(allow_policy_v2=True)
    await store.apply_bundle(bundle)
    outbox = SqliteEventOutbox(tmp_path / "outbox.sqlite3")
    credential = _credential()
    context = _delivery_context(bundle)
    backend = _FakeTrackedBackend()
    detections_by_seq = {frame.sequence_number: dets for frame, dets in frames_with_detections}
    queue: asyncio.Queue[FrameEnvelope | None] = asyncio.Queue()
    for frame, _ in frames_with_detections:
        queue.put_nowait(frame)
    queue.put_nowait(None)

    stream_cfg = StreamConfig(
        stream_id="stream-01",
        camera_external_id="camera-01",
        source_url="rtsp://example.local/live",
        is_live=True,
    )

    class _QueueStream:
        config = stream_cfg

        async def start(self) -> None:
            return None

        async def stop(self) -> None:
            return None

        async def get_frame(self) -> FrameEnvelope:
            item = await queue.get()
            if item is None:
                raise QueueClosedError("stream ended")
            return item

        def snapshot(self) -> StreamStatus:
            return StreamStatus(
                stream_id="stream-01",
                camera_external_id="camera-01",
                sanitized_url="rtsp://example.local/live",
                state=StreamState.STOPPED,
                metrics=StreamMetrics(),
                queue_size=0,
                queue_capacity=1,
            )

    class _Detector:
        async def detect(self, frame: FrameEnvelope) -> DetectionBatch:
            return sampled_batch(
                frame.sequence_number,
                *detections_by_seq[frame.sequence_number],
                offset_ms=int((frame.captured_at - BASE_TIME).total_seconds() * 1000),
                width=frame.width,
                height=frame.height,
            )

    pipeline = Mf05Mf06Pipeline(
        tracker=IoUPersonTracker(continuity_policy=policy),
        ppe=PpePipeline(),
        zones=RestrictedZonePipeline(continuity_policy=policy),
        ppe_region_id=str(PPE_REGION_ID),
        tracked_v17=True,
    )
    dispatcher = OutboxDispatcher(
        outbox,
        backend,  # type: ignore[arg-type]
        generation_credential=credential,
        tracked_batch_limit=4,
    )
    worker = HeadlessCameraProcessingWorker(
        stream=_QueueStream(),  # type: ignore[arg-type]
        detector=_Detector(),  # type: ignore[arg-type]
        pipeline=pipeline,
        ppe_region_id=str(PPE_REGION_ID),
        configurations=store,
        outbox=outbox,
        dispatcher=dispatcher,
        delivery_interval_seconds=0.01,
        tracked_capture=TrackedCapturePolicy(
            credential=credential,
            site_id=SITE_ID,
            profile_sha256=PROFILE_SHA,
            pending_seconds=10.0,
            predecessor_wait_seconds=2.0,
            max_pending=8,
            max_bytes=1_000_000,
            continuity_policy=policy,
        ),
        delivery_context_provider=lambda _cfg, _f, _kind: asyncio.sleep(0, result=context),
    )
    result = await worker.run()
    assert result.frames_processed == len(frames_with_detections)
    return backend.raw_events


@pytest.mark.anyio
async def test_worker_sampled_continuity_temporal_ppe_zone_and_two_people(
    tmp_path: Path,
) -> None:
    policy = _sampled_policy()
    bundle = _policy_bundle()
    p1 = person(0.05, 0.25)
    no_hat = detection("no helmet", (0.08, 0.12, 0.22, 0.25))
    p2_out = person(0.35, 0.70)
    hat_out = detection("helmet", (0.45, 0.12, 0.60, 0.25))
    p2_in = person(0.45, 0.80)
    hat_in = detection("helmet", (0.55, 0.12, 0.70, 0.25))

    frames = [
        (_worker_frame(0, 0), (p1, no_hat, p2_out, hat_out)),
        (_worker_frame(50, 1613), (p1, no_hat, p2_in, hat_in)),
        (_worker_frame(56, 1863), (p1, no_hat, p2_in, hat_in)),
        (_worker_frame(65, 2113), (p1, no_hat, p2_in, hat_in)),
    ]
    events = await _run_worker_sequence(tmp_path, bundle, policy, frames)
    assert [e["framePartition"]["sequenceNumber"] for e in events] == ["0", "50", "56", "65"]  # type: ignore[index]

    # Two distinct subjects never merge and each continues from exact prior raw frame.
    seg0_ids = [s["segmentId"] for s in events[0]["continuity"]["subjects"]]  # type: ignore[index]
    assert len(set(seg0_ids)) == 2
    for prev_event, curr_event in zip(events[:-1], events[1:], strict=True):
        curr_subs = curr_event["continuity"]["subjects"]  # type: ignore[index]
        assert [s["segmentId"] for s in curr_subs] == seg0_ids
        assert [s["link"]["previousEventId"] for s in curr_subs] == [
            prev_event["eventId"],
            prev_event["eventId"],
        ]

    # Temporal PPE missing confirms after 3 sampled frames across 1613ms warmup (seq 56),
    # and Zone crossing confirms OUTSIDE_TO_INSIDE after 3 inside frames (seq 65).
    missing_ppe_seqs = [
        e["framePartition"]["sequenceNumber"]  # type: ignore[index]
        for e in events
        if any(o["type"] == "PPE" and o["status"] == "MISSING" for o in e["observations"])  # type: ignore[attr-defined]
    ]
    assert missing_ppe_seqs == ["56"]
    zone_entries = [
        (e["framePartition"]["sequenceNumber"], o["entryOrigin"])  # type: ignore[index]
        for e in events
        for o in e["observations"]  # type: ignore[attr-defined]
        if o["type"] == "ZONE_ENTRY"
    ]
    assert zone_entries == [("65", "OUTSIDE_TO_INSIDE")]


@pytest.mark.anyio
async def test_worker_sampled_discontinuity_resets_temporal_ppe_and_zone(
    tmp_path: Path,
) -> None:
    policy = _sampled_policy()
    bundle = _policy_bundle()
    p1 = person(0.05, 0.25)
    no_hat = detection("no helmet", (0.08, 0.12, 0.22, 0.25))
    p2_out = person(0.35, 0.70)
    p2_in = person(0.45, 0.80)

    # Discontinuity at seq 115 (gap 65 > 64) resets state; post-reset gate still keeps >1s expiry.
    frames = [
        (_worker_frame(0, 0), (p1, no_hat, p2_out)),
        (_worker_frame(50, 250), (p1, no_hat, p2_in)),
        (_worker_frame(115, 500), (p1, no_hat, p2_in)),
        (_worker_frame(120, 2113), (p1, no_hat, p2_in)),
        (_worker_frame(125, 2363), (p1, no_hat, p2_in)),
    ]
    events = await _run_worker_sequence(tmp_path, bundle, policy, frames)
    by_seq = {e["framePartition"]["sequenceNumber"]: e for e in events}  # type: ignore[index]

    assert all(
        s["link"] == {"kind": "START", "reason": "FRAME_DISCONTINUITY"}
        for s in by_seq["115"]["continuity"]["subjects"]  # type: ignore[index]
    )
    missing_ppe_seqs = [
        seq
        for seq, e in by_seq.items()
        if any(o["type"] == "PPE" and o["status"] == "MISSING" for o in e["observations"])  # type: ignore[attr-defined]
    ]
    assert missing_ppe_seqs == ["125"]
    zone_entries = [
        (seq, o["entryOrigin"])
        for seq, e in by_seq.items()
        for o in e["observations"]  # type: ignore[attr-defined]
        if o["type"] == "ZONE_ENTRY"
    ]
    assert zone_entries == [("125", "INSIDE_ENTRY_UNOBSERVED")]
