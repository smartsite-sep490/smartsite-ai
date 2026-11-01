"""Original part-local PERSON mapping, not wire activation or identity qualification."""

from dataclasses import replace
from uuid import UUID, uuid4

import pytest
from test_mf05_mf06_pipeline import batch, configuration, detection, make_pipeline

from smartsite_ai.domain.observations import PpeObservation, TechnicalObservationEvent
from smartsite_ai.tracking.models import SubjectSegment


def observed(sequence=1):
    frame = make_pipeline().observe(
        batch(sequence, detection("person", (0.1, 0.1, 0.3, 0.8))),
        region_configuration=configuration(),
        event_id="00000000-0000-4000-8000-000000000009",
    )
    assert frame is not None
    return frame


def test_pipeline_keeps_internal_tracking_context_without_changing_event_bytes():
    pipeline = make_pipeline()
    frames = [
        pipeline.observe(
            batch(sequence, detection("person", (0.1, 0.1, 0.3, 0.8))),
            region_configuration=configuration(),
            event_id=f"00000000-0000-4000-8000-{sequence:012d}",
        )
        for sequence in (1, 2)
    ]
    first, second = frames
    assert first is not None and second is not None
    assert first.source_sequence_number == 1 and second.source_sequence_number == 2
    assert first.subject_segments[0].segment_id == second.subject_segments[0].segment_id
    assert second.subject_segments[0].previous_sequence_number == 1
    legacy = replace(second, source_sequence_number=None, subject_segments=(), retired_segments=())
    assert second.to_event().to_wire_dict() == legacy.to_event().to_wire_dict()


def test_many_people_partition_references_use_each_original_person_index():
    frame = observed()
    template = next(o for o in frame.observations if o.type == "PERSON")
    limit = next(
        m.max_length
        for m in TechnicalObservationEvent.model_fields["observations"].metadata
        if hasattr(m, "max_length")
    )
    people = tuple(template.model_copy(update={"track_id": i}) for i in range(1, limit + 2))
    claims = tuple(SubjectSegment(i, uuid4(), None, "INITIAL") for i in range(1, limit + 2))
    frame = replace(frame, observations=people, subject_segments=claims)
    packed = frame.pack_with_tracking(sequence_number=1)
    assert len(packed) == 2
    assert tuple(p.event.to_wire_dict() for p in packed) == tuple(
        p.to_wire_dict() for p in frame.to_partitioned_events(sequence_number=1)
    )
    references = [reference for part in packed for reference in part.subjects]
    assert len(references) == len(claims)
    assert {r.subject.segment_id for r in references} == {c.segment_id for c in claims}
    for part in packed:
        for reference in part.subjects:
            person = part.event.observations[reference.person_observation_index]
            assert person.type == "PERSON" and person.track_id == reference.subject.track_id
            assert reference.event_id == part.event.event_id


def test_split_large_subject_repeats_same_segment_without_reindexing_evidence():
    frame = observed()
    person = next(o for o in frame.observations if o.type == "PERSON")
    limit = next(
        m.max_length
        for m in TechnicalObservationEvent.model_fields["observations"].metadata
        if hasattr(m, "max_length")
    )
    findings = tuple(
        PpeObservation.model_validate(
            {
                "type": "PPE",
                "trackId": person.track_id,
                "regionId": str(UUID(int=i)),
                "geometryVersion": 1,
                "ppeItem": "HARD_HAT",
                "status": "MISSING",
                "confidence": 0.9,
            }
        )
        for i in range(1, limit + 1)
    )
    frame = replace(frame, observations=(person, *findings))
    packed = frame.pack_with_tracking(sequence_number=1)
    assert len(packed) == 2
    for part in packed:
        assert len(part.subjects) == 1
        assert part.subjects[0].person_observation_index == 0
        assert part.subjects[0].subject == frame.subject_segments[0]
    assert sum(sum(o.type == "PPE" for o in p.event.observations) for p in packed) == limit


def test_mapping_refuses_missing_mismatched_or_duplicated_claims():
    frame = observed()
    claim = frame.subject_segments[0]
    for claims in [(), (replace(claim, track_id=99),), (claim, claim)]:
        with pytest.raises(ValueError):
            replace(frame, subject_segments=claims).pack_with_tracking(sequence_number=1)
    with pytest.raises(ValueError):
        frame.pack_with_tracking(sequence_number=2)


def test_pipeline_preserves_retirement_while_another_person_remains():
    pipeline = make_pipeline()
    left = detection("person", (0.1, 0.1, 0.3, 0.8))
    right = detection("person", (0.7, 0.1, 0.9, 0.8))
    first = pipeline.observe(
        batch(1, left, right),
        region_configuration=configuration(),
        event_id="00000000-0000-4000-8000-000000000001",
    )
    second = pipeline.observe(
        batch(2, left),
        region_configuration=configuration(),
        event_id="00000000-0000-4000-8000-000000000002",
    )
    assert first is not None and second is not None
    assert [(s.segment_id, s.reason) for s in second.retired_segments] == [
        (first.subject_segments[1].segment_id, "LOST_TRACK")
    ]
    assert (
        second.pack_with_tracking(sequence_number=2)[0].retired_segments == second.retired_segments
    )


def test_ten_people_six_ppe_findings_keep_three_subject_references():
    frame = observed()
    template = next(o for o in frame.observations if o.type == "PERSON")
    people = tuple(template.model_copy(update={"track_id": i}) for i in range(1, 11))
    claims = tuple(SubjectSegment(i, uuid4(), None, "INITIAL") for i in range(1, 11))
    findings = tuple(
        PpeObservation.model_validate(
            {
                "type": "PPE",
                "trackId": i,
                "regionId": str(UUID(int=1)),
                "geometryVersion": 1,
                "ppeItem": item,
                "status": "MISSING",
                "confidence": 0.9,
            }
        )
        for i in (1, 2, 3)
        for item in ("HARD_HAT", "SAFETY_VEST")
    )
    parts = replace(
        frame, observations=(*people, *findings), subject_segments=claims
    ).pack_with_tracking(sequence_number=1)
    by_track = {r.subject.track_id: r.subject.segment_id for part in parts for r in part.subjects}
    assert len(by_track) == 10
    affected = {
        by_track[o.track_id] for part in parts for o in part.event.observations if o.type == "PPE"
    }
    assert affected == {c.segment_id for c in claims[:3]}
    assert len(affected) == 3
