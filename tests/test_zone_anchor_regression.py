from test_zone_pipeline import batch, configuration, person

from smartsite_ai.pipelines.zones import RestrictedZonePipeline
from smartsite_ai.tracking.models import TrackedFrame, TrackedPerson


def test_box_center_confirms_occupancy_when_bottom_center_is_outside() -> None:
    centered = RestrictedZonePipeline(anchor="BOX_CENTER")
    default = RestrictedZonePipeline()
    detection = person((0.45, 0.3, 0.65, 1.0))
    for sequence in (1, 2, 3, 4):
        frame = TrackedFrame(
            batch=batch(sequence, detection),
            persons=(TrackedPerson(track_id=1, detection=detection),),
        )
        assert default.process(frame, configuration()) == ()
        entries = centered.process(frame, configuration())
        assert len(entries) == (1 if sequence == 3 else 0)
    assert centered.occupied_track_regions(
        stream_id=frame.batch.stream_id,
        session_id=frame.batch.session_id,
        active_track_ids=(1,),
    ) == frozenset({(1, configuration().regions[0].region_id)})
