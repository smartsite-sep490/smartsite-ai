import pytest

from smartsite_ai.realtime import parse_zone_polygon, safe_source_label


def test_zone_polygon_rejects_out_of_range_points() -> None:
    with pytest.raises(ValueError, match="within 0 and 1"):
        parse_zone_polygon("1.2,0.2;0.9,0.2;0.9,0.9")


def test_zone_polygon_rejects_fewer_than_three_points() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        parse_zone_polygon("0.1,0.1;0.2,0.2")


def test_zone_polygon_keeps_in_range_points() -> None:
    assert parse_zone_polygon("0,0;1,0;1,1") == [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]


def test_source_errors_hide_credentials() -> None:
    assert safe_source_label("rtsp://user:secret@camera.local/live") == "rtsp://camera.local"
    assert safe_source_label(r"D:\videos\site.mp4") == "configured source"
