from smartsite_ai.ingestion.config import StreamConfig
from smartsite_ai.ingestion.source import SourceConnectionError, sanitize_stream_url


def test_sanitize_rtsp_url_with_credentials() -> None:
    raw = "rtsp://admin:supersecret@192.168.1.100:554/live/ch0"
    sanitized = sanitize_stream_url(raw)
    assert sanitized == "rtsp://***:***@192.168.1.100:554/live/ch0"
    assert "supersecret" not in sanitized
    assert "admin" not in sanitized


def test_sanitize_rtsp_url_with_user_only() -> None:
    raw = "rtsp://viewer@192.168.1.100:554/live"
    sanitized = sanitize_stream_url(raw)
    assert sanitized == "rtsp://***@192.168.1.100:554/live"
    assert "viewer" not in sanitized


def test_sanitize_rtsp_url_without_credentials() -> None:
    raw = "rtsp://192.168.1.100:554/live/stream1"
    assert sanitize_stream_url(raw) == "rtsp://192.168.1.100:554/live/stream1"


def test_sanitize_http_stream_url() -> None:
    raw = "http://user:pass@camera.local:8080/mjpg"
    sanitized = sanitize_stream_url(raw)
    assert sanitized == "http://***:***@camera.local:8080/mjpg"
    assert "pass" not in sanitized


def test_sanitize_local_device_or_path() -> None:
    assert sanitize_stream_url("/dev/video0") == "/dev/video0"
    assert sanitize_stream_url("0") == "0"
    assert sanitize_stream_url("C:\\videos\\sample.mp4") == "C:\\videos\\sample.mp4"


def test_stream_config_repr_masks_source_url() -> None:
    config = StreamConfig(
        stream_id="cam-01",
        camera_external_id="CAM_EXTERNAL_01",
        source_url="rtsp://operator:p@ssword123@10.0.1.25:554/stream",
    )
    repr_str = repr(config)
    str_str = str(config)
    assert "p@ssword123" not in repr_str
    assert "operator" not in repr_str
    assert "p@ssword123" not in str_str
    assert "rtsp://***:***@10.0.1.25:554/stream" in repr_str
    # Verify raw URL is still accessible for actual connection
    assert config.get_raw_source_url() == "rtsp://operator:p@ssword123@10.0.1.25:554/stream"


def test_source_connection_error_masks_url() -> None:
    err = SourceConnectionError(
        "Connection refused to rtsp://admin:mysecretpass@192.168.1.5:554/stream"
    )
    msg = str(err)
    assert "mysecretpass" not in msg
    assert "admin" not in msg
    assert "rtsp://***:***@192.168.1.5:554/stream" in msg
