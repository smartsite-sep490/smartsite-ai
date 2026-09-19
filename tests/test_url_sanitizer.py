import pytest
from pydantic import ValidationError

from smartsite_ai.ingestion.config import StreamConfig
from smartsite_ai.ingestion.source import (
    SourceConnectionError,
    classify_error_reason,
    sanitize_stream_url,
)


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


def test_sanitize_url_fail_closed_query_parameters() -> None:
    """Only allowlisted query parameters keep values; all others are masked."""
    raw = "http://camera.local/live?token=secret123&unknown_param=foo&channel=1&fps=30"
    sanitized = sanitize_stream_url(raw)
    assert "secret123" not in sanitized
    assert "foo" not in sanitized
    assert "channel=1" in sanitized
    assert "fps=30" in sanitized
    assert "token=***" in sanitized
    assert "unknown_param=***" in sanitized


def test_sanitize_url_strips_fragments_completely() -> None:
    """Fragments must be stripped completely to avoid leaking tokens embedded in #fragments."""
    raw = "http://camera.local/live?ch=0#access_token=super_secret_fragment_token"
    sanitized = sanitize_stream_url(raw)
    assert "super_secret_fragment_token" not in sanitized
    assert "#" not in sanitized
    assert sanitized == "http://camera.local/live?ch=0"


def test_sanitize_local_device_or_path() -> None:
    assert sanitize_stream_url("/dev/video0") == "/dev/video0"
    assert sanitize_stream_url("0") == "0"
    assert sanitize_stream_url("/var/videos/sample.mp4") == "/var/videos/sample.mp4"


def test_stream_config_repr_and_str_mask_source_url() -> None:
    config = StreamConfig(
        stream_id="cam-01",
        camera_external_id="CAM_EXTERNAL_01",
        source_url="rtsp://operator:p@ssword123@10.0.1.25:554/stream?token=topsecret",
    )
    repr_str = repr(config)
    str_str = str(config)
    assert "p@ssword123" not in repr_str
    assert "operator" not in repr_str
    assert "topsecret" not in repr_str
    assert "p@ssword123" not in str_str
    assert "topsecret" not in str_str
    # Verify raw URL is accessible via explicit accessor only
    assert (
        config.get_raw_source_url()
        == "rtsp://operator:p@ssword123@10.0.1.25:554/stream?token=topsecret"
    )


def test_stream_config_model_dump_does_not_leak_secret() -> None:
    raw_url = "rtsp://admin:mysecretpassword@10.0.1.50:554/live"
    config = StreamConfig(
        stream_id="cam-02",
        camera_external_id="CAM_02",
        source_url=raw_url,
    )
    dumped = config.model_dump()
    assert str(dumped["source_url"]) != raw_url
    assert "mysecretpassword" not in str(dumped["source_url"])

    json_str = config.model_dump_json()
    assert "mysecretpassword" not in json_str


def test_stream_config_validation_error_masks_credentials_fragments_and_substrings() -> None:
    """Validation errors must never leak credentials or substrings, even if truncated."""
    bad_url = "ftp://admin:super_secret_ftp_pass@camera.local/feed"
    with pytest.raises(ValidationError) as exc_info:
        StreamConfig(
            stream_id="cam-bad",
            camera_external_id="CAM_BAD",
            source_url=bad_url,
        )

    err_str = str(exc_info.value)
    err_repr = repr(exc_info.value)
    err_errors = str(exc_info.value.errors())

    for fragment in ["super_secret", "ftp_pass", "admin"]:
        assert fragment not in err_str
        assert fragment not in err_repr
        assert fragment not in err_errors


def test_stream_config_allowed_schemes() -> None:
    assert (
        StreamConfig(
            stream_id="c1", camera_external_id="e1", source_url="rtsp://10.0.0.1/live"
        ).source_url.get_secret_value()
        == "rtsp://10.0.0.1/live"
    )
    assert (
        StreamConfig(
            stream_id="c2", camera_external_id="e2", source_url="rtsps://10.0.0.1/live"
        ).source_url.get_secret_value()
        == "rtsps://10.0.0.1/live"
    )
    assert (
        StreamConfig(
            stream_id="c3", camera_external_id="e3", source_url="http://10.0.0.1/live.mjpg"
        ).source_url.get_secret_value()
        == "http://10.0.0.1/live.mjpg"
    )
    assert (
        StreamConfig(
            stream_id="c4", camera_external_id="e4", source_url="https://10.0.0.1/live.mjpg"
        ).source_url.get_secret_value()
        == "https://10.0.0.1/live.mjpg"
    )
    assert (
        StreamConfig(
            stream_id="c5", camera_external_id="e5", source_url="/var/video/sample.mp4"
        ).source_url.get_secret_value()
        == "/var/video/sample.mp4"
    )
    assert (
        StreamConfig(
            stream_id="c6", camera_external_id="e6", source_url="0"
        ).source_url.get_secret_value()
        == "0"
    )


def test_stream_config_rejects_disallowed_schemes_and_traversal() -> None:
    """Disallowed schemes, relative paths, and parent traversals must be strictly rejected."""
    disallowed_urls = [
        "ftp://host/stream",
        "javascript:alert(1)",
        "data:text/plain;base64,xyz",
        "ssh://user:pass@host",
        "file:///etc/shadow",
        "../../secret.mp4",
        "./local_video.mp4",
        "relative/path/video.mp4",
        "/var/video/../../etc/passwd.mp4",
    ]
    for bad_url in disallowed_urls:
        with pytest.raises(ValidationError):
            StreamConfig(stream_id="c_err", camera_external_id="e_err", source_url=bad_url)


def test_source_connection_error_masks_url_and_tokens() -> None:
    err = SourceConnectionError(
        "Connection refused to rtsp://admin:mysecretpass@192.168.1.5:554/stream?token=leakme123"
    )
    msg = str(err)
    assert "mysecretpass" not in msg
    assert "admin" not in msg
    assert "leakme123" not in msg
    assert "rtsp://***:***@192.168.1.5:554/stream" in msg


def test_classify_error_reason_never_echoes_arbitrary_messages() -> None:
    """Unknown exception messages must never be echoed in classified error reasons."""
    leak_msg = "Database query failed with internal password secret_pwd_456"
    err = RuntimeError(leak_msg)
    classified = classify_error_reason(err)
    assert classified == "RuntimeError: unclassified_error"
    assert "secret_pwd_456" not in classified
    assert "Database query" not in classified


def test_sanitize_url_and_message_with_at_in_password() -> None:
    """Sanitizer must handle unencoded @ character in password correctly."""
    raw_url = "rtsp://operator:p@ssword123@10.0.1.25/live"
    sanitized = sanitize_stream_url(raw_url)
    assert sanitized == "rtsp://***:***@10.0.1.25/live"
    assert "p@ssword123" not in sanitized
    assert "operator" not in sanitized

    msg = "failed rtsp://operator:p@ssword123@10.0.1.25/live"
    err = SourceConnectionError(msg)
    assert "p@ssword123" not in str(err)
    assert "operator" not in str(err)
    assert "rtsp://***:***@10.0.1.25/live" in str(err)
