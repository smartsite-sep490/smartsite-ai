from pathlib import Path

import pytest
from pydantic import SecretStr

from smartsite_ai.source_control_models import SourceDraft, SourcePolicy, SourceRejected


def policy(tmp_path: Path) -> SourcePolicy:
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"permitted fixture, not decoded")
    return SourcePolicy(
        webcams={"usb-front": 0}, videos={"sample": video}, networks=("192.168.10.0/24",)
    )


def test_host_webcam_and_approved_video_are_not_urls(tmp_path):
    p = policy(tmp_path)
    assert p.resolve(SourceDraft(kind="WEBCAM", deviceId="usb-front")) == "0"
    assert p.resolve(SourceDraft(kind="VIDEO", assetId="sample")) == str(tmp_path / "sample.mp4")
    for draft in [
        SourceDraft(kind="WEBCAM", deviceId="other"),
        SourceDraft(kind="VIDEO", assetId="../secret"),
    ]:
        with pytest.raises(SourceRejected):
            p.resolve(draft)


def test_rtsp_requires_approved_literal_host_and_separate_write_only_credentials(tmp_path):
    draft = SourceDraft(
        kind="RTSP",
        endpoint=SecretStr("rtsp://192.168.10.25:554/stream1"),
        username=SecretStr("camera"),
        password=SecretStr("not-a-real-pass"),
    )
    assert (
        policy(tmp_path).resolve(draft) == "rtsp://camera:not-a-real-pass@192.168.10.25:554/stream1"
    )
    assert "not-a-real-pass" not in repr(draft)
    for endpoint in [
        "http://192.168.10.25/x",
        "rtsp://127.0.0.1/x",
        "rtsp://169.254.169.254/x",
        "rtsp://8.8.8.8/x",
        "rtsp://camera.example/x",
        "rtsp://u:p@192.168.10.25/x",
    ]:
        with pytest.raises(SourceRejected):
            policy(tmp_path).resolve(SourceDraft(kind="RTSP", endpoint=SecretStr(endpoint)))


def test_rtsp_validation_error_never_echoes_secret_or_endpoint(tmp_path):
    with pytest.raises(SourceRejected) as error:
        policy(tmp_path).resolve(
            SourceDraft(kind="RTSP", endpoint=SecretStr("rtsp://outside:bad-secret@127.0.0.1/x"))
        )
    assert str(error.value) == "SOURCE_NOT_ALLOWED"
    assert "bad-secret" not in str(error.value)
