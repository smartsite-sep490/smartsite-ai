from fastapi.testclient import TestClient

from smartsite_ai.app import create_app
from smartsite_ai.config import Settings


def test_production_cannot_use_legacy_query_token_to_start_preview(monkeypatch):
    starts = []

    async def produce(sink, settings):
        starts.append(True)
        await sink.accept()
        await sink.send_json({"type": "frame"})
        await sink.close()

    monkeypatch.setattr("smartsite_ai.app.stream_realtime", produce)
    app = create_app(
        Settings(
            _env_file=None,
            environment="production",
            realtime_model_path="not-loaded.pt",
            realtime_source="0",
            backend_service_token="fixture-token",
        )
    )
    with (
        TestClient(app) as client,
        client.websocket_connect("/ws/realtime?token=fixture-token") as socket,
    ):
        assert socket.receive_json() == {
            "type": "error",
            "message": "Use the authorized Backend preview",
        }
    assert starts == []


def test_terminal_shared_producer_cannot_be_restarted_through_legacy_socket(monkeypatch):
    starts = []

    async def produce(sink, settings):
        starts.append(True)
        await sink.accept()
        await sink.send_json({"type": "frame"})
        await sink.close()

    monkeypatch.setattr("smartsite_ai.app.stream_realtime", produce)
    app = create_app(
        Settings(
            _env_file=None,
            realtime_snapshot_enabled=True,
            realtime_model_path="not-loaded.pt",
            realtime_source="0",
            backend_service_token="fixture-token",
        )
    )
    with TestClient(app) as client:
        response = client.get(
            "/v1/realtime/preview/latest?cameraExternalId=ppe-demo",
            headers={"Authorization": "Bearer fixture-token"},
        )
        assert response.status_code == 503
        assert app.state.preview_hub.started
        with client.websocket_connect("/ws/realtime?token=fixture-token") as socket:
            assert socket.receive_json() == {
                "type": "error",
                "message": "Use the authorized Backend preview",
            }
    assert starts == [True]
