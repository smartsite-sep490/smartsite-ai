from uvicorn.config import Config


def test_uvicorn_runtime_has_a_websocket_protocol() -> None:
    config = Config("smartsite_ai.app:create_app", factory=True, ws="auto")

    config.load()

    assert config.ws_protocol_class is not None
    assert config.ws_protocol_class.__module__.startswith("uvicorn.protocols.websockets.")
