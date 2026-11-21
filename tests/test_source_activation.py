import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from cryptography.fernet import Fernet
from pydantic import SecretStr

from smartsite_ai.source_activation import SourceActivationError, SourceActivator, SourceRegistry

SCOPE = ("11111111-1111-4111-8111-111111111111", "camera-a")
OLD = "22222222-2222-4222-8222-222222222222"
COMMAND = "33333333-3333-4333-8333-333333333333"
RECEIPT = "44444444-4444-4444-8444-444444444444"


class Driver:
    def __init__(self):
        self.calls = []
        self.running = True

    def supports(self, scope):
        return scope == SCOPE

    async def generation_id(self, scope):
        return OLD

    async def stop(self, scope):
        self.calls.append("stop")
        self.running = False

    async def start(self, scope, source, generation):
        assert not self.running
        self.calls.append(("start", source, generation["workerGenerationId"]))
        self.running = True

    async def is_running(self, scope, generation_id):
        return self.running


def test_receipt_bound_activation_is_encrypted_durable_and_idempotent(tmp_path):
    async def run():
        key = SecretStr(Fernet.generate_key().decode())
        path = tmp_path / "sources.enc"
        driver = Driver()
        store = SourceRegistry(path, key)
        active = SourceActivator(store, driver)
        active.remember(
            SCOPE, RECEIPT, "rtsp://operator:private-password@192.168.1.50/live", "LIVE"
        )
        prepared = await active.prepare(SCOPE, COMMAND, RECEIPT, 0)
        assert prepared["state"] == "PREPARED"
        assert prepared["currentGenerationId"] == OLD
        assert driver.calls == ["stop"]
        generation = {
            "workerGenerationId": "55555555-5555-4555-8555-555555555555",
            "streamSessionId": "66666666-6666-4666-8666-666666666666",
            "secret": "private-generation-token-for-test-only",
            "expiresAt": (datetime.now(UTC) + timedelta(minutes=10)).isoformat(),
        }
        result = await active.commit(SCOPE, COMMAND, generation)
        assert result["state"] == "RUNNING" and result["sourceRevision"] == 1
        assert "private-password" not in repr(
            result
        ) and "private-generation-token-for-test-only" not in repr(result)
        assert b"private-password" not in path.read_bytes()
        assert b"private-generation-token-for-test-only" not in path.read_bytes()
        again = SourceActivator(SourceRegistry(path, key), driver)
        assert await again.prepare(SCOPE, COMMAND, RECEIPT, 0) == result
        assert len(driver.calls) == 2
        with pytest.raises(SourceActivationError):
            await again.prepare(SCOPE, COMMAND, "different-receipt", 0)
        with pytest.raises(SourceActivationError):
            await again.prepare(SCOPE, "77777777-7777-4777-8777-777777777777", RECEIPT, 0)
        with pytest.raises(SourceActivationError):
            SourceRegistry(path, SecretStr(Fernet.generate_key().decode())).read(SCOPE)

    asyncio.run(run())


def test_expired_foreign_or_stale_receipt_never_stops_a_camera(tmp_path):
    async def run():
        now = [datetime.now(UTC)]
        driver = Driver()
        active = SourceActivator(
            SourceRegistry(tmp_path / "state.enc", SecretStr(Fernet.generate_key().decode())),
            driver,
            clock=lambda: now[0],
        )
        active.remember(SCOPE, RECEIPT, "0", "LIVE")
        for scope, receipt, revision in [
            (("foreign", "camera-a"), RECEIPT, 0),
            (SCOPE, "unknown", 0),
            (SCOPE, RECEIPT, 1),
        ]:
            with pytest.raises(SourceActivationError):
                await active.prepare(scope, COMMAND, receipt, revision)
        now[0] += timedelta(seconds=61)
        with pytest.raises(SourceActivationError):
            await active.prepare(SCOPE, COMMAND, RECEIPT, 0)
        assert driver.calls == []

    asyncio.run(run())


def test_activation_api_accepts_only_server_receipt_then_fresh_generation(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from smartsite_ai.source_control_api import source_control_router
    from smartsite_ai.source_control_models import SourcePolicy

    driver = Driver()
    active = SourceActivator(
        SourceRegistry(tmp_path / "state.enc", SecretStr(Fernet.generate_key().decode())), driver
    )

    async def probe(source, live, scope):
        return {
            "width": 640,
            "height": 480,
            "capturedAt": datetime.now(UTC).isoformat(),
            "imageDataUrl": "data:image/jpeg;base64,/9j/2Q==",
        }

    app = FastAPI()
    app.include_router(
        source_control_router(
            host_id="edge",
            label="Host",
            scopes={SCOPE},
            policy=SourcePolicy({"usb0": 0}, {}, ()),
            service_token=SecretStr("a" * 48),
            probe=probe,
            activator=active,
        )
    )
    client = TestClient(app)
    url = f"/v1/source-control/{SCOPE[0]}/{SCOPE[1]}"
    headers = {"Authorization": "Bearer " + "a" * 48}
    assert client.get(url + "/capabilities", headers=headers).json()["applySupported"] is True
    receipt = client.post(
        url + "/probe", headers=headers, json={"kind": "WEBCAM", "deviceId": "usb0"}
    ).json()["receiptId"]
    command = {"commandId": COMMAND, "receiptId": receipt, "expectedRevision": 0}
    assert client.post(url + "/prepare", json=command).status_code == 401
    assert (
        client.post(
            url + "/prepare", headers=headers, json={**command, "source": "arbitrary"}
        ).status_code
        == 400
    )
    result = client.post(url + "/prepare", headers=headers, json=command)
    assert result.status_code == 200 and result.json()["state"] == "PREPARED"
    assert "source" not in result.json()
    generation = {
        "workerGenerationId": "55555555-5555-4555-8555-555555555555",
        "streamSessionId": "66666666-6666-4666-8666-666666666666",
        "secret": "b" * 48,
        "expiresAt": (datetime.now(UTC) + timedelta(minutes=10)).isoformat(),
    }
    result = client.post(
        url + "/commit", headers=headers, json={"commandId": COMMAND, "generation": generation}
    )
    assert result.status_code == 200 and result.json()["state"] == "RUNNING"
    assert "secret" not in result.text
