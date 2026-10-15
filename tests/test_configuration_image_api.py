"""Observer-only API qualification, using an in-memory ASGI app and synthetic source."""

import asyncio
import base64
import hashlib
import importlib
import importlib.util
import io
from datetime import timedelta

import httpx
import pytest
from fastapi import FastAPI
from PIL import Image
from pydantic import SecretStr

from tests.configuration_capture_fakes import NOW, FakeSource, finish, frame, until
from tests.test_configuration_capture import capture_pair

TOKEN = "synthetic-service-token-for-configuration-tests"
PATH = "/v1/realtime/configuration-image"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


def make_app(capture):
    name = "smartsite_ai.configuration_image_api"
    assert importlib.util.find_spec(name) is not None, "Observer-only route is not implemented"
    module = importlib.import_module(name)
    app = FastAPI()
    app.include_router(module.configuration_image_router(capture, SecretStr(TOKEN)))
    return app


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong"}])
def test_unauthenticated_requests_never_start_source_and_error_is_not_cacheable(headers):
    async def scenario():
        capture, source, _, _ = capture_pair()
        app = make_app(capture)
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client,
        ):
            response = await client.get(
                PATH, params={"cameraExternalId": "camera-a"}, headers=headers
            )
            assert response.status_code == 401
            assert response.headers["Cache-Control"] == "no-store"
            assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert source.connects == source.reads == 0 and capture.state == "IDLE"

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "query, expected",
    [
        ("", 400),
        ("cameraExternalId=camera-a&source=anything", 400),
        ("cameraExternalId=camera-a&cameraExternalId=camera-a", 400),
        ("cameraExternalId=", 400),
        ("cameraExternalId=another-camera", 404),
        ("cameraExternalId=camera-a", 503),
    ],
)
def test_query_scope_and_unavailable_owner_never_create_or_restart_capture(query, expected):
    async def scenario():
        capture, source, _, _ = capture_pair()
        app = make_app(capture)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            for _ in range(2):
                response = await client.get(PATH + "?" + query, headers=AUTH)
                assert response.status_code == expected
                assert response.headers["Cache-Control"] == "no-store"
                assert "frame" not in response.json()
        assert source.connects == source.reads == 0

    asyncio.run(scenario())


def test_authenticated_route_returns_exact_jpeg_then_refuses_stale_or_stopped_owner():
    async def scenario():
        now = [NOW]
        source = FakeSource()
        capture, _, _, _ = capture_pair(source, kind="REPLAY", clock=lambda: now[0])
        app = make_app(capture)
        await capture.start()
        try:
            await source.reading.wait()
            source.items.put_nowait(frame())
            await until(lambda: capture.latest() is not None)
            original = capture.latest()
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.get(PATH + "?cameraExternalId=camera-a", headers=AUTH)
                assert response.status_code == 200 and response.json() == original
                assert response.headers["Cache-Control"] == "no-store"
                raw = base64.b64decode(response.json()["frame"]["imageDataUrl"].split(",", 1)[1])
                assert hashlib.sha256(raw).hexdigest() == response.json()["frame"]["imageSha256"]
                image = Image.open(io.BytesIO(raw))
                image.load()
                assert image.size == (64, 64)
                now[0] += timedelta(seconds=3)
                assert (
                    await client.get(PATH + "?cameraExternalId=camera-a", headers=AUTH)
                ).status_code == 503
                await finish(capture, source)
                assert (
                    await client.get(PATH + "?cameraExternalId=camera-a", headers=AUTH)
                ).status_code == 503
            assert source.connects == source.closes == 1
        finally:
            await finish(capture, source)

    asyncio.run(scenario())
