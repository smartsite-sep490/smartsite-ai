"""Model-free MF06 shutdown checks; a live task is not a graceful release."""

import asyncio

import pytest
from fastapi import FastAPI

from smartsite_ai import mf06_identity_api


def test_shutdown_reports_unfinished_resolver_instead_of_claiming_graceful_release(monkeypatch):
    monkeypatch.setattr(mf06_identity_api, "VERIFICATION_TIMEOUT_SECONDS", 0.01)

    async def scenario():
        app = FastAPI()
        active = None
        try:
            with pytest.raises(RuntimeError, match="MF06_FACE_DRAIN_TIMEOUT"):
                async with mf06_identity_api.router.lifespan_context(app):
                    active = asyncio.create_task(asyncio.Event().wait())
                    app.state.mf06_face_active = active
            assert app.state.mf06_face_draining is True
            assert not active.done()
        finally:
            if active is not None:
                active.cancel()
                await asyncio.gather(active, return_exceptions=True)

    asyncio.run(scenario())


def test_shutdown_waits_for_actual_resolver_completion():
    async def scenario():
        app = FastAPI()
        async with mf06_identity_api.router.lifespan_context(app):
            active = asyncio.create_task(asyncio.sleep(0))
            app.state.mf06_face_active = active
        assert active.done()
        assert app.state.mf06_face_draining is True

    asyncio.run(scenario())


def test_shutdown_observes_late_resolver_failure_without_calling_it_a_drain_timeout():
    async def failing():
        await asyncio.sleep(0)
        raise ValueError("Synthetic resolver failure")

    async def scenario():
        app = FastAPI()
        async with mf06_identity_api.router.lifespan_context(app):
            active = asyncio.create_task(failing())
            app.state.mf06_face_active = active
        assert isinstance(active.exception(), ValueError)

    asyncio.run(scenario())
