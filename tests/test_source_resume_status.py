"""Public resume retries use durable original bindings, never reconstruct new requests."""

import asyncio
import json
from uuid import uuid4

import pytest
from test_source_resume import OLD, SCOPE, request, setup

from smartsite_ai.source_activation import SourceActivationError


def test_status_recovers_original_request_after_response_loss_and_blocks_delayed_command(tmp_path):
    async def run():
        active, driver = setup(tmp_path)
        first = request()
        with pytest.raises(SourceActivationError, match="NOT_FOUND"):
            await active.resume_status(SCOPE, first.commandId)
        await active.prepare_resume(SCOPE, first)
        assert (await active.resume_status(SCOPE, first.commandId))["state"] == "PREPARED"
        next_generation = {
            **OLD,
            "workerGenerationId": str(uuid4()),
            "streamSessionId": str(uuid4()),
        }
        await active.commit_resume(SCOPE, first.commandId, next_generation)
        # Read a registry written by the preceding single-command implementation.
        legacy = active.registry.read(SCOPE)
        legacy.pop("resumeHistory")
        active.registry.write(SCOPE, legacy)
        # A new coordinator instance reads original metadata after the response was lost.
        from smartsite_ai.source_activation import SourceActivator

        restarted = SourceActivator(active.registry, driver)
        status = await restarted.resume_status(SCOPE, first.commandId)
        assert status["generationBinding"] == first.generationBinding.model_dump()
        assert status["configuration"] == first.configuration.model_dump()
        assert (
            status["newGenerationBinding"]["workerGenerationId"]
            == next_generation["workerGenerationId"]
        )
        assert set(status) == {
            "commandId",
            "sourceRevision",
            "sourceKind",
            "state",
            "generationBinding",
            "configuration",
            "newGenerationBinding",
        }
        assert OLD["secret"] not in json.dumps(status)
        assert len(driver.calls) == 2
        second = first.model_copy(
            update={
                "commandId": str(uuid4()),
                "generationBinding": first.generationBinding.model_copy(
                    update={
                        "workerGenerationId": next_generation["workerGenerationId"],
                        "streamSessionId": next_generation["streamSessionId"],
                    }
                ),
            }
        )
        await restarted.prepare_resume(SCOPE, second)
        third_generation = {
            **OLD,
            "workerGenerationId": str(uuid4()),
            "streamSessionId": str(uuid4()),
        }
        await restarted.commit_resume(SCOPE, second.commandId, third_generation)
        assert len(active.registry.read(SCOPE)["resumeHistory"]) == 2
        before = len(driver.calls)
        for original_or_rebound in [
            first,
            first.model_copy(update={"generationBinding": second.generationBinding}),
        ]:
            with pytest.raises(SourceActivationError):
                await restarted.prepare_resume(SCOPE, original_or_rebound)
        with pytest.raises(SourceActivationError, match="SUPERSEDED"):
            await restarted.resume_status(SCOPE, first.commandId)
        assert len(driver.calls) == before

    asyncio.run(run())


def test_status_of_uncertain_start_is_readonly_and_has_no_new_binding(tmp_path):
    async def run():
        active, driver = setup(tmp_path)
        req = request()
        await active.prepare_resume(SCOPE, req)

        async def fail(*args):
            raise RuntimeError("Synthetic startup failure")

        driver.start_resume = fail
        new = {**OLD, "workerGenerationId": str(uuid4()), "streamSessionId": str(uuid4())}
        with pytest.raises(RuntimeError):
            await active.commit_resume(SCOPE, req.commandId, new)
        before = active.registry.path.read_bytes()
        status = await active.resume_status(SCOPE, req.commandId)
        assert status["state"] == "STARTING" and "newGenerationBinding" not in status
        assert active.registry.path.read_bytes() == before

    asyncio.run(run())


def test_history_quota_fails_before_stopping_and_keeps_prior_record(tmp_path):
    async def run():
        active, driver = setup(tmp_path)
        row = active.registry.read(SCOPE)
        row["padding"] = "p" * 748500
        active.registry.write(SCOPE, row)
        before = active.registry.path.read_bytes()
        assert 998000 < len(before) < 1000000
        with pytest.raises(SourceActivationError, match="STORAGE_UNAVAILABLE"):
            await active.prepare_resume(SCOPE, request())
        assert driver.calls == []
        assert active.registry.path.read_bytes() == before

    asyncio.run(run())


def test_status_api_auth_scope_strict_readonly_and_legacy_capability_compatibility(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from pydantic import SecretStr
    from test_source_activation import Driver as LegacyDriver

    from smartsite_ai.source_control_api import source_control_router
    from smartsite_ai.source_control_models import SourcePolicy

    active, driver = setup(tmp_path)

    async def probe(*args):
        raise AssertionError("No probe during status")

    app = FastAPI()
    app.include_router(
        source_control_router(
            host_id="edge",
            label="Host",
            scopes={SCOPE},
            policy=SourcePolicy({}, {}, ()),
            service_token=SecretStr("a" * 48),
            probe=probe,
            activator=active,
        )
    )
    client = TestClient(app)
    url = f"/v1/source-control/{SCOPE[0]}/{SCOPE[1]}"
    headers = {"Authorization": "Bearer " + "a" * 48}
    req = request()
    payload = {"commandId": req.commandId}
    for malformed in [1, [], {}, None]:
        assert (
            client.post(
                url + "/resume-status", headers=headers, json={"commandId": malformed}
            ).status_code
            == 400
        )
        assert (
            client.post(
                url + "/commit-resume",
                headers=headers,
                json={"commandId": malformed, "generation": OLD},
            ).status_code
            == 400
        )
    assert client.post(url + "/resume-status", json=payload).status_code == 401
    assert (
        client.post(
            url.replace("camera-a", "foreign") + "/resume-status", headers=headers, json=payload
        ).status_code
        == 404
    )
    assert (
        client.post(
            url + "/resume-status", headers=headers, json={**payload, "source": "evil"}
        ).status_code
        == 400
    )
    assert client.post(url + "/resume-status", headers=headers, json=payload).status_code == 404
    assert (
        client.post(url + "/prepare-resume", headers=headers, json=req.model_dump()).status_code
        == 200
    )
    assert (
        client.post(url + "/resume-status", headers=headers, json=payload).json()["state"]
        == "PREPARED"
    )
    new = {**OLD, "workerGenerationId": str(uuid4()), "streamSessionId": str(uuid4())}
    assert (
        client.post(
            url + "/commit-resume", headers=headers, json={**payload, "generation": new}
        ).status_code
        == 200
    )
    response = client.post(url + "/resume-status", headers=headers, json=payload)
    assert response.status_code == 200 and response.json()["state"] == "RUNNING"
    assert response.json()["generationBinding"] == req.generationBinding.model_dump()
    assert response.json()["newGenerationBinding"]["streamSessionId"] == new["streamSessionId"]
    assert set(response.json()) == {
        "schemaVersion",
        "commandId",
        "sourceRevision",
        "sourceKind",
        "state",
        "generationBinding",
        "configuration",
        "newGenerationBinding",
    }
    assert OLD["secret"] not in response.text
    active.driver = LegacyDriver()
    assert client.post(url + "/resume-status", headers=headers, json=payload).status_code == 409
    assert (
        "configurationResumeSupported"
        not in client.get(url + "/capabilities", headers=headers).json()
    )


def test_source_replacement_retains_history_until_revision_advances(tmp_path):
    async def run():
        active, driver = setup(tmp_path)
        req = request()
        await active.prepare_resume(SCOPE, req)
        new = {**OLD, "workerGenerationId": str(uuid4()), "streamSessionId": str(uuid4())}
        await active.commit_resume(SCOPE, req.commandId, new)

        async def generation_id(scope):
            return new["workerGenerationId"]

        driver.generation_id = generation_id
        receipt = str(uuid4())
        active.remember(SCOPE, receipt, "1", "LIVE")
        replace = str(uuid4())
        await active.prepare(SCOPE, replace, receipt, 3)
        assert req.commandId in active.registry.read(SCOPE)["resumeHistory"]
        with pytest.raises(SourceActivationError, match="RECONCILIATION"):
            await active.resume_status(SCOPE, req.commandId)

        async def start(scope, source, generation):
            driver.running = True

        driver.start = start
        replacement = {**OLD, "workerGenerationId": str(uuid4()), "streamSessionId": str(uuid4())}
        assert (await active.commit(SCOPE, replace, replacement))["sourceRevision"] == 4
        assert "resumeHistory" not in active.registry.read(SCOPE)
        with pytest.raises(SourceActivationError, match="NOT_FOUND"):
            await active.resume_status(SCOPE, req.commandId)
        before = len(driver.calls)
        with pytest.raises(SourceActivationError):
            await active.prepare_resume(SCOPE, req)
        assert len(driver.calls) == before

    asyncio.run(run())
