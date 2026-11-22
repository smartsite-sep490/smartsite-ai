from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from cryptography.fernet import Fernet
from fastapi import FastAPI
from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError
from pydantic import SecretStr

from smartsite_ai.source_activation import (
    Scope,
    SourceActivationError,
    SourceActivator,
    SourceRegistry,
)
from smartsite_ai.source_control_api import source_control_router
from smartsite_ai.source_control_models import SourcePolicy

CONTRACTS_DIR = Path(__file__).resolve().parents[1] / "contracts"
METADATA_PATH = CONTRACTS_DIR / "metadata.json"
SCHEMA_PATH = CONTRACTS_DIR / "schemas/v1.1/source-activation-state.json"
EXAMPLE_PATH = CONTRACTS_DIR / "examples/v1.1/source-unowned-prepared.json"

EXPECTED_COMMIT_SHA = "6bcecad9d15729cfa61f08603b08e33135ea6634"
EXPECTED_SCHEMA_SHA256 = "eec9839184c594879dde077ca8e799d6cf5322908144efb96b1ebf1129285dc1"
EXPECTED_EXAMPLE_SHA256 = "68bab3bf0f3a211472fc13f70242ac34b75f7ebaf1497919d52cbe9ff71738bf"

SCOPE: Scope = ("11111111-1111-4111-8111-111111111111", "camera-a")
CMD1 = "22222222-2222-4222-8222-222222222222"
CMD2 = "33333333-3333-4333-8333-333333333333"
GEN1 = "66666666-6666-4666-8666-666666666666"
GEN2 = "77777777-7777-4777-8777-777777777777"
SESS1 = "88888888-8888-4888-8888-888888888888"
SESS2 = "99999999-9999-4999-8999-999999999999"


def _load_schema() -> dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _load_example() -> dict[str, Any]:
    return json.loads(EXAMPLE_PATH.read_text(encoding="utf-8"))


def _validator() -> Draft202012Validator:
    schema = _load_schema()
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _owned_baseline(*, state: str = "RUNNING", source_revision: int = 1) -> dict[str, Any]:
    return {
        "schemaVersion": "1.0.0",
        "state": state,
        "commandId": CMD1,
        "sourceRevision": source_revision,
        "sourceKind": "LIVE",
        "currentGenerationId": GEN1,
    }


def _generation(worker_generation_id: str, stream_session_id: str) -> dict[str, str]:
    return {
        "workerGenerationId": worker_generation_id,
        "streamSessionId": stream_session_id,
        "secret": "s" * 48,
        "expiresAt": (datetime.now(UTC) + timedelta(minutes=10)).isoformat(),
    }


class _UnownedFakeDriver:
    def __init__(self, *, initial_owner: str | None = None) -> None:
        self.owner = initial_owner
        self.running = initial_owner is not None

    def supports(self, scope: Scope) -> bool:
        return scope == SCOPE

    async def generation_id(self, scope: Scope) -> str | None:
        if not self.supports(scope):
            raise SourceActivationError()
        return self.owner

    async def stop(self, scope: Scope) -> None:
        self.running = False

    async def start(self, scope: Scope, source: str, generation: Mapping[str, str]) -> None:
        assert not self.running
        self.owner = generation["workerGenerationId"]
        self.running = True

    async def is_running(self, scope: Scope, generation_id: str) -> bool:
        return self.running and self.owner == generation_id


def _build_client(
    store_path: Path, *, initial_owner: str | None = None
) -> tuple[TestClient, dict[str, str], str]:
    key = SecretStr(Fernet.generate_key().decode())
    store = SourceRegistry(store_path, key)
    driver = _UnownedFakeDriver(initial_owner=initial_owner)
    activator = SourceActivator(store, driver)

    async def probe(source: str, live: bool, scope: Scope) -> dict[str, object]:
        return {
            "width": 640,
            "height": 480,
            "capturedAt": datetime.now(UTC).isoformat(),
            "imageDataUrl": "data:image/jpeg;base64,/9j/2Q==",
        }

    token = "a" * 48
    app = FastAPI()
    app.include_router(
        source_control_router(
            host_id="edge",
            label="Host",
            scopes={SCOPE},
            policy=SourcePolicy({"usb0": 0}, {}, ()),
            service_token=SecretStr(token),
            probe=probe,
            activator=activator,
        )
    )
    return (
        TestClient(app),
        {"Authorization": f"Bearer {token}"},
        f"/v1/source-control/{SCOPE[0]}/{SCOPE[1]}",
    )


def test_source_activation_metadata_provenance_and_lf_bytes() -> None:
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    entry = metadata["cameraSourceActivationState"]

    assert entry["schemaVersion"] == "1.1.0"
    assert entry["sourceRepository"] == "smartsite-sep490/smartsite"
    assert entry["sourceCommitSha"] == EXPECTED_COMMIT_SHA
    assert re.fullmatch(r"[0-9a-f]{40}", entry["sourceCommitSha"])
    assert entry["sourceCommitSha"] not in {
        metadata["sourceCommitSha"],
        metadata["cameraPolicyConfigurationBundle"]["sourceCommitSha"],
        metadata["trackedTechnicalObservationVersions"][-1]["sourceCommitSha"],
        metadata["aiFrameEvidenceReceipt"]["sourceCommitSha"],
    }
    assert entry["path"] == "contracts/schemas/v1.1/source-activation-state.json"
    assert entry["sourcePath"] == entry["path"]
    assert entry["examplePath"] == "contracts/examples/v1.1/source-unowned-prepared.json"
    assert entry["sourceExamplePath"] == entry["examplePath"]

    schema_bytes = SCHEMA_PATH.read_bytes()
    example_bytes = EXAMPLE_PATH.read_bytes()
    assert b"\r\n" not in schema_bytes
    assert b"\r\n" not in example_bytes

    assert re.fullmatch(r"[0-9a-f]{64}", entry["schemaSha256"])
    assert re.fullmatch(r"[0-9a-f]{64}", entry["exampleSha256"])
    assert entry["schemaSha256"] == EXPECTED_SCHEMA_SHA256
    assert entry["exampleSha256"] == EXPECTED_EXAMPLE_SHA256
    assert hashlib.sha256(schema_bytes).hexdigest() == entry["schemaSha256"]
    assert hashlib.sha256(example_bytes).hexdigest() == entry["exampleSha256"]


def test_source_activation_schema_accepts_vendored_example_and_owned_vectors() -> None:
    validator = _validator()

    example = _load_example()
    validator.validate(example)
    assert example["schemaVersion"] == "1.1.0"
    assert example["state"] == "PREPARED"
    assert example["sourceRevision"] == 0
    assert example["sourceKind"] == "LIVE"
    assert example["currentGenerationId"] is None

    positive_vectors = [
        _owned_baseline(state="PREPARED", source_revision=0),
        _owned_baseline(state="PREPARED", source_revision=1),
        _owned_baseline(state="RUNNING", source_revision=1),
        {
            "schemaVersion": "1.0.0",
            "state": "RUNNING",
            "commandId": CMD1.upper(),
            "sourceRevision": 9007199254740991,
            "sourceKind": "LIVE",
            "currentGenerationId": GEN2.upper(),
        },
    ]
    for vector in positive_vectors:
        validator.validate(vector)


@pytest.mark.parametrize(
    ("use_example", "mutator"),
    [
        (True, lambda d: d.__setitem__("state", "RUNNING")),
        (True, lambda d: d.__setitem__("sourceRevision", 1)),
        (True, lambda d: d.__setitem__("sourceRevision", -1)),
        (True, lambda d: d.__setitem__("currentGenerationId", GEN1)),
        (True, lambda d: d.pop("currentGenerationId")),
        (True, lambda d: d.pop("schemaVersion")),
        (True, lambda d: d.__setitem__("sourceKind", "REPLAY")),
        (True, lambda d: d.__setitem__("permission", "admin")),
        (False, lambda d: d.__setitem__("currentGenerationId", None)),
        (False, lambda d: d.__setitem__("sourceRevision", -1)),
        (False, lambda d: d.__setitem__("sourceRevision", 9007199254740992)),
        (False, lambda d: d.__setitem__("state", "IDLE")),
        (False, lambda d: d.__setitem__("sourceKind", "WEBCAM")),
        (False, lambda d: d.__setitem__("secret", "leak")),
        (False, lambda d: d.__setitem__("commandId", "11111111-1111-4111-8111-11111111111G")),
        (False, lambda d: d.__setitem__("currentGenerationId", "not-a-uuid")),
        (False, lambda d: d.__setitem__("schemaVersion", "1.2.0")),
    ],
)
def test_source_activation_schema_rejects_boundary_and_tampered_vectors(
    use_example: bool,
    mutator: Callable[[dict[str, Any]], Any],
) -> None:
    validator = _validator()
    candidate = copy.deepcopy(_load_example() if use_example else _owned_baseline())
    mutator(candidate)
    with pytest.raises(JsonSchemaValidationError):
        validator.validate(candidate)


def test_source_control_api_outputs_validate_against_vendored_schema(tmp_path: Path) -> None:
    validator = _validator()

    unowned_client, headers, base = _build_client(tmp_path / "unowned.enc", initial_owner=None)
    receipt1 = unowned_client.post(
        f"{base}/probe", headers=headers, json={"kind": "WEBCAM", "deviceId": "usb0"}
    ).json()["receiptId"]

    prep1_body = {"commandId": CMD1, "receiptId": receipt1, "expectedRevision": 0}
    prep1 = unowned_client.post(f"{base}/prepare", headers=headers, json=prep1_body)
    assert prep1.status_code == 200
    prep1_json = prep1.json()
    assert prep1_json["schemaVersion"] == "1.1.0"
    assert prep1_json["state"] == "PREPARED"
    assert prep1_json["currentGenerationId"] is None
    validator.validate(prep1_json)

    prep1_retry = unowned_client.post(f"{base}/prepare", headers=headers, json=prep1_body)
    assert prep1_retry.status_code == 200
    assert prep1_retry.json() == prep1_json
    validator.validate(prep1_retry.json())

    commit1 = unowned_client.post(
        f"{base}/commit",
        headers=headers,
        json={"commandId": CMD1, "generation": _generation(GEN1, SESS1)},
    )
    assert commit1.status_code == 200
    commit1_json = commit1.json()
    assert commit1_json["schemaVersion"] == "1.0.0"
    assert commit1_json["state"] == "RUNNING"
    assert commit1_json["sourceRevision"] == 1
    assert commit1_json["currentGenerationId"] == GEN1
    validator.validate(commit1_json)

    prep1_after_commit = unowned_client.post(f"{base}/prepare", headers=headers, json=prep1_body)
    assert prep1_after_commit.status_code == 200
    assert prep1_after_commit.json() == commit1_json
    validator.validate(prep1_after_commit.json())

    receipt2 = unowned_client.post(
        f"{base}/probe", headers=headers, json={"kind": "WEBCAM", "deviceId": "usb0"}
    ).json()["receiptId"]
    prep2 = unowned_client.post(
        f"{base}/prepare",
        headers=headers,
        json={"commandId": CMD2, "receiptId": receipt2, "expectedRevision": 1},
    )
    assert prep2.status_code == 200
    prep2_json = prep2.json()
    assert prep2_json["schemaVersion"] == "1.0.0"
    assert prep2_json["state"] == "PREPARED"
    assert prep2_json["sourceRevision"] == 1
    assert prep2_json["currentGenerationId"] == GEN1
    validator.validate(prep2_json)

    commit2 = unowned_client.post(
        f"{base}/commit",
        headers=headers,
        json={"commandId": CMD2, "generation": _generation(GEN2, SESS2)},
    )
    assert commit2.status_code == 200
    commit2_json = commit2.json()
    assert commit2_json["schemaVersion"] == "1.0.0"
    assert commit2_json["state"] == "RUNNING"
    assert commit2_json["sourceRevision"] == 2
    assert commit2_json["currentGenerationId"] == GEN2
    validator.validate(commit2_json)

    owned_client, owned_headers, owned_base = _build_client(
        tmp_path / "owned.enc", initial_owner=GEN1
    )
    receipt3 = owned_client.post(
        f"{owned_base}/probe", headers=owned_headers, json={"kind": "WEBCAM", "deviceId": "usb0"}
    ).json()["receiptId"]
    owned_prep = owned_client.post(
        f"{owned_base}/prepare",
        headers=owned_headers,
        json={"commandId": CMD1, "receiptId": receipt3, "expectedRevision": 0},
    )
    assert owned_prep.status_code == 200
    owned_prep_json = owned_prep.json()
    assert owned_prep_json["schemaVersion"] == "1.0.0"
    assert owned_prep_json["state"] == "PREPARED"
    assert owned_prep_json["sourceRevision"] == 0
    assert owned_prep_json["currentGenerationId"] == GEN1
    validator.validate(owned_prep_json)

    owned_commit = owned_client.post(
        f"{owned_base}/commit",
        headers=owned_headers,
        json={"commandId": CMD1, "generation": _generation(GEN2, SESS2)},
    )
    assert owned_commit.status_code == 200
    owned_commit_json = owned_commit.json()
    assert owned_commit_json["schemaVersion"] == "1.0.0"
    assert owned_commit_json["state"] == "RUNNING"
    assert owned_commit_json["sourceRevision"] == 1
    assert owned_commit_json["currentGenerationId"] == GEN2
    validator.validate(owned_commit_json)
