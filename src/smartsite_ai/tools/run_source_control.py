"""Explicit source-control host. Running this module never starts an inference model."""

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from smartsite_ai.source_control_api import source_control_router
from smartsite_ai.source_control_models import SourcePolicy
from smartsite_ai.source_control_probe import capture_source


class ActiveSource(BaseModel):
    model_config = ConfigDict(extra="forbid")
    site_id: str
    camera_external_id: str
    source: SecretStr
    preview_origin: str


class HostManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    host_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,128}$")
    label: str = Field(min_length=1, max_length=128)
    scopes: list[tuple[str, str]] = Field(min_length=1, max_length=128)
    webcams: dict[str, int] = Field(default_factory=dict, max_length=32)
    videos: dict[str, Path] = Field(default_factory=dict, max_length=128)
    networks: tuple[str, ...] = ()
    active_sources: list[ActiveSource] = Field(default_factory=list, max_length=16)


def build_source_control_app(manifest: HostManifest, token: SecretStr) -> FastAPI:
    for active in manifest.active_sources:
        url = urlsplit(active.preview_origin)
        if (
            url.username
            or url.password
            or url.path not in {"", "/"}
            or url.query
            or url.fragment
            or not (
                url.scheme == "https"
                or (url.scheme == "http" and url.hostname in {"127.0.0.1", "::1"})
            )
        ):
            raise ValueError("Invalid configured preview origin")
        if (active.site_id, active.camera_external_id) not in manifest.scopes and (
            active.site_id,
            "*",
        ) not in manifest.scopes:
            raise ValueError("Active preview must belong to an assigned source scope")
    policy = SourcePolicy(manifest.webcams, manifest.videos, manifest.networks)

    async def probe(source: str, live: bool, scope: tuple[str, str]) -> dict[str, object]:
        matching = [a for a in manifest.active_sources if a.source.get_secret_value() == source]
        if not matching:
            return await capture_source(source, live)
        active = next((a for a in matching if (a.site_id, a.camera_external_id) == scope), None)
        if active is None:
            # A physical device already belongs to another capture owner.
            raise RuntimeError("SOURCE_BUSY")
        async with (
            httpx.AsyncClient(timeout=3, follow_redirects=False) as client,
            client.stream(
                "GET",
                active.preview_origin.rstrip("/") + "/v1/realtime/configuration-image",
                params={"cameraExternalId": active.camera_external_id},
                headers={"Authorization": "Bearer " + token.get_secret_value()},
            ) as response,
        ):
            if (
                response.status_code != 200
                or response.headers.get("content-type", "").split(";")[0] != "application/json"
            ):
                raise RuntimeError("SOURCE_UNAVAILABLE")
            body = bytearray()
            async for chunk in response.aiter_bytes():
                body.extend(chunk)
                if len(body) > 400_000:
                    raise RuntimeError("SOURCE_UNAVAILABLE")
        payload = json.loads(body)
        frame = payload["frame"]
        captured_at = datetime.fromisoformat(frame["capturedAt"])
        age = (datetime.now(UTC) - captured_at).total_seconds()
        if (
            frame.get("cameraExternalId") != scope[1]
            or not 0 <= age <= 5
            or payload.get("sourceKind") != ("LIVE" if live else "REPLAY")
        ):
            raise RuntimeError("SOURCE_UNAVAILABLE")
        return {key: frame[key] for key in ("width", "height", "capturedAt", "imageDataUrl")}

    app = FastAPI(openapi_url=None, docs_url=None, redoc_url=None)
    app.include_router(
        source_control_router(
            host_id=manifest.host_id,
            label=manifest.label,
            scopes=set(manifest.scopes),
            policy=policy,
            service_token=token,
            probe=probe,
        )
    )
    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--port", type=int, default=62035)
    args = parser.parse_args()
    # Manifest paths and network policy are operator-owned, never supplied by the browser.
    manifest = HostManifest.model_validate_json(args.manifest.read_bytes())
    token = SecretStr(os.environ["SMARTSITE_AI_SOURCE_CONTROL_TOKEN"])
    app = build_source_control_app(manifest, token)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="error", access_log=False)


if __name__ == "__main__":
    main()
