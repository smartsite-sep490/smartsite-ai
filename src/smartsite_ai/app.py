"""HTTP contract for the API process, independent of future inference workers."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, Response
from pydantic import BaseModel

from smartsite_ai.config import Settings


class LiveHealth(BaseModel):
    status: Literal["ok"] = "ok"
    service: Literal["smartsite-ai"] = "smartsite-ai"


class ReadyHealth(BaseModel):
    status: Literal["ready", "not_ready"]
    service: Literal["smartsite-ai"] = "smartsite-ai"
    scope: Literal["api"] = "api"
    inference_ready: Literal[False] = False


class Capability(BaseModel):
    status: Literal["not_configured"] = "not_configured"
    provider: str | None = None
    reason: str


class Capabilities(BaseModel):
    service: Literal["smartsite-ai"] = "smartsite-ai"
    api_version: Literal["v1"] = "v1"
    inference_ready: Literal[False] = False
    capabilities: dict[str, Capability]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.api_ready = True
    try:
        yield
    finally:
        app.state.api_ready = False


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings if settings is not None else Settings()
    expose_docs = settings.environment != "production"
    app = FastAPI(
        title="SmartSite AI",
        version="0.1.0",
        description="API foundation. Camera ingestion and inference are not configured.",
        lifespan=lifespan,
        docs_url="/docs" if expose_docs else None,
        redoc_url=None,
        openapi_url="/openapi.json" if expose_docs else None,
    )
    app.state.api_ready = False

    @app.get("/health/live", response_model=LiveHealth, tags=["health"])
    async def live() -> LiveHealth:
        return LiveHealth()

    @app.get(
        "/health/ready",
        response_model=ReadyHealth,
        responses={503: {"model": ReadyHealth, "description": "API lifecycle is not ready"}},
        tags=["health"],
        summary="API process readiness, independent of inference readiness",
    )
    async def ready(response: Response) -> ReadyHealth:
        if not app.state.api_ready:
            response.status_code = 503
            return ReadyHealth(status="not_ready")
        return ReadyHealth(status="ready")

    @app.get("/v1/capabilities", response_model=Capabilities, tags=["capabilities"])
    async def capabilities() -> Capabilities:
        return Capabilities(
            capabilities={
                "camera": Capability(reason="No camera ingestion worker or stream is configured."),
                "detector": Capability(
                    provider="ultralytics-yolo11s + supervision",
                    reason="Pipeline direction only; no model weights, PPE model or worker loaded.",
                ),
                "zone": Capability(
                    provider="supervision",
                    reason="No zones, tracking worker or backend policy contract configured.",
                ),
                "identity": Capability(
                    provider="insightface (candidate)",
                    reason="Candidate only; no identity adapter, model or enrollment store.",
                ),
                "openai": Capability(
                    provider="openai",
                    reason="Supplementary evidence analysis planned; no API adapter configured.",
                ),
            }
        )

    return app
