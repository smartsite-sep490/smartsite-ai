"""Local and container entry point."""

import uvicorn

from smartsite_ai.app import create_app
from smartsite_ai.config import Settings


def run() -> None:
    settings = Settings()
    uvicorn.run(
        create_app(settings),
        host=str(settings.host),
        port=settings.port,
        log_level=settings.log_level,
    )


if __name__ == "__main__":
    run()
