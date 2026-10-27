import asyncio
import sqlite3
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from smartsite_ai.integrations.outbox import SqliteEventOutbox


@pytest.mark.anyio
async def test_generation_and_session_cannot_be_claimed_again_after_restart(tmp_path: Path):
    path = tmp_path / "outbox.sqlite3"
    camera, generation, session = uuid4(), uuid4(), uuid4()
    await SqliteEventOutbox(path).claim_worker_session(camera, generation, session)
    for ids in [
        (camera, generation, session),
        (camera, generation, uuid4()),
        (uuid4(), uuid4(), session),
    ]:
        with pytest.raises(RuntimeError, match="already consumed"):
            await SqliteEventOutbox(path).claim_worker_session(*ids)
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT count(*) FROM worker_session_claim").fetchone()[0] == 1
    await SqliteEventOutbox(path).claim_worker_session(camera, uuid4(), uuid4())


@pytest.mark.anyio
async def test_concurrent_instances_have_one_winner(tmp_path: Path):
    path = tmp_path / "outbox.sqlite3"
    first, second = SqliteEventOutbox(path), SqliteEventOutbox(path)
    await first.initialize()
    await second.initialize()
    ids = uuid4(), uuid4(), uuid4()
    results = await asyncio.gather(
        first.claim_worker_session(*ids), second.claim_worker_session(*ids), return_exceptions=True
    )
    assert sum(value is None for value in results) == 1
    assert sum(isinstance(value, RuntimeError) for value in results) == 1


@pytest.mark.anyio
async def test_claim_rejects_untyped_ids_before_storage(tmp_path: Path):
    path = tmp_path / "outbox.sqlite3"
    with pytest.raises(ValueError):
        await SqliteEventOutbox(path).claim_worker_session("bad", UUID(int=1), uuid4())
    assert not path.exists()
