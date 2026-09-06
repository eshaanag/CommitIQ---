import asyncio
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import text

from backend.config import MAX_CONCURRENT_INGESTIONS
from backend.database import _IS_SQLITE, commit_with_retry, engine
from backend.features.repo_ingestion.router import _update_job, get_ingestion_semaphore
from backend.shared.models import AnalysisJob, Repo


@pytest.mark.anyio
async def test_commit_with_retry_succeeds_on_first_attempt():
    session = AsyncMock()
    session.commit = AsyncMock()

    await commit_with_retry(session, max_retries=3)

    assert session.commit.call_count == 1


@pytest.mark.anyio
async def test_commit_with_retry_retries_up_to_3_times_on_lock():
    session = AsyncMock()
    call_count = 0

    async def mock_commit():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise RuntimeError("(sqlite3.OperationalError) database is locked")
        return None

    session.commit = mock_commit

    await commit_with_retry(session, max_retries=3, initial_delay=0.01)

    assert call_count == 3


@pytest.mark.anyio
async def test_commit_with_retry_fails_after_3_retries():
    session = AsyncMock()
    session.commit = AsyncMock(
        side_effect=RuntimeError("(sqlite3.OperationalError) database is locked")
    )

    with pytest.raises(RuntimeError) as exc_info:
        await commit_with_retry(session, max_retries=3, initial_delay=0.01)

    assert "database is locked" in str(exc_info.value)
    assert session.commit.call_count == 3


@pytest.mark.anyio
async def test_commit_with_retry_raises_non_lock_error_immediately():
    session = AsyncMock()
    session.commit = AsyncMock(side_effect=ValueError("connection dropped"))

    with pytest.raises(ValueError) as exc_info:
        await commit_with_retry(session, max_retries=3, initial_delay=0.01)

    assert "connection dropped" in str(exc_info.value)
    assert session.commit.call_count == 1


@pytest.mark.anyio
async def test_sqlite_pragmas_active():
    if not _IS_SQLITE:
        pytest.skip("Test requires SQLite database")

    async with engine.connect() as conn:
        res = await conn.execute(text("PRAGMA journal_mode"))
        mode = res.scalar()
        assert mode.lower() == "wal"

        res_timeout = await conn.execute(text("PRAGMA busy_timeout"))
        timeout_ms = res_timeout.scalar()
        assert timeout_ms >= 10000


@pytest.mark.anyio
async def test_get_ingestion_semaphore():
    sem = get_ingestion_semaphore()
    assert isinstance(sem, asyncio.Semaphore)
    assert sem._value == MAX_CONCURRENT_INGESTIONS


@pytest.mark.anyio
async def test_update_job_with_provided_session():
    mock_db = AsyncMock()
    mock_job = AnalysisJob(id=1, status="queued")
    mock_db.get = AsyncMock(return_value=mock_job)
    mock_db.commit = AsyncMock()

    await _update_job(1, db=mock_db, status="analyzing", current_stage="Analyzing...")

    assert mock_job.status == "analyzing"
    assert mock_job.current_stage == "Analyzing..."
    assert mock_db.commit.call_count == 1


@pytest.mark.anyio
async def test_concurrent_sqlite_writes_with_retry(tmp_path):
    """Simulate multiple concurrent background tasks writing to the database simultaneously."""
    import uuid

    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from backend.database import Base

    db_path = tmp_path / "test_concurrency.db"
    test_engine = create_async_engine(
        f"sqlite+aiosqlite:///{db_path}",
        connect_args={"check_same_thread": False, "timeout": 60},
    )
    test_session_maker = async_sessionmaker(bind=test_engine, expire_on_commit=False)

    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    test_run_id = uuid.uuid4().hex[:6]

    async def write_worker(worker_id: int):
        async with test_session_maker() as session:
            repo = Repo(
                name=f"test/concurrent-repo-{test_run_id}-{worker_id}",
                owner="test",
                url=f"https://github.com/test/concurrent-repo-{test_run_id}-{worker_id}",
                repo_slug=f"test-concurrent-repo-{test_run_id}-{worker_id}",
                status="pending",
            )
            session.add(repo)
            await commit_with_retry(session, max_retries=5, initial_delay=0.05)
            await session.refresh(repo)

            job = AnalysisJob(
                repo_id=repo.id,
                status="queued",
                triggered_by=f"worker-{worker_id}",
            )
            session.add(job)
            await commit_with_retry(session, max_retries=5, initial_delay=0.05)

    # Launch 6 concurrent workers simultaneously
    tasks = [write_worker(i) for i in range(6)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    await test_engine.dispose()

    for res in results:
        assert not isinstance(res, Exception), f"Concurrent worker failed: {res}"
