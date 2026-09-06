import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import CORS_ORIGINS, ENVIRONMENT
from backend.database import AsyncSessionLocal, engine, init_db
from backend.demo_seeder import seed_demo_data_if_empty
from backend.features.llm_analysis.router import router as llm_router
from backend.features.metrics.digest_router import router as digest_router
from backend.features.metrics.recommendations_router import router as recs_router
from backend.features.metrics.router import router as metrics_router
from backend.features.repo_ingestion.router import router as ingestion_router
from backend.features.reports.router import router as reports_router
from backend.features.reports.schedule_router import router as schedule_router
from backend.features.webhooks.router import router as webhooks_router
from backend.scheduler import start_scheduler, stop_scheduler

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db(env=ENVIRONMENT)
    async with AsyncSessionLocal() as session:
        try:
            await seed_demo_data_if_empty(session)
        except Exception as exc:
            logger.error("Failed to auto-seed demo data: %s", exc, exc_info=True)
    start_scheduler()
    yield
    stop_scheduler()
    await engine.dispose()


app = FastAPI(
    title="CommitIQ API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingestion_router, prefix="/api")
app.include_router(llm_router, prefix="/api")
app.include_router(metrics_router, prefix="/api")
app.include_router(digest_router, prefix="/api")
app.include_router(recs_router, prefix="/api")
app.include_router(reports_router, prefix="/api")
app.include_router(schedule_router, prefix="/api")
app.include_router(webhooks_router, prefix="/api/webhooks", tags=["webhooks"])


@app.get("/health")
async def health():
    from backend.scheduler import get_scheduler_status

    return {
        "status": "ok",
        "service": "commitiq-api",
        "scheduler": get_scheduler_status(),
    }
