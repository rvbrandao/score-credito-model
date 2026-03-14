from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.core.model_store import setup_model_store
from api.routers.scoring import router as scoring_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_model_store(app)
    yield


app = FastAPI(
    title="Credit Scoring API",
    description="Simple educational API for default probability scoring.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(scoring_router)

