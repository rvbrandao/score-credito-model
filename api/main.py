from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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

# Open CORS for educational local frontend usage.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(scoring_router)


