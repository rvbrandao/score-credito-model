from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI

from api.model_loader import load_model
from api.schemas import ScoreRequest, ScoreResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model()
    yield


app = FastAPI(
    title="Credit Scoring API",
    description="Simple educational API for default probability scoring.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.post("/score", response_model=ScoreResponse)
def score(request: ScoreRequest) -> ScoreResponse:
    features = np.array(
        [
            [
                request.age,
                request.income,
                request.number_of_loans,
                request.payment_delays,
            ]
        ],
        dtype=float,
    )

    probability_default = float(app.state.model.predict_proba(features)[0][1])
    return ScoreResponse(probability_default=round(probability_default, 4))
