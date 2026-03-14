from pydantic import BaseModel, Field


class ScoreRequest(BaseModel):
    age: int = Field(..., ge=18, le=100, description="Customer age in years")
    income: float = Field(..., gt=0, description="Monthly income")
    number_of_loans: int = Field(..., ge=0, le=20, description="Active loans")
    payment_delays: int = Field(
        ...,
        ge=0,
        le=120,
        description="Late payments count",
    )


class ScoreResponse(BaseModel):
    probability_default: float = Field(..., ge=0.0, le=1.0)
