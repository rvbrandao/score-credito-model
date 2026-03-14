from pydantic import BaseModel, Field


class ChatScoreRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=10,
        description="Natural language description of the customer profile",
    )


class ExtractedCreditProfile(BaseModel):
    age: int | None = Field(default=None, ge=18, le=100)
    income: float | None = Field(default=None, gt=0)
    number_of_loans: int | None = Field(default=None, ge=0, le=20)
    payment_delays: int | None = Field(default=None, ge=0, le=120)


class ChatScoreResponse(BaseModel):
    extracted_data: ExtractedCreditProfile
    missing_fields: list[str] = Field(default_factory=list)
    probability_default: float | None = Field(default=None, ge=0.0, le=1.0)
    explanation: str
