from fastapi import APIRouter, Request

from api.core.config import MODEL_FEATURES
from api.core.model_store import get_model_from_app
from api.schemas import ChatScoreRequest
from api.schemas import ChatScoreResponse
from api.schemas import ScoreRequest
from api.schemas import ScoreResponse
from api.services.llm_service import extract_credit_profile
from api.services.scoring_service import build_risk_explanation
from api.services.scoring_service import predict_probability

router = APIRouter(tags=["scoring"])


@router.post("/score", response_model=ScoreResponse)
def score(request: ScoreRequest, http_request: Request) -> ScoreResponse:
    model = get_model_from_app(http_request.app)
    probability_default = predict_probability(model, request)
    return ScoreResponse(probability_default=probability_default)


@router.post("/score/chat", response_model=ChatScoreResponse)
def score_chat(
    request: ChatScoreRequest,
    http_request: Request,
) -> ChatScoreResponse:
    extracted_profile = extract_credit_profile(request.message)
    missing_fields = [
        field_name
        for field_name in MODEL_FEATURES
        if getattr(extracted_profile, field_name) is None
    ]

    if missing_fields:
        missing_fields_text = ", ".join(missing_fields)
        return ChatScoreResponse(
            extracted_data=extracted_profile,
            missing_fields=missing_fields,
            explanation=(
                "Nao foi possivel calcular o score ainda. "
                f"Informe os campos faltantes: {missing_fields_text}."
            ),
        )

    score_request = ScoreRequest(
        age=extracted_profile.age,
        income=extracted_profile.income,
        number_of_loans=extracted_profile.number_of_loans,
        payment_delays=extracted_profile.payment_delays,
    )

    model = get_model_from_app(http_request.app)
    probability_default = predict_probability(model, score_request)
    explanation = build_risk_explanation(probability_default)

    return ChatScoreResponse(
        extracted_data=extracted_profile,
        missing_fields=[],
        probability_default=probability_default,
        explanation=explanation,
    )
