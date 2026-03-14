from fastapi import APIRouter, Request

from api.core.config import MODEL_FEATURES
from api.core.model_store import get_model_from_app
from api.schemas import ChatScoreRequest
from api.schemas import ChatScoreResponse
from api.schemas import ScoreRequest
from api.schemas import ScoreResponse
from api.services.chat_memory_service import build_context_for_extraction
from api.services.chat_memory_service import register_user_message
from api.services.llm_service import extract_credit_profile
from api.services.scoring_service import build_credit_decision_guidance
from api.services.scoring_service import build_risk_explanation
from api.services.scoring_service import is_credit_decision_question
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
    conversation_id, turn, recent_messages = register_user_message(
        request.message,
        request.conversation_id,
    )
    context_message = build_context_for_extraction(recent_messages)
    extracted_profile = extract_credit_profile(context_message)

    missing_fields = [
        field_name
        for field_name in MODEL_FEATURES
        if getattr(extracted_profile, field_name) is None
    ]

    if missing_fields:
        missing_fields_text = ", ".join(missing_fields)
        return ChatScoreResponse(
            conversation_id=conversation_id,
            turn=turn,
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
    if is_credit_decision_question(request.message):
        explanation = build_credit_decision_guidance(probability_default)
    else:
        explanation = build_risk_explanation(probability_default)

    return ChatScoreResponse(
        conversation_id=conversation_id,
        turn=turn,
        extracted_data=extracted_profile,
        missing_fields=[],
        probability_default=probability_default,
        explanation=explanation,
    )

