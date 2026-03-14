import numpy as np

from api.schemas import ScoreRequest


def predict_probability(model: object, request: ScoreRequest) -> float:
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

    probability = float(model.predict_proba(features)[0][1])
    return round(probability, 4)


def build_risk_explanation(probability_default: float) -> str:
    if probability_default < 0.2:
        risk_level = "baixo"
    elif probability_default < 0.5:
        risk_level = "moderado"
    else:
        risk_level = "alto"

    return (
        "O perfil foi convertido em dados estruturados e avaliado pelo "
        "modelo de regressao logistica. "
        f"A probabilidade estimada de default e {probability_default:.4f}, "
        f"o que indica risco {risk_level} neste modelo educacional."
    )
