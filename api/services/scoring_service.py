import numpy as np
import unicodedata

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


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return ascii_text.lower().strip()


def is_credit_decision_question(message: str) -> bool:
    normalized = _normalize_text(message)
    decision_keywords = (
        "liberar credito",
        "aprovar credito",
        "aprovar emprestimo",
        "conceder credito",
        "dar credito",
        "posso liberar",
        "devo liberar",
        "pode aprovar",
        "vale aprovar",
        "libera",
        "aprovado",
    )
    return any(keyword in normalized for keyword in decision_keywords)


def build_credit_decision_guidance(probability_default: float) -> str:
    if probability_default < 0.2:
        return (
            "Neste exemplo educacional, esse perfil sugere risco baixo. "
            "Em uma politica simples de estudo, ele poderia seguir para uma "
            "aprovacao inicial ou analise simplificada. Ainda assim, isso nao "
            "significa liberacao automatica: em um cenario real seria preciso "
            "avaliar politica de credito, documentos, fraude, limite e renda "
            "compativel."
        )

    if probability_default < 0.5:
        return (
            "Neste exemplo educacional, o perfil ficou em risco moderado. "
            "Eu nao recomendaria liberacao automatica. "
            "O caminho mais coerente "
            "seria enviar para revisao manual e aplicar regras adicionais de "
            "politica de credito antes de decidir."
        )

    return (
        "Neste exemplo educacional, o perfil indica risco alto. Eu nao "
        "recomendaria liberar credito automaticamente com esse resultado. "
        "O mais prudente seria reprovar na esteira automatica ou exigir uma "
        "analise manual mais restritiva."
    )
