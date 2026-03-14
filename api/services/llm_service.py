import json
import os
from urllib import error, request

from fastapi import HTTPException
from pydantic import ValidationError

from api.core.config import DEFAULT_GEMINI_MODEL
from api.core.config import load_env_file_if_present
from api.schemas import ExtractedCreditProfile


def _clean_json_text(text: str) -> str:
    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json\n", "", 1)

    return cleaned.strip()


def _extract_text(payload: dict) -> str:
    candidates = payload.get("candidates", [])
    if not candidates:
        raise HTTPException(
            status_code=502,
            detail="Gemini returned no candidates for extraction.",
        )

    content = candidates[0].get("content", {})
    parts = content.get("parts", [])
    text_parts = [part.get("text", "") for part in parts if part.get("text")]

    if not text_parts:
        raise HTTPException(
            status_code=502,
            detail="Gemini returned an empty extraction response.",
        )

    return "".join(text_parts).strip()


def _build_prompt(message: str) -> str:
    return f"""
You extract credit scoring fields from natural language.

Return only valid JSON with exactly these keys:
- age
- income
- number_of_loans
- payment_delays

Rules:
- Use numbers only.
- If a value is missing or uncertain, return null.
- Do not invent values.
- Do not include extra keys.

User message:
{message}
""".strip()


def extract_credit_profile(message: str) -> ExtractedCreditProfile:
    load_env_file_if_present()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail=(
                "GEMINI_API_KEY is not configured. Set this environment "
                "variable to use the conversational scoring endpoint."
            ),
        )

    model_name = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model_name}:generateContent?key={api_key}"
    )
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": _build_prompt(message),
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
        },
    }

    body = json.dumps(payload).encode("utf-8")
    http_request = request.Request(
        url=url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(http_request, timeout=30) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="ignore")
        raise HTTPException(
            status_code=502,
            detail=(
                "Gemini request failed. "
                f"Provider response: {error_body or exc.reason}"
            ),
        ) from exc
    except error.URLError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Could not reach Gemini: {exc.reason}",
        ) from exc

    raw_text = _extract_text(response_payload)
    cleaned_text = _clean_json_text(raw_text)

    try:
        extracted_payload = json.loads(cleaned_text)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=502,
            detail="Gemini returned invalid JSON for extracted fields.",
        ) from exc

    try:
        return ExtractedCreditProfile.model_validate(extracted_payload)
    except ValidationError as exc:
        raise HTTPException(
            status_code=502,
            detail="Gemini returned fields outside the expected schema.",
        ) from exc
