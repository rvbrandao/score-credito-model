from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"
MODEL_PATH = PROJECT_ROOT / "models" / "credit_model.pkl"

MODEL_FEATURES = (
    "age",
    "income",
    "number_of_loans",
    "payment_delays",
)

DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"

_ENV_LOADED = False


def load_env_file_if_present() -> None:
    global _ENV_LOADED

    if _ENV_LOADED or not ENV_FILE.exists():
        _ENV_LOADED = True
        return

    for raw_line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()

        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key:
            # Do not override values injected by the process environment.
            import os

            os.environ.setdefault(key, value)

    _ENV_LOADED = True
