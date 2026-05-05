"""Configuration for the CE compliance experiment pipeline."""

from __future__ import annotations

import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "results"

API_BASE_URL = os.getenv("AUTODL_API_BASE_URL", "https://www.autodl.art/api/v1/")

DEFAULT_MODEL_ALIASES = {
    "gpt-5.4": "gpt-5.4",
    "claude-opus-4.6": "claude-opus-4-6",
    "gemini-3.1-pro": "gemini-3.1-pro-preview",
    "deepseek-v4-pro": "deepseek-v4-pro",
}

DEFAULT_N = int(os.getenv("CE_EXPERIMENT_N", "20"))
DEFAULT_CONCURRENCY = int(os.getenv("CE_EXPERIMENT_CONCURRENCY", "5"))
DEFAULT_TEMPERATURE = float(os.getenv("CE_EXPERIMENT_TEMPERATURE", "1.0"))
DEFAULT_TOP_P = float(os.getenv("CE_EXPERIMENT_TOP_P", "1.0"))
DEFAULT_MAX_TOKENS = int(os.getenv("CE_EXPERIMENT_MAX_TOKENS", "2048"))
DEFAULT_REQUEST_TIMEOUT = float(os.getenv("CE_EXPERIMENT_REQUEST_TIMEOUT", "120"))


def _read_key_file(path: Path) -> str | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return None
    if "=" in text:
        for line in text.splitlines():
            if line.strip().startswith("AUTODL_API_KEY"):
                return line.split("=", 1)[1].strip().strip("\"'")
    return text.splitlines()[0].strip()


def get_api_key() -> str:
    """Return API key from env first, then local key files."""

    key = os.getenv("AUTODL_API_KEY")
    if key:
        return key
    for filename in ("API_KEY.md", "API_KEY"):
        key = _read_key_file(ROOT_DIR / filename)
        if key:
            return key
    raise RuntimeError(
        "Missing AutoDL API key. Set AUTODL_API_KEY or put it in API_KEY.md."
    )


def parse_model_alias_overrides() -> dict[str, str]:
    """Parse CE_MODEL_ALIASES='friendly=actual,other=actual2'."""

    aliases = dict(DEFAULT_MODEL_ALIASES)
    raw = os.getenv("CE_MODEL_ALIASES", "").strip()
    if not raw:
        return aliases
    for item in raw.split(","):
        if "=" not in item:
            continue
        friendly, actual = item.split("=", 1)
        aliases[friendly.strip()] = actual.strip()
    return aliases


MODEL_ALIASES = parse_model_alias_overrides()
DEFAULT_MODELS = list(MODEL_ALIASES)
