"""Lightweight safety and grounding guardrails for financial RAG.

Tune FINRAG_MIN_RERANK_SCORE against a labeled validation set before production.
"""
import os
import re

ABSTAIN = "Insufficient information in the provided documents."

_ADVICE_RE = re.compile(
    r"\b(should i (buy|sell|hold|invest)|recommend (a|me|whether)|"
    r"what should i do|personal financial advice|which stock should i buy|"
    r"portfolio allocation)\b", re.I
)
_PRICE_PREDICTION_RE = re.compile(
    r"\b(price target|predict.*(stock|share|crypto|bitcoin).*price|"
    r"(stock|share|crypto|bitcoin).*price.*(next|future|tomorrow|end of)|"
    r"will .*stock.*(go up|go down|reach)|forecast.*share price)\b", re.I
)

def blocked_request(query: str) -> str | None:
    """Return a reason if the query requests advice or future price prediction."""
    if _ADVICE_RE.search(query):
        return "personalized financial advice"
    if _PRICE_PREDICTION_RE.search(query):
        return "future asset-price prediction"
    return None

def min_reranker_score() -> float:
    """Configurable abstention threshold; calibrate using held-out examples."""
    try:
        return float(os.getenv("FINRAG_MIN_RERANK_SCORE", "-5.0"))
    except ValueError:
        return -5.0

def numeric_consistency(answer: str, context: str) -> tuple[bool, list[str]]:
    """Flag answer numbers not present in the retrieved context (format-tolerant)."""
    def numbers(text: str) -> set[str]:
        out = set()
        for raw in re.findall(r"(?<![A-Za-z])[-+]?\$?\d[\d,]*(?:\.\d+)?%?", text):
            token = raw.replace("$", "").replace(",", "").replace("%", "")
            try:
                value = float(token)
                out.add(str(int(value)) if value.is_integer() else str(value))
            except ValueError:
                pass
        return out
    context_numbers = numbers(context)
    unsupported = sorted(numbers(answer) - context_numbers)
    return not unsupported, unsupported
