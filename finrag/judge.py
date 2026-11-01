import json
import re
import time

from .models import get_provider_llm

PROMPT = """You are grading an answer to a financial question against a reference answer.
Question: {q}
Reference answer: {gold}
Candidate answer: {pred}

The candidate is CORRECT if its final value/conclusion matches the reference (rounding and unit/format differences are fine; extra reasoning is fine). Otherwise INCORRECT.
Return only JSON: {{"verdict": "correct" | "incorrect", "reason": "<one sentence>"}}"""

PROMPT_EV = """You are the reference labeler for a financial QA evaluation. Treat the SOURCE EVIDENCE as ground truth.
Question: {q}
Source evidence: {ev}
Reference answer (may be imperfect): {gold}
Candidate answer: {pred}

The candidate is CORRECT if its final value/conclusion is supported by the evidence (rounding and unit/format differences are fine; extra reasoning is fine). Otherwise INCORRECT.
Return only JSON: {{"verdict": "correct" | "incorrect", "reason": "<one sentence>"}}"""


def _text(resp):
    c = resp.content if hasattr(resp, "content") else resp
    if isinstance(c, list):
        return "\n".join(p if isinstance(p, str) else p.get("text", "") for p in c)
    return str(c)


class Judge:
    def __init__(self, provider, model, other_provider, other_model):
        if provider == other_provider and (model or None) == (other_model or None):
            raise SystemExit(f"{provider}:{model} must differ from {other_provider}:{other_model}.")
        self.llm = get_provider_llm(provider, model)
        self.name = f"{provider}:{model or 'default'}"

    def grade(self, q, gold, pred, evidence=None, retries=4):
        prompt = (PROMPT_EV.format(q=q, ev=evidence[:3000], gold=gold, pred=pred) if evidence
                  else PROMPT.format(q=q, gold=gold, pred=pred))
        err = ""
        for i in range(retries):
            try:
                raw = _text(self.llm.invoke(prompt))
                m = re.search(r"\{.*\}", raw, re.S)
                if m:
                    d = json.loads(m.group(0))
                    v = str(d.get("verdict", "")).lower()
                    if v in ("correct", "incorrect"):
                        return v, d.get("reason", "")
                low = raw.lower()
                return ("incorrect" if "incorrect" in low else "correct" if "correct" in low else "incorrect"), raw[:200]
            except Exception as e:
                err = str(e)
                time.sleep(2 ** i)
        return "error", err
