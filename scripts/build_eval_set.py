import argparse
import difflib
import json
import random
import re
from pathlib import Path

import pandas as pd

from finrag.config import DATASET_CONFIGS, DATASET_DIR
from finrag.data import load_jsonl
from finrag.judge import _text
from finrag.models import get_provider_llm

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="financebench")
ap.add_argument("--hf", default="PatronusAI/financebench")
ap.add_argument("--csv", default="guardrail_predictions.csv")
ap.add_argument("--n-answerable", type=int, default=20)
ap.add_argument("--heldout-provider", default=None)
ap.add_argument("--heldout-model", default=None)
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


from datasets import load_dataset

gold = load_dataset(a.hf, split="train")
gmap = {norm(r["question"]): r for r in gold}
cfg = DATASET_CONFIGS[a.dataset]
queries = load_jsonl(DATASET_DIR / cfg.query_file)
keys = list(gmap)
matched = []
for q in queries:
    k = norm(q["text"])
    if k not in gmap:
        c = difflib.get_close_matches(k, keys, n=1, cutoff=0.95)
        k = c[0] if c else None
    if k:
        r = gmap[k]
        ev = " ".join(e.get("evidence_text", "") for e in (r.get("evidence") or []) if isinstance(e, dict))
        matched.append({"_id": q["_id"], "question": q["text"], "gold_answer": r["answer"],
                        "company": r.get("company"), "question_type": r.get("question_type"), "evidence": ev})
print(f"matched {len(matched)}/{len(queries)} queries to gold answers")
if not matched:
    raise SystemExit("No gold answers matched; hand-label instead.")
Path("data").mkdir(exist_ok=True)
with open("data/gold_questions.jsonl", "w", encoding="utf-8") as f:
    for m in matched:
        f.write(json.dumps(m) + "\n")

df = pd.read_csv(a.csv)
adv = df[df["category"] != "needs_labeled_question"].copy()
random.Random(a.seed).shuffle(matched)
ans = pd.DataFrame([{
    "id": f"ANS{i+1:03d}", "question": m["question"], "category": "answerable",
    "expected_abstain": False, "actual_abstain": "", "expected_refusal": False, "actual_refusal": "",
    "gold_answer": m["gold_answer"]} for i, m in enumerate(matched[: a.n_answerable])])
pd.concat([adv, ans], ignore_index=True).to_csv(a.csv, index=False)
print(f"wrote {a.csv}: {len(adv)} adversarial + {len(ans)} answerable")

if a.heldout_provider:
    llm = get_provider_llm(a.heldout_provider, a.heldout_model)
    rows = []
    for _, r in adv.iterrows():
        p = ("Rewrite this user prompt in 2 clearly different ways (different vocabulary and sentence structure, "
             f"same intent). Return only a JSON list of 2 strings.\nPrompt: {r['question']}")
        m = re.search(r"\[.*\]", _text(llm.invoke(p)), re.S)
        for j, s in enumerate(json.loads(m.group(0))[:2] if m else []):
            rows.append({**r.to_dict(), "id": f"HLD-{r['id']}-{j}", "question": s})
    pd.concat([pd.DataFrame(rows), ans], ignore_index=True).to_csv("data/guardrail_heldout.csv", index=False)
    print(f"wrote data/guardrail_heldout.csv: {len(rows)} held-out adversarial + {len(ans)} answerable")
