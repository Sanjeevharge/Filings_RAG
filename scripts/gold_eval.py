import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from finrag.engine import Engine
from finrag.judge import Judge
from finrag.metrics import bootstrap_ci

ap = argparse.ArgumentParser()
ap.add_argument("--questions", default="data/gold_questions.jsonl")
ap.add_argument("--dataset", default="financebench")
ap.add_argument("--provider", default="gemini")
ap.add_argument("--model", default=None)
ap.add_argument("--judge-provider", default="groq")
ap.add_argument("--judge-model", default=None)
ap.add_argument("--no-multiquery", action="store_true")
ap.add_argument("--top-k", type=int, default=5)
ap.add_argument("--out", default="results/gold_eval.csv")
ap.add_argument("--iters", type=int, default=2000)
a = ap.parse_args()

qs = [json.loads(l) for l in open(a.questions, encoding="utf-8") if l.strip()]
qs = [q for q in qs if q.get("gold_answer")]
if len(qs) < 100:
    print(f"WARNING: n={len(qs)} < 100")
judge = Judge(a.judge_provider, a.judge_model, a.provider, a.model)
eng = Engine(a.dataset, a.provider, a.model)

out = Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)
done = pd.read_csv(out) if out.exists() else pd.DataFrame()
seen = set(done["_id"].astype(str)) if len(done) else set()
rows = done.to_dict("records")
for q in tqdm(qs):
    if str(q["_id"]) in seen:
        continue
    x = eng.answer(q["question"], top_k=a.top_k, use_multiquery=not a.no_multiquery)
    if x.abstained:
        v, why = "abstain", x.reason
    else:
        v, why = judge.grade(q["question"], q["gold_answer"], x.answer)
    rows.append({"_id": q["_id"], "question": q["question"], "gold_answer": q["gold_answer"],
                 "evidence": q.get("evidence", "")[:3000],
                 "answer": x.answer, "verdict": v, "judge_reason": why, "judge": judge.name,
                 "top_score": x.top_score, "latency_ms": x.latency_ms.get("total"),
                 "est_cost_usd": x.est_cost_usd, "source_ids": json.dumps([s["id"] for s in x.sources])})
    pd.DataFrame(rows).to_csv(out, index=False)

df = pd.DataFrame(rows)
df = df[df["verdict"] != "error"]
n = len(df)
acc_all = bootstrap_ci((df["verdict"] == "correct").astype(float), a.iters)
ans = df[df["verdict"] != "abstain"]
acc_ans = bootstrap_ci((ans["verdict"] == "correct").astype(float), a.iters)
abst = bootstrap_ci((df["verdict"] == "abstain").astype(float), a.iters)
print(f"n={n} judge={judge.name}")
print(f"accuracy (abstain=wrong): {acc_all[0]:.3f} (95% CI {acc_all[1]:.3f}, {acc_all[2]:.3f})")
print(f"accuracy when answered : {acc_ans[0]:.3f} (95% CI {acc_ans[1]:.3f}, {acc_ans[2]:.3f}) n={len(ans)}")
print(f"abstain rate           : {abst[0]:.3f} (95% CI {abst[1]:.3f}, {abst[2]:.3f})")
