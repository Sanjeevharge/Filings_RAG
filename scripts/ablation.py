import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from finrag.config import DATASET_CONFIGS, DATASET_DIR
from finrag.data import load_jsonl, load_qrels
from finrag.engine import MODES, Engine
from finrag.metrics import bootstrap_ci, mrr_at_k, ndcg_at_k, paired_bootstrap, recall_at_k

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="financebench")
ap.add_argument("--provider", default="gemini")
ap.add_argument("--model", default=None)
ap.add_argument("--modes", nargs="+", default=MODES)
ap.add_argument("--k", type=int, default=10)
ap.add_argument("--iters", type=int, default=2000)
ap.add_argument("--out", default="results/ablation")
a = ap.parse_args()

out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
cfg = DATASET_CONFIGS[a.dataset]
qrels = load_qrels(DATASET_DIR / cfg.qrels_file)
queries = [q for q in load_jsonl(DATASET_DIR / cfg.query_file) if q["_id"] in qrels]
print(f"n={len(queries)} labeled queries")
eng = Engine(a.dataset, a.provider, a.model)

per = {}
for mode in a.modes:
    cache = out / f"{a.dataset}_{mode}.json"
    runs = json.loads(cache.read_text()) if cache.exists() else {}
    for q in tqdm(queries, desc=mode):
        if q["_id"] in runs:
            continue
        runs[q["_id"]] = [d for d, _ in eng.retrieve(q["text"], mode, k=a.k)]
        cache.write_text(json.dumps(runs))
    per[mode] = pd.DataFrame({
        "qid": [q["_id"] for q in queries],
        "ndcg": [ndcg_at_k(runs[q["_id"]], qrels[q["_id"]], a.k) for q in queries],
        "recall": [recall_at_k(runs[q["_id"]], qrels[q["_id"]], a.k) for q in queries],
        "mrr": [mrr_at_k(runs[q["_id"]], qrels[q["_id"]], a.k) for q in queries],
    })

rows, prev = [], None
for mode, df in per.items():
    r = {"mode": mode, "n": len(df)}
    for m in ("ndcg", "recall", "mrr"):
        mean, lo, hi = bootstrap_ci(df[m], a.iters)
        r[f"{m}@{a.k}"] = f"{mean:.3f} [{lo:.3f}, {hi:.3f}]"
    if prev is not None:
        d, lo, hi, p = paired_bootstrap(per[prev]["ndcg"], df["ndcg"], a.iters)
        r[f"Δndcg vs {prev}"] = f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}] p={p:.3f}"
    rows.append(r)
    prev = mode

res = pd.DataFrame(rows)
res.to_csv(out / f"{a.dataset}_ablation.csv", index=False)
(out / f"{a.dataset}_ablation.md").write_text(res.to_markdown(index=False))
print(res.to_markdown(index=False))
