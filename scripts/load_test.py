import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://localhost:8000/query")
ap.add_argument("--questions", default="data/gold_questions.jsonl")
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--concurrency", type=int, default=4)
ap.add_argument("--warmup", type=int, default=3)
a = ap.parse_args()

qs = [json.loads(l)["question"] for l in open(a.questions, encoding="utf-8") if l.strip()]
qs = (qs * (a.n // len(qs) + 1))[: a.n + a.warmup]


def call(q):
    t = time.perf_counter()
    try:
        r = httpx.post(a.url, json={"query": q}, timeout=120)
        r.raise_for_status()
        return (time.perf_counter() - t) * 1000, r.json(), None
    except Exception as e:
        return (time.perf_counter() - t) * 1000, {}, str(e)


for q in qs[: a.warmup]:
    call(q)
t0 = time.perf_counter()
with ThreadPoolExecutor(a.concurrency) as ex:
    res = list(ex.map(call, qs[a.warmup:]))
wall = time.perf_counter() - t0
ok = [r for r in res if r[2] is None]
lat = np.array([r[0] for r in ok])
cost = np.array([r[1].get("est_cost_usd", 0) for r in ok])
print(f"requests={len(res)} errors={len(res) - len(ok)} concurrency={a.concurrency} throughput={len(ok) / wall:.2f} rps")
print(f"latency ms  p50={np.percentile(lat, 50):.0f}  p95={np.percentile(lat, 95):.0f}  p99={np.percentile(lat, 99):.0f}")
print(f"est cost/query USD  mean={cost.mean():.6f}  p95={np.percentile(cost, 95):.6f}")
