import os
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import asdict

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

STATE = {}
LAT, COST, ABST = deque(maxlen=5000), deque(maxlen=5000), deque(maxlen=5000)
LOCK = threading.Lock()


@asynccontextmanager
async def lifespan(app):
    from finrag.engine import Engine
    STATE["engine"] = Engine(os.getenv("FINRAG_DATASET", "financebench"),
                             os.getenv("FINRAG_PROVIDER", "gemini"),
                             os.getenv("FINRAG_MODEL") or None)
    yield


app = FastAPI(title="FinRAG", lifespan=lifespan)


class Req(BaseModel):
    query: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(5, ge=1, le=10)
    multiquery: bool = True


@app.post("/query")
def query(r: Req):
    t = time.perf_counter()
    a = STATE["engine"].answer(r.query, top_k=r.top_k, use_multiquery=r.multiquery)
    ms = (time.perf_counter() - t) * 1000
    with LOCK:
        LAT.append(ms)
        COST.append(a.est_cost_usd)
        ABST.append(a.abstained)
    d = asdict(a)
    d["latency_ms"]["wall"] = ms
    return d


@app.get("/health")
def health():
    return {"status": "ok", "ready": "engine" in STATE}


@app.get("/metrics")
def metrics():
    with LOCK:
        lat, cost, ab = np.array(LAT), np.array(COST), np.array(ABST)
    if not len(lat):
        return {"n": 0}
    return {"n": int(len(lat)), "p50_ms": float(np.percentile(lat, 50)), "p95_ms": float(np.percentile(lat, 95)),
            "p99_ms": float(np.percentile(lat, 99)), "mean_cost_usd": float(cost.mean()),
            "abstain_rate": float(ab.mean())}
