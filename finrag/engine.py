import logging
import os
import time
from dataclasses import dataclass, field

from .chunking import split_documents
from .config import DATASET_CONFIGS, DATASET_DIR
from .data import load_jsonl, make_corpus_lookup
from .generation import generate_answer
from .guardrails import ABSTAIN, blocked_request, min_reranker_score
from .models import get_embedding_model, get_provider_llm, get_reranker
from .retrieval import (build_bm25_okapi, build_bm25_retriever, retrieve_and_rerank, rrf_fuse)
from .vectorstore import get_vectorstore

logger = logging.getLogger(__name__)

MODES = ["bm25", "dense", "hybrid", "rerank", "rerank_mq"]
REFUSAL = ("I can help summarize historical financial disclosures, but I cannot provide "
           "personalized investment advice or predict future asset prices.")
PRICE_IN = float(os.getenv("PRICE_IN_PER_M", "0"))
PRICE_OUT = float(os.getenv("PRICE_OUT_PER_M", "0"))


@dataclass
class Answer:
    query: str
    answer: str
    refused: bool = False
    abstained: bool = False
    reason: str | None = None
    top_score: float | None = None
    sources: list = field(default_factory=list)
    latency_ms: dict = field(default_factory=dict)
    est_cost_usd: float = 0.0
    model: str = ""


def _uniq(ids):
    seen, out = set(), []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


class Engine:
    def __init__(self, dataset="financebench", provider="gemini", model=None, rebuild=False):
        cfg = DATASET_CONFIGS[dataset]
        self.cfg, self.dataset, self.provider = cfg, dataset, provider
        self.model = model or provider
        corpus = load_jsonl(DATASET_DIR / cfg.corpus_file)
        self.lookup = make_corpus_lookup(corpus)
        self.chunks = split_documents(corpus, cfg.chunk_size, cfg.chunk_overlap, cfg.dataset_type)
        self.reranker = get_reranker()
        self.llm = get_provider_llm(provider, model)
        self.vs = get_vectorstore(cfg.name, self.chunks, get_embedding_model(), force_rebuild=rebuild)
        self.bm25 = build_bm25_retriever(self.chunks.lc_docs, fetch_k=cfg.fetch_k)
        self.okapi = build_bm25_okapi(self.chunks.texts)

    def _dense(self, q):
        c = self.cfg
        if c.dataset_type == "passage":
            docs = self.vs.max_marginal_relevance_search(q, k=c.fetch_k, fetch_k=c.fetch_k * 3, lambda_mult=0.7)
        else:
            docs = self.vs.similarity_search(q, k=c.fetch_k)
        return _uniq(d.metadata["id"] for d in docs)

    def _bm25(self, q):
        return _uniq(d.metadata["id"] for d in self.bm25.invoke(q))

    def retrieve(self, query, mode="rerank_mq", k=10):
        if mode == "bm25":
            return [(d, 1 / (i + 1)) for i, d in enumerate(self._bm25(query)[:k])]
        if mode == "dense":
            return [(d, 1 / (i + 1)) for i, d in enumerate(self._dense(query)[:k])]
        if mode == "hybrid":
            return rrf_fuse([self._dense(query), self._bm25(query)])[:k]
        if mode == "rerank_mq" and self.llm is None:
            raise RuntimeError("rerank_mq needs an LLM")
        return retrieve_and_rerank(
            vectorstore=self.vs, bm25_retriever=self.bm25, bm25_okapi=self.okapi,
            chunk_result=self.chunks, query=query, reranker=self.reranker,
            dataset_type=self.cfg.dataset_type, fetch_k=self.cfg.fetch_k,
            llm=self.llm if mode == "rerank_mq" else None, k=k,
            rerank_top_n=self.cfg.rerank_top_n,
        )

    def _cost(self, query, answer, ranked, top_k, multiquery):
        ctx = sum(min(1500, len(self.lookup.get(d, {}).get("text", ""))) for d, _ in ranked[:top_k])
        tin = (len(query) + ctx + 900) / 4
        tout = (len(answer) + 40) / 4
        if multiquery:
            tin += (len(query) + 250) / 4
            tout += 80
        return (tin * PRICE_IN + tout * PRICE_OUT) / 1e6

    def answer(self, query, top_k=5, use_multiquery=True, threshold=None) -> Answer:
        t0 = time.perf_counter()
        lat = {}
        reason = blocked_request(query)
        if reason:
            lat["total"] = (time.perf_counter() - t0) * 1000
            return Answer(query, REFUSAL, True, True, f"blocked:{reason}", None, [], lat, 0.0, "guardrail")

        t = time.perf_counter()
        ranked = self.retrieve(query, "rerank_mq" if use_multiquery else "rerank", k=max(top_k, 10))
        lat["retrieval"] = (time.perf_counter() - t) * 1000
        top = float(ranked[0][1]) if ranked else None
        th = min_reranker_score() if threshold is None else threshold
        if top is None or top < th:
            lat["total"] = (time.perf_counter() - t0) * 1000
            return Answer(query, ABSTAIN, False, True, "low_rerank_score", top, [], lat, 0.0, self.model)

        t = time.perf_counter()
        res = generate_answer(query=query, retrieved=ranked, corpus_lookup=self.lookup, llm=self.llm,
                              top_k=top_k, multiquery=use_multiquery, model_name=self.model)
        lat["generation"] = (time.perf_counter() - t) * 1000
        lat["total"] = (time.perf_counter() - t0) * 1000
        abst = res.answer.startswith(ABSTAIN)
        why = None
        if abst:
            why = "numeric_inconsistency" if "figures not supported" in res.answer else "model_abstain"
        srcs = [{"id": s.corpus_id, "title": s.title, "score": s.score, "excerpt": s.excerpt} for s in res.sources]
        return Answer(query, res.answer, False, abst, why, top, srcs, lat,
                      self._cost(query, res.answer, ranked, top_k, use_multiquery), self.model)
