"""Retrieval evaluation — NDCG@k over ALL labeled queries (zero-hit queries score 0)."""
import logging

import numpy as np

from .metrics import ndcg_at_k, ranked_ids

logger = logging.getLogger(__name__)


def compute_ndcg(qrels, results, k=10) -> float:
    vals = [
        ndcg_at_k(ranked_ids(results[q]), qrels[q], k)
        for q in qrels if q in results
    ]
    vals = [v for v in vals if v == v]
    if not vals:
        logger.warning("No scoreable queries found — check qrels alignment.")
        return float("nan")
    hit = sum(v > 0 for v in vals)
    logger.info("Hit-rate@%d: %d / %d (%.1f%%)", k, hit, len(vals), 100 * hit / len(vals))
    return float(np.mean(vals))
