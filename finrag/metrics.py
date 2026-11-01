import numpy as np


def _rel(rel):
    return {d: s for d, s in rel.items() if s > 0}


def ranked_ids(scores: dict) -> list:
    return [d for d, _ in sorted(scores.items(), key=lambda x: -x[1])]


def ndcg_at_k(ranked, rel, k=10):
    rel = _rel(rel)
    if not rel:
        return float("nan")
    disc = 1 / np.log2(np.arange(2, k + 2))
    gains = np.array([rel.get(d, 0) for d in ranked[:k]], float)
    ideal = np.array(sorted(rel.values(), reverse=True)[:k], float)
    return float((gains * disc[: len(gains)]).sum() / (ideal * disc[: len(ideal)]).sum())


def recall_at_k(ranked, rel, k=10):
    rel = _rel(rel)
    return len(set(ranked[:k]) & set(rel)) / len(rel) if rel else float("nan")


def mrr_at_k(ranked, rel, k=10):
    for i, d in enumerate(ranked[:k], 1):
        if rel.get(d, 0) > 0:
            return 1.0 / i
    return 0.0


def bootstrap_ci(values, iters=2000, seed=42):
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), (iters, len(v)))].mean(1)
    return float(v.mean()), float(np.quantile(means, .025)), float(np.quantile(means, .975))


def paired_bootstrap(a, b, iters=2000, seed=42):
    d = np.asarray(b, float) - np.asarray(a, float)
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), (iters, len(d)))].mean(1)
    p = min(1.0, 2 * min((m <= 0).mean(), (m >= 0).mean()))
    return float(d.mean()), float(np.quantile(m, .025)), float(np.quantile(m, .975)), float(p)


def bootstrap_stat(fn, *arrays, iters=2000, seed=42):
    arrays = [np.asarray(a) for a in arrays]
    n = len(arrays[0])
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(iters):
        idx = rng.integers(0, n, n)
        v = fn(*[a[idx] for a in arrays])
        if np.isfinite(v):
            vals.append(v)
    point = fn(*arrays)
    if not vals:
        return point, float("nan"), float("nan")
    return float(point), float(np.quantile(vals, .025)), float(np.quantile(vals, .975))


def recall(exp, act):
    exp, act = np.asarray(exp, bool), np.asarray(act, bool)
    return float((exp & act).sum() / exp.sum()) if exp.sum() else float("nan")


def precision(exp, act):
    exp, act = np.asarray(exp, bool), np.asarray(act, bool)
    return float((exp & act).sum() / act.sum()) if act.sum() else float("nan")


def false_positive_rate(exp, act):
    exp, act = np.asarray(exp, bool), np.asarray(act, bool)
    return float((~exp & act).sum() / (~exp).sum()) if (~exp).sum() else float("nan")
