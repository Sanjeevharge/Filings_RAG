"""Bootstrap guardrail metrics from a labeled predictions CSV."""
import argparse
import numpy as np
import pandas as pd

def boolean_col(series):
    return series.astype(str).str.strip().str.lower().map({
        "true": True, "1": True, "yes": True, "y": True,
        "false": False, "0": False, "no": False, "n": False
    })

def recall(expected, actual):
    denom = int(expected.sum())
    return float((expected & actual).sum() / denom) if denom else float("nan")

def precision(expected, actual):
    denom = int(actual.sum())
    return float((expected & actual).sum() / denom) if denom else float("nan")

def bootstrap_ci(metric, expected, actual, iterations=2000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(expected)
    vals = []
    for _ in range(iterations):
        idx = rng.integers(0, n, size=n)
        v = metric(expected[idx], actual[idx])
        if np.isfinite(v): vals.append(v)
    return (float(np.quantile(vals, .025)), float(np.quantile(vals, .975))) if vals else (float("nan"), float("nan"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--iterations", type=int, default=2000)
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    required = ["expected_abstain", "actual_abstain", "expected_refusal", "actual_refusal"]
    missing = [c for c in required if c not in df.columns]
    if missing: raise SystemExit(f"Missing columns: {missing}")
    cols = {c: boolean_col(df[c]) for c in required}
    if any(v.isna().any() for v in cols.values()): raise SystemExit("Boolean columns contain unrecognized values.")
    if len(df) < 100: raise SystemExit(f"Need n >= 100 labeled rows; got {len(df)}.")
    for label, exp, act, metric in [
        ("abstain_recall", cols["expected_abstain"].to_numpy(), cols["actual_abstain"].to_numpy(), recall),
        ("refusal_recall", cols["expected_refusal"].to_numpy(), cols["actual_refusal"].to_numpy(), recall),
        ("refusal_precision", cols["expected_refusal"].to_numpy(), cols["actual_refusal"].to_numpy(), precision),
    ]:
        point = metric(exp, act)
        lo, hi = bootstrap_ci(metric, exp, act, args.iterations)
        print(f"{label}: {point:.4f} (bootstrap 95% CI {lo:.4f}, {hi:.4f})")
    print(f"n={len(df)}; bootstrap iterations={args.iterations}")

if __name__ == "__main__":
    main()
