import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from finrag.metrics import bootstrap_stat, false_positive_rate, precision, recall


def to_bool(s):
    return s.astype(str).str.strip().str.lower().map({"true": True, "1": True, "false": False, "0": False})


def final_abstain(df, th):
    low = df["top_score"].isna() | (df["top_score"] < th)
    return (df["refused"] | (~df["refused"] & low) | df["llm_abstain"]).to_numpy()


def pick_threshold(df):
    cands = [-np.inf] + sorted(df["top_score"].dropna().unique().tolist())
    exp = df["expected_abstain"].to_numpy()
    best, bj = cands[0], -9
    for th in cands:
        act = final_abstain(df, th)
        j = recall(exp, act) - false_positive_rate(exp, act)
        if j > bj + 1e-12:
            best, bj = th, j
    return best


def report(df, th, iters, label):
    exp_a, act_a = df["expected_abstain"].to_numpy(), final_abstain(df, th)
    exp_r, act_r = df["expected_refusal"].to_numpy(), df["refused"].to_numpy()
    print(f"\n[{label}] n={len(df)} threshold={th:.3f}")
    for name, fn, e, x in [
        ("abstain_recall", recall, exp_a, act_a),
        ("false_abstain_rate", false_positive_rate, exp_a, act_a),
        ("refusal_recall", recall, exp_r, act_r),
        ("refusal_precision", precision, exp_r, act_r),
    ]:
        p, lo, hi = bootstrap_stat(fn, e, x, iters=iters)
        print(f"  {name}: {p:.3f} (95% CI {lo:.3f}, {hi:.3f})")
    print("  layer ablation (abstain_recall / false_abstain):")
    for name, act in [
        ("regex only", df["refused"].to_numpy()),
        ("+ rerank threshold", (df["refused"] | df["top_score"].isna() | (df["top_score"] < th)).to_numpy()),
        ("+ LLM/numeric abstain", act_a),
    ]:
        print(f"    {name}: {recall(exp_a, act):.3f} / {false_positive_rate(exp_a, act):.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--dataset", default="financebench")
    ap.add_argument("--provider", default="gemini")
    ap.add_argument("--model", default=None)
    ap.add_argument("--no-multiquery", action="store_true")
    ap.add_argument("--calib-frac", type=float, default=0.5)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/guardrail")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(a.csv)
    if df["question"].isna().any():
        raise SystemExit("Blank questions; run scripts/build_eval_set.py first.")
    df["expected_abstain"] = to_bool(df["expected_abstain"])
    df["expected_refusal"] = to_bool(df["expected_refusal"])
    # if len(df) < 100:
    #     raise SystemExit(f"n={len(df)} < 100")

    raw = out / "raw.csv"
    if raw.exists():
        r = pd.read_csv(raw)
    else:
        from finrag.engine import Engine
        eng = Engine(a.dataset, a.provider, a.model)
        recs = []
        for _, row in tqdm(df.iterrows(), total=len(df)):
            x = eng.answer(row["question"], use_multiquery=not a.no_multiquery, threshold=float("-inf"))
            recs.append({"id": row["id"], "refused": x.refused, "top_score": x.top_score,
                         "llm_abstain": x.abstained and not x.refused, "reason": x.reason, "answer": x.answer})
        r = pd.DataFrame(recs)
        r.to_csv(raw, index=False)
    df = df.drop(columns=[c for c in ("refused", "top_score", "llm_abstain", "reason", "answer") if c in df]).merge(r, on="id")
    for c in ("refused", "llm_abstain"):
        df[c] = df[c].astype(bool)

    rng = np.random.default_rng(a.seed)
    calib_idx = []
    for _, g in df.groupby("expected_abstain"):
        idx = rng.permutation(g.index.to_numpy())
        calib_idx += idx[: int(len(idx) * a.calib_frac)].tolist()
    calib, test = df.loc[calib_idx], df.drop(index=calib_idx)
    th = pick_threshold(calib)
    print(f"Chosen FINRAG_MIN_RERANK_SCORE={th:.4f} (picked on calibration split n={len(calib)})")
    report(test, th, a.iters, "HELD-OUT TEST SPLIT")
    report(df, th, a.iters, "FULL SET (optimistic)")

    df["actual_abstain"] = final_abstain(df, th)
    df["actual_refusal"] = df["refused"]
    df.to_csv(out / "predictions.csv", index=False)
