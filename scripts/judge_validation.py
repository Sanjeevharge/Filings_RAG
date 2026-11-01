import argparse

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from finrag.metrics import bootstrap_stat

ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd", required=True)
for name in ("export", "auto"):
    p = sub.add_parser(name)
    p.add_argument("--gold-eval", default="results/gold_eval.csv")
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default="results/judge_labels.csv")
    if name == "auto":
        p.add_argument("--labeler-provider", default="groq")
        p.add_argument("--labeler-model", default="openai/gpt-oss-120b")
        p.add_argument("--judge-provider", default="groq")
        p.add_argument("--judge-model", default="llama-3.3-70b-versatile")
        p.add_argument("--gen-provider", default="gemini")
        p.add_argument("--gen-model", default="gemini-3.6-flash")
s = sub.add_parser("score")
s.add_argument("--gold-eval", default="results/gold_eval.csv")
s.add_argument("--labels", default="results/judge_labels.csv")
s.add_argument("--iters", type=int, default=2000)
a = ap.parse_args()


def sample(df, n, seed):
    df = df[df["verdict"].isin(["correct", "incorrect"])]
    parts = [g.sample(min(len(g), n // 2), random_state=seed) for _, g in df.groupby("verdict")]
    smp = pd.concat(parts)
    if len(smp) < n:
        smp = pd.concat([smp, df.drop(smp.index).sample(min(n - len(smp), len(df) - len(smp)), random_state=seed)])
    return smp


if a.cmd == "export":
    smp = sample(pd.read_csv(a.gold_eval), a.n, a.seed)
    smp[["_id", "question", "gold_answer", "answer"]].assign(reference_verdict="", label_source="human").to_csv(a.out, index=False)
    print(f"wrote {a.out}")
elif a.cmd == "auto":
    from finrag.judge import Judge
    if (a.labeler_provider, a.labeler_model) == (a.gen_provider, a.gen_model):
        raise SystemExit("Labeler must differ from generator.")
    lab = Judge(a.labeler_provider, a.labeler_model, a.judge_provider, a.judge_model)
    df = pd.read_csv(a.gold_eval)
    if "evidence" not in df:
        df["evidence"] = ""
    smp = sample(df, a.n, a.seed).copy()
    smp["reference_verdict"] = [
        lab.grade(r.question, r.gold_answer, r.answer, evidence=(r.evidence if isinstance(r.evidence, str) else ""))[0]
        for r in smp.itertuples()
    ]
    smp["label_source"] = f"LLM:{lab.name}"
    smp[["_id", "question", "gold_answer", "answer", "reference_verdict", "label_source"]].to_csv(a.out, index=False)
    print(f"wrote {a.out} ({len(smp)} rows, labeler={lab.name}; NOT human labels)")
else:
    g = pd.read_csv(a.gold_eval)[["_id", "verdict"]]
    h = pd.read_csv(a.labels)
    col = "reference_verdict" if "reference_verdict" in h else "human_verdict"
    m = h.merge(g, on="_id")
    m[col] = m[col].astype(str).str.strip().str.lower()
    m = m[m[col].isin(["correct", "incorrect"])]
    src = h["label_source"].iloc[0] if "label_source" in h and len(h) else "human"
    rv, jv = (m[col] == "correct").to_numpy(), (m["verdict"] == "correct").to_numpy()
    p, lo, hi = bootstrap_stat(lambda x, y: float((x == y).mean()), rv, jv, iters=a.iters)
    print(f"labels from: {src}  n={len(m)}")
    print(f"agreement={p:.3f} (95% CI {lo:.3f}, {hi:.3f})  cohen_kappa={cohen_kappa_score(rv, jv):.3f}")
    print(pd.crosstab(m[col], m["verdict"], rownames=["reference"], colnames=["judge"]))
