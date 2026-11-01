import argparse
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", default="Dataset/indian_reports_corpus.jsonl/corpus.jsonl")
ap.add_argument("--n", type=int, default=120)
ap.add_argument("--writer-provider", default="groq")
ap.add_argument("--writer-model", default="qwen/qwen3-32b")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--sleep", type=float, default=0.5)
ap.add_argument("--out", default="data/indian_questions.jsonl")
ap.add_argument("--queries", default="Dataset/indian_reports_queries.jsonl/queries.jsonl")
ap.add_argument("--qrels", default="Dataset/IndianReports_qrels.tsv")
a = ap.parse_args()

PROMPT = """Passage from an Indian company's annual report ({title}):
\"\"\"{text}\"\"\"

Write ONE question a retail investor could ask whose answer is a specific figure or fact stated in this passage.
Rules: name the company and fiscal year explicitly in the question; never mention "the passage" or "above"; the question must be answerable from this passage alone; the answer must include units (e.g. ₹ crore, %) exactly as the passage states them.
If the passage has no clear standalone fact, return {{"skip": true}}.
Return only JSON: {{"question": "...", "answer": "..."}}"""


def nums(s):
    return {x.replace(",", "") for x in re.findall(r"\d[\d,]*(?:\.\d+)?", s)}


def parse(raw):
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S)
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except Exception:
        return None


def valid(d, text):
    if not d or d.get("skip"):
        return False
    q, ans = str(d.get("question", "")).strip(), str(d.get("answer", "")).strip()
    if len(q) < 25 or not ans or re.search(r"\b(passage|above|following text)\b", q, re.I):
        return False
    n = nums(ans)
    return bool(n) and n <= nums(text)


def main():
    from finrag.judge import _text
    from finrag.models import get_provider_llm
    llm = get_provider_llm(a.writer_provider, a.writer_model)
    corpus = [json.loads(l) for l in open(a.corpus, encoding="utf-8") if l.strip()]
    by_co = defaultdict(list)
    for d in corpus:
        if len(d["text"]) >= 400 and len(nums(d["text"])) >= 6:
            by_co[d["_id"].split("_")[0]].append(d)
    rng = random.Random(a.seed)
    for v in by_co.values():
        rng.shuffle(v)
    order, i = [], 0
    while any(i < len(v) for v in by_co.values()):
        order += [v[i] for v in by_co.values() if i < len(v)]
        i += 1
    out, seen = [], set()
    for d in order:
        if len(out) >= a.n:
            break
        try:
            r = parse(_text(llm.invoke(PROMPT.format(title=d["title"], text=d["text"]))))
        except Exception:
            time.sleep(3)
            continue
        if valid(r, d["text"]) and r["question"].lower() not in seen:
            seen.add(r["question"].lower())
            out.append({"_id": f"in{len(out)+1:03d}", "question": r["question"].strip(), "gold_answer": str(r["answer"]).strip(),
                        "company": d["_id"].split("_")[0], "source_id": d["_id"], "evidence": d["text"]})
            print(f"{len(out)}/{a.n}", end="\r")
        time.sleep(a.sleep)
    for p in (a.out, a.queries, a.qrels):
        Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(o) + "\n" for o in out)
    with open(a.queries, "w", encoding="utf-8") as f:
        f.writelines(json.dumps({"_id": o["_id"], "text": o["question"]}) + "\n" for o in out)
    with open(a.qrels, "w", encoding="utf-8") as f:
        f.write("query_id\tcorpus_id\tscore\n")
        f.writelines(f"{o['_id']}\t{o['source_id']}\t1\n" for o in out)
    print(f"\nwrote {len(out)} questions | {a.out} | {a.queries} | {a.qrels}")


if __name__ == "__main__":
    main()
