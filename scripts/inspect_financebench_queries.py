"""Check whether the local FinanceBench queries JSONL contains reference answers."""
import json
from pathlib import Path

path = Path("Dataset/financebench_queries.jsonl/queries.jsonl")
if not path.exists():
    raise SystemExit(f"Not found: {path}. Update this path to your local queries.jsonl.")
with path.open(encoding="utf-8") as f:
    rows = [json.loads(line) for line in f if line.strip()]
print(f"Rows: {len(rows)}")
print("First row keys:", sorted(rows[0]) if rows else [])
candidates = {"answer", "answers", "reference_answer", "gold_answer", "ground_truth", "response"}
for key in sorted(candidates):
    present = sum(key in row and row[key] not in (None, "", []) for row in rows)
    if present:
        print(f"{key}: present in {present}/{len(rows)} rows")
if not any(any(k in row and row[k] not in (None, "", []) for k in candidates) for row in rows):
    print("No obvious reference-answer field found. Inspect dataset documentation before assuming answers are absent.")
