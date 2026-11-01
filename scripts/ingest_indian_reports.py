import argparse
import json
import re
from pathlib import Path

import fitz

DROP = re.compile(r"proxy form|attendance slip|route map|business responsibility and sustainability|form no\. ?mgt-11|notice of the .{0,30}annual general meeting", re.I)

ap = argparse.ArgumentParser()
ap.add_argument("pdf_dir", help="PDFs named COMPANY_FY2024.pdf")
ap.add_argument("--out", default="Dataset/indian_reports_corpus.jsonl/corpus.jsonl")
ap.add_argument("--max-chars", type=int, default=1400)
ap.add_argument("--keep-all", action="store_true")
a = ap.parse_args()

out = Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)
n = 0
with out.open("w", encoding="utf-8") as f:
    for pdf in sorted(Path(a.pdf_dir).glob("*.pdf")):
        company, _, fy = pdf.stem.partition("_")
        for pno, page in enumerate(fitz.open(pdf), 1):
            text = page.get_text("text")
            if len(text.strip()) < 200 or (not a.keep_all and DROP.search(text[:600])):
                continue
            paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
            buf, i = "", 0
            for p in paras + [None]:
                if p is None or (buf and len(buf) + len(p) > a.max_chars):
                    if buf:
                        f.write(json.dumps({"_id": f"{pdf.stem}_p{pno}_{i}",
                                            "title": f"{company} {fy} annual report p.{pno}",
                                            "text": buf}) + "\n")
                        n += 1
                        i += 1
                    buf = ""
                if p is not None:
                    while len(p) > a.max_chars:
                        f.write(json.dumps({"_id": f"{pdf.stem}_p{pno}_{i}",
                                            "title": f"{company} {fy} annual report p.{pno}",
                                            "text": p[: a.max_chars]}) + "\n")
                        n += 1
                        i += 1
                        p = p[a.max_chars:]
                    buf = (buf + "\n" + p).strip()
print(f"wrote {n} passages -> {out}")
