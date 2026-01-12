import json, os, re
from typing import List, Optional

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_jsonl(path: str, limit: Optional[int] = None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out

def clean_titles_flat(titles: List[str]) -> List[str]:
    cleaned = []
    for t in titles:
        if not isinstance(t, str) or not t.strip():
            cleaned.append("")
            continue
        t = " ".join(t.strip().replace("\n", " ").replace("\t", " ").split())
        t = re.sub(r"[^A-Za-z0-9,.:;\-()\[\]\s]", "", t)
        cleaned.append(t)
    return cleaned

def filter_pairs(refs: List[str], preds: List[str]):
    out_r, out_p = [], []
    for r, p in zip(refs, preds):
        if isinstance(r, str) and isinstance(p, str) and r.strip() and p.strip():
            out_r.append(r.strip())
            out_p.append(p.strip())
    return out_r, out_p
