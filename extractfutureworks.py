#!/usr/bin/env python3
"""


End-to-end pipeline for:
1) Loading JSONL shards (e.g., 2023/*.jsonl)
2) Extracting "future work / discussion / conclusion" text from arXiv sources (LaTeX preferred)
3) Attaching extracted future-work text to the latest-year papers in each record
"""


from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import re
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple


# -----------------------------
# Logging
# -----------------------------
def setup_logger(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def format_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}h:{m:02d}m:{s:02d}s"


# -----------------------------
# I/O helpers
# -----------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: Any, path: str, indent: int = 2) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


def read_jsonl(path: str, limit: Optional[int] = None) -> List[dict]:
    """
    Read a JSONL file safely, skipping empty/malformed lines.
    Also collapses accidental list objects: if a line is a JSON list, take first element.
    """
    records: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            if limit is not None and len(records) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, list):
                    obj = obj[0] if obj else None
                if isinstance(obj, dict):
                    records.append(obj)
            except json.JSONDecodeError as e:
                logging.warning("[WARN] %s line %d: %s", path, line_num, e)
                continue
    return records


def append_jsonl(records: Iterable[dict], path: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()


def load_done_ids_jsonl(path: str, key: str) -> set:
    """
    Read JSONL file and collect values of `key` to support resume.
    """
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if key in obj:
                    done.add(obj[key])
            except Exception:
                continue
    return done


# -----------------------------
# Stage 1: load multiple JSONL shards
# -----------------------------
def load_folder_jsonl(data_folder: str, pattern: str = "*.jsonl", limit_per_file: Optional[int] = None) -> List[dict]:
    files = sorted(glob.glob(os.path.join(data_folder, pattern)))
    logging.info("Found %d files in %s", len(files), data_folder)
    out: List[dict] = []
    for fp in files:
        recs = read_jsonl(fp, limit=limit_per_file)
        out.extend(recs)
        logging.info("Loaded %d from %s (total=%d)", len(recs), os.path.basename(fp), len(out))
    return out


# -----------------------------
# Stage 2: arXiv future-work extraction (LaTeX first; optional PDF fallback)
# -----------------------------
FUTURE_SECTION_TOKENS = [
    "future", "outlook", "open", "discussion", "conclusion",
    "concluding", "summary", "limitations", "remark"
]

EXCLUDED_SECTION_TOKENS = [
    "final copy", "camera ready", "copyright", "ieee", "cvpr",
    "paper id", "author guidelines", "latex guidelines", "response",
    "rebuttal", "appendix", "supplementary"
]

SECTION_BLOCK_RE = re.compile(
    r"\\section\*?\{(?P<title>[^}]+)\}(?P<body>[\s\S]*?)(?=\\section|\Z)",
    flags=re.IGNORECASE
)

ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?")


def normalize_arxiv_id(raw: Any) -> Optional[str]:
    """
    Extract clean arXiv ID (no version) from:
      - plain id: '2301.00309v2'
      - URL: 'http://arxiv.org/abs/2301.00309v1'
      - list/dict wrappers
    """
    if raw is None:
        return None
    if isinstance(raw, list):
        raw = raw[0] if raw else None
    if isinstance(raw, dict):
        raw = raw.get("id") or raw.get("value")
    if not isinstance(raw, str):
        return None
    m = ARXIV_ID_RE.search(raw)
    return m.group(1) if m else None


def truncate_at_end_document(tex: str) -> str:
    idx = tex.lower().find("\\end{document}")
    return tex[:idx] if idx != -1 else tex


def load_latex_from_tar(tar_path: str) -> Optional[str]:
    """
    Concatenate all .tex files found in tarball.
    (Simple + robust; for better speed you can score and pick main .tex.)
    """
    try:
        texts = []
        with tarfile.open(tar_path) as tar:
            for m in tar.getmembers():
                if m.name.lower().endswith(".tex"):
                    f = tar.extractfile(m)
                    if f:
                        texts.append(f.read().decode("utf-8", errors="ignore"))
        return "\n".join(texts) if texts else None
    except Exception:
        return None


def extract_future_sections_from_latex(tex: str) -> List[dict]:
    tex = truncate_at_end_document(tex)
    sections: List[dict] = []

    for m in SECTION_BLOCK_RE.finditer(tex):
        title = m.group("title").strip()
        body = m.group("body").strip()
        title_l = title.lower()

        if any(bad in title_l for bad in EXCLUDED_SECTION_TOKENS):
            continue
        if any(tok in title_l for tok in FUTURE_SECTION_TOKENS):
            sections.append({"title": title, "text": body})

    return sections


def extract_pdf_tail_text(pdf_path: str, max_pages: int = 3) -> Optional[str]:
    """
    PDF fallback (optional). Reads last `max_pages` pages.
    Requires PyMuPDF: pip install pymupdf
    """
    try:
        import fitz  # type: ignore
    except Exception:
        logging.warning("PyMuPDF not installed; skipping PDF fallback.")
        return None

    if not os.path.exists(pdf_path):
        return None
    try:
        doc = fitz.open(pdf_path)
        pages = doc[-max_pages:] if len(doc) > max_pages else doc
        return "\n".join(p.get_text() for p in pages)
    except Exception:
        return None


@dataclass
class ArxivExtractorConfig:
    out_dir: str = "arxiv_sources"
    page_size: int = 100
    delay_seconds: int = 3
    pdf_fallback: bool = False  # set True only if you want PDF fallback


def download_source_and_pdf(arxiv_id: str, cfg: ArxivExtractorConfig) -> Tuple[Optional[str], Optional[str]]:
    """
    Downloads source tarball + pdf for an arXiv id.
    Requires: pip install arxiv
    """
    try:
        import arxiv  # type: ignore
    except Exception as e:
        raise RuntimeError("Missing dependency `arxiv`. Install: pip install arxiv") from e

    ensure_dir(cfg.out_dir)
    client = arxiv.Client(page_size=cfg.page_size, delay_seconds=cfg.delay_seconds)

    try:
        paper = next(client.results(arxiv.Search(id_list=[arxiv_id])))
        src_path = paper.download_source(dirpath=cfg.out_dir)
        pdf_path = None
        if cfg.pdf_fallback:
            pdf_path = paper.download_pdf(dirpath=cfg.out_dir, filename=f"{arxiv_id}.pdf")
        return src_path, pdf_path
    except Exception as e:
        logging.error("[download] %s: %s", arxiv_id, e)
        return None, None


def process_paper_future_work(arxiv_id: str, cfg: ArxivExtractorConfig) -> dict:
    src_path, pdf_path = download_source_and_pdf(arxiv_id, cfg)

    if not src_path and not pdf_path:
        return {"paper_id": arxiv_id, "status": "download_failed"}

    # 1) LaTeX first
    if src_path:
        tex = load_latex_from_tar(src_path)
        if tex:
            sections = extract_future_sections_from_latex(tex)
            if sections:
                return {
                    "paper_id": arxiv_id,
                    "status": "ok",
                    "source": "latex_section",
                    "num_sections": len(sections),
                    "future_sections": sections,
                }
            return {"paper_id": arxiv_id, "status": "no_future_section", "source": "latex_section"}

    # 2) PDF fallback only if enabled and no LaTeX
    if cfg.pdf_fallback and pdf_path:
        txt = extract_pdf_tail_text(pdf_path)
        if txt:
            return {"paper_id": arxiv_id, "status": "ok", "source": "pdf_tail", "future_text": txt}

    return {"paper_id": arxiv_id, "status": "no_latex"}


def run_future_work_batch(
    arxiv_ids: List[str],
    out_jsonl: str,
    workers: int = 16,
    cfg: Optional[ArxivExtractorConfig] = None,
) -> None:
    cfg = cfg or ArxivExtractorConfig()
    done = load_done_ids_jsonl(out_jsonl, key="paper_id")
    to_run = [aid for aid in arxiv_ids if aid not in done]

    logging.info("Future-work extraction: %d total | %d already done | %d to process",
                 len(arxiv_ids), len(done), len(to_run))
    if not to_run:
        return

    ensure_dir(os.path.dirname(out_jsonl) or ".")

    start = time.time()
    completed = 0

    with open(out_jsonl, "a", encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(process_paper_future_work, aid, cfg): aid for aid in to_run}
            for fut in as_completed(futures):
                rec = fut.result()
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()

                completed += 1
                if completed % 25 == 0:
                    elapsed = time.time() - start
                    rate = completed / max(elapsed, 1e-9)
                    logging.info("Progress: %d/%d | rate=%.2f/s | elapsed=%s",
                                 completed, len(to_run), rate, format_time(elapsed))


# -----------------------------
# Stage 3: attach + merge future-work to max-year papers
# -----------------------------
def get_latest_year_papers(record: dict) -> Dict[str, dict]:
    """
    Return dict of papers in record['past_papers'] where year == max_year.
    Expects record['past_papers'] to be dict-like: {paper_key: {...}}
    """
    past = record.get("past_papers", {})
    if not isinstance(past, dict):
        return {}

    years = [p.get("year") for p in past.values() if isinstance(p.get("year"), int)]
    if not years:
        return {}

    max_year = max(years)
    return {k: p for k, p in past.items() if p.get("year") == max_year}


def parse_future_jsonl(future_jsonl: str) -> Dict[str, dict]:
    """
    Loads future-work jsonl results keyed by paper_id.
    """
    future_map: Dict[str, dict] = {}
    if not os.path.exists(future_jsonl):
        return future_map

    with open(future_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                pid = obj.get("paper_id") or obj.get("arxiv_id")
                if pid:
                    future_map[pid] = obj
            except Exception:
                continue
    return future_map


def extract_future_text(future_rec: dict) -> Optional[str]:
    """
    Normalize different output shapes into one string.
    """
    if not future_rec:
        return None

    # latex sections
    if future_rec.get("future_sections"):
        secs = future_rec["future_sections"]
        texts = [s.get("text", "") for s in secs if isinstance(s, dict)]
        joined = "\n\n".join(t.strip() for t in texts if t and t.strip()).strip()
        return joined or None

    # pdf tail fallback
    if future_rec.get("future_text"):
        t = str(future_rec.get("future_text", "")).strip()
        return t or None

    return None


def merge_future_work_max_year_strict(records: List[dict], future_jsonl: str) -> List[dict]:
    """
    Keep a record ONLY if all max-year past papers have a future-work entry available.
    Attach future_work_text/source/status to max-year papers only.
    """
    future_map = parse_future_jsonl(future_jsonl)

    updated: List[dict] = []
    dropped = 0

    for record in records:
        past = record.get("past_papers", {})
        if not isinstance(past, dict) or not past:
            dropped += 1
            continue

        latest = get_latest_year_papers(record)
        if not latest:
            dropped += 1
            continue

        # validate all max-year papers exist in future_map
        ok = True
        for p in latest.values():
            aid = normalize_arxiv_id(p.get("arxiv_id"))
            if not aid or aid not in future_map:
                ok = False
                break
        if not ok:
            dropped += 1
            continue

        # attach
        new_record = dict(record)
        new_past = {}
        for k, p in past.items():
            p2 = dict(p)
            if p2.get("year") == latest[next(iter(latest))].get("year"):
                aid = normalize_arxiv_id(p2.get("arxiv_id"))
                fut = future_map.get(aid, {})
                p2["future_work_text"] = extract_future_text(fut)
                p2["future_work_source"] = fut.get("source")
                p2["future_work_status"] = fut.get("status")
            new_past[k] = p2

        new_record["past_papers"] = new_past
        updated.append(new_record)

    logging.info("merge_future_work_max_year_strict: kept=%d | dropped=%d", len(updated), dropped)
    return updated


