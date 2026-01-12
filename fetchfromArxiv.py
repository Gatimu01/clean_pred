# arxiv_corpus.py
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import feedparser


JsonDict = Dict[str, Any]


@dataclass
class ArxivFetchConfig:
    """
    Configuration for fetching arXiv papers via the arXiv Atom API.

    Attributes
    ----------
    max_results : int
        Max records per API call. arXiv caps at 2000; keep <= 2000.
    interval_days : int
        Month is split into intervals to avoid hitting the 2000-result cap per query.
    sleep_s : float
        Sleep between requests to be polite and reduce 429s.
    base_url : str
        arXiv API endpoint.
    """
    max_results: int = 2000
    interval_days: int = 7
    sleep_s: float = 3.0
    base_url: str = "http://export.arxiv.org/api/query?"


class ArxivCorpusFetcher:
    """
    Fetch arXiv papers for a given month using the arXiv Atom API (export.arxiv.org).

    The arXiv API may return at most ~2000 results for a given query window.
    To reliably fetch all papers for a month, this class:
      1) splits the month into smaller date intervals (e.g., 7-day windows)
      2) paginates within each interval using `start` offsets
      3) deduplicates by arXiv id
      4) optionally streams output to a JSONL file (recommended for large downloads)

    Output JSON schema (per paper)
    ------------------------------
    {
      "id": "<entry URL>",
      "arxiv_id": "YYMM.NNNNN",
      "title": "...",
      "authors": [{"name": "...", "id": "...", "link": "..."}],
      "published": "<ISO datetime string>",
      "abstract": "...",
      "categories": ["cs.LG", ...]
    }
    """

    ARXIV_ID_RE = re.compile(r"arxiv\.org/abs/([^/]+)")
    AUTHOR_ID_RE = re.compile(r"/a/([^/]+)")

    def __init__(self, config: Optional[ArxivFetchConfig] = None):
        self.config = config or ArxivFetchConfig()

    # -------------------------
    # Parsing helpers
    # -------------------------
    @staticmethod
    def _safe_strip(x: Any) -> str:
        return (x or "").strip() if isinstance(x, str) else ""

    def extract_arxiv_id(self, entry_id: str) -> str:
        """
        Extract arXiv id from entry.id URL. Falls back to raw entry_id if no match.
        """
        m = self.ARXIV_ID_RE.search(entry_id or "")
        return m.group(1) if m else (entry_id or "")

    def extract_author_id(self, author_obj: Any) -> Optional[str]:
        """
        Extract arXiv author ID if present (from arxiv_author.identifier).
        Returns None if not present.
        """
        try:
            arxiv_author = getattr(author_obj, "arxiv_author", None)
            identifier = getattr(arxiv_author, "identifier", None)
            if identifier:
                m = self.AUTHOR_ID_RE.search(identifier)
                if m:
                    return m.group(1)
        except Exception:
            pass
        return None

    def _author_link(self, name: str, author_id: Optional[str]) -> str:
        if author_id:
            return f"https://arxiv.org/a/{author_id}.html"
        # fallback to search query
        return f"https://arxiv.org/search/?searchtype=author&query={name.replace(' ', '+')}"

    # -------------------------
    # Core fetch logic
    # -------------------------
    def fetch_range_page(self, start_date: str, end_date: str, *, start: int = 0) -> List[JsonDict]:
        """
        Fetch one page of results for a submittedDate range.

        Parameters
        ----------
        start_date, end_date : str
            arXiv date range in the format YYYYMMDDHHMM.
            Example: 202301010000
        start : int
            Pagination offset (0, 2000, 4000, ...)

        Returns
        -------
        list of dict
            Parsed papers for that page (may be empty).
        """
        query = f"submittedDate:[{start_date} TO {end_date}]"
        url = (
            f"{self.config.base_url}search_query={quote(query)}"
            f"&start={start}&max_results={self.config.max_results}"
            f"&sortBy=submittedDate&sortOrder=ascending"
        )

        feed = feedparser.parse(url)
        if not getattr(feed, "entries", None):
            return []

        batch: List[JsonDict] = []
        for entry in feed.entries:
            authors = []
            for a in getattr(entry, "authors", []) or []:
                aid = self.extract_author_id(a)
                name = getattr(a, "name", "") or ""
                authors.append({
                    "name": name,
                    "id": aid,
                    "link": self._author_link(name, aid),
                })

            batch.append({
                "id": getattr(entry, "id", ""),
                "arxiv_id": self.extract_arxiv_id(getattr(entry, "id", "")),
                "title": self._safe_strip(getattr(entry, "title", "")),
                "authors": authors,
                "published": getattr(entry, "published", ""),
                "abstract": getattr(entry, "summary", ""),
                "categories": [t.term for t in getattr(entry, "tags", [])] if hasattr(entry, "tags") else [],
            })

        return batch

    @staticmethod
    def split_month(year: int, month: int, interval_days: int) -> List[Tuple[str, str]]:
        """
        Split a month into [start,end) intervals expressed as YYYYMMDDHHMM.

        interval_days determines the step size (e.g., 7 days).
        """
        if interval_days <= 0:
            raise ValueError("interval_days must be > 0")

        intervals: List[Tuple[str, str]] = []
        cur = datetime(year, month, 1)
        nxt = datetime(year + 1, 1, 1) if month == 12 else datetime(year, month + 1, 1)

        while cur < nxt:
            end = min(cur + timedelta(days=interval_days), nxt)
            intervals.append((cur.strftime("%Y%m%d0000"), end.strftime("%Y%m%d0000")))
            cur = end

        return intervals

    @staticmethod
    def _append_jsonl(path: str, items: List[JsonDict]) -> None:
        if not items:
            return
        with open(path, "a", encoding="utf-8") as f:
            for obj in items:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def fetch_month(
        self,
        year: int,
        month: int,
        *,
        save_file: Optional[str] = None,
        return_list: bool = True,
        verbose: bool = True,
    ) -> List[JsonDict]:
        """
        Fetch all papers for a month.

        Parameters
        ----------
        year, month : int
            Month to fetch.
        save_file : str | None
            If provided, stream new papers to this JSONL file as they are fetched.
        return_list : bool
            If True, return the list of all unique papers. If False, return [] (useful when streaming to disk).
        verbose : bool
            Print progress information.

        Returns
        -------
        list of dict
            The unique papers for that month (unless return_list=False).
        """
        seen: set[str] = set()
        all_papers: List[JsonDict] = []

        intervals = self.split_month(year, month, self.config.interval_days)
        if verbose:
            print(f"📅 {year}-{month:02d} split into {len(intervals)} intervals")

        for (start_date, end_date) in intervals:
            if verbose:
                print(f"  Interval {start_date} → {end_date}")

            start_idx = 0
            while True:
                page = self.fetch_range_page(start_date, end_date, start=start_idx)
                if not page:
                    break

                new_items: List[JsonDict] = []
                for p in page:
                    aid = p.get("arxiv_id")
                    if aid and aid not in seen:
                        seen.add(aid)
                        new_items.append(p)

                if save_file:
                    self._append_jsonl(save_file, new_items)

                if return_list:
                    all_papers.extend(new_items)

                if verbose:
                    total = len(all_papers) if return_list else len(seen)
                    print(f"    fetched={len(page)} new={len(new_items)} total={total}")

                # If less than max_results, this interval is exhausted
                if len(page) < self.config.max_results:
                    break

                start_idx += len(page)
                time.sleep(self.config.sleep_s)

            time.sleep(self.config.sleep_s)

        if verbose:
            total = len(all_papers) if return_list else len(seen)
            print(f"🎯 Total unique papers for {year}-{month:02d}: {total}")

        return all_papers if return_list else []
