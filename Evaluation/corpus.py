from dataclasses import dataclass
from typing import Dict, List
from collections import defaultdict

@dataclass
class CorpusIndex:
    candidate_titles: List[str]
    title2idx: Dict[str, int]
    cat2idxs: Dict[str, List[int]]

class CorpusBuilder:
    @staticmethod
    def flatten_2024_corpus(records: List[dict]) -> List[dict]:
        flat = []
        for rec in records:
            authors = rec.get("author_group", []) or []
            for cp in rec.get("current_papers", []) or []:
                title = cp.get("title")
                if not title:
                    continue
                flat.append({
                    "title": title,
                    "authors": authors,
                    "categories": cp.get("categories", []) or []
                })
        return flat

    @staticmethod
    def build_index(papers_2024: List[dict]) -> CorpusIndex:
        candidate_titles = [p["title"] for p in papers_2024]
        title2idx = {t: i for i, t in enumerate(candidate_titles)}

        cat2idxs = defaultdict(list)
        for i, p in enumerate(papers_2024):
            for c in (p.get("categories", []) or []):
                cat2idxs[c].append(i)

        return CorpusIndex(candidate_titles, title2idx, dict(cat2idxs))

    @staticmethod
    def allowed_categories_from_true_title(true_title: str, idx: CorpusIndex, papers_2024: List[dict]) -> List[str]:
        j = idx.title2idx.get(true_title)
        return [] if j is None else (papers_2024[j].get("categories", []) or [])

    @staticmethod
    def pool_idxs_from_categories(allowed_categories: List[str], cat2idxs: Dict[str, List[int]]) -> List[int]:
        s = set()
        for c in allowed_categories:
            s.update(cat2idxs.get(c, []))
        return sorted(s)
