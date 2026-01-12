# evaluation.py
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np

# Optional heavy deps: only imported when used
# from bert_score import score
# from sentence_transformers import SentenceTransformer, util
# from keybert import KeyBERT
# from rank_bm25 import BM25Okapi


# -----------------------------
# Utilities
# -----------------------------
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
    """Basic cleaning for titles; returns a flat list."""
    cleaned = []
    for t in titles:
        if not t or not isinstance(t, str):
            cleaned.append("")
            continue
        t = t.strip().replace("\n", " ").replace("\t", " ")
        t = " ".join(t.split())
        t = re.sub(r"[^A-Za-z0-9,.:;\-()\[\]\s]", "", t)
        cleaned.append(t)
    return cleaned


def filter_pairs(
    refs: List[str],
    preds: List[str],
) -> Tuple[List[str], List[str]]:
    """Keep only pairs where both are non-empty strings."""
    out_r, out_p = [], []
    for r, p in zip(refs, preds):
        if isinstance(r, str) and isinstance(p, str) and r.strip() and p.strip():
            out_r.append(r.strip())
            out_p.append(p.strip())
    return out_r, out_p


def hits_at_k(ranks: List[int], k: int) -> float:
    ranks = [r for r in ranks if r is not None]
    if not ranks:
        return 0.0
    return sum(r <= k for r in ranks) / len(ranks)


def mean_reciprocal_rank(ranks: List[int]) -> float:
    ranks = [r for r in ranks if r is not None]
    if not ranks:
        return 0.0
    return float(np.mean([1.0 / r for r in ranks]))


def tokenize_bm25(s: str) -> List[str]:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    return [w for w in s.split() if w]


# -----------------------------
# Data Structures
# -----------------------------
@dataclass
class CorpusIndex:
    """Index structures for fast category pooling and title lookup."""
    candidate_titles: List[str]
    title2idx: Dict[str, int]
    cat2idxs: Dict[str, List[int]]


# -----------------------------
# Evaluation Class
# -----------------------------
class TitleForecastEvaluator:
    """
    TitleForecastEvaluator runs evaluation for title forecasting experiments.

    Supported metrics/workflows:
    - BERTScore (SciBERT) on (pred, true) pairs
    - SPECTER cosine similarity on (pred, true) pairs
    - Retrieval-style ranking:
        * global pool (all candidate titles)
        * category-restricted pool (oracle categories from the true paper)
        * optional KeyBERT shrink within category pool (top-M)
        * optional BM25 rerank within the shrunk pool

    Typical usage:
        ev = TitleForecastEvaluator()
        papers_2024 = ev.flatten_2024_corpus(records)
        index = ev.build_corpus_index(papers_2024)
        ev.build_candidate_embeddings(index.candidate_titles)

        ranks = ev.rank_many(preds, trues, index, use_oracle_categories=True)
        print(ev.summary(ranks))

    Notes:
    - Oracle categories are for analysis/debug only (upper bound).
    - For large corpora, embedding all candidates can be memory-heavy.
    """

    def __init__(
        self,
        specter_model_name: str = "sentence-transformers/allenai-specter",
        bertscore_model_type: str = "allenai/scibert_scivocab_uncased",
        device: Optional[str] = None,   # e.g., "cpu" or "cuda"
    ):
        self.specter_model_name = specter_model_name
        self.bertscore_model_type = bertscore_model_type
        self.device = device

        self._st_model = None          # SentenceTransformer
        self._E_cands = None           # np.ndarray [N, d], float32
        self._kw_model = None          # KeyBERT
        self._bm25_cache = {}          # optional cache per pool key

    # -----------------------------
    # Models
    # -----------------------------
    def _get_st(self):
        if self._st_model is None:
            from sentence_transformers import SentenceTransformer
            self._st_model = SentenceTransformer(self.specter_model_name)
        return self._st_model

    def enable_keybert(self, keybert_backbone: str = "allenai/scibert_scivocab_uncased"):
        """Optional: enable KeyBERT phrase extraction."""
        from keybert import KeyBERT
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer(keybert_backbone)
        self._kw_model = KeyBERT(model=embedder)

    # -----------------------------
    # Corpus prep
    # -----------------------------
    @staticmethod
    def flatten_2024_corpus(records: List[dict]) -> List[dict]:
        """
        records: list of dicts with keys: author_group, current_papers (list)
        returns: list of paper dicts: {"title":..., "authors":[...], "categories":[...]}
        """
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
    def build_corpus_index(papers_2024: List[dict]) -> CorpusIndex:
        candidate_titles = [p["title"] for p in papers_2024]
        title2idx = {t: i for i, t in enumerate(candidate_titles)}

        cat2idxs = defaultdict(list)
        for i, p in enumerate(papers_2024):
            for c in (p.get("categories", []) or []):
                cat2idxs[c].append(i)

        return CorpusIndex(
            candidate_titles=candidate_titles,
            title2idx=title2idx,
            cat2idxs=dict(cat2idxs),
        )

    # -----------------------------
    # Embeddings
    # -----------------------------
    def build_candidate_embeddings(self, candidate_titles: List[str], batch_size: int = 256):
        """
        Precompute embeddings for all candidate titles once.
        Stores in self._E_cands as float32 numpy array.
        """
        st = self._get_st()
        E = st.encode(
            candidate_titles,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=True,
        )
        self._E_cands = np.asarray(E, dtype=np.float32)

    def specter_pairwise_scores(self, pred_titles: List[str], true_titles: List[str]) -> List[float]:
        """
        Compute cosine similarity for each (pred_i, true_i) pair using SPECTER embeddings.
        """
        st = self._get_st()
        e_pred = st.encode(pred_titles, normalize_embeddings=True)
        e_true = st.encode(true_titles, normalize_embeddings=True)
        e_pred = np.asarray(e_pred, dtype=np.float32)
        e_true = np.asarray(e_true, dtype=np.float32)
        return np.sum(e_pred * e_true, axis=1).tolist()

    # -----------------------------
    # BERTScore
    # -----------------------------
    def bertscore(self, pred_titles: List[str], true_titles: List[str], rescale_with_baseline: bool = True):
        """
        Returns: (P_mean, R_mean, F1_mean)
        """
        from bert_score import score
        P, R, F1 = score(
            pred_titles,
            true_titles,
            lang="en",
            model_type=self.bertscore_model_type,
            rescale_with_baseline=rescale_with_baseline,
            verbose=True,
        )
        return float(P.mean()), float(R.mean()), float(F1.mean())

    # -----------------------------
    # Pooling helpers
    # -----------------------------
    @staticmethod
    def pool_idxs_from_categories(allowed_categories: List[str], cat2idxs: Dict[str, List[int]]) -> List[int]:
        idxs = set()
        for c in allowed_categories:
            idxs.update(cat2idxs.get(c, []))
        return sorted(idxs)

    @staticmethod
    def allowed_categories_from_true_title(true_title: str, idx: CorpusIndex, papers_2024: List[dict]) -> List[str]:
        j = idx.title2idx.get(true_title)
        if j is None:
            return []
        return papers_2024[j].get("categories", []) or []

    # -----------------------------
    # Ranking
    # -----------------------------
    def rank_in_pool_cached(
        self,
        pred_title: str,
        true_title: str,
        pool_idxs: List[int],
        idx: CorpusIndex,
    ) -> Optional[int]:
        """
        Rank of true_title in a given pool, using dot product with precomputed candidate embeddings.
        Requires self._E_cands to be built.
        """
        if self._E_cands is None:
            raise RuntimeError("Call build_candidate_embeddings(...) before ranking.")

        true_global = idx.title2idx.get(true_title)
        if true_global is None:
            return None

        pool_set = set(pool_idxs)
        if true_global not in pool_set:
            return None

        st = self._get_st()
        e_pred = st.encode(pred_title, normalize_embeddings=True)
        e_pred = np.asarray(e_pred, dtype=np.float32)

        scores = self._E_cands[pool_idxs] @ e_pred

        # position of true in pool
        pos_map = {g: j for j, g in enumerate(pool_idxs)}
        true_pos = pos_map[true_global]
        true_score = scores[true_pos]

        # exact rank without sorting all
        return int(1 + np.sum(scores > true_score))

    # -----------------------------
    # KeyBERT shrink (optional)
    # -----------------------------
    def extract_keyphrases(self, text: str, top_n: int = 8) -> List[str]:
        if self._kw_model is None:
            raise RuntimeError("Call enable_keybert(...) before using KeyBERT features.")
        kws = self._kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 3),
            stop_words="english",
            top_n=top_n,
        )
        return [k for k, _ in kws]

    def keybert_topM_within_pool(
        self,
        pred_title: str,
        pool_idxs: List[int],
        top_phrases: int = 8,
        M: int = 800,
    ) -> Tuple[List[int], List[str]]:
        """
        Shrink a pool by scoring candidates with an embedding of KeyBERT-derived query.
        Returns: (top_global_idxs, phrases_used)
        """
        if self._E_cands is None:
            raise RuntimeError("Call build_candidate_embeddings(...) before ranking.")
        phrases = self.extract_keyphrases(pred_title, top_n=top_phrases)
        query = " ".join(phrases) if phrases else pred_title

        st = self._get_st()
        e_q = st.encode(query, normalize_embeddings=True)
        e_q = np.asarray(e_q, dtype=np.float32)

        scores = self._E_cands[pool_idxs] @ e_q

        M = min(M, len(pool_idxs))
        top_local = np.argpartition(scores, -M)[-M:]
        top_local = top_local[np.argsort(scores[top_local])[::-1]]

        top_global = [pool_idxs[j] for j in top_local.tolist()]
        return top_global, phrases

    # -----------------------------
    # BM25 rerank (optional)
    # -----------------------------
    def bm25_rank_in_pool(
        self,
        query_title: str,
        true_title: str,
        pool_idxs: List[int],
        idx: CorpusIndex,
    ) -> Optional[int]:
        """
        BM25 rank within pool based on token overlap.
        """
        from rank_bm25 import BM25Okapi

        true_global = idx.title2idx.get(true_title)
        if true_global is None:
            return None
        pool_set = set(pool_idxs)
        if true_global not in pool_set:
            return None

        pool_titles = [idx.candidate_titles[i] for i in pool_idxs]
        tokenized_docs = [tokenize_bm25(t) for t in pool_titles]
        bm25 = BM25Okapi(tokenized_docs)

        q = tokenize_bm25(query_title)
        scores = np.asarray(bm25.get_scores(q), dtype=np.float32)

        true_pos = pool_idxs.index(true_global)
        true_score = scores[true_pos]
        return int(1 + np.sum(scores > true_score))

    # -----------------------------
    # High-level evaluation runners
    # -----------------------------
    def rank_many(
        self,
        pred_titles: List[str],
        true_titles: List[str],
        idx: CorpusIndex,
        papers_2024: Optional[List[dict]] = None,
        use_oracle_categories: bool = False,
        keybert_shrink: bool = False,
        keybert_M: int = 800,
        bm25_rerank: bool = False,
        top_phrases: int = 8,
    ) -> List[Optional[int]]:
        """
        Computes ranks for many pairs.

        Pools:
        - If use_oracle_categories=True: category pool derived from the true title's categories (requires papers_2024).
        - Else: global pool (all candidates).

        Optional:
        - keybert_shrink: shrink pool to top-M by KeyBERT query embedding
        - bm25_rerank: rerank inside the (optionally shrunk) pool by BM25
        """
        if use_oracle_categories and papers_2024 is None:
            raise ValueError("papers_2024 must be provided when use_oracle_categories=True")

        all_idxs = list(range(len(idx.candidate_titles)))
        ranks: List[Optional[int]] = []

        for pred, true in zip(pred_titles, true_titles):
            # pick pool
            if use_oracle_categories:
                allowed = self.allowed_categories_from_true_title(true, idx, papers_2024)
                pool = self.pool_idxs_from_categories(allowed, idx.cat2idxs)
                if len(pool) < 2:
                    pool = all_idxs
            else:
                pool = all_idxs

            # optional shrink
            if keybert_shrink:
                pool, _phrases = self.keybert_topM_within_pool(
                    pred, pool, top_phrases=top_phrases, M=keybert_M
                )

            # rank
            if bm25_rerank:
                r = self.bm25_rank_in_pool(pred, true, pool, idx)
            else:
                r = self.rank_in_pool_cached(pred, true, pool, idx)

            ranks.append(r)

        return ranks

    @staticmethod
    def summary(ranks: List[Optional[int]], ks: Tuple[int, ...] = (10, 25, 50, 100, 200, 500)) -> Dict[str, float]:
        rr = [r for r in ranks if r is not None]
        out = {"n": float(len(rr))}
        for k in ks:
            out[f"hits@{k}"] = hits_at_k(rr, k)
        out["mrr"] = mean_reciprocal_rank(rr)
        out["median_rank"] = float(np.median(rr)) if rr else 0.0
        out["mean_rank"] = float(np.mean(rr)) if rr else 0.0
        return out


# -----------------------------
# Example CLI-like usage
# -----------------------------
if __name__ == "__main__":
    # Example wiring based on your notebook variables/files.
    # Adjust paths to your repo structure.
    records = load_jsonl("Data_to_run50000.jsonl", limit=10)

    # True titles list
    current_titles = records.get("current_titles", []   )
    current_titles = clean_titles_flat(current_titles)

    # Predictions list (loaded from your json dump)
    preds = load_json("predictions/2024_run_all_resultstitdate.json")
    preds = clean_titles_flat(preds)

    # Align & filter
    true_clean, pred_clean = filter_pairs(current_titles, preds)
    true_clean = true_clean[:1000]
    pred_clean = pred_clean[:1000]

    ev = TitleForecastEvaluator()

    # Build corpus/index
    papers_2024 = ev.flatten_2024_corpus(records)
    idx = ev.build_corpus_index(papers_2024)

    # Precompute embeddings for ranking (may take time/memory)
    ev.build_candidate_embeddings(idx.candidate_titles, batch_size=256)

    # Optional: enable KeyBERT (if you want that stage)
    # ev.enable_keybert()

    # Rank: oracle categories (analysis upper-bound)
    ranks = ev.rank_many(
        pred_titles=pred_clean,
        true_titles=true_clean,
        idx=idx,
        papers_2024=papers_2024,
        use_oracle_categories=True,
        keybert_shrink=False,   # set True if you enabled keybert
        bm25_rerank=False,
    )
    print("Ranking summary:", ev.summary(ranks))

    # Similarity metrics (optional)
    # Pm, Rm, Fm = ev.bertscore(pred_clean, true_clean)
    # print("BERTScore:", {"P": Pm, "R": Rm, "F1": Fm})

    pair_scores = ev.specter_pairwise_scores(pred_clean, true_clean)
    print("Avg SPECTER cosine:", float(np.mean(pair_scores)))
