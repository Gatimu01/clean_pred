import numpy as np
import re
from typing import List, Optional
from .corpus import CorpusIndex

def tokenize_bm25(s: str) -> List[str]:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    return [w for w in s.split() if w]

class EmbeddingRanker:
    def __init__(self, embedder, E_candidates: np.ndarray):
        self.embedder = embedder
        self.E = E_candidates  # [N, d] float32

    def rank(self, pred_title: str, true_title: str, pool_idxs: List[int], idx: CorpusIndex) -> Optional[int]:
        true_global = idx.title2idx.get(true_title)
        if true_global is None or true_global not in set(pool_idxs):
            return None

        e_pred = self.embedder.encode([pred_title], batch_size=1, normalize=True)[0]  # [d]
        scores = self.E[pool_idxs] @ e_pred

        true_pos = pool_idxs.index(true_global)
        true_score = scores[true_pos]
        return int(1 + np.sum(scores > true_score))

class BM25Ranker:
    def rank(self, query_title: str, true_title: str, pool_idxs: List[int], idx: CorpusIndex) -> Optional[int]:
        from rank_bm25 import BM25Okapi

        true_global = idx.title2idx.get(true_title)
        if true_global is None or true_global not in set(pool_idxs):
            return None

        pool_titles = [idx.candidate_titles[i] for i in pool_idxs]
        tokenized_docs = [tokenize_bm25(t) for t in pool_titles]
        bm25 = BM25Okapi(tokenized_docs)

        q = tokenize_bm25(query_title)
        scores = np.asarray(bm25.get_scores(q), dtype=np.float32)

        true_pos = pool_idxs.index(true_global)
        true_score = scores[true_pos]
        return int(1 + np.sum(scores > true_score))

class KeyBERTShrinker:
    def __init__(self, backbone: str = "allenai/scibert_scivocab_uncased"):
        from keybert import KeyBERT
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer(backbone)
        self.kw = KeyBERT(model=embedder)

    def extract(self, text: str, top_n: int = 8):
        kws = self.kw.extract_keywords(text, keyphrase_ngram_range=(1,3), stop_words="english", top_n=top_n)
        return [k for k, _ in kws]

    def shrink(self, pred_title: str, pool_idxs: List[int], idx: CorpusIndex, E_candidates: np.ndarray, embedder, M: int = 800, top_phrases: int = 8):
        phrases = self.extract(pred_title, top_n=top_phrases)
        query = " ".join(phrases) if phrases else pred_title
        e_q = embedder.encode([query], batch_size=1, normalize=True)[0]
        scores = E_candidates[pool_idxs] @ e_q
        M = min(M, len(pool_idxs))
        top_local = np.argpartition(scores, -M)[-M:]
        top_local = top_local[np.argsort(scores[top_local])[::-1]]
        return [pool_idxs[j] for j in top_local.tolist()], phrases
