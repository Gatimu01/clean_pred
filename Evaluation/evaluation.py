import numpy as np
from typing import List, Optional
from .corpus import CorpusBuilder, CorpusIndex
from .embdding import EmbeddingModel
from .ranker import EmbeddingRanker, BM25Ranker, KeyBERTShrinker
from .metrics import summary

class TitleForecastEvaluator:
    def __init__(self, specter_model: str = "sentence-transformers/allenai-specter", device: Optional[str] = None):
        self.embedder = EmbeddingModel(specter_model, device=device)
        self.E_candidates: Optional[np.ndarray] = None

    def build(self, records: List[dict]):
        papers_2024 = CorpusBuilder.flatten_2024_corpus(records)
        idx = CorpusBuilder.build_index(papers_2024)
        self.E_candidates = self.embedder.encode(idx.candidate_titles, batch_size=256, normalize=True, show_progress=True)
        return papers_2024, idx

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
    ):
        if self.E_candidates is None:
            raise RuntimeError("Call build(...) first to compute candidate embeddings.")

        all_idxs = list(range(len(idx.candidate_titles)))
        ranks = []

        emb_ranker = EmbeddingRanker(self.embedder, self.E_candidates)
        bm25_ranker = BM25Ranker() if bm25_rerank else None
        shrinker = KeyBERTShrinker() if keybert_shrink else None

        for pred, true in zip(pred_titles, true_titles):
            # pool
            if use_oracle_categories:
                if papers_2024 is None:
                    raise ValueError("papers_2024 must be provided for oracle category pooling.")
                allowed = CorpusBuilder.allowed_categories_from_true_title(true, idx, papers_2024)
                pool = CorpusBuilder.pool_idxs_from_categories(allowed, idx.cat2idxs)
                if len(pool) < 2:
                    pool = all_idxs
            else:
                pool = all_idxs

            # shrink
            if shrinker is not None:
                pool, _ = shrinker.shrink(pred, pool, idx, self.E_candidates, self.embedder, M=keybert_M, top_phrases=top_phrases)

            # rank
            if bm25_ranker is not None:
                r = bm25_ranker.rank(pred, true, pool, idx)
            else:
                r = emb_ranker.rank(pred, true, pool, idx)

            ranks.append(r)

        return ranks

    def summarize(self, ranks):
        return summary(ranks)
