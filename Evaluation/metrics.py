import numpy as np
from typing import Dict, List, Optional, Tuple

def hits_at_k(ranks: List[int], k: int) -> float:
    ranks = [r for r in ranks if r is not None]
    return 0.0 if not ranks else sum(r <= k for r in ranks) / len(ranks)

def mean_reciprocal_rank(ranks: List[int]) -> float:
    ranks = [r for r in ranks if r is not None]
    return 0.0 if not ranks else float(np.mean([1.0 / r for r in ranks]))

def summary(ranks: List[Optional[int]], ks: Tuple[int, ...] = (10, 25, 50, 100, 200, 500)) -> Dict[str, float]:
    rr = [r for r in ranks if r is not None]
    out = {"n": float(len(rr))}
    for k in ks:
        out[f"hits@{k}"] = hits_at_k(rr, k)
    out["mrr"] = mean_reciprocal_rank(rr)
    out["median_rank"] = float(np.median(rr)) if rr else 0.0
    out["mean_rank"] = float(np.mean(rr)) if rr else 0.0
    return out
