import numpy as np
from typing import List, Optional

class EmbeddingModel:
    def __init__(self, model_name: str, device: Optional[str] = None):
        self.model_name = model_name
        self.device = device
        self._st = None

    def _get(self):
        if self._st is None:
            from sentence_transformers import SentenceTransformer
            self._st = SentenceTransformer(self.model_name, device=self.device)
        return self._st

    def encode(self, texts: List[str], batch_size: int = 256, normalize: bool = True, show_progress: bool = False):
        st = self._get()
        emb = st.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=show_progress,
        )
        return np.asarray(emb, dtype=np.float32)
