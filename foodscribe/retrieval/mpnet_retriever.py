"""Semantic food retrieval using MPNet embeddings."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class RetrievalResult:
    fdc_id: int
    description: str
    score: float   # cosine similarity [-1, 1]
    rank: int      # 1 = best match


class MPNetRetriever:
    """
    Offline phase  : embeddings pre-computed by scripts/build_data.py
    Online phase   : encode query -> cosine similarity -> top-k fdc_ids

    Model          : sentence-transformers/all-mpnet-base-v2
                     768-dim L2-normalised vectors
    """

    def __init__(self, data_dir: str | Path = "data/", top_k: int = 3) -> None:
        self.data_dir = Path(data_dir)
        self.top_k = top_k
        self._embeddings: np.ndarray | None = None
        self._metadata: pd.DataFrame | None = None
        self._model = None  # lazy-loaded

    # ------------------------------------------------------------------
    # Public API (stubs — filled in Stage 3)
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        """Embed query, cosine-search embeddings matrix, return top-k results."""
        return self.retrieve_batch([query], top_k=top_k)[0]

    def retrieve_batch(
        self, queries: list[str], top_k: int | None = None
    ) -> list[list[RetrievalResult]]:
        """Encode all queries in one batch, return top-k results per query."""
        if self._embeddings is None:
            self._load()
        k = top_k if top_k is not None else self.top_k
        model = self._get_model()
        q_embs = model.encode(queries, normalize_embeddings=True, convert_to_numpy=True)
        # scores shape: (n_queries, n_foods)
        scores = q_embs @ self._embeddings.T
        results: list[list[RetrievalResult]] = []
        for row_scores in scores:
            top_idx = np.argsort(row_scores)[::-1][:k]
            top_results = []
            for rank, idx in enumerate(top_idx, start=1):
                meta_row = self._metadata.iloc[idx]
                top_results.append(
                    RetrievalResult(
                        fdc_id=int(meta_row["fdc_id"]),
                        description=str(meta_row["description"]),
                        score=float(row_scores[idx]),
                        rank=rank,
                    )
                )
            results.append(top_results)
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        emb_path = self.data_dir / "food_embeddings_mpnet.npy"
        meta_path = self.data_dir / "food_metadata.csv"
        if not emb_path.exists():
            raise FileNotFoundError(
                f"Embeddings not found at {emb_path}. "
                "Run: python scripts/build_data.py --usda-dir <path> --data-dir data/"
            )
        self._embeddings = np.load(emb_path, mmap_mode="r")
        self._metadata = pd.read_csv(meta_path)

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("all-mpnet-base-v2")
        return self._model
