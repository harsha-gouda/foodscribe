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

    # Score bonus added to cosine similarity before ranking.
    # Keeps Foundation foods preferred over SR Legacy / Survey when scores are close.
    _DEFAULT_BOOST = {"foundation": 0.02, "sr_legacy": 0.01}

    def __init__(
        self,
        data_dir: str | Path = "data/",
        top_k: int = 3,
        source_boost: dict | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.top_k = top_k
        self._boost = source_boost if source_boost is not None else self._DEFAULT_BOOST
        self._embeddings: np.ndarray | None = None
        self._metadata: pd.DataFrame | None = None
        self._source_bonus: np.ndarray | None = None
        self._model = None  # lazy-loaded

    # ------------------------------------------------------------------
    # Public API (stubs — filled in Stage 3)
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        """Embed query, cosine-search embeddings matrix, return top-k results."""
        return self.retrieve_batch([query], top_k=top_k)[0]

    def retrieve_batch(
        self,
        queries: list[str],
        top_k: int | None = None,
        contexts: list[str] | None = None,
    ) -> list[list[RetrievalResult]]:
        """Encode all queries in one batch, return top-k results per query.

        Args:
            queries: Ingredient names (or USDA-style descriptions) to search.
            top_k: Number of results per query.
            contexts: Optional meal descriptions, one per query. When provided,
                      each query is enriched as "{query} [from: {context}]" before
                      encoding, giving the embedding model meal-level context.
        """
        if self._embeddings is None:
            self._load()
        k = top_k if top_k is not None else self.top_k
        model = self._get_model()
        if contexts:
            enriched = [
                f"{q} [from: {c}]" if c else q
                for q, c in zip(queries, contexts)
            ]
        else:
            enriched = queries
        q_embs = model.encode(enriched, normalize_embeddings=True, convert_to_numpy=True)
        # scores shape: (n_queries, n_foods)
        scores = q_embs @ self._embeddings.T
        results: list[list[RetrievalResult]] = []
        for row_scores in scores:
            # Add source-priority bonus before ranking; report raw score to caller
            adjusted = row_scores + self._source_bonus
            top_idx = np.argsort(adjusted)[::-1][:k]
            top_results = []
            for rank, idx in enumerate(top_idx, start=1):
                meta_row = self._metadata.iloc[idx]
                top_results.append(
                    RetrievalResult(
                        fdc_id=int(meta_row["fdc_id"]),
                        description=str(meta_row["description"]),
                        score=float(row_scores[idx]),  # raw cosine similarity
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
        # Build per-food source bonus array aligned with metadata row order
        self._source_bonus = np.array(
            [self._boost.get(str(dt).lower(), 0.0) for dt in self._metadata["data_type"]],
            dtype=np.float32,
        )

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("all-mpnet-base-v2")
        return self._model
