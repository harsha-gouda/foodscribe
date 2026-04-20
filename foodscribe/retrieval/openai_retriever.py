"""OpenAI text-embedding-3-small retriever for USDA food matching."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_EMBED_MODEL = "text-embedding-3-large"
_DIM = 3072
_DEFAULT_BOOST: dict[str, float] = {"foundation": 0.02, "sr_legacy": 0.01}


@dataclass
class RetrievalResult:
    fdc_id: int
    description: str
    score: float
    rank: int


class OpenAIRetriever:
    """
    Retrieves USDA food matches using OpenAI text-embedding-3-small embeddings.

    Requires:
      - data/food_embeddings_openai.npy  (N, 1536) float32 L2-normalised
      - data/food_metadata.csv           fdc_id, description, food_category, data_type

    Build the index with:
      python scripts/build_data.py --skip-embeddings=false --embedder openai
    """

    def __init__(
        self,
        data_dir: Path,
        top_k: int = 3,
        source_boost: dict[str, float] | None = None,
        foundation_threshold: float = 0.70,
    ) -> None:
        self._data_dir = Path(data_dir)
        self._top_k = top_k
        self._boost_cfg = source_boost or _DEFAULT_BOOST
        self._foundation_threshold = foundation_threshold
        # Lazy-loaded
        self._embeddings: np.ndarray | None = None
        self._metadata: pd.DataFrame | None = None
        self._source_bonus: np.ndarray | None = None
        self._foundation_indices: np.ndarray | None = None
        self._foundation_embeddings: np.ndarray | None = None
        self._client = None

    def _load(self) -> None:
        if self._embeddings is not None:
            return

        from openai import OpenAI

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable not set")
        self._client = OpenAI(api_key=api_key)

        emb_path = self._data_dir / "food_embeddings_openai.npy"
        meta_path = self._data_dir / "food_metadata.csv"

        if not emb_path.exists():
            raise FileNotFoundError(
                f"OpenAI embedding index not found: {emb_path}\n"
                "Build it with: python scripts/build_data.py --skip-embeddings false --embedder openai"
            )

        self._embeddings = np.load(emb_path, mmap_mode="r")
        self._metadata = pd.read_csv(meta_path)

        # Per-food source priority bonus
        data_types = self._metadata["data_type"].str.lower().tolist()
        self._source_bonus = np.array(
            [self._boost_cfg.get(dt, 0.0) for dt in data_types], dtype=np.float32
        )

        # Foundation sub-index for threshold routing
        foundation_mask = self._metadata["data_type"].str.lower() == "foundation"
        self._foundation_indices = np.where(foundation_mask.values)[0]
        self._foundation_embeddings = self._embeddings[self._foundation_indices]

    _EMBED_BATCH_SIZE = 2048  # OpenAI API limit

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Call OpenAI API and return L2-normalised (N, 1536) float32 array.

        Chunks into batches of at most _EMBED_BATCH_SIZE to stay within the
        OpenAI API limit of 2048 inputs per request.
        """
        all_vecs: list[np.ndarray] = []
        for start in range(0, len(texts), self._EMBED_BATCH_SIZE):
            chunk = texts[start : start + self._EMBED_BATCH_SIZE]
            response = self._client.embeddings.create(model=_EMBED_MODEL, input=chunk)
            all_vecs.append(np.array([e.embedding for e in response.data], dtype=np.float32))
        vecs = np.concatenate(all_vecs, axis=0)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return vecs / norms

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        return self.retrieve_batch([query], top_k=top_k)[0]

    def retrieve_batch(
        self,
        queries: list[str],
        top_k: int | None = None,
        contexts: list[str] | None = None,
    ) -> list[list[RetrievalResult]]:
        self._load()
        k = top_k or self._top_k
        q_embs = self._embed(queries)
        scores = q_embs @ self._embeddings.T  # (n_queries, n_foods)

        results: list[list[RetrievalResult]] = []
        for i, row_scores in enumerate(scores):
            # Try foundation-only first; fall back to full index if no strong match
            f_scores = q_embs[i] @ self._foundation_embeddings.T
            best_f = float(f_scores.max()) if len(f_scores) > 0 else 0.0

            if best_f >= self._foundation_threshold:
                top_f_idx = np.argsort(f_scores)[::-1][:k]
                top_idx = self._foundation_indices[top_f_idx]
            else:
                adjusted = row_scores + self._source_bonus
                top_idx = np.argsort(adjusted)[::-1][:k]

            cands: list[RetrievalResult] = []
            for rank, idx in enumerate(top_idx, start=1):
                row = self._metadata.iloc[int(idx)]
                cands.append(
                    RetrievalResult(
                        fdc_id=int(row["fdc_id"]),
                        description=str(row["description"]),
                        score=float(row_scores[idx]),
                        rank=rank,
                    )
                )
            results.append(cands)

        return results
