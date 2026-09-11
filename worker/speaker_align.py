"""Cross-chunk speaker label alignment.

Each transcribed chunk labels its speakers independently (S01, S02, ...),
so the same person may get different labels in different chunks. When a
voice-embedding function is available we match chunk-local speakers to
global labels by centroid cosine similarity; unmatched speakers get
fresh global labels (a false split is safer than a false merge for
meeting minutes).
"""

from __future__ import annotations

import numpy as np

DEFAULT_THRESHOLD = 0.72


class SpeakerAligner:
    def __init__(self, threshold: float = DEFAULT_THRESHOLD):
        self.threshold = threshold
        self.centroids: dict[str, np.ndarray] = {}

    def _best_match(self, emb: np.ndarray) -> str | None:
        best, best_sim = None, self.threshold
        for label, centroid in self.centroids.items():
            denom = np.linalg.norm(emb) * np.linalg.norm(centroid) + 1e-9
            sim = float(np.dot(emb, centroid) / denom)
            if sim > best_sim:
                best, best_sim = label, sim
        return best

    def add_chunk(self, local_embeddings: dict[str, np.ndarray]) -> dict[str, str]:
        """Map one chunk's local speaker labels to global labels.

        Decisions for the whole chunk are made before any centroid update,
        so speakers from the same chunk cannot absorb each other.
        """
        decided: dict[str, str | None] = {
            local: self._best_match(emb) for local, emb in local_embeddings.items()
        }
        mapping: dict[str, str] = {}
        for local, emb in local_embeddings.items():
            label = decided[local]
            if label is None:
                label = f"S{len(self.centroids) + 1:02d}"
                self.centroids[label] = emb / (np.linalg.norm(emb) + 1e-9)
            else:
                merged = self.centroids[label] + emb
                self.centroids[label] = merged / (np.linalg.norm(merged) + 1e-9)
            mapping[local] = label
        return mapping
