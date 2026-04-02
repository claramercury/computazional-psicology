"""
Motor de recuperación de memoria.

Simula búsqueda híbrida (70% vector + 30% keyword) inspirada
en OpenClaw/Qdrant. La calidad de la recuperación afecta directamente
el rendimiento del agente en la sesión.
"""

import numpy as np
from typing import List, Dict

from ..config import MemoryParams, DEFAULT_MEMORY_PARAMS
from .layers import MemoryStore, MemoryEntry


class RetrievalEngine:
    """Motor de recuperación de memorias relevantes."""

    def __init__(
        self,
        store: MemoryStore,
        params: MemoryParams = None,
        rng: np.random.Generator = None,
    ):
        self.store = store
        self.params = params or DEFAULT_MEMORY_PARAMS
        self.rng = rng or np.random.default_rng(42)

    def retrieve(
        self,
        query_vector: float = None,
        current_session: int = 0,
    ) -> List[MemoryEntry]:
        """
        Recupera las top-k memorias más relevantes.

        Scoring = vector_weight × similitud_vector
                + (1 - vector_weight) × relevancia × recencia
        """
        if query_vector is None:
            query_vector = float(self.rng.random())

        valid = [e for e in self.store.entries if e.is_valid]
        if not valid:
            return []

        scored = []
        for entry in valid:
            # Similitud vectorial simulada
            vec_sim = 1.0 - abs(entry.content_hash - query_vector)

            # Recencia normalizada
            age = max(1, current_session - entry.last_accessed + 1)
            recency = 1.0 / np.sqrt(age)

            # Score combinado
            score = (
                self.params.vector_weight * vec_sim
                + (1 - self.params.vector_weight) * entry.relevance_score * recency
            )

            # Bonus por validación
            if entry.validation_state == "validated":
                score *= 1.2

            scored.append((score, entry))

        # Ordenar por score descendente
        scored.sort(key=lambda x: x[0], reverse=True)

        # Top-k
        top_k = scored[:self.params.retrieval_top_k]

        # Actualizar last_accessed
        for _, entry in top_k:
            entry.last_accessed = current_session

        return [entry for _, entry in top_k]

    def compute_retrieval_quality(
        self,
        retrieved: List[MemoryEntry],
    ) -> float:
        """
        Calidad de la recuperación: media ponderada de confianza
        y estado de validación de las memorias recuperadas.
        """
        if not retrieved:
            return 0.0

        scores = []
        for entry in retrieved:
            base = entry.confidence
            if entry.validation_state == "validated":
                base *= 1.2
            scores.append(min(1.0, base))

        return float(np.mean(scores))

    def compute_memory_boost(
        self,
        retrieval_quality: float,
        memory_quality: float,
    ) -> float:
        """
        Calcula el boost al rendimiento basado en la calidad de
        recuperación y la calidad global de la memoria.

        Retorna un multiplicador ∈ [1.0, 1.0 + max_boost].
        """
        combined = (retrieval_quality + memory_quality) / 2.0
        boost = combined * self.params.memory_quality_boost_max
        return 1.0 + boost
