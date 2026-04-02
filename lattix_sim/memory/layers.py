"""
Sistema de memoria por capas para Lattix.

4 capas con diferentes propósitos, TTL y protocolos de escritura:
- Episódica: eventos del día (TTL corto, alta rotación)
- Semántica: conocimiento validado (TTL largo, baja rotación)
- Procedimental: workflows estables (sin TTL)
- Hipótesis: hipótesis activas con evidencia (TTL medio)

Principio rector: "La memoria no decide: recupera, resume y prioriza
evidencia para que el nodo correcto decida."
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from ..config import (
    MemoryLayer, MemoryParams, DEFAULT_MEMORY_PARAMS,
    AGENT_MEMORY_WRITE_PROTOCOL, AGENT_MEMORY_QUALITY,
)


@dataclass
class MemoryEntry:
    """Una entrada individual de memoria."""
    entry_id: int
    layer: MemoryLayer
    agent_source: str
    confidence: float
    validation_state: str       # "candidate" | "validated" | "deprecated"
    created_session: int
    last_accessed: int
    relevance_score: float
    content_hash: float         # Simulado como valor numérico para similitud
    evidence_count: int = 0

    @property
    def is_valid(self) -> bool:
        return self.validation_state != "deprecated"


class MemoryStore:
    """
    Almacén de memoria de 4 capas.

    Gestiona el ciclo de vida completo: creación → acceso → decaimiento →
    promoción → deprecación.
    """

    def __init__(
        self,
        params: MemoryParams = None,
        rng: np.random.Generator = None,
    ):
        self.params = params or DEFAULT_MEMORY_PARAMS
        self.rng = rng or np.random.default_rng(42)
        self._entries: List[MemoryEntry] = []
        self._next_id = 0

    @property
    def entries(self) -> List[MemoryEntry]:
        return [e for e in self._entries if e.is_valid]

    def count_by_layer(self) -> Dict[MemoryLayer, int]:
        """Cuenta entradas válidas por capa."""
        counts = {layer: 0 for layer in MemoryLayer}
        for e in self.entries:
            counts[e.layer] += 1
        return counts

    def count_by_state(self) -> Dict[str, int]:
        """Cuenta entradas por estado de validación."""
        states = {"candidate": 0, "validated": 0, "deprecated": 0}
        for e in self._entries:
            states[e.validation_state] += 1
        return states

    def add_entry(
        self,
        layer: MemoryLayer,
        agent_source: str,
        session: int,
        confidence: Optional[float] = None,
    ) -> MemoryEntry:
        """Añade una nueva entrada de memoria."""
        if confidence is None:
            base_quality = AGENT_MEMORY_QUALITY.get(agent_source, 0.5)
            confidence = float(np.clip(
                self.rng.normal(base_quality, 0.15), 0.05, 0.99
            ))

        entry = MemoryEntry(
            entry_id=self._next_id,
            layer=layer,
            agent_source=agent_source,
            confidence=confidence,
            validation_state="candidate",
            created_session=session,
            last_accessed=session,
            relevance_score=confidence,
            content_hash=float(self.rng.random()),
        )
        self._entries.append(entry)
        self._next_id += 1
        return entry

    def generate_session_entries(
        self,
        agent_key: str,
        session: int,
    ) -> List[MemoryEntry]:
        """
        Genera entradas de memoria para un agente en una sesión.

        Respeta el protocolo de escritura (qué capas puede escribir cada agente).
        """
        allowed_layers = AGENT_MEMORY_WRITE_PROTOCOL.get(
            agent_key, [MemoryLayer.EPISODIC]
        )

        n_entries = max(1, int(self.rng.normal(
            self.params.entries_per_session_mean / 4,  # Por agente
            self.params.entries_per_session_std / 4,
        )))

        new_entries = []
        for _ in range(n_entries):
            idx = int(self.rng.integers(0, len(allowed_layers)))
            layer = allowed_layers[idx]
            entry = self.add_entry(layer, agent_key, session)
            new_entries.append(entry)

        return new_entries

    def apply_decay(self, current_session: int):
        """Aplica decaimiento temporal a las entradas."""
        decay_rates = {
            MemoryLayer.EPISODIC: self.params.episodic_decay_rate,
            MemoryLayer.HYPOTHESIS: self.params.hypothesis_decay_rate,
            MemoryLayer.SEMANTIC: self.params.semantic_decay_rate,
            MemoryLayer.PROCEDURAL: 0.0,  # Sin decaimiento
        }

        for entry in self._entries:
            if entry.validation_state == "deprecated":
                continue
            age = current_session - entry.last_accessed
            rate = decay_rates.get(entry.layer, 0.1)
            entry.relevance_score *= np.exp(-rate * age)

            # Deprecar entradas episódicas muy antiguas
            if (entry.layer == MemoryLayer.EPISODIC
                    and entry.relevance_score < 0.05):
                entry.validation_state = "deprecated"

    def get_quality_score(self) -> float:
        """
        Calidad global de la memoria: media ponderada de confidence
        de entradas semánticas y procedurales validadas.
        """
        valid = [
            e for e in self.entries
            if e.validation_state == "validated"
            and e.layer in (MemoryLayer.SEMANTIC, MemoryLayer.PROCEDURAL)
        ]
        if not valid:
            return 0.5  # Neutral si no hay memorias validadas
        return float(np.mean([e.confidence for e in valid]))

    def get_contradiction_rate(self) -> float:
        """Proporción de entradas que son contradictorias (heurística)."""
        valid = [e for e in self.entries if e.is_valid]
        if len(valid) < 2:
            return 0.0
        # Simular contradicciones: entradas con content_hash muy similar
        # pero confianza divergente
        hashes = np.array([e.content_hash for e in valid])
        confs = np.array([e.confidence for e in valid])

        contradictions = 0
        n = len(valid)
        sample_size = min(100, n * (n - 1) // 2)
        for _ in range(sample_size):
            i, j = self.rng.choice(n, size=2, replace=False)
            hash_sim = 1.0 - abs(hashes[i] - hashes[j])
            conf_diff = abs(confs[i] - confs[j])
            if hash_sim > 0.9 and conf_diff > 0.4:
                contradictions += 1

        return contradictions / max(sample_size, 1)

    def get_metrics(self, session: int) -> Dict[str, float]:
        """Métricas completas del estado de la memoria."""
        counts = self.count_by_layer()
        states = self.count_by_state()
        total = sum(counts.values())
        return {
            "total_entries": total,
            "episodic_count": counts[MemoryLayer.EPISODIC],
            "semantic_count": counts[MemoryLayer.SEMANTIC],
            "procedural_count": counts[MemoryLayer.PROCEDURAL],
            "hypothesis_count": counts[MemoryLayer.HYPOTHESIS],
            "validated_count": states["validated"],
            "candidate_count": states["candidate"],
            "deprecated_count": states["deprecated"],
            "memory_quality": self.get_quality_score(),
            "contradiction_rate": self.get_contradiction_rate(),
            "session": session,
        }
