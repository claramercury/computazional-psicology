"""
Motor de consolidación nocturna ("Auto-Dream").

Simula el ciclo de aprendizaje sin reentrenamiento de pesos:
1. Ingesta de entradas episódicas del día
2. Deduplicación semántica
3. Extracción de lecciones candidatas
4. Scoring de calidad
5. Promoción a memoria semántica validada
6. Detección de contradicciones
7. Decaimiento temporal

Inspirado en el sistema Auto-Dream de OpenClaw/ClaudeCode
y en la consolidación de memoria durante el sueño (neurociencia).
"""

import numpy as np
from typing import Dict, List

from ..config import MemoryLayer, MemoryParams, DEFAULT_MEMORY_PARAMS
from .layers import MemoryStore, MemoryEntry


class ConsolidationEngine:
    """Motor de consolidación nocturna de memoria."""

    def __init__(
        self,
        store: MemoryStore,
        params: MemoryParams = None,
        rng: np.random.Generator = None,
    ):
        self.store = store
        self.params = params or DEFAULT_MEMORY_PARAMS
        self.rng = rng or np.random.default_rng(42)

    def run_nightly_cycle(self, current_session: int) -> Dict[str, int]:
        """
        Ejecuta el ciclo completo de consolidación nocturna.

        Retorna métricas del ciclo.
        """
        metrics = {
            "deduplicated": 0,
            "promoted": 0,
            "contradictions_found": 0,
            "deprecated": 0,
            "lessons_extracted": 0,
        }

        # 1. Deduplicación semántica
        metrics["deduplicated"] = self._deduplicate()

        # 2. Extracción de lecciones (episodic → hypothesis candidates)
        metrics["lessons_extracted"] = self._extract_lessons(current_session)

        # 3. Scoring y promoción (hypothesis/candidate → semantic/validated)
        metrics["promoted"] = self._promote_knowledge(current_session)

        # 4. Detección de contradicciones
        metrics["contradictions_found"] = self._detect_contradictions()

        # 5. Decaimiento temporal
        self.store.apply_decay(current_session)

        # 6. Contar deprecadas tras decaimiento
        metrics["deprecated"] = sum(
            1 for e in self.store._entries
            if e.validation_state == "deprecated"
        )

        return metrics

    def _deduplicate(self) -> int:
        """
        Elimina duplicados semánticos.

        Compara content_hash de entradas en la misma capa.
        Si similitud > umbral, depreca la más antigua.
        """
        removed = 0
        valid = [e for e in self.store.entries if e.is_valid]

        # Agrupar por capa
        by_layer: Dict[MemoryLayer, List[MemoryEntry]] = {}
        for entry in valid:
            by_layer.setdefault(entry.layer, []).append(entry)

        for layer, entries in by_layer.items():
            if len(entries) < 2:
                continue

            hashes = np.array([e.content_hash for e in entries])
            n = len(entries)

            # Comparar pares (limitado para eficiencia)
            max_comparisons = min(200, n * (n - 1) // 2)
            for _ in range(max_comparisons):
                i, j = self.rng.choice(n, size=2, replace=False)
                similarity = 1.0 - abs(hashes[i] - hashes[j])

                if similarity > self.params.dedup_similarity_threshold:
                    # Deprecar la de menor confianza
                    if entries[i].confidence < entries[j].confidence:
                        entries[i].validation_state = "deprecated"
                    else:
                        entries[j].validation_state = "deprecated"
                    removed += 1

        return removed

    def _extract_lessons(self, current_session: int) -> int:
        """
        Extrae lecciones de entradas episódicas de alta confianza.

        Episodic con confidence > umbral → nuevo candidato en hypothesis.
        """
        extracted = 0
        episodic = [
            e for e in self.store.entries
            if e.layer == MemoryLayer.EPISODIC
            and e.confidence > self.params.promotion_confidence_threshold * 0.8
            and e.is_valid
        ]

        for entry in episodic:
            # Probabilidad de extracción proporcional a confianza
            if self.rng.random() < entry.confidence * 0.3:
                new_entry = self.store.add_entry(
                    layer=MemoryLayer.HYPOTHESIS,
                    agent_source=entry.agent_source,
                    session=current_session,
                    confidence=entry.confidence * 0.9,
                )
                new_entry.evidence_count = 1
                extracted += 1

        return extracted

    def _promote_knowledge(self, current_session: int) -> int:
        """
        Promueve hipótesis bien fundamentadas a conocimiento semántico.

        Requisitos:
        - confidence > umbral de promoción
        - evidence_count >= mínimo requerido
        """
        promoted = 0
        candidates = [
            e for e in self.store.entries
            if e.layer == MemoryLayer.HYPOTHESIS
            and e.validation_state == "candidate"
            and e.is_valid
        ]

        for entry in candidates:
            # Simular acumulación de evidencia con el tiempo
            age = current_session - entry.created_session
            if age > 0:
                entry.evidence_count += int(self.rng.poisson(0.3))

            if (entry.confidence >= self.params.promotion_confidence_threshold
                    and entry.evidence_count >= self.params.promotion_min_evidence):
                # Promover: crear entrada semántica validada
                new_entry = self.store.add_entry(
                    layer=MemoryLayer.SEMANTIC,
                    agent_source=entry.agent_source,
                    session=current_session,
                    confidence=min(0.99, entry.confidence * 1.1),
                )
                new_entry.validation_state = "validated"
                new_entry.evidence_count = entry.evidence_count

                # Deprecar la hipótesis original
                entry.validation_state = "deprecated"
                promoted += 1

        return promoted

    def _detect_contradictions(self) -> int:
        """
        Detecta contradicciones entre entradas semánticas.

        Heurística: entradas con content_hash similar pero
        agentes fuente diferentes y confianzas divergentes.
        """
        semantic = [
            e for e in self.store.entries
            if e.layer == MemoryLayer.SEMANTIC
            and e.is_valid
        ]

        if len(semantic) < 2:
            return 0

        contradictions = 0
        hashes = np.array([e.content_hash for e in semantic])
        n = len(semantic)

        max_checks = min(100, n * (n - 1) // 2)
        for _ in range(max_checks):
            i, j = self.rng.choice(n, size=2, replace=False)
            hash_sim = 1.0 - abs(hashes[i] - hashes[j])
            conf_diff = abs(semantic[i].confidence - semantic[j].confidence)

            if (hash_sim > 0.85 and conf_diff > 0.3
                    and semantic[i].agent_source != semantic[j].agent_source):
                contradictions += 1
                # Reducir confianza de ambas entradas
                semantic[i].confidence *= 0.9
                semantic[j].confidence *= 0.9

        return contradictions
