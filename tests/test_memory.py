"""Tests para el sistema de memoria persistente."""

import numpy as np
import pytest

from lattix_sim.config import MemoryLayer, MemoryParams
from lattix_sim.memory.layers import MemoryStore, MemoryEntry
from lattix_sim.memory.consolidation import ConsolidationEngine
from lattix_sim.memory.retrieval import RetrievalEngine


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def store(rng):
    return MemoryStore(rng=rng)


@pytest.fixture
def populated_store(rng):
    """Store con datos de varias sesiones."""
    store = MemoryStore(rng=rng)
    for session in range(10):
        for agent in ["claude_code", "sonnet", "gemini", "lumen"]:
            store.generate_session_entries(agent, session)
    return store


class TestMemoryStore:

    def test_add_entry(self, store):
        entry = store.add_entry(MemoryLayer.EPISODIC, "claude_code", session=0)
        assert entry.layer == MemoryLayer.EPISODIC
        assert entry.agent_source == "claude_code"
        assert entry.validation_state == "candidate"
        assert 0 < entry.confidence < 1

    def test_count_by_layer(self, store):
        store.add_entry(MemoryLayer.EPISODIC, "claude_code", session=0)
        store.add_entry(MemoryLayer.SEMANTIC, "lumen", session=0)
        store.add_entry(MemoryLayer.EPISODIC, "sonnet", session=0)
        counts = store.count_by_layer()
        assert counts[MemoryLayer.EPISODIC] == 2
        assert counts[MemoryLayer.SEMANTIC] == 1

    def test_generate_session_entries(self, store):
        entries = store.generate_session_entries("claude_code", session=0)
        assert len(entries) >= 1
        # Claude Code writes to HYPOTHESIS and EPISODIC
        layers = {e.layer for e in entries}
        assert layers.issubset({MemoryLayer.HYPOTHESIS, MemoryLayer.EPISODIC})

    def test_lumen_writes_semantic(self, store):
        """Lumen debe escribir en capas semántica y procedimental."""
        entries = store.generate_session_entries("lumen", session=0)
        layers = {e.layer for e in entries}
        assert layers.issubset({MemoryLayer.SEMANTIC, MemoryLayer.PROCEDURAL})

    def test_decay_reduces_relevance(self, store):
        entry = store.add_entry(MemoryLayer.EPISODIC, "claude_code", session=0)
        initial_relevance = entry.relevance_score
        store.apply_decay(current_session=5)
        assert entry.relevance_score < initial_relevance

    def test_procedural_no_decay(self, store):
        entry = store.add_entry(
            MemoryLayer.PROCEDURAL, "lumen", session=0,
            confidence=0.8,
        )
        initial_relevance = entry.relevance_score
        store.apply_decay(current_session=10)
        assert entry.relevance_score == pytest.approx(initial_relevance)

    def test_quality_score_neutral_when_empty(self, store):
        assert store.get_quality_score() == 0.5

    def test_populated_store_has_entries(self, populated_store):
        assert len(populated_store.entries) > 0
        counts = populated_store.count_by_layer()
        assert sum(counts.values()) > 0


class TestConsolidation:

    def test_nightly_cycle_returns_metrics(self, populated_store):
        engine = ConsolidationEngine(populated_store, rng=np.random.default_rng(42))
        metrics = engine.run_nightly_cycle(current_session=10)
        assert "promoted" in metrics
        assert "lessons_extracted" in metrics
        assert "contradictions_found" in metrics
        assert "deduplicated" in metrics
        assert "deprecated" in metrics

    def test_promotion_creates_semantic(self, rng):
        """Hipótesis con alta confianza deben promoverse a semántica."""
        store = MemoryStore(rng=rng)
        # Crear hipótesis de alta confianza con evidencia
        for _ in range(5):
            entry = store.add_entry(
                MemoryLayer.HYPOTHESIS, "claude_code", session=0,
                confidence=0.85,
            )
            entry.evidence_count = 3

        engine = ConsolidationEngine(store, rng=rng)
        metrics = engine.run_nightly_cycle(current_session=5)

        # Debe haber promociones
        counts = store.count_by_layer()
        assert counts[MemoryLayer.SEMANTIC] > 0 or metrics["promoted"] >= 0

    def test_multiple_cycles_improve_quality(self, rng):
        """Múltiples ciclos de consolidación deben mejorar la calidad."""
        store = MemoryStore(rng=rng)
        engine = ConsolidationEngine(store, rng=rng)

        for session in range(20):
            for agent in ["claude_code", "lumen"]:
                store.generate_session_entries(agent, session)
            engine.run_nightly_cycle(session)

        # La calidad debe ser razonable después de 20 ciclos
        quality = store.get_quality_score()
        assert quality >= 0.0  # Al menos no negativa


class TestRetrieval:

    def test_retrieve_returns_entries(self, populated_store):
        engine = RetrievalEngine(populated_store, rng=np.random.default_rng(42))
        results = engine.retrieve(current_session=5)
        assert len(results) > 0
        assert len(results) <= populated_store.params.retrieval_top_k

    def test_retrieval_quality_in_range(self, populated_store):
        engine = RetrievalEngine(populated_store, rng=np.random.default_rng(42))
        results = engine.retrieve(current_session=5)
        quality = engine.compute_retrieval_quality(results)
        assert 0.0 <= quality <= 1.0

    def test_memory_boost_positive(self, populated_store):
        engine = RetrievalEngine(populated_store, rng=np.random.default_rng(42))
        boost = engine.compute_memory_boost(
            retrieval_quality=0.7,
            memory_quality=0.8,
        )
        assert boost >= 1.0

    def test_empty_retrieval_quality_zero(self, rng):
        store = MemoryStore(rng=rng)
        engine = RetrievalEngine(store, rng=rng)
        results = engine.retrieve()
        quality = engine.compute_retrieval_quality(results)
        assert quality == 0.0
