"""Tests para el modelo psicométrico Lattix 01."""

import numpy as np
import pandas as pd
import pytest

from lattix_sim.config import (
    LattixVariable, AGENT_PROFILES, Condition,
    CONDITION_MODIFIERS, SimulationParams,
)
from lattix_sim.models.agents import Agent
from lattix_sim.models.conditions import ConditionEngine
from lattix_sim.models.psychometric import LattixScorer


class TestAgentSampling:
    """Verifica que los agentes generan muestras válidas."""

    def setup_method(self):
        self.rng = np.random.default_rng(42)

    def test_beta_variables_in_range(self):
        """Variables Beta deben estar en [0, 1]."""
        agent = Agent(AGENT_PROFILES["claude_code"], self.rng)
        modifier = CONDITION_MODIFIERS[Condition.DUAL]

        for var in LattixVariable:
            if var == LattixVariable.META_PROPOSALS:
                continue
            samples = agent.sample(var, modifier, n=1000)
            assert samples.min() >= 0.0
            assert samples.max() <= 1.0

    def test_poisson_non_negative(self):
        """Meta-propuestas (Poisson) deben ser >= 0."""
        agent = Agent(AGENT_PROFILES["lumen"], self.rng)
        modifier = CONDITION_MODIFIERS[Condition.TRIADIC]
        samples = agent.sample(LattixVariable.META_PROPOSALS, modifier, n=1000)
        assert (samples >= 0).all()

    def test_different_agents_different_profiles(self):
        """Agentes distintos deben tener distribuciones diferentes."""
        claude = Agent(AGENT_PROFILES["claude_code"], np.random.default_rng(42))
        gemini = Agent(AGENT_PROFILES["gemini"], np.random.default_rng(42))
        modifier = CONDITION_MODIFIERS[Condition.DUAL]

        claude_samples = claude.sample(LattixVariable.FUNC_DIST, modifier, n=5000)
        gemini_samples = gemini.sample(LattixVariable.FUNC_DIST, modifier, n=5000)

        # Claude Code debería tener mayor func_dist que Gemini
        assert claude_samples.mean() > gemini_samples.mean()


class TestConditionEngine:
    """Verifica la lógica de condiciones experimentales."""

    def setup_method(self):
        self.rng = np.random.default_rng(42)
        self.engine = ConditionEngine(self.rng)

    def test_solo_includes_all_agents(self):
        active = self.engine.get_active_agents(Condition.SOLO)
        assert len(active) == 4

    def test_dual_excludes_lumen(self):
        active = self.engine.get_active_agents(Condition.DUAL)
        assert "lumen" not in active
        assert len(active) == 3

    def test_triadic_includes_all(self):
        active = self.engine.get_active_agents(Condition.TRIADIC)
        assert "lumen" in active
        assert len(active) == 4

    def test_coupling_increases_with_condition(self):
        """El acoplamiento debe ser 0 < dual < triádico."""
        assert CONDITION_MODIFIERS[Condition.SOLO].coupling_weight == 0.0
        assert CONDITION_MODIFIERS[Condition.DUAL].coupling_weight < \
               CONDITION_MODIFIERS[Condition.TRIADIC].coupling_weight

    def test_sample_condition_returns_correct_shape(self):
        n_trials = 100
        samples = self.engine.sample_condition(Condition.DUAL, n_trials)
        for agent_key, var_samples in samples.items():
            for var in LattixVariable:
                assert len(var_samples[var]) == n_trials


class TestLattixScorer:
    """Verifica el scoring psicométrico."""

    def test_normalize_meta_proposals(self):
        values = np.array([0, 5, 10, 15, 20])
        normalized = LattixScorer.normalize_meta_proposals(values, max_clip=15)
        assert normalized[0] == 0.0
        assert normalized[-1] == 1.0  # 20 se clipea a 15, luego /15 = 1.0

    def test_lattix_index_range(self):
        """El índice Lattix debe estar en [0, ~1]."""
        df = pd.DataFrame({
            "func_dist": [0.5, 0.8, 0.3],
            "enunc_stability": [0.6, 0.7, 0.4],
            "gap_detection": [0.7, 0.9, 0.5],
            "meta_proposals": [3.0, 5.0, 1.0],
            "choral_utility": [0.6, 0.8, 0.2],
        })
        index = LattixScorer.compute_lattix_index(df)
        assert (index >= 0).all()
        assert (index <= 1.5).all()  # Puede ser > 1 por meta_proposals normalizadas
