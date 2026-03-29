"""Tests para el módulo de análisis estadístico."""

import numpy as np
import pandas as pd
import pytest

from lattix_sim.config import SimulationParams
from lattix_sim.simulation.engine import LattixSimulation
from lattix_sim.analysis.statistics import LattixStatistics
from lattix_sim.analysis.reliability import LattixReliability


@pytest.fixture
def simulation_data():
    """Dataset de simulación para tests."""
    params = SimulationParams(n_trials=200, seed=42)
    sim = LattixSimulation(params)
    return sim.run()


class TestANOVA:
    def test_anova_returns_results(self, simulation_data):
        stats = LattixStatistics(simulation_data)
        result = stats.run_anova("choral_utility")
        assert result.f_statistic > 0
        assert 0 <= result.p_value <= 1
        assert 0 <= result.eta_squared <= 1

    def test_all_anova_five_results(self, simulation_data):
        stats = LattixStatistics(simulation_data)
        results = stats.run_all_anova()
        assert len(results) == 5

    def test_choral_utility_significant(self, simulation_data):
        """Utilidad coral debe diferir significativamente entre condiciones."""
        stats = LattixStatistics(simulation_data)
        result = stats.run_anova("choral_utility")
        assert result.p_value < 0.05


class TestPostHoc:
    def test_posthoc_three_comparisons(self, simulation_data):
        stats = LattixStatistics(simulation_data)
        results = stats.posthoc_tukey("choral_utility")
        assert len(results) == 3  # solo-dual, solo-triadic, dual-triadic

    def test_cohens_d_finite(self, simulation_data):
        stats = LattixStatistics(simulation_data)
        results = stats.posthoc_tukey("func_dist")
        for r in results:
            assert np.isfinite(r.cohens_d)


class TestMANOVA:
    def test_manova_significant(self, simulation_data):
        stats = LattixStatistics(simulation_data)
        result = stats.run_manova()
        assert result.pillai_trace > 0
        assert result.p_value < 0.05


class TestBayesian:
    def test_directional_test(self, simulation_data):
        stats = LattixStatistics(simulation_data)
        result = stats.bayesian_directional_test("choral_utility")
        assert 0 <= result["posterior_probability"] <= 1
        assert result["bayes_factor"] > 0


class TestReliability:
    def test_cronbachs_alpha_reasonable(self, simulation_data):
        rel = LattixReliability(simulation_data)
        alpha = rel.cronbachs_alpha()
        # Alpha debería ser razonable (no perfecto, no terrible)
        assert -0.5 < alpha < 1.0

    def test_icc_in_range(self, simulation_data):
        rel = LattixReliability(simulation_data)
        icc = rel.icc()
        assert -1 <= icc <= 1

    def test_split_half(self, simulation_data):
        rel = LattixReliability(simulation_data)
        result = rel.split_half_reliability(n_splits=50)
        assert 0 < result["spearman_brown_reliability"] < 1
