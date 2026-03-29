"""Tests para el motor de simulación."""

import numpy as np
import pytest

from lattix_sim.config import SimulationParams, LattixVariable
from lattix_sim.simulation.engine import LattixSimulation
from lattix_sim.simulation.irt import IRTModel


class TestLattixSimulation:
    """Verifica la simulación Monte Carlo completa."""

    def setup_method(self):
        self.params = SimulationParams(n_trials=200, seed=42)
        self.sim = LattixSimulation(self.params)

    def test_run_produces_dataframe(self):
        data = self.sim.run()
        assert len(data) > 0
        assert "trial_id" in data.columns
        assert "agent" in data.columns
        assert "condition" in data.columns

    def test_all_variables_present(self):
        data = self.sim.run()
        for var in LattixVariable:
            assert var.value in data.columns

    def test_three_conditions(self):
        data = self.sim.run()
        conditions = set(data["condition"].unique())
        assert conditions == {"solo", "dual", "triadic"}

    def test_reproducibility(self):
        """Misma seed → mismos resultados."""
        sim1 = LattixSimulation(SimulationParams(n_trials=50, seed=123))
        sim2 = LattixSimulation(SimulationParams(n_trials=50, seed=123))
        data1 = sim1.run()
        data2 = sim2.run()
        np.testing.assert_array_almost_equal(
            data1["choral_utility"].values,
            data2["choral_utility"].values,
        )

    def test_triadic_higher_choral_utility(self):
        """Condición triádica debe tener mayor utilidad coral que solo."""
        data = self.sim.run()
        solo_mean = data[data["condition"] == "solo"]["choral_utility"].mean()
        triadic_mean = data[data["condition"] == "triadic"]["choral_utility"].mean()
        assert triadic_mean > solo_mean

    def test_summary_table(self):
        self.sim.run()
        summary = self.sim.summary()
        assert "mean" in summary.columns
        assert "std" in summary.columns


class TestIRTModel:
    """Verifica el modelo IRT 2PL."""

    def setup_method(self):
        self.rng = np.random.default_rng(42)
        self.irt = IRTModel(n_items=10, rng=self.rng)

    def test_probability_range(self):
        """Probabilidades deben estar en (0, 1)."""
        for item in self.irt.items:
            for theta in np.linspace(-3, 3, 20):
                p = self.irt.probability(theta, item)
                assert 0 < p < 1

    def test_higher_theta_higher_probability(self):
        """Mayor theta → mayor probabilidad (monotonicidad)."""
        item = self.irt.items[0]
        p_low = self.irt.probability(-2.0, item)
        p_high = self.irt.probability(2.0, item)
        assert p_high > p_low

    def test_simulate_responses_shape(self):
        thetas = np.linspace(-2, 2, 50)
        responses = self.irt.simulate_responses(thetas)
        assert responses.shape == (50, 10)
        assert set(np.unique(responses)).issubset({0.0, 1.0})

    def test_information_positive(self):
        """Información del test debe ser positiva."""
        info = self.irt.test_information(0.0)
        assert info > 0

    def test_theta_estimation_correlation(self):
        """Thetas estimados deben correlacionar con los reales."""
        thetas = np.linspace(-2, 2, 100)
        responses = self.irt.simulate_responses(thetas)
        estimated = self.irt.estimate_theta(responses)
        from scipy.stats import pearsonr
        r, _ = pearsonr(thetas, estimated)
        assert r > 0.3  # Correlación razonable con solo 10 ítems
