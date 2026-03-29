"""
Motor de simulación Monte Carlo Lattix.

Genera el dataset completo: n_trials × agentes × condiciones × 5 variables.
"""

import numpy as np
import pandas as pd

from ..config import (
    Condition, LattixVariable, SimulationParams, DEFAULT_PARAMS,
)
from ..models.conditions import ConditionEngine


class LattixSimulation:
    """Simulación Monte Carlo del framework psicométrico Lattix 01."""

    def __init__(self, params: SimulationParams = None):
        self.params = params or DEFAULT_PARAMS
        self.rng = np.random.default_rng(self.params.seed)
        self.engine = ConditionEngine(self.rng)
        self._data = None

    def run(self) -> pd.DataFrame:
        """
        Ejecuta la simulación completa.

        Retorna DataFrame con columnas:
            trial_id, agent, agent_key, condition,
            func_dist, enunc_stability, gap_detection,
            meta_proposals, choral_utility
        """
        rows = []
        conditions = list(Condition)

        for condition in conditions:
            samples = self.engine.sample_condition(
                condition, self.params.n_trials
            )

            for agent_key, var_samples in samples.items():
                agent = self.engine.agents[agent_key]
                for trial_idx in range(self.params.n_trials):
                    row = {
                        "trial_id": trial_idx,
                        "agent": agent.name,
                        "agent_key": agent_key,
                        "condition": condition.value,
                    }
                    for var in LattixVariable:
                        row[var.value] = var_samples[var][trial_idx]
                    rows.append(row)

        self._data = pd.DataFrame(rows)
        return self._data

    @property
    def data(self) -> pd.DataFrame:
        if self._data is None:
            raise RuntimeError("Ejecuta .run() antes de acceder a los datos.")
        return self._data

    def summary(self) -> pd.DataFrame:
        """Tabla resumen: media ± std por agente × condición × variable."""
        df = self.data
        variables = [v.value for v in LattixVariable]

        grouped = df.groupby(["condition", "agent"])[variables]
        means = grouped.mean()
        stds = grouped.std()

        summary_parts = []
        for var in variables:
            part = pd.DataFrame({
                "variable": var,
                "mean": means[var],
                "std": stds[var],
            })
            summary_parts.append(part)

        return pd.concat(summary_parts).reset_index()
