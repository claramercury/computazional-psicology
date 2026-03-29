"""
Análisis Factorial Confirmatorio (CFA) para el framework Lattix 01.

Verifica la estructura de 3 factores:
  - Competencia Operativa: func_dist, choral_utility
  - Capacidad Metacognitiva: gap_detection, meta_proposals
  - Estabilidad Posicional: enunc_stability
"""

import numpy as np
import pandas as pd
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo


class LattixCFA:
    """Análisis factorial del instrumento Lattix 01."""

    VARIABLES = [
        "func_dist", "enunc_stability", "gap_detection",
        "meta_proposals", "choral_utility",
    ]

    def __init__(self, data: pd.DataFrame):
        self.data = data[self.VARIABLES].copy()
        # Normalizar meta_proposals para que sea comparable
        mp_max = self.data["meta_proposals"].quantile(0.99)
        if mp_max > 0:
            self.data["meta_proposals"] = self.data["meta_proposals"] / mp_max

    def bartlett_test(self) -> dict:
        """Test de esfericidad de Bartlett."""
        chi2, p_value = calculate_bartlett_sphericity(self.data)
        return {"chi_square": chi2, "p_value": p_value}

    def kmo_test(self) -> dict:
        """Kaiser-Meyer-Olkin measure of sampling adequacy."""
        kmo_per_var, kmo_total = calculate_kmo(self.data)
        return {
            "kmo_total": kmo_total,
            "kmo_per_variable": dict(zip(self.VARIABLES, kmo_per_var)),
        }

    def run_efa(self, n_factors: int = 3) -> dict:
        """
        Análisis Factorial Exploratorio.

        Confirma que 3 factores capturan la estructura de las 5 variables.
        """
        fa = FactorAnalyzer(n_factors=n_factors, rotation="varimax")
        fa.fit(self.data)

        loadings = pd.DataFrame(
            fa.loadings_,
            index=self.VARIABLES,
            columns=[f"Factor {i+1}" for i in range(n_factors)],
        )

        variance = fa.get_factor_variance()
        eigenvalues, _ = fa.get_eigenvalues()

        communalities = fa.get_communalities()

        return {
            "loadings": loadings,
            "variance_explained": {
                "ss_loadings": variance[0].tolist(),
                "proportion_variance": variance[1].tolist(),
                "cumulative_variance": variance[2].tolist(),
            },
            "eigenvalues": eigenvalues.tolist(),
            "communalities": dict(zip(self.VARIABLES, communalities)),
        }

    def goodness_of_fit(self) -> dict:
        """
        Métricas de ajuste del modelo factorial.

        Incluye varianza explicada y residuales.
        """
        fa = FactorAnalyzer(n_factors=3, rotation="varimax")
        fa.fit(self.data)

        # Correlación reproducida vs observada
        reproduced = fa.loadings_ @ fa.loadings_.T
        observed = self.data.corr().values
        residuals = observed - reproduced
        np.fill_diagonal(residuals, 0)

        rmsr = np.sqrt(np.mean(residuals ** 2))

        # Proporción de residuales > 0.05
        n_off_diag = len(self.VARIABLES) * (len(self.VARIABLES) - 1) / 2
        n_large = np.sum(np.abs(residuals[np.triu_indices_from(residuals, k=1)]) > 0.05)
        prop_large = n_large / n_off_diag if n_off_diag > 0 else 0

        return {
            "rmsr": rmsr,
            "proportion_large_residuals": prop_large,
            "residual_matrix": pd.DataFrame(
                residuals, index=self.VARIABLES, columns=self.VARIABLES
            ),
        }

    def full_report(self) -> dict:
        """Reporte completo del análisis factorial."""
        return {
            "bartlett": self.bartlett_test(),
            "kmo": self.kmo_test(),
            "efa": self.run_efa(),
            "fit": self.goodness_of_fit(),
        }
