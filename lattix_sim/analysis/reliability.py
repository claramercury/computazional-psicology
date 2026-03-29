"""
Análisis de fiabilidad del instrumento Lattix 01.

Cronbach's alpha, split-half reliability, ICC.
"""

import numpy as np
import pandas as pd
from scipy import stats


class LattixReliability:
    """Métricas de fiabilidad del instrumento psicométrico Lattix."""

    VARIABLES = [
        "func_dist", "enunc_stability", "gap_detection",
        "meta_proposals", "choral_utility",
    ]

    def __init__(self, data: pd.DataFrame):
        self.data = data[self.VARIABLES].copy()
        # Normalizar meta_proposals
        mp_max = self.data["meta_proposals"].quantile(0.99)
        if mp_max > 0:
            self.data["meta_proposals"] = self.data["meta_proposals"] / mp_max

    def cronbachs_alpha(self) -> float:
        """
        Cronbach's alpha para consistencia interna.

        alpha = (k / (k-1)) * (1 - sum(var_i) / var_total)
        """
        k = len(self.VARIABLES)
        item_vars = self.data.var(axis=0, ddof=1)
        total_var = self.data.sum(axis=1).var(ddof=1)

        if total_var == 0:
            return 0.0

        alpha = (k / (k - 1)) * (1 - item_vars.sum() / total_var)
        return float(alpha)

    def split_half_reliability(self, n_splits: int = 100) -> dict:
        """
        Fiabilidad por mitades con corrección Spearman-Brown.

        Promedia sobre múltiples splits aleatorios.
        """
        rng = np.random.default_rng(42)
        correlations = []

        n_rows = len(self.data)
        indices = np.arange(n_rows)

        for _ in range(n_splits):
            rng.shuffle(indices)
            half = n_rows // 2
            first_half = self.data.iloc[indices[:half]].sum(axis=1)
            second_half = self.data.iloc[indices[half:2 * half]].sum(axis=1)
            r, _ = stats.pearsonr(first_half.values, second_half.values)
            correlations.append(r)

        mean_r = np.mean(correlations)
        # Corrección Spearman-Brown
        reliability = 2 * mean_r / (1 + mean_r) if (1 + mean_r) != 0 else 0.0

        return {
            "mean_split_half_r": float(mean_r),
            "spearman_brown_reliability": float(reliability),
            "std_split_half_r": float(np.std(correlations)),
        }

    def icc(self, icc_type: str = "ICC(3,1)") -> float:
        """
        Intraclass Correlation Coefficient (ICC 3,1).

        Mide la consistencia de las puntuaciones entre las variables
        como si fueran "calificadores" del mismo constructo.
        """
        data_matrix = self.data.values
        n, k = data_matrix.shape

        # Medias
        row_means = data_matrix.mean(axis=1)
        col_means = data_matrix.mean(axis=0)
        grand_mean = data_matrix.mean()

        # Sumas de cuadrados
        ss_rows = k * np.sum((row_means - grand_mean) ** 2)
        ss_cols = n * np.sum((col_means - grand_mean) ** 2)
        ss_total = np.sum((data_matrix - grand_mean) ** 2)
        ss_error = ss_total - ss_rows - ss_cols

        # Mean squares
        ms_rows = ss_rows / (n - 1) if n > 1 else 0
        ms_error = ss_error / ((n - 1) * (k - 1)) if (n - 1) * (k - 1) > 0 else 0
        ms_cols = ss_cols / (k - 1) if k > 1 else 0

        # ICC(3,1) — two-way mixed, consistency
        if ms_error == 0:
            return 1.0
        icc_value = (ms_rows - ms_error) / (ms_rows + (k - 1) * ms_error)
        return float(np.clip(icc_value, -1, 1))

    def item_total_correlations(self) -> dict:
        """Correlación ítem-total corregida para cada variable."""
        total = self.data.sum(axis=1)
        correlations = {}
        for var in self.VARIABLES:
            corrected_total = total - self.data[var]
            r, p = stats.pearsonr(self.data[var].values, corrected_total.values)
            correlations[var] = {"r": float(r), "p_value": float(p)}
        return correlations

    def full_report(self) -> dict:
        """Reporte completo de fiabilidad."""
        return {
            "cronbachs_alpha": self.cronbachs_alpha(),
            "split_half": self.split_half_reliability(),
            "icc": self.icc(),
            "item_total_correlations": self.item_total_correlations(),
        }
