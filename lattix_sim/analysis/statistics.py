"""
Análisis estadístico para comparación de condiciones Lattix.

ANOVA, MANOVA (via Pillai's trace), tamaños de efecto, Tukey HSD,
y comparaciones bayesianas simplificadas.
"""

import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations
from dataclasses import dataclass


@dataclass
class ANOVAResult:
    variable: str
    f_statistic: float
    p_value: float
    eta_squared: float
    condition_means: dict


@dataclass
class PostHocResult:
    variable: str
    group1: str
    group2: str
    mean_diff: float
    cohens_d: float
    t_statistic: float
    p_value: float


@dataclass
class MANOVAResult:
    pillai_trace: float
    approx_f: float
    p_value: float
    df_hypothesis: int
    df_error: int


class LattixStatistics:
    """Análisis estadístico del framework Lattix 01."""

    VARIABLES = [
        "func_dist", "enunc_stability", "gap_detection",
        "meta_proposals", "choral_utility",
    ]

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.conditions = sorted(data["condition"].unique())

    def run_anova(self, variable: str) -> ANOVAResult:
        """One-way ANOVA para una variable entre condiciones."""
        groups = [
            self.data[self.data["condition"] == c][variable].values
            for c in self.conditions
        ]

        f_stat, p_val = stats.f_oneway(*groups)

        # Eta-squared
        grand_mean = self.data[variable].mean()
        ss_between = sum(
            len(g) * (g.mean() - grand_mean) ** 2 for g in groups
        )
        ss_total = sum((self.data[variable] - grand_mean) ** 2)
        eta_sq = ss_between / ss_total if ss_total > 0 else 0.0

        means = {
            c: self.data[self.data["condition"] == c][variable].mean()
            for c in self.conditions
        }

        return ANOVAResult(
            variable=variable,
            f_statistic=f_stat,
            p_value=p_val,
            eta_squared=eta_sq,
            condition_means=means,
        )

    def run_all_anova(self) -> list:
        """ANOVA para todas las variables Lattix."""
        return [self.run_anova(v) for v in self.VARIABLES]

    def posthoc_tukey(self, variable: str) -> list:
        """Comparaciones post-hoc por pares con corrección Bonferroni."""
        results = []
        pairs = list(combinations(self.conditions, 2))
        n_comparisons = len(pairs)

        for c1, c2 in pairs:
            g1 = self.data[self.data["condition"] == c1][variable].values
            g2 = self.data[self.data["condition"] == c2][variable].values

            t_stat, p_val = stats.ttest_ind(g1, g2)
            # Corrección Bonferroni
            p_corrected = min(1.0, p_val * n_comparisons)

            # Cohen's d
            pooled_std = np.sqrt(
                ((len(g1) - 1) * g1.std() ** 2 + (len(g2) - 1) * g2.std() ** 2)
                / (len(g1) + len(g2) - 2)
            )
            d = (g1.mean() - g2.mean()) / pooled_std if pooled_std > 0 else 0.0

            results.append(PostHocResult(
                variable=variable,
                group1=c1,
                group2=c2,
                mean_diff=g1.mean() - g2.mean(),
                cohens_d=d,
                t_statistic=t_stat,
                p_value=p_corrected,
            ))

        return results

    def run_manova(self) -> MANOVAResult:
        """
        MANOVA simplificada usando Pillai's trace.

        Compara el vector de 5 variables entre las 3 condiciones.
        """
        groups = []
        for c in self.conditions:
            subset = self.data[self.data["condition"] == c][self.VARIABLES].values
            groups.append(subset)

        n_total = sum(len(g) for g in groups)
        k = len(groups)  # n condiciones
        p = len(self.VARIABLES)  # n variables

        # Media global
        all_data = np.vstack(groups)
        grand_mean = all_data.mean(axis=0)

        # Matrices SSCP
        # H (between-groups)
        H = np.zeros((p, p))
        for g in groups:
            diff = g.mean(axis=0) - grand_mean
            H += len(g) * np.outer(diff, diff)

        # E (within-groups)
        E = np.zeros((p, p))
        for g in groups:
            centered = g - g.mean(axis=0)
            E += centered.T @ centered

        # Pillai's trace = tr(H @ inv(H + E))
        HE = H + E
        try:
            HE_inv = np.linalg.inv(HE)
            pillai = np.trace(H @ HE_inv)
        except np.linalg.LinAlgError:
            pillai = 0.0

        # Aproximación F
        s = min(k - 1, p)
        m = (abs(p - k + 1) - 1) / 2
        n_param = (n_total - k - p - 1) / 2

        df_hyp = s * max(p, k - 1)
        df_err = s * (n_total - k - p + s)

        if df_err > 0 and s > 0:
            approx_f = (pillai / s) / ((1 - pillai / s)) * (df_err / df_hyp)
            p_value = 1 - stats.f.cdf(approx_f, df_hyp, df_err)
        else:
            approx_f = 0.0
            p_value = 1.0

        return MANOVAResult(
            pillai_trace=pillai,
            approx_f=approx_f,
            p_value=p_value,
            df_hypothesis=int(df_hyp),
            df_error=int(df_err),
        )

    def bayesian_directional_test(
        self, variable: str, order: list = None
    ) -> dict:
        """
        Test bayesiano simplificado para la hipótesis direccional:
        triadic > dual > solo.

        Usa el porcentaje de muestras bootstrap que cumplen el orden.
        """
        if order is None:
            order = ["solo", "dual", "triadic"]

        n_bootstrap = 5000
        rng = np.random.default_rng(42)
        count_order_holds = 0

        group_data = {
            c: self.data[self.data["condition"] == c][variable].values
            for c in order
        }

        for _ in range(n_bootstrap):
            means = []
            for c in order:
                sample = rng.choice(group_data[c], size=len(group_data[c]), replace=True)
                means.append(sample.mean())
            # Verificar orden estricto
            if all(means[i] < means[i + 1] for i in range(len(means) - 1)):
                count_order_holds += 1

        posterior_prob = count_order_holds / n_bootstrap

        return {
            "variable": variable,
            "hypothesis": " < ".join(order),
            "posterior_probability": posterior_prob,
            "bayes_factor": posterior_prob / (1 - posterior_prob) if posterior_prob < 1 else float("inf"),
        }

    def full_report(self) -> dict:
        """Genera el reporte estadístico completo."""
        anova_results = self.run_all_anova()
        manova = self.run_manova()

        posthoc = {}
        bayesian = {}
        for v in self.VARIABLES:
            posthoc[v] = self.posthoc_tukey(v)
            bayesian[v] = self.bayesian_directional_test(v)

        return {
            "manova": manova,
            "anova": anova_results,
            "posthoc": posthoc,
            "bayesian": bayesian,
        }
