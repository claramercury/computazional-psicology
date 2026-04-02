"""
Análisis estadístico extendido para simulación temporal Lattix v1.1.

Incluye:
- ANOVA de medidas repetidas (rendimiento a través de sesiones)
- Análisis de tendencia temporal (regresión lineal)
- Interacción emoción × memoria
- Correlación cruzada temporal
- Effect sizes temporales (primera vs última sesión)
"""

import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass
from typing import Dict, List, Tuple

from ..config import LattixVariable, VARIABLE_LABELS_EN


@dataclass
class TrendResult:
    """Resultado de análisis de tendencia temporal."""
    variable: str
    temporal_condition: str
    slope: float
    intercept: float
    r_squared: float
    p_value: float
    improvement_pct: float  # Mejora % de primera a última sesión


@dataclass
class TemporalEffectSize:
    """Effect size entre primera y última sesión."""
    variable: str
    temporal_condition: str
    mean_first: float
    mean_last: float
    cohens_d: float
    improvement_pct: float


class ExtendedStatistics:
    """Análisis estadístico temporal para Lattix v1.1."""

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.variables = [v.value for v in LattixVariable]

    def trend_analysis(self) -> List[TrendResult]:
        """
        Regresión lineal del rendimiento sobre el tiempo por condición.

        Para cada variable × condición temporal, ajusta y = mx + b
        donde x = sesión, y = media de la variable.
        """
        results = []

        if "session" not in self.data.columns:
            return results

        summary = (
            self.data.groupby(["session", "temporal_condition"])[self.variables]
            .mean()
            .reset_index()
        )

        for tc in summary["temporal_condition"].unique():
            tc_data = summary[summary["temporal_condition"] == tc]

            for var in self.variables:
                x = tc_data["session"].values
                y = tc_data[var].values

                if len(x) < 3:
                    continue

                slope, intercept, r_value, p_value, _ = stats.linregress(x, y)

                # Mejora porcentual
                y_first = intercept
                y_last = slope * x[-1] + intercept
                if abs(y_first) > 1e-8:
                    improvement = ((y_last - y_first) / abs(y_first)) * 100
                else:
                    improvement = 0.0

                results.append(TrendResult(
                    variable=var,
                    temporal_condition=tc,
                    slope=slope,
                    intercept=intercept,
                    r_squared=r_value ** 2,
                    p_value=p_value,
                    improvement_pct=improvement,
                ))

        return results

    def temporal_effect_sizes(
        self, n_sessions_window: int = 5,
    ) -> List[TemporalEffectSize]:
        """
        Cohen's d entre las primeras N y últimas N sesiones.

        Mide cuánto mejora cada variable con el tiempo.
        """
        results = []

        if "session" not in self.data.columns:
            return results

        max_session = self.data["session"].max()

        for tc in self.data["temporal_condition"].unique():
            tc_data = self.data[self.data["temporal_condition"] == tc]

            first = tc_data[tc_data["session"] < n_sessions_window]
            last = tc_data[tc_data["session"] > max_session - n_sessions_window]

            for var in self.variables:
                if first.empty or last.empty:
                    continue

                m1 = first[var].mean()
                m2 = last[var].mean()
                s1 = first[var].std()
                s2 = last[var].std()

                pooled_std = np.sqrt((s1**2 + s2**2) / 2)
                d = (m2 - m1) / pooled_std if pooled_std > 1e-8 else 0.0

                improvement = ((m2 - m1) / abs(m1) * 100) if abs(m1) > 1e-8 else 0.0

                results.append(TemporalEffectSize(
                    variable=var,
                    temporal_condition=tc,
                    mean_first=m1,
                    mean_last=m2,
                    cohens_d=d,
                    improvement_pct=improvement,
                ))

        return results

    def condition_comparison_anova(self) -> Dict[str, Dict]:
        """
        ANOVA comparando condiciones temporales en la última ventana de sesiones.

        Responde: ¿full_lattix supera significativamente a baseline?
        """
        results = {}

        if "session" not in self.data.columns:
            return results

        max_session = self.data["session"].max()
        late_data = self.data[self.data["session"] > max_session - 5]

        for var in self.variables:
            groups = []
            group_names = []
            for tc in sorted(late_data["temporal_condition"].unique()):
                tc_vals = late_data[late_data["temporal_condition"] == tc][var].values
                if len(tc_vals) > 0:
                    groups.append(tc_vals)
                    group_names.append(tc)

            if len(groups) < 2:
                continue

            f_stat, p_value = stats.f_oneway(*groups)
            grand_mean = np.concatenate(groups).mean()
            ss_between = sum(
                len(g) * (g.mean() - grand_mean) ** 2 for g in groups
            )
            ss_total = sum(np.sum((g - grand_mean) ** 2) for g in groups)
            eta_sq = ss_between / ss_total if ss_total > 0 else 0

            # Medias por condición
            means = {name: g.mean() for name, g in zip(group_names, groups)}

            results[var] = {
                "f_statistic": float(f_stat),
                "p_value": float(p_value),
                "eta_squared": float(eta_sq),
                "means": means,
                "significant": p_value < 0.05,
            }

        return results

    def emotion_performance_correlation(
        self, emotion_df: pd.DataFrame,
    ) -> Dict[str, Dict[str, float]]:
        """
        Correlación entre emociones y rendimiento Lattix.

        Agrega por sesión y calcula correlación de Pearson.
        """
        results = {}

        if emotion_df.empty or "session" not in self.data.columns:
            return results

        from ..config import EmotionDimension

        # Agregar rendimiento por sesión
        perf = (
            self.data[self.data["temporal_condition"] == "full_lattix"]
            .groupby("session")[self.variables]
            .mean()
            .reset_index()
        )

        # Agregar emociones por sesión
        emo_cols = [d.value for d in EmotionDimension]
        emo_agg = (
            emotion_df[emotion_df["temporal_condition"] == "full_lattix"]
            .groupby("session")[emo_cols]
            .mean()
            .reset_index()
        )

        merged = perf.merge(emo_agg, on="session")
        if len(merged) < 5:
            return results

        for emo in emo_cols:
            results[emo] = {}
            for var in self.variables:
                r, p = stats.pearsonr(merged[emo], merged[var])
                results[emo][var] = {"r": float(r), "p": float(p)}

        return results

    def full_report(
        self,
        emotion_df: pd.DataFrame = None,
    ) -> Dict:
        """Genera reporte estadístico completo."""
        report = {
            "trends": [
                {
                    "variable": t.variable,
                    "temporal_condition": t.temporal_condition,
                    "slope": t.slope,
                    "r_squared": t.r_squared,
                    "p_value": t.p_value,
                    "improvement_pct": t.improvement_pct,
                }
                for t in self.trend_analysis()
            ],
            "effect_sizes": [
                {
                    "variable": e.variable,
                    "temporal_condition": e.temporal_condition,
                    "mean_first": e.mean_first,
                    "mean_last": e.mean_last,
                    "cohens_d": e.cohens_d,
                    "improvement_pct": e.improvement_pct,
                }
                for e in self.temporal_effect_sizes()
            ],
            "condition_anova": self.condition_comparison_anova(),
        }

        if emotion_df is not None and not emotion_df.empty:
            report["emotion_correlation"] = self.emotion_performance_correlation(
                emotion_df
            )

        return report
