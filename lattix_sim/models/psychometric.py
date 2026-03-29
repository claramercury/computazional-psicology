"""
Scoring psicométrico Lattix 01.

Transforma muestras brutas en puntuaciones normalizadas y computa
índices compuestos del framework.
"""

import numpy as np
import pandas as pd

from ..config import LattixVariable, VARIABLE_LABELS_EN


class LattixScorer:
    """Calcula puntuaciones compuestas e índices del framework Lattix."""

    # Estructura factorial hipotética
    FACTOR_STRUCTURE = {
        "Operational Competence": [
            LattixVariable.FUNC_DIST,
            LattixVariable.CHORAL_UTILITY,
        ],
        "Metacognitive Capacity": [
            LattixVariable.GAP_DETECTION,
            LattixVariable.META_PROPOSALS,
        ],
        "Positional Stability": [
            LattixVariable.ENUNC_STABILITY,
        ],
    }

    @staticmethod
    def normalize_meta_proposals(values: np.ndarray, max_clip: float = 15.0) -> np.ndarray:
        """Normaliza conteos de Poisson a [0, 1] para comparabilidad."""
        clipped = np.clip(values, 0, max_clip)
        return clipped / max_clip

    @classmethod
    def compute_factor_scores(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula puntuaciones por factor a partir de las 5 variables."""
        result = df.copy()
        for factor_name, variables in cls.FACTOR_STRUCTURE.items():
            cols = [v.value for v in variables]
            if LattixVariable.META_PROPOSALS in variables:
                # Normalizar meta_proposals antes de promediar
                temp = df[cols].copy()
                if "meta_proposals" in temp.columns:
                    temp["meta_proposals"] = cls.normalize_meta_proposals(
                        temp["meta_proposals"].values
                    )
                result[factor_name] = temp.mean(axis=1)
            else:
                result[factor_name] = df[cols].mean(axis=1)
        return result

    @staticmethod
    def compute_lattix_index(df: pd.DataFrame) -> pd.Series:
        """
        Índice compuesto Lattix: media ponderada de las 5 variables.

        Pesos derivados de la importancia teórica en el framework:
        - Utilidad coral y detección de huecos reciben mayor peso
          (son las variables más discriminantes entre condiciones).
        """
        weights = {
            LattixVariable.FUNC_DIST.value: 0.15,
            LattixVariable.ENUNC_STABILITY.value: 0.15,
            LattixVariable.GAP_DETECTION.value: 0.25,
            LattixVariable.META_PROPOSALS.value: 0.20,
            LattixVariable.CHORAL_UTILITY.value: 0.25,
        }
        normalized = df.copy()
        normalized["meta_proposals"] = LattixScorer.normalize_meta_proposals(
            normalized["meta_proposals"].values
        )
        score = sum(normalized[col] * w for col, w in weights.items())
        return score
