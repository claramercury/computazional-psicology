"""
Heatmaps de correlaciones e interacciones Lattix.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

from ..config import LattixVariable, VARIABLE_LABELS_EN


class HeatmapPlotter:
    """Genera heatmaps de correlación e interacción."""

    def __init__(self, data: pd.DataFrame, output_dir: str = "results/figures"):
        self.data = data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def correlation_matrix(self, save: bool = True) -> plt.Figure:
        """Heatmap de correlaciones entre las 5 variables Lattix."""
        variables = [v.value for v in LattixVariable]
        labels = [VARIABLE_LABELS_EN[v] for v in LattixVariable]

        corr = self.data[variables].corr()

        fig, ax = plt.subplots(figsize=(8, 7))
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

        sns.heatmap(
            corr, mask=mask, annot=True, fmt=".2f",
            cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            xticklabels=labels, yticklabels=labels,
            square=True, ax=ax,
        )
        ax.set_title(
            "Lattix 01 — Variable Correlation Matrix",
            fontsize=13, fontweight="bold",
        )
        fig.tight_layout()

        if save:
            fig.savefig(self.output_dir / "correlation_matrix.png", dpi=300, bbox_inches="tight")
            fig.savefig(self.output_dir / "correlation_matrix.pdf", bbox_inches="tight")

        return fig

    def agent_variable_heatmap(self, save: bool = True) -> plt.Figure:
        """Heatmap agente × variable, facetado por condición."""
        variables = [v.value for v in LattixVariable]
        labels = [VARIABLE_LABELS_EN[v] for v in LattixVariable]
        conditions = ["solo", "dual", "triadic"]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

        for ax, cond in zip(axes, conditions):
            subset = self.data[self.data["condition"] == cond]
            means = subset.groupby("agent")[variables].mean()

            # Normalizar por columna para comparabilidad visual
            normalized = means.copy()
            for col in variables:
                col_max = normalized[col].max()
                if col_max > 0:
                    normalized[col] = normalized[col] / col_max

            sns.heatmap(
                normalized, annot=means.round(3).values, fmt="",
                cmap="YlOrRd", vmin=0, vmax=1,
                xticklabels=labels, yticklabels=means.index,
                ax=ax,
            )
            ax.set_title(f"{cond.capitalize()}", fontsize=12, fontweight="bold")
            ax.set_xlabel("")

        fig.suptitle(
            "Lattix 01 — Agent × Variable Scores by Condition",
            fontsize=14, fontweight="bold", y=1.02,
        )
        fig.tight_layout()

        if save:
            fig.savefig(self.output_dir / "agent_variable_heatmap.png", dpi=300, bbox_inches="tight")
            fig.savefig(self.output_dir / "agent_variable_heatmap.pdf", bbox_inches="tight")

        return fig

    def condition_difference_heatmap(self, save: bool = True) -> plt.Figure:
        """
        Heatmap de diferencias: (triádico - solo) por agente × variable.

        Muestra el efecto de la condición colaborativa.
        """
        variables = [v.value for v in LattixVariable]
        labels = [VARIABLE_LABELS_EN[v] for v in LattixVariable]

        solo_means = self.data[self.data["condition"] == "solo"].groupby("agent")[variables].mean()
        triadic_means = self.data[self.data["condition"] == "triadic"].groupby("agent")[variables].mean()

        # Alinear índices
        common_agents = solo_means.index.intersection(triadic_means.index)
        diff = triadic_means.loc[common_agents] - solo_means.loc[common_agents]

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(
            diff, annot=True, fmt=".3f",
            cmap="RdYlGn", center=0,
            xticklabels=labels, yticklabels=diff.index,
            ax=ax,
        )
        ax.set_title(
            "Lattix 01 — Triadic vs Solo Difference (Δ scores)",
            fontsize=13, fontweight="bold",
        )
        fig.tight_layout()

        if save:
            fig.savefig(self.output_dir / "condition_difference.png", dpi=300, bbox_inches="tight")
            fig.savefig(self.output_dir / "condition_difference.pdf", bbox_inches="tight")

        return fig
