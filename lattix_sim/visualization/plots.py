"""
Visualizaciones de comparación entre condiciones experimentales.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

from ..config import (
    LattixVariable, VARIABLE_LABELS_EN, CONDITION_LABELS, Condition,
)

# Estilo publicación
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})

PALETTE = {"solo": "#4C72B0", "dual": "#DD8452", "triadic": "#55A868"}


class ConditionPlotter:
    """Genera gráficos de comparación entre condiciones Lattix."""

    def __init__(self, data: pd.DataFrame, output_dir: str = "results/figures"):
        self.data = data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def violin_plots(self, save: bool = True) -> plt.Figure:
        """Violin plots: una variable por panel, 3 condiciones."""
        variables = list(LattixVariable)
        fig, axes = plt.subplots(1, 5, figsize=(18, 4.5), sharey=False)

        for ax, var in zip(axes, variables):
            sns.violinplot(
                data=self.data, x="condition", y=var.value,
                order=["solo", "dual", "triadic"],
                palette=PALETTE, inner="quartile", ax=ax,
            )
            ax.set_title(VARIABLE_LABELS_EN[var], fontweight="bold")
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_xticklabels(["Solo", "Dual", "Triadic"])

        fig.suptitle(
            "Lattix 01 — Variable Distribution by Experimental Condition",
            fontsize=14, fontweight="bold", y=1.02,
        )
        fig.tight_layout()

        if save:
            fig.savefig(self.output_dir / "violin_conditions.png", dpi=300, bbox_inches="tight")
            fig.savefig(self.output_dir / "violin_conditions.pdf", bbox_inches="tight")

        return fig

    def box_plots(self, save: bool = True) -> plt.Figure:
        """Box plots con swarm overlay para cada variable."""
        variables = list(LattixVariable)
        fig, axes = plt.subplots(1, 5, figsize=(18, 4.5), sharey=False)

        for ax, var in zip(axes, variables):
            sns.boxplot(
                data=self.data, x="condition", y=var.value,
                order=["solo", "dual", "triadic"],
                palette=PALETTE, ax=ax, fliersize=2,
            )
            ax.set_title(VARIABLE_LABELS_EN[var], fontweight="bold")
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_xticklabels(["Solo", "Dual", "Triadic"])

        fig.suptitle(
            "Lattix 01 — Condition Comparison (Box Plots)",
            fontsize=14, fontweight="bold", y=1.02,
        )
        fig.tight_layout()

        if save:
            fig.savefig(self.output_dir / "box_conditions.png", dpi=300, bbox_inches="tight")
            fig.savefig(self.output_dir / "box_conditions.pdf", bbox_inches="tight")

        return fig

    def bar_summary(self, save: bool = True) -> plt.Figure:
        """Barras con error bars: media ± 95% CI por condición."""
        variables = [v.value for v in LattixVariable]
        conditions = ["solo", "dual", "triadic"]

        means = self.data.groupby("condition")[variables].mean()
        sems = self.data.groupby("condition")[variables].sem()

        x = np.arange(len(variables))
        width = 0.25

        fig, ax = plt.subplots(figsize=(12, 5))

        for i, cond in enumerate(conditions):
            offset = (i - 1) * width
            ax.bar(
                x + offset,
                means.loc[cond, variables],
                width,
                yerr=1.96 * sems.loc[cond, variables],
                label=CONDITION_LABELS[Condition(cond)],
                color=PALETTE[cond],
                capsize=3,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [VARIABLE_LABELS_EN[LattixVariable(v)] for v in variables],
            rotation=15, ha="right",
        )
        ax.set_ylabel("Mean Score")
        ax.set_title(
            "Lattix 01 — Mean Scores by Condition (±95% CI)",
            fontweight="bold",
        )
        ax.legend()
        fig.tight_layout()

        if save:
            fig.savefig(self.output_dir / "bar_summary.png", dpi=300, bbox_inches="tight")
            fig.savefig(self.output_dir / "bar_summary.pdf", bbox_inches="tight")

        return fig

    def effect_size_plot(self, posthoc_results: dict, save: bool = True) -> plt.Figure:
        """Visualiza Cohen's d para cada comparación por pares."""
        rows = []
        for var, comparisons in posthoc_results.items():
            for comp in comparisons:
                rows.append({
                    "variable": VARIABLE_LABELS_EN[LattixVariable(var)],
                    "comparison": f"{comp.group1} vs {comp.group2}",
                    "cohens_d": abs(comp.cohens_d),
                    "significant": comp.p_value < 0.05,
                })

        df = pd.DataFrame(rows)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(
            data=df, x="cohens_d", y="variable", hue="comparison",
            ax=ax, palette="viridis",
        )

        # Líneas de referencia para tamaños de efecto
        ax.axvline(0.2, color="gray", ls="--", alpha=0.5, label="Small (0.2)")
        ax.axvline(0.5, color="gray", ls="-.", alpha=0.5, label="Medium (0.5)")
        ax.axvline(0.8, color="gray", ls=":", alpha=0.5, label="Large (0.8)")

        ax.set_xlabel("|Cohen's d|")
        ax.set_title("Effect Sizes by Variable and Comparison", fontweight="bold")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        fig.tight_layout()

        if save:
            fig.savefig(self.output_dir / "effect_sizes.png", dpi=300, bbox_inches="tight")
            fig.savefig(self.output_dir / "effect_sizes.pdf", bbox_inches="tight")

        return fig
