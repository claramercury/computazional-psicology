"""
Radar charts (spider charts) para perfiles de agentes Lattix.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from ..config import LattixVariable, VARIABLE_LABELS_EN, AGENT_PROFILES
from ..models.psychometric import LattixScorer


AGENT_COLORS = {
    "Claude Code": "#4C72B0",
    "Sonnet": "#DD8452",
    "Gemini": "#55A868",
    "Lumen": "#C44E52",
}


class RadarPlotter:
    """Genera radar charts de perfiles psicométricos de agentes."""

    def __init__(self, data: pd.DataFrame, output_dir: str = "results/figures"):
        self.data = data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _normalize_for_radar(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normaliza todas las variables a [0, 1] para el radar."""
        result = df.copy()
        for col in [v.value for v in LattixVariable]:
            col_max = result[col].max()
            if col_max > 0:
                result[col] = result[col] / col_max
        return result

    def agent_profiles(self, condition: str = "triadic", save: bool = True) -> plt.Figure:
        """Radar chart comparando perfiles de agentes en una condición."""
        subset = self.data[self.data["condition"] == condition]
        if subset.empty:
            raise ValueError(f"No hay datos para condición '{condition}'")

        variables = [v.value for v in LattixVariable]
        labels = [VARIABLE_LABELS_EN[v] for v in LattixVariable]

        means = subset.groupby("agent")[variables].mean()
        normalized = self._normalize_for_radar(means)

        angles = np.linspace(0, 2 * np.pi, len(variables), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        for agent_name in normalized.index:
            values = normalized.loc[agent_name, variables].values.tolist()
            values += values[:1]
            color = AGENT_COLORS.get(agent_name, "#888888")
            ax.plot(angles, values, "o-", linewidth=2, label=agent_name, color=color)
            ax.fill(angles, values, alpha=0.1, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.set_title(
            f"Agent Psychometric Profiles — {condition.capitalize()} Condition",
            fontsize=13, fontweight="bold", pad=20,
        )
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
        fig.tight_layout()

        if save:
            fig.savefig(
                self.output_dir / f"radar_agents_{condition}.png",
                dpi=300, bbox_inches="tight",
            )
            fig.savefig(
                self.output_dir / f"radar_agents_{condition}.pdf",
                bbox_inches="tight",
            )

        return fig

    def agent_across_conditions(self, agent_name: str = "Lumen", save: bool = True) -> plt.Figure:
        """Radar de un mismo agente en las 3 condiciones."""
        subset = self.data[self.data["agent"] == agent_name]
        if subset.empty:
            raise ValueError(f"No hay datos para agente '{agent_name}'")

        variables = [v.value for v in LattixVariable]
        labels = [VARIABLE_LABELS_EN[v] for v in LattixVariable]

        means = subset.groupby("condition")[variables].mean()
        normalized = self._normalize_for_radar(means)

        angles = np.linspace(0, 2 * np.pi, len(variables), endpoint=False).tolist()
        angles += angles[:1]

        cond_colors = {"solo": "#4C72B0", "dual": "#DD8452", "triadic": "#55A868"}

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        for cond in ["solo", "dual", "triadic"]:
            if cond not in normalized.index:
                continue
            values = normalized.loc[cond, variables].values.tolist()
            values += values[:1]
            ax.plot(angles, values, "o-", linewidth=2, label=cond.capitalize(),
                    color=cond_colors[cond])
            ax.fill(angles, values, alpha=0.1, color=cond_colors[cond])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.set_title(
            f"{agent_name} — Profile Across Conditions",
            fontsize=13, fontweight="bold", pad=20,
        )
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
        fig.tight_layout()

        if save:
            safe_name = agent_name.lower().replace(" ", "_")
            fig.savefig(
                self.output_dir / f"radar_{safe_name}_conditions.png",
                dpi=300, bbox_inches="tight",
            )
            fig.savefig(
                self.output_dir / f"radar_{safe_name}_conditions.pdf",
                bbox_inches="tight",
            )

        return fig
