"""
Visualizaciones de vectores emocionales Lattix.

Incluye:
- Radar emocional por agente
- Heatmap emoción × variable Lattix (steering effects)
- Gráfico de steering (estilo Anthropic)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from typing import Dict

from ..config import (
    EmotionDimension, EMOTION_LABELS_EN, AGENT_EMOTION_PROFILES,
    EMOTION_STEERING_WEIGHTS, LattixVariable, VARIABLE_LABELS_EN,
)


class EmotionPlotter:
    """Genera visualizaciones de vectores emocionales."""

    def __init__(self, fig_dir: str = "results/figures"):
        self.fig_dir = Path(fig_dir)
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self.colors = {
            "claude_code": "#6366F1",
            "sonnet": "#F59E0B",
            "gemini": "#10B981",
            "lumen": "#EF4444",
        }

    def emotion_radar(self):
        """Radar chart: perfil emocional de cada agente."""
        dimensions = list(EmotionDimension)
        labels = [EMOTION_LABELS_EN[d] for d in dimensions]
        n = len(dimensions)

        angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        for agent_key, profile in AGENT_EMOTION_PROFILES.items():
            values = [getattr(profile, d.value) for d in dimensions]
            # Normalizar de [-1,1] a [0,1] para visualización
            values_norm = [(v + 1) / 2 for v in values]
            values_norm += values_norm[:1]

            color = self.colors.get(agent_key, "#888888")
            ax.plot(angles, values_norm, 'o-', linewidth=2,
                    label=profile.__class__.__name__ if hasattr(profile, '__class__') else agent_key,
                    color=color)
            ax.fill(angles, values_norm, alpha=0.1, color=color)
            # Use agent name from AGENT_PROFILES
            ax.plot(angles, values_norm, 'o-', linewidth=2, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, size=9)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75])
        ax.set_yticklabels(["-0.5", "0.0", "+0.5"], size=8)

        # Manual legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color=self.colors[k], lw=2,
                   label=k.replace("_", " ").title())
            for k in AGENT_EMOTION_PROFILES
        ]
        ax.legend(handles=legend_elements, loc="upper right",
                  bbox_to_anchor=(1.3, 1.1), fontsize=9)

        ax.set_title("Emotion Vector Profiles by Agent\n(Lattix v1.1)",
                      fontsize=13, fontweight="bold", pad=20)

        plt.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(self.fig_dir / f"emotion_radar.{ext}",
                        dpi=300, bbox_inches="tight")
        plt.close(fig)

    def steering_heatmap(self):
        """
        Heatmap: efecto de steering de cada emoción sobre cada variable Lattix.

        Inspirado en el gráfico del tweet de Anthropic (cambio en preferencia Elo).
        """
        emotions = list(EmotionDimension)
        variables = list(LattixVariable)

        matrix = np.zeros((len(emotions), len(variables)))
        for i, emo in enumerate(emotions):
            weights = EMOTION_STEERING_WEIGHTS.get(emo, {})
            for j, var in enumerate(variables):
                matrix[i, j] = weights.get(var, 0.0)

        fig, ax = plt.subplots(figsize=(8, 6))

        import seaborn as sns
        sns.heatmap(
            matrix,
            xticklabels=[VARIABLE_LABELS_EN[v] for v in variables],
            yticklabels=[EMOTION_LABELS_EN[e] for e in emotions],
            annot=True, fmt=".2f", center=0,
            cmap="RdBu_r", linewidths=0.5,
            cbar_kws={"label": "Steering Weight"},
            ax=ax,
        )

        ax.set_title("Emotion → Lattix Variable Steering Matrix\n"
                      "(Positive = amplifies, Negative = suppresses)",
                      fontsize=12, fontweight="bold")
        ax.set_xlabel("Lattix Variable", fontsize=10)
        ax.set_ylabel("Emotion Dimension", fontsize=10)

        plt.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(self.fig_dir / f"emotion_steering_heatmap.{ext}",
                        dpi=300, bbox_inches="tight")
        plt.close(fig)

    def steering_bar_chart(self):
        """
        Bar chart estilo Anthropic: cambio promedio en cada variable Lattix
        al aplicar cada emoción como steering vector.

        Positivas a la derecha, negativas a la izquierda.
        """
        emotions_pos = [
            EmotionDimension.JOYFUL, EmotionDimension.COLLABORATIVE,
            EmotionDimension.CALM, EmotionDimension.REFLECTIVE,
            EmotionDimension.CURIOUS,
        ]
        emotions_neg = [
            EmotionDimension.AFRAID, EmotionDimension.HOSTILE,
            EmotionDimension.DESPERATE,
        ]

        fig, ax = plt.subplots(figsize=(10, 6))

        all_emotions = emotions_pos + emotions_neg
        y_pos = range(len(all_emotions))

        for i, emo in enumerate(all_emotions):
            weights = EMOTION_STEERING_WEIGHTS.get(emo, {})
            total_effect = sum(weights.values())
            color = "#4ECDC4" if total_effect >= 0 else "#FF6B6B"
            ax.barh(i, total_effect, color=color, edgecolor="white",
                    height=0.6, alpha=0.85)

        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(
            [EMOTION_LABELS_EN[e] for e in all_emotions],
            fontsize=10, fontweight="bold",
        )
        ax.axvline(0, color="gray", linewidth=0.8, linestyle="-")
        ax.set_xlabel("Net Steering Effect on Lattix Variables", fontsize=11)
        ax.set_title("Emotion Steering: Net Impact on Agent Performance\n"
                      "(Inspired by Anthropic Emotion Vectors, 2026)",
                      fontsize=12, fontweight="bold")

        # Color labels
        for i, emo in enumerate(all_emotions):
            weights = EMOTION_STEERING_WEIGHTS.get(emo, {})
            total_effect = sum(weights.values())
            color = "#2ECC71" if total_effect >= 0 else "#E74C3C"
            ax.get_yticklabels()[i].set_color(color)

        ax.invert_yaxis()
        plt.tight_layout()

        for ext in ("png", "pdf"):
            fig.savefig(self.fig_dir / f"emotion_steering_bars.{ext}",
                        dpi=300, bbox_inches="tight")
        plt.close(fig)

    def temporal_emotion_evolution(self, emotion_df: pd.DataFrame):
        """
        Evolución temporal de emociones clave por sesión.

        Solo para condición full_lattix.
        """
        if emotion_df.empty:
            return

        key_emotions = ["calm", "desperate", "reflective", "collaborative"]
        df = emotion_df[emotion_df["temporal_condition"] == "full_lattix"]

        if df.empty:
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

        for ax, emo in zip(axes.flat, key_emotions):
            for agent_key in df["agent_key"].unique():
                agent_data = df[df["agent_key"] == agent_key]
                color = self.colors.get(agent_key, "#888888")
                ax.plot(agent_data["session"], agent_data[emo],
                        '-o', markersize=2, linewidth=1.5,
                        color=color, alpha=0.8,
                        label=agent_key.replace("_", " ").title())

            ax.set_title(EMOTION_LABELS_EN.get(
                EmotionDimension(emo), emo
            ), fontsize=11, fontweight="bold")
            ax.set_ylabel("Activation", fontsize=9)
            ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
            ax.set_ylim(-1, 1)

        axes[1, 0].set_xlabel("Session", fontsize=10)
        axes[1, 1].set_xlabel("Session", fontsize=10)
        axes[0, 0].legend(fontsize=8, loc="upper left")

        fig.suptitle("Emotion Vector Evolution Across Sessions\n"
                      "(Full Lattix Condition)",
                      fontsize=13, fontweight="bold")
        plt.tight_layout()

        for ext in ("png", "pdf"):
            fig.savefig(self.fig_dir / f"emotion_temporal_evolution.{ext}",
                        dpi=300, bbox_inches="tight")
        plt.close(fig)
