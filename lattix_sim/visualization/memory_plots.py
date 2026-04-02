"""
Visualizaciones del sistema de memoria Lattix.

Incluye:
- Curvas de aprendizaje (rendimiento por sesión)
- Crecimiento de memoria por capa (stacked area)
- Métricas de consolidación
- Comparación temporal (baseline vs memory vs full_lattix)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from ..config import LattixVariable, VARIABLE_LABELS_EN


class MemoryPlotter:
    """Genera visualizaciones del sistema de memoria."""

    def __init__(self, fig_dir: str = "results/figures"):
        self.fig_dir = Path(fig_dir)
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self.condition_colors = {
            "baseline_no_memory": "#95A5A6",
            "memory_no_consolidation": "#F39C12",
            "full_lattix": "#2ECC71",
        }
        self.condition_labels = {
            "baseline_no_memory": "Sin memoria",
            "memory_no_consolidation": "Memoria sin consolidar",
            "full_lattix": "Lattix completo",
        }

    def learning_curves(self, data: pd.DataFrame):
        """
        Curvas de aprendizaje: media de choral_utility por sesión.

        Compara las 3 condiciones temporales.
        """
        if "session" not in data.columns:
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Panel 1: Choral Utility
        key_var = "choral_utility"
        summary = (
            data.groupby(["session", "temporal_condition"])[key_var]
            .mean()
            .reset_index()
        )

        ax = axes[0]
        for tc, group in summary.groupby("temporal_condition"):
            color = self.condition_colors.get(tc, "#888888")
            label = self.condition_labels.get(tc, tc)
            ax.plot(group["session"], group[key_var],
                    linewidth=2, color=color, label=label)

        ax.set_xlabel("Session (day)", fontsize=10)
        ax.set_ylabel("Mean Choral Utility", fontsize=10)
        ax.set_title("Learning Curve: Choral Utility", fontsize=12,
                      fontweight="bold")
        ax.legend(fontsize=9)

        # Panel 2: Gap Detection
        key_var2 = "gap_detection"
        summary2 = (
            data.groupby(["session", "temporal_condition"])[key_var2]
            .mean()
            .reset_index()
        )

        ax = axes[1]
        for tc, group in summary2.groupby("temporal_condition"):
            color = self.condition_colors.get(tc, "#888888")
            label = self.condition_labels.get(tc, tc)
            ax.plot(group["session"], group[key_var2],
                    linewidth=2, color=color, label=label)

        ax.set_xlabel("Session (day)", fontsize=10)
        ax.set_ylabel("Mean Gap Detection", fontsize=10)
        ax.set_title("Learning Curve: Gap Detection", fontsize=12,
                      fontweight="bold")
        ax.legend(fontsize=9)

        fig.suptitle("Temporal Learning Curves — Impact of Memory & Consolidation",
                      fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()

        for ext in ("png", "pdf"):
            fig.savefig(self.fig_dir / f"learning_curves.{ext}",
                        dpi=300, bbox_inches="tight")
        plt.close(fig)

    def all_variables_learning(self, data: pd.DataFrame):
        """
        Grid de curvas de aprendizaje para todas las 5 variables Lattix.
        """
        if "session" not in data.columns:
            return

        variables = [v.value for v in LattixVariable]
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        axes_flat = axes.flat

        for i, var in enumerate(variables):
            ax = axes_flat[i]
            summary = (
                data.groupby(["session", "temporal_condition"])[var]
                .mean()
                .reset_index()
            )

            for tc, group in summary.groupby("temporal_condition"):
                color = self.condition_colors.get(tc, "#888888")
                label = self.condition_labels.get(tc, tc)
                ax.plot(group["session"], group[var],
                        linewidth=1.8, color=color, label=label)

            ax.set_title(
                VARIABLE_LABELS_EN.get(LattixVariable(var), var),
                fontsize=10, fontweight="bold",
            )
            ax.set_xlabel("Session", fontsize=8)
            if i == 0:
                ax.legend(fontsize=7)

        # Ocultar el subplot sobrante
        axes_flat[5].set_visible(False)

        fig.suptitle("All Lattix Variables — Temporal Evolution by Condition",
                      fontsize=13, fontweight="bold")
        plt.tight_layout()

        for ext in ("png", "pdf"):
            fig.savefig(self.fig_dir / f"all_variables_learning.{ext}",
                        dpi=300, bbox_inches="tight")
        plt.close(fig)

    def memory_growth(self, memory_df: pd.DataFrame):
        """
        Stacked area: crecimiento de memoria por capa sobre el tiempo.
        """
        if memory_df.empty or "session" not in memory_df.columns:
            return

        # Usar solo post-consolidation si existe
        if "phase" in memory_df.columns:
            df = memory_df[memory_df["phase"] == "post_consolidation"]
        else:
            df = memory_df

        if df.empty:
            return

        fig, ax = plt.subplots(figsize=(10, 5))

        layers = ["episodic_count", "hypothesis_count",
                   "semantic_count", "procedural_count"]
        layer_labels = ["Episodic", "Hypothesis", "Semantic", "Procedural"]
        colors = ["#3498DB", "#E67E22", "#2ECC71", "#9B59B6"]

        sessions = sorted(df["session"].unique())
        layer_data = []
        for layer in layers:
            values = []
            for s in sessions:
                s_data = df[df["session"] == s]
                values.append(s_data[layer].mean() if not s_data.empty else 0)
            layer_data.append(values)

        ax.stackplot(sessions, *layer_data, labels=layer_labels,
                      colors=colors, alpha=0.8)

        ax.set_xlabel("Session (day)", fontsize=10)
        ax.set_ylabel("Memory Entries", fontsize=10)
        ax.set_title("Memory Growth by Layer\n(Full Lattix Condition)",
                      fontsize=12, fontweight="bold")
        ax.legend(loc="upper left", fontsize=9)

        plt.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(self.fig_dir / f"memory_growth.{ext}",
                        dpi=300, bbox_inches="tight")
        plt.close(fig)

    def consolidation_metrics(self, consolidation_df: pd.DataFrame):
        """
        Métricas de consolidación nocturna por sesión.
        """
        if consolidation_df.empty:
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        metrics = ["promoted", "lessons_extracted",
                    "contradictions_found", "deduplicated"]
        titles = ["Knowledge Promoted", "Lessons Extracted",
                   "Contradictions Found", "Entries Deduplicated"]
        colors = ["#2ECC71", "#3498DB", "#E74C3C", "#F39C12"]

        for ax, metric, title, color in zip(axes.flat, metrics, titles, colors):
            ax.bar(consolidation_df["session"],
                   consolidation_df[metric],
                   color=color, alpha=0.7, width=0.8)
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.set_xlabel("Session", fontsize=9)
            ax.set_ylabel("Count", fontsize=9)

        fig.suptitle("Nightly Consolidation Metrics (Auto-Dream Cycle)",
                      fontsize=13, fontweight="bold")
        plt.tight_layout()

        for ext in ("png", "pdf"):
            fig.savefig(self.fig_dir / f"consolidation_metrics.{ext}",
                        dpi=300, bbox_inches="tight")
        plt.close(fig)

    def memory_quality_evolution(self, memory_df: pd.DataFrame):
        """
        Evolución de la calidad de memoria y tasa de contradicciones.
        """
        if memory_df.empty:
            return

        if "phase" in memory_df.columns:
            df = memory_df[memory_df["phase"] == "post_consolidation"]
        else:
            df = memory_df

        if df.empty:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Quality
        ax1.plot(df["session"], df["memory_quality"],
                 'o-', color="#2ECC71", linewidth=2, markersize=3)
        ax1.set_xlabel("Session", fontsize=10)
        ax1.set_ylabel("Memory Quality", fontsize=10)
        ax1.set_title("Memory Quality Over Time", fontsize=12,
                        fontweight="bold")
        ax1.set_ylim(0, 1)
        ax1.axhline(0.5, color="gray", linestyle="--", linewidth=0.8,
                     label="Neutral baseline")
        ax1.legend(fontsize=9)

        # Contradictions
        ax2.plot(df["session"], df["contradiction_rate"],
                 'o-', color="#E74C3C", linewidth=2, markersize=3)
        ax2.set_xlabel("Session", fontsize=10)
        ax2.set_ylabel("Contradiction Rate", fontsize=10)
        ax2.set_title("Contradiction Rate Over Time", fontsize=12,
                        fontweight="bold")
        ax2.set_ylim(0, max(0.2, df["contradiction_rate"].max() * 1.2))

        fig.suptitle("Memory System Health Indicators",
                      fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()

        for ext in ("png", "pdf"):
            fig.savefig(self.fig_dir / f"memory_quality_evolution.{ext}",
                        dpi=300, bbox_inches="tight")
        plt.close(fig)
