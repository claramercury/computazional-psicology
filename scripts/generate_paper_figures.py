#!/usr/bin/env python3
"""
Genera todas las figuras para el paper de arXiv.

Requiere haber ejecutado run_simulation.py primero.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")

import pandas as pd
from lattix_sim.analysis.statistics import LattixStatistics
from lattix_sim.visualization.plots import ConditionPlotter
from lattix_sim.visualization.radar import RadarPlotter
from lattix_sim.visualization.heatmaps import HeatmapPlotter


def main():
    data_path = Path("results/simulation_data.csv")
    if not data_path.exists():
        print("ERROR: Ejecuta primero 'python scripts/run_simulation.py'")
        sys.exit(1)

    data = pd.read_csv(data_path)
    fig_dir = "results/figures"

    print("Generando figuras para el paper...")

    # Condition comparisons
    plotter = ConditionPlotter(data, fig_dir)
    plotter.violin_plots()
    plotter.box_plots()
    plotter.bar_summary()

    stats = LattixStatistics(data)
    report = stats.full_report()
    plotter.effect_size_plot(report["posthoc"])

    # Agent profiles
    radar = RadarPlotter(data, fig_dir)
    for cond in ["solo", "dual", "triadic"]:
        try:
            radar.agent_profiles(cond)
        except ValueError:
            pass
    for agent in ["Lumen", "Claude Code", "Sonnet", "Gemini"]:
        try:
            radar.agent_across_conditions(agent)
        except ValueError:
            pass

    # Heatmaps
    heatmap = HeatmapPlotter(data, fig_dir)
    heatmap.correlation_matrix()
    heatmap.agent_variable_heatmap()
    heatmap.condition_difference_heatmap()

    import matplotlib.pyplot as plt
    plt.close("all")

    print(f"Figuras guardadas en {fig_dir}/")


if __name__ == "__main__":
    main()
