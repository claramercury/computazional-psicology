#!/usr/bin/env python3
"""
Lattix v1.1 — Simulación Extendida con Emociones y Memoria.

Ejecuta el pipeline temporal: emociones → memoria → consolidación →
análisis estadístico → visualizaciones.

Uso:
    python scripts/run_extended_simulation.py [--sessions N] [--trials T] [--seed S]
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lattix_sim.config import (
    TemporalParams, Condition, LattixVariable, VARIABLE_LABELS_EN,
    EmotionDimension, EMOTION_LABELS_EN,
)
from lattix_sim.simulation.temporal_engine import (
    TemporalSimulation, TemporalCondition, ALL_TEMPORAL_CONDITIONS,
)
from lattix_sim.analysis.extended_statistics import ExtendedStatistics
from lattix_sim.visualization.emotion_plots import EmotionPlotter
from lattix_sim.visualization.memory_plots import MemoryPlotter
from lattix_sim.utils.reproducibility import save_results


def print_header(text: str):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def print_section(text: str):
    print(f"\n--- {text} ---")


def main():
    parser = argparse.ArgumentParser(
        description="Lattix v1.1 — Extended Temporal Simulation"
    )
    parser.add_argument("--sessions", type=int, default=30,
                        help="Number of sessions (days)")
    parser.add_argument("--trials", type=int, default=100,
                        help="Trials per session per condition")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    t_params = TemporalParams(
        n_sessions=args.sessions,
        n_trials_per_session=args.trials,
        seed=args.seed,
    )

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    # =====================================================================
    # 1. SIMULACIÓN TEMPORAL
    # =====================================================================
    print_header("LATTIX v1.1 — SIMULACIÓN TEMPORAL EXTENDIDA")
    print(f"Parámetros: {t_params.n_sessions} sesiones, "
          f"{t_params.n_trials_per_session} trials/sesión, seed={t_params.seed}")

    sim = TemporalSimulation(
        temporal_params=t_params,
        experimental_condition=Condition.TRIADIC,
    )

    print("\nEjecutando 3 condiciones temporales...")
    data = sim.run_all_conditions()
    print(f"  Dataset total: {len(data)} observaciones")
    print(f"  Condiciones temporales: {', '.join(data['temporal_condition'].unique())}")

    # Guardar datos
    data.to_csv(output_dir / "temporal_data.csv", index=False)
    print("  → temporal_data.csv")

    # Guardar datos de emociones
    emotion_df = sim.emotion_data
    if not emotion_df.empty:
        emotion_df.to_csv(output_dir / "emotion_data.csv", index=False)
        print(f"  → emotion_data.csv ({len(emotion_df)} registros)")

    # Guardar métricas de memoria
    memory_df = sim.memory_metrics
    if not memory_df.empty:
        memory_df.to_csv(output_dir / "memory_metrics.csv", index=False)
        print(f"  → memory_metrics.csv ({len(memory_df)} registros)")

    # Guardar métricas de consolidación
    cons_df = sim.consolidation_metrics
    if not cons_df.empty:
        cons_df.to_csv(output_dir / "consolidation_metrics.csv", index=False)
        print(f"  → consolidation_metrics.csv ({len(cons_df)} registros)")

    # =====================================================================
    # 2. RESUMEN DESCRIPTIVO
    # =====================================================================
    print_header("RESUMEN POR CONDICIÓN TEMPORAL")

    summary = sim.session_summary(data)
    variables = [v.value for v in LattixVariable]

    # Comparar primera vs última sesión
    for tc in ALL_TEMPORAL_CONDITIONS:
        tc_data = summary[summary["temporal_condition"] == tc]
        if tc_data.empty:
            continue

        first = tc_data[tc_data["session"] < 5]
        last = tc_data[tc_data["session"] >= t_params.n_sessions - 5]

        print(f"\n  {tc}:")
        for var in variables:
            m_first = first[var].mean()
            m_last = last[var].mean()
            change = ((m_last - m_first) / abs(m_first) * 100) if abs(m_first) > 1e-8 else 0
            label = VARIABLE_LABELS_EN.get(LattixVariable(var), var)
            arrow = "↑" if change > 1 else "↓" if change < -1 else "→"
            print(f"    {label:30s}  {m_first:.3f} → {m_last:.3f}  {arrow} {change:+.1f}%")

    # =====================================================================
    # 3. ANÁLISIS ESTADÍSTICO EXTENDIDO
    # =====================================================================
    print_header("ANÁLISIS ESTADÍSTICO TEMPORAL")

    ext_stats = ExtendedStatistics(data)

    # Tendencias
    print_section("Tendencias temporales (regresión lineal)")
    trends = ext_stats.trend_analysis()
    for t in trends:
        if t.temporal_condition == "full_lattix":
            label = VARIABLE_LABELS_EN.get(LattixVariable(t.variable), t.variable)
            sig = "***" if t.p_value < 0.001 else "**" if t.p_value < 0.01 else "*" if t.p_value < 0.05 else "ns"
            print(f"  {label:30s}  slope={t.slope:+.5f}  R²={t.r_squared:.3f}  "
                  f"p={t.p_value:.4f} {sig}  Δ={t.improvement_pct:+.1f}%")

    # Effect sizes
    print_section("Effect sizes temporales (primeras 5 vs últimas 5 sesiones)")
    effects = ext_stats.temporal_effect_sizes()
    for e in effects:
        if e.temporal_condition == "full_lattix":
            label = VARIABLE_LABELS_EN.get(LattixVariable(e.variable), e.variable)
            d_str = f"d={e.cohens_d:+.3f}"
            print(f"  {label:30s}  {e.mean_first:.3f} → {e.mean_last:.3f}  "
                  f"{d_str}  Δ={e.improvement_pct:+.1f}%")

    # ANOVA de condiciones
    print_section("ANOVA: comparación entre condiciones temporales (últimas 5 sesiones)")
    anova = ext_stats.condition_comparison_anova()
    for var, result in anova.items():
        label = VARIABLE_LABELS_EN.get(LattixVariable(var), var)
        sig = "***" if result["p_value"] < 0.001 else "**" if result["p_value"] < 0.01 else "*" if result["p_value"] < 0.05 else "ns"
        print(f"  {label:30s}  F={result['f_statistic']:8.2f}  "
              f"p={result['p_value']:.2e}  η²={result['eta_squared']:.3f} {sig}")

    # Correlación emoción-rendimiento
    if not emotion_df.empty:
        print_section("Correlación emoción × rendimiento (full_lattix)")
        corr = ext_stats.emotion_performance_correlation(emotion_df)
        key_pairs = [
            ("desperate", "choral_utility"),
            ("calm", "enunc_stability"),
            ("reflective", "gap_detection"),
            ("collaborative", "choral_utility"),
        ]
        for emo, var in key_pairs:
            if emo in corr and var in corr[emo]:
                r = corr[emo][var]["r"]
                p = corr[emo][var]["p"]
                sig = "*" if p < 0.05 else "ns"
                emo_label = EMOTION_LABELS_EN.get(EmotionDimension(emo), emo)
                var_label = VARIABLE_LABELS_EN.get(LattixVariable(var), var)
                print(f"  {emo_label:15s} × {var_label:25s}  r={r:+.3f}  p={p:.4f} {sig}")

    # =====================================================================
    # 4. VISUALIZACIONES
    # =====================================================================
    print_header("GENERANDO VISUALIZACIONES")

    # Emociones (estáticas - no necesitan datos temporales)
    emo_plotter = EmotionPlotter(str(fig_dir))
    emo_plotter.emotion_radar()
    print("  → emotion_radar.png/pdf")
    emo_plotter.steering_heatmap()
    print("  → emotion_steering_heatmap.png/pdf")
    emo_plotter.steering_bar_chart()
    print("  → emotion_steering_bars.png/pdf")

    if not emotion_df.empty:
        emo_plotter.temporal_emotion_evolution(emotion_df)
        print("  → emotion_temporal_evolution.png/pdf")

    # Memoria
    mem_plotter = MemoryPlotter(str(fig_dir))
    mem_plotter.learning_curves(data)
    print("  → learning_curves.png/pdf")
    mem_plotter.all_variables_learning(data)
    print("  → all_variables_learning.png/pdf")

    if not memory_df.empty:
        mem_plotter.memory_growth(memory_df)
        print("  → memory_growth.png/pdf")
        mem_plotter.memory_quality_evolution(memory_df)
        print("  → memory_quality_evolution.png/pdf")

    if not cons_df.empty:
        mem_plotter.consolidation_metrics(cons_df)
        print("  → consolidation_metrics.png/pdf")

    plt.close("all")

    # =====================================================================
    # 5. GUARDAR REPORTE
    # =====================================================================
    print_header("GUARDANDO REPORTE")

    report = ext_stats.full_report(emotion_df)
    report["simulation_params"] = {
        "n_sessions": t_params.n_sessions,
        "n_trials_per_session": t_params.n_trials_per_session,
        "seed": t_params.seed,
        "experimental_condition": "triadic",
    }

    filepath = save_results(report, filename="extended_stats.json")
    print(f"  → {filepath}")

    # =====================================================================
    # RESUMEN FINAL
    # =====================================================================
    print_header("SIMULACIÓN v1.1 COMPLETA")
    print("Archivos generados:")
    print("  Datos: temporal_data.csv, emotion_data.csv, memory_metrics.csv")
    print("  Análisis: extended_stats.json")
    print(f"  Figuras: {sum(1 for f in fig_dir.iterdir())} archivos en results/figures/")
    print()

    # Conclusión clave
    full_effects = [e for e in effects if e.temporal_condition == "full_lattix"]
    baseline_effects = [e for e in effects if e.temporal_condition == "baseline_no_memory"]

    if full_effects and baseline_effects:
        full_choral = next((e for e in full_effects if e.variable == "choral_utility"), None)
        base_choral = next((e for e in baseline_effects if e.variable == "choral_utility"), None)

        if full_choral and base_choral:
            print(f"  HALLAZGO CLAVE:")
            print(f"  Full Lattix mejora choral_utility {full_choral.improvement_pct:+.1f}% "
                  f"(d={full_choral.cohens_d:+.3f})")
            print(f"  Baseline sin memoria: {base_choral.improvement_pct:+.1f}% "
                  f"(d={base_choral.cohens_d:+.3f})")


if __name__ == "__main__":
    main()
