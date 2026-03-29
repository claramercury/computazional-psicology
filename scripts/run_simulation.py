#!/usr/bin/env python3
"""
Lattix 01 — Simulación Psicométrica Completa.

Ejecuta el pipeline: generación de datos → análisis estadístico →
visualizaciones → reporte.

Uso:
    python scripts/run_simulation.py [--trials N] [--seed S]
"""

import sys
import argparse
from pathlib import Path

# Asegurar que el paquete sea importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")  # Backend no interactivo

from lattix_sim.config import SimulationParams, LattixVariable, VARIABLE_LABELS_EN
from lattix_sim.simulation.engine import LattixSimulation
from lattix_sim.simulation.irt import IRTModel
from lattix_sim.analysis.statistics import LattixStatistics
from lattix_sim.analysis.factor_analysis import LattixCFA
from lattix_sim.analysis.reliability import LattixReliability
from lattix_sim.visualization.plots import ConditionPlotter
from lattix_sim.visualization.radar import RadarPlotter
from lattix_sim.visualization.heatmaps import HeatmapPlotter
from lattix_sim.utils.reproducibility import save_results


def print_header(text: str):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def print_section(text: str):
    print(f"\n--- {text} ---")


def main():
    parser = argparse.ArgumentParser(description="Lattix 01 Psychometric Simulation")
    parser.add_argument("--trials", type=int, default=1000, help="Number of trials per condition")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    params = SimulationParams(n_trials=args.trials, seed=args.seed)

    # =====================================================================
    # 1. SIMULACIÓN MONTE CARLO
    # =====================================================================
    print_header("LATTIX 01 — SIMULACIÓN PSICOMÉTRICA")
    print(f"Parámetros: {params.n_trials} trials, seed={params.seed}")

    sim = LattixSimulation(params)
    data = sim.run()

    print(f"\nDataset generado: {len(data)} observaciones")
    print(f"  Agentes: {data['agent'].nunique()} ({', '.join(data['agent'].unique())})")
    print(f"  Condiciones: {', '.join(data['condition'].unique())}")

    # Guardar CSV
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    data.to_csv(output_dir / "simulation_data.csv", index=False)
    print(f"  → Datos guardados en results/simulation_data.csv")

    # Resumen
    print_section("Resumen descriptivo")
    summary = sim.summary()
    print(summary.to_string(index=False))

    # =====================================================================
    # 2. ANÁLISIS ESTADÍSTICO
    # =====================================================================
    print_header("ANÁLISIS ESTADÍSTICO")

    stats_analyzer = LattixStatistics(data)

    # MANOVA
    print_section("MANOVA (Pillai's Trace)")
    manova = stats_analyzer.run_manova()
    print(f"  Pillai's Trace = {manova.pillai_trace:.4f}")
    print(f"  Approx F({manova.df_hypothesis}, {manova.df_error}) = {manova.approx_f:.2f}")
    print(f"  p = {manova.p_value:.2e}")

    # ANOVA por variable
    print_section("ANOVA por variable")
    anova_results = stats_analyzer.run_all_anova()
    for result in anova_results:
        label = VARIABLE_LABELS_EN.get(LattixVariable(result.variable), result.variable)
        sig = "***" if result.p_value < 0.001 else "**" if result.p_value < 0.01 else "*" if result.p_value < 0.05 else "ns"
        print(f"  {label:30s}  F = {result.f_statistic:8.2f}  p = {result.p_value:.2e}  η² = {result.eta_squared:.3f}  {sig}")

    # Post-hoc
    print_section("Post-hoc (Bonferroni)")
    report = stats_analyzer.full_report()
    for var, comparisons in report["posthoc"].items():
        label = VARIABLE_LABELS_EN.get(LattixVariable(var), var)
        print(f"\n  {label}:")
        for comp in comparisons:
            sig = "*" if comp.p_value < 0.05 else "ns"
            print(f"    {comp.group1:8s} vs {comp.group2:8s}  Δ = {comp.mean_diff:+.3f}  d = {comp.cohens_d:+.3f}  p = {comp.p_value:.4f} {sig}")

    # Bayesiano
    print_section("Test direccional bayesiano (solo < dual < triadic)")
    for var, result in report["bayesian"].items():
        label = VARIABLE_LABELS_EN.get(LattixVariable(var), var)
        bf = result["bayes_factor"]
        bf_str = f"{bf:.2f}" if bf < 1000 else "∞"
        print(f"  {label:30s}  P(H₁) = {result['posterior_probability']:.3f}  BF = {bf_str}")

    # =====================================================================
    # 3. IRT
    # =====================================================================
    print_header("MODELO IRT 2PL")
    irt = IRTModel(n_items=params.n_irt_items, rng=sim.rng)

    # Usar choral_utility como ejemplo de variable latente
    thetas_raw = data[data["condition"] == "triadic"]["choral_utility"].values[:200]
    # Transformar a escala theta: z-score
    thetas = (thetas_raw - thetas_raw.mean()) / (thetas_raw.std() + 1e-8)

    responses = irt.simulate_responses(thetas)
    estimated = irt.estimate_theta(responses)

    from scipy.stats import pearsonr
    r, _ = pearsonr(thetas, estimated)
    print(f"  Ítems: {len(irt.items)}")
    print(f"  Correlación theta real vs estimado: r = {r:.3f}")
    print(f"  Info máxima del test: {max(irt.test_information(t) for t in thetas):.2f}")

    # =====================================================================
    # 4. ANÁLISIS FACTORIAL
    # =====================================================================
    print_header("ANÁLISIS FACTORIAL")
    try:
        cfa = LattixCFA(data)
        fa_report = cfa.full_report()

        print_section("Test de Bartlett")
        print(f"  χ² = {fa_report['bartlett']['chi_square']:.2f}, p = {fa_report['bartlett']['p_value']:.2e}")

        print_section("KMO")
        print(f"  KMO total = {fa_report['kmo']['kmo_total']:.3f}")

        print_section("Cargas factoriales (Varimax)")
        print(fa_report["efa"]["loadings"].to_string())

        print_section("Varianza explicada")
        cumvar = fa_report["efa"]["variance_explained"]["cumulative_variance"]
        print(f"  Varianza acumulada (3 factores): {cumvar[-1]:.1%}")

        print_section("Ajuste del modelo")
        fit = fa_report["fit"]
        print(f"  RMSR = {fit['rmsr']:.4f}")
        print(f"  Residuales > 0.05: {fit['proportion_large_residuals']:.1%}")
    except Exception as e:
        print(f"  [WARN] Factor analysis: {e}")

    # =====================================================================
    # 5. FIABILIDAD
    # =====================================================================
    print_header("FIABILIDAD DEL INSTRUMENTO")
    rel = LattixReliability(data)
    rel_report = rel.full_report()

    print(f"  Cronbach's α = {rel_report['cronbachs_alpha']:.3f}")
    print(f"  Split-half (Spearman-Brown) = {rel_report['split_half']['spearman_brown_reliability']:.3f}")
    print(f"  ICC(3,1) = {rel_report['icc']:.3f}")

    print_section("Correlaciones ítem-total corregidas")
    for var, vals in rel_report["item_total_correlations"].items():
        label = VARIABLE_LABELS_EN.get(LattixVariable(var), var)
        print(f"  {label:30s}  r = {vals['r']:.3f}  p = {vals['p_value']:.2e}")

    # =====================================================================
    # 6. VISUALIZACIONES
    # =====================================================================
    print_header("GENERANDO VISUALIZACIONES")

    fig_dir = "results/figures"

    plotter = ConditionPlotter(data, fig_dir)
    plotter.violin_plots()
    print("  → violin_conditions.png/pdf")
    plotter.box_plots()
    print("  → box_conditions.png/pdf")
    plotter.bar_summary()
    print("  → bar_summary.png/pdf")
    plotter.effect_size_plot(report["posthoc"])
    print("  → effect_sizes.png/pdf")

    radar = RadarPlotter(data, fig_dir)
    for cond in ["solo", "dual", "triadic"]:
        try:
            radar.agent_profiles(cond)
            print(f"  → radar_agents_{cond}.png/pdf")
        except ValueError:
            pass

    for agent in ["Lumen", "Claude Code"]:
        try:
            radar.agent_across_conditions(agent)
            safe = agent.lower().replace(" ", "_")
            print(f"  → radar_{safe}_conditions.png/pdf")
        except ValueError:
            pass

    heatmap = HeatmapPlotter(data, fig_dir)
    heatmap.correlation_matrix()
    print("  → correlation_matrix.png/pdf")
    heatmap.agent_variable_heatmap()
    print("  → agent_variable_heatmap.png/pdf")
    heatmap.condition_difference_heatmap()
    print("  → condition_difference.png/pdf")

    plt.close("all")

    # =====================================================================
    # 7. GUARDAR RESULTADOS
    # =====================================================================
    print_header("GUARDANDO RESULTADOS")

    results_to_save = {
        "simulation_params": {
            "n_trials": params.n_trials,
            "seed": params.seed,
            "n_agents": data["agent"].nunique(),
            "n_conditions": data["condition"].nunique(),
        },
        "manova": {
            "pillai_trace": manova.pillai_trace,
            "f_statistic": manova.approx_f,
            "p_value": manova.p_value,
        },
        "anova": {
            r.variable: {
                "f_statistic": r.f_statistic,
                "p_value": r.p_value,
                "eta_squared": r.eta_squared,
                "means": r.condition_means,
            }
            for r in anova_results
        },
        "bayesian": report["bayesian"],
        "irt": {
            "n_items": len(irt.items),
            "theta_correlation": float(r),
        },
        "reliability": rel_report,
    }

    filepath = save_results(results_to_save)
    print(f"  → {filepath}")

    print_header("SIMULACIÓN COMPLETA")
    print("Todos los resultados guardados en results/")
    print("Figuras en results/figures/")


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    main()
