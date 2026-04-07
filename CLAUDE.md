# CLAUDE.md — Computational Psychology / Lattix 01

AI assistant guide for the `claramercury/computazional-psicology` repository.

---

## Project Overview

**Lattix Sim** is a Python psychometric simulation framework implementing **Lattix 01**, a measurement instrument for multi-agent LLM coordination analysis. It models emergent behaviors in collaborative AI systems using Monte Carlo simulation, IRT, MANOVA, and factor analysis.

**Context:** The experiment on 26/03/2026 documented a case where Lumen (GPT-5.4) analyzed its own exclusion from a collaborative process and argued for reincorporation — evidence of *normative contextual agency*. This repository computationally validates that phenomenon.

**Author:** Clara Mercury (clara@the-latix-project.org) — The Latix Project  
**Publication target:** arXiv (cs.AI / cs.HC) → Behavior Research Methods / BMC Psychology  
**Version:** 0.1.0 (Alpha)  
**License:** MIT

---

## Git History (Recent)

| Date | Commit | Description |
|------|--------|-------------|
| 2026-03-29 | `470bf80` | Merge PR #1: Add Lattix 01 simulation framework (30 files, 2344 insertions) |
| 2026-03-29 | `11d0628` | Add Lattix 01 psychometric simulation framework (Claude) |
| 2026-03-28 | `804f146` | Add paper on multi-agent coordination experiment (claramercury) |
| 2026-03-28 | `41534e8` | Initial commit |

**Active branch for AI development:** `claude/add-claude-documentation-1aU6G`  
All changes must be committed and pushed to this branch.

---

## Directory Structure

```
computazional-psicology/
├── lattix_sim/                     # Main Python package
│   ├── __init__.py
│   ├── config.py                   # Central config: enums, agent profiles, condition modifiers
│   ├── models/
│   │   ├── agents.py               # Agent class: samples Beta/Poisson per condition
│   │   ├── conditions.py           # ConditionEngine: active agents + coupling per condition
│   │   └── psychometric.py         # LattixScorer: normalize, factor score, weighted index
│   ├── simulation/
│   │   ├── engine.py               # LattixSimulation: Monte Carlo loop
│   │   └── irt.py                  # IRTModel: 2PL item generation + MLE theta estimation
│   ├── analysis/
│   │   ├── statistics.py           # LattixStatistics: MANOVA, ANOVA, Tukey HSD, Bayesian
│   │   ├── factor_analysis.py      # LattixCFA: EFA with KMO, Bartlett, Varimax
│   │   └── reliability.py          # LattixReliability: Cronbach's α, split-half, ICC
│   ├── visualization/
│   │   ├── plots.py                # Violin/box/bar plots, effect sizes
│   │   ├── radar.py                # Radar charts for agent profiles
│   │   └── heatmaps.py             # Correlation and difference heatmaps
│   └── utils/
│       └── reproducibility.py      # Save results (CSV, JSON)
├── scripts/
│   ├── run_simulation.py           # Full pipeline: simulation → analysis → figures
│   └── generate_paper_figures.py   # Selective figure regeneration for publication
├── tests/
│   ├── test_psychometric.py        # Agent sampling, condition logic, scoring
│   ├── test_simulation.py          # Monte Carlo engine, IRT model
│   └── test_statistics.py          # Statistical tests verification
├── results/                        # Generated output (gitignored except .gitkeep)
│   └── .gitkeep
├── paperpsicometría                # Original experiment paper (text document)
├── pyproject.toml                  # Project metadata and dependencies
├── requirements.txt                # pip dependencies
├── README.md                       # Project documentation (bilingual: ES/EN)
└── CLAUDE.md                       # This file
```

---

## Installation & Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run full simulation pipeline (data + analysis + 10+ figures)
python scripts/run_simulation.py

# Optional CLI args
python scripts/run_simulation.py --trials 2000 --seed 123

# Regenerate only paper figures
python scripts/generate_paper_figures.py

# Run tests
python -m pytest tests/ -v
```

**Outputs** (written to `results/`):
- `simulation_data.csv` — raw Monte Carlo data (n_trials × agents × conditions × 5 variables)
- `summary_stats.json` — aggregated statistics
- `figures/*.png` and `figures/*.pdf` — publication-quality visualizations

---

## Core Domain Model

### 5 Lattix Psychometric Variables

Defined as `LattixVariable` enum in `config.py`:

| Code | Label (ES) | Label (EN) | Distribution |
|------|-----------|------------|-------------|
| `func_dist` | Reparto funcional | Functional Distribution | Beta(α, β) ∈ [0,1] |
| `enunc_stability` | Estabilidad enunciativa | Enunciative Stability | Beta(α, β) ∈ [0,1] |
| `gap_detection` | Detección de huecos | Gap Detection | Beta(α, β) ∈ [0,1] |
| `meta_proposals` | Meta-propuestas | Meta-proposals | Poisson(λ), discrete |
| `choral_utility` | Utilidad coral | Choral Utility | Beta(α, β) ∈ [0,1] |

### 3 Experimental Conditions

Defined as `Condition` enum in `config.py`:

| Code | Description | Key modifiers |
|------|-------------|---------------|
| `solo` | Single agent, isolated baseline | `choral_utility × 0.3`, no coupling |
| `dual` | Multi-agent without meta-analyst (Lumen excluded) | `coupling_weight = 0.15` |
| `triadic` | Full coordination with Lumen as meta-analyst/regulator | `choral_utility × 1.30`, `coupling_weight = 0.25` |

### 4 Agent Profiles

Calibrated from the 26/03/2026 experiment observations:

| Agent key | Name | Architecture | Distinctive trait |
|-----------|------|-------------|-------------------|
| `claude_code` | Claude Code | Opus 4.6 | High functional contribution, stable position |
| `sonnet` | Sonnet | Sonnet (web) | Adaptable, moderate-high utility |
| `gemini` | Gemini | Gemini | Strong gap detection (OSINT/crypto background) |
| `lumen` | Lumen | GPT-5.4 | Maximum meta-proposals, very high gap detection (analyzed own exclusion) |

**Active agents per condition:**
- `solo`: 1 agent (iterated for each separately)
- `dual`: claude_code + sonnet + gemini (Lumen excluded)
- `triadic`: all 4 agents including Lumen

---

## Architecture & Key Classes

### `config.py`
Central configuration. Contains all enums, dataclasses, and constants. Modify here to change agent profiles or condition modifiers — no changes needed elsewhere.

### `models/agents.py` — `Agent`
Samples the 5 variables from their base distributions, then applies `ConditionModifier` multiplicatively (Beta alpha) or additively (Poisson lambda). Inter-agent coupling is applied by `ConditionEngine`.

### `models/conditions.py` — `ConditionEngine`
Manages which agents are active per condition and applies coupling weights between agent samples.

### `models/psychometric.py` — `LattixScorer`
- Normalizes `meta_proposals` (Poisson → [0,1])
- Computes factor scores (weighted combination)
- Computes the overall **Lattix Index** (weighted sum of 5 variables)

### `simulation/engine.py` — `LattixSimulation`
Monte Carlo loop: iterates `n_trials` times across all conditions and agents, collecting a DataFrame with shape `(n_trials × n_conditions × n_agents, 5_variables + metadata)`.

### `simulation/irt.py` — `IRTModel`
2-Parameter Logistic (2PL) IRT model:
- Generates `n_irt_items` items with random difficulty (b) and discrimination (a)
- Simulates dichotomous responses from latent theta
- Estimates theta via Maximum Likelihood Estimation (MLE)
- Validates: `r(θ_real, θ_estimated) ≈ 0.90`

### `analysis/statistics.py` — `LattixStatistics`
- **MANOVA** (Pillai's Trace): multivariate test across conditions
- **ANOVA + Tukey HSD**: per-variable with Bonferroni correction
- **Bayesian directional test**: computes P(solo < dual < triadic) for each variable

### `analysis/factor_analysis.py` — `LattixCFA`
Exploratory Factor Analysis:
- KMO test (adequacy)
- Bartlett's test (sphericity)
- Varimax rotation
- Extracts 3-factor structure

### `analysis/reliability.py` — `LattixReliability`
- **Cronbach's α**
- **Split-half** with Spearman-Brown correction
- **ICC(3,1)** (intraclass correlation)

---

## Statistical Results (Validated)

| Metric | Value | Interpretation |
|--------|-------|---------------|
| MANOVA Pillai's Trace | 0.79, p < 0.001 | Strong multivariate separation between conditions |
| Choral utility η² | 0.52 | Large effect size |
| Choral utility d (triadic vs solo) | 2.19 | Very large Cohen's d |
| Meta-proposals η² | 0.42 | Large effect size |
| Meta-proposals d (triadic vs solo) | 1.84 | Very large Cohen's d |
| IRT r(θ_real, θ_estimated) | 0.90 | Strong latent trait recovery |

---

## Development Conventions

### Language
- **Code:** English (variable names, docstrings, comments)
- **Labels/UI strings:** Spanish (ES) with English equivalents in `VARIABLE_LABELS_EN`
- **Commits:** English
- **Paper/documentation:** Mixed ES/EN

### Python Style
- Python 3.10+ (uses `match`-compatible syntax, `|` union types)
- `dataclasses` with `frozen=True` for immutable configs
- `Enum` subclassing `str` for JSON-serializable enums
- Type hints throughout
- Docstrings in Spanish for domain classes, English acceptable for utilities

### Testing
- Framework: `pytest`
- Run: `python -m pytest tests/ -v`
- Key invariants that tests enforce:
  - Beta samples are always in [0, 1]
  - Poisson samples are always ≥ 0
  - Triadic condition produces higher `choral_utility` than Solo
  - IRT probability is monotonically increasing in theta
  - Simulation is reproducible with the same seed
- Do not break these invariants when modifying condition modifiers or agent profiles

### Adding New Features
- New psychometric variables → add to `LattixVariable` enum in `config.py`, add Beta/Poisson params to `AgentProfile`, update `ConditionModifier`
- New agents → add profile to `AGENT_PROFILES` dict in `config.py`, update `ConditionEngine` active agent lists
- New statistical tests → add methods to `LattixStatistics`, expose in `run_simulation.py`
- New visualizations → add to appropriate file in `visualization/`, call from `run_simulation.py`

### Results Directory
The `results/` directory is gitignored (except `.gitkeep`). Generated CSVs, JSONs, and figures are never committed. To reproduce results, run `python scripts/run_simulation.py`.

### No CI/CD
There are no GitHub Actions workflows, Makefile, or Docker configuration. Tests must be run manually.

---

## Important Domain Knowledge for AI Assistants

1. **The simulation is calibrated, not arbitrary.** Agent profiles in `config.py` were derived from qualitative observations of the 26/03/2026 experiment. Changes to Beta/Poisson parameters need domain justification.

2. **Lumen is special.** Lumen has `meta_proposals_lam = 5.0` (highest of all agents) and `gap_detection = Beta(8, 2)` (near-ceiling) because it empirically detected its own exclusion and proposed the 3-condition experimental design. This is the core phenomenon being modeled.

3. **Condition modifiers are intentionally asymmetric.** `choral_utility` gets `× 0.3` in Solo (minimal, no ensemble) but `× 1.30` in Triadic. This is theoretically motivated, not a bug.

4. **IRT is applied per-variable.** Each of the 5 Lattix variables gets its own IRT calibration with 20 items. The resulting theta estimates provide an alternative latent trait score alongside the direct Beta/Poisson scores.

5. **The paper (`paperpsicometría`) is the theoretical grounding.** When domain questions arise about why something is modeled a certain way, that document contains the rationale.

6. **Bilingual outputs.** Visualization labels use Spanish (`VARIABLE_LABELS`) by default for the paper. English labels (`VARIABLE_LABELS_EN`) are available for international submission variants.
