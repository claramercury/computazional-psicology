# Lattix 01 — Simulación Psicométrica Computacional

Simulación Monte Carlo del framework psicométrico **Lattix 01** para análisis de coordinación multiagente en sistemas LLM.

Parte del [The Latix Project](https://the-latix-project.org/).

## Contexto

Este repositorio implementa la simulación computacional del instrumento psicométrico Lattix 01, diseñado para medir variables emergentes en la coordinación entre múltiples arquitecturas LLM (Claude, Sonnet, Gemini, Lumen/GPT-5.4).

El experimento original (26/03/2026) documentó un caso de **negociación emergente de roles** donde un LLM (Lumen) analizó su propia exclusión de un proceso colaborativo y argumentó funcionalmente por su reincorporación — evidencia de agencia normativa contextual.

## Las 5 Variables Psicométricas Lattix

| Variable | Descripción |
|----------|-------------|
| **Reparto funcional** | Distribución de trabajo entre agentes |
| **Estabilidad enunciativa** | Consistencia del rol/posición de cada agente |
| **Detección de huecos** | Capacidad de identificar lo que falta |
| **Meta-propuestas** | Reflexión sobre el propio dispositivo experimental |
| **Utilidad coral** | Contribución real al resultado grupal |

## 3 Condiciones Experimentales

1. **Solo** — Arquitectura aislada (baseline)
2. **Dual** — Dos+ arquitecturas sin meta-analista (sesión 26/03)
3. **Triádico** — Con Lumen como meta-analista/regulador

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

```bash
# Simulación completa (datos + análisis + figuras)
python scripts/run_simulation.py

# Solo regenerar figuras para el paper
python scripts/generate_paper_figures.py

# Tests
python -m pytest tests/ -v
```

## Métodos Estadísticos

- **IRT 2PL** — Teoría de Respuesta al Ítem
- **MANOVA** — Comparación multivariada (Pillai's Trace)
- **ANOVA + Tukey HSD** — Comparaciones por variable con corrección Bonferroni
- **Test bayesiano direccional** — P(triádico > dual > solo)
- **Análisis Factorial** — Estructura de 3 factores (Varimax)
- **Fiabilidad** — Cronbach's α, split-half, ICC

## Resultados Clave

La simulación confirma que el framework Lattix discrimina entre condiciones:

- **MANOVA**: Pillai's Trace = 0.79, p < 0.001
- **Utilidad coral**: η² = 0.52 (efecto grande), triádico >> solo (d = 2.19)
- **Meta-propuestas**: η² = 0.42 (efecto grande), triádico >> solo (d = 1.84)
- **IRT**: r(θ_real, θ_estimado) = 0.90

## Estructura

```
lattix_sim/
├── config.py              # Perfiles de agentes y parámetros
├── models/                # Agentes, variables, condiciones
├── simulation/            # Motor Monte Carlo + IRT
├── analysis/              # Estadística, factorial, fiabilidad
└── visualization/         # Plots, radar, heatmaps
```

## Publicación

- **Target**: arXiv (cs.AI / cs.HC) → Behavior Research Methods / BMC Psychology
- **Título provisional**: "Emergent Role Negotiation in Multi-Architecture LLM Collaboration: A Case Study in Reverse Engineering"
