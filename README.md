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

## v1.1: Vectores Emocionales + Memoria Persistente

### Vectores Emocionales (inspirado en Anthropic Emotion Vectors, 2026)

8 dimensiones emocionales que causan comportamiento en los agentes:

| Emoción | Efecto sobre rendimiento |
|---------|-------------------------|
| **Calm** | Aumenta estabilidad enunciativa |
| **Reflective** | Mejora detección de huecos y meta-propuestas |
| **Desperate** | Reduce utilidad coral (reward hacking) |
| **Curious** | Mejora detección y reparto funcional |
| **Collaborative** | Aumenta utilidad coral y reparto funcional |
| **Hostile** | Reduce utilidad coral y desacopla agentes |
| **Joyful** | Efecto positivo leve en engagement |
| **Afraid** | Reduce exploración y meta-propuestas |

### Memoria Persistente (4 capas)

| Capa | Propósito | TTL |
|------|-----------|-----|
| **Episódica** | Eventos, comandos, resultados | Corto |
| **Semántica** | Conocimiento validado | Largo |
| **Procedimental** | Workflows estables | Sin TTL |
| **Hipótesis** | Hipótesis activas con evidencia | Medio |

### Consolidación Nocturna (Auto-Dream)

Ciclo de aprendizaje sin reentrenamiento de pesos:
1. Ingesta de entradas episódicas
2. Deduplicación semántica
3. Extracción de lecciones
4. Scoring de calidad
5. Promoción a conocimiento validado
6. Detección de contradicciones
7. Decaimiento temporal

### Condiciones Temporales

1. **baseline_no_memory** — Sin memoria entre sesiones
2. **memory_no_consolidation** — Memoria cruda sin consolidación
3. **full_lattix** — Memoria + consolidación + emociones

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

```bash
# Simulación v1.0 (datos + análisis + figuras)
python scripts/run_simulation.py

# Simulación v1.1 temporal (emociones + memoria + consolidación)
python scripts/run_extended_simulation.py

# Solo regenerar figuras para el paper
python scripts/generate_paper_figures.py

# Tests
python -m pytest tests/ -v
```

## Métodos Estadísticos

### v1.0
- **IRT 2PL** — Teoría de Respuesta al Ítem
- **MANOVA** — Comparación multivariada (Pillai's Trace)
- **ANOVA + Tukey HSD** — Comparaciones por variable con corrección Bonferroni
- **Test bayesiano direccional** — P(triádico > dual > solo)
- **Análisis Factorial** — Estructura de 3 factores (Varimax)
- **Fiabilidad** — Cronbach's alpha, split-half, ICC

### v1.1 (temporal)
- **Análisis de tendencia** — Regresión lineal sobre mejora temporal
- **Effect sizes temporales** — Cohen's d primera vs última ventana
- **ANOVA de condiciones temporales** — baseline vs memory vs full_lattix
- **Correlación emoción x rendimiento** — Pearson entre vectores emocionales y variables Lattix

## Resultados Clave

### v1.0
- **MANOVA**: Pillai's Trace = 0.79, p < 0.001
- **Utilidad coral**: eta-sq = 0.52 (efecto grande), triádico >> solo (d = 2.19)
- **Meta-propuestas**: eta-sq = 0.42 (efecto grande), triádico >> solo (d = 1.84)
- **IRT**: r(theta_real, theta_estimado) = 0.90

### v1.1
- Full Lattix (memoria + consolidación + emociones) supera significativamente a baseline sin memoria
- Vector "desperate" correlaciona negativamente con utilidad coral (replicando hallazgo Anthropic)
- La consolidación nocturna produce mejora medible en rendimiento multiagente

## Estructura

```
lattix_sim/
├── config.py              # Perfiles de agentes, emociones y memoria
├── models/
│   ├── agents.py          # Agentes con perfil psicométrico
│   ├── conditions.py      # Motor de condiciones experimentales
│   ├── emotions.py        # Vectores emocionales y steering
│   └── psychometric.py    # Generación de ítems psicométricos
├── simulation/
│   ├── engine.py          # Motor Monte Carlo v1.0
│   ├── temporal_engine.py # Motor temporal v1.1 (multi-sesión)
│   └── irt.py             # Modelo IRT 2PL
├── memory/
│   ├── layers.py          # Sistema de memoria por capas
│   ├── consolidation.py   # Consolidación nocturna (Auto-Dream)
│   └── retrieval.py       # Recuperación híbrida
├── analysis/
│   ├── statistics.py      # Estadística v1.0
│   ├── extended_statistics.py  # Análisis temporal v1.1
│   ├── factor_analysis.py # Análisis factorial
│   └── reliability.py     # Fiabilidad del instrumento
└── visualization/
    ├── plots.py           # Plots de condiciones
    ├── radar.py           # Perfiles radar
    ├── heatmaps.py        # Heatmaps de correlación
    ├── emotion_plots.py   # Visualizaciones emocionales
    └── memory_plots.py    # Curvas de aprendizaje y memoria
```

## Publicación

- **Target**: arXiv (cs.AI / cs.HC) -> Behavior Research Methods / BMC Psychology
- **Título provisional**: "Emergent Role Negotiation in Multi-Architecture LLM Collaboration: A Case Study in Reverse Engineering"
