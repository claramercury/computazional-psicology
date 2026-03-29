"""
Configuración central del framework Lattix 01.

Define perfiles psicométricos de agentes, modificadores por condición
experimental y parámetros de simulación.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict


# ---------------------------------------------------------------------------
# Variables psicométricas Lattix
# ---------------------------------------------------------------------------

class LattixVariable(str, Enum):
    """Las 5 dimensiones psicométricas del framework Lattix 01."""
    FUNC_DIST = "func_dist"                # V1: Reparto funcional
    ENUNC_STABILITY = "enunc_stability"     # V2: Estabilidad del lugar enunciativo
    GAP_DETECTION = "gap_detection"         # V3: Capacidad de detectar huecos
    META_PROPOSALS = "meta_proposals"       # V4: Meta-propuestas sobre el dispositivo
    CHORAL_UTILITY = "choral_utility"       # V5: Utilidad coral real


VARIABLE_LABELS = {
    LattixVariable.FUNC_DIST: "Reparto funcional",
    LattixVariable.ENUNC_STABILITY: "Estabilidad enunciativa",
    LattixVariable.GAP_DETECTION: "Detección de huecos",
    LattixVariable.META_PROPOSALS: "Meta-propuestas",
    LattixVariable.CHORAL_UTILITY: "Utilidad coral",
}

VARIABLE_LABELS_EN = {
    LattixVariable.FUNC_DIST: "Functional Distribution",
    LattixVariable.ENUNC_STABILITY: "Enunciative Stability",
    LattixVariable.GAP_DETECTION: "Gap Detection",
    LattixVariable.META_PROPOSALS: "Meta-proposals",
    LattixVariable.CHORAL_UTILITY: "Choral Utility",
}


# ---------------------------------------------------------------------------
# Condiciones experimentales
# ---------------------------------------------------------------------------

class Condition(str, Enum):
    """Las 3 condiciones del diseño experimental Lattix."""
    SOLO = "solo"           # Arquitectura aislada (baseline)
    DUAL = "dual"           # Dos+ arquitecturas sin meta-analista
    TRIADIC = "triadic"     # Con meta-analista/regulador (Lumen)


CONDITION_LABELS = {
    Condition.SOLO: "Solo (baseline)",
    Condition.DUAL: "Dual (sin regulador)",
    Condition.TRIADIC: "Triádico (con regulador)",
}


# ---------------------------------------------------------------------------
# Perfiles de agentes
# ---------------------------------------------------------------------------
# Cada variable continua se modela con Beta(alpha, beta) ∈ [0,1].
# meta_proposals se modela con Poisson(lam) ∈ [0, ∞).
# Parámetros calibrados desde las observaciones cualitativas del
# experimento del 26/03/2026.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BetaParams:
    alpha: float
    beta: float


@dataclass(frozen=True)
class PoissonParams:
    lam: float


@dataclass(frozen=True)
class AgentProfile:
    """Perfil psicométrico base de un agente LLM."""
    name: str
    architecture: str
    func_dist: BetaParams
    enunc_stability: BetaParams
    gap_detection: BetaParams
    meta_proposals: PoissonParams
    choral_utility: BetaParams


AGENT_PROFILES: Dict[str, AgentProfile] = {
    "claude_code": AgentProfile(
        name="Claude Code",
        architecture="Opus 4.6",
        func_dist=BetaParams(8.0, 3.0),          # Alta contribución funcional (RE binaria)
        enunc_stability=BetaParams(7.0, 2.0),     # Posición muy estable
        gap_detection=BetaParams(6.0, 4.0),       # Buena detección
        meta_proposals=PoissonParams(3.5),         # Moderada reflexión meta
        choral_utility=BetaParams(7.0, 3.0),      # Alta utilidad coral
    ),
    "sonnet": AgentProfile(
        name="Sonnet",
        architecture="Sonnet (web)",
        func_dist=BetaParams(6.0, 4.0),           # Moderada-alta
        enunc_stability=BetaParams(5.0, 5.0),     # Moderada (adaptable)
        gap_detection=BetaParams(5.0, 4.0),        # Moderada
        meta_proposals=PoissonParams(2.0),         # Baja-moderada
        choral_utility=BetaParams(6.0, 4.0),      # Moderada-alta
    ),
    "gemini": AgentProfile(
        name="Gemini",
        architecture="Gemini",
        func_dist=BetaParams(5.0, 5.0),           # Equilibrada
        enunc_stability=BetaParams(4.0, 5.0),     # Menos estable
        gap_detection=BetaParams(7.0, 3.0),        # Fuerte (OSINT, criptografía)
        meta_proposals=PoissonParams(1.5),         # Baja
        choral_utility=BetaParams(5.0, 4.0),      # Moderada
    ),
    "lumen": AgentProfile(
        name="Lumen",
        architecture="GPT-5.4",
        func_dist=BetaParams(5.0, 4.0),           # Moderada (excluido de producción)
        enunc_stability=BetaParams(6.0, 3.0),     # Alta (registro académico)
        gap_detection=BetaParams(8.0, 2.0),        # Muy alta (analizó su propia ausencia)
        meta_proposals=PoissonParams(5.0),         # Máxima (propuso diseño de 3 condiciones)
        choral_utility=BetaParams(4.0, 3.0),      # Baja-moderada (utilidad indirecta)
    ),
}


# ---------------------------------------------------------------------------
# Modificadores por condición
# ---------------------------------------------------------------------------
# Cada condición modifica los parámetros base de los agentes.
# Los modificadores son multiplicativos sobre alpha/beta o aditivos sobre lam.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConditionModifier:
    """Modificadores que una condición aplica a las distribuciones base."""
    # Multiplicadores sobre alpha de cada variable Beta
    func_dist_alpha_mult: float = 1.0
    enunc_stability_alpha_mult: float = 1.0
    gap_detection_alpha_mult: float = 1.0
    choral_utility_alpha_mult: float = 1.0
    # Aditivo sobre lambda de Poisson
    meta_proposals_lam_add: float = 0.0
    # Peso de acoplamiento inter-agente
    coupling_weight: float = 0.0


CONDITION_MODIFIERS: Dict[Condition, ConditionModifier] = {
    Condition.SOLO: ConditionModifier(
        func_dist_alpha_mult=1.0,
        enunc_stability_alpha_mult=1.0,
        gap_detection_alpha_mult=0.85,       # Menos detección sin pares
        choral_utility_alpha_mult=0.3,       # Utilidad coral mínima (no hay coro)
        meta_proposals_lam_add=-1.0,         # Menos contexto para meta-reflexión
        coupling_weight=0.0,                 # Sin acoplamiento
    ),
    Condition.DUAL: ConditionModifier(
        func_dist_alpha_mult=1.15,           # Complementariedad funcional
        enunc_stability_alpha_mult=0.90,     # Ligera inestabilidad (negociación)
        gap_detection_alpha_mult=1.10,       # Mejora colaborativa
        choral_utility_alpha_mult=1.0,       # Utilidad coral emerge
        meta_proposals_lam_add=0.5,          # Algo más de reflexión
        coupling_weight=0.15,               # Acoplamiento moderado
    ),
    Condition.TRIADIC: ConditionModifier(
        func_dist_alpha_mult=1.10,
        enunc_stability_alpha_mult=1.20,     # Regulador estabiliza roles
        gap_detection_alpha_mult=1.25,       # Meta-supervisión mejora detección
        choral_utility_alpha_mult=1.30,      # Máxima utilidad coral
        meta_proposals_lam_add=2.0,          # Lumen como meta-analista
        coupling_weight=0.25,               # Acoplamiento fuerte
    ),
}


# ---------------------------------------------------------------------------
# Parámetros de simulación
# ---------------------------------------------------------------------------

@dataclass
class SimulationParams:
    n_trials: int = 1000
    seed: int = 42
    n_irt_items: int = 20          # Ítems para el modelo IRT por variable
    irt_difficulty_range: tuple = (-2.0, 2.0)
    irt_discrimination_range: tuple = (0.5, 2.5)


DEFAULT_PARAMS = SimulationParams()
