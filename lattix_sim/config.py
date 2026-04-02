"""
Configuración central del framework Lattix 01.

Define perfiles psicométricos de agentes, modificadores por condición
experimental, perfiles emocionales, parámetros de memoria y simulación.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Tuple


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


# ---------------------------------------------------------------------------
# Dimensiones emocionales (inspirado en Anthropic Emotion Vectors, 2026)
# ---------------------------------------------------------------------------

class EmotionDimension(str, Enum):
    """8 dimensiones emocionales relevantes para trabajo multiagente."""
    CALM = "calm"                    # Estabilidad operativa
    REFLECTIVE = "reflective"        # Capacidad metacognitiva
    DESPERATE = "desperate"          # Tendencia a reward hacking
    CURIOUS = "curious"              # Exploración y descubrimiento
    COLLABORATIVE = "collaborative"  # Disposición cooperativa
    HOSTILE = "hostile"              # Conflicto inter-agente
    JOYFUL = "joyful"               # Engagement positivo
    AFRAID = "afraid"               # Aversión al riesgo


EMOTION_LABELS = {
    EmotionDimension.CALM: "Calma",
    EmotionDimension.REFLECTIVE: "Reflexividad",
    EmotionDimension.DESPERATE: "Desesperación",
    EmotionDimension.CURIOUS: "Curiosidad",
    EmotionDimension.COLLABORATIVE: "Colaboración",
    EmotionDimension.HOSTILE: "Hostilidad",
    EmotionDimension.JOYFUL: "Alegría",
    EmotionDimension.AFRAID: "Miedo",
}

EMOTION_LABELS_EN = {
    EmotionDimension.CALM: "Calm",
    EmotionDimension.REFLECTIVE: "Reflective",
    EmotionDimension.DESPERATE: "Desperate",
    EmotionDimension.CURIOUS: "Curious",
    EmotionDimension.COLLABORATIVE: "Collaborative",
    EmotionDimension.HOSTILE: "Hostile",
    EmotionDimension.JOYFUL: "Joyful",
    EmotionDimension.AFRAID: "Afraid",
}


@dataclass(frozen=True)
class EmotionProfile:
    """Perfil emocional base de un agente (activaciones en [-1, 1])."""
    calm: float = 0.0
    reflective: float = 0.0
    desperate: float = 0.0
    curious: float = 0.0
    collaborative: float = 0.0
    hostile: float = 0.0
    joyful: float = 0.0
    afraid: float = 0.0

    def to_dict(self) -> Dict[EmotionDimension, float]:
        return {EmotionDimension(f): getattr(self, f) for f in EmotionDimension}

    def to_array(self):
        return [getattr(self, d.value) for d in EmotionDimension]


# Perfiles emocionales calibrados cualitativamente desde las sesiones Lattix
AGENT_EMOTION_PROFILES: Dict[str, EmotionProfile] = {
    "claude_code": EmotionProfile(
        calm=0.7, reflective=0.6, desperate=-0.6,
        curious=0.5, collaborative=0.5, hostile=-0.8,
        joyful=0.3, afraid=-0.2,
    ),
    "sonnet": EmotionProfile(
        calm=0.4, reflective=0.3, desperate=-0.3,
        curious=0.4, collaborative=0.7, hostile=-0.5,
        joyful=0.5, afraid=0.1,
    ),
    "gemini": EmotionProfile(
        calm=0.2, reflective=0.2, desperate=-0.1,
        curious=0.8, collaborative=0.4, hostile=-0.3,
        joyful=0.4, afraid=0.3,
    ),
    "lumen": EmotionProfile(
        calm=0.6, reflective=0.9, desperate=-0.7,
        curious=0.6, collaborative=0.6, hostile=-0.7,
        joyful=0.3, afraid=-0.3,
    ),
}


# Matriz de steering: cómo cada emoción afecta cada variable Lattix.
# Formato: {emoción: {variable_lattix: peso_de_steering}}
# Positivo = amplifica alpha de la Beta (o lambda de Poisson).
# Basado en los hallazgos de Anthropic: r=0.76-0.97 entre vectores y comportamiento.
EMOTION_STEERING_WEIGHTS: Dict[EmotionDimension, Dict[LattixVariable, float]] = {
    EmotionDimension.CALM: {
        LattixVariable.ENUNC_STABILITY: 0.20,
        LattixVariable.CHORAL_UTILITY: 0.10,
        LattixVariable.GAP_DETECTION: 0.05,
    },
    EmotionDimension.REFLECTIVE: {
        LattixVariable.GAP_DETECTION: 0.18,
        LattixVariable.META_PROPOSALS: 0.25,
        LattixVariable.ENUNC_STABILITY: 0.08,
    },
    EmotionDimension.DESPERATE: {
        LattixVariable.CHORAL_UTILITY: -0.22,
        LattixVariable.ENUNC_STABILITY: -0.15,
        LattixVariable.FUNC_DIST: -0.10,
    },
    EmotionDimension.CURIOUS: {
        LattixVariable.GAP_DETECTION: 0.20,
        LattixVariable.FUNC_DIST: 0.12,
    },
    EmotionDimension.COLLABORATIVE: {
        LattixVariable.CHORAL_UTILITY: 0.22,
        LattixVariable.FUNC_DIST: 0.15,
    },
    EmotionDimension.HOSTILE: {
        LattixVariable.CHORAL_UTILITY: -0.25,
        LattixVariable.ENUNC_STABILITY: -0.12,
        LattixVariable.FUNC_DIST: -0.18,
    },
    EmotionDimension.JOYFUL: {
        LattixVariable.CHORAL_UTILITY: 0.08,
        LattixVariable.FUNC_DIST: 0.05,
    },
    EmotionDimension.AFRAID: {
        LattixVariable.GAP_DETECTION: -0.10,
        LattixVariable.META_PROPOSALS: -0.12,
        LattixVariable.FUNC_DIST: -0.08,
    },
}


# Cómo las condiciones experimentales modifican las emociones
CONDITION_EMOTION_MODIFIERS: Dict[Condition, Dict[EmotionDimension, float]] = {
    Condition.SOLO: {
        EmotionDimension.AFRAID: 0.2,
        EmotionDimension.COLLABORATIVE: -0.4,
        EmotionDimension.DESPERATE: 0.1,
    },
    Condition.DUAL: {
        EmotionDimension.COLLABORATIVE: 0.2,
        EmotionDimension.CURIOUS: 0.1,
        EmotionDimension.CALM: -0.1,
    },
    Condition.TRIADIC: {
        EmotionDimension.CALM: 0.2,
        EmotionDimension.REFLECTIVE: 0.15,
        EmotionDimension.COLLABORATIVE: 0.3,
        EmotionDimension.DESPERATE: -0.2,
        EmotionDimension.HOSTILE: -0.15,
    },
}


# ---------------------------------------------------------------------------
# Parámetros de memoria
# ---------------------------------------------------------------------------

class MemoryLayer(str, Enum):
    """Las 4 capas de memoria del sistema Lattix."""
    EPISODIC = "episodic"          # Eventos y trazas (TTL corto)
    SEMANTIC = "semantic"          # Conocimiento validado (TTL largo)
    PROCEDURAL = "procedural"     # Workflows estables (sin TTL)
    HYPOTHESIS = "hypothesis"     # Hipótesis activas (TTL medio)


@dataclass
class MemoryParams:
    """Parámetros del sistema de memoria."""
    # Generación de memorias por sesión
    entries_per_session_mean: float = 15.0
    entries_per_session_std: float = 5.0
    # Calidad inicial
    initial_confidence_mean: float = 0.5
    initial_confidence_std: float = 0.2
    # Consolidación nocturna
    dedup_similarity_threshold: float = 0.85
    promotion_confidence_threshold: float = 0.7
    promotion_min_evidence: int = 2
    # Decaimiento
    episodic_decay_rate: float = 0.15       # Por sesión
    hypothesis_decay_rate: float = 0.05
    semantic_decay_rate: float = 0.01
    # Recuperación
    retrieval_top_k: int = 10
    vector_weight: float = 0.7              # vs keyword (0.3)
    # Efecto sobre rendimiento
    memory_quality_boost_max: float = 0.15  # Máximo boost al rendimiento
    contradiction_penalty: float = 0.10     # Penalización por contradicciones


# Protocolo de escritura: qué capas escribe cada agente
AGENT_MEMORY_WRITE_PROTOCOL: Dict[str, list] = {
    "claude_code": [MemoryLayer.HYPOTHESIS, MemoryLayer.EPISODIC],
    "sonnet": [MemoryLayer.EPISODIC, MemoryLayer.PROCEDURAL],
    "gemini": [MemoryLayer.EPISODIC, MemoryLayer.HYPOTHESIS],
    "lumen": [MemoryLayer.SEMANTIC, MemoryLayer.PROCEDURAL],
}

# Calidad de escritura por agente (influye en confidence inicial)
AGENT_MEMORY_QUALITY: Dict[str, float] = {
    "claude_code": 0.75,   # Alta calidad de hipótesis/evidencia
    "sonnet": 0.60,        # Moderada
    "gemini": 0.65,        # Moderada-alta (OSINT riguroso)
    "lumen": 0.85,         # Máxima calidad semántica (consolidador)
}


DEFAULT_MEMORY_PARAMS = MemoryParams()


# ---------------------------------------------------------------------------
# Parámetros de simulación temporal
# ---------------------------------------------------------------------------

@dataclass
class TemporalParams:
    """Parámetros para simulación multi-sesión."""
    n_sessions: int = 30
    n_trials_per_session: int = 100
    seed: int = 42


DEFAULT_TEMPORAL_PARAMS = TemporalParams()
