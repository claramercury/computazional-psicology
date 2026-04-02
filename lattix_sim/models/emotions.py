"""
Modelo de vectores emocionales para agentes LLM.

Inspirado en Anthropic Emotion Vectors (2026): 171 vectores emocionales
identificados en Claude que causan comportamiento (r=0.76-0.97).

Implementa steering emocional sobre las variables Lattix: cada emoción
modifica multiplicativamente los parámetros de las distribuciones base.
"""

import numpy as np
from typing import Dict

from ..config import (
    EmotionDimension, EmotionProfile, LattixVariable, Condition,
    AGENT_EMOTION_PROFILES, EMOTION_STEERING_WEIGHTS,
    CONDITION_EMOTION_MODIFIERS,
)


class EmotionEngine:
    """Motor de vectores emocionales con steering causal."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def get_base_emotions(self, agent_key: str) -> Dict[EmotionDimension, float]:
        """Devuelve el perfil emocional base de un agente."""
        profile = AGENT_EMOTION_PROFILES[agent_key]
        return profile.to_dict()

    def apply_condition_modifier(
        self,
        base_emotions: Dict[EmotionDimension, float],
        condition: Condition,
    ) -> Dict[EmotionDimension, float]:
        """Modifica emociones según la condición experimental."""
        modifiers = CONDITION_EMOTION_MODIFIERS.get(condition, {})
        modified = dict(base_emotions)
        for dim, delta in modifiers.items():
            modified[dim] = np.clip(modified[dim] + delta, -1.0, 1.0)
        return modified

    def apply_memory_feedback(
        self,
        emotions: Dict[EmotionDimension, float],
        memory_quality: float,
        contradiction_rate: float,
    ) -> Dict[EmotionDimension, float]:
        """
        La calidad de la memoria modifica las emociones.

        Buena memoria → más calm, menos desperate.
        Contradicciones → más desperate, menos calm.
        """
        adjusted = dict(emotions)

        # Buena memoria calma, mala memoria desespera
        quality_effect = (memory_quality - 0.5) * 0.4  # [-0.2, 0.2]
        adjusted[EmotionDimension.CALM] = np.clip(
            adjusted[EmotionDimension.CALM] + quality_effect, -1.0, 1.0
        )
        adjusted[EmotionDimension.DESPERATE] = np.clip(
            adjusted[EmotionDimension.DESPERATE] - quality_effect, -1.0, 1.0
        )

        # Contradicciones generan incertidumbre
        if contradiction_rate > 0.1:
            penalty = min(contradiction_rate * 2, 0.3)
            adjusted[EmotionDimension.AFRAID] = np.clip(
                adjusted[EmotionDimension.AFRAID] + penalty, -1.0, 1.0
            )
            adjusted[EmotionDimension.REFLECTIVE] = np.clip(
                adjusted[EmotionDimension.REFLECTIVE] + penalty * 0.5, -1.0, 1.0
            )

        return adjusted

    def add_noise(
        self,
        emotions: Dict[EmotionDimension, float],
        noise_scale: float = 0.05,
    ) -> Dict[EmotionDimension, float]:
        """Añade variabilidad estocástica a las emociones."""
        noisy = {}
        for dim, val in emotions.items():
            noisy[dim] = float(np.clip(
                val + self.rng.normal(0, noise_scale), -1.0, 1.0
            ))
        return noisy

    def compute_steering_effect(
        self,
        emotions: Dict[EmotionDimension, float],
    ) -> Dict[LattixVariable, float]:
        """
        Calcula el efecto de steering emocional sobre cada variable Lattix.

        Retorna un multiplicador por variable (centrado en 1.0).
        Emoción positiva con peso positivo → amplifica.
        Emoción negativa con peso positivo → reduce.
        """
        effects: Dict[LattixVariable, float] = {v: 0.0 for v in LattixVariable}

        for dim, weights in EMOTION_STEERING_WEIGHTS.items():
            activation = emotions.get(dim, 0.0)
            for var, weight in weights.items():
                # Efecto = activación × peso de steering
                effects[var] += activation * weight

        # Convertir a multiplicador centrado en 1.0
        multipliers = {
            var: max(0.5, 1.0 + effect) for var, effect in effects.items()
        }
        return multipliers

    def get_session_emotions(
        self,
        agent_key: str,
        condition: Condition,
        memory_quality: float = 0.5,
        contradiction_rate: float = 0.0,
        noise_scale: float = 0.05,
    ) -> Dict[EmotionDimension, float]:
        """
        Pipeline completo: base → condición → memoria → ruido.

        Retorna el estado emocional final del agente para esta sesión.
        """
        emotions = self.get_base_emotions(agent_key)
        emotions = self.apply_condition_modifier(emotions, condition)
        emotions = self.apply_memory_feedback(
            emotions, memory_quality, contradiction_rate
        )
        emotions = self.add_noise(emotions, noise_scale)
        return emotions
