"""
Motor de condiciones experimentales.

Aplica modificadores de condición y acoplamiento inter-agente.
"""

import numpy as np

from ..config import (
    Condition, ConditionModifier, LattixVariable,
    CONDITION_MODIFIERS, AGENT_PROFILES,
)
from .agents import Agent


class ConditionEngine:
    """Gestiona la lógica de condiciones experimentales."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.agents = {
            key: Agent(profile, rng)
            for key, profile in AGENT_PROFILES.items()
        }

    def get_active_agents(self, condition: Condition) -> dict:
        """Devuelve los agentes activos según la condición."""
        if condition == Condition.SOLO:
            # En solo, cada agente trabaja aislado - devolvemos todos
            # pero sin acoplamiento
            return dict(self.agents)
        if condition == Condition.DUAL:
            # Sesión del 26/03: Claude Code + Sonnet + Gemini (sin Lumen)
            return {k: v for k, v in self.agents.items() if k != "lumen"}
        # Triádico: todos incluido Lumen como meta-analista
        return dict(self.agents)

    def sample_condition(
        self,
        condition: Condition,
        n_trials: int,
    ) -> dict:
        """
        Genera muestras para todos los agentes activos en una condición.

        Retorna dict: {agent_key: {variable: np.ndarray(n_trials,)}}
        """
        modifier = CONDITION_MODIFIERS[condition]
        active = self.get_active_agents(condition)
        variables = list(LattixVariable)

        # Paso 1: muestras independientes
        raw_samples = {}
        for agent_key, agent in active.items():
            raw_samples[agent_key] = {}
            for var in variables:
                raw_samples[agent_key][var] = agent.sample(var, modifier, n_trials)

        # Paso 2: acoplamiento inter-agente
        if modifier.coupling_weight > 0 and len(active) > 1:
            raw_samples = self._apply_coupling(
                raw_samples, modifier.coupling_weight, variables
            )

        return raw_samples

    def _apply_coupling(
        self,
        samples: dict,
        weight: float,
        variables: list,
    ) -> dict:
        """
        Acoplamiento: score_efectivo = (1-w)*score_base + w*mean(otros_agentes).

        Para meta_proposals (conteo), el acoplamiento es aditivo.
        """
        agent_keys = list(samples.keys())
        coupled = {k: {} for k in agent_keys}

        for var in variables:
            # Matriz: agentes × trials
            matrix = np.stack([samples[k][var] for k in agent_keys])
            n_agents = len(agent_keys)

            for i, key in enumerate(agent_keys):
                others_mean = (matrix.sum(axis=0) - matrix[i]) / (n_agents - 1)

                if var == LattixVariable.META_PROPOSALS:
                    # Aditivo para conteos
                    coupled[key][var] = np.maximum(
                        0, matrix[i] + weight * others_mean
                    )
                else:
                    # Ponderado para proporciones [0, 1]
                    coupled[key][var] = np.clip(
                        (1 - weight) * matrix[i] + weight * others_mean,
                        0.0, 1.0,
                    )

        return coupled
