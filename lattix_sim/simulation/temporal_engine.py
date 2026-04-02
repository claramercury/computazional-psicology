"""
Motor de simulación temporal Lattix v1.1.

Extiende la simulación Monte Carlo con dimensión temporal:
- Multi-sesión (30 días por defecto)
- Memoria persistente entre sesiones
- Vectores emocionales con steering causal
- Consolidación nocturna ("Auto-Dream")

Compara 3 condiciones temporales:
- baseline_no_memory: Sin memoria entre sesiones
- memory_no_consolidation: Memoria cruda sin consolidación
- full_lattix: Memoria + consolidación + emociones
"""

import numpy as np
import pandas as pd
from typing import Dict, List

from ..config import (
    Condition, LattixVariable, TemporalParams, DEFAULT_TEMPORAL_PARAMS,
    MemoryParams, DEFAULT_MEMORY_PARAMS,
    CONDITION_MODIFIERS, AGENT_PROFILES,
    EmotionDimension,
)
from ..models.conditions import ConditionEngine
from ..models.emotions import EmotionEngine
from ..memory.layers import MemoryStore
from ..memory.consolidation import ConsolidationEngine
from ..memory.retrieval import RetrievalEngine


class TemporalCondition:
    """Condiciones temporales para comparación."""
    BASELINE = "baseline_no_memory"
    MEMORY_ONLY = "memory_no_consolidation"
    FULL_LATTIX = "full_lattix"


ALL_TEMPORAL_CONDITIONS = [
    TemporalCondition.BASELINE,
    TemporalCondition.MEMORY_ONLY,
    TemporalCondition.FULL_LATTIX,
]


class TemporalSimulation:
    """
    Simulación multi-sesión con memoria y emociones.

    Cada sesión simula un día de trabajo multiagente.
    Entre sesiones, se ejecuta consolidación nocturna.
    """

    def __init__(
        self,
        temporal_params: TemporalParams = None,
        memory_params: MemoryParams = None,
        experimental_condition: Condition = Condition.TRIADIC,
    ):
        self.t_params = temporal_params or DEFAULT_TEMPORAL_PARAMS
        self.m_params = memory_params or DEFAULT_MEMORY_PARAMS
        self.condition = experimental_condition
        self.rng = np.random.default_rng(self.t_params.seed)

        self._session_data: List[pd.DataFrame] = []
        self._memory_metrics: List[Dict] = []
        self._emotion_data: List[Dict] = []
        self._consolidation_metrics: List[Dict] = []

    def run(self, temporal_condition: str = TemporalCondition.FULL_LATTIX) -> pd.DataFrame:
        """
        Ejecuta la simulación temporal completa.

        Args:
            temporal_condition: Una de las 3 condiciones temporales.

        Returns:
            DataFrame con columnas:
                session, trial_id, agent, agent_key, condition,
                temporal_condition, + 5 variables Lattix
        """
        use_memory = temporal_condition != TemporalCondition.BASELINE
        use_consolidation = temporal_condition == TemporalCondition.FULL_LATTIX
        use_emotions = temporal_condition == TemporalCondition.FULL_LATTIX

        # Inicializar componentes
        condition_engine = ConditionEngine(self.rng)
        emotion_engine = EmotionEngine(self.rng) if use_emotions else None
        memory_store = MemoryStore(self.m_params, self.rng) if use_memory else None
        consolidation = (
            ConsolidationEngine(memory_store, self.m_params, self.rng)
            if use_consolidation and memory_store
            else None
        )
        retrieval = (
            RetrievalEngine(memory_store, self.m_params, self.rng)
            if use_memory and memory_store
            else None
        )

        all_rows = []

        for session in range(self.t_params.n_sessions):
            # --- Pre-sesión: recuperar memoria y calcular emociones ---
            memory_quality = 0.5
            contradiction_rate = 0.0
            memory_boost = 1.0
            session_emotions = {}

            if use_memory and memory_store:
                memory_quality = memory_store.get_quality_score()
                contradiction_rate = memory_store.get_contradiction_rate()

                if retrieval:
                    retrieved = retrieval.retrieve(
                        current_session=session
                    )
                    ret_quality = retrieval.compute_retrieval_quality(retrieved)
                    memory_boost = retrieval.compute_memory_boost(
                        ret_quality, memory_quality
                    )

            if use_emotions and emotion_engine:
                modifier = CONDITION_MODIFIERS[self.condition]
                active_agents = condition_engine.get_active_agents(self.condition)

                for agent_key in active_agents:
                    emotions = emotion_engine.get_session_emotions(
                        agent_key, self.condition,
                        memory_quality, contradiction_rate,
                    )
                    session_emotions[agent_key] = emotions

                    # Registrar emociones
                    emotion_record = {
                        "session": session,
                        "agent_key": agent_key,
                        "temporal_condition": temporal_condition,
                    }
                    for dim in EmotionDimension:
                        emotion_record[dim.value] = emotions[dim]
                    self._emotion_data.append(emotion_record)

            # --- Simulación de la sesión ---
            samples = condition_engine.sample_condition(
                self.condition, self.t_params.n_trials_per_session
            )

            # Aplicar steering emocional y boost de memoria
            for agent_key, var_samples in samples.items():
                # Steering emocional
                if agent_key in session_emotions and emotion_engine:
                    multipliers = emotion_engine.compute_steering_effect(
                        session_emotions[agent_key]
                    )
                    for var in LattixVariable:
                        if var == LattixVariable.META_PROPOSALS:
                            # Aditivo para conteos
                            var_samples[var] = np.maximum(
                                0, var_samples[var] * multipliers[var]
                            )
                        else:
                            var_samples[var] = np.clip(
                                var_samples[var] * multipliers[var],
                                0.0, 1.0,
                            )

                # Boost de memoria
                if memory_boost > 1.0:
                    for var in LattixVariable:
                        if var == LattixVariable.META_PROPOSALS:
                            var_samples[var] = np.maximum(
                                0, var_samples[var] * memory_boost
                            )
                        else:
                            var_samples[var] = np.clip(
                                var_samples[var] * memory_boost,
                                0.0, 1.0,
                            )

                # Registrar datos
                agent = condition_engine.agents[agent_key]
                for trial_idx in range(self.t_params.n_trials_per_session):
                    row = {
                        "session": session,
                        "trial_id": trial_idx,
                        "agent": agent.name,
                        "agent_key": agent_key,
                        "condition": self.condition.value,
                        "temporal_condition": temporal_condition,
                    }
                    for var in LattixVariable:
                        row[var.value] = var_samples[var][trial_idx]
                    all_rows.append(row)

            # --- Post-sesión: generar memorias ---
            if use_memory and memory_store:
                active_agents = condition_engine.get_active_agents(self.condition)
                for agent_key in active_agents:
                    memory_store.generate_session_entries(agent_key, session)

                # Métricas de memoria pre-consolidación
                self._memory_metrics.append({
                    **memory_store.get_metrics(session),
                    "temporal_condition": temporal_condition,
                    "phase": "pre_consolidation",
                })

            # --- Noche: consolidación ---
            if use_consolidation and consolidation:
                cons_metrics = consolidation.run_nightly_cycle(session)
                cons_metrics["session"] = session
                cons_metrics["temporal_condition"] = temporal_condition
                self._consolidation_metrics.append(cons_metrics)

                # Métricas post-consolidación
                self._memory_metrics.append({
                    **memory_store.get_metrics(session),
                    "temporal_condition": temporal_condition,
                    "phase": "post_consolidation",
                })

        return pd.DataFrame(all_rows)

    def run_all_conditions(self) -> pd.DataFrame:
        """
        Ejecuta las 3 condiciones temporales y combina resultados.
        """
        all_data = []
        for tc in ALL_TEMPORAL_CONDITIONS:
            # Reset RNG para comparabilidad
            self.rng = np.random.default_rng(self.t_params.seed)
            self._emotion_data = []
            self._memory_metrics = []
            self._consolidation_metrics = []

            data = self.run(tc)
            all_data.append(data)

        return pd.concat(all_data, ignore_index=True)

    @property
    def emotion_data(self) -> pd.DataFrame:
        if not self._emotion_data:
            return pd.DataFrame()
        return pd.DataFrame(self._emotion_data)

    @property
    def memory_metrics(self) -> pd.DataFrame:
        if not self._memory_metrics:
            return pd.DataFrame()
        return pd.DataFrame(self._memory_metrics)

    @property
    def consolidation_metrics(self) -> pd.DataFrame:
        if not self._consolidation_metrics:
            return pd.DataFrame()
        return pd.DataFrame(self._consolidation_metrics)

    def session_summary(self, data: pd.DataFrame) -> pd.DataFrame:
        """Media por sesión × temporal_condition × variable."""
        variables = [v.value for v in LattixVariable]
        return (
            data.groupby(["session", "temporal_condition"])[variables]
            .mean()
            .reset_index()
        )
