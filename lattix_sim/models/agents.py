"""
Modelo de agentes LLM con perfiles psicométricos.
"""

import numpy as np

from ..config import (
    AgentProfile, BetaParams, PoissonParams,
    ConditionModifier, LattixVariable,
)


class Agent:
    """Un agente LLM con perfil psicométrico Lattix."""

    def __init__(self, profile: AgentProfile, rng: np.random.Generator):
        self.profile = profile
        self.rng = rng

    @property
    def name(self) -> str:
        return self.profile.name

    def _get_base_params(self, variable: LattixVariable):
        return getattr(self.profile, variable.value)

    def _apply_modifier(
        self, variable: LattixVariable, base_params, modifier: ConditionModifier
    ):
        """Aplica modificadores de condición a los parámetros base."""
        if variable == LattixVariable.META_PROPOSALS:
            assert isinstance(base_params, PoissonParams)
            new_lam = max(0.1, base_params.lam + modifier.meta_proposals_lam_add)
            return PoissonParams(lam=new_lam)

        assert isinstance(base_params, BetaParams)
        mult_attr = f"{variable.value}_alpha_mult"
        mult = getattr(modifier, mult_attr, 1.0)
        return BetaParams(
            alpha=max(0.5, base_params.alpha * mult),
            beta=base_params.beta,
        )

    def sample(
        self,
        variable: LattixVariable,
        modifier: ConditionModifier,
        n: int = 1,
    ) -> np.ndarray:
        """Genera n muestras de una variable bajo una condición."""
        base = self._get_base_params(variable)
        modified = self._apply_modifier(variable, base, modifier)

        if isinstance(modified, PoissonParams):
            return self.rng.poisson(lam=modified.lam, size=n).astype(float)

        return self.rng.beta(a=modified.alpha, b=modified.beta, size=n)
