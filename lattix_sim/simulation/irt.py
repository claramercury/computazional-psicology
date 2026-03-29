"""
Modelo de Teoría de Respuesta al Ítem (IRT) 2PL.

Proporciona fundamentación psicométrica al framework Lattix midiendo
la capacidad latente (theta) de cada agente en cada dimensión.
"""

import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass


@dataclass
class IRTItem:
    """Un ítem del instrumento psicométrico."""
    discrimination: float    # parámetro a (pendiente)
    difficulty: float        # parámetro b (umbral)


class IRTModel:
    """
    Modelo 2PL (Two-Parameter Logistic).

    P(X=1 | theta, a, b) = 1 / (1 + exp(-a * (theta - b)))
    """

    def __init__(self, n_items: int = 20, rng: np.random.Generator = None):
        self.rng = rng or np.random.default_rng(42)
        self.items = self._generate_items(n_items)

    def _generate_items(self, n: int) -> list:
        """Genera ítems con discriminación y dificultad variadas."""
        items = []
        for _ in range(n):
            a = self.rng.uniform(0.5, 2.5)
            b = self.rng.uniform(-2.0, 2.0)
            items.append(IRTItem(discrimination=a, difficulty=b))
        return items

    @staticmethod
    def probability(theta: float, item: IRTItem) -> float:
        """Probabilidad de respuesta correcta."""
        z = item.discrimination * (theta - item.difficulty)
        return 1.0 / (1.0 + np.exp(-z))

    def simulate_responses(self, thetas: np.ndarray) -> np.ndarray:
        """
        Simula respuestas dicotómicas para un vector de thetas.

        Retorna: matriz (n_sujetos × n_items)
        """
        n_subjects = len(thetas)
        n_items = len(self.items)
        responses = np.zeros((n_subjects, n_items))

        for j, item in enumerate(self.items):
            probs = np.array([self.probability(t, item) for t in thetas])
            responses[:, j] = self.rng.binomial(1, probs)

        return responses

    def estimate_theta(self, responses: np.ndarray) -> np.ndarray:
        """
        Estima theta (MLE) para cada patrón de respuesta.

        Usa Maximum Likelihood Estimation con scipy.optimize.
        """
        n_subjects = responses.shape[0]
        thetas = np.zeros(n_subjects)

        for i in range(n_subjects):
            pattern = responses[i]

            def neg_log_likelihood(theta):
                ll = 0.0
                for j, item in enumerate(self.items):
                    p = self.probability(theta[0], item)
                    p = np.clip(p, 1e-10, 1 - 1e-10)
                    ll += pattern[j] * np.log(p) + (1 - pattern[j]) * np.log(1 - p)
                return -ll

            result = minimize(neg_log_likelihood, x0=[0.0], method="Nelder-Mead")
            thetas[i] = result.x[0]

        return thetas

    def item_information(self, theta: float) -> np.ndarray:
        """Función de información por ítem en theta."""
        info = np.zeros(len(self.items))
        for j, item in enumerate(self.items):
            p = self.probability(theta, item)
            q = 1.0 - p
            info[j] = (item.discrimination ** 2) * p * q
        return info

    def test_information(self, theta: float) -> float:
        """Función de información total del test en theta."""
        return self.item_information(theta).sum()

    def information_curve(self, theta_range: np.ndarray = None) -> tuple:
        """Curva de información del test."""
        if theta_range is None:
            theta_range = np.linspace(-3, 3, 100)
        info = np.array([self.test_information(t) for t in theta_range])
        return theta_range, info
