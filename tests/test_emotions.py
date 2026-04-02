"""Tests para el módulo de vectores emocionales."""

import numpy as np
import pytest

from lattix_sim.config import (
    EmotionDimension, LattixVariable, Condition,
    AGENT_EMOTION_PROFILES, EMOTION_STEERING_WEIGHTS,
)
from lattix_sim.models.emotions import EmotionEngine


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def engine(rng):
    return EmotionEngine(rng)


class TestEmotionProfiles:
    """Tests para perfiles emocionales de agentes."""

    def test_all_agents_have_profiles(self):
        for key in ["claude_code", "sonnet", "gemini", "lumen"]:
            assert key in AGENT_EMOTION_PROFILES

    def test_profile_values_in_range(self):
        for key, profile in AGENT_EMOTION_PROFILES.items():
            for dim in EmotionDimension:
                val = getattr(profile, dim.value)
                assert -1.0 <= val <= 1.0, f"{key}.{dim.value} = {val} out of range"

    def test_lumen_most_reflective(self):
        """Lumen debe tener la mayor reflexividad."""
        lumen_ref = AGENT_EMOTION_PROFILES["lumen"].reflective
        for key, profile in AGENT_EMOTION_PROFILES.items():
            if key != "lumen":
                assert lumen_ref >= profile.reflective

    def test_gemini_most_curious(self):
        """Gemini debe tener la mayor curiosidad."""
        gemini_cur = AGENT_EMOTION_PROFILES["gemini"].curious
        for key, profile in AGENT_EMOTION_PROFILES.items():
            if key != "gemini":
                assert gemini_cur >= profile.curious


class TestEmotionEngine:
    """Tests para el motor de emociones."""

    def test_base_emotions_correct(self, engine):
        emotions = engine.get_base_emotions("claude_code")
        assert len(emotions) == len(EmotionDimension)
        assert emotions[EmotionDimension.CALM] == 0.7

    def test_condition_modifier_triadic_increases_calm(self, engine):
        base = engine.get_base_emotions("claude_code")
        modified = engine.apply_condition_modifier(base, Condition.TRIADIC)
        assert modified[EmotionDimension.CALM] > base[EmotionDimension.CALM]

    def test_condition_modifier_solo_reduces_collaborative(self, engine):
        base = engine.get_base_emotions("sonnet")
        modified = engine.apply_condition_modifier(base, Condition.SOLO)
        assert modified[EmotionDimension.COLLABORATIVE] < base[EmotionDimension.COLLABORATIVE]

    def test_memory_feedback_good_quality_calms(self, engine):
        base = engine.get_base_emotions("claude_code")
        adjusted = engine.apply_memory_feedback(base, memory_quality=0.9, contradiction_rate=0.0)
        assert adjusted[EmotionDimension.CALM] > base[EmotionDimension.CALM]

    def test_memory_feedback_contradictions_increase_afraid(self, engine):
        base = engine.get_base_emotions("claude_code")
        adjusted = engine.apply_memory_feedback(base, memory_quality=0.5, contradiction_rate=0.3)
        assert adjusted[EmotionDimension.AFRAID] > base[EmotionDimension.AFRAID]

    def test_noise_preserves_range(self, engine):
        base = engine.get_base_emotions("lumen")
        for _ in range(100):
            noisy = engine.add_noise(base, noise_scale=0.3)
            for dim, val in noisy.items():
                assert -1.0 <= val <= 1.0

    def test_steering_effect_desperate_reduces_choral(self, engine):
        """Vector 'desperate' alto debe reducir choral_utility."""
        high_desperate = {d: 0.0 for d in EmotionDimension}
        high_desperate[EmotionDimension.DESPERATE] = 0.8

        neutral = {d: 0.0 for d in EmotionDimension}

        effect_desp = engine.compute_steering_effect(high_desperate)
        effect_neutral = engine.compute_steering_effect(neutral)

        assert effect_desp[LattixVariable.CHORAL_UTILITY] < effect_neutral[LattixVariable.CHORAL_UTILITY]

    def test_steering_effect_reflective_boosts_gap_detection(self, engine):
        """Vector 'reflective' alto debe mejorar gap_detection."""
        high_reflective = {d: 0.0 for d in EmotionDimension}
        high_reflective[EmotionDimension.REFLECTIVE] = 0.9

        effect = engine.compute_steering_effect(high_reflective)
        assert effect[LattixVariable.GAP_DETECTION] > 1.0

    def test_full_pipeline(self, engine):
        """Pipeline completo no debe crashear y valores en rango."""
        emotions = engine.get_session_emotions(
            "lumen", Condition.TRIADIC,
            memory_quality=0.8, contradiction_rate=0.05,
        )
        assert len(emotions) == len(EmotionDimension)
        for dim, val in emotions.items():
            assert -1.0 <= val <= 1.0
