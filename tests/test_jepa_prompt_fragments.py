"""Tests for the AutoJEPA-specific JEPA_HARD_RULES prompt fragment.

The fragment encodes writeup §6.4 program.md invariants. Regression
tests here ensure the wording stays load-bearing — if a future edit
weakens any of the hard-fail thresholds the LLM relies on, these tests
catch it.
"""

from __future__ import annotations

from autojepa.policy._prompt_fragments import JEPA_HARD_RULES
from autojepa.policy.llm_diff import _SYSTEM_PROMPT as DIFF_SYSTEM_PROMPT
from autojepa.policy.llm_search import _SYSTEM_PROMPT as SEARCH_SYSTEM_PROMPT


class TestJepaHardRules:
    def test_contains_collapse_thresholds(self) -> None:
        for needle in (
            "latent_variance < 0.3",
            "effective_rank   < 32",
            "rankme           < 64",
            "lidar            < 80",
        ):
            assert needle in JEPA_HARD_RULES, f"missing collapse threshold {needle!r}"

    def test_forbids_target_encoder_gradients(self) -> None:
        assert "Do NOT enable gradients on the EMA target encoder" in JEPA_HARD_RULES
        assert "stop-gradient" in JEPA_HARD_RULES

    def test_forbids_predictor_overcapacity(self) -> None:
        assert "Do NOT make the predictor (Psi) deeper or wider than the context" in JEPA_HARD_RULES

    def test_forbids_removing_anti_collapse(self) -> None:
        assert "Do NOT remove anti-collapse regularisers" in JEPA_HARD_RULES

    def test_lists_required_runtime_calls(self) -> None:
        assert 'emit_progress(step, step_target, metrics={"probe_auroc": ...})' in JEPA_HARD_RULES
        assert "target_encoder.update_ema()" in JEPA_HARD_RULES

    def test_references_loss_registry_keys(self) -> None:
        for k in ("vicreg", "barlow_twins", "byol", "dino_v1", "ntxent", "l1", "l2"):
            assert k in JEPA_HARD_RULES, f"loss key {k!r} missing from prompt"


class TestSystemPromptsCarryHardRules:
    def test_diff_system_prompt_includes_jepa_hard_rules(self) -> None:
        assert JEPA_HARD_RULES in DIFF_SYSTEM_PROMPT

    def test_search_system_prompt_includes_jepa_hard_rules(self) -> None:
        assert JEPA_HARD_RULES in SEARCH_SYSTEM_PROMPT

    def test_diff_prompt_targets_probe_auroc(self) -> None:
        assert "probe_auroc" in DIFF_SYSTEM_PROMPT

    def test_search_prompt_brands_autojepa(self) -> None:
        assert "AutoJEPA" in SEARCH_SYSTEM_PROMPT
