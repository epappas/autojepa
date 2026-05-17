"""Structural skeleton tests for examples/ijepa-cifar10.

Validates the example's file shape and key contract elements without
actually running training (which needs a GPU + CIFAR download). Catches
regressions like:
- accidental mutation of frozen prepare.py
- removal of required emit_progress / assert_no_grad_on_target calls
  from train.py (these would normally be enforced by the AST validator
  during a campaign — this test catches them at PR time)
- drift between program.md hard-rules wording and the JEPA_HARD_RULES
  prompt fragment
- config.yaml losing the Basilica target / probe_auroc objective /
  recalibrated forecaster defaults

The actual end-to-end smoke (`./run.sh smoke`) and the Basilica
campaign run are tracked in `TODO.md` Phase-2 batch 2 / batch 3 — they
require runtime resources outside CI.
"""

from __future__ import annotations

from pathlib import Path

import yaml

EXAMPLE_DIR = Path(__file__).parent.parent / "examples" / "ijepa-cifar10"


class TestExampleFilesPresent:
    def test_all_required_files_exist(self) -> None:
        for name in (
            "prepare.py",
            "train.py",
            "program.md",
            "config.yaml",
            "README.md",
            "run.sh",
            "requirements.txt",
        ):
            path = EXAMPLE_DIR / name
            assert path.is_file(), f"missing required example file: {name}"

    def test_run_sh_is_executable(self) -> None:
        import os

        path = EXAMPLE_DIR / "run.sh"
        assert os.access(path, os.X_OK), "run.sh must be executable"


class TestPrepareFrozen:
    """Frozen contract: prepare.py exposes exactly the documented surface."""

    def test_prepare_declares_main(self) -> None:
        src = (EXAMPLE_DIR / "prepare.py").read_text()
        assert "def main()" in src

    def test_prepare_writes_documented_outputs(self) -> None:
        src = (EXAMPLE_DIR / "prepare.py").read_text()
        for output_file in (
            "cifar10_train.pt",
            "cifar10_test.pt",
            "probe_eval.pt",
            "canary.pt",
        ):
            assert output_file in src, f"prepare.py must mention {output_file}"


class TestTrainContract:
    """Mutable but contract-bound: train.py must keep the AST validator's
    required calls. The AST validator enforces this at runtime; this
    test catches violations earlier."""

    def test_emit_progress_is_called(self) -> None:
        src = (EXAMPLE_DIR / "train.py").read_text()
        assert "from autojepa.target.progress import emit_progress" in src
        assert 'emit_progress(' in src
        assert '"probe_auroc"' in src

    def test_assert_no_grad_on_target_is_called(self) -> None:
        src = (EXAMPLE_DIR / "train.py").read_text()
        assert "assert_no_grad_on_target" in src

    def test_canary_loss_is_emitted(self) -> None:
        src = (EXAMPLE_DIR / "train.py").read_text()
        assert '"canary_loss"' in src

    def test_uses_stable_pretraining_ijepa(self) -> None:
        src = (EXAMPLE_DIR / "train.py").read_text()
        assert "from stable_pretraining.methods import IJEPA" in src

    def test_ar_params_json_contract(self) -> None:
        src = (EXAMPLE_DIR / "train.py").read_text()
        assert "AR_PARAMS_JSON" in src
        assert "AR_MODEL_DIR" in src

    def test_outcome_json_helper_present(self) -> None:
        """ADR-015: train.py must write outcome.json on every exit path."""
        src = (EXAMPLE_DIR / "train.py").read_text()
        assert "_write_outcome" in src
        assert "OUTCOME_FILENAME" in src
        # Required at canary-fail, success, import-error and last-resort paths.
        assert src.count("_write_outcome(") >= 4
        assert "_entrypoint" in src


class TestProgramMdHardRules:
    """The example's program.md must encode the same JEPA invariants
    the global JEPA_HARD_RULES prompt fragment lists."""

    def test_lists_collapse_thresholds(self) -> None:
        src = (EXAMPLE_DIR / "program.md").read_text()
        for needle in (
            "latent_variance < 0.3",
            "effective_rank   < 32",
            "rankme           < 64",
            "lidar            < 80",
        ):
            assert needle in src, f"program.md missing collapse threshold {needle!r}"

    def test_lists_required_runtime_calls(self) -> None:
        src = (EXAMPLE_DIR / "program.md").read_text()
        assert "emit_progress(step, step_target, metrics={" in src
        assert "assert_no_grad_on_target" in src

    def test_states_phase2_kill_criterion(self) -> None:
        # Whitespace-tolerant: the writeup §11 reference may wrap onto
        # the next line, breaking a naive substring search.
        normalised = " ".join((EXAMPLE_DIR / "program.md").read_text().split()).lower()
        assert "kill criterion" in normalised or "phase-2 falsifier" in normalised


class TestConfigYaml:
    def _config(self) -> dict:
        return yaml.safe_load((EXAMPLE_DIR / "config.yaml").read_text())

    def test_objective_is_probe_auroc_max(self) -> None:
        cfg = self._config()
        assert cfg["objective"]["metric"] == "probe_auroc"
        assert cfg["objective"]["direction"] == "max"

    def test_target_is_basilica(self) -> None:
        cfg = self._config()
        assert cfg["target"]["type"] == "basilica"

    def test_basilica_accepts_a100(self) -> None:
        cfg = self._config()
        assert "A100" in cfg["target"]["basilica"]["gpu_models"]

    def test_policy_is_hybrid_with_jepa_widened_defaults(self) -> None:
        cfg = self._config()
        assert cfg["policy"]["type"] == "hybrid"
        # The meaningful invariant: diff-mode MUST be reachable within
        # the campaign budget. With hybrid_param_explore_iters >=
        # max_iterations, the hybrid policy stays in param mode for
        # the whole run (cf. v21 post-mortem 2026-05-16). The old
        # literal >=25 assertion locked in that bug. Replace with the
        # real reachability constraint.
        max_iters = cfg["controller"]["max_iterations"]
        explore = cfg["policy"]["hybrid_param_explore_iters"]
        stall = cfg["policy"]["hybrid_stall_threshold"]
        assert explore < max_iters, (
            f"hybrid_param_explore_iters ({explore}) >= max_iterations "
            f"({max_iters}): diff mode will never trigger in this campaign"
        )
        assert explore + stall < max_iters, (
            f"explore+stall ({explore}+{stall}) >= max_iterations "
            f"({max_iters}): no room for ratchet evidence after diff "
            f"mode triggers"
        )
        assert stall >= 1, "stall threshold of 0 would switch on every iter"

    def test_intra_iteration_cancel_uses_ssl_defaults(self) -> None:
        cfg = self._config()
        ic = cfg["controller"]["intra_iteration_cancel"]
        assert ic["enabled"] is True
        assert ic["min_steps"] >= 2000
        assert ic["poll_interval_s"] >= 30.0
        assert ic["min_reports_before_decide"] >= 10

    def test_max_iterations_is_phase2_budget(self) -> None:
        cfg = self._config()
        assert cfg["controller"]["max_iterations"] == 20
