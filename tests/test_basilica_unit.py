"""Unit tests for BasilicaTarget helpers -- no cloud, no cost."""
from __future__ import annotations

import json

from autojepa.target.basilica import BasilicaTarget


class TestParseMetrics:
    """Test _parse_metrics static method."""

    def test_extracts_key_value_pairs(self) -> None:
        logs = "loss=0.432100\nval_bpb=0.150000\nf1=0.850000\naccuracy=0.875000"
        metrics = BasilicaTarget._parse_metrics(logs)
        assert abs(metrics["loss"] - 0.4321) < 1e-6
        assert abs(metrics["val_bpb"] - 0.15) < 1e-6
        assert abs(metrics["f1"] - 0.85) < 1e-6
        assert abs(metrics["accuracy"] - 0.875) < 1e-6

    def test_handles_scientific_notation(self) -> None:
        logs = "lr=2e-5\nloss=1.23e+02"
        metrics = BasilicaTarget._parse_metrics(logs)
        assert abs(metrics["lr"] - 2e-5) < 1e-10
        assert abs(metrics["loss"] - 123.0) < 1e-6

    def test_empty_logs_returns_empty(self) -> None:
        assert BasilicaTarget._parse_metrics("") == {}

    def test_no_metrics_in_text(self) -> None:
        assert BasilicaTarget._parse_metrics("Starting training...\nDone.") == {}

    def test_ignores_invalid_float_values(self) -> None:
        logs = "loss=0.5\nbad=not_a_float"
        metrics = BasilicaTarget._parse_metrics(logs)
        assert "loss" in metrics
        assert "bad" not in metrics

    def test_mixed_text_and_metrics(self) -> None:
        logs = (
            "Epoch 1/3\n"
            "Step 10: loss=0.6543\n"
            "Evaluation results:\n"
            "val_bpb=0.234000 f1=0.766000\n"
        )
        metrics = BasilicaTarget._parse_metrics(logs)
        assert "loss" in metrics
        assert "val_bpb" in metrics
        assert "f1" in metrics

    def test_keys_lowercased(self) -> None:
        logs = "LOSS=0.5\nVal_BPB=0.2"
        metrics = BasilicaTarget._parse_metrics(logs)
        assert "loss" in metrics
        assert "val_bpb" in metrics

    def test_negative_values(self) -> None:
        logs = "improvement=-0.05"
        metrics = BasilicaTarget._parse_metrics(logs)
        assert abs(metrics["improvement"] - (-0.05)) < 1e-6


class TestExtractMessages:
    """Test _extract_messages static method for SSE JSON log parsing."""

    def test_plain_text_passthrough(self) -> None:
        raw = "Hello world\nTraining started"
        result = BasilicaTarget._extract_messages(raw)
        assert "Hello world" in result
        assert "Training started" in result

    def test_sse_json_extraction(self) -> None:
        lines = [
            'data: {"message": "Starting epoch 1"}',
            'data: {"message": "loss=0.5432"}',
        ]
        raw = "\n".join(lines)
        result = BasilicaTarget._extract_messages(raw)
        assert "Starting epoch 1" in result
        assert "loss=0.5432" in result

    def test_empty_messages_skipped(self) -> None:
        raw = 'data: {"message": ""}\ndata: {"message": "real log"}'
        result = BasilicaTarget._extract_messages(raw)
        assert result == "real log"

    def test_malformed_json_kept_as_text(self) -> None:
        raw = "data: {broken json\nnormal text"
        result = BasilicaTarget._extract_messages(raw)
        assert "{broken json" in result
        assert "normal text" in result

    def test_empty_input(self) -> None:
        assert BasilicaTarget._extract_messages("") == ""

    def test_blank_lines_skipped(self) -> None:
        raw = "line1\n\n\nline2"
        result = BasilicaTarget._extract_messages(raw)
        assert result == "line1\nline2"

    def test_mixed_sse_and_plain(self) -> None:
        raw = (
            "plain log line\n"
            'data: {"message": "from sse"}\n'
            "another plain line\n"
        )
        result = BasilicaTarget._extract_messages(raw)
        assert "plain log line" in result
        assert "from sse" in result
        assert "another plain line" in result

    def test_json_without_message_key(self) -> None:
        raw = 'data: {"level": "info", "text": "something"}'
        result = BasilicaTarget._extract_messages(raw)
        # No "message" key, so empty message is skipped; the raw json line
        # should not appear since it was valid json with no message.
        assert result == ""

    def test_realistic_basilica_logs(self) -> None:
        entries = [
            {"message": "Container started"},
            {"message": "pip install complete"},
            {"message": "Epoch 1/1"},
            {"message": "loss=0.432100"},
            {"message": "val_bpb=0.150000"},
            {"message": "f1=0.850000"},
        ]
        raw = "\n".join(f"data: {json.dumps(e)}" for e in entries)
        result = BasilicaTarget._extract_messages(raw)
        assert "loss=0.432100" in result
        assert "val_bpb=0.150000" in result


class TestBuildBootstrapCmd:
    """Test _build_bootstrap_cmd static method."""

    def test_basic_command_without_setup(self) -> None:
        script = BasilicaTarget._build_bootstrap_cmd(["python3", "train.py"])
        assert "python3" in script
        assert "train.py" in script
        assert "HTTPServer" in script
        assert "subprocess.call" in script

    def test_setup_cmd_injected(self) -> None:
        script = BasilicaTarget._build_bootstrap_cmd(
            ["python3", "train.py"],
            setup_cmd="pip install torch",
        )
        assert "pip install torch" in script
        assert "check_call" in script

    def test_no_setup_cmd(self) -> None:
        script = BasilicaTarget._build_bootstrap_cmd(
            ["python3", "train.py"],
            setup_cmd=None,
        )
        assert "check_call" not in script

    def test_health_port_injected(self) -> None:
        script = BasilicaTarget._build_bootstrap_cmd(["echo", "hi"])
        assert "8080" in script

    def test_script_is_valid_python(self) -> None:
        script = BasilicaTarget._build_bootstrap_cmd(
            ["python3", "train.py"],
            setup_cmd="pip install foo",
        )
        # Verify it compiles without syntax errors
        compile(script, "<bootstrap>", "exec")

    def test_script_without_setup_is_valid_python(self) -> None:
        script = BasilicaTarget._build_bootstrap_cmd(["python3", "train.py"])
        compile(script, "<bootstrap>", "exec")

    def test_complex_setup_cmd(self) -> None:
        setup = (
            "pip install -q transformers datasets "
            "&& mkdir -p /app/data "
            "&& echo done"
        )
        script = BasilicaTarget._build_bootstrap_cmd(
            ["python3", "/app/train.py"],
            setup_cmd=setup,
        )
        compile(script, "<bootstrap>", "exec")
        assert "transformers" in script

    def test_user_cmd_preserved_exactly(self) -> None:
        cmd = ["python3", "train.py", "--epochs", "3", "--lr", "1e-5"]
        script = BasilicaTarget._build_bootstrap_cmd(cmd)
        assert repr(cmd) in script

    def test_post_trial_sleep_default_is_90(self) -> None:
        """Bootstrap stays alive 90s post-trial so the controller can pull
        the model files before container shutdown. Earlier 15s default
        caused HTTP 500/503 on _download_model — see probe5 logs."""
        script = BasilicaTarget._build_bootstrap_cmd(["python3", "train.py"])
        assert "time.sleep(90)" in script

    def test_post_trial_sleep_is_configurable(self) -> None:
        script = BasilicaTarget._build_bootstrap_cmd(
            ["python3", "train.py"], post_trial_sleep_s=300,
        )
        assert "time.sleep(300)" in script
        assert "time.sleep(90)" not in script

    def test_dict_literals_stay_literal(self) -> None:
        # After string.Template refactor, JSON dict literals in the script
        # must round-trip without {{}} escaping.
        script = BasilicaTarget._build_bootstrap_cmd(["python3", "train.py"])
        assert '{"files": [], "model_dir": _model_dir}' in script
        assert '{"path": str(f.relative_to(base)), "size": f.stat().st_size}' in script

    def test_fstring_braces_stay_literal(self) -> None:
        script = BasilicaTarget._build_bootstrap_cmd(["python3", "train.py"])
        assert "f\"[ar] wrote modified source to {_tgt} ({len(_src)} b64 chars)\"" in script

    def test_no_double_braces_remain(self) -> None:
        # If the old .format() escaping leaked, we'd see {{ or }} in the output.
        script = BasilicaTarget._build_bootstrap_cmd(["python3", "train.py"])
        assert "{{" not in script
        assert "}}" not in script

    def test_bootstrap_exposes_progress_endpoint(self) -> None:
        script = BasilicaTarget._build_bootstrap_cmd(["python3", "train.py"])
        assert "/progress" in script
        assert "_serve_progress" in script
        assert 'AR_PROGRESS_FILE' in script

    def test_bootstrap_exposes_control_endpoint(self) -> None:
        script = BasilicaTarget._build_bootstrap_cmd(["python3", "train.py"])
        assert "/control" in script
        assert "_accept_control" in script
        assert 'AR_CONTROL_FILE' in script
        assert "do_POST" in script


class TestBootstrapEnvInjection:
    """ADR-018 contract: env values are inlined into the bootstrap script
    so they survive any Basilica-SDK env-propagation gap with custom
    Docker images (verified live on commit b3fbad1: kubectl exec env
    showed empty AR_* even when env=env was passed to create_deployment)."""

    def test_no_env_yields_empty_inject_dict(self) -> None:
        script = BasilicaTarget._build_bootstrap_cmd(["python3", "train.py"])
        assert "_os.environ.update(json.loads(" in script
        # Empty env still produces a no-op update with {}
        assert "_os.environ.update(json.loads('{}'))" in script

    def test_env_dict_is_inlined_as_json_literal(self) -> None:
        script = BasilicaTarget._build_bootstrap_cmd(
            ["python3", "train.py"],
            env={"AR_MODEL_DIR": "artifacts/test/v0001",
                 "AR_PARAMS_JSON": '{"lr": 0.001}',
                 "AR_PARAM_LR": "0.001"},
        )
        # The whole values must round-trip into the script as a json string
        assert "AR_MODEL_DIR" in script
        assert "artifacts/test/v0001" in script
        assert "AR_PARAM_LR" in script
        # The injection happens BEFORE _model_dir is read
        inject_pos = script.index("_os.environ.update(json.loads(")
        read_pos = script.index('_model_dir = _os.environ.get("AR_MODEL_DIR"')
        assert inject_pos < read_pos, (
            "env_inject must precede the AR_MODEL_DIR read; otherwise "
            "the bootstrap server's _model_dir is empty when the update "
            "runs and the model-files endpoint returns no files"
        )

    def test_env_with_special_chars_round_trips(self) -> None:
        # Backslashes, quotes, unicode — must survive json.dumps + eval.
        script = BasilicaTarget._build_bootstrap_cmd(
            ["python3", "train.py"],
            env={"AR_PARAMS_JSON": '{"path": "C:\\\\x", "name": "α"}'},
        )
        # Round-trip the env back out by extracting the json payload.
        import json
        marker = "_os.environ.update(json.loads("
        idx = script.index(marker) + len(marker)
        end = script.index("))", idx)
        repr_payload = script[idx:end]
        # repr_payload is like  '{"AR_PARAMS_JSON": "..."}'  (a Python string literal)
        decoded = json.loads(eval(repr_payload))
        assert decoded["AR_PARAMS_JSON"] == '{"path": "C:\\\\x", "name": "α"}'


class TestPropagateControl:
    """_propagate_control uploads run_dir/control.json to deployment /control."""

    def _build_target(self) -> BasilicaTarget:
        # Bypass __init__ (which requires basilica-sdk) by allocating directly.
        target = BasilicaTarget.__new__(BasilicaTarget)
        target._client = None  # type: ignore[attr-defined]
        target._cfg = None  # type: ignore[attr-defined]
        target._bcfg = None  # type: ignore[attr-defined]
        target._last_train_outcome = None  # type: ignore[attr-defined]
        return target

    def test_no_control_file_is_noop(self, tmp_path) -> None:
        from unittest.mock import MagicMock, patch

        target = self._build_target()
        deployment = MagicMock()
        deployment.url = "http://example/"
        with patch("urllib.request.urlopen") as mock_urlopen:
            target._propagate_control(deployment, str(tmp_path))
        mock_urlopen.assert_not_called()

    def test_uploads_when_control_present(self, tmp_path) -> None:
        from unittest.mock import MagicMock, patch

        import json
        target = self._build_target()
        deployment = MagicMock()
        deployment.url = "http://example/"
        ctrl = tmp_path / "control.json"
        ctrl.write_text(json.dumps({"action": "cancel", "reason": "test"}))

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_resp.read = MagicMock(return_value=b"")
            mock_urlopen.return_value = mock_resp
            target._propagate_control(deployment, str(tmp_path))

        assert mock_urlopen.call_count == 1
        req = mock_urlopen.call_args[0][0]
        assert req.full_url.endswith("/control")
        assert req.method == "POST"
        assert b"cancel" in req.data

    def test_skips_duplicate_uploads_same_size(self, tmp_path) -> None:
        from unittest.mock import MagicMock, patch

        import json
        target = self._build_target()
        deployment = MagicMock()
        deployment.url = "http://example/"
        ctrl = tmp_path / "control.json"
        ctrl.write_text(json.dumps({"action": "cancel"}))

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_resp.read = MagicMock(return_value=b"")
            mock_urlopen.return_value = mock_resp
            target._propagate_control(deployment, str(tmp_path))
            target._propagate_control(deployment, str(tmp_path))
        # Second call must NOT re-upload (content hash cached).
        assert mock_urlopen.call_count == 1

    def test_reuploads_when_content_changes_at_same_length(self, tmp_path) -> None:
        """Edit reason text to a same-length string -> hash differs -> re-upload."""
        from unittest.mock import MagicMock, patch

        import json
        target = self._build_target()
        deployment = MagicMock()
        deployment.url = "http://example/"
        ctrl = tmp_path / "control.json"

        # Two payloads of identical byte length, different content.
        first = json.dumps({"action": "cancel", "reason": "AAAAA"})
        second = json.dumps({"action": "cancel", "reason": "BBBBB"})
        assert len(first) == len(second), "test fixture broken: lengths differ"

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_resp.read = MagicMock(return_value=b"")
            mock_urlopen.return_value = mock_resp

            ctrl.write_text(first)
            target._propagate_control(deployment, str(tmp_path))
            ctrl.write_text(second)
            target._propagate_control(deployment, str(tmp_path))

        # Must have uploaded BOTH (content differs even though length is same).
        assert mock_urlopen.call_count == 2, (
            "size-cache regression: equal-length payload edit was silently dropped"
        )

    def test_upload_failure_is_swallowed(self, tmp_path) -> None:
        from unittest.mock import MagicMock, patch

        import json
        target = self._build_target()
        deployment = MagicMock()
        deployment.url = "http://example/"
        ctrl = tmp_path / "control.json"
        ctrl.write_text(json.dumps({"action": "cancel"}))

        with patch("urllib.request.urlopen", side_effect=ConnectionError("boom")):
            target._propagate_control(deployment, str(tmp_path))  # must not raise


class TestRunEvalCachePerRunDir:
    """Race-free run()/eval() cache (fix for parallel engine + Basilica)."""

    def _build_target(self) -> BasilicaTarget:
        target = BasilicaTarget.__new__(BasilicaTarget)
        target._client = None  # type: ignore[attr-defined]
        # Replicate __init__ tail explicitly so we don't need basilica-sdk
        # available in the test env.
        import threading
        target._cfg = None  # type: ignore[attr-defined]
        target._bcfg = None  # type: ignore[attr-defined]
        target._last_train_outcome = {}  # type: ignore[attr-defined]
        target._cache_lock = threading.Lock()  # type: ignore[attr-defined]
        return target

    def test_eval_returns_per_run_dir_outcome_not_global_last(self) -> None:
        """Pre-fix: eval() returned the LAST run() outcome regardless of run_dir.
        Under parallel mode that meant Thread A could see Thread B's metrics.
        """
        from unittest.mock import MagicMock

        target = self._build_target()
        target._cfg = MagicMock()
        target._cfg.eval_cmd = None  # the no-eval-cmd code path

        from autojepa.target.interface import RunOutcome

        out_a = RunOutcome(
            status="ok", metrics={"score": 0.4}, stdout="A", stderr="",
            elapsed_s=10.0, run_dir="/run-A",
        )
        out_b = RunOutcome(
            status="ok", metrics={"score": 0.7}, stdout="B", stderr="",
            elapsed_s=20.0, run_dir="/run-B",
        )
        # Simulate two parallel run() calls landing outcomes into the cache.
        target._last_train_outcome["/run-A"] = out_a
        target._last_train_outcome["/run-B"] = out_b

        # Each eval() must return ITS run_dir's outcome, not the most-recent.
        eval_a = target.eval(run_dir="/run-A", params={})
        eval_b = target.eval(run_dir="/run-B", params={})
        assert eval_a.metrics == {"score": 0.4}, eval_a
        assert eval_b.metrics == {"score": 0.7}, eval_b
        # Cache is consumed on eval (pop), so the dict is empty after both.
        assert target._last_train_outcome == {}

    def test_eval_returns_empty_when_no_cached_run(self) -> None:
        from unittest.mock import MagicMock

        target = self._build_target()
        target._cfg = MagicMock()
        target._cfg.eval_cmd = None
        outcome = target.eval(run_dir="/no-prior-run", params={})
        assert outcome.status == "ok"
        assert outcome.metrics == {}


class TestFetchOutcome:
    """ADR-015 outcome.json contract: adapter pulls outcome via /model/files."""

    def _build_target(self) -> BasilicaTarget:
        target = BasilicaTarget.__new__(BasilicaTarget)
        target._client = None  # type: ignore[attr-defined]
        target._cfg = None  # type: ignore[attr-defined]
        target._bcfg = None  # type: ignore[attr-defined]
        return target

    def _mock_resp(self, body: bytes):
        from unittest.mock import MagicMock
        resp = MagicMock()
        resp.read = MagicMock(return_value=body)
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        return resp

    def test_returns_payload_when_present(self) -> None:
        from unittest.mock import MagicMock, patch

        target = self._build_target()
        deployment = MagicMock()
        deployment.url = "http://example/"

        listing = json.dumps({
            "files": [
                {"path": "outcome.json", "size": 80},
                {"path": "model.pt", "size": 4096},
            ],
            "model_dir": "/m",
        }).encode()
        outcome_payload = json.dumps({
            "status": "ok",
            "metrics": {"probe_auroc": 0.281, "loss": 0.008},
            "elapsed_s": 1417,
            "completed_steps": 4000,
            "step_target": 4000,
            "ts": 1778900000,
        }).encode()

        with patch(
            "autojepa.target.basilica.urllib.request.urlopen",
            side_effect=[self._mock_resp(listing), self._mock_resp(outcome_payload)],
        ):
            payload = target._fetch_outcome(deployment)
        assert payload is not None
        assert payload["status"] == "ok"
        assert payload["metrics"]["probe_auroc"] == 0.281

    def test_returns_none_when_outcome_absent(self) -> None:
        from unittest.mock import MagicMock, patch

        target = self._build_target()
        deployment = MagicMock()
        deployment.url = "http://example/"

        listing = json.dumps({"files": [{"path": "model.pt", "size": 4096}]}).encode()
        with patch(
            "autojepa.target.basilica.urllib.request.urlopen",
            return_value=self._mock_resp(listing),
        ):
            payload = target._fetch_outcome(deployment)
        assert payload is None

    def test_returns_none_on_listing_error(self) -> None:
        from unittest.mock import MagicMock, patch

        target = self._build_target()
        deployment = MagicMock()
        deployment.url = "http://example/"

        with patch(
            "autojepa.target.basilica.urllib.request.urlopen",
            side_effect=ConnectionError("boom"),
        ):
            payload = target._fetch_outcome(deployment)
        assert payload is None

    def test_returns_none_on_malformed_payload(self) -> None:
        from unittest.mock import MagicMock, patch

        target = self._build_target()
        deployment = MagicMock()
        deployment.url = "http://example/"

        listing = json.dumps({"files": [{"path": "outcome.json", "size": 80}]}).encode()
        with patch(
            "autojepa.target.basilica.urllib.request.urlopen",
            side_effect=[self._mock_resp(listing), self._mock_resp(b"not-json{")],
        ):
            payload = target._fetch_outcome(deployment)
        assert payload is None

    def test_handles_nested_outcome_path(self) -> None:
        from unittest.mock import MagicMock, patch

        target = self._build_target()
        deployment = MagicMock()
        deployment.url = "http://example/"

        listing = json.dumps({"files": [{"path": "subdir/outcome.json", "size": 80}]}).encode()
        outcome_payload = json.dumps({"status": "failed", "reason": "x", "metrics": {}}).encode()
        with patch(
            "autojepa.target.basilica.urllib.request.urlopen",
            side_effect=[self._mock_resp(listing), self._mock_resp(outcome_payload)],
        ):
            payload = target._fetch_outcome(deployment)
        assert payload is not None
        assert payload["status"] == "failed"


class TestFinalizeOutcome:
    """_finalize_outcome converts a payload to RunOutcome and cleans up."""

    def _build_target(self) -> BasilicaTarget:
        target = BasilicaTarget.__new__(BasilicaTarget)
        target._client = None  # type: ignore[attr-defined]
        target._cfg = None  # type: ignore[attr-defined]
        target._bcfg = None  # type: ignore[attr-defined]
        return target

    def test_ok_status_propagates(self, tmp_path) -> None:
        from unittest.mock import MagicMock, patch

        target = self._build_target()
        deployment = MagicMock()
        deployment.url = "http://example/"
        deployment.logs = MagicMock(return_value="")
        deployment.delete = MagicMock()

        outcome = {
            "status": "ok",
            "metrics": {"probe_auroc": 0.31, "loss": 0.01},
            "elapsed_s": 100.0,
            "completed_steps": 4000,
            "step_target": 4000,
            "ts": 1,
        }
        # Stub _download_model so this stays a unit test
        with patch.object(target, "_download_model", return_value=None):
            result = target._finalize_outcome(
                deployment, "ar-test", t0=0.0, run_dir=str(tmp_path),
                outcome=outcome,
            )
        assert result.status == "ok"
        assert result.metrics["probe_auroc"] == 0.31
        deployment.delete.assert_called_once()

    def test_failed_status_with_reason(self, tmp_path) -> None:
        from unittest.mock import MagicMock, patch

        target = self._build_target()
        deployment = MagicMock()
        deployment.url = "http://example/"
        deployment.logs = MagicMock(return_value="")
        deployment.delete = MagicMock()

        outcome = {
            "status": "failed",
            "metrics": {"canary_loss": 0.06},
            "elapsed_s": 60.0,
            "reason": "canary_loss=0.0600 > threshold=0.05",
            "ts": 1,
        }
        with patch.object(target, "_download_model", return_value=None):
            result = target._finalize_outcome(
                deployment, "ar-test", t0=0.0, run_dir=str(tmp_path),
                outcome=outcome,
            )
        assert result.status == "failed"
        assert "canary_loss" in result.metrics
        assert "canary_loss=0.0600" in result.stderr

    def test_non_numeric_metrics_skipped(self, tmp_path) -> None:
        from unittest.mock import MagicMock, patch

        target = self._build_target()
        deployment = MagicMock()
        deployment.url = "http://example/"
        deployment.logs = MagicMock(return_value="")
        deployment.delete = MagicMock()

        outcome = {
            "status": "ok",
            "metrics": {"probe_auroc": 0.4, "label": "ok-string"},
            "elapsed_s": 1.0,
            "ts": 1,
        }
        with patch.object(target, "_download_model", return_value=None):
            result = target._finalize_outcome(
                deployment, "ar-test", t0=0.0, run_dir=str(tmp_path),
                outcome=outcome,
            )
        assert "probe_auroc" in result.metrics
        assert "label" not in result.metrics
