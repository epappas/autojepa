from __future__ import annotations

import json
import random
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from autojepa.policy.llm_search import (
    LLMParamPolicy,
    _call_chat_api_messages,
    _coerce_value,
    _format_prompt,
    _normalise_models,
    _parse_response,
    _random_fallback,
)


SPACE = {"learning_rate": [0.00001, 0.00002, 0.00003], "epochs": [1, 2, 3]}


# --- _format_prompt ---


def test_format_prompt_with_program():
    prompt = _format_prompt(SPACE, [], "val_bpb", "min", program="Train a small LM on wikitext.")
    assert prompt.startswith("Task specification:")
    assert "Train a small LM on wikitext." in prompt
    assert "Objective:" in prompt
    # program comes before objective
    assert prompt.index("Task specification:") < prompt.index("Objective:")


def test_format_prompt_empty_program_omits_section():
    prompt = _format_prompt(SPACE, [], "val_bpb", "min", program="")
    assert "Task specification:" not in prompt
    prompt2 = _format_prompt(SPACE, [], "val_bpb", "min")
    assert "Task specification:" not in prompt2


def test_format_prompt_with_history():
    history = [
        {"params": {"learning_rate": 0.00001, "epochs": 1}, "metrics": {"val_bpb": 1.5}, "status": "ok"},
        {"params": {"learning_rate": 0.00002, "epochs": 2}, "metrics": {"val_bpb": 1.3}, "status": "ok"},
    ]
    prompt = _format_prompt(SPACE, history, "val_bpb", "min")
    assert "minimize" in prompt.lower()
    assert "learning_rate" in prompt
    assert "1.5" in prompt
    assert "1.3" in prompt
    assert "last 2 of 2" in prompt


def test_format_prompt_empty_history():
    prompt = _format_prompt(SPACE, [], "val_bpb", "min")
    assert "No experiment history" in prompt
    assert "learning_rate" in prompt


def test_format_prompt_truncation():
    history = [
        {"params": {"learning_rate": 0.00001, "epochs": 1}, "metrics": {"val_bpb": float(i)}, "status": "ok"}
        for i in range(100)
    ]
    prompt = _format_prompt(SPACE, history, "val_bpb", "min")
    assert "last 50 of 100" in prompt


def test_format_prompt_max_direction():
    prompt = _format_prompt(SPACE, [], "accuracy", "max")
    assert "maximize" in prompt.lower()


# --- _parse_response ---


def test_parse_valid_json():
    raw = '{"learning_rate": 0.00002, "epochs": 2}'
    result = _parse_response(raw, SPACE)
    assert result == {"learning_rate": 0.00002, "epochs": 2}


def test_parse_markdown_fences():
    raw = '```json\n{"learning_rate": 0.00001, "epochs": 3}\n```'
    result = _parse_response(raw, SPACE)
    assert result == {"learning_rate": 0.00001, "epochs": 3}


def test_parse_markdown_fences_no_lang():
    raw = '```\n{"learning_rate": 0.00003, "epochs": 1}\n```'
    result = _parse_response(raw, SPACE)
    assert result == {"learning_rate": 0.00003, "epochs": 1}


def test_parse_string_coercion():
    raw = '{"learning_rate": "0.00002", "epochs": "2"}'
    result = _parse_response(raw, SPACE)
    assert result == {"learning_rate": 0.00002, "epochs": 2}


def test_parse_missing_key():
    raw = '{"learning_rate": 0.00002}'
    with pytest.raises(ValueError, match="Missing key"):
        _parse_response(raw, SPACE)


def test_parse_invalid_value():
    raw = '{"learning_rate": 0.99, "epochs": 2}'
    with pytest.raises(ValueError, match="not in allowed"):
        _parse_response(raw, SPACE)


def test_parse_no_json():
    with pytest.raises(ValueError, match="No JSON object"):
        _parse_response("I think you should try these params", SPACE)


def test_parse_surrounding_text():
    raw = 'Here is my suggestion:\n{"learning_rate": 0.00001, "epochs": 1}\nGood luck!'
    result = _parse_response(raw, SPACE)
    assert result == {"learning_rate": 0.00001, "epochs": 1}


# --- _coerce_value ---


def test_coerce_direct_match():
    assert _coerce_value(0.00002, [0.00001, 0.00002]) == 0.00002


def test_coerce_string_to_float():
    assert _coerce_value("0.00002", [0.00001, 0.00002]) == 0.00002


def test_coerce_string_to_int():
    assert _coerce_value("2", [1, 2, 3]) == 2


def test_coerce_float_to_int():
    assert _coerce_value(2.0, [1, 2, 3]) == 2


def test_coerce_bool():
    assert _coerce_value("true", [True, False]) is True
    assert _coerce_value("false", [True, False]) is False


def test_coerce_no_match():
    assert _coerce_value(999, [1, 2, 3]) is None


def test_coerce_string_values():
    assert _coerce_value("adam", ["adam", "sgd"]) == "adam"


# --- _random_fallback ---


def test_random_fallback_reproducible():
    rng1 = random.Random(42)
    rng2 = random.Random(42)
    p1 = _random_fallback(SPACE, rng1)
    p2 = _random_fallback(SPACE, rng2)
    assert p1.params == p2.params
    assert p1.rationale == "llm-fallback-random"


def test_random_fallback_values_in_space():
    rng = random.Random(0)
    for _ in range(20):
        p = _random_fallback(SPACE, rng)
        for k, v in p.params.items():
            assert v in SPACE[k]


# --- LLMParamPolicy.next ---


def _mock_urlopen_response(content: str):
    body = json.dumps({
        "choices": [{"message": {"content": content}}]
    }).encode("utf-8")
    resp = MagicMock()
    resp.read.return_value = body
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    return resp


class TestLLMParamPolicyNext:

    def _make_policy(self, **kwargs):
        defaults = {
            "api_url": "http://localhost:8000/v1",
            "model": "test-model",
            "api_key_env": "TEST_LLM_KEY",
            "seed": 42,
        }
        defaults.update(kwargs)
        return LLMParamPolicy(SPACE, **defaults)

    def test_success(self):
        policy = self._make_policy()
        response = '{"learning_rate": 0.00002, "epochs": 2}'
        with (
            patch.dict("os.environ", {"TEST_LLM_KEY": "sk-test"}),
            patch("autojepa.policy.llm_search.urllib.request.urlopen")
            as mock_urlopen,
        ):
            mock_urlopen.return_value = _mock_urlopen_response(response)
            proposal = policy.propose({"history": []})

        assert proposal.params == {"learning_rate": 0.00002, "epochs": 2}
        assert proposal.rationale == "llm"

    def test_api_error_falls_back(self):
        policy = self._make_policy()
        with (
            patch.dict("os.environ", {"TEST_LLM_KEY": "sk-test"}),
            patch("autojepa.policy.llm_search.urllib.request.urlopen")
            as mock_urlopen,
        ):
            mock_urlopen.side_effect = Exception("API timeout")
            proposal = policy.propose({"history": []})

        assert proposal.rationale == "llm-fallback-random"
        for k, v in proposal.params.items():
            assert v in SPACE[k]

    def test_parse_error_falls_back(self):
        policy = self._make_policy()
        response = "I cannot help with that"
        with (
            patch.dict("os.environ", {"TEST_LLM_KEY": "sk-test"}),
            patch("autojepa.policy.llm_search.urllib.request.urlopen")
            as mock_urlopen,
        ):
            mock_urlopen.return_value = _mock_urlopen_response(response)
            proposal = policy.propose({"history": []})

        assert proposal.rationale == "llm-fallback-random"

    def test_missing_api_key_falls_back(self):
        policy = self._make_policy()
        with patch.dict("os.environ", {}, clear=True):
            proposal = policy.propose({"history": []})

        assert proposal.rationale == "llm-fallback-random"

    def test_api_call_sends_correct_payload(self):
        policy = self._make_policy()
        response = '{"learning_rate": 0.00001, "epochs": 1}'
        with (
            patch.dict("os.environ", {"TEST_LLM_KEY": "sk-test"}),
            patch("autojepa.policy.llm_search.urllib.request.urlopen")
            as mock_urlopen,
        ):
            mock_urlopen.return_value = _mock_urlopen_response(response)
            policy.propose({"history": []})

            req = mock_urlopen.call_args[0][0]
            assert req.full_url == "http://localhost:8000/v1/chat/completions"
            assert req.get_header("Authorization") == "Bearer sk-test"
            assert req.get_header("Content-type") == "application/json"
            body = json.loads(req.data)
            assert body["model"] == "test-model"
            assert len(body["messages"]) == 2
            assert body["messages"][0]["role"] == "system"
            assert body["messages"][1]["role"] == "user"

    def test_with_history(self):
        policy = self._make_policy()
        history = [
            {"params": {"learning_rate": 0.00001, "epochs": 1}, "metrics": {"val_bpb": 1.5}, "status": "ok"},
        ]
        response = '{"learning_rate": 0.00003, "epochs": 3}'
        with (
            patch.dict("os.environ", {"TEST_LLM_KEY": "sk-test"}),
            patch("autojepa.policy.llm_search.urllib.request.urlopen")
            as mock_urlopen,
        ):
            mock_urlopen.return_value = _mock_urlopen_response(response)
            proposal = policy.propose({"history": history})

        assert proposal.params == {"learning_rate": 0.00003, "epochs": 3}

    def test_program_flows_to_api_prompt(self):
        policy = self._make_policy()
        response = '{"learning_rate": 0.00002, "epochs": 2}'
        with (
            patch.dict("os.environ", {"TEST_LLM_KEY": "sk-test"}),
            patch("autojepa.policy.llm_search.urllib.request.urlopen")
            as mock_urlopen,
        ):
            mock_urlopen.return_value = _mock_urlopen_response(response)
            policy.propose({"history": [], "program": "Minimize perplexity on wikitext."})

            req = mock_urlopen.call_args[0][0]
            body = json.loads(req.data)
            user_msg = body["messages"][1]["content"]
            assert "Task specification:" in user_msg
            assert "Minimize perplexity on wikitext." in user_msg

    def test_no_program_omits_section_in_api(self):
        policy = self._make_policy()
        response = '{"learning_rate": 0.00002, "epochs": 2}'
        with (
            patch.dict("os.environ", {"TEST_LLM_KEY": "sk-test"}),
            patch("autojepa.policy.llm_search.urllib.request.urlopen")
            as mock_urlopen,
        ):
            mock_urlopen.return_value = _mock_urlopen_response(response)
            policy.propose({"history": []})

            req = mock_urlopen.call_args[0][0]
            body = json.loads(req.data)
            user_msg = body["messages"][1]["content"]
            assert "Task specification:" not in user_msg

    def test_fallback_randomness_advances(self):
        """Two consecutive fallbacks should return different params (most of the time)."""
        policy = self._make_policy()
        results = []
        with patch.dict("os.environ", {}, clear=True):
            for _ in range(10):
                results.append(policy.propose({"history": []}).params)
        # With 10 draws from a small space, we should see at least 2 distinct combos
        unique = {tuple(sorted(r.items())) for r in results}
        assert len(unique) >= 2


# --- ADR-017: model-name fallback list ---


class TestNormaliseModels:
    def test_single_string(self) -> None:
        assert _normalise_models("foo") == ["foo"]

    def test_comma_separated_string(self) -> None:
        assert _normalise_models("foo, bar,baz") == ["foo", "bar", "baz"]

    def test_explicit_list(self) -> None:
        assert _normalise_models(["a", "b", "c"]) == ["a", "b", "c"]

    def test_drops_empties(self) -> None:
        assert _normalise_models([" ", "a", ""]) == ["a"]
        assert _normalise_models("a,, ,b") == ["a", "b"]

    def test_empty_list_raises(self) -> None:
        with pytest.raises(ValueError, match="empty list"):
            _normalise_models([])

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError, match="empty list"):
            _normalise_models("   ")


class TestCallChatApiFallback:
    """ADR-017: 404 on a model triggers fallback to the next list entry."""

    def _make_404(self, body: bytes = b'{"error": "model not found"}') -> urllib.error.HTTPError:
        from io import BytesIO
        return urllib.error.HTTPError(
            url="http://example/v1/chat/completions",
            code=404,
            msg="Not Found",
            hdrs={},  # type: ignore[arg-type]
            fp=BytesIO(body),
        )

    def _make_ok(self, content: str):
        body = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
        resp = MagicMock()
        resp.read = MagicMock(return_value=body)
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        return resp

    def test_falls_back_to_next_model_on_404(self) -> None:
        """First model 404s, second succeeds — caller sees the second's response."""
        models = ["primary-404", "secondary-ok"]
        with patch(
            "autojepa.policy.llm_search.urllib.request.urlopen",
            side_effect=[self._make_404(), self._make_ok("hello world")],
        ) as mock_urlopen:
            result = _call_chat_api_messages(
                "http://example/v1", models, "sk-test",
                [{"role": "user", "content": "x"}], timeout=5,
                max_retries=0,
            )
        assert result == "hello world"
        # Two HTTP calls: first 404 on primary, second OK on secondary.
        assert mock_urlopen.call_count == 2
        first_payload = json.loads(mock_urlopen.call_args_list[0][0][0].data)
        second_payload = json.loads(mock_urlopen.call_args_list[1][0][0].data)
        assert first_payload["model"] == "primary-404"
        assert second_payload["model"] == "secondary-ok"

    def test_all_models_404_propagates(self) -> None:
        """No remaining candidate -> last 404 re-raised so caller's seeded-random fires."""
        models = ["first-404", "second-404"]
        with patch(
            "autojepa.policy.llm_search.urllib.request.urlopen",
            side_effect=[self._make_404(), self._make_404()],
        ):
            with pytest.raises(urllib.error.HTTPError) as exc_info:
                _call_chat_api_messages(
                    "http://example/v1", models, "sk-test",
                    [{"role": "user", "content": "x"}], timeout=5,
                    max_retries=0,
                )
        assert exc_info.value.code == 404

    def test_single_string_model_back_compat(self) -> None:
        """The legacy single-string config form still works."""
        with patch(
            "autojepa.policy.llm_search.urllib.request.urlopen",
            return_value=self._make_ok("ok"),
        ) as mock_urlopen:
            result = _call_chat_api_messages(
                "http://example/v1", "only-one", "sk-test",
                [{"role": "user", "content": "x"}], timeout=5,
                max_retries=0,
            )
        assert result == "ok"
        payload = json.loads(mock_urlopen.call_args[0][0].data)
        assert payload["model"] == "only-one"

    def test_500_does_not_trigger_fallback(self) -> None:
        """5xx errors should retry (or propagate) on the SAME model, not advance."""
        from io import BytesIO

        models = ["primary", "secondary"]
        err500 = urllib.error.HTTPError(
            url="http://example/v1/chat/completions",
            code=500, msg="Server Error", hdrs={},  # type: ignore[arg-type]
            fp=BytesIO(b""),
        )
        with patch(
            "autojepa.policy.llm_search.urllib.request.urlopen",
            side_effect=err500,
        ) as mock_urlopen:
            with pytest.raises(urllib.error.HTTPError) as exc_info:
                _call_chat_api_messages(
                    "http://example/v1", models, "sk-test",
                    [{"role": "user", "content": "x"}], timeout=5,
                    max_retries=0,
                )
        assert exc_info.value.code == 500
        # Exactly one call — no fallback to "secondary".
        assert mock_urlopen.call_count == 1


class TestPolicyAcceptsModelList:
    """LLMParamPolicy must accept the list form end-to-end."""

    def test_init_accepts_list(self) -> None:
        policy = LLMParamPolicy(
            SPACE,
            api_url="http://x/v1",
            model=["m1", "m2"],
            api_key_env="TEST_LLM_KEY",
            seed=1,
        )
        assert policy._model == ["m1", "m2"]

    def test_init_accepts_string(self) -> None:
        policy = LLMParamPolicy(
            SPACE,
            api_url="http://x/v1",
            model="single",
            api_key_env="TEST_LLM_KEY",
            seed=1,
        )
        assert policy._model == "single"
