"""Regression tests for LLMDiffPolicy conversation hygiene (ADR-025).

Before this fix, v30 iters 9, 10, 11 all proposed essentially the
same CosineAnnealingLR diff after iter=4 was kept, because the
multi-turn conversation anchored on iter=4's success and refused to
explore alternatives. The fallback to GreedyLLMPolicy then fired,
masking the policy's failure to diversify.

Two complementary defences:
1. Reset conversation history when a new kept diff appears in
   `state['history']` — the LLM re-reasons from the patched baseline
   instead of building on stale context.
2. Inject a "PREVIOUSLY PROPOSED APPROACHES (DO NOT propose any of
   these again)" section listing one-line summaries of recent diffs.
   Steers the LLM away from repetition even within a single
   conversation (e.g. across retry attempts on validation failure).
"""
from __future__ import annotations

from unittest.mock import patch

from autojepa.policy.llm_diff import (
    LLMDiffPolicy,
    _extract_prior_diff_summaries,
    _format_diff_prompt,
    _last_kept_diff_iter,
    _summarize_diff,
)


SAMPLE_DIFF_COSINE = """\
--- a/train.py
+++ b/train.py
@@ -10,3 +10,7 @@
 import torch
+scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer)
+scheduler.step()
"""

SAMPLE_DIFF_VICREG = """\
--- a/train.py
+++ b/train.py
@@ -10,3 +10,5 @@
 import torch
+from autojepa.models.losses import LOSS_REGISTRY
+loss_fn = LOSS_REGISTRY["vicreg"](var_coef=25.0)
"""

SAMPLE_SOURCE = """\
import torch
LEARNING_RATE = 0.0026
"""


# --- _summarize_diff ---


def test_summarize_diff_picks_first_substantive_add():
    summary = _summarize_diff(SAMPLE_DIFF_COSINE)
    assert "CosineAnnealingLR" in summary
    # The summary must not include the +++ header line.
    assert summary != "+ b/train.py"


def test_summarize_diff_handles_empty():
    assert _summarize_diff("") == "<empty>"


def test_summarize_diff_handles_context_only():
    ctx_only = "--- a/x\n+++ b/x\n@@ -1 +1 @@\n unchanged\n"
    assert _summarize_diff(ctx_only) == "<context-only>"


def test_summarize_diff_skips_comment_added_lines():
    """A pure-comment addition shouldn't be the gist — pick the next
    line. This prevents `# fix bug` from masking the real change."""
    diff = (
        "--- a/x\n+++ b/x\n@@ -1 +1,3 @@\n"
        "+# just a comment\n+lr = 0.001\n"
    )
    summary = _summarize_diff(diff)
    assert "lr = 0.001" in summary


# --- _extract_prior_diff_summaries ---


def test_extract_prior_diff_summaries_dedups_repeats():
    """v30 iters 9, 10, 11 emitted the SAME diff three times. The
    prior-approaches list must dedupe so the prompt isn't spammed."""
    history = [
        {"iter": 4, "status": "ok", "decision": "keep",
         "params": {"_type": "diff", "diff": SAMPLE_DIFF_COSINE}},
        {"iter": 9, "status": "failed", "decision": "discard",
         "params": {"_type": "diff", "diff": SAMPLE_DIFF_COSINE}},
        {"iter": 10, "status": "failed", "decision": "discard",
         "params": {"_type": "diff", "diff": SAMPLE_DIFF_COSINE}},
    ]
    summaries = _extract_prior_diff_summaries(history)
    assert len(summaries) == 1, (
        f"three identical diffs must collapse to one summary; got {summaries}"
    )


def test_extract_prior_diff_summaries_skips_param_iters():
    """Only diff iters count; param iters shouldn't pollute the list."""
    history = [
        {"iter": 0, "status": "ok", "decision": "keep",
         "params": {"_type": "param", "lr": 0.001}},
        {"iter": 1, "status": "ok", "decision": "keep",
         "params": {"_type": "diff", "diff": SAMPLE_DIFF_VICREG}},
    ]
    summaries = _extract_prior_diff_summaries(history)
    assert len(summaries) == 1
    # First added line in the vicreg sample is `from ... LOSS_REGISTRY` —
    # that's the gist (and is distinctive enough to dedupe against).
    assert "LOSS_REGISTRY" in summaries[0]


def test_extract_prior_diff_summaries_orders_recent_first():
    history = [
        {"iter": 0, "status": "ok", "decision": "discard",
         "params": {"_type": "diff", "diff": SAMPLE_DIFF_VICREG}},
        {"iter": 1, "status": "ok", "decision": "keep",
         "params": {"_type": "diff", "diff": SAMPLE_DIFF_COSINE}},
    ]
    summaries = _extract_prior_diff_summaries(history)
    # Most-recent first: iter=1 (cosine) before iter=0 (vicreg).
    assert "CosineAnnealingLR" in summaries[0]
    assert "LOSS_REGISTRY" in summaries[1]


# --- _last_kept_diff_iter ---


def test_last_kept_diff_iter_finds_latest_kept():
    history = [
        {"iter": 0, "status": "ok", "decision": "keep",
         "params": {"_type": "diff", "diff": SAMPLE_DIFF_VICREG}},
        {"iter": 1, "status": "failed", "decision": "discard",
         "params": {"_type": "diff", "diff": SAMPLE_DIFF_COSINE}},
        {"iter": 2, "status": "ok", "decision": "keep",
         "params": {"_type": "diff", "diff": SAMPLE_DIFF_COSINE}},
    ]
    assert _last_kept_diff_iter(history) == 2


def test_last_kept_diff_iter_returns_none_when_no_kept_diff():
    history = [
        {"iter": 0, "status": "ok", "decision": "keep",
         "params": {"_type": "param", "lr": 0.001}},
        {"iter": 1, "status": "failed", "decision": "discard",
         "params": {"_type": "diff", "diff": SAMPLE_DIFF_COSINE}},
    ]
    assert _last_kept_diff_iter(history) is None


# --- _format_diff_prompt with prior_approaches ---


def test_format_prompt_includes_prior_approaches_section():
    prompt = _format_diff_prompt(
        source=SAMPLE_SOURCE,
        filename="train.py",
        history=[],
        metric="probe_auroc",
        direction="max",
        prior_approaches=["iter=4 status=ok decision=keep :: CosineAnnealingLR"],
    )
    assert "PREVIOUSLY PROPOSED APPROACHES" in prompt
    assert "CosineAnnealingLR" in prompt
    assert "DO NOT propose any of these again" in prompt


def test_format_prompt_omits_prior_approaches_when_empty():
    prompt = _format_diff_prompt(
        source=SAMPLE_SOURCE, filename="train.py", history=[],
        metric="probe_auroc", direction="max", prior_approaches=[],
    )
    assert "PREVIOUSLY PROPOSED APPROACHES" not in prompt


# --- LLMDiffPolicy: reset after kept diff ---


_PATCH = "autojepa.policy.llm_diff._call_chat_api_messages"


def _make_policy(**kwargs):
    defaults = {
        "mutable_file": "/tmp/test_train_hygiene.py",
        "api_url": "http://localhost:8000/v1",
        "model": "test-model",
        "api_key_env": "TEST_LLM_KEY",
        "seed": 42,
    }
    defaults.update(kwargs)
    return LLMDiffPolicy(**defaults)


def test_policy_resets_conversation_after_new_kept_diff():
    """When `history` shows a kept diff iter that the policy hasn't
    yet seen, the next propose() call MUST start fresh — the prior
    conversation about pre-keep baselines is now stale."""
    policy = _make_policy()
    # Seed a non-empty conversation as if prior iters were proposed.
    policy._conversation = [
        {"role": "user", "content": "old stuff"},
        {"role": "assistant", "content": "old diff"},
    ]
    assert policy._last_seen_kept_diff_iter is None

    history = [
        {"iter": 4, "status": "ok", "decision": "keep",
         "params": {"_type": "diff", "diff": SAMPLE_DIFF_COSINE}},
    ]

    with (
        patch.dict("os.environ", {"TEST_LLM_KEY": "sk-test"}),
        patch(_PATCH, return_value=SAMPLE_DIFF_VICREG),
    ):
        proposal = policy.propose({
            "history": history, "source": SAMPLE_SOURCE,
        })

    assert proposal.rationale == "llm-diff"
    # _last_seen_kept_diff_iter must be updated so we don't re-reset.
    assert policy._last_seen_kept_diff_iter == 4
    # Conversation should hold exactly one fresh (user, assistant) pair
    # for the just-completed propose call — not the seeded two-pair
    # stale conversation plus the new one.
    assert len(policy._conversation) == 2
    assert "old stuff" not in policy._conversation[0]["content"]


def test_policy_does_not_reset_when_no_new_keep():
    """If history shows the same kept-diff iter we've already
    acknowledged, do NOT clear — we want conversation continuity."""
    policy = _make_policy()
    policy._conversation = [
        {"role": "user", "content": "msg1"},
        {"role": "assistant", "content": "rsp1"},
    ]
    policy._last_seen_kept_diff_iter = 4  # already acknowledged

    history = [
        {"iter": 4, "status": "ok", "decision": "keep",
         "params": {"_type": "diff", "diff": SAMPLE_DIFF_COSINE}},
        {"iter": 5, "status": "failed", "decision": "discard",
         "params": {"_type": "diff", "diff": SAMPLE_DIFF_VICREG}},
    ]

    with (
        patch.dict("os.environ", {"TEST_LLM_KEY": "sk-test"}),
        patch(_PATCH, return_value=SAMPLE_DIFF_VICREG),
    ):
        policy.propose({"history": history, "source": SAMPLE_SOURCE})

    # Conversation should have appended (not been cleared).
    # Pre-call had 2 entries; one successful propose adds 2 more = 4.
    assert len(policy._conversation) == 4
    assert policy._conversation[0]["content"] == "msg1"


def test_policy_propose_includes_prior_approaches_in_call():
    """Beyond reset, even within a single conversation the LLM call
    must include the anti-repeat section in the user message so
    Claude is steered away from approaches it has already tried."""
    policy = _make_policy()

    history = [
        {"iter": 1, "status": "ok", "decision": "discard",
         "params": {"_type": "diff", "diff": SAMPLE_DIFF_COSINE}},
        {"iter": 2, "status": "failed", "decision": "discard",
         "params": {"_type": "diff", "diff": SAMPLE_DIFF_VICREG}},
    ]

    captured: dict[str, list[dict]] = {}

    def _capture(api_url, model, key, messages, *args, **kwargs):  # noqa: ARG001
        captured["messages"] = messages
        return SAMPLE_DIFF_COSINE

    with (
        patch.dict("os.environ", {"TEST_LLM_KEY": "sk-test"}),
        patch(_PATCH, side_effect=_capture),
    ):
        policy.propose({"history": history, "source": SAMPLE_SOURCE})

    assert "messages" in captured
    # Find the user message (last role=user).
    user_msgs = [m for m in captured["messages"] if m["role"] == "user"]
    assert user_msgs, "no user message sent"
    last_user = user_msgs[-1]["content"]
    assert "PREVIOUSLY PROPOSED APPROACHES" in last_user
    assert "CosineAnnealingLR" in last_user
    assert "LOSS_REGISTRY" in last_user
