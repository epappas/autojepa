"""LLM-powered diff policy for code modification proposals.

Sends the current training script source, experiment history, and task
description to an LLM via a persistent multi-turn conversation. The LLM
builds cumulative reasoning across iterations. On validation failure the
policy sends a correction request and retries before falling back to
GreedyLLMPolicy.
"""
from __future__ import annotations

import logging
import os
import re

from autojepa.policy._prompt_fragments import (
    CANCELLATION_CONTEXT_RULES,
    JEPA_HARD_RULES,
    PROGRESS_PROTOCOL_RULES,
    render_progress_series,
    render_progress_summary,
)
from autojepa.policy.interface import DiffProposal
from autojepa.policy.llm_context import (
    extract_recent_errors,
    extract_recent_logs,
    format_history_section,
)
from autojepa.policy.llm_search import _call_chat_api_messages
from autojepa.sandbox.validator import validate_diff

logger = logging.getLogger(__name__)

_MAX_CONVERSATION_PAIRS = 10
_MAX_CORRECTION_RETRIES = 2
_MAX_PRIOR_DIFF_SUMMARIES = 8


def _summarize_diff(diff: str, *, max_len: int = 120) -> str:
    """One-line gist of a diff for the anti-repeat prompt section.

    Picks the first non-context added line (or removed line if none),
    strips the leading +/- and pads short lines so the LLM can recognise
    the distinct approach (CosineAnnealingLR vs VICReg vs partial-credit
    reward etc.) without being overwhelmed by full hunks.
    """
    if not diff:
        return "<empty>"
    for raw in diff.splitlines():
        if raw.startswith("+") and not raw.startswith("+++"):
            cleaned = raw[1:].strip()
            if cleaned and not cleaned.startswith("#"):
                return cleaned[:max_len]
    for raw in diff.splitlines():
        if raw.startswith("-") and not raw.startswith("---"):
            cleaned = raw[1:].strip()
            if cleaned:
                return f"-{cleaned[:max_len - 1]}"
    return "<context-only>"


def _extract_prior_diff_summaries(history: list[dict]) -> list[str]:
    """Pull a deduped list of recent diff approaches from history.

    Reads `params['diff']` populated by `_hybrid_extractor` for diff
    iters. Skips non-diff iters and empty diffs. Returns most-recent
    first, deduped on the gist string to avoid spamming the prompt
    when the LLM has been hammering the same approach.
    """
    seen: set[str] = set()
    summaries: list[str] = []
    for entry in reversed(history):
        params = entry.get("params") or {}
        if params.get("_type") != "diff":
            continue
        diff = params.get("diff")
        if not isinstance(diff, str) or not diff.strip():
            continue
        gist = _summarize_diff(diff)
        if gist in seen:
            continue
        seen.add(gist)
        status = entry.get("status", "?")
        decision = entry.get("decision", "?")
        iter_idx = entry.get("iter", "?")
        summaries.append(
            f"iter={iter_idx} status={status} decision={decision} :: {gist}"
        )
        if len(summaries) >= _MAX_PRIOR_DIFF_SUMMARIES:
            break
    return summaries


def _last_kept_diff_iter(history: list[dict]) -> int | None:
    """Return the iter index of the most recent kept diff, or None."""
    for entry in reversed(history):
        params = entry.get("params") or {}
        if params.get("_type") != "diff":
            continue
        if entry.get("decision") == "keep":
            iter_idx = entry.get("iter")
            if isinstance(iter_idx, int):
                return iter_idx
    return None


_SYSTEM_PROMPT = (
    "You are a code optimization assistant for AutoJEPA, a self-supervised "
    "Joint-Embedding Predictive Architecture pretraining framework. "
    "Given a training script, experiment history, and task description, "
    "propose a code modification as a unified diff. "
    "Respond with ONLY a valid unified diff (starting with --- a/ and +++ b/). "
    "Make targeted, minimal changes to improve probe_auroc.\n\n"
    "DIFF QUALITY REQUIREMENTS (enforced):\n"
    "  - The diff MUST change runtime behavior. NO-OP diffs are rejected.\n"
    "  - Forbidden no-op patterns: appending unused module-level assignments "
    "(e.g. `use_qk_norm = True` with no consumer), comments-only changes, "
    "whitespace-only changes, dead code after `sys.exit(...)` or `return`.\n"
    "  - Your diff MUST modify a function body, a class method, a model "
    "construction call, the loss formulation, the optimizer, the masking "
    "strategy, OR the EMA schedule. If you cannot identify a specific runtime "
    "behavior change, do NOT respond.\n"
    "  - Prefer ONE substantive change over many superficial ones.\n\n"
    "EXAMPLE OF A GOOD DIFF (swap plain L2 latent loss for VICReg anti-collapse):\n"
    "--- a/train.py\n"
    "+++ b/train.py\n"
    "@@ -120,7 +120,12 @@ def main():\n"
    "     model = IJEPA(\n"
    "         encoder_name=\"vit_tiny_patch16_224\",\n"
    "         predictor_embed_dim=PREDICTOR_EMBED_DIM,\n"
    "         predictor_depth=PREDICTOR_DEPTH,\n"
    "+        # Swap plain L2 latent loss (collapse-prone) for VICReg\n"
    "+        # variance/invariance/covariance loss per C-JEPA writeup §3.\n"
    "+        from autojepa.models.losses import LOSS_REGISTRY\n"
    "+        loss_fn=LOSS_REGISTRY[\"vicreg\"](var_coef=25.0, cov_coef=1.0),\n"
    "         num_targets=NUM_TARGETS,\n\n"
    f"{PROGRESS_PROTOCOL_RULES}\n\n"
    f"{CANCELLATION_CONTEXT_RULES}\n\n"
    f"{JEPA_HARD_RULES}"
)


def _format_diff_prompt(
    source: str,
    filename: str,
    history: list[dict],
    metric: str,
    direction: str,
    program: str = "",
    prior_approaches: list[str] | None = None,
) -> str:
    lines: list[str] = []
    if program:
        lines.append("Task specification:")
        lines.append(program)
        lines.append("")

    lines.append(f"Objective: {direction}imize '{metric}'")
    lines.append("")

    lines.append(f"Current source ({filename}):")
    lines.append("```python")
    lines.append(source)
    lines.append("```")
    lines.append("")

    history_section = format_history_section(history, metric)
    lines.append(history_section)
    lines.append("")

    summary = render_progress_summary(history)
    if summary:
        lines.append(summary)
        lines.append("")
    series = render_progress_series(history, metric)
    if series:
        lines.append(series)
        lines.append("")

    recent_errors = extract_recent_errors(history)
    if recent_errors:
        lines.append("Recent errors:")
        for err in recent_errors:
            lines.append(f"  - {err}")
        lines.append("")

    recent_logs = extract_recent_logs(history)
    if recent_logs:
        lines.append("Recent training logs:")
        for log_entry in recent_logs:
            lines.append(f"  {log_entry}")
        lines.append("")

    if prior_approaches:
        # Anti-repeat guard. v30 iters 9, 10, 11 all proposed the same
        # CosineAnnealingLR approach after iter=4 was kept because the
        # conversation anchored on the success and refused to explore.
        # See docs/phase-2-fix-diary.md 2026-05-19 / ADR-025.
        lines.append(
            "PREVIOUSLY PROPOSED APPROACHES (DO NOT propose any of these again):"
        )
        for approach in prior_approaches:
            lines.append(f"  - {approach}")
        lines.append(
            "If your next idea matches any of the above, you MUST pick a "
            "different one (e.g. a different loss formulation, different "
            "masking strategy, different EMA schedule, different optimizer)."
        )
        lines.append("")

    lines.append(
        f"Respond with ONLY a unified diff. "
        f"Use '--- a/{filename}' and '+++ b/{filename}' as file paths. "
        f"Make targeted, minimal changes to improve {metric}."
    )
    return "\n".join(lines)


def _parse_diff_response(raw: str, filename: str) -> str:
    """Extract unified diff from LLM response."""
    text = raw.strip()

    # Strip markdown fences
    match = re.search(r"```(?:diff)?\s*\n?(.*)", text, re.DOTALL)
    if match:
        inner = match.group(1)
        inner = re.sub(r"\s*```\s*$", "", inner)
        text = inner.strip()

    # Find diff start
    diff_start: int | None = None
    text_lines = text.splitlines()
    for i, line in enumerate(text_lines):
        if line.startswith("---") or line.startswith("diff --git"):
            diff_start = i
            break

    if diff_start is None:
        raise ValueError(f"No unified diff found in response: {raw[:200]}")

    diff_lines = text_lines[diff_start:]
    diff_text = "\n".join(diff_lines) + "\n"

    has_minus = any(ln.startswith("---") for ln in diff_lines)
    has_plus = any(ln.startswith("+++") for ln in diff_lines)
    has_hunk = any(ln.startswith("@@") for ln in diff_lines)

    if not (has_minus and has_plus and has_hunk):
        raise ValueError("Diff missing required sections (---, +++, or @@)")

    return diff_text


class LLMDiffPolicy:
    """Calls an OpenAI-compatible chat API to propose code diffs.

    Maintains a multi-turn conversation across iterations so the LLM
    accumulates context about what changes worked. On validation failure,
    sends a correction request and retries up to _MAX_CORRECTION_RETRIES
    times before falling back to GreedyLLMPolicy.
    """

    def __init__(
        self,
        *,
        mutable_file: str,
        api_url: str,
        model: str | list[str],
        api_key_env: str = "OPENAI_API_KEY",
        timeout_s: int = 60,
        metric: str = "val_bpb",
        direction: str = "min",
        seed: int = 7,
    ):
        self._mutable_file = mutable_file
        self._api_url = api_url
        self._model = model
        self._api_key_env = api_key_env
        self._timeout_s = timeout_s
        self._metric = metric
        self._direction = direction
        self._filename = os.path.basename(mutable_file)
        self._conversation: list[dict] = []
        self._last_seen_kept_diff_iter: int | None = None

    def propose(self, state: dict) -> DiffProposal:
        history: list[dict] = state.get("history", [])
        program: str = state.get("program", "")
        source: str = state.get("source", "")

        api_key = os.environ.get(self._api_key_env)
        if not api_key:
            logger.warning(
                "LLM diff policy: %s not set, falling back to greedy",
                self._api_key_env,
            )
            return self._greedy_fallback()

        if not source:
            logger.warning("LLM diff policy: no source in state, falling back to greedy")
            return self._greedy_fallback()

        # ADR-025 hygiene: after a kept diff lands, the patched source
        # is the new baseline — any conversation history built on top of
        # the OLD baseline is now misleading. Clear conversation once
        # per newly-observed kept-diff iter so the LLM re-evaluates from
        # scratch instead of anchoring on its previous success and
        # proposing trivial variations of the same approach.
        last_kept = _last_kept_diff_iter(history)
        if last_kept is not None and last_kept != self._last_seen_kept_diff_iter:
            self._conversation.clear()
            self._last_seen_kept_diff_iter = last_kept
            logger.info(
                "LLM diff policy: reset conversation after kept diff at iter=%d",
                last_kept,
            )

        prior_approaches = _extract_prior_diff_summaries(history)
        user_msg = _format_diff_prompt(
            source=source,
            filename=self._filename,
            history=history,
            metric=self._metric,
            direction=self._direction,
            program=program,
            prior_approaches=prior_approaches,
        )

        # Build local messages for this attempt (includes conversation + new user msg).
        # The local list may grow with correction messages on retry; _conversation
        # only stores clean successful (user, assistant) pairs.
        messages: list[dict] = list(self._trimmed_conversation())
        messages.append({"role": "user", "content": user_msg})

        raw = ""
        for attempt in range(_MAX_CORRECTION_RETRIES + 1):
            try:
                full_messages = [{"role": "system", "content": _SYSTEM_PROMPT}] + messages
                raw = _call_chat_api_messages(
                    self._api_url, self._model, api_key,
                    full_messages, self._timeout_s, max_tokens=4096,
                )
                diff = _parse_diff_response(raw, self._filename)
                result = validate_diff(diff)
                if not result.ok:
                    raise ValueError(f"diff validation failed: {result.reason}")

                # Success: commit to conversation (clean pair only)
                self._conversation.append({"role": "user", "content": user_msg})
                self._conversation.append({"role": "assistant", "content": raw})
                self._trim_conversation()
                return DiffProposal(diff=diff, rationale="llm-diff")

            except Exception as exc:
                # Promoted from debug -> warning 2026-05-19 after Phase-3
                # v1 surfaced "LLM diff policy failed after 3 attempts,
                # falling back to greedy" with no detail on what the 3
                # failures actually were. Without the exception type +
                # message, post-mortems have to add print statements and
                # rerun. Including type(exc).__name__ makes HTTPError vs
                # ValueError vs JSONDecodeError grep-able immediately.
                logger.warning(
                    "LLM diff attempt %d/%d failed: %s: %s",
                    attempt + 1, _MAX_CORRECTION_RETRIES + 1,
                    type(exc).__name__, exc,
                )
                if attempt < _MAX_CORRECTION_RETRIES:
                    messages.append({"role": "assistant", "content": raw})
                    messages.append({
                        "role": "user",
                        "content": (
                            f"That response was invalid: {exc}. "
                            f"Please provide a correct unified diff for {self._filename}."
                        ),
                    })

        logger.warning(
            "LLM diff policy failed after %d attempts, falling back to greedy",
            _MAX_CORRECTION_RETRIES + 1,
        )
        return self._greedy_fallback()

    def reset_conversation(self) -> None:
        """Clear conversation history (use when starting a new experiment)."""
        self._conversation.clear()

    def _trimmed_conversation(self) -> list[dict]:
        limit = _MAX_CONVERSATION_PAIRS * 2
        return self._conversation[-limit:]

    def _trim_conversation(self) -> None:
        limit = _MAX_CONVERSATION_PAIRS * 2
        if len(self._conversation) > limit:
            self._conversation = self._conversation[-limit:]

    def _greedy_fallback(self) -> DiffProposal:
        """Delegate to GreedyLLMPolicy as a fallback."""
        from autojepa.policy.baselines import GreedyLLMPolicy

        greedy = GreedyLLMPolicy()
        state = {"mutable_file": self._mutable_file, "workdir": "."}
        try:
            return greedy.propose(state)
        except Exception:
            logger.warning("Greedy fallback also failed", exc_info=True)
            return DiffProposal(diff="", rationale="llm-diff-fallback-empty")
