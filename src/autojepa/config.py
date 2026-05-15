from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ObjectiveConfig(BaseModel):
    """Campaign objective metric.

    AutoJEPA defaults to `probe_auroc` per ADR-004 (training loss
    collapses, so probe-based downstream score is the only safe
    objective). The metric name resolves against the dict passed to
    `emit_progress(..., metrics={...})` from the trial subprocess.
    Same field is used by the intra-iteration forecaster as the
    series to extrapolate (writeup §6.2 forecast_target).
    """

    metric: str = "probe_auroc"
    direction: Literal["min", "max"] = "max"


class BasilicaConfig(BaseModel):
    image: str = "pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel"
    gpu_count: int = 1
    gpu_models: list[str] = Field(default_factory=lambda: ["A100", "H100"])
    memory: str = "32Gi"
    cpu: str = "8"
    storage: str | None = "/data"
    ttl_seconds: int = 7200
    min_gpu_memory_gb: int | None = None
    setup_cmd: str | None = None
    # Max seconds to wait for the container to reach ready state. Default
    # 600 covers a serial run with light setup_cmd. Heavy setup_cmd
    # (large pip install + model preload) under parallel mode (multiple
    # concurrent deployments contending for network) requires more —
    # observed 600s timeouts at K=4 with the security-judge example.
    ready_timeout_s: int = 600
    # Seconds the bootstrap script sleeps AFTER the trial exits before
    # killing itself. The controller needs a window to (a) notice the
    # final metrics in the polled logs, (b) call /model/files +
    # /model/download/<path> on the bootstrap's HTTP server. With the
    # earlier 15s value, a fast-finishing trial whose stdout flushed
    # just after a poll-cycle would have its container shut down before
    # the controller's next poll → HTTP 500/503 on download. 90s gives
    # comfortable headroom even with adaptive 20s poll backoff.
    post_trial_sleep_s: int = 90


class TargetConfig(BaseModel):
    type: Literal["command", "http", "basilica"] = "command"
    prepare_cmd: list[str] | None = None
    train_cmd: list[str] | None = None
    eval_cmd: list[str] | None = None
    url: str | None = None
    headers: dict[str, str] | None = None
    timeout_s: int = 3600
    workdir: str = "."
    basilica: BasilicaConfig = Field(default_factory=BasilicaConfig)


class PolicyConfig(BaseModel):
    type: Literal[
        "grid", "random", "static", "learned", "llm", "llm_diff", "hybrid"
    ] = "static"
    params: dict[str, list[float] | list[int] | list[str] | list[bool]] = Field(
        default_factory=dict
    )
    seed: int = 7
    llm_api_url: str | None = None
    # Per ADR-017, llm_model accepts a list (try-each-on-404) or a
    # single string (back-compat). YAML can express either:
    #   llm_model: "deepseek-ai/DeepSeek-V3-0324-TEE"
    #   llm_model: ["deepseek-ai/DeepSeek-V3-0324-TEE", "deepseek-ai/DeepSeek-V3-0324"]
    # A comma-separated string is also accepted and split on first use.
    llm_model: str | list[str] | None = None
    llm_api_key_env: str = "OPENAI_API_KEY"
    llm_timeout_s: int = 30
    # Diff mode fields
    mutable_file: str | None = None
    frozen_file: str | None = None
    program_file: str | None = None
    contract_strict: bool = True
    required_calls: list[str] = Field(default_factory=lambda: ["emit_progress"])
    # Hybrid mode fields
    hybrid_param_explore_iters: int = 5
    hybrid_stall_threshold: int = 3
    hybrid_diff_failure_limit: int = 3

    @model_validator(mode="after")
    def _validate_llm_fields(self) -> "PolicyConfig":
        if self.type in ("llm", "llm_diff", "hybrid"):
            if not self.llm_api_url:
                raise ValueError(
                    f"llm_api_url is required when policy type is '{self.type}'"
                )
            if not self.llm_model:
                raise ValueError(
                    f"llm_model is required when policy type is '{self.type}'"
                )
            # Empty list is falsy already; defensive: a list of empty
            # strings is still meaningless and should fail.
            if isinstance(self.llm_model, list) and not any(
                isinstance(m, str) and m.strip() for m in self.llm_model
            ):
                raise ValueError(
                    f"llm_model list must contain at least one non-empty model name "
                    f"when policy type is '{self.type}'"
                )
        if self.type in ("llm_diff", "hybrid"):
            if not self.mutable_file:
                raise ValueError(
                    f"mutable_file is required when policy type is '{self.type}'"
                )
        return self


class ComparabilityConfig(BaseModel):
    budget_mode: Literal["fixed_wallclock", "parallel_wallclock"] = "fixed_wallclock"
    expected_budget_s: int = 300
    expected_hardware_fingerprint: str | None = None
    strict: bool = True


class IntraIterationCancelConfig(BaseModel):
    """Intra-iteration cancellation guard.

    Defaults are recalibrated for SSL learning-curve shapes per ADR-008
    (writeup §6.2): JEPA loss curves have a long plateau where only the
    probe score moves, so the upstream autoresearch-rl defaults
    (min_steps=5, poll_interval_s=5.0, min_reports_before_decide=5)
    over-cancel JEPA trials in the plateau phase.

    The metric extrapolated by the forecaster is `objective.metric`
    (default `probe_auroc`); no separate `forecast_target` field is
    needed because the autoresearch-rl plumbing already wires
    `objective.metric` through to the IntraIterationGuard constructor.
    """

    enabled: bool = False
    min_steps: int = 2000
    poll_interval_s: float = 30.0
    min_reports_before_decide: int = 10


class ParallelConfig(BaseModel):
    enabled: bool = False
    max_concurrency: int = 1
    resources: dict[str, int] = Field(default_factory=lambda: {"gpu": 1})
    submit_poll_interval_s: float = 0.5


class ControllerConfig(BaseModel):
    seed: int | None = None
    max_wall_time_s: int | None = None
    max_iterations: int | None = None
    no_improve_limit: int | None = None
    failure_rate_limit: float | None = None
    failure_window: int = 10
    checkpoint_path: str | None = None
    intra_iteration_cancel: IntraIterationCancelConfig = Field(
        default_factory=IntraIterationCancelConfig
    )
    parallel: ParallelConfig = Field(default_factory=ParallelConfig)


class ScoringConfig(BaseModel):
    val_bpb: float = 1.0
    loss: float = 0.15
    fail_penalty: float = 0.8
    timeout_penalty: float = 1.2
    neutral_penalty: float = 0.05
    directional_bonus: float = 0.2
    early_stop_penalty: float = 0.4
    eval_score_weight: float = 0.25


class TelemetryConfig(BaseModel):
    trace_path: str = "traces/events.jsonl"
    ledger_path: str = "artifacts/results.tsv"
    artifacts_dir: str = "artifacts/runs"
    versions_dir: str = "artifacts/versions"
    model_output_dir: str | None = None
    timeline_path: str | None = None
    max_file_size_bytes: int = 50 * 1024 * 1024  # 50MB
    max_rotated_files: int = 5


class RunConfig(BaseModel):
    name: str = "autoresearch-run"
    program_path: str | None = None
    objective: ObjectiveConfig = Field(default_factory=ObjectiveConfig)
    target: TargetConfig = Field(default_factory=TargetConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    controller: ControllerConfig = Field(default_factory=ControllerConfig)
    comparability: ComparabilityConfig = Field(default_factory=ComparabilityConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
