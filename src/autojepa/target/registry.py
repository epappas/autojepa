from __future__ import annotations

from autojepa.config import TargetConfig
from autojepa.target.command import CommandTarget
from autojepa.target.http import HttpTarget
from autojepa.target.interface import TargetAdapter


def build_target(cfg: TargetConfig) -> TargetAdapter:
    if cfg.type == "basilica":
        from autojepa.target.basilica import BasilicaTarget

        return BasilicaTarget(cfg)

    if cfg.type == "http":
        if not cfg.url:
            raise ValueError("target.url is required for http target")
        return HttpTarget(url=cfg.url, headers=cfg.headers, timeout_s=cfg.timeout_s)

    if not cfg.train_cmd:
        raise ValueError("target.train_cmd is required for command target")
    return CommandTarget(
        train_cmd=cfg.train_cmd, eval_cmd=cfg.eval_cmd,
        workdir=cfg.workdir, timeout_s=cfg.timeout_s,
        prepare_cmd=cfg.prepare_cmd,
    )
