#!/usr/bin/env python3
"""Launch the AutoJEPA Phase-3 trace-jepa campaign on Basilica.

Mirrors examples/ijepa-cifar10/deploy.py: read `train.py` and
`prepare.py` locally, base64-inject them into the Basilica container
via `setup_cmd`, then invoke `autojepa run` with
`target.basilica.setup_cmd` overridden to the augmented command.

Why deploy.py rather than `autojepa run config.yaml` directly:
The Basilica container only has the base image plus what `setup_cmd`
installs. Pip-installing autojepa from GitHub provides `autojepa.*`
imports but NOT `examples/*` (only `src/` ships in the wheel). The
example's `train.py` + `prepare.py` therefore must be injected
separately. Once injected, `train_cmd` and `prepare_cmd` reference
`/app/train.py` and `/app/prepare.py`.

Usage:
    cd examples/trace-jepa
    BASILICA_API_TOKEN=... CHUTES_API_KEY=... python3 deploy.py
    # or with overrides:
    python3 deploy.py --policy hybrid -- controller.max_iterations=5
"""

from __future__ import annotations

import argparse
import base64
import os
import subprocess
import sys
from pathlib import Path

DIR = Path(__file__).resolve().parent
CONFIG = DIR / "config.yaml"
REPO_ROOT = DIR.parent.parent

INJECT_FILES = {
    "/app/train.py": DIR / "train.py",
    "/app/prepare.py": DIR / "prepare.py",
}

# Pin a specific commit so re-runs against the same SHA are reproducible.
# Pass --git-ref to override (useful when iterating before push).
DEFAULT_GIT_REF = "HEAD"


def _build_file_injection_cmd() -> str:
    parts: list[str] = ["mkdir -p /app"]
    for dest, src in INJECT_FILES.items():
        content = src.read_text(encoding="utf-8")
        encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
        parts.append(
            f'python3 -c "import base64; '
            f"open('{dest}','w').write(base64.b64decode('{encoded}').decode('utf-8'))\""
        )
    return " && ".join(parts)


def _build_setup_cmd(git_ref: str) -> str:
    """Compose the container setup: pip install + file injection.

    Same pattern as the Phase-2 deploy.py. The Basilica base image
    (`pytorch/pytorch:...-cudnn9-devel`) does NOT ship `git`. pip
    cannot install `autojepa @ git+https://...` without it, so the
    first step apt-installs git.

    Trace-jepa adds `webdataset` to the heavy pip batch (Phase-2
    omits it). The transformers pin (4.47.x) is the same as Phase-2
    to dodge the torch 2.4 infer_schema crash on
    `from __future__ import annotations` integration code paths.

    Setup_cmd order:
    1. apt-install git (~30 s).
    2. Heavy GPU/ML pip batch (includes webdataset for trace-jepa).
    3. autojepa core deps pip batch.
    4. autojepa from git WITHOUT extras.
    5. Inject train.py + prepare.py via base64.
    6. Sanity-import + nvidia-smi check.

    Once the baked image (ghcr.io/epappas/autojepa-runtime:phase2 or
    a phase3 successor) lands and config.yaml is switched to it, this
    setup_cmd shrinks to just the autojepa-from-git + file_inject
    steps.
    """
    apt = "apt-get update -qq && apt-get install -y -qq git"
    deps = (
        "pip install --no-cache-dir "
        "torch>=2.4 lightning>=2.4 torchmetrics>=1.4 torchvision "
        "'transformers>=4.47,<4.48' datasets "
        "'stable-pretraining>=0.1.6,<0.2' timm webdataset"
    )
    autojepa_deps = (
        "pip install --no-cache-dir 'numpy>=1.24' 'pydantic>=2.7' "
        "'pyyaml>=6.0' 'typer>=0.12' 'basilica-sdk>=0.20'"
    )
    autojepa = (
        f"pip install --no-cache-dir --no-deps "
        f"'autojepa @ git+https://github.com/epappas/autojepa.git@{git_ref}'"
    )
    file_inject = _build_file_injection_cmd()
    sanity = (
        'python3 -c "import autojepa, stable_pretraining, torch; '
        "print('autojepa OK; spt', stable_pretraining.__version__, "
        "'cuda', torch.cuda.is_available(), 'devices', torch.cuda.device_count())\""
    )
    return (
        f"{apt} && {deps} && {autojepa_deps} && {autojepa} "
        f"&& {file_inject} && {sanity}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Deploy trace-jepa to Basilica")
    parser.add_argument("--policy", choices=["llm", "hybrid", "grid"], default=None)
    parser.add_argument(
        "--git-ref",
        default=DEFAULT_GIT_REF,
        help="GitHub commit/branch ref to pip install autojepa from (default: HEAD)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Override controller.max_iterations (default: 30 from config.yaml)",
    )
    parser.add_argument("overrides", nargs="*", help="Extra key=value --override pairs")
    args = parser.parse_args()

    for var in ("BASILICA_API_TOKEN", "CHUTES_API_KEY"):
        if not os.environ.get(var):
            print(f"ERROR: {var} not set in environment", file=sys.stderr)
            return 1

    setup_cmd = _build_setup_cmd(args.git_ref)
    print(f"setup_cmd length: {len(setup_cmd)} chars", file=sys.stderr)

    cmd = [
        "uv", "run", "autojepa", "run", str(CONFIG),
        "--override", f"target.basilica.setup_cmd={setup_cmd}",
    ]
    if args.policy:
        cmd += ["--override", f"policy.type={args.policy}"]
    if args.max_iterations is not None:
        cmd += ["--override", f"controller.max_iterations={args.max_iterations}"]
    for ov in args.overrides:
        cmd += ["--override", ov]

    print(f"running from {REPO_ROOT} ...", file=sys.stderr)
    return subprocess.run(cmd, cwd=str(REPO_ROOT), check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
