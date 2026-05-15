#!/usr/bin/env python3
"""Launch the AutoJEPA Phase-2 falsifier campaign on Basilica.

Mirrors the upstream `autoresearch-rl/examples/basilica-grpo/deploy.py`
pattern: read `train.py` and `prepare.py` locally, base64-inject them
into the Basilica container via `setup_cmd`, then invoke `autojepa run`
with `target.basilica.setup_cmd` overridden to the augmented command.

Why deploy.py rather than `autojepa run config.yaml` directly:
The Basilica container only has the base image plus what `setup_cmd`
installs. Pip-installing autojepa from GitHub provides `autojepa.*`
imports but NOT `examples/*` (only `src/` ships in the wheel). The
example's `train.py` + `prepare.py` therefore must be injected
separately. Once injected, `train_cmd` and `prepare_cmd` reference
`/app/train.py` and `/app/prepare.py`.

Usage:
    cd examples/ijepa-cifar10
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

    Trimmed install: the [jepa] aggregator extra pulls webdataset
    (~10 MB) which ijepa-cifar10 does not use (it's for trace-jepa
    shards). transformers + datasets are unfortunately HARD imports
    of stable-pretraining v0.1.x (`import stable_pretraining` itself
    pulls them) so we cannot omit them. Verified live on commit
    d23af7c — that build crashed because spt could not import.

    Setup_cmd order:
    1. Install heavy GPU/ML deps via pip in one batch (best wheel
       resolution + cache locality).
    2. Install autojepa core deps.
    3. Install autojepa from git WITHOUT extras (we already installed
       what we need).
    4. Inject train.py + prepare.py via base64.
    5. Sanity-import to fail fast if anything is missing.
    """
    deps = (
        "pip install --no-cache-dir "
        "torch>=2.4 lightning>=2.4 torchmetrics>=1.4 torchvision "
        "transformers>=4.44 datasets "
        "'stable-pretraining>=0.1.6,<0.2' timm"
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
        'python3 -c "import autojepa, stable_pretraining; '
        "print('autojepa OK; spt', stable_pretraining.__version__)\""
    )
    return f"{deps} && {autojepa_deps} && {autojepa} && {file_inject} && {sanity}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Deploy ijepa-cifar10 to Basilica")
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
        help="Override controller.max_iterations (default: 20 from config.yaml)",
    )
    parser.add_argument("overrides", nargs="*", help="Extra key=value --override pairs")
    args = parser.parse_args()

    for var in ("BASILICA_API_TOKEN", "CHUTES_API_KEY"):
        if not os.environ.get(var):
            print(f"ERROR: {var} not set in environment", file=sys.stderr)
            return 1

    setup_cmd = _build_setup_cmd(args.git_ref)

    # Bound the override-string length: setup_cmd contains base64'd
    # train.py + prepare.py and is ~20 KB. Most argv parsers handle this
    # fine but we surface the size so a debugger can verify.
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
