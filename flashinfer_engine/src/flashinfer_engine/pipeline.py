"""End-to-end validation pipeline.

Wraps the kernel build + smoke test sequence into a single Python entry point.
Used by CI and as a sanity check after compiling new kernel versions.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


_REPO_ROOT = Path(__file__).parent.parent.parent.parent
_ENGINE_DIR = _REPO_ROOT / "flashinfer_engine"


@dataclass
class StepResult:
    command: list[str]
    returncode: int
    cwd: str | None = None


def _run(command: list[str], cwd: str | Path | None = None) -> StepResult:
    completed = subprocess.run(command, check=False, cwd=cwd)
    return StepResult(command=command, returncode=completed.returncode, cwd=str(cwd) if cwd else None)


def run_validation_suite() -> list[StepResult]:
    """Build and smoke-test all available kernel versions.

    Currently runs the build + smoke-test path for the kernels in `csrc/`.
    Future versions will be added as separate steps here.
    """
    steps = [
        ([sys.executable, "-m", "csrc.compile"], _ENGINE_DIR),
        ([sys.executable, "-m", "pytest", "tests/test_smoke.py", "-v"], _ENGINE_DIR),
    ]
    return [_run(cmd, cwd=cwd) for cmd, cwd in steps]
