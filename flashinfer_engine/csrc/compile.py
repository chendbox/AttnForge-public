"""Build all v0 CUDA kernels in-place.

Run from the flashinfer_engine/ root:

    python -m csrc.compile

Or compile individual kernels:

    python -m csrc.compile --prefill-only
    python -m csrc.compile --decode-only
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_CSRC = Path(__file__).parent


def _build(kernel_dir: Path) -> None:
    print(f"\n[compile] Building {kernel_dir.name} ...")
    result = subprocess.run(
        [sys.executable, "setup.py", "build_ext", "--inplace"],
        cwd=kernel_dir,
    )
    if result.returncode != 0:
        sys.exit(f"[compile] FAILED: {kernel_dir.name}")
    print(f"[compile] OK: {kernel_dir.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefill-only", action="store_true")
    parser.add_argument("--decode-only",  action="store_true")
    parser.add_argument("--v0-only",      action="store_true")
    parser.add_argument("--v1-only",      action="store_true")
    args = parser.parse_args()

    build_prefill = not args.decode_only
    build_decode  = not args.prefill_only
    build_v0      = not args.v1_only
    build_v1      = not args.v0_only

    if build_prefill and build_v0:
        _build(_CSRC / "flash_v0_prefill")
    if build_decode and build_v0:
        _build(_CSRC / "flash_v0_decode")
    if build_prefill and build_v1:
        _build(_CSRC / "flash_v1_prefill")


if __name__ == "__main__":
    main()
