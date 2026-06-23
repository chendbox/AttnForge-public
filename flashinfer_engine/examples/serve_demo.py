#!/usr/bin/env python3
"""Minimal placeholder for future runtime demos.

This repository currently focuses on benchmark, validation, and kernel
experiments. End-to-end serving demos will be added as the runtime layer
stabilizes.
"""

from __future__ import annotations

import sys


def main() -> None:
    sys.exit(
        "examples/serve_demo.py is intentionally kept minimal in the public repo. "
        "Use the benchmark scripts under benchmarks/ or integrate the backends "
        "through your own runtime wrapper."
    )


if __name__ == "__main__":
    main()
