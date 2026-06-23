"""Shared pytest fixtures and configuration."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# Ensure the package is importable without `pip install -e .`
_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ---------- pytest collection hooks ----------

def pytest_collection_modifyitems(config, items):
    """Auto-skip GPU-marked tests when CUDA is unavailable."""
    if torch.cuda.is_available():
        return
    skip_gpu = pytest.mark.skip(reason="CUDA not available")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)


# ---------- fixtures ----------

@pytest.fixture(scope="session")
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(autouse=True)
def _set_seed():
    """Deterministic randomness for every test."""
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)


# ---------- tolerance constants ----------

# fp32 attention should agree very closely across implementations.
# Larger atol than rtol because softmax output values are in [0, 1] so absolute
# tolerance dominates the loose tail of the distribution.
TOL_FP32 = dict(rtol=1e-3, atol=1e-4)

# bf16/fp16 tolerances (for future use)
TOL_BF16 = dict(rtol=1e-2, atol=1e-2)
TOL_FP16 = dict(rtol=1e-3, atol=1e-3)
