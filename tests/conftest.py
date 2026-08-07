"""Shared test configuration.

Tests marked `gpu` need CUDA + Triton. On a CPU-only machine (e.g. a Mac)
they are skipped rather than failing, so the CPU-testable parts of the repo
stay runnable with no GPU pod running.
"""

import pytest
import torch


def pytest_collection_modifyitems(config, items):
    if torch.cuda.is_available():
        return
    skip_gpu = pytest.mark.skip(reason="no CUDA device available")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
