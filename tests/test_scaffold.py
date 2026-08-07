"""Scaffold smoke tests: prove the package layout and GPU-skip plumbing work.

Placeholder for the real correctness suite, which arrives with the naive
baseline (step 2) and the Triton kernel (steps 3+).
"""

import pytest
import torch


def test_package_imports_without_triton():
    """The package must import on a CPU-only machine (no Triton available)."""
    import flash_attn

    assert flash_attn.__version__


@pytest.mark.gpu
def test_gpu_marker_is_wired():
    """Sanity-check the `gpu` marker: skipped on CPU, runs on the pod."""
    assert torch.cuda.is_available()
