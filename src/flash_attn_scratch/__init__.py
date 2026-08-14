"""FlashAttention-2, implemented from scratch as a Triton kernel.

Deliberately does NOT import the Triton kernels at package import time.
Triton is Linux/GPU-only, so an eager import would make this package
unimportable on macOS and break local (CPU) development of the baseline.
Import GPU code explicitly from its own module instead.
"""

__version__ = "0.1.0"
