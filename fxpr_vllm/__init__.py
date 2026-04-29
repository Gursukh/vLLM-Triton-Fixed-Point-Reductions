"""Fixed point reductions for VLLM."""

# Eager import registers fake impls so torch.compile / Dynamo can trace.
from . import library_ops  # noqa: F401
