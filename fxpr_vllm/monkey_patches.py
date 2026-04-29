"""Monkey-patches for vLLM internals lacking official extension points."""

from __future__ import annotations

import logging
import sys

logger = logging.getLogger("fxpr_vllm")


def patch_rms_norm() -> int:
    """Install DeterministicRMSNorm and rebind already-imported model modules.

    Idempotent. Modules imported after this call keep the original symbol
    (re-call to rebind them); the op-registry swap still intercepts most paths.
    """
    from vllm.model_executor.custom_op import op_registry, op_registry_oot
    import vllm.model_executor.layers.layernorm as layernorm_mod

    from .rms_norm import DeterministicRMSNorm

    if op_registry.get("rms_norm") is not DeterministicRMSNorm:
        DeterministicRMSNorm.name = "rms_norm"
        op_registry["rms_norm"] = DeterministicRMSNorm
    if op_registry_oot.get("RMSNorm") is not DeterministicRMSNorm:
        op_registry_oot["RMSNorm"] = DeterministicRMSNorm

    original_rms_norm = layernorm_mod.RMSNorm
    if original_rms_norm is not DeterministicRMSNorm:
        layernorm_mod.RMSNorm = DeterministicRMSNorm

    patched = 0
    for mod_name, mod in list(sys.modules.items()):
        if mod is None or not mod_name.startswith("vllm.model_executor.models."):
            continue
        bound = getattr(mod, "RMSNorm", None)
        if bound is not None and bound is not DeterministicRMSNorm:
            setattr(mod, "RMSNorm", DeterministicRMSNorm)
            patched += 1

    if patched == 0:
        logger.warning(
            "DeterministicRMSNorm: no vllm.model_executor.models.* modules patched "
            "(no models loaded yet, or vLLM import path changed)."
        )
    return patched


def patch_attention_backend() -> None:
    """Bind the deterministic attention backend to vLLM's CUSTOM enum slot."""
    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum,
        register_backend,
    )

    from .attention_backend import DeterministicAttentionBackend

    backend_path = (
        f"{DeterministicAttentionBackend.__module__}."
        f"{DeterministicAttentionBackend.__qualname__}"
    )
    register_backend(AttentionBackendEnum.CUSTOM, class_path=backend_path)


def patch_sampler() -> None:
    """Replace Sampler.compute_logprobs with the deterministic log-softmax."""
    from vllm.v1.sample.sampler import Sampler

    from .sampling import deterministic_log_softmax

    if getattr(Sampler, "_fxp_logprobs_patched", False):
        return

    Sampler.compute_logprobs = staticmethod(deterministic_log_softmax)
    Sampler._fxp_logprobs_patched = True
    logger.info("Sampler log-softmax patched")
