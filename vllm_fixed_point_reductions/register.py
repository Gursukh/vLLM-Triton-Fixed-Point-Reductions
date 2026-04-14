import logging
import sys

from .vllm_modules.config import get_runtime_config

logger = logging.getLogger("vllm_fixed_point_reductions")


_registered = False


def _register_rms_norm() -> None:
    """Register the deterministic RMSNorm implementation, replacing the default one."""

    from vllm.model_executor.custom_op import op_registry
    import vllm.model_executor.layers.layernorm as layernorm_mod

    if "rms_norm" in op_registry:
        del op_registry["rms_norm"]

    from .vllm_modules.rms_norm import DeterministicRMSNorm

    original_rms_norm = layernorm_mod.RMSNorm
    layernorm_mod.RMSNorm = DeterministicRMSNorm

    patched_modules = 0
    for mod_name, mod in list(sys.modules.items()):
        if mod is None or not mod_name.startswith("vllm.model_executor.models."):
            continue

        if getattr(mod, "RMSNorm", None) is original_rms_norm:
            setattr(mod, "RMSNorm", DeterministicRMSNorm)
            patched_modules += 1

    logger.info("RMSNorm patched (preloaded_model_modules=%d)", patched_modules)


def _register_quant_config() -> None:
    """Register the quantisation config for the fixed-point reductions in gemms."""

    from .vllm_modules.quantisation_config import FixedPointConfig

    logger.info("Quant config registered: %s", FixedPointConfig.get_name())


def _register_attention_backend() -> None:
    """Register the deterministic attention backend, replacing the "CUSTOM" backend enum."""

    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum,
        register_backend,
    )
    from .vllm_modules.attention_backend import DeterministicAttentionBackend

    backend_path = (
        f"{DeterministicAttentionBackend.__module__}."
        f"{DeterministicAttentionBackend.__qualname__}"
    )

    register_backend(
        AttentionBackendEnum.CUSTOM,
        class_path=backend_path,
    )

    logger.info(
        "Attention backend registered under CUSTOM (%s). "
        "Activate with VLLM_ATTENTION_BACKEND=CUSTOM.",
        backend_path,
    )


def _register_sampler() -> None:
    """Patch the Sampler class to use the deterministic log-softmax implementation."""

    from vllm.v1.sample.sampler import Sampler
    from .vllm_modules.sampling import deterministic_log_softmax

    if getattr(Sampler, "_fxp_logprobs_patched", False):
        return

    Sampler.compute_logprobs = staticmethod(deterministic_log_softmax)
    Sampler._fxp_logprobs_patched = True
    logger.info("Sampler log-softmax patched")


def register() -> None:

    global _registered
    if _registered:
        return
    _registered = True

    logger.info("vllm-deterministic: registering components")

    cfg = get_runtime_config()
    logger.info(
        "  frac_bits=%d num_kv_splits=%d fxp_int_bits=%d",
        cfg.frac_bits,
        cfg.num_kv_splits,
        cfg.fxp_int_bits,
    )

    _register_rms_norm()
    _register_quant_config()
    _register_attention_backend()
    _register_sampler()
