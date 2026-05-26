import logging

from .config import get_runtime_config

logger = logging.getLogger("fxpr_vllm")

_registered = False


def register() -> None:
    """vLLM plugin entry point; auto-called via vllm.general_plugins."""
    global _registered
    if _registered:
        return

    cfg = get_runtime_config()
    logger.info(
        "fxpr_vllm: int_bits=%d frac_bits=%d rms_norm=%s lm_head=%s",
        cfg.fxp_int_bits, cfg.fxp_frac_bits,
        cfg.enable_rms_norm, cfg.enable_lm_head,
    )

    from vllm.model_executor.layers.quantization import register_quantization_config
    from . import monkey_patches
    from .quantisation_config import FixedPointConfig

    register_quantization_config("fixed_point_det")(FixedPointConfig)
    logger.info("registered fixed_point_det quant config")

    monkey_patches.patch_attention_backend()
    logger.info("registered CUSTOM attention backend")

    if cfg.enable_rms_norm:
        monkey_patches.patch_rms_norm()
        logger.info("patched RMSNorm")

    if cfg.enable_lm_head:
        logger.info("lm_head matmul will route through gemm_fxp")

    _registered = True
