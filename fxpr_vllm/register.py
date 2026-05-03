import logging
from typing import Callable

from .config import get_runtime_config

from . import quantisation_config  # noqa: F401
from . import monkey_patches

logger = logging.getLogger("fxpr_vllm")


_registered = False


def register() -> None:
    global _registered
    if _registered:
        return
    _registered = True

    logger.info("fxpr_vllm: registering components")

    cfg = get_runtime_config()
    logger.info(
        "Runtime Config: fxp_int_bits=%d num_kv_splits=%d",
        cfg.fxp_int_bits,
        cfg.num_kv_splits,
    )

    from . import library_ops  # noqa: F401

    steps: list[tuple[str, Callable[[], object], Callable[[], None]]] = [
        ("RMSNorm", monkey_patches.patch_rms_norm, _undo_rms_norm),
        ("Attention", monkey_patches.patch_attention_backend, _noop),
        ("Sampler", monkey_patches.patch_sampler, _undo_sampler),
    ]

    rollback: list[Callable[[], None]] = []
    name = "<unknown>"
    try:
        for name, do, undo in steps:
            do()
            rollback.append(undo)
            logger.info("%s registered", name)
    except Exception as e:
        logger.error("Error during %s registration: %s; rolling back", name, e)
        for undo in reversed(rollback):
            try:
                undo()
            except Exception as undo_err:
                logger.error("Rollback step failed: %s", undo_err)
        _registered = False
        raise


def _noop() -> None:
    return None


def _undo_rms_norm() -> None:
    try:
        from vllm.model_executor.custom_op import op_registry

        op_registry.pop("rms_norm", None)
    except Exception as e:
        logger.warning("RMSNorm rollback failed: %s", e)


def _undo_sampler() -> None:
    try:
        from vllm.v1.sample.sampler import Sampler

        if getattr(Sampler, "_fxp_logprobs_patched", False):
            del Sampler.compute_logprobs
            Sampler._fxp_logprobs_patched = False
    except Exception as e:
        logger.warning("Sampler rollback failed: %s", e)
