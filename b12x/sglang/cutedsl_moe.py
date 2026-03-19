"""Optional masked-MoE adapter for sglang-backed deployments.

This module keeps the import lightweight by delegating to sglang lazily when
the masked path is requested.
"""


def b12x_moe_masked(*args, **kwargs):
    """Dispatch masked MoE through sglang's CuteDSL implementation."""
    try:
        from sglang.srt.layers.moe.flashinfer_cutedsl_moe import (
            flashinfer_cutedsl_moe_masked,
        )
    except ImportError as exc:
        raise ImportError(
            "b12x.sglang.b12x_moe_masked requires the optional 'sglang' dependency."
        ) from exc

    return flashinfer_cutedsl_moe_masked(*args, **kwargs)


__all__ = ["b12x_moe_masked"]
