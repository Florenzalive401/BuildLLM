from __future__ import annotations

import torch


def resolve_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is not available. Use --device cpu or --device auto."
        )
    return torch.device(requested_device)


def resolve_precision(
    requested_precision: str,
    device: torch.device,
) -> tuple[str, torch.dtype | None]:
    if device.type == "cpu":
        if requested_precision in {"auto", "fp32"}:
            return "fp32", None
        raise RuntimeError(
            "CPU training uses FP32 in this lab. Use --precision fp32 or auto."
        )

    if requested_precision == "auto":
        requested_precision = "bf16" if torch.cuda.is_bf16_supported() else "fp16"

    if requested_precision == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                "BF16 was requested but is not supported by this CUDA device. "
                "Use --precision fp16 or fp32."
            )
        return "bf16", torch.bfloat16
    if requested_precision == "fp16":
        return "fp16", torch.float16
    return "fp32", None
