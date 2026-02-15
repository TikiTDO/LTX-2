from __future__ import annotations

from collections.abc import Mapping


def ltx_split_2gpu_device_map(
    *,
    num_layers: int = 48,
    primary: str = "cuda:0",
    secondary: str = "cuda:1",
    split_at: int | None = None,
) -> dict[str, str]:
    """
    Minimal device_map for accelerate.dispatch_model(model, device_map=...).

    We dispatch the heavy transformer blocks across 2 GPUs and keep everything else on `primary`.
    This assumes the dispatched module is the LTX velocity model (ltx_core.model.transformer.model.LTXModel),
    where the blocks live at `transformer_blocks.{i}`.
    """
    device_map: dict[str, str] = {"": primary}
    split_at = (num_layers // 2) if split_at is None else int(split_at)
    if split_at < 0 or split_at > num_layers:
        raise ValueError(f"split_at must be in [0, {num_layers}], got {split_at}")
    for idx in range(num_layers):
        device_map[f"transformer_blocks.{idx}"] = primary if idx < split_at else secondary
    return device_map


def ltx_interleave_2gpu_device_map(
    *,
    num_layers: int = 48,
    primary: str = "cuda:0",
    secondary: str = "cuda:1",
) -> dict[str, str]:
    """
    Interleave transformer blocks across 2 GPUs (even -> primary, odd -> secondary).

    This reduces the chance of OOM during accelerate dispatch if it moves blocks in-order and a contiguous
    half of the network doesn't fit on a single GPU.
    """
    device_map: dict[str, str] = {"": primary}
    for idx in range(num_layers):
        device_map[f"transformer_blocks.{idx}"] = primary if (idx % 2 == 0) else secondary
    return device_map


def ltx_split_2gpu_cpu_device_map(
    *,
    num_layers: int = 48,
    primary: str = "cuda:0",
    secondary: str = "cuda:1",
    cpu: str = "cpu",
) -> dict[str, str]:
    """
    Split blocks across 2 GPUs, with the final third offloaded to CPU RAM.
    This is a fallback if 2x24GB VRAM is still tight after dtype/attention optimizations.
    """
    device_map: dict[str, str] = {"": primary}
    split_1 = num_layers // 3
    split_2 = (2 * num_layers) // 3
    for idx in range(num_layers):
        if idx < split_1:
            dev = primary
        elif idx < split_2:
            dev = secondary
        else:
            dev = cpu
        device_map[f"transformer_blocks.{idx}"] = dev
    return device_map


def resolve_transformer_device_map_preset(preset: str) -> Mapping[str, str] | None:
    preset = (preset or "").strip()
    if preset == "":
        return None
    # Parameterized preset: split-2gpu@N where N is the split index.
    # Useful to account for root modules living on cuda:0 (biasing more blocks to cuda:1).
    if preset.startswith("split-2gpu@"):
        _, split_s = preset.split("@", 1)
        split_at = int(split_s.strip())
        return ltx_split_2gpu_device_map(split_at=split_at)
    if preset == "split-2gpu":
        return ltx_split_2gpu_device_map()
    if preset == "interleave-2gpu":
        return ltx_interleave_2gpu_device_map()
    if preset == "split-2gpu+cpu":
        return ltx_split_2gpu_cpu_device_map()
    raise ValueError(f"Unknown transformer device map preset: {preset!r}")
