from __future__ import annotations

from collections.abc import Mapping


def gemma_middle_2gpu_layer_device_map(
    *,
    num_layers: int = 48,
    primary: str = "cuda:0",
    secondary: str = "cuda:1",
    secondary_start: int = 8,
    secondary_end: int = 40,
) -> dict[int, str]:
    """
    Map a contiguous *middle* range of Gemma decoder layers to the secondary GPU.

    This minimizes inter-device transfers: only two boundaries (primary->secondary->primary),
    while allowing the tied embedding/lm_head (typically on primary) to remain consistent.
    """
    if secondary_start < 0 or secondary_end < 0 or secondary_start > secondary_end:
        raise ValueError(f"Invalid range: {secondary_start}:{secondary_end}")
    if secondary_end > num_layers:
        raise ValueError(f"secondary_end must be <= num_layers ({num_layers}), got {secondary_end}")
    out: dict[int, str] = {}
    for i in range(num_layers):
        out[i] = secondary if (secondary_start <= i < secondary_end) else primary
    return out


def gemma_interleave_2gpu_layer_device_map(
    *,
    num_layers: int = 48,
    primary: str = "cuda:0",
    secondary: str = "cuda:1",
) -> dict[int, str]:
    """
    Interleave layers across GPUs (0->primary, 1->secondary, ...).

    This maximizes cross-device hops; prefer middle-2gpu for performance unless
    you're only trying to fit the model into VRAM.
    """
    return {i: (primary if (i % 2 == 0) else secondary) for i in range(num_layers)}


def resolve_gemma_layer_device_map_preset(preset: str) -> Mapping[int, str] | None:
    """
    Presets:
    - "" -> None
    - "middle-2gpu" -> middle-2gpu@8:40
    - "middle-2gpu@S:E" -> user-defined range on cuda:1 (half-open)
    - "interleave-2gpu" -> alternating layers
    """
    preset = (preset or "").strip()
    if preset == "":
        return None
    if preset == "middle-2gpu":
        return gemma_middle_2gpu_layer_device_map()
    if preset.startswith("middle-2gpu@"):
        _, rng = preset.split("@", 1)
        s_s, e_s = rng.split(":", 1)
        return gemma_middle_2gpu_layer_device_map(secondary_start=int(s_s.strip()), secondary_end=int(e_s.strip()))
    if preset == "interleave-2gpu":
        return gemma_interleave_2gpu_layer_device_map()
    raise ValueError(f"Unknown gemma layer device map preset: {preset!r}")

