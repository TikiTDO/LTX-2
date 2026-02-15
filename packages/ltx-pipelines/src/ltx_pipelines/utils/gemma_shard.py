from __future__ import annotations

import logging
from collections.abc import Mapping

import torch

logger = logging.getLogger(__name__)


def _move_tensors(obj, device: torch.device):  # noqa: ANN001
    if obj is None:
        return None
    if torch.is_tensor(obj):
        if obj.device == device:
            return obj
        return obj.to(device=device, non_blocking=True)
    if isinstance(obj, tuple):
        return tuple(_move_tensors(x, device) for x in obj)
    if isinstance(obj, list):
        return [_move_tensors(x, device) for x in obj]
    if isinstance(obj, dict):
        return {k: _move_tensors(v, device) for k, v in obj.items()}
    return obj


def shard_gemma3_language_layers_inplace(
    model: torch.nn.Module,
    *,
    layer_device_map: Mapping[int, str],
    primary: torch.device = torch.device("cuda:0"),
    move_vision_tower_to: torch.device | None = None,
) -> None:
    """
    Manual multi-GPU sharding for Gemma3ForConditionalGeneration-like models.

    Key constraint: Gemma ties `embed_tokens.weight` and `lm_head.weight`, so those must
    stay on the same device (we keep them on `primary`). To reduce cross-device transfers,
    prefer a mapping that assigns a contiguous *middle* block of layers to the secondary GPU.

    This function:
    - Moves token embedding + final norm + lm_head to `primary`
    - Moves each decoder layer to its assigned device
    - Registers a per-layer forward pre-hook to move tensor inputs/kwargs to that layer's device
    - Optionally moves the vision tower + multi-modal projector
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for sharding Gemma across GPUs")

    # Paths validated against transformers Gemma3ForConditionalGeneration in this repo.
    try:
        language_model = model.model.language_model
        layers = language_model.layers
        embed = language_model.embed_tokens
        norm = language_model.norm
        lm_head = model.lm_head
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Unexpected Gemma3 model structure; cannot shard") from e

    # Keep tied weights and final projection on primary.
    embed.to(primary)
    norm.to(primary)
    lm_head.to(primary)

    # Optional: push vision tower away from primary VRAM.
    # Default to primary to keep `model.device` stable (the processor uses it for .to()).
    vt_dev = primary if move_vision_tower_to is None else move_vision_tower_to
    try:
        model.model.vision_tower.to(vt_dev)
        model.model.multi_modal_projector.to(vt_dev)
    except Exception:
        logger.warning("Failed moving vision tower to %s (continuing)", vt_dev)

    # Install hooks and move layers.
    counts: dict[str, int] = {}
    for idx, layer in enumerate(layers):
        dev_str = layer_device_map.get(idx, str(primary))
        dev = torch.device(dev_str)
        layer.to(dev)
        counts[dev_str] = counts.get(dev_str, 0) + 1

        # Move tensor inputs for this layer right before execution.
        def _pre_hook(_mod, args, kwargs, *, _dev=dev):  # noqa: ANN001
            return (_move_tensors(args, _dev), _move_tensors(kwargs, _dev))

        layer.register_forward_pre_hook(_pre_hook, with_kwargs=True)

    logger.info("gemma_layer_devices %s", " ".join(f"{k}={v}" for k, v in sorted(counts.items())))
