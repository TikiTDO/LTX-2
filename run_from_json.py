#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import safe_open

from ltx_core.components.guiders import MultiModalGuiderParams
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.quantization import QuantizationPolicy
from ltx_pipelines.ti2vid_one_stage import TI2VidOneStagePipeline
from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
from ltx_pipelines.utils.device_map import resolve_transformer_device_map_preset
from ltx_pipelines.utils.telemetry import Timer, log_cuda_memory, log_nvidia_smi, log_ram_memory, log_system_summary


logger = logging.getLogger("run_from_json")

def _configure_logging(*, level: int, log_file: str | None) -> None:
    fmt = "%(asctime)s %(levelname)s %(name)s %(message)s"
    root = logging.getLogger()
    root.setLevel(level)

    if not root.handlers:
        logging.basicConfig(level=level, format=fmt)
    else:
        for h in root.handlers:
            h.setFormatter(logging.Formatter(fmt))

    if not log_file:
        return
    p = Path(log_file).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    already = any(isinstance(h, logging.FileHandler) and Path(getattr(h, "baseFilename", "")) == p for h in root.handlers)
    if already:
        return
    fh = logging.FileHandler(p, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter(fmt))
    root.addHandler(fh)


def _abs(path: str) -> str:
    return str(Path(path).expanduser().resolve())


def _load_json(path: str) -> dict:
    p = Path(path)
    raw = p.read_text()
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("input.json must be an object")
    return data


def _maybe_max_frames_from_checkpoint(checkpoint_path: str) -> int | None:
    # Best-effort: parse the safetensors metadata `config` JSON.
    try:
        with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
            md = f.metadata() or {}
        cfg_raw = md.get("config", "")
        if not cfg_raw:
            return None
        cfg = json.loads(cfg_raw)
        tr = cfg.get("transformer", {})
        max_pos = tr.get("positional_embedding_max_pos")
        if isinstance(max_pos, list) and max_pos and isinstance(max_pos[0], int):
            return int(max_pos[0])
        return None
    except Exception:
        return None


def _maybe(data: dict, key: str, default):  # noqa: ANN001
    return data[key] if key in data else default


def _as_bool_auto(v, default: bool) -> bool:  # noqa: ANN001
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, str) and v.strip().lower() == "auto":
        return default
    raise ValueError(f"Expected bool or 'auto', got {v!r}")


@dataclass(frozen=True)
class ImageCondition:
    path: str
    frame_idx: int
    strength: float


def _parse_images(data: dict) -> list[tuple[str, int, float]]:
    imgs = _maybe(data, "images", [])
    if not isinstance(imgs, list):
        raise ValueError("images must be a list")
    out: list[tuple[str, int, float]] = []
    for item in imgs:
        if not isinstance(item, dict):
            raise ValueError("each images[] item must be an object")
        out.append((_abs(item["path"]), int(item.get("frame_idx", 0)), float(item.get("strength", 0.85))))
    return out


def _parse_loras(items) -> list[LoraPathStrengthAndSDOps]:  # noqa: ANN001
    if items is None:
        return []
    if not isinstance(items, list):
        raise ValueError("loras/distilled_lora must be a list")
    out: list[LoraPathStrengthAndSDOps] = []
    for it in items:
        if not isinstance(it, dict):
            raise ValueError("each lora item must be an object")
        out.append(
            LoraPathStrengthAndSDOps(_abs(it["path"]), float(it.get("strength", 1.0)), LTXV_LORA_COMFY_RENAMING_MAP)
        )
    return out


def main() -> None:
    # Minimal default logging (may be reconfigured after reading JSON).
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    ap = argparse.ArgumentParser()
    ap.add_argument("input_json", type=str, help="Path to an input.json describing the run.")
    args = ap.parse_args()

    cfg = _load_json(args.input_json)
    # Optional: allow JSON to control python logging without adding CLI flags.
    log_level = str(_maybe(cfg, "log_level", "")).strip().upper()
    level = getattr(logging, log_level, logging.INFO) if log_level else logging.INFO
    log_file_raw = str(_maybe(cfg, "log_file", "")).strip()
    log_file = _abs(log_file_raw) if log_file_raw else os.environ.get("LTX_LOG_FILE")
    _configure_logging(level=level, log_file=log_file)

    log_system_summary("run_from_json")
    # Required core fields
    checkpoint_path = _abs(cfg["checkpoint_path"])
    output_path = _abs(cfg["output_path"])
    prompt = str(cfg["prompt"])

    # Optional fields
    pipeline = str(_maybe(cfg, "pipeline", "auto"))
    negative_prompt = str(_maybe(cfg, "negative_prompt", ""))
    height = int(_maybe(cfg, "height", 384))
    width = int(_maybe(cfg, "width", 640))
    num_frames = int(_maybe(cfg, "num_frames", 17))
    frame_rate = float(_maybe(cfg, "frame_rate", 24.0))
    num_inference_steps = int(_maybe(cfg, "num_inference_steps", 12))
    seed = int(_maybe(cfg, "seed", 10))
    enhance_prompt = bool(_maybe(cfg, "enhance_prompt", False))
    context_in_path_raw = str(_maybe(cfg, "context_in_path", "")).strip()
    context_in_path = _abs(context_in_path_raw) if context_in_path_raw else None
    context_in_paths_raw = _maybe(cfg, "context_in_paths", None)
    context_in_paths: list[str] | None = None
    if context_in_paths_raw is not None:
        if not isinstance(context_in_paths_raw, list):
            raise ValueError("context_in_paths must be a list of paths")
        context_in_paths = [_abs(str(p)) for p in context_in_paths_raw]
    context_out_path_raw = str(_maybe(cfg, "context_out_path", "")).strip()
    context_out_path = _abs(context_out_path_raw) if context_out_path_raw else None

    if context_in_path is None and context_in_paths:
        # Default to the first existing context file; if none exist yet, still pick the first path
        # so the downstream error is clear.
        for p in context_in_paths:
            if Path(p).exists():
                context_in_path = p
                break
        if context_in_path is None:
            context_in_path = context_in_paths[0]
        logger.info("context_in_path_selected=%s", context_in_path)

    text_encoder = str(_maybe(cfg, "text_encoder", "gemma-hf"))
    gemma_root_raw = _maybe(cfg, "gemma_root", "")
    gemma_root = _abs(gemma_root_raw) if gemma_root_raw else ""

    images = _parse_images(cfg)
    loras = _parse_loras(_maybe(cfg, "loras", []))

    distilled_lora = _parse_loras(_maybe(cfg, "distilled_lora", []))
    spatial_upsampler_path_raw = _maybe(cfg, "spatial_upsampler_path", "")
    spatial_upsampler_path = _abs(spatial_upsampler_path_raw) if spatial_upsampler_path_raw else ""

    temporal_upsampler_path_raw = str(_maybe(cfg, "temporal_upsampler_path", "")).strip()
    temporal_upsampler_path = _abs(temporal_upsampler_path_raw) if temporal_upsampler_path_raw else ""
    temporal_upsample = bool(_maybe(cfg, "temporal_upsample", False))

    # Default upsamplers / distilled LoRA if not provided and present in ./models.
    root = Path(__file__).resolve().parent
    default_spatial = root / "models" / "lightricksLTXV2_ltx2SpatialUpscaler.safetensors"
    default_temporal = root / "models" / "lightricksLTXV2_ltx2TemporalUpscaler.safetensors"
    default_distilled = root / "models" / "ltx-2-19b-distilled-lora-384.safetensors"

    if not spatial_upsampler_path and default_spatial.exists():
        spatial_upsampler_path = str(default_spatial.resolve())
    if not temporal_upsampler_path and default_temporal.exists():
        temporal_upsampler_path = str(default_temporal.resolve())
    if not distilled_lora and default_distilled.exists():
        distilled_lora = [LoraPathStrengthAndSDOps(str(default_distilled.resolve()), 1.0, None)]

    # Auto selection of pipeline and dispatch settings
    if pipeline == "auto":
        pipeline = "ti2vid_two_stages" if (distilled_lora and spatial_upsampler_path) else "ti2vid_one_stage"

    want_dispatch_default = torch.cuda.is_available() and torch.cuda.device_count() >= 2
    dispatch_transformer = _as_bool_auto(_maybe(cfg, "dispatch_transformer", "auto"), want_dispatch_default)

    transformer_device_map = str(_maybe(cfg, "transformer_device_map", "auto"))
    if transformer_device_map == "auto":
        transformer_device_map = "interleave-2gpu" if (torch.cuda.is_available() and torch.cuda.device_count() >= 2) else ""

    transformer_device_map_resolved = (
        resolve_transformer_device_map_preset(transformer_device_map) if dispatch_transformer else None
    )
    transformer_offload_dir_raw = _maybe(cfg, "transformer_offload_dir", "")
    transformer_offload_dir = _abs(transformer_offload_dir_raw) if transformer_offload_dir_raw else None

    if text_encoder not in ("gemma-hf", "gemma-hf-cpu", "gemma-bnb4"):
        raise ValueError("text_encoder must be 'gemma-hf', 'gemma-hf-cpu', or 'gemma-bnb4'")
    if not gemma_root:
        raise ValueError("gemma_root is required")

    gemma_device_map = str(_maybe(cfg, "gemma_device_map", "")).strip()
    gemma_move_vision_tower_to = str(_maybe(cfg, "gemma_move_vision_tower_to", "")).strip()

    quant_raw = _maybe(cfg, "quantization", "")
    quantization: QuantizationPolicy | None = None
    if isinstance(quant_raw, str):
        q = quant_raw.strip()
        if q:
            if q == "fp8-cast":
                quantization = QuantizationPolicy.fp8_cast()
            elif q == "fp8-scaled-mm":
                quantization = QuantizationPolicy.fp8_scaled_mm()
            else:
                raise ValueError("quantization must be '', 'fp8-cast', or 'fp8-scaled-mm'")
    elif quant_raw:
        raise ValueError("quantization must be a string (e.g. 'fp8-cast')")

    max_frames = _maybe_max_frames_from_checkpoint(checkpoint_path)
    if max_frames is not None and num_frames > max_frames:
        # This is not necessarily a hard limit: `positional_embedding_max_pos` is used to normalize
        # RoPE/positional coordinates, so exceeding it typically means *extrapolation* (quality may degrade).
        # Many longer-video examples are produced by chunking or by accepting extrapolation.
        strict = bool(_maybe(cfg, "strict_max_frames", False))
        msg = (
            f"num_frames={num_frames} exceeds checkpoint config.transformer.positional_embedding_max_pos[0]={max_frames}. "
            "This usually means positional extrapolation; it may work but can degrade quality or increase instability/OOM risk. "
            "If you want to hard-enforce this, set strict_max_frames=true. "
            f"Workarounds: lower frame_rate (e.g. {max_frames} frames at 4 fps = 5 sec), or generate in temporal chunks and stitch."
        )
        if strict:
            raise ValueError(msg)
        logger.warning("%s", msg)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    logger.info(
        "config pipeline=%s text_encoder=%s dispatch=%s device_map=%s output=%s",
        pipeline,
        text_encoder,
        dispatch_transformer,
        transformer_device_map,
        output_path,
    )

    if quantization is not None and dispatch_transformer:
        raise ValueError(
            "quantization currently requires dispatch_transformer=false (multi-device dispatch + quantization not wired)"
        )

    log_cuda_memory("pre_run")
    log_ram_memory("pre_run")
    log_nvidia_smi("pre_run")

    try:
        with torch.inference_mode():
            if pipeline == "ti2vid_one_stage":
                runner = TI2VidOneStagePipeline(
                    checkpoint_path=checkpoint_path,
                    gemma_root=gemma_root or None,
                    loras=loras,
                    device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
                    quantization=quantization,
                    transformer_device_map=dict(transformer_device_map_resolved) if transformer_device_map_resolved else None,
                    transformer_offload_dir=transformer_offload_dir,
                    text_encoder_backend=text_encoder,
                    gemma_device_map=gemma_device_map,
                    gemma_move_vision_tower_to=gemma_move_vision_tower_to,
                    temporal_upsampler_path=temporal_upsampler_path or None,
                    temporal_upsample=temporal_upsample,
                )
                t = Timer("pipeline_call")
                video, audio = runner(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    seed=seed,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    frame_rate=frame_rate,
                    num_inference_steps=num_inference_steps,
                    video_guider_params=MultiModalGuiderParams(),
                    audio_guider_params=MultiModalGuiderParams(),
                    images=images,
                    enhance_prompt=enhance_prompt,
                    context_in_path=context_in_path,
                    context_out_path=context_out_path,
                )
                t.done()
                from ltx_pipelines.utils.media_io import encode_video
                from ltx_pipelines.utils.constants import AUDIO_SAMPLE_RATE

                encode_video(video=video, fps=frame_rate, audio=audio, audio_sample_rate=AUDIO_SAMPLE_RATE, output_path=output_path, video_chunks_number=1)
            elif pipeline == "ti2vid_two_stages":
                if not distilled_lora or not spatial_upsampler_path:
                    raise ValueError("ti2vid_two_stages requires distilled_lora and spatial_upsampler_path")
                runner = TI2VidTwoStagesPipeline(
                    checkpoint_path=checkpoint_path,
                    distilled_lora=distilled_lora,
                    spatial_upsampler_path=spatial_upsampler_path,
                    temporal_upsampler_path=temporal_upsampler_path or None,
                    gemma_root=gemma_root or None,
                    loras=loras,
                    device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
                    quantization=quantization,
                    transformer_device_map=dict(transformer_device_map_resolved) if transformer_device_map_resolved else None,
                    transformer_offload_dir=transformer_offload_dir,
                    text_encoder_backend=text_encoder,
                    gemma_device_map=gemma_device_map,
                    gemma_move_vision_tower_to=gemma_move_vision_tower_to,
                    temporal_upsample=temporal_upsample,
                )
                t = Timer("pipeline_call")
                video, audio = runner(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    seed=seed,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    frame_rate=frame_rate,
                    num_inference_steps=num_inference_steps,
                    video_guider_params=MultiModalGuiderParams(),
                    audio_guider_params=MultiModalGuiderParams(),
                    images=images,
                    tiling_config=None,
                    enhance_prompt=enhance_prompt,
                    context_in_path=context_in_path,
                    context_out_path=context_out_path,
                )
                t.done()
                from ltx_pipelines.utils.media_io import encode_video
                from ltx_pipelines.utils.constants import AUDIO_SAMPLE_RATE

                encode_video(video=video, fps=frame_rate, audio=audio, audio_sample_rate=AUDIO_SAMPLE_RATE, output_path=output_path, video_chunks_number=1)
            else:
                raise ValueError(f"Unknown pipeline: {pipeline!r}")
    except torch.OutOfMemoryError:
        # Make sure we get a memory snapshot in logs for post-mortem.
        logger.exception("OOM: pipeline=%s output=%s", pipeline, output_path)
        log_cuda_memory("oom")
        log_ram_memory("oom")
        log_nvidia_smi("oom")
        raise

    log_cuda_memory("post_run")
    log_ram_memory("post_run")
    log_nvidia_smi("post_run")
    logger.info("done output=%s", output_path)


if __name__ == "__main__":
    main()
