#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import torch

from ltx_core.text_encoders.gemma import encode_text
from ltx_pipelines.utils import ModelLedger
from ltx_pipelines.utils.telemetry import log_ram_memory, log_system_summary
from ltx_pipelines.utils.text_context import TextContexts, save_text_contexts


logger = logging.getLogger("extract_text_contexts")


def _abs(path: str) -> str:
    return str(Path(path).expanduser().resolve())


def _load_json(path: str) -> dict:
    p = Path(path)
    raw = p.read_text()
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("input.json must be an object")
    return data


def _maybe(data: dict, key: str, default):  # noqa: ANN001
    return data[key] if key in data else default


def _slug(s: str) -> str:
    out = []
    for ch in s.strip():
        if ch.isalnum():
            out.append(ch.lower())
        elif ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "unnamed"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    log_system_summary("extract_text_contexts")

    ap = argparse.ArgumentParser()
    ap.add_argument("input_json", type=str, help="Path to the same input.json used for video generation.")
    ap.add_argument(
        "--out-dir",
        type=str,
        default="output/text_contexts",
        help="Directory to write .pt context files into (default: output/text_contexts).",
    )
    ap.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run text encoding on. Default 'cpu'. (GPU may OOM.)",
    )
    args = ap.parse_args()

    cfg = _load_json(args.input_json)
    prompt = str(cfg["prompt"])
    negative_prompt = str(_maybe(cfg, "negative_prompt", ""))
    enhance_prompt = bool(_maybe(cfg, "enhance_prompt", False))
    if enhance_prompt:
        raise ValueError("enhance_prompt is not supported in extract_text_contexts.py (requires image processor/generate)")

    # Use CPU by default, and ensure we don't accidentally use GPUs.
    if args.device == "cpu":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    device = torch.device(args.device)

    checkpoint_path = _abs(cfg["checkpoint_path"])

    variants = _maybe(cfg, "text_encoder_extract", None)
    if variants is None:
        # Default: extract using the main gemma_root from the JSON.
        variants = [
            {
                "name": "default",
                "gemma_root": str(cfg["gemma_root"]),
                "text_encoder": "gemma-hf-cpu",
            }
        ]

    if not isinstance(variants, list) or not variants:
        raise ValueError("text_encoder_extract must be a non-empty list (or omitted for default behavior)")

    out_dir = Path(_abs(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    wrote: list[str] = []
    for idx, item in enumerate(variants):
        if not isinstance(item, dict):
            raise ValueError(f"text_encoder_extract[{idx}] must be an object")
        name = _slug(str(_maybe(item, "name", f"variant_{idx}")))
        gemma_root = _abs(str(item["gemma_root"]))
        backend = str(_maybe(item, "text_encoder", "gemma-hf-cpu"))
        if backend != "gemma-hf-cpu":
            raise ValueError(f"text_encoder_extract[{idx}].text_encoder must be 'gemma-hf-cpu' (got {backend!r})")

        out_path = str(_maybe(item, "out_path", str(out_dir / f"{name}.pt")))
        out_path = _abs(out_path)

        logger.info("extract variant=%s backend=%s gemma_root=%s out=%s", name, backend, gemma_root, out_path)
        log_ram_memory(f"before {name}")

        ledger = ModelLedger(
            dtype=torch.bfloat16,
            device=device,
            checkpoint_path=checkpoint_path,
            gemma_root_path=gemma_root,
            text_encoder_backend=backend,
        )

        with torch.inference_mode():
            enc = ledger.text_encoder()
            (v_context_p, a_context_p), (v_context_n, a_context_n) = encode_text(enc, [prompt, negative_prompt])

        save_text_contexts(
            out_path,
            TextContexts(
                v_context_p=v_context_p,
                a_context_p=a_context_p,
                v_context_n=v_context_n,
                a_context_n=a_context_n,
                prompt=prompt,
                negative_prompt=negative_prompt,
            ),
        )
        wrote.append(out_path)
        logger.info("wrote %s", out_path)
        log_ram_memory(f"after {name}")

    # Print as a machine-readable snippet that can be pasted back into input.json.
    print(json.dumps({"context_in_paths": wrote}, indent=2))


if __name__ == "__main__":
    main()

