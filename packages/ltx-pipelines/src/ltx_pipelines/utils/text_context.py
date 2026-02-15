from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class TextContexts:
    v_context_p: torch.Tensor
    a_context_p: torch.Tensor
    v_context_n: torch.Tensor
    a_context_n: torch.Tensor
    prompt: str | None = None
    negative_prompt: str | None = None


def load_text_contexts(path: str, *, device: torch.device) -> TextContexts:
    p = Path(path).expanduser().resolve()
    obj = torch.load(str(p), map_location="cpu", weights_only=True)
    if not isinstance(obj, dict):
        raise ValueError(f"Invalid contexts file (expected dict): {p}")
    for k in ("v_context_p", "a_context_p", "v_context_n", "a_context_n"):
        if k not in obj:
            raise ValueError(f"Invalid contexts file (missing {k}): {p}")
        if not isinstance(obj[k], torch.Tensor):
            raise ValueError(f"Invalid contexts file ({k} is not a Tensor): {p}")

    return TextContexts(
        v_context_p=obj["v_context_p"].to(device),
        a_context_p=obj["a_context_p"].to(device),
        v_context_n=obj["v_context_n"].to(device),
        a_context_n=obj["a_context_n"].to(device),
        prompt=str(obj["prompt"]) if "prompt" in obj and obj["prompt"] is not None else None,
        negative_prompt=str(obj["negative_prompt"]) if "negative_prompt" in obj and obj["negative_prompt"] is not None else None,
    )


def save_text_contexts(path: str, contexts: TextContexts) -> None:
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "v_context_p": contexts.v_context_p.detach().to("cpu"),
            "a_context_p": contexts.a_context_p.detach().to("cpu"),
            "v_context_n": contexts.v_context_n.detach().to("cpu"),
            "a_context_n": contexts.a_context_n.detach().to("cpu"),
            "prompt": contexts.prompt,
            "negative_prompt": contexts.negative_prompt,
        },
        str(p),
    )

