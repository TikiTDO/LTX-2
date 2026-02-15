from __future__ import annotations

import logging
import os
import subprocess
import time
from dataclasses import dataclass

import torch


logger = logging.getLogger(__name__)


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
    except Exception as e:  # pragma: no cover
        return f"<failed {cmd!r}: {e!r}>"


def log_system_summary(prefix: str = "") -> None:
    prefix = (prefix + " ") if prefix else ""
    logger.info("%spython_pid=%s", prefix, os.getpid())
    logger.info("%storch=%s cuda=%s gpus=%s", prefix, torch.__version__, torch.version.cuda, torch.cuda.device_count())
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            try:
                props = torch.cuda.get_device_properties(i)
                logger.info("%sgpu[%s]=%s vram_gb=%.2f", prefix, i, props.name, props.total_memory / (1024**3))
            except Exception:  # pragma: no cover
                logger.info("%sgpu[%s]=<failed to read props>", prefix, i)
    logger.info("%sPYTORCH_CUDA_ALLOC_CONF=%r", prefix, os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))
    logger.info("%sLTX_LOG_MEM_EVERY=%r", prefix, os.environ.get("LTX_LOG_MEM_EVERY"))


@dataclass(frozen=True)
class RamInfo:
    total_b: int
    available_b: int


def _read_proc_meminfo() -> RamInfo | None:
    # Linux-only fallback; keeps telemetry dependency-free.
    try:
        total_kb = None
        avail_kb = None
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    total_kb = int(line.split()[1])
                elif line.startswith("MemAvailable:"):
                    avail_kb = int(line.split()[1])
        if total_kb is None or avail_kb is None:
            return None
        return RamInfo(total_b=total_kb * 1024, available_b=avail_kb * 1024)
    except Exception:  # pragma: no cover
        return None


def log_ram_memory(tag: str = "") -> None:
    tag_s = f" tag={tag}" if tag else ""
    info = _read_proc_meminfo()
    if info is None:
        logger.info("ram_mem%s <unavailable>", tag_s)
        return
    logger.info(
        "ram_mem%s available_gb=%.2f total_gb=%.2f",
        tag_s,
        info.available_b / (1024**3),
        info.total_b / (1024**3),
    )


def log_cuda_memory(tag: str = "") -> None:
    tag_s = f" tag={tag}" if tag else ""
    if not torch.cuda.is_available():
        logger.info("cuda_mem%s <cuda unavailable>", tag_s)
        return
    for i in range(torch.cuda.device_count()):
        try:
            free_b, total_b = torch.cuda.mem_get_info(i)
            allocated = torch.cuda.memory_allocated(i)
            reserved = torch.cuda.memory_reserved(i)
            logger.info(
                "cuda_mem%s gpu=%s free_gb=%.2f total_gb=%.2f allocated_gb=%.2f reserved_gb=%.2f",
                tag_s,
                i,
                free_b / (1024**3),
                total_b / (1024**3),
                allocated / (1024**3),
                reserved / (1024**3),
            )
        except Exception:  # pragma: no cover
            logger.info("cuda_mem%s gpu=%s <failed>", tag_s, i)


def reset_cuda_peak_memory_stats() -> None:
    if not torch.cuda.is_available():
        return
    for i in range(torch.cuda.device_count()):
        try:
            torch.cuda.reset_peak_memory_stats(i)
        except Exception:  # pragma: no cover
            pass


def log_cuda_peak_memory(tag: str = "") -> None:
    tag_s = f" tag={tag}" if tag else ""
    if not torch.cuda.is_available():
        logger.info("cuda_peak%s <cuda unavailable>", tag_s)
        return
    for i in range(torch.cuda.device_count()):
        try:
            max_alloc = torch.cuda.max_memory_allocated(i)
            max_reserved = torch.cuda.max_memory_reserved(i)
            logger.info(
                "cuda_peak%s gpu=%s max_allocated_gb=%.2f max_reserved_gb=%.2f",
                tag_s,
                i,
                max_alloc / (1024**3),
                max_reserved / (1024**3),
            )
        except Exception:  # pragma: no cover
            logger.info("cuda_peak%s gpu=%s <failed>", tag_s, i)


def log_nvidia_smi(tag: str = "") -> None:
    tag_s = f" tag={tag}" if tag else ""
    out = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader",
        ]
    )
    logger.info("nvidia_smi%s %s", tag_s, out.replace("\n", " | "))


class Timer:
    def __init__(self, name: str):
        self.name = name
        self.start = time.time()

    def done(self) -> float:
        dt = time.time() - self.start
        logger.info("timer name=%s seconds=%.3f", self.name, dt)
        return dt
