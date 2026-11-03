"""Conservative, approximate VRAM preflight for the external llama-server.

Only ComfyUI's own model manager can be asked to release memory. This is not a
reservation: other processes and inference allocations can change after checking.
"""
from __future__ import annotations

import os
import re
import subprocess

from . import llama_server, model_images

_MIB = 1024 * 1024
_GIB = 1024 * _MIB
_SPLIT = re.compile(r"^(.*-)(\d{5})-of-(\d{5})(\.gguf)$", re.IGNORECASE)


def _model_bytes(path: str) -> int:
    match = _SPLIT.match(path)
    paths = [path]
    if match:
        prefix, _, count, suffix = match.groups()
        paths = [f"{prefix}{i:05d}-of-{count}{suffix}" for i in range(1, int(count) + 1)]
    try:
        return sum(os.path.getsize(item) for item in paths)
    except OSError as exc:
        raise llama_server.LlamaServerError(f"Cannot estimate model VRAM: {exc}") from exc


def estimate_bytes(config: llama_server.ServerConfig) -> int:
    """File-size estimate plus padding, not a guarantee of actual GPU use.

    Auto is treated as full offload, so it does not silently fit a mostly-CPU
    model merely because other processes currently occupy the card. MoE CPU
    offload is ignored, deliberately overestimating it.
    """
    model_bytes = _model_bytes(config.model_path)
    if config.gpu_layers == 0:
        fraction = 0.0
    elif config.gpu_layers < 0:
        fraction = 1.0
    else:
        strings, numbers = model_images.gguf_metadata(
            config.model_path, string_keys=("general.architecture",), number_key_parts=("block_count",)
        )
        blocks = numbers.get(f"{strings.get('general.architecture', '')}.block_count")
        fraction = min(1.0, (config.gpu_layers + 2) / blocks) if blocks and blocks > 0 else 1.0
    projector = os.path.getsize(config.projector_path) if config.projector_path else 0
    # Projector workspace, KV/compute, CUDA overhead, and a safety margin.
    return int(model_bytes * fraction + projector * 2 + 2 * _GIB + config.context_size * 64 * 1024)


def _free_bytes(gpu: int) -> int:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--id", str(gpu), "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        return int(result.stdout.strip()) * _MIB
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise llama_server.LlamaServerError(f"Cannot check free VRAM on GPU {gpu}: {exc}") from exc


def free_comfy_vram(gpu: int) -> None:
    # Check the mapped physical device before unloading anything. Never evict
    # ComfyUI models on another card, and never touch a different process.
    import comfy.model_management as mm
    import torch

    device = mm.get_torch_device()
    if getattr(device, "type", None) != "cuda":
        return
    try:
        physical_index = torch.cuda._get_nvml_device_index(device)
    except (AttributeError, RuntimeError, ValueError, IndexError) as exc:
        raise llama_server.LlamaServerError(
            f"Cannot safely map ComfyUI's CUDA device to GPU {gpu}; no models were unloaded: {exc}"
        ) from exc
    if physical_index != gpu:
        return
    print(f"[LLM Session] Unloading ComfyUI-managed models from GPU {gpu} before LLM inference.", flush=True)
    # ComfyUI itself uses 1e32 as the sentinel for unloading a whole model.
    # Request more than the device's capacity to unload all eligible models.
    mm.free_memory(1e32, device)
    # ComfyUI's accounting includes unused torch cache; a separate process
    # cannot use that cache until it is returned to the CUDA driver.
    mm.soft_empty_cache()


def preflight(config: llama_server.ServerConfig, free_vram: bool) -> None:
    required = estimate_bytes(config)
    available = _free_bytes(config.gpu)
    if available < required:
        raise llama_server.LlamaServerError(
            f"Not enough estimated VRAM on GPU {config.gpu}: {available / _GIB:.2f} GiB free, "
            f"about {required / _GIB:.2f} GiB required including padding. "
            + ("ComfyUI could not free enough memory; other processes are not touched. " if free_vram else
               "Enable free_vram to release ComfyUI models, or free memory manually. ")
            + "The estimate is approximate; reduce offload or context if appropriate."
        )
    print(f"[LLM Session] VRAM preflight GPU {config.gpu}: {available / _GIB:.2f} GiB free, "
          f"estimated {required / _GIB:.2f} GiB including padding.", flush=True)
