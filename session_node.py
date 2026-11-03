"""The LLM session node: inputs, session policy and execution.

One node, one text output. The node keeps a managed local server running so
that continuing a conversation does not reload the model, and it keeps the
conversation's own history separately from that server:

* "continue"  - later runs append to the stored conversation. Earlier
  instructions and answers therefore keep influencing the answer.
* "fresh"     - "Reset session", or the "Clear conversation" button, drops the
  stored conversation. The server and its model stay loaded.

Only one conversation is active at a time in a ComfyUI process. If two node
instances are used, the last one that ran owns the conversation; the other one
simply starts a new conversation when it runs again.
"""

from __future__ import annotations

import asyncio
import threading
from contextlib import contextmanager
from dataclasses import dataclass

from aiohttp import web
from server import PromptServer

from . import llama_server, model_images, vram_guard

NODE_NAME = "ComfyLLMSession"
UI_STATS_KEY = "llm_session_stats"

# Sampling settings proven on this machine. Kept as constants so a future
# backend can expose its own defaults instead.
SAMPLING_TOP_P = 0.8
SAMPLING_TOP_K = 20
SAMPLING_MIN_P = 0.0

# Model-specific chat-template option. It is only sent when the model's own
# chat template mentions it, so models that do not know it are unaffected.
THINKING_OPTION = "enable_thinking"

MAX_OVERFLOW_RETRIES = 12
CONVERSATION_LOG_PREFIX = "[LLM Session]"

_lock = threading.RLock()
_conversation: dict | None = None
_clear_revisions: dict[str, int] = {}
_template_option_cache: dict[str, bool] = {}
_server_build_logged: str | None = None


@dataclass
class RequestOutcome:
    """Parsed answer plus the metrics the server actually reported."""

    text: str
    tokens_per_second: float | None
    predicted_tokens: int | None
    prompt_tokens: int | None
    cached_tokens: int | None
    context_used: int | None
    context_estimate: bool
    draft_tokens: int | None
    accepted_tokens: int | None
    prefill_ms: float | None = None
    prefill_tokens_per_second: float | None = None
    generation_ms: float | None = None


def _log(message: str) -> None:
    print(f"{CONVERSATION_LOG_PREFIX} {message}", flush=True)


def _clear_conversation_locked(node_id: str | None) -> None:
    global _conversation
    if node_id is not None:
        _clear_revisions[node_id] = _clear_revisions.get(node_id, 0) + 1
    if _conversation is not None and (node_id is None or _conversation["key"][0] == node_id):
        _conversation = None


@PromptServer.instance.routes.post("/comfyui_llm_session/clear")
async def clear_conversation_route(request):
    """Clear one node's conversation so the next queued run starts fresh."""
    try:
        payload = await request.json()
    except ValueError:
        return web.json_response({"error": "The clear request body is not JSON."}, status=400)
    node_id = str(payload.get("node_id", ""))
    if not node_id:
        return web.json_response({"error": "The clear request is missing node_id."}, status=400)
    if not _lock.acquire(blocking=False):
        return web.json_response(
            {"error": "Busy: a generation is running. Wait for it to finish, then clear again."},
            status=409,
        )
    try:
        _clear_conversation_locked(node_id)
    finally:
        _lock.release()
    return web.json_response({"ok": True, "server_running": llama_server.is_running()})


@PromptServer.instance.routes.get("/comfyui_llm_session/server")
async def server_status_route(request):
    """Live state of the llama-server this process manages (model, port, health...)."""
    return web.json_response(await asyncio.to_thread(llama_server.status))


@PromptServer.instance.routes.get("/comfyui_llm_session/server/log")
async def server_log_route(request):
    """Tail of the llama-server log for the 'LLM Server log' bottom-panel tab.

    On purpose: the server's own output never goes through ComfyUI's stdout, so
    ComfyUI's log stays clean and this is only read when somebody looks.
    """
    query = request.query
    try:
        tail = int(query.get("tail", 400))
    except ValueError:
        tail = 400
    try:
        since = int(query.get("since", -1))
    except ValueError:
        since = -1
    which = query.get("which", "current")
    if which not in ("current", "1", "2"):
        which = "current"
    return web.json_response(await asyncio.to_thread(llama_server.log_tail, tail, since, which))


@PromptServer.instance.routes.get("/comfyui_llm_session/projector")
async def projector_choice_route(request):
    """Which vision projector to use for a model (see model_images.select_projector).

    The frontend asks this when the node loads and when the model changes, so the
    name-matching rule lives in one place instead of being duplicated in JS.
    """
    model = request.query.get("model", "")
    current = request.query.get("current", "")
    pinned = request.query.get("text_only_pinned", "") in ("1", "true")
    if not model:
        return web.json_response({"error": "model query parameter required"}, status=400)
    decision = await asyncio.to_thread(model_images.select_projector, model, current, pinned)
    decision["model"] = model
    decision["choices"] = model_images.projector_choices()
    return web.json_response(decision)


def _is_context_overflow(message: str) -> bool:
    lowered = message.lower()
    return ("exceed" in lowered and "context" in lowered) or "context_size_error" in lowered


def _shrink_messages(messages: list[dict], fresh_messages: list[dict]) -> tuple[list[dict], str]:
    """Drop the oldest exchange after the initial image-bearing turn."""
    system_turn = 1 if messages and messages[0]["role"] == "system" else 0
    if len(messages) > system_turn + 3:
        shortened = messages[: system_turn + 2] + messages[system_turn + 4 :]
        return shortened, "context full: removed the oldest follow-up exchange"
    return list(fresh_messages), "context full: started a new conversation with the current prompt and images"


def _thinking_option_supported(model_path: str) -> bool:
    supported = _template_option_cache.get(model_path)
    if supported is None:
        supported = model_images.template_supports_option(model_path, THINKING_OPTION)
        _template_option_cache[model_path] = supported
        _log(
            f"chat template {'mentions' if supported else 'does not mention'} '{THINKING_OPTION}': "
            f"{'sending' if supported else 'not sending'} it to the server."
        )
    return supported


def _parse_outcome(result: dict) -> RequestOutcome:
    choices = result.get("choices") or [{}]
    text = (choices[0].get("message") or {}).get("content") or ""
    timings = result.get("timings") or {}
    usage = result.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}

    def number(*values):
        for value in values:
            if isinstance(value, (int, float)):
                return value
        return None

    prompt_tokens = number(usage.get("prompt_tokens"), timings.get("prompt_n"))
    cached_tokens = number(details.get("cached_tokens"), timings.get("cache_n"))
    predicted = number(timings.get("predicted_n"), usage.get("completion_tokens"))

    # The server documents the context as prompt_n + cache_n + predicted_n.
    parts = (timings.get("prompt_n"), timings.get("cache_n"), timings.get("predicted_n"))
    if all(isinstance(part, (int, float)) for part in parts):
        context_used, estimate = int(sum(parts)), False
    else:
        context_used, estimate = number(usage.get("total_tokens")), True

    return RequestOutcome(
        text=text.strip(),
        tokens_per_second=number(timings.get("predicted_per_second")),
        predicted_tokens=predicted,
        prompt_tokens=prompt_tokens,
        cached_tokens=cached_tokens,
        context_used=context_used,
        context_estimate=estimate,
        draft_tokens=number(timings.get("draft_n")),
        accepted_tokens=number(timings.get("draft_n_accepted")),
        prefill_ms=number(timings.get("prompt_ms")),
        prefill_tokens_per_second=number(timings.get("prompt_per_second")),
        generation_ms=number(timings.get("predicted_ms")),
    )


@contextmanager
def _server_lifetime(policy: str):
    """Release the owned server after inference when requested, even on failure."""
    if policy not in ("keep_loaded", "unload_after_generation"):
        raise ValueError(f"Unknown server lifetime {policy!r}.")
    result = {}
    try:
        yield result
    finally:
        if policy == "unload_after_generation":
            llama_server.stop()
            if "stats" in result:
                result["stats"]["server_unloaded"] = not llama_server.is_running()


class LLMSessionNode:
    @classmethod
    def INPUT_TYPES(cls):
        models = model_images.model_choices() or ["no LLM models found"]
        projectors = model_images.projector_choices()
        # Show 'CUDA1: NVIDIA GeForce RTX 3090' rather than a bare number, using
        # llama.cpp's own device list so the numbering matches what it will use.
        try:
            gpu_input = (
                llama_server.gpu_choices(),
                {
                    "tooltip": "GPU for the llama.cpp server, by name. The server is given this device "
                    "alone and cannot touch the others. Independent of ComfyUI's own GPU selection."
                },
            )
        except llama_server.LlamaServerError as exc:
            # Keep the input a combo even when discovery fails so old named
            # selections are not converted to integers during validation.
            _log(f"GPU list unavailable: {exc}")
            gpu_input = (
                ["(GPU unavailable)"],
                {"tooltip": f"Could not list GPUs: {exc}"},
            )
        return {
            "required": {
                # Order here is the order of the rows on the node. Keeping 'gpu'
                # first puts the server choice above the model choice.
                "gpu": gpu_input,
                "model": (models, {"tooltip": "GGUF language model from the ComfyUI LLM model folders."}),
                "vision_projector": (
                    projectors,
                    {"tooltip": f"Matching mmproj file. The node fills this in from the model name; a projector you pick yourself is kept as long as it plausibly belongs to the model. '{model_images.NO_PROJECTOR}' runs the model as text-only and is remembered as a deliberate choice."},
                ),
                "prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "The request for this run. Must not be blank."}),
                "system_prompt": (
                    "STRING",
                    {"default": "Describe only what is visible.", "multiline": True, "tooltip": "Starts a new conversation when changed. May be blank."},
                ),
                "context_size": ("INT", {"default": 4096, "min": 2048, "max": 32768, "step": 256, "tooltip": "Prompt + generated tokens share this. Images cost a lot: a 1024 px image is roughly 700 tokens."}),
                "max_tokens": ("INT", {"default": 512, "min": 1, "max": 4096, "tooltip": "Maximum tokens generated by this run."}),
                "max_image_edge": ("INT", {"default": 1024, "min": 128, "max": 4096, "step": 64, "tooltip": "Images are resized to this before the vision encoder; smaller means fewer prompt tokens."}),
                "temperature": ("FLOAT", {"default": 0.55, "min": 0.0, "max": 2.0, "step": 0.01, "tooltip": "Only temperature varies between runs; top_p/top_k/min_p stay at llama.cpp defaults."}),
                "seed": ("INT", {"default": 37, "min": 0, "max": 4294967295, "control_after_generate": True}),
                "reset_session": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "On: every run starts a new conversation. Off: the conversation continues."},
                ),
                "mtp_draft_tokens": ("INT", {"default": 2, "min": 1, "max": 8, "tooltip": "Draft tokens for MTP speculative decoding (requires a model with MTP/NextN layers)."}),
            },
            "optional": {
                # Optional for API workflows saved before these controls existed.
                "server_lifetime": (["keep_loaded", "unload_after_generation"], {"default": "keep_loaded", "tooltip": "Keep the model loaded for reuse, or stop the server after generation to release its GPU memory before downstream nodes run."}),
                "gpu_layers": ("INT", {"default": 99, "min": -2, "max": 999, "step": 1, "tooltip": "GPU layer count; 0 = CPU, -1 = llama.cpp auto, -2 = all. Default 99 preserves previous behavior."}),
                "cpu_moe_layers": ("INT", {"default": 0, "min": 0, "max": 999, "step": 1, "tooltip": "Keep the expert weights of the first N MoE layers on CPU; 0 leaves llama.cpp's default unchanged."}),
                "free_vram": ("BOOLEAN", {"default": False, "tooltip": "On every execution, ask ComfyUI to unload its own models from the selected GPU before LLM inference. Other processes are not touched. The node still refuses to start if estimated free VRAM is insufficient."}),
                **{slot: ("IMAGE", {"tooltip": "Optional image. The first frame of a batch is used."})
                   for slot in model_images.IMAGE_SLOTS},
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "run"
    CATEGORY = "LLM Session"
    SEARCH_ALIASES = ["llm", "vision", "describe image", "prompt enhance", "chat", "llama.cpp"]
    # Shown by the frontend as the tooltip when hovering this node's title bar.
    # The full reference lives in web/docs/ComfyLLMSession.md (Info tab).
    DESCRIPTION = (
        "Prompt plus optional images in, text out, using a llama.cpp server this "
        "node starts on its own GPU. One conversation per node: re-running "
        "unchanged is cached and the conversation continues, until you press Clear "
        "conversation or change the image, system prompt, model, projector, GPU or "
        "context size. The bottom line of the node shows speed, turn number and "
        "context use; hover a row for that parameter; the Info tab has the full "
        "reference."
    )

    @classmethod
    def VALIDATE_INPUTS(cls, gpu):
        # Let a saved, currently missing GPU reach run(), which produces a clear
        # execution error without silently changing the selected device. Other
        # inputs keep ComfyUI's normal validation.
        return True

    @classmethod
    def IS_CHANGED(cls, unique_id=None, **kwargs):
        # Normal ComfyUI caching otherwise applies: unchanged inputs do not run
        # again. Clearing the conversation changes this value, so the next run
        # is executed.
        return _clear_revisions.get(str(unique_id), 0)

    def run(self, model, vision_projector, prompt, system_prompt, gpu, context_size, max_tokens,
            max_image_edge, temperature, seed, reset_session, mtp_draft_tokens, unique_id=None,
            server_lifetime="keep_loaded", gpu_layers=99, cpu_moe_layers=0, free_vram=False, **slots):
        global _conversation
        if not str(prompt).strip():
            raise ValueError("The prompt must not be empty.")

        model_path = model_images.resolve_model_path(model, label="Model")
        projector_name = vision_projector
        if projector_name == model_images.AUTO_PROJECTOR:
            # Only workflows saved before the projector row stopped offering
            # '(auto)' can still carry it; resolve it instead of failing.
            projector_name = model_images.auto_match_projector(model)
            if projector_name is None:
                _log(
                    f"'{model_images.AUTO_PROJECTOR}' found no mmproj whose name matches '{model}'. "
                    "Running without a vision projector."
                )
                projector_name = model_images.NO_PROJECTOR
            else:
                _log(f"'{model_images.AUTO_PROJECTOR}' resolved to vision projector '{projector_name}'.")
        projector_path = model_images.resolve_model_path(projector_name, label="Vision projector")
        gpu_index = llama_server.parse_gpu(gpu)
        device = llama_server.validate_gpu(gpu_index)

        prepared = self._prepare_images(slots, int(max_image_edge))
        if prepared and projector_path is None:
            raise ValueError(
                "Image inputs need a vision projector. Choose the model's mmproj file in "
                "'vision_projector', or disconnect the images to run the model as text-only."
            )
        identity = tuple((image.slot, image.digest) for image in prepared)
        mtp_layers = model_images.mtp_predict_layers(model_path)
        mtp_requested = int(mtp_draft_tokens)
        if mtp_requested > 0 and mtp_layers <= 0:
            _log(
                f"MTP requested with {mtp_requested} draft tokens, but this model declares no "
                "MTP/NextN layers. Starting the server without speculative decoding."
            )
        mtp_tokens = mtp_requested if mtp_layers > 0 else 0
        config = llama_server.ServerConfig(
            executable=llama_server.resolve_executable(),
            model_path=model_path,
            projector_path=projector_path,
            gpu=gpu_index,
            context_size=int(context_size),
            mtp_draft_tokens=mtp_tokens,
            gpu_layers=int(gpu_layers),
            cpu_moe_layers=int(cpu_moe_layers),
        )
        session_key = (str(unique_id), identity, system_prompt, config)

        with _lock, _server_lifetime(server_lifetime) as lifetime:
            started = _conversation is None or _conversation["key"] != session_key
            if reset_session:
                started = True
            first_turn_content = self._user_content(prompt, prepared)
            messages = list(_conversation["messages"]) if not started else []
            if not started:
                # Continuation keeps the system prompt, the image-bearing first
                # turn and the previous exchanges, then appends this request.
                messages.append({"role": "user", "content": prompt})
                turns = int(_conversation.get("turns", 0)) + 1
            else:
                messages = self._initial_messages(system_prompt, first_turn_content)
                turns = 1

            if free_vram:
                vram_guard.free_comfy_vram(config.gpu)
            handle = llama_server.ensure_running(
                config, before_start=lambda cfg: vram_guard.preflight(cfg, bool(free_vram))
            )
            server = llama_server.status(probe=False)
            self._log_server_build(config)
            payload = self._request_payload(max_tokens, temperature, seed, model_path)

            attempt = 0
            while True:
                payload["messages"] = messages
                try:
                    result = llama_server.post_chat_completion(handle.port, payload)
                    break
                except llama_server.LlamaServerError as exc:
                    message = str(exc)
                    if not _is_context_overflow(message):
                        raise
                    if attempt >= MAX_OVERFLOW_RETRIES:
                        raise RuntimeError(
                            f"The conversation still does not fit after {attempt} retries "
                            f"(context size {context_size}). Raise the context size or send fewer/smaller images."
                        ) from exc
                    attempt += 1
                    fresh = self._initial_messages(system_prompt, first_turn_content)
                    shrunk, reason = _shrink_messages(messages, fresh)
                    if shrunk == messages:
                        raise RuntimeError(
                            "The current prompt and images do not fit in the configured context "
                            f"({context_size} tokens). Raise the context size, lower 'max image edge', "
                            "or shorten the prompt. Server message: " + message[:400]
                        ) from exc
                    messages = shrunk
                    if messages == fresh:
                        turns = 1
                        started = True
                    _log(f"{reason} (attempt {attempt})")

            outcome = _parse_outcome(result)
            assistant_message = (result.get("choices") or [{}])[0].get("message") or {"role": "assistant", "content": outcome.text}
            _conversation = {
                "key": session_key,
                "messages": messages + [assistant_message],
                "turns": turns,
            }

            self._log_outcome(outcome, turns, started, config, device, attempt, server.get("load_seconds"))
            stats = {
                "turns": turns,
                "continued": not started,
                "tokens_per_second": outcome.tokens_per_second,
                "predicted_tokens": outcome.predicted_tokens,
                "prompt_tokens": outcome.prompt_tokens,
                "cached_tokens": outcome.cached_tokens,
                "context_used": outcome.context_used,
                "context_size": int(context_size),
                "context_estimate": outcome.context_estimate,
                "draft_tokens": outcome.draft_tokens,
                "accepted_tokens": outcome.accepted_tokens,
                "mtp_requested": mtp_requested,
                "mtp_draft_tokens": mtp_tokens,
                "server_lifetime": server_lifetime,
                "server_unloaded": False,
                "gpu_layers": int(gpu_layers),
                "cpu_moe_layers": int(cpu_moe_layers),
                "free_vram": bool(free_vram),
                "images": len(prepared),
                "overflow_retries": attempt,
                "device": device,
                # Timing breakdown: prefill/generation come from the server,
                # model load is measured by this package when the server starts.
                "prefill_seconds": None if outcome.prefill_ms is None else round(outcome.prefill_ms / 1000, 3),
                "prefill_tokens_per_second": outcome.prefill_tokens_per_second,
                "generation_seconds": None if outcome.generation_ms is None else round(outcome.generation_ms / 1000, 3),
                "model_load_seconds": server.get("load_seconds"),
                # Cheap summary (no extra /health probe) so the panel can point at
                # the running server without another request.
                "server_pid": server.get("pid"),
                "server_port": server.get("port"),
                "server_uptime_seconds": server.get("uptime_seconds"),
                "llama_version": server.get("version", ""),
                # State for the node's title pill and colour: "ok" or "warn".
                # "idle" (never run) and "error" are decided by the frontend,
                # which also sees execution_error events. A skipped MTP is only
                # a note: most GGUFs have no MTP layers, and the node is fine.
                "state": "warn" if attempt else "ok",
                "note": (
                    f"context full: dropped the oldest exchange ({attempt}x)"
                    if attempt
                    else ("MTP skipped: this GGUF has no MTP/NextN layers" if mtp_requested > 0 and mtp_tokens == 0 else "")
                ),
            }
            lifetime["stats"] = stats
            return {"ui": {UI_STATS_KEY: [stats]}, "result": (outcome.text,)}

    @staticmethod
    def _prepare_images(slots: dict, max_image_edge: int) -> list[model_images.PreparedImage]:
        prepared = []
        for slot in model_images.IMAGE_SLOTS:
            tensor = slots.get(slot)
            if tensor is None:
                continue
            if getattr(tensor, "shape", (0,)) and tensor.shape[0] == 0:
                raise ValueError(f"The connected image slot '{slot}' is empty.")
            image = model_images.prepare_image(tensor, slot, 0, max_image_edge)
            prepared.append(image)
            if image.sent_size != image.source_size:
                _log(
                    f"{slot}: resized {image.source_size[0]}x{image.source_size[1]} -> "
                    f"{image.sent_size[0]}x{image.sent_size[1]}, JPEG {image.byte_size / 1024:.1f} KiB"
                )
        if prepared:
            _log("prepared images: " + ", ".join(f"{image.slot} {image.digest[:12]}" for image in prepared))
        return prepared

    @staticmethod
    def _user_content(prompt: str, prepared: list[model_images.PreparedImage]):
        if not prepared:
            return prompt
        # Text first, then the images in slot order: the same order every turn,
        # which is what lets the server reuse the shared prompt prefix.
        return [
            {"type": "text", "text": prompt},
            *(
                {"type": "image_url", "image_url": {"url": image.data_url()}}
                for image in prepared
            ),
        ]

    @staticmethod
    def _initial_messages(system_prompt: str, first_turn_content) -> list[dict]:
        messages = []
        if str(system_prompt).strip():
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": first_turn_content})
        return messages

    @staticmethod
    def _request_payload(max_tokens: int, temperature: float, seed: int, model_path: str) -> dict:
        payload = {
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "seed": int(seed),
            "top_p": SAMPLING_TOP_P,
            "top_k": SAMPLING_TOP_K,
            "min_p": SAMPLING_MIN_P,
            "cache_prompt": True,
        }
        if _thinking_option_supported(model_path):
            payload["chat_template_kwargs"] = {THINKING_OPTION: False}
        return payload

    @staticmethod
    def _log_server_build(config: llama_server.ServerConfig) -> None:
        global _server_build_logged
        if _server_build_logged == config.executable:
            return
        _server_build_logged = config.executable
        _log(f"server build: {llama_server.server_version()}")

    @staticmethod
    def _log_outcome(outcome: RequestOutcome, turns: int, started: bool, config: llama_server.ServerConfig,
                     device: str, retries: int, load_seconds: float | None = None) -> None:
        def show(value, suffix=""):
            if value is None:
                return "unknown"
            if isinstance(value, float):
                value = round(value, 2)
            return f"{value}{suffix}"

        _log(
            f"{'new' if started else 'continued'} conversation, turn {turns} | {config.description()} | {device}"
        )
        _log(
            f"prompt_tokens={show(outcome.prompt_tokens)} cached_prompt_tokens={show(outcome.cached_tokens)} "
            f"generated={show(outcome.predicted_tokens)} speed={show(outcome.tokens_per_second, ' tok/s')} "
            f"MTP draft/accepted={show(outcome.draft_tokens)}/{show(outcome.accepted_tokens)}"
            + (f" | context retries={retries}" if retries else "")
        )
        if outcome.cached_tokens == 0:
            _log("The server reported no reused prompt prefix for this request.")
        elif outcome.cached_tokens is None:
            _log("The server did not report a cached-token count for this request.")
        prefill_seconds = None if outcome.prefill_ms is None else outcome.prefill_ms / 1000
        generation_seconds = None if outcome.generation_ms is None else outcome.generation_ms / 1000
        _log(
            f"timing: model load={show(load_seconds, ' s')} prefill={show(prefill_seconds, ' s')}"
            f" ({show(outcome.prefill_tokens_per_second, ' tok/s')}) generation={show(generation_seconds, ' s')}"
        )
