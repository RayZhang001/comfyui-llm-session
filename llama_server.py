"""Managed ``llama-server`` process and its HTTP API.

Scope of this module: start a loopback-only llama.cpp server for one model /
projector / GPU / context / MTP configuration, reuse it while that
configuration is unchanged, restart it when the configuration changes, and
stop only the process this package started. It never changes ComfyUI's own
CUDA device and never touches a server it does not own.
"""

from __future__ import annotations

import atexit
import json
import os
import re
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

# Where to find llama-server. Override with the COMFYUI_LLM_SESSION_SERVER
# environment variable if your llama.cpp build lives elsewhere.
EXECUTABLE_ENV_VAR = "COMFYUI_LLM_SESSION_SERVER"
DEFAULT_EXECUTABLE = Path.home() / "software" / "llama.cpp" / "current" / "llama-server"

SERVER_LOG_PATH = "/tmp/comfyui-llm-session-server.log"
# A server that dies at startup is the case where the log matters most, so the
# previous runs' logs are kept instead of being truncated away by the next start.
PREVIOUS_LOG_PATHS = (SERVER_LOG_PATH + ".1", SERVER_LOG_PATH + ".2")
SERVER_PID_PATH = "/tmp/comfyui-llm-session-server.pid"

STARTUP_TIMEOUT_SECONDS = 300.0
REQUEST_TIMEOUT_SECONDS = 900.0

# Server settings proven on this machine (see specification): selected GPU for
# model and projector, all layers on GPU, flash attention, q8 KV cache, one
# slot, no context shift, MTP draft mode when the model supports it.
GPU_LAYERS = "99"
KV_CACHE_TYPE = "q8_0"

# llama.cpp's default verbosity (3) does not say which device received the
# weights; level 4 adds the device list, the per-device memory breakdown and
# the layer offload summary while staying compact.
LOG_VERBOSITY = "4"

# Log lines that show where the model and projector were actually placed.
_PLACEMENT_MARKERS = (
    "device_info",
    "using device",
    "offloaded",
    "model buffer size",
    "KV buffer size",
    "RS buffer size",
    "mmproj",
    "mtmd",
    "spec",
    "MTP",
    "n_slots",
    "CUDA error",
)

_lock = threading.RLock()
_handle: "_ServerHandle | None" = None
_devices: list[str] | None = None
_devices_for: str | None = None


class LlamaServerError(RuntimeError):
    """A server startup or request failure that the node shows to the user."""


@dataclass(frozen=True)
class ServerConfig:
    """Everything that requires a server restart when it changes.

    ``gpu`` is the device number as llama.cpp lists it in ComfyUI's own
    environment. ``mtp_draft_tokens`` of 0 means "no speculative decoding".
    """

    executable: str
    model_path: str
    projector_path: str | None
    gpu: int
    context_size: int
    mtp_draft_tokens: int
    gpu_layers: int = int(GPU_LAYERS)
    cpu_moe_layers: int = 0

    def description(self) -> str:
        projector = self.projector_path or "none"
        mtp = "off" if self.mtp_draft_tokens <= 0 else f"{self.mtp_draft_tokens} draft tokens"
        return (
            f"model={os.path.basename(self.model_path)} projector={os.path.basename(projector)} "
            f"gpu={self.gpu} ctx={self.context_size} MTP={mtp} "
            f"gpu_layers={self.gpu_layers} cpu_moe_layers={self.cpu_moe_layers} server={self.executable}"
        )


def resolve_executable() -> str:
    configured = os.environ.get(EXECUTABLE_ENV_VAR, "").strip()
    path = Path(configured) if configured else DEFAULT_EXECUTABLE
    if not path.is_file():
        raise LlamaServerError(
            f"llama-server was not found at '{path}'. Install llama.cpp or point "
            f"{EXECUTABLE_ENV_VAR} at the llama-server executable."
        )
    return str(path)


def device_list() -> list[str]:
    """llama.cpp's own device list (its numbering can differ from torch's)."""
    global _devices, _devices_for
    executable = resolve_executable()
    with _lock:
        if _devices is not None and _devices_for == executable:
            return _devices
        try:
            completed = subprocess.run(
                [executable, "--list-devices"],
                capture_output=True,
                text=True,
                timeout=120,
                env=_child_env(executable),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise LlamaServerError(f"Could not run '{executable} --list-devices': {exc}") from exc
        lines = [line.strip() for line in (completed.stdout or "").splitlines()
                 if re.match(r"^[A-Z]+\d*\s*:", line.strip())]
        if not lines:
            raise LlamaServerError(
                f"'{executable} --list-devices' reported no devices. "
                f"Output: {(completed.stderr or completed.stdout or 'empty').strip()[:400]}"
            )
        _devices, _devices_for = lines, executable
        return lines


# Raw device list for the GPU dropdown. Free memory changes constantly, so this is
# re-read with a short TTL instead of being cached for the process lifetime the way
# device_list() (which only matters for numbering) is.
_DEVICE_SNAPSHOT_TTL = 5.0
_device_snapshot: tuple[float, list[str]] | None = None
_GPU_LINE_RE = re.compile(
    r"^(?P<dev>[A-Za-z]+\d*)\s*:\s*(?P<name>.*?)\s*\((?P<total>\d+)\s*MiB,\s*(?P<free>\d+)\s*MiB free\)$"
)


def _snapshot_devices() -> list[str]:
    """llama.cpp's device lines, including current memory, at most every 20 s."""
    global _device_snapshot
    now = time.monotonic()
    if _device_snapshot and now - _device_snapshot[0] < _DEVICE_SNAPSHOT_TTL:
        return _device_snapshot[1]
    executable = resolve_executable()
    try:
        completed = subprocess.run(
            [executable, "--list-devices"], capture_output=True, text=True, timeout=120, env=_child_env(executable)
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise LlamaServerError(f"Could not run '{executable} --list-devices': {exc}") from exc
    lines = [
        line.strip()
        for line in (completed.stdout or "").splitlines()
        if re.match(r"^[A-Z]+\d*\s*:", line.strip())
    ]
    if not lines:
        raise LlamaServerError(f"'{executable} --list-devices' reported no devices.")
    _device_snapshot = (now, lines)
    return lines


def _short_device_name(name: str) -> str:
    """'NVIDIA GeForce RTX 3090' -> 'RTX 3090' (the vendor prefix says nothing)."""
    short = re.sub(r"^(NVIDIA|AMD|Intel|ATI)\s+", "", name)
    short = re.sub(r"^(GeForce|Radeon|Arc|Quadro|Tesla)\s+", "", short)
    return short or name


def gpu_choices() -> list[str]:
    """Dropdown labels for the GPU row: the device name, nothing else.

    A bare '0'/'1' says nothing about whether that device can take the model, and
    the llama.cpp index is not something you choose by, so the row shows the name.
    No memory figures: the label is the saved value, and a value that changes
    between page loads makes ComfyUI refuse the run. Memory is in the server state
    (status route, log tab). The index behind each label is kept in _label_devices
    so a picked label can be turned back into a device number.
    """
    global _label_devices
    lines = _snapshot_devices()
    parsed = []
    for line in lines:
        match = _GPU_LINE_RE.match(line)
        if not match:
            # Some other llama.cpp format (no memory shown): keep the line as is.
            parsed.append((line, line))
            continue
        parsed.append((match.group("dev"), _short_device_name(match.group("name"))))
    counts = {}
    for _dev, name in parsed:
        counts[name] = counts.get(name, 0) + 1
    labels = []
    devices = {}
    for dev, name in parsed:
        # Two cards of the same model need telling apart; the index only shows up
        # when the name alone would not do it.
        shown = f"{name} #{dev[-1:] if dev[-1:].isdigit() else dev}" if counts[name] > 1 else name
        labels.append(shown)
        devices[labels[-1]] = int(re.search(r"\d+", dev).group(0))
    if not labels:
        raise LlamaServerError(f"Could not read device names from: {'; '.join(lines)}")
    _label_devices = devices
    return labels


_label_devices: dict[str, int] = {}


def parse_gpu(value) -> int:
    """Turn a gpu_choices() label, 'CUDA1' or a plain number into a device number."""
    text = str(value).strip()
    if re.fullmatch(r"\d+", text):
        return int(text)
    match = re.fullmatch(r"CUDA(\d+)", text, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    try:
        labels = gpu_choices()
    except LlamaServerError as exc:
        raise LlamaServerError(
            f"Cannot verify selected GPU {value!r}: {exc} No other GPU was selected."
        ) from exc
    if text in labels:
        return _label_devices[text]
    # Old labels contained free-memory text. Only accept an unambiguous exact
    # name match; never reinterpret a missing device as a different GPU.
    wanted_name = text.split(" - ")[0].strip()
    matches = [label for label in labels if label == wanted_name]
    if len(matches) == 1:
        return _label_devices[matches[0]]
    raise LlamaServerError(
        f"Selected GPU {value!r} is unavailable to llama.cpp. "
        f"Available devices: {', '.join(labels) or 'none'}. "
        "Select an available GPU and retry; no fallback was used."
    )


def validate_gpu(gpu: int) -> str:
    """Return the llama.cpp device name for ``gpu`` or explain what is available."""
    device_name = f"CUDA{gpu}"
    available = _snapshot_devices()
    if not any(line.split(":", 1)[0].strip().upper() == device_name.upper() for line in available):
        raise LlamaServerError(
            f"llama.cpp does not have device '{device_name}'. Available devices: {'; '.join(available)}. "
            f"Set the GPU input to one of those numbers."
        )
    for line in available:
        if line.split(":", 1)[0].strip().upper() == device_name.upper():
            return line
    return device_name


@dataclass
class _ServerHandle:
    process: subprocess.Popen
    port: int
    config: ServerConfig
    log: object
    started_at: float = 0.0
    version: str = ""
    placement: tuple = ()
    load_seconds: float = 0.0
    exit_notice: bool = False


def _child_env(executable: str, visible_gpu: int | None = None) -> dict[str, str]:
    """Environment for the server process.

    Two deliberate isolation choices, both limited to the child process:

    * prepend the llama.cpp build's own library directory, so the server uses
      its own CUDA libraries instead of ComfyUI's torch ones;
    * when ``visible_gpu`` is set, hide every other GPU from the server. CUDA
      then numbers that physical device as 0, which means llama.cpp cannot
      create a context on ComfyUI's GPU (it would otherwise allocate memory
      there just by measuring devices, which can evict a diffusion model or
      fail outright when that GPU is nearly full).
    """
    env = dict(os.environ)
    library_dir = str(Path(executable).resolve().parent)
    env["LD_LIBRARY_PATH"] = os.pathsep.join([library_dir, env.get("LD_LIBRARY_PATH", "")]).rstrip(os.pathsep)
    if visible_gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(visible_gpu)
    return env


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _read_pidfile() -> dict[str, str]:
    try:
        with open(SERVER_PID_PATH, encoding="utf-8") as handle:
            fields = handle.read().strip().split("\n")
        return {line.split("=", 1)[0]: line.split("=", 1)[1] for line in fields if "=" in line}
    except (OSError, IndexError):
        return {}


def _owned_process_is_ours(pid: int, executable: str, port: str) -> bool:
    """Only accept a process we started, so no foreign server is ever killed."""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as handle:
            command_line = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return False
    return executable in command_line and "--port" in command_line and port in command_line


def _reap_stale_servers(executable: str) -> None:
    """Clean up a server left behind by a previous ComfyUI process.

    Only a server whose owning process is gone may be reaped, so a server owned
    by another live process is never killed.
    """
    fields = _read_pidfile()
    pid, port = fields.get("pid"), fields.get("port")
    owner = fields.get("owner")
    if not pid or not port:
        return
    try:
        pid_number = int(pid)
    except ValueError:
        return
    if pid_number <= 0 or not _owned_process_is_ours(pid_number, executable, port):
        return
    try:
        owner_number = int(owner or "-1")
    except ValueError:
        owner_number = -1
    if owner_number > 0 and owner_number != os.getpid() and os.path.exists(f"/proc/{owner_number}"):
        raise LlamaServerError(
            f"Another running process (pid {owner_number}) already owns a llama-server started by this "
            "package. Stop that ComfyUI instance first: two sessions would each load the model into VRAM."
        )
    print(f"[LLM Session] Stopping llama-server (pid {pid_number}) left over by a previous session.", flush=True)
    try:
        os.kill(pid_number, 15)
        for _ in range(40):
            if not os.path.exists(f"/proc/{pid_number}"):
                break
            time.sleep(0.25)
        else:
            os.kill(pid_number, 9)
    except ProcessLookupError:
        pass
    except PermissionError as exc:
        print(f"[LLM Session] Could not stop the leftover server: {exc}", flush=True)
    try:
        os.remove(SERVER_PID_PATH)
    except OSError:
        pass


def _failure_lines(limit: int = 8) -> list[str]:
    """The informative lines of a failed startup, gdb backtrace noise removed.

    llama.cpp's crash handler attaches gdb, so the tail of the log is usually a
    stack trace. The root-cause lines come first.
    """
    noise = ("#", "[New LWP", "warning:", "Using host libthread_db", "[Thread debugging", "[Inferior")
    strong = (
        "out of memory", "failed to load model", "failed to allocate", "unable to allocate",
        "failed to create", "cudamalloc failed", "cuda error", "exiting due to", "not supported",
        "unsupported", "does not contain", "abort", "invalid", "exceed",
    )
    weak = ("error", "failed", "cannot", "assert")
    try:
        with open(SERVER_LOG_PATH, encoding="utf-8", errors="replace") as handle:
            lines = [line.strip() for line in handle if line.strip()]
    except OSError:
        return []
    clean = [line for line in lines if not line.startswith(noise) and "0x0000" not in line
             and not any(token in line.lower() for token in noise)]
    strongest = [line for line in clean if any(token in line.lower() for token in strong)]
    if strongest:
        return strongest[:limit]
    informative = [line for line in clean if any(token in line.lower() for token in weak)]
    return (informative or clean[:limit])[:limit]


def _placement_lines() -> list[str]:
    """Server log lines that show which device actually received the weights."""
    try:
        with open(SERVER_LOG_PATH, encoding="utf-8", errors="replace") as handle:
            lines = [line.strip() for line in handle if line.strip()]
    except OSError:
        return []
    markers = tuple(marker.lower() for marker in _PLACEMENT_MARKERS)
    return [line for line in lines if any(marker in line.lower() for marker in markers)][:14]


def _build_line() -> str:
    """Build string the server prints at startup, e.g. 'version: ... (build 11870 ...)'."""
    try:
        with open(SERVER_LOG_PATH, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if "server build:" in line:
                    return line.split("server build:", 1)[1].strip()
    except OSError:
        pass
    return ""


def _start(config: ServerConfig) -> _ServerHandle:
    executable = config.executable
    _reap_stale_servers(executable)
    port = _free_port()
    # The server only ever sees the selected GPU, so llama.cpp numbers it as 0.
    device_name = "CUDA0"
    arguments = [
        executable,
        "-m", config.model_path,
        "--device", device_name,
        "-ngl", "auto" if config.gpu_layers == -1 else "all" if config.gpu_layers == -2 else str(config.gpu_layers),
        "-c", str(config.context_size),
        "-np", "1",
        "-fa", "on",
        "-ctk", KV_CACHE_TYPE,
        "-ctv", KV_CACHE_TYPE,
        "--no-context-shift",
        "-lv", LOG_VERBOSITY,
        "--host", "127.0.0.1",
        "--port", str(port),
    ]
    if config.cpu_moe_layers > 0:
        arguments += ["--n-cpu-moe", str(config.cpu_moe_layers)]
    if config.mtp_draft_tokens > 0:
        arguments += ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(config.mtp_draft_tokens)]
    if config.projector_path:
        arguments += ["--mmproj", config.projector_path, "--mmproj-device", device_name]

    selected = validate_gpu(config.gpu)
    _rotate_server_log()
    log = open(SERVER_LOG_PATH, "w", encoding="utf-8")
    print(
        f"[LLM Session] Starting llama-server for '{selected}'. The process only sees this GPU "
        f"(CUDA_VISIBLE_DEVICES={config.gpu}), so llama.cpp numbers it as {device_name} and cannot "
        "allocate memory on any other GPU.",
        flush=True,
    )
    try:
        spawned = time.monotonic()
        process = subprocess.Popen(
            arguments, stdout=log, stderr=subprocess.STDOUT, env=_child_env(executable, config.gpu)
        )
    except OSError as exc:
        log.close()
        raise LlamaServerError(f"Could not start '{executable}': {exc}") from exc

    try:
        with open(SERVER_PID_PATH, "w", encoding="utf-8") as handle:
            handle.write(f"pid={process.pid}\nport={port}\nexecutable={executable}\nowner={os.getpid()}\n")
    except OSError:
        pass

    handle = _ServerHandle(
        process=process,
        port=port,
        config=config,
        log=log,
        started_at=time.time(),
        version=_build_line(),
        placement=tuple(_placement_lines()),
    )
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if process.poll() is not None:
            log.close()
            detail = " | ".join(_failure_lines())
            raise LlamaServerError(
                f"llama-server exited during startup with code {process.returncode} "
                f"({config.description()}). GPU {config.gpu} may be busy or out of memory. "
                f"Log: {SERVER_LOG_PATH}. Server said: {detail}"
            )
        status = _health(port)
        if status == "ok":
            handle.load_seconds = round(time.monotonic() - spawned, 2)
            print(
                f"[LLM Session] Server ready on 127.0.0.1:{port} (pid {process.pid}) after "
                f"{handle.load_seconds:.2f} s (process start + model load). {config.description()}",
                flush=True,
            )
            for line in _placement_lines():
                print(f"[LLM Session] server: {line}", flush=True)
            return handle
        time.sleep(0.25)

    # _start() has not yet returned, so this handle is not in _handle and
    # stop() would not find it. Stop the process that actually timed out.
    _stop_handle(handle)
    raise LlamaServerError(
        f"llama-server did not become ready within {int(STARTUP_TIMEOUT_SECONDS)} s "
        f"({config.description()}). Log: {SERVER_LOG_PATH}. Server said: {' | '.join(_failure_lines())}"
    )


def _health(port: int) -> str:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
            if response.status == 200:
                payload = json.load(response)
                return "ok" if payload.get("status") == "ok" else "loading"
    except (OSError, urllib.error.URLError, ValueError):
        return "unreachable"
    return "unreachable"


def ensure_running(config: ServerConfig, before_start=None) -> _ServerHandle:
    """Reuse the running server, or restart it when the configuration changed.

    Run the optional preflight after stopping an old server and only when a new
    one is needed, so its own existing allocations are not counted twice.
    """
    global _handle
    with _lock:
        if _handle is not None:
            if _handle.process.poll() is None and _handle.config == config:
                return _handle
            reason = "configuration changed" if _handle.process.poll() is None else "the previous server exited"
            print(f"[LLM Session] Restarting llama-server because {reason}.", flush=True)
            _stop_locked()
        if before_start is not None:
            before_start(config)
        _handle = _start(config)
        return _handle


def stop() -> None:
    with _lock:
        _stop_locked()


def _stop_locked() -> None:
    global _handle
    handle, _handle = _handle, None
    if handle is None:
        return
    _stop_handle(handle)


def _stop_handle(handle: _ServerHandle) -> None:
    if handle.process.poll() is None:
        handle.process.terminate()
        try:
            handle.process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            handle.process.kill()
            handle.process.wait()
    try:
        handle.log.close()
    except OSError:
        pass
    try:
        os.remove(SERVER_PID_PATH)
    except OSError:
        pass
    print(f"[LLM Session] llama-server (pid {handle.process.pid}) stopped.", flush=True)


atexit.register(stop)


def _install_shutdown_handler() -> None:
    """Stop the owned server on SIGTERM too (atexit does not run for signals).

    The previous handler is restored and re-invoked, so stopping ComfyUI keeps
    working exactly as before.
    """
    try:
        previous = signal.getsignal(signal.SIGTERM)
    except (ValueError, OSError):
        return  # not the main thread: leave signal handling alone

    def handler(signum, frame):
        stop()
        signal.signal(signal.SIGTERM, previous if callable(previous) else signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    try:
        signal.signal(signal.SIGTERM, handler)
    except (ValueError, OSError):
        pass


_install_shutdown_handler()


def _reap_stale_on_import() -> None:
    """Clean up a server orphaned by a previous ComfyUI process at startup."""
    try:
        if os.path.exists(SERVER_PID_PATH):
            _reap_stale_servers(resolve_executable())
    except Exception as exc:  # never block ComfyUI startup
        print(f"[LLM Session] Could not clean up a previous llama-server: {exc}", flush=True)


_reap_stale_on_import()


def post_chat_completion(port: int, payload: dict) -> dict:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise LlamaServerError(f"llama-server request failed with HTTP {exc.code}: {body[:1200]}") from exc
    except (OSError, urllib.error.URLError) as exc:
        raise LlamaServerError(
            f"Could not reach llama-server on 127.0.0.1:{port}: {exc}. "
            f"The server may have crashed; see {SERVER_LOG_PATH}."
        ) from exc
    except ValueError as exc:
        raise LlamaServerError(f"llama-server returned a response that is not valid JSON: {exc}") from exc


def is_running() -> bool:
    with _lock:
        return _handle is not None and _handle.process.poll() is None


def _rotate_server_log() -> None:
    """Shift the current log aside before a new server truncates it."""
    try:
        if os.path.getsize(SERVER_LOG_PATH) == 0:
            return
    except OSError:
        return
    for index in range(len(PREVIOUS_LOG_PATHS) - 1, -1, -1):
        target = PREVIOUS_LOG_PATHS[index]
        source = PREVIOUS_LOG_PATHS[index - 1] if index else SERVER_LOG_PATH
        try:
            os.replace(source, target)
        except OSError:
            pass


def _append_log_line(line: str) -> None:
    """Mark events in the llama.cpp log itself, so they survive a rotation."""
    try:
        with open(SERVER_LOG_PATH, "a", encoding="utf-8") as handle:
            handle.write(line.rstrip() + "\n")
    except OSError:
        pass


def _log_summary(path: str) -> dict | None:
    try:
        stat = os.stat(path)
    except OSError:
        return None
    first = last = ""
    count = 0
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.strip():
                    continue
                count += 1
                if count == 1:
                    first = line.strip()
                last = line.strip()
    except OSError:
        return None
    return {
        "path": path,
        "lines": count,
        "size_bytes": stat.st_size,
        "modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)),
        "first_line": first[:200],
        "last_line": last[:200],
    }


def previous_logs() -> list[dict]:
    """The logs of earlier server runs that were rotated aside."""
    out = []
    for path in PREVIOUS_LOG_PATHS:
        summary = _log_summary(path)
        if summary:
            out.append(summary)
    return out


def device_label(gpu: int) -> str:
    """llama.cpp's own line for a device number, '' when it cannot be resolved."""
    try:
        return validate_gpu(int(gpu))
    except (LlamaServerError, TypeError, ValueError):
        return ""


def status(probe: bool = True) -> dict:
    """State of the server owned by this process, for the node's status panel.

    ``probe=False`` skips the ``/health`` request, so it can be called while a
    generation holds the server.
    """
    with _lock:
        handle = _handle
    if handle is None:
        return {
            "running": False,
            "health": "not started",
            "message": "No llama-server is running in this ComfyUI process yet. It starts on the next queued run.",
            "log_path": SERVER_LOG_PATH,
            "previous_logs": previous_logs(),
        }
    alive = handle.process.poll() is None
    if not alive:
        health = "exited"
        code = handle.process.poll()
        if not handle.exit_notice:
            # One line only: the llama.cpp output itself never enters ComfyUI's log.
            handle.exit_notice = True
            code_text = f"{code} (killed by signal {-code})" if code and code < 0 else str(code)
            _append_log_line(
                f"[comfyui-llm-session] llama-server (pid {handle.process.pid}) exited with code "
                f"{code_text} at {time.strftime('%H:%M:%S')}."
            )
            print(
                f"[LLM Session] llama-server (pid {handle.process.pid}) exited with code {code}. "
                f"Open the 'LLM Server log' tab, or tail {SERVER_LOG_PATH}.",
                flush=True,
            )
    elif probe:
        health = _health(handle.port)
    else:
        health = "ok"
    if not handle.version:
        # The server prints its build line while loading, which may not have
        # reached the log file yet; fall back to asking the executable.
        handle.version = _build_line() or server_version()
    if not handle.placement:
        # Same buffering reason: the device/offload lines can land in the log
        # slightly after the server reports healthy.
        handle.placement = tuple(_placement_lines())
    return {
        "running": alive and health == "ok",
        "health": health,
        "exit_code": None if alive else handle.process.poll(),
        "pid": handle.process.pid,
        "owner_pid": os.getpid(),
        "port": handle.port,
        "url": f"http://127.0.0.1:{handle.port}",
        "executable": handle.config.executable,
        "version": handle.version,
        "uptime_seconds": round(time.time() - handle.started_at, 1) if handle.started_at else None,
        "load_seconds": handle.load_seconds or None,
        "model": os.path.basename(handle.config.model_path),
        "projector": os.path.basename(handle.config.projector_path) if handle.config.projector_path else "none",
        "gpu": handle.config.gpu,
        "device": device_label(handle.config.gpu),
        "context_size": handle.config.context_size,
        "mtp_draft_tokens": handle.config.mtp_draft_tokens,
        "log_path": SERVER_LOG_PATH,
        "pid_file": SERVER_PID_PATH,
        "placement": list(handle.placement),
        "previous_logs": previous_logs(),
    }


# Kept separate from ComfyUI's own log on purpose: llama.cpp is chatty at the
# verbosity we run it with, and ComfyUI's log buffer only holds ~300 lines.
LOG_TAIL_MAX_LINES = 2000
LOG_TAIL_MAX_BYTES = 4_000_000


def log_tail(tail: int = 400, since: int = -1, which: str = "current") -> dict:
    """Newest lines of a llama-server log, for the 'LLM Server log' tab.

    ``which`` selects the running server's log ("current") or a rotated one
    ("1", "2"). The caller passes ``since`` = how many lines it already holds for
    this log generation; it should re-read from scratch when ``generation``
    changes or when ``absolute`` is false. Nothing here is copied into ComfyUI's
    log.
    """
    if which in {"1", "2"}:
        return _rotated_log_tail(PREVIOUS_LOG_PATHS[int(which) - 1], tail, since)
    info = status(probe=False)
    try:
        wanted = max(1, min(int(tail), LOG_TAIL_MAX_LINES))
    except (TypeError, ValueError):
        wanted = 400
    try:
        stat = os.stat(SERVER_LOG_PATH)
    except OSError:
        info.update({"generation": "", "total": 0, "absolute": True, "lines": [], "truncated": False})
        return info
    start = max(0, stat.st_size - LOG_TAIL_MAX_BYTES)
    absolute = start == 0
    with open(SERVER_LOG_PATH, encoding="utf-8", errors="replace") as handle:
        handle.seek(start)
        chunk = handle.read()
    lines = [line.rstrip() for line in chunk.splitlines() if line.strip()]
    if start:
        lines = lines[1:]  # the first line was cut mid-way by the seek
    total = len(lines)
    if absolute and 0 <= since <= total:
        window = lines[since + 1 :]
    else:
        window = lines[-wanted:]
    # The generation identifies *this* server's log, not each write: the pid
    # changes when the server restarts, and the file is truncated at startup.
    pid = info.get("pid") or 0
    info.update({
        "generation": f"{pid}-{stat.st_ino}",
        "log_age_seconds": round(time.time() - stat.st_mtime, 1),
        "total": total if absolute else None,
        "absolute": absolute,
        "lines": window[-LOG_TAIL_MAX_LINES:],
        "truncated": not absolute,
        "line_limit": LOG_TAIL_MAX_LINES,
    })
    return info


def _rotated_log_tail(path: str, tail: int, since: int) -> dict:
    """Same shape as log_tail(), but for a log that is no longer being written."""
    summary = _log_summary(path) or {"path": path, "lines": 0, "size_bytes": 0, "modified": "", "first_line": "", "last_line": ""}
    info = {
        "running": False,
        "health": "not started",
        "message": f"Log of an earlier llama-server run (written until {summary['modified']}, {summary['lines']} lines).",
        "log_path": path,
        "previous_logs": previous_logs(),
    }
    try:
        stat = os.stat(path)
    except OSError:
        info.update({"which": "missing", "generation": "", "total": 0, "absolute": True, "lines": [], "truncated": False})
        return info
    start = max(0, stat.st_size - LOG_TAIL_MAX_BYTES)
    with open(path, encoding="utf-8", errors="replace") as handle:
        handle.seek(start)
        chunk = handle.read()
    lines = [line.rstrip() for line in chunk.splitlines() if line.strip()]
    if start:
        lines = lines[1:]
    info.update({
        "which": "rotated",
        "log_path": path,
        "summary": summary,
        "generation": f"rotated-{int(stat.st_mtime_ns)}",
        "log_age_seconds": round(time.time() - stat.st_mtime, 1),
        "total": len(lines),
        "absolute": True,
        "lines": lines[-tail:] if tail and tail > 0 else lines[-400:],
        "truncated": start > 0,
        "line_limit": LOG_TAIL_MAX_LINES,
    })
    return info


_version_cache: str | None = None


def server_version() -> str:
    """Server build string, used in logs so results can be attributed."""
    global _version_cache
    if _version_cache:
        return _version_cache
    executable = resolve_executable()
    try:
        completed = subprocess.run(
            [executable, "--version"], capture_output=True, text=True, timeout=60, env=_child_env(executable)
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"unknown ({exc})"
    first = (completed.stdout or completed.stderr or "").strip().splitlines()
    _version_cache = first[0].strip() if first else "unknown"
    return _version_cache
