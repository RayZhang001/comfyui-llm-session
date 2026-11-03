# ComfyUI LLM Session

One ComfyUI node that answers a text request, optionally with up to eight
images, using a local inference server. The initial backend is
**llama.cpp** (`llama-server`). The package name and the node contract are
backend-neutral; no other backend is implemented yet.

Keep a model loaded between requests, continue a conversation, and reuse its
prompt prefix when llama.cpp reports a cache hit. You can also start a fresh
conversation each run or unload the server before downstream image-generation
nodes need the GPU.

## Current platform support

This baseline was developed and integration-tested on Linux with NVIDIA CUDA.
It currently uses Linux `/tmp` paths and `/proc` process inspection. Windows
support is pending; copying this version into Windows ComfyUI is not sufficient.
macOS, Metal, Vulkan, ROCm and CPU-only operation have not been validated.
Setting `gpu_layers=0` does not remove the CUDA device and NVIDIA VRAM-check
requirements.

## Requirements

- An existing ComfyUI installation (no extra Python environment, no extra pip
  packages: only Pillow/NumPy/aiohttp, which ComfyUI already has).
- An existing `llama-server` binary. Default location:
  `~/software/llama.cpp/current/llama-server`. Override with the environment
  variable `COMFYUI_LLM_SESSION_SERVER`.
- An NVIDIA GPU, `nvidia-smi` available on PATH, and enough free VRAM for the
  selected configuration.
- A GGUF chat model supported by your llama.cpp build. For image input, also
  provide that model's matching vision projector (`mmproj` GGUF).

The node manages its own server process. It does not download llama.cpp, invoke
an updater, or attach to a server you started separately. Its command-line
options must be supported by your installed llama.cpp build.

## Install

Copy this folder into `ComfyUI/custom_nodes/` and restart ComfyUI. The node is
**LLM Session (llama.cpp)** (class `ComfyLLMSession`) in the `LLM Session`
category.

The installed layout should be:

```text
ComfyUI/custom_nodes/comfyui-llm-session/
    __init__.py
    session_node.py
    llama_server.py
    model_images.py
    vram_guard.py
    web/
```

If your server is elsewhere, set its path in the script that launches ComfyUI,
before the Python launch command. No global environment configuration is needed:

```sh
export COMFYUI_LLM_SESSION_SERVER="/path/to/llama-server"
# Your existing ComfyUI launch command follows here.
```

The current code checks this variable, then the default path above; it does not
search PATH for the server executable.

## First run

1. Put a GGUF chat model in `ComfyUI/models/LLM` (or configure an existing model
   directory as described below). Add the matching mmproj if using images.
2. Restart ComfyUI and add **LLM Session (llama.cpp)**.
3. Choose the GPU and model. Check the automatically selected projector; choose
   `(none)` for text-only use. Matching by filename is a convenience, not a
   compatibility guarantee.
4. Enter a prompt. For general chat, replace or clear the image-description
   system prompt. Optionally connect an image.
5. Connect the `text` output to a text display or downstream prompt input and
   queue the workflow. The first run starts the server and loads the model.
6. Change the prompt or seed and queue again to continue. Unchanged inputs may
   return ComfyUI's cached result without executing the node.

Use **Clear conversation** to start over without unloading the model. If an
image-generation node needs the same GPU afterwards, consider
`server_lifetime=unload_after_generation`; the next LLM run will reload it.

## Model folders

GGUF files are scanned in ComfyUI folder order, first match wins:

1. the `LLM` folder (`<ComfyUI>/models/LLM`, or your `extra_model_paths.yaml`
   mapping). The folder key is registered automatically if it does not exist.
2. the `text_encoders` folder - which in ComfyUI also covers `<ComfyUI>/models/clip`
   and whatever `extra_model_paths.yaml` maps to those keys.

Add more places without touching the code (folder keys or absolute paths,
comma-separated): `COMFYUI_LLM_SESSION_MODEL_FOLDERS="LLM,text_encoders,/mnt/other/ggufs"`.
Only `.gguf` files are offered, up to 4 folder levels deep; names from
subfolders are shown as `subfolder/name.gguf`. A file whose name contains
`mmproj` is offered as the **vision projector**, everything else as the
**model**; split shards (`name-00001-of-00002.gguf`) are offered by their first
shard only, and a file reachable through several folders is listed once. Text
encoders that are not chat models (for example `t5-v1_1-xxl-encoder-Q8_0.gguf`)
are listed too - llama.cpp will load them, but they are not useful here. The
folders are re-scanned on every node-info refresh, so a UI refresh is enough
after adding a file.

## Inputs

| Input | Default | Notes |
| --- | --- | --- |
| `model` | first GGUF | GGUF language model from the ComfyUI `LLM` folders (only `.gguf` is offered, because llama.cpp loads GGUF). Split GGUFs are offered by their first shard. |
| `vision_projector` | matched from the model | The `mmproj` whose name best fits the model (see “Vision projector selection”). `(none)` starts the server without a projector (text-only models, or a VL model used for text). |
| `prompt` | empty | Must not be blank. |
| `system_prompt` | `Describe only what is visible.` | May be blank. Changing it starts a new conversation. |
| `gpu` | the first device in the list | A dropdown of the GPUs llama.cpp can see, by name (from `llama-server --list-devices`). The llama.cpp index is kept behind the scenes. Does not change ComfyUI's GPU. |
| `context_size` | `4096` | 2048-32768, step 256. Raise it to keep more turns. |
| `max_tokens` | `512` | 1-4096 per run. |
| `max_image_edge` | `1024` | 128-4096, step 64. Images are resized/compressed for transport. |
| `temperature` | `0.55` | 0-2. |
| `seed` | `37` | Supports ComfyUI's seed controls. |
| `reset_session` | `false` | On: every run is a new conversation. |
| `mtp_draft_tokens` | `2` | 1-8. MTP speculative decoding; needs a model with MTP/NextN layers. |
| `server_lifetime` | `keep_loaded` | `unload_after_generation` stops the owned server before downstream nodes run; the next LLM execution reloads it. |
| `gpu_layers` | `99` | GPU layer count. `-1` = llama.cpp auto, `-2` = all, `0` = no model layers on GPU. |
| `cpu_moe_layers` | `0` | Keep the first N MoE expert layers on CPU; zero leaves the llama.cpp default. |
| `free_vram` | `false` | On every execution, ask ComfyUI to unload its own models from the selected GPU before LLM inference. Other processes are not touched. |
| `image` .. `image8` | unconnected | Optional. One more slot appears when the previous one is connected. The first frame of each connected batch is used. |

Output: `text` (`STRING`).

What the node shows (deliberately little):

- **Two status lines** under the buttons -

  ```
  122.7 tok/s · turn 3 · ctx 1.4k/4k · +512 tok · cache 3.6k/4k · pre 0.4s · img 2
  server up 12 min · pid 119332 · :49751
  ```

  The first is this node's **last run**: generation speed, turn count (`(new)` when
  the conversation had just reset), how much of the context is used, tokens
  generated, how much of the prompt llama.cpp reused from its prompt cache, prefill
  time, images sent. The second is the **server process**: uptime, PID and port
  (`server: not running - starts on the next run` when there is none, `health ...`
  / `exited` when something is wrong). Nothing that is already a row on the node -
  GPU, context size, model - is repeated there. While generating it says
  `generating...`; after a failed run the error text; before any run in this
  browser session, `no run yet`. The current implementation uses DOM status
  rows. Legacy canvas rendering has known layout problems; the previous canvas
  renderer remains commented out and is not active. The full breakdown
  (prompt/cache token split, prefill and generation times, model load time,
  llama.cpp build, device placement, notes) is in the **Logs** tab. The status
  text is informational, not an editable input.
- **`Clear conversation`** - forgets this node's conversation; the server and its
  loaded model keep running. Busy requests answer with a "busy" message.
- **`Server log`** - opens the bottom-panel tab **LLM Server log**: health, PID,
  URL, uptime, model load time, context size, model/projector, GPU, MTP, llama.cpp
  build, device placement, and the tail of the llama.cpp server's own output
  (polled only while the tab is open). llama.cpp output never goes into ComfyUI's
  own log; if the server exits, ComfyUI's log gets one line saying so. If you
  turned that tab off in settings (see **Settings**), the button instead tells you
  the log file path and where the setting lives.
- Per-run summaries (turn count, prompt/cached tokens, prefill and generation
  seconds, speed) are printed to ComfyUI's **Logs** tab once per run.

How the **LLM Server log** tab behaves (it is live, but llama.cpp is chatty only
while working):

- On open it shows up to the last **800 lines** of the current server's log. The
  file (`/tmp/comfyui-llm-session-server.log`) is truncated when a server starts,
  so that is that server's whole history - currently ~270 startup lines plus
  ~10-12 lines per request.
- While the tab is on screen it polls every **1.5 s** and appends only new lines
  (incremental `since=` request, not a re-read). Switching tabs or closing the
  bottom panel issues no requests at all.
- Between requests llama.cpp writes nothing, so the tab looks frozen until you run
  the node. The header says so: `last log write: 43 s ago` plus
  `the server is idle - llama.cpp only writes while it is processing a request`.
- `Refresh` re-reads the whole tail, `Open log file` prints the path (raw stream:
  `tail -f /tmp/comfyui-llm-session-server.log`).
- The log body is bounded inside the bottom panel and scrolls on its own (wheel /
  trackpad, or click it and use PgUp/PgDown). Opening the tab also pulls the panel
  back to ~30% of the window if the remembered splitter size was eating the canvas
  (drag it yourself afterwards and it stays as you left it).
- **`follow` only sticks while you are already at the bottom**: scroll up and the
  view stays put, with an `N new lines ↓` button to jump back down.
- A dropdown next to `Refresh` switches between **this server run** and the logs of
  the **previous runs** that were rotated aside, so a llama-server that died (crash,
  OOM, failed model load) can still be read after the next attempt overwrites the
  current file.

The node uses two status rows rather than the older large status panel.

## Vision projector selection

The `vision_projector` row is filled in from the model name: names are compared
after removing folders, `.gguf`, the `mmproj` marker, quantisation tags
(`q4_k_m`, `q8_0`, `bf16`, `fp8_scaled`, …) and split-shard suffixes, and the
best-scoring candidate wins (on a tie the higher-precision file).

A projector you pick by hand overrides that decision and persists through re-runs:
it is kept while its name fits the model as well as the best candidate does, so
choosing the Q8 `mmproj` instead of the F16 one sticks, including across quant-only
model changes. A pick that clearly belongs to another model (a 27B `mmproj` on a 9B
model) or a name that has disappeared from the folders is replaced the next time the
model changes or the workflow loads. `(none)` is remembered as a deliberate
text-only choice; it is also what the node falls back to when nothing matches -
text-only models such as `gemma-...-Q8_0.gguf` or `t5-v1_1-xxl-encoder-Q8_0.gguf`.

The rule lives in `model_images.select_projector()`; the frontend asks for the
answer over `/comfyui_llm_session/projector` when the node loads and when the model
changes, so there is one matcher rather than two. Workflows saved with the
`(auto)` placeholder are still resolved at run time.

## Settings

**Settings > LLM Session > Server log tab** controls whether the bottom-panel
tab is shown (`Comfy.LLMSession.ServerLogTab`, default on). Toggling applies
without a reload. This feature uses frontend internals, so behavior can depend
on your ComfyUI frontend version.

## Sessions

- **Continue** (default): the system prompt, the first image-bearing request and
  the previous exchanges are kept, and the new prompt is appended. Earlier
  instructions therefore keep influencing later answers.
- **Fresh**: `reset_session`, the **Clear conversation** button, a changed image,
  a changed system prompt, a different node or a changed server configuration
  (model, projector, GPU, context size, MTP, offload settings) all start a new
  conversation. The server remains loaded only when `server_lifetime` is `keep_loaded`.
- Clearing also invalidates ComfyUI's cached result, so the next queued run
  actually executes. Clearing during a generation returns a "busy" message.
- Conversation history and KV-cache reuse are different things: history is kept
  by this node, while prefix reuse is only reported when llama.cpp returns the
  cached-token count.
- Context overflow: the oldest exchange after the initial image-bearing turn is
  dropped and the request is retried; if nothing droppable remains, the node
  restarts with the current prompt and images. If even that does not fit, the
  node reports it. History removals are logged.

## Server ownership and limits

- The node starts one loopback-only `llama-server` (127.0.0.1, random port) with
  the selected device for model and projector, configurable GPU layers and MoE
  CPU offload, flash attention, q8 KV cache, one slot and context shift off.
- By default it is reused while the configuration is unchanged; choose
  `unload_after_generation` to release it before downstream nodes run. Stopping
  the process loses its prompt cache, though node conversation history remains.
  A missing saved GPU causes an execution error, never a fallback to another GPU.
  Before starting a server, the node checks NVIDIA free VRAM against a rough,
  padded model/projector/context estimate. With `free_vram` on, the node first
  asks ComfyUI to unload its own models on the selected device on every execution,
  even if the server is reused. It refuses an estimated shortfall either way.
  Other processes are never unloaded. This is not a reservation or an OOM guarantee; estimates can reject
  models that would fit, and memory can change after the check. For `gpu_layers`
  auto the estimate assumes full offload. A reused server is not checked again.
  Only the process this package started is ever stopped; a leftover
  process from a previous ComfyUI session is detected through its recorded PID,
  port and command line.
- **One conversation per ComfyUI process.** Two node instances do not have
  independent persistent conversations: the last one that ran owns the session,
  and the other one starts a new conversation when it runs.
- Sampling defaults `top_p=0.8`, `top_k=20`, `min_p=0.0` are fixed constants for
  now. The model-specific `enable_thinking=false` chat-template option is only
  sent when the selected model's own chat template mentions it.

Every run also prints a `timing:` line to the ComfyUI console, e.g.
`timing: model load=3.87 s prefill=0.61 s (993.42 tok/s) generation=0.42 s`, and
the startup line reports how long the server took to become healthy.

Server log: `/tmp/comfyui-llm-session-server.log`. PID record:
`/tmp/comfyui-llm-session-server.pid` (`pid`, `port`, `executable`, `owner`). The
port is picked per start, so read it there instead of hard-coding it. The same
state is available over HTTP for scripts: `GET /comfyui_llm_session/server`
returns JSON with health, PID, port, uptime, model, projector, GPU, context
size, MTP, executable, log path and the placement lines. Startup and request
failures quote the log path and the last server output.

## Display lab (experiment, not part of the session node)

`LLM Session Display Lab` (`ComfyLLMSessionDisplayLab`) is a display-only
experiment node that renders the same status information through every on-node
display channel this ComfyUI frontend offers, so one can be chosen by eye: pick
`display_mode` = `dom_block`, `dom_compact`, `dom_preview`, `readonly_row`,
`canvas_widget`, `title_pill`, `footer_strip`, `background_band`, `node_color`,
or `title_only`.

- It does not touch the llama.cpp server, conversations, or models; it only
  paints. The session node is unaffected.
- The frontend half is `web/display_lab.js`. It can also tag *any* node with the
  property `llm_session.lab_mode` (`window.comfyLLMSessionLab.tag(node, mode)`) so
  the modes can be compared without adding nodes.
- The node class only appears after a ComfyUI restart; the tagging path works on
  a running instance.

## Troubleshooting

- **Server not found / GPU unavailable:** check the executable path in the
  ComfyUI launch environment. Run that binary with `--list-devices` to check
  CUDA support. The node never silently substitutes another GPU.
- **No models listed:** check the model folders, refresh the browser, and make
  sure the files are GGUF chat models rather than diffusion text encoders.
- **Server exits at startup:** open **Server log** for the actual error. Check
  model support, projector compatibility, available VRAM and server CLI options.
- **Estimated VRAM shortfall:** reduce context or GPU offload, or free memory.
  `free_vram` only releases ComfyUI-managed models when ComfyUI's active CUDA
  device maps to the selected physical GPU; it does not stop other processes.
- **No new answer when queued:** change the prompt/seed or clear the conversation
  to invalidate ComfyUI's cached result.
- **Context full:** increase `context_size` or reduce the prompt/images. The
  node first retries by dropping old follow-up exchanges, as described above.
