# LLM Session (llama.cpp)

Prompt plus optional images in, text out. The node starts and owns its own
`llama-server` (llama.cpp) process. It uses the GPU selected in the node,
independently of the device used by ComfyUI.

## Reading the node

Two lines of text sit under the last row of the node:

```
120.8 tok/s  ·  turn 3  ·  ctx 1.4k/4.1k  ·  +512 tok  ·  cache 3.6k/4k  ·  pre 0.4s
server up 12 min  ·  pid 119332  ·  :49751
```

The first is **this node's last run** (generation speed, turn, how full the context
is, then tokens generated, prompt-cache reuse, prefill time, images sent). The
second is **the llama.cpp server process**: uptime, PID, port. If there is no
server yet it says `server: not running - starts on the next run`; a crashed one
shows `health ...`. Nothing that is already a row on the node (GPU, context size,
model) is repeated here.

They are text drawn on the node, not input rows: nothing to click, nothing to edit,
and the drawing is clipped to the node so it cannot spill outside it. A narrow node
cuts the line with `...`, keeping the most important parts first; the full
breakdown (prompt/cache token split, prefill and generation times, model load time,
llama.cpp build, device placement, notes) is in the **Logs** tab.

The text itself says what happened:

| text | meaning |
|------|---------|
| `no run yet` | never run in this browser session, or the conversation was just cleared |
| `generating...` | running right now |
| `120.8 tok/s ...` | ran fine |
| `! ...` prefix | this run degraded: the context filled up, the oldest exchange was dropped and retried |
| an error line | this run failed; the detail is in the log tabs |

Two buttons: **Clear conversation** (forget this node's conversation, keep the
server and its model loaded) and **Server log** (open the bottom-panel tab
**LLM Server log**). The output is one `STRING` named `text` - feed it to any
text node, or wire a `Preview Any` to see it.

## What continues, what starts fresh

Each node carries one conversation, keyed by the node, the images, the system
prompt and the server configuration.

* Queueing the same workflow again without changing anything: ComfyUI's cache
  skips the node, so nothing runs and the answer is reused.
* Changing only `prompt` re-runs the node and **continues** the conversation
  (turn 2, 3, ...). Earlier turns still influence the answer.
* **Clear conversation**, a changed image, a changed `system_prompt`, or a
  changed model / projector / GPU / context size / MTP setting starts a new
  conversation. The model stays loaded, so the first turn is still fast.
* `reset_session` on: every run starts fresh.
* The conversation lives inside one ComfyUI process. Two ComfyUI instances, or
  two copies of this node, do not share one.

## Server, GPU and restarts

On the first run the node starts `llama-server` on a random loopback port and
waits for `/health`. `server_lifetime` controls what happens after generation:

* `keep_loaded` (default) keeps the process for later runs; ComfyUI exit stops it.
* `unload_after_generation` stops the owned server before the node returns, so a
  downstream sampler can use the released GPU memory. The next execution reloads
  the model and cannot reuse the stopped server's prompt cache. Python-side
  conversation history is retained unless reset or changed.

Changing model, projector, GPU, context size, MTP, GPU layers or CPU MoE layers
restarts the server. Server load time is measured and reported.

`gpu` is a dropdown of the GPUs llama.cpp sees, by name (e.g. `RTX 3090`); the
llama.cpp index (`CUDA0`, `CUDA1`, ...) stays behind the scenes, and so does the
VRAM, which the server state and the log tab report. The label is what gets saved,
so it deliberately carries no live figures - a value that changes between page
loads would make the node's stored value invalid and ComfyUI would refuse to queue
the run. It is independent of ComfyUI's own device selection; selecting the same
GPU as ComfyUI is allowed. The child sees only the selected GPU. If a saved GPU
is missing, execution fails clearly without selecting a different card. If it is
too full, startup fails with llama.cpp's own message in **LLM Server log**.

`gpu_layers` sets llama.cpp's GPU layer count: `99` preserves the previous default,
`-1` asks llama.cpp for `auto`, `-2` means `all`, and `0` means no model layers
on the GPU. `cpu_moe_layers` keeps the first N MoE expert layers on CPU; `0`
leaves llama.cpp's default. These controls do not guarantee a specific VRAM
budget, and the vision projector may still use the selected GPU.

`free_vram` defaults to **off**. Turn it on to ask ComfyUI to unload its own
models on the selected GPU before every node execution, even when the server is
reused. Other processes are never unloaded.

Before starting a server, the node compares NVIDIA free VRAM with a rough,
padded estimate from the model/projector sizes and context. If the estimate is
above free memory, the node refuses to start, whether the toggle is on or off.

The estimate can reject a model that would fit and cannot prevent all OOMs:
other processes can allocate after the check, and inference may need more than
estimated. `gpu_layers=-1` is estimated as full offload to avoid silently
accepting a mostly CPU-fitted model. An already-running reused server does not
need another load and skips this preflight.

Default binary is the llama.cpp build this pack was installed with; override it
with the `COMFYUI_LLM_SESSION_SERVER` environment variable.

## Model and projector

* Only `.gguf` language models are offered, scanned recursively from the ComfyUI
  `LLM` and `text_encoders` folders (which include `models/clip`). Extra folders:
  `COMFYUI_LLM_SESSION_MODEL_FOLDERS="LLM,text_encoders,/abs/path"`.
* `vision_projector` is filled in for you: it holds the `mmproj` whose name best
  matches the selected model. Names are compared after dropping folders, `.gguf`,
  the `mmproj` marker, quantisation tags (`q4_k_m`, `q8_0`, `bf16`, `fp8_scaled`, ...)
  and split-shard suffixes; on a tie the higher-precision file wins.
* A projector you pick yourself wins. It is kept across re-runs and across model
  changes as long as its name fits the model as well as the best candidate does -
  that is how choosing the Q8 `mmproj` over the F16 one sticks. A pick that clearly
  belongs to another model (a 27B `mmproj` on a 9B model) is replaced on the next
  model change or reload, and a name that is no longer on offer is replaced too.
* `(none)` runs the model as text-only. Picking it by hand is remembered as a
  deliberate choice and stays; it is also what the node falls back to when nothing
  matches, e.g. `t5-v1_1-xxl-encoder-Q8_0.gguf`.
* **MTP speculative decoding** (`mtp_draft_tokens`) only helps models that ship
  MTP/NextN layers. Most GGUFs do not: the node then starts the server without
  `--spec-type draft-mtp` and says so in the log. That is a note, not a failure.

## Images

Up to eight optional image inputs (`image`, `image2`, ... `image8`); leave the
ones you do not use unconnected. For a batch, only the **first frame** is used.
`max_image_edge` resizes before the vision encoder - images dominate the prompt:
a 1024 px image costs roughly 700 tokens of context.

## Context and overflow

`context_size` is shared by the prompt and everything generated. When a turn no
longer fits, the node drops the oldest follow-up exchange (the first
image-bearing turn is kept when possible), retries, and only then falls back to a
fresh conversation; the status line goes amber and says what happened. If even a
fresh request does not fit, the run fails with llama.cpp's own wording, for
example:

```
request (3810 tokens) exceeds the available context size (2048 tokens), try increasing it
```

## Timing

The status line reports generation speed. The per-run breakdown - prefill
seconds and tokens/s, generation seconds, and model load seconds when the server
had to (re)start - is written to ComfyUI's own **Logs** tab and repeated in the
**LLM Server log** tab header, together with server pid, port, model, projector,
GPU, context size, MTP state and device placement.

Measured on this machine: `Qwen3.8-9B-heretic-uncensored.Q4_K_M.gguf` + its
`mmproj-f16`, context 4096, RTX 3090, **116-123 tok/s**, model load ~1.3 s.

## Server log tab

The llama.cpp server's own output stays in
`/tmp/comfyui-llm-session-server.log` (earlier runs: `.1`, `.2`, selectable in
the tab) and is read on demand while the tab is open - it never enters
ComfyUI's log or the websocket. Add or remove the tab in
**Settings - LLM Session - Server log tab** (`Comfy.LLMSession.ServerLogTab`).
With the tab off, the **Server log** button only reports the log file path.

Clearing during a generation is refused on purpose: the route answers
`409 Busy: a generation is running. Wait for it to finish, then clear again.`

## Where this text comes from

Three native help surfaces: hover the node's **title bar** for the short summary,
hover any **widget row** for that parameter, and this page in the **Info** tab
(or the Node Info button in the selection toolbox). The markdown lives in
`web/docs/ComfyLLMSession.md` of this pack.
