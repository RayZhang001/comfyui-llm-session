"""Experiment node: a display surface lab, not a feature.

Purpose: render the same information through each of the on-node display
methods ComfyUI supports, so the visual difference can be judged side by side
and one method can be picked for the real LLM Session node. It does not touch
the inference server, sessions, models or any other part of the package.

The interesting half of this node lives in ``web/display_lab.js``: this file
only supplies the content to display (headline, metric, detail lines, state,
progress, a small series) plus the mode selector.
"""

from __future__ import annotations

# Same websocket output key the real session node uses, so the frontend can
# reuse one code path for "content arrived from the server".
UI_STATS_KEY = "llm_session_stats"

MODES = [
    "dom_block",       # HTML widget in the node body, like the node has today
    "dom_compact",     # same HTML widget, two lines only
    "dom_preview",     # HTML widget styled like an output preview (monospace)
    "readonly_row",    # read-only text widget: one normal widget row
    "canvas_widget",   # custom widget drawn on the canvas (pill + bar + spark)
    "title_pill",      # drawn inside the title bar, no extra node height
    "footer_strip",    # drawn in a reserved strip below the widgets
    "background_band", # state colour as a band/tint behind the widgets
    "node_color",      # state colour on the node body and title
    "title_only",      # the node title carries the information
]


class DisplayLabNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "display_mode": (MODES, {"default": "title_pill", "tooltip": "Which on-node display method to use."}),
                "headline": ("STRING", {"default": "Qwen3.8-9B-heretic-uncensored.Q4_K_M", "tooltip": "Model / server identity line."}),
                "metric": ("STRING", {"default": "122.7 tok/s - turn 1", "tooltip": "The one number that matters."}),
                "detail": (
                    "STRING",
                    {
                        "default": "prompt 608 tok (0 prefix reuse)\ngenerated 51 tok in 0.42 s\ncontext 4,096 - GPU 1 - MTP off\nllama-server pid <pid> port <port>",
                        "multiline": True,
                        "tooltip": "Longer status lines; only some methods can show these.",
                    },
                ),
                "state": (["running", "idle", "warn", "error"], {"default": "running", "tooltip": "Server state, shown as colour."}),
                "progress": ("FLOAT", {"default": 0.72, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Context used, as a bar."}),
                "series": ("STRING", {"default": "118,121,124,119,123,126,122,120", "tooltip": "Comma separated values for the sparkline."}),
            },
            "optional": {
                "text": ("STRING", {"forceInput": True, "tooltip": "Optional real output text (e.g. from the LLM Session node) to preview."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "render"
    CATEGORY = "LLM Session"
    OUTPUT_NODE = True
    DESCRIPTION = "Displays the same status information through a selectable on-node method."
    SEARCH_ALIASES = ["display", "status", "demo", "experiment", "badge", "sparkline"]

    def render(self, display_mode, headline, metric, detail, state, progress, series, text=None, unique_id=None):
        values = [float(part) for part in series.replace(";", ",").split(",") if part.strip()]
        payload = {
            "lab_mode": display_mode,
            "headline": headline,
            "metric": metric,
            "detail": detail,
            "state": state,
            "progress": progress,
            "series": values,
            "preview": (text or "").strip()[:400],
        }
        print(f"[LLM Session] display lab node {unique_id}: mode={display_mode} state={state}", flush=True)
        body = text if text else f"{headline}\n{metric}\n{detail}"
        return (body, {"ui": {UI_STATS_KEY: [payload]}})
