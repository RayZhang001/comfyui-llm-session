// Display lab: draws the same status information through every on-node display
// method ComfyUI offers in this frontend, so one can be picked by eye.
//
// Two ways to run it:
//   * the real lab node (class ComfyLLMSessionDisplayLab, needs a ComfyUI
//     restart to appear) - pick a method with its display_mode widget;
//   * any node tagged with the property "llm_session.lab_mode" (used by
//     ui_docs/spawn_lab_nodes.js so the methods can be compared without a
//     restart) - content then falls back to numbers from a real earlier run.
//
// Nothing here talks to the server; it only paints.

import { app } from "../../scripts/app.js";

const LAB_CLASS = "ComfyLLMSessionDisplayLab";
const SESSION_CLASS = "ComfyLLMSession";
const LAB_PROPERTY = "llm_session.lab_mode";
const STATS_KEY = "llm_session_stats";

const STATE_COLORS = {
    running: { accent: "#38a169", bg: "#152a1c", title: "#8fe3b0" },
    idle: { accent: "#718096", bg: "#1c2129", title: "#cbd5e0" },
    warn: { accent: "#d69e2e", bg: "#2e2513", title: "#f2cf7b" },
    error: { accent: "#e53e3e", bg: "#2f1717", title: "#f6a3a3" },
};

// From an actual run of the session node on this machine (see RESULTS.md).
const DEMO_CONTENT = {
    headline: "Qwen3.8-9B-heretic-uncensored.Q4_K_M",
    metric: "122.7 tok/s - turn 1",
    detail: "prompt 608 tok (0 prefix reuse)\ngenerated 51 tok in 0.42 s\ncontext 4,096 - GPU 1 - MTP off\nllama-server pid 48377 port 46953",
    state: "running",
    progress: 0.72,
    series: [118, 121, 124, 119, 123, 126, 122, 120],
    preview: "A crayon drawing: a figure with raised arms on a green field under a blue sky.",
};

const titleHeight = () => window.LiteGraph?.NODE_TITLE_HEIGHT ?? 30;
const isLab = (node) => node?.comfyClass === LAB_CLASS || !!node?.properties?.[LAB_PROPERTY];

function widgetValue(node, name, fallback) {
    const w = node.widgets?.find((item) => item.name === name);
    const value = w?.value;
    if (value === undefined || value === null || value === "") return fallback;
    return value;
}

function content(node) {
    // After a run the executed payload wins, because it also carries the
    // optional text input, which has no widget of its own.
    const live = node.__labData ?? {};
    const pick = (name, fallback) =>
        live[name] !== undefined && live[name] !== null ? live[name] : widgetValue(node, name, fallback);
    const out = {
        headline: String(pick("headline", DEMO_CONTENT.headline)),
        metric: String(pick("metric", DEMO_CONTENT.metric)),
        detail: String(pick("detail", DEMO_CONTENT.detail)),
        state: String(pick("state", DEMO_CONTENT.state)),
        progress: Number(pick("progress", DEMO_CONTENT.progress)),
        series: String(pick("series", DEMO_CONTENT.series.join(","))),
        preview: String(live.preview || DEMO_CONTENT.preview),
    };
    const values = out.series.split(/[,;]/).map((part) => Number(part.trim())).filter(Number.isFinite);
    out.series = values.length ? values : DEMO_CONTENT.series;
    out.lines = out.detail.split("\n").map((line) => line.trim()).filter(Boolean);
    out.colors = STATE_COLORS[out.state] ?? STATE_COLORS.idle;
    return out;
}

function mode(node) {
    return String(node.properties?.[LAB_PROPERTY] ?? widgetValue(node, "display_mode", "title_pill"));
}

// ---------- drawing helpers ----------

function roundRect(ctx, x, y, w, h, r) {
    const radius = Math.min(r, h / 2, w / 2);
    ctx.beginPath();
    if (ctx.roundRect) {
        ctx.roundRect(x, y, w, h, radius);
        return;
    }
    ctx.moveTo(x + radius, y);
    ctx.arcTo(x + w, y, x + w, y + h, radius);
    ctx.arcTo(x + w, y + h, x, y + h, radius);
    ctx.arcTo(x, y + h, x, y, radius);
    ctx.arcTo(x, y, x + w, y, radius);
    ctx.closePath();
}

function pill(ctx, x, y, text, fg, bg, size = 12) {
    ctx.font = `${size}px sans-serif`;
    const w = ctx.measureText(text).width + 12;
    const h = size + 8;
    ctx.fillStyle = bg;
    roundRect(ctx, x - w, y, w, h, h / 2);
    ctx.fill();
    ctx.fillStyle = fg;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(text, x - w / 2, y + h / 2 + 0.5);
    ctx.textAlign = "left";
    return w;
}

function bar(ctx, x, y, w, h, fraction, fg, bg) {
    ctx.fillStyle = bg;
    roundRect(ctx, x, y, w, h, h / 2);
    ctx.fill();
    ctx.fillStyle = fg;
    roundRect(ctx, x, y, Math.max(h, w * Math.min(1, Math.max(0, fraction))), h, h / 2);
    ctx.fill();
}

function sparkline(ctx, x, y, w, h, values, color) {
    if (values.length < 2) return;
    const min = Math.min(...values);
    const max = Math.max(...values);
    const span = max - min || 1;
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    values.forEach((value, index) => {
        const px = x + (index / (values.length - 1)) * w;
        const py = y + h - ((value - min) / span) * h;
        if (index === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
    });
    ctx.stroke();
    ctx.fillStyle = color;
    const last = values[values.length - 1];
    ctx.beginPath();
    ctx.arc(x + w, y + h - ((last - min) / span) * h, 2.2, 0, Math.PI * 2);
    ctx.fill();
}

function truncate(ctx, text, maxWidth) {
    if (ctx.measureText(text).width <= maxWidth) return text;
    let out = text;
    while (out.length > 4 && ctx.measureText(`${out}...`).width > maxWidth) out = out.slice(0, -1);
    return `${out}...`;
}

// ---------- teardown ----------

function disposeLab(node) {
    const lab = node.__lab;
    if (!lab) return;
    for (const name of lab.widgetNames ?? []) {
        const widget = node.widgets?.find((item) => item.name === name);
        if (widget?.element?.parentElement) widget.element.remove();
        const index = node.widgets?.findIndex((item) => item.name === name) ?? -1;
        if (index >= 0) node.widgets.splice(index, 1);
    }
    for (const [key, original] of Object.entries(lab.hooks ?? {})) node[key] = original;
    if (lab.title !== undefined) node.title = lab.title;
    if (lab.color !== undefined) node.color = lab.color;
    if (lab.bgcolor !== undefined) node.bgcolor = lab.bgcolor;
    if (lab.titleColor !== undefined) node.titleColor = lab.titleColor;
    if (lab.size) node.size = lab.size;
    node.__lab = null;
}

// The LLM Session node paints its own status panel; remove it on tagged copies
// so the compared methods are not mixed with it.
function stripSessionPanel(node) {
    if (node.comfyClass !== SESSION_CLASS) return;
    for (const name of ["llm_session_status", "Clear conversation", "Server log", "auto_match_projector"]) {
        const widget = node.widgets?.find((item) => item.name === name);
        if (!widget) continue;
        widget.element?.parentElement?.remove();
        const index = node.widgets.findIndex((item) => item.name === name);
        if (index >= 0) node.widgets.splice(index, 1);
    }
}

// ---------- the methods ----------

const RENDERERS = {
    // HTML widget in the node body - what the node does today.
    dom_block(node, c) {
        const el = document.createElement("div");
        el.style.cssText =
            "padding:8px;color:#ddd;font-size:12px;line-height:1.45;min-height:132px;max-height:132px;" +
            "overflow-y:auto;box-sizing:border-box;white-space:pre-line;";
        el.innerHTML =
            `<div style="color:${c.colors.title};font-weight:600">${c.metric}</div>` +
            `<div style="opacity:.75;margin-bottom:4px">${c.headline}</div>` +
            c.lines.map((line) => `<div>${line}</div>`).join("");
        const widget = node.addDOMWidget("lab_dom_block", "lab_dom_block", el, { serialize: false });
        widget.computeSize = (width) => [width, 140];
        node.__lab.widgetNames.push("lab_dom_block");
    },

    // Same idea, two lines.
    dom_compact(node, c) {
        const el = document.createElement("div");
        el.style.cssText =
            "padding:4px 6px;color:#ddd;font-size:12px;line-height:1.4;box-sizing:border-box;" +
            "min-height:34px;max-height:34px;overflow:hidden;white-space:nowrap;";
        el.innerHTML = `<span style="color:${c.colors.title}">${c.metric}</span> · <span style="opacity:.7">${c.headline}</span>`;
        const widget = node.addDOMWidget("lab_dom_compact", "lab_dom_compact", el, { serialize: false });
        widget.computeSize = (width) => [width, 36];
        node.__lab.widgetNames.push("lab_dom_compact");
    },

    // Preview-style block: what an output-preview looks like inside a node.
    dom_preview(node, c) {
        const el = document.createElement("div");
        el.style.cssText =
            "padding:6px;color:#cfd8dc;font:11px/1.45 monospace;background:#101317;border:1px solid #2c333c;" +
            "min-height:74px;max-height:74px;overflow-y:auto;box-sizing:border-box;white-space:pre-wrap;";
        el.textContent = c.preview || c.lines[0] || "";
        const widget = node.addDOMWidget("lab_dom_preview", "lab_dom_preview", el, { serialize: false });
        widget.computeSize = (width) => [width, 80];
        node.__lab.widgetNames.push("lab_dom_preview");
    },

    // One ordinary widget row, text only.
    readonly_row(node, c) {
        const text = `${c.metric}  |  ${c.headline}`;
        const widget = node.addWidget("string", "lab_readonly_row", text, () => {}, { serialize: false });
        widget.options.read_only = true;
        widget.callback = () => {
            widget.value = text;
        };
        node.__lab.widgetNames.push("lab_readonly_row");
    },

    // Custom widget drawn on the canvas: pill + bar + sparkline in one row.
    // Note: LiteGraph hands the widget draw a fixed NODE_WIDGET_HEIGHT, so the
    // row height used for layout is tracked here and used for the drawing too.
    canvas_widget(node, c) {
        const ROW_H = 44;
        const widget = node.addWidget("lab_canvas", "lab_canvas_widget", "", () => {}, { serialize: false });
        widget.computeSize = (width) => [width, ROW_H];
        widget.mouse = () => false;
        widget.draw = (ctx, target, width) => {
            // Same canvas-state hygiene as the pill: LiteGraph draws the rows
            // after this and inherits whatever font / align we leave behind.
            ctx.save();
            // This frontend draws node content with y = 0 at the body top, and
            // this widget is the last row, so anchor it to the bottom of the body.
            const top = target.size[1] - ROW_H;
            ctx.fillStyle = "rgba(255,255,255,0.05)";
            roundRect(ctx, 6, top + 2, width - 12, ROW_H - 6, 6);
            ctx.fill();
            ctx.strokeStyle = "rgba(255,255,255,0.14)";
            ctx.lineWidth = 1;
            roundRect(ctx, 6, top + 2, width - 12, ROW_H - 6, 6);
            ctx.stroke();
            ctx.font = "600 12px sans-serif";
            ctx.fillStyle = c.colors.title;
            ctx.textBaseline = "top";
            ctx.fillText(truncate(ctx, c.metric, width * 0.55), 12, top + 8);
            bar(ctx, 12, top + 26, width * 0.5, 7, c.progress, c.colors.accent, "rgba(255,255,255,0.12)");
            sparkline(ctx, width * 0.6, top + 20, width * 0.32, ROW_H - 26, c.series, c.colors.accent);
            ctx.restore();
        };
        node.__lab.widgetNames.push("lab_canvas_widget");
    },

    // Status pill inside the title bar. Costs no node height.
    // In this frontend the drawing origin is the body top-left, so the title
    // bar lives at negative y.
    title_pill(node, c) {
        const original = node.onDrawForeground;
        node.onDrawForeground = function (ctx) {
            original?.apply(this, arguments);
            if (this.flags?.collapsed || this.collapsed) return;
            let right = this.size[0] - 6;
            for (const button of this.title_buttons ?? []) {
                if (button.visible === false) continue;
                right -= (button.getWidth?.(ctx) ?? 0) + 2;
            }
            ctx.save();
            const top = -titleHeight() + 5;
            pill(ctx, right, top, c.metric, "#101317", c.colors.accent, 11);
            ctx.restore();
        };
        node.__lab.hooks.onDrawForeground = original;
        // Make room for the pill on the title line.
        node.__lab.title = node.title;
        const label = String(node.title ?? "");
        if (label.length > 18) node.title = `${label.slice(0, 17)}…`;
    },

    // Extra strip drawn under the widgets. Grows the node by one row.
    footer_strip(node, c) {
        const height = 58;
        node.__lab.size = [node.size[0], node.size[1]];
        node.size = [node.size[0], node.size[1] + height];
        const original = node.onDrawForeground;
        node.onDrawForeground = function (ctx) {
            original?.apply(this, arguments);
            const top = this.size[1] - height;
            ctx.save();
            ctx.fillStyle = "rgba(255,255,255,0.05)";
            roundRect(ctx, 6, top, this.size[0] - 12, height - 10, 6);
            ctx.fill();
            ctx.font = "600 12px sans-serif";
            ctx.fillStyle = c.colors.title;
            ctx.textBaseline = "top";
            ctx.fillText(truncate(ctx, c.metric, this.size[0] - 20), 12, top + 6);
            bar(ctx, 12, top + 24, this.size[0] * 0.52, 7, c.progress, c.colors.accent, "rgba(255,255,255,0.12)");
            sparkline(ctx, this.size[0] * 0.6, top + 20, this.size[0] * 0.32, 16, c.series, c.colors.accent);
            ctx.restore();
        };
        node.__lab.hooks.onDrawForeground = original;
    },

    // State as a band behind the widgets. y = 0 is the body top in this
    // frontend, so the band covers the whole body and the title bar is painted
    // through the node colours.
    background_band(node, c) {
        const original = node.onDrawBackground;
        node.onDrawBackground = function (ctx) {
            original?.apply(this, arguments);
            ctx.save();
            ctx.fillStyle = c.colors.bg;
            ctx.fillRect(0, 0, this.size[0], this.size[1]);
            ctx.fillStyle = c.colors.accent;
            ctx.fillRect(0, 0, 5, this.size[1]);
            ctx.restore();
        };
        node.__lab.hooks.onDrawBackground = original;
        node.__lab.titleColor = node.titleColor;
        node.titleColor = c.colors.title;
    },

    // State as node colour.
    node_color(node, c) {
        node.__lab.bgcolor = node.bgcolor;
        node.__lab.titleColor = node.titleColor;
        node.bgcolor = c.colors.bg;
        node.titleColor = c.colors.title;
    },

    // The title line carries everything.
    title_only(node, c) {
        node.__lab.title = node.title;
        node.title = `${c.metric} - ${c.headline}`;
    },
};

function applyLab(node) {
    disposeLab(node);
    stripSessionPanel(node);
    const current = mode(node);
    const render = RENDERERS[current];
    if (!render) return;
    node.__lab = { widgetNames: [], hooks: {} };
    try {
        render(node, content(node));
    } catch (error) {
        console.warn(`[LLM Session] display lab mode '${current}' failed:`, error);
        node.__lab = null;
        return;
    }
    node.setDirtyCanvas?.(true, true);
}

function refreshWhenInputsChange(node) {
    for (const name of ["display_mode", "headline", "metric", "detail", "state", "progress", "series"]) {
        const widget = node.widgets?.find((item) => item.name === name);
        if (!widget) continue;
        const original = widget.callback;
        widget.callback = function (...args) {
            const result = original?.apply(this, args);
            setTimeout(() => applyLab(node), 0);
            return result;
        };
    }
    const originalExecuted = node.onExecuted;
    node.onExecuted = function (detail) {
        const result = originalExecuted?.apply(this, arguments);
        const entry = detail?.[STATS_KEY];
        const payload = Array.isArray(entry) ? entry[0] : entry;
        if (payload) node.__labData = payload;
        if (payload?.lab_mode) node.properties[LAB_PROPERTY] = payload.lab_mode;
        setTimeout(() => applyLab(node), 0);
        return result;
    };
}

app.registerExtension({
    name: "comfyui-llm-session.display-lab",

    async nodeCreated(node) {
        if (!isLab(node)) return;
        refreshWhenInputsChange(node);
        // Let the session node's own extension finish first, then replace it.
        setTimeout(() => applyLab(node), 80);
    },

    async loadedGraphNode(node) {
        if (!isLab(node)) return;
        setTimeout(() => applyLab(node), 120);
    },
});

// Handle for the console helper (ui_docs/spawn_lab_nodes.js): tagging a node
// after it was created has no hook of its own, so do the apply explicitly.
window.comfyLLMSessionLab = {
    modes: Object.keys(RENDERERS),
    apply: applyLab,
    tag(node, nextMode) {
        node.properties = node.properties || {};
        node.properties[LAB_PROPERTY] = nextMode;
        applyLab(node);
    },
};
