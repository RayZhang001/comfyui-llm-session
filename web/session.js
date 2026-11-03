import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Debug: how many times this module was evaluated in the page.
window.__llmSessionLoads = (window.__llmSessionLoads ?? 0) + 1;

// Status uses DOM text in both renderers for the legacy compatibility test.
// Detailed llama.cpp output is in the "LLM Server log" bottom-panel tab.

// Keep in sync with NODE_NAME in session_node.py.
const NODE_NAME = "ComfyLLMSession";
const IMAGE_SLOTS = ["image", "image2", "image3", "image4", "image5", "image6", "image7", "image8"];
const STATS_KEY = "llm_session_stats";
const CLEAR_ROUTE = "/comfyui_llm_session/clear";
const SERVER_ROUTE = "/comfyui_llm_session/server";
const SERVER_LOG_ROUTE = `${SERVER_ROUTE}/log`;
const PROJECTOR_ROUTE = "/comfyui_llm_session/projector";
const NO_PROJECTOR = "(none)";
// Set when the user picks '(none)' by hand, so the node does not put a projector
// back on the next load. Any other projector is remembered by its own value.
const TEXT_ONLY_PIN_PROPERTY = "llm_session.text_only_pinned";
const AUTO_TOGGLE_ROW_PROPERTY = "llm_session.auto_toggle_row_removed";
// Colours an earlier version of this extension wrote onto nodes. They now live
// in saved workflows and in the browser's workflow draft, so clear them once -
// only exact matches, never a colour the user picked.
const LEGACY_COLORS = {
    bgcolor: new Set(["#1b1f26", "#152232", "#16251c", "#2a2214", "#2b1717"]),
    titleColor: new Set(["#cbd5e0", "#a9cdf5", "#8fe3b0", "#f2cf7b", "#f6a3a3"]),
};
// idle = never run in this page, busy = this node is generating right now,
// ok/warn come from the backend stats, error from execution_error.
const LOG_TAB_ID = "llm-session-server-log";
const LOG_TAB_TITLE = "LLM Server log";
const TAB_SETTING = "Comfy.LLMSession.ServerLogTab";
const LOG_POLL_MS = 1500;

// Two status rows at the bottom of the node: this node's last run, and the
// llama.cpp server process it talks to. Each is one plain widget row - LiteGraph
// reserves the height for widget rows itself, so nothing here touches node.size.
const SERVER_REFRESH_MS = 20000;

let serverInfo = { at: 0, data: null, pending: null };
const sessionNodes = new Set();

async function refreshServerInfo(force = false) {
    if (!force && serverInfo.data && Date.now() - serverInfo.at < SERVER_REFRESH_MS) return serverInfo.data;
    if (serverInfo.pending) return serverInfo.pending;
    serverInfo.pending = (async () => {
        let data = serverInfo.data;
        try {
            const response = await api.fetchApi(SERVER_ROUTE);
            data = await response.json();
        } catch {
            /* ComfyUI just restarted, or the route is from an older build */
        }
        serverInfo = { at: Date.now(), data, pending: null };
        return data;
    })();
    return serverInfo.pending;
}

function serverSegments(info) {
    if (!info) return ["server: checking..."];
    if (!info.running) {
        return info.health === "exited"
            ? [`server: exited (code ${info.exit_code})`]
            : ["server: not running - starts on the next run"];
    }
    // No device, no context size: the `gpu` and `context_size` rows above already
    // say that. What is not visible anywhere else is whether this server process
    // is alive, for how long, and how to reach it.
    const parts = [];
    if (info.health && info.health !== "ok") parts.push(`health ${info.health}`);
    if (Number.isFinite(info.uptime_seconds)) parts.push(`server up ${Math.max(1, Math.round(info.uptime_seconds / 60))} min`);
    if (Number.isFinite(info.pid)) parts.push(`pid ${info.pid}`);
    if (Number.isFinite(info.port)) parts.push(`:${info.port}`);
    return parts;
}

function serverLineText(info) {
    return serverSegments(info).join("  \u00b7  ");
}

async function refreshAllStrips(force = false) {
    const info = await refreshServerInfo(force);
    for (const node of sessionNodes) {
        node.__llm = node.__llm ?? { state: "idle", stats: null, errorText: "" };
        node.__llm.server = info;
        renderStatus(node);
    }
}

// One timer for every node, so the server line keeps showing real uptime and
// notices a server that stopped (ComfyUI restarted, or it died mid-run).
setInterval(() => {
    if (!document.hidden) refreshAllStrips();
}, SERVER_REFRESH_MS);

// ---- formatting ----

function formatNumber(value) {
    return Number.isFinite(value) ? value.toLocaleString() : "unknown";
}

function formatSpeed(value) {
    return Number.isFinite(value) ? `${value.toFixed(1)} tok/s` : "? tok/s";
}

function compactCount(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) return "?";
    if (number < 1000) return String(Math.round(number));
    if (number < 10000) return `${(number / 1000).toFixed(1).replace(/\.0$/, "")}k`;
    return `${Math.round(number / 1000)}k`;
}

function formatSeconds(value, digits = 2) {
    return Number.isFinite(value) ? `${value.toFixed(digits)} s` : "unknown";
}

function formatUptime(seconds) {
    if (!Number.isFinite(seconds)) return "unknown";
    if (seconds < 90) return `${Math.round(seconds)} s`;
    if (seconds < 5400) return `${Math.round(seconds / 60)} min`;
    return `${(seconds / 3600).toFixed(1)} h`;
}

function oneLine(text, limit = 90) {
    const value = String(text).replace(/\s+/g, " ").trim();
    return value.length > limit ? `${value.slice(0, limit - 1)}...` : value;
}

// ---- the two status lines ----
// Plain DOM text below the buttons, in both renderers.

const STATE_STYLE = {
    idle: { accent: "#718096", text: "#cbd5e0" },
    busy: { accent: "#3182ce", text: "#a9cdf5" },
    ok: { accent: "#38a169", text: "#8fe3b0" },
    warn: { accent: "#d69e2e", text: "#f2cf7b" },
    error: { accent: "#e53e3e", text: "#f6a3a3" },
};

// The most important segments come first; text wraps naturally.
function statusSegments(status) {
    const stats = status.stats;
    if (status.state === "idle") return ["no run yet"];
    if (status.state === "busy") return ["generating..."];
    // Long enough that a narrow node cuts it with "..."; the important parts are first.
    if (status.state === "error") return [oneLine(status.errorText || "run failed", 260)];
    if (!stats) return ["no run yet"];
    const parts = [];
    // Only real problems get the "!" prefix; informational notes (MTP skipped,
    // which is normal for most GGUFs) stay out of the strip.
    if (stats.note && status.state === "warn") parts.unshift(`! ${oneLine(stats.note, 40)}`);
    parts.push(formatSpeed(stats.tokens_per_second));
    parts.push(`turn ${stats.turns}${stats.continued ? "" : " (new)"}`);
    parts.push(`ctx ${compactCount(stats.context_used)}/${compactCount(stats.context_size)}`);
    if (Number.isFinite(stats.predicted_tokens)) parts.push(`+${stats.predicted_tokens} tok`);
    if (Number.isFinite(stats.prompt_tokens)) {
        const cached = Number.isFinite(stats.cached_tokens) ? stats.cached_tokens : 0;
        parts.push(cached ? `cache ${compactCount(cached)}/${compactCount(stats.prompt_tokens)}` : "no cache");
    }
    if (Number.isFinite(stats.prefill_seconds)) parts.push(`pre ${stats.prefill_seconds.toFixed(2)}s`);
    if (Number.isFinite(stats.images) && stats.images > 0) parts.push(`img ${stats.images}`);
    // The GPU is the first row of the node; repeating it here is just noise.
    return parts;
}

function statusText(status) {
    return statusSegments(status).join("  \u00b7  ");
}

function installStatusRows(node) {
    node.__llm = node.__llm ?? { state: "idle", stats: null, errorText: "", lines: [] };
    clearLegacyColors(node);
    if (node.__llm.drawInstalled) return;
    node.__llm.drawInstalled = true;
    installStatusElements(node);
    renderStatus(node);
}

// Canvas renderer backup (disabled while testing DOM widgets in legacy mode).
/*
const STATUS_LINE_HEIGHT = 18;
const STATUS_LINE_GAP = 4;
const STATUS_EDGE = 16;
const STATUS_RESERVE = STATUS_LINE_GAP + STATUS_LINE_HEIGHT * 2;
const STATUS_COLOUR = "#a7b6c8";
const STATUS_DIM_COLOUR = "#8593a5";

function statusRenderer() {
    return window.LiteGraph?.vueNodesMode ? "dom" : "canvas";
}

function installStatusDraw(node) {
    if (statusRenderer() === "dom") {
        installStatusElements(node);
        return;
    }
    // Legacy canvas renderer: the node's own size maths accounts for the two
    // lines...
    const computeSize = node.computeSize?.bind(node);
    if (computeSize) {
        node.computeSize = (...args) => {
            const size = computeSize(...args);
            if (Array.isArray(size)) size[1] += STATUS_RESERVE;
            return size;
        };
    }
    // ...and the multi-line text rows give that height back, so the node stays the
    // height its content needs and the band below the last row is real space
    // rather than a stretched node.
    capFlexibleRows(node);
    const foreground = node.onDrawForeground;
    node.onDrawForeground = function (ctx) {
        foreground?.apply(this, arguments);
        drawStatusLines(this, ctx);
    };
}
*/

const STATUS_ROW_NAMES = ["run", "server"];

function installStatusElements(node) {
    node.__llm.elements = STATUS_ROW_NAMES.map((name) => {
        const el = document.createElement("div");
        const widget = node.addDOMWidget(name, "div", el, { serialize: false });
        return { el, widget };
    });
}

/* Canvas renderer backup, continued.
function capFlexibleRows(node) {
    const flex = (node.widgets ?? []).filter(
        (widget) => typeof widget.computeLayoutSize === "function" && !widget.__llmRowCapped
    );
    if (!flex.length) return;
    const share = Math.ceil(STATUS_RESERVE / flex.length);
    for (const widget of flex) {
        const original = widget.computeLayoutSize.bind(widget);
        widget.__llmRowCapped = true;
        widget.computeLayoutSize = (host) => {
            const size = original(host);
            size.minHeight = Math.max(24, (size.minHeight ?? 24) - share);
            size.maxHeight = Math.max(size.minHeight, (size.maxHeight ?? size.minHeight) - share);
            return size;
        };
    }
    node.setDirtyCanvas?.(true, true);
}

function statusFont() {
    const LG = window.LiteGraph;
    return `12px ${LG?.NODE_FONT ?? LG?.DEFAULT_FONT ?? "sans-serif"}`;
}

function drawStatusLines(node, ctx) {
    const lines = node.__llm?.lines ?? [];
    if (!lines.length) return;
    // Directly under the last widget row, so the block sits with the content
    // instead of floating at the bottom of a node the user made taller by hand.
    let bottom = 0;
    for (const widget of node.widgets ?? []) {
        if (Number.isFinite(widget.last_y)) bottom = Math.max(bottom, widget.last_y + (widget.height ?? 20));
    }
    const box = Math.max(0, node.size[0] - STATUS_EDGE * 2);
    ctx.save();
    ctx.beginPath();
    ctx.rect(0, 0, node.size[0], node.size[1]);
    ctx.clip();
    ctx.font = statusFont();
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    lines.forEach((text, i) => {
        const y = bottom + STATUS_LINE_GAP + i * STATUS_LINE_HEIGHT + STATUS_LINE_HEIGHT / 2;
        if (y + STATUS_LINE_HEIGHT / 2 > node.size[1]) return; // no room left: skip that line
        ctx.fillStyle = i === 0 ? STATUS_COLOUR : STATUS_DIM_COLOUR;
        ctx.fillText(ellipsisToWidth(ctx, node, i, text, box), STATUS_EDGE, y);
    });
    ctx.restore();
}

function ellipsisToWidth(ctx, node, index, text, width) {
    if (!text) return text;
    // Called on every repaint of the node, and the scan below is O(character) of
    // canvas measuring, so each node remembers the answer for each line, keyed on
    // the text, width and font it last drew.
    const slots = node.__llm.crop ??= [];
    const cache = (slots[index] ??= { text: null, width: 0, font: "", out: "" });
    if (cache.text === text && cache.width === width && cache.font === ctx.font) return cache.out;
    let out = text;
    if (ctx.measureText(out).width > width) {
        while (out.length > 1 && ctx.measureText(`${out}...`).width > width) out = out.slice(0, -1);
        out = `${out}...`;
    }
    cache.text = text;
    cache.width = width;
    cache.font = ctx.font;
    cache.out = out;
    return out;
}
*/

function renderStatus(node) {
    const status = node.__llm;
    if (!status) return;
    status.lines = [statusText(status), serverLineText(status.server)];
    for (const [i, row] of (status.elements ?? []).entries()) {
        const text = status.lines[i] ?? "";
        if (row.text === text) continue;
        row.text = text;
        // Allow long unbroken paths or error messages to wrap without CSS or
        // changing their text. Normal words and spaces retain native wrapping.
        const fragment = document.createDocumentFragment();
        for (const part of text.split(/(\S{12})(?=\S)/u)) {
            if (!part) continue;
            if (fragment.lastChild && /\S$/.test(fragment.lastChild.textContent) && /^\S/.test(part)) {
                fragment.append(document.createElement("wbr"));
            }
            fragment.append(document.createTextNode(part));
        }
        row.el.replaceChildren(fragment);
    }
    node.setDirtyCanvas?.(true, false);
}

function clearLegacyColors(node) {
    // Exact match against the palette this extension used to write, so a colour
    // you picked yourself is never touched (if you happen to want one of these
    // five exact greys on this node, nudge it one shade).
    if (LEGACY_COLORS.bgcolor.has(String(node.bgcolor ?? "").toLowerCase())) node.bgcolor = null;
    if (LEGACY_COLORS.titleColor.has(String(node.titleColor ?? "").toLowerCase())) node.titleColor = undefined;
}

function setStatus(node, state, extra = {}) {
    const status = (node.__llm = node.__llm ?? { state: "idle", stats: null, errorText: "" });
    status.state = STATE_STYLE[state] ? state : "idle";
    if (extra.stats) status.stats = extra.stats;
    if (extra.errorText !== undefined) status.errorText = extra.errorText;
    if (state === "ok" || state === "warn") status.errorText = "";
    renderStatus(node);
    // A run can have (re)started the server, so its identity line is refreshed
    // too; refreshAllStrips dedupes concurrent calls.
    if (state === "ok" || state === "warn" || state === "error") refreshAllStrips(true);
}

function toast(severity, summary, detail) {
    try {
        app.extensionManager.toast.add({ severity, summary, detail, life: 6000, closable: true });
    } catch {
        /* older clients without the toast API: the console line is enough */
        console.warn(`[LLM Session] ${summary}${detail ? `: ${detail}` : ""}`);
    }
}

const LOG_TAB = {
    id: LOG_TAB_ID,
    title: LOG_TAB_TITLE,
    type: "custom",
    render: (container) => buildServerLogTab(container),
    destroy: () => logView?.stop(),
};

// Every extension's declarative `bottomPanelTabs` is registered at startup and the
// frontend has no public way to take one away again, so turning the tab off goes
// through the workspace store. One place, and if the store cannot be found the tab
// simply stays - the setting then has no effect until a frontend update fixes it.
function bottomPanelStore() {
    for (const el of [document.getElementById("vue-app"), document.body]) {
        const pinia =
            el?.__vue_app__?._context?.config?.globalProperties?.$pinia ??
            el?.__vueParentComponent?.appContext?.config?.globalProperties?.$pinia;
        const store = pinia?._s?.get("bottomPanel");
        if (store?.panels?.terminal) return store;
    }
    return null;
}

function logTabRegistered() {
    return bottomPanelStore()?.panels.terminal.tabs.some((tab) => tab.id === LOG_TAB_ID) ?? false;
}

// The declarative registration is what puts the tab in the store; we only ever
// put it back after we took it away ourselves. Without this guard the load-time
// check can run before the declarative registration lands and add a second copy.
let logTabDropped = false;

function applyTabSetting(enabled) {
    const store = bottomPanelStore();
    if (!store) return false;
    const panel = store.panels.terminal;
    const present = panel.tabs.some((tab) => tab.id === LOG_TAB_ID);
    if (enabled) {
        if (logTabDropped && !present) panel.tabs = [...panel.tabs, LOG_TAB];
        logTabDropped = false;
        return true;
    }
    if (!present) {
        logTabDropped = true;
        return true;
    }
    logView?.stop();
    panel.tabs = panel.tabs.filter((tab) => tab.id !== LOG_TAB_ID);
    logTabDropped = true;
    // Without this the panel keeps an id that no longer exists.
    if (store.activeBottomPanelTabId === LOG_TAB_ID) store.setActiveTab?.(panel.tabs[0]?.id ?? "");
    return true;
}

async function openServerLogTab() {
    if (!logTabRegistered()) {
        toast(
            "info",
            "The server log tab is turned off",
            `Turn it on under Settings > LLM Session > Server log tab. Log file: ${await logPath()}`
        );
        return false;
    }
    const command = `Workspace.ToggleBottomPanelTab.${LOG_TAB_ID}`;
    try {
        app.extensionManager.command.execute(command);
        // The tab may already be mounted (then its render hook does not run again).
        setTimeout(armPanelClamp, 250);
        return true;
    } catch (error) {
        toast("warning", "Could not open the server log tab", `${error.message}. File: ${await logPath()}`);
        return false;
    }
}

async function logPath() {
    try {
        const response = await api.fetchApi(SERVER_ROUTE);
        return (await response.json())?.log_path ?? "unknown";
    } catch {
        return "unknown";
    }
}

// ---- "LLM Server log" bottom-panel tab ----

let logView = null;

// ComfyUI's bottom panel is a splitter the user can drag to any size, and a tab
// that fills 60-99% of the window hides the canvas. When our log opens, pull the
// panel back to a usable strip - by writing the same inline flex sizes the
// splitter itself uses, and only on the closed -> open transition, so a later
// drag by the user is never fought.
const PANEL_BLOCK_FRACTION = 0.45; // only step in when the panel really dominates
const PANEL_TARGET_FRACTION = 0.34;
let clampWhenShown = false;
let panelWatcher = null;

// Small handle for debugging from the console (and for automated checks).
window.comfyLLMSession = {
    loads: () => window.__llmSessionLoads ?? 0,
    tabSetting: TAB_SETTING,
    tabRegistered: logTabRegistered,
    openLog: openServerLogTab,
    clampPanel: clampBottomPanel,
    armClamp: armPanelClamp,
    setStatus: (node, state, extra = {}) => setStatus(node, state, extra),
    stripState: (node) => ({
        lines: node.__llm?.lines ?? null,
        nodeSize: node.size.map((v) => Math.round(v)),
        computed: node.computeSize?.().map((v) => Math.round(v)) ?? null,
        widgets: (node.widgets ?? []).map((w) => `${w.name}:${w.type}`),
    }),
    panelFraction: () => {
        const panel = document.querySelector("#bottom-panel");
        const canvasPanel = document.querySelector("#graph-canvas-panel");
        const ph = panel?.getBoundingClientRect().height ?? 0;
        const total = ph + (canvasPanel?.getBoundingClientRect().height ?? 0);
        return total ? +(ph / total).toFixed(3) : 0;
    },
};

// The header is shown in full, so the panel has to be tall enough for it plus a
// readable chunk of log.
const PANEL_MIN_LOG_HEIGHT = 260;

function panelSizeFor(total) {
    const chrome =
        (document.getElementById("llm-session-log-bar")?.getBoundingClientRect().height ?? 36) +
        (document.getElementById("llm-session-log-header")?.getBoundingClientRect().height ?? 170);
    const wanted = chrome + PANEL_MIN_LOG_HEIGHT;
    const fraction = Math.min(PANEL_BLOCK_FRACTION, Math.max(PANEL_TARGET_FRACTION, wanted / total));
    return Math.round(fraction * 100);
}

function clampBottomPanel() {
    const panel = document.querySelector("#bottom-panel");
    const canvasPanel = document.querySelector("#graph-canvas-panel");
    if (!panel || !canvasPanel) return;
    const panelHeight = panel.getBoundingClientRect().height;
    const total = panelHeight + canvasPanel.getBoundingClientRect().height;
    if (!panelHeight || !total) return; // panel closed or tiny: nothing to fix
    if (panelHeight / total <= PANEL_BLOCK_FRACTION) return;
    const panelSize = panelSizeFor(total);
    const canvasSize = 100 - panelSize;
    canvasPanel.style.flex = `${canvasSize} 1 0px`;
    panel.style.flex = `${panelSize} 1 0px`;
    const remember = () => {
        try {
            localStorage.setItem("bottom-panel-splitter", JSON.stringify([canvasSize, panelSize]));
        } catch (error) {
            /* private mode: the size just is not remembered */
        }
    };
    remember();
    // The splitter persists its own (old) sizes right after, so write it again
    // once it has settled - otherwise the huge size comes back on the next load.
    setTimeout(remember, 400);
    setTimeout(remember, 1200);
}

function armPanelClamp() {
    clampWhenShown = true;
    const panel = document.querySelector("#bottom-panel");
    if (!panel) return;
    if (panel.getBoundingClientRect().height > 0) {
        clampWhenShown = false;
        clampBottomPanel();
        return;
    }
    // At render time the panel is often not laid out yet (height 0): wait for the
    // first real size, then clamp once.
    panelWatcher ??= new ResizeObserver(() => {
        if (!clampWhenShown) return;
        if (panel.getBoundingClientRect().height > 0) {
            clampWhenShown = false;
            clampBottomPanel();
        }
    });
    panelWatcher.disconnect();
    panelWatcher.observe(panel);
    setTimeout(() => {
        if (!clampWhenShown) return;
        clampWhenShown = false;
        clampBottomPanel();
    }, 1500); // fallback in case no resize is reported
}

function buildServerLogTab(container) {
    // ComfyUI hands the custom tab a plain div with no height and no positioning
    // (and Vue may rewrite its style attribute on re-render). Without a positioned
    // box our absolutely-positioned root escapes to the page and the log grows to
    // the full window height, covering the canvas. Give that div a real box and
    // keep it that way.
    const styleBox = () => {
        container.style.setProperty("position", "relative", "important");
        container.style.setProperty("height", "100%", "important");
        container.style.setProperty("width", "100%", "important");
    };
    styleBox();
    const restyle = new MutationObserver(() => {
        if (!container.isConnected) {
            restyle.disconnect();
            return;
        }
        if (!/position:\s*relative/.test(container.getAttribute("style") ?? "")) styleBox();
    });
    restyle.observe(container, { attributes: true, attributeFilter: ["style", "class"] });

    const root = document.createElement("div");
    root.style.cssText =
        "position:absolute;inset:0;display:flex;flex-direction:column;min-height:0;box-sizing:border-box;overflow:hidden;" +
        "background:#0f1216;color:#d7dee7;font:12px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;";

    const bar = document.createElement("div");
    bar.id = "llm-session-log-bar";
    bar.style.cssText = "display:flex;gap:14px;align-items:center;padding:6px 10px;border-bottom:1px solid #262c34;flex-wrap:wrap;flex:0 0 auto;";
    const header = document.createElement("div");
    header.id = "llm-session-log-header";
    header.style.cssText = "padding:6px 10px 8px;border-bottom:1px solid #262c34;color:#a9b4c0;white-space:pre-wrap;flex:0 0 auto;";
    const body = document.createElement("pre");
    body.tabIndex = 0;
    body.style.cssText =
        "flex:1 1 auto;min-height:0;margin:0;padding:8px 10px;overflow:auto;" +
        "white-space:pre-wrap;word-break:break-word;outline:none;";

    const follow = document.createElement("label");
    follow.style.cssText = "display:flex;gap:5px;align-items:center;color:#a9b4c0;";
    const followBox = document.createElement("input");
    followBox.type = "checkbox";
    followBox.checked = true;
    follow.append(followBox, document.createTextNode("follow"));

    const refresh = document.createElement("button");
    refresh.textContent = "Refresh";
    refresh.style.cssText = "padding:3px 10px;background:#232a33;color:#d7dee7;border:1px solid #39424e;border-radius:4px;cursor:pointer;";
    const reveal = document.createElement("button");
    reveal.textContent = "Open log file";
    reveal.style.cssText = refresh.style.cssText;

    // Shown when lines arrived while the user had scrolled up.
    const jump = document.createElement("button");
    jump.style.cssText = "display:none;padding:3px 10px;background:#1f2c22;color:#8fe3b0;border:1px solid #38a169;border-radius:4px;cursor:pointer;";

    // "this run" or one of the rotated logs of earlier servers - a llama-server
    // that dies at startup is exactly the case where the old log matters.
    const whichSelect = document.createElement("select");
    whichSelect.style.cssText = "padding:3px 6px;background:#232a33;color:#d7dee7;border:1px solid #39424e;border-radius:4px;";
    let which = "current";
    let renderedOptions = "";

    // Scrolling up must not be undone by the next poll.
    let pinned = true;
    const nearBottom = () => body.scrollHeight - body.scrollTop - body.clientHeight < 24;
    body.addEventListener("scroll", () => {
        pinned = nearBottom();
        if (pinned) jump.style.display = "none";
    });

    bar.append(follow, refresh, whichSelect, reveal, jump, document.createElement("span"));
    root.append(bar, header, body);
    container.append(root);
    armPanelClamp();

    let generation = "";
    let seen = -1;
    let lines = [];
    let path = "";
    let legacy = false;

    // Rows are appended as elements rather than rewriting the whole <pre>: that
    // keeps the scroll position and any text selection stable while polling.
    const appendRows = (items) => {
        const frag = document.createDocumentFragment();
        for (const line of items) {
            const row = document.createElement("div");
            row.textContent = line;
            frag.append(row);
        }
        body.append(frag);
    };

    const paintHeader = (status, note) => {
        if (status.health === "not started") {
            header.textContent = status.message ?? "No llama-server has been started by this ComfyUI process yet.";
            return;
        }
        const bits = [
            [
                `state: ${status.health}${Number.isInteger(status.exit_code) ? ` (code ${status.exit_code})` : ""}`,
                `pid ${status.pid}`,
                status.url,
                `up ${formatUptime(status.uptime_seconds)}`,
            ].join("  ·  "),
            `model ${status.model}${status.projector && status.projector !== "none" ? ` + ${status.projector}` : ""}`,
            `GPU ${status.gpu}  ·  ctx ${formatNumber(status.context_size)}  ·  MTP ${
                Number.isFinite(status.mtp_draft_tokens) && status.mtp_draft_tokens > 0 ? `${status.mtp_draft_tokens} drafts` : "off"
            }  ·  model load ${formatSeconds(status.load_seconds)}`,
            `${status.version ?? "version unknown"}`,   // the build line already names the binary's build
            `executable ${status.executable}`,
        ];
        // The device facts worth seeing without scrolling; the rest of the
        // placement output is in the log body below.
        const placement = (status.placement ?? [])
            .filter((line) => /using device|offloaded \d+\/\d+ layers|CUDA0 model buffer size/i.test(line))
            .slice(0, 3);
        const writeAge = Number.isFinite(status.log_age_seconds)
            ? `last log write: ${Math.round(status.log_age_seconds)} s ago${
                  status.log_age_seconds > 30 && status.health === "ok"
                      ? "  ·  the server is idle - llama.cpp only writes while it is processing a request"
                      : ""
              }`
            : "";
        header.textContent =
            bits.join("\n") +
            (placement.length ? `\ndevice:\n${placement.map((line) => `  ${line.trim()}`).join("\n")}` : "") +
            (writeAge ? `\n${writeAge}` : "") +
            (note ? `\n\n${note}` : "");
    };

    const pullHeaderOnly = async () => {
        // Works on a backend that predates the log route, so the tab still tells
        // something useful instead of only an HTTP status.
        try {
            const status = await api.fetchApi(SERVER_ROUTE).then((response) => response.json());
            path = status.log_path ?? path;
            paintHeader(status, "This ComfyUI process was started before the LLM Server log route existed. Restart ComfyUI to read the log here.");
            if (!lines.length) body.textContent = `server log file: ${status.log_path ?? "unknown"}`;
        } catch (error) {
            header.textContent = `Could not read the server state: ${error.message}`;
        }
    };

    const pull = async () => {
        if (!container.isConnected) return; // tab is not on screen: no requests
        if (legacy) {
            await pullHeaderOnly();
            return;
        }
        try {
            const response = await api.fetchApi(`${SERVER_LOG_ROUTE}?tail=800&since=${seen}&which=${encodeURIComponent(which)}`);
            const data = await response.json().catch(() => ({}));
            if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
            path = data.log_path ?? path;
            const options = [{ value: "current", label: "this server run" }].concat(
                (data.previous_logs ?? []).map((entry, index) => ({
                    value: String(index + 1),
                    label: `previous run - ${entry.modified} - ${entry.lines} lines`,
                }))
            );
            if (JSON.stringify(options) !== renderedOptions) {
                renderedOptions = JSON.stringify(options);
                whichSelect.replaceChildren(
                    ...options.map((option) => {
                        const element = document.createElement("option");
                        element.value = option.value;
                        element.textContent = option.label;
                        return element;
                    })
                );
            }
            if (!options.some((option) => option.value === which)) which = "current";
            whichSelect.value = which;
            paintHeader(data);
            const incoming = data.lines ?? [];
            const incremental = data.generation === generation && data.absolute !== false && seen >= 0;
            if (incremental) lines.push(...incoming);
            else lines = [...incoming];
            generation = data.generation;
            if (Number.isFinite(data.total)) seen = data.total;

            let trimmed = false;
            if (lines.length > 4000) {
                lines = lines.slice(-3000);
                trimmed = true;
            }
            const wasPinned = pinned;
            if (!incremental || trimmed) {
                body.textContent = "";
                appendRows(lines);
                if (!lines.length) body.textContent = data.health === "not started" ? "" : "(no output yet)";
            } else if (incoming.length) {
                appendRows(incoming);
            }

            if (followBox.checked && wasPinned) {
                body.scrollTop = body.scrollHeight;
                jump.style.display = "none";
            } else if (incremental && incoming.length) {
                jump.textContent = `${incoming.length} new line${incoming.length === 1 ? "" : "s"} \u2193`;
                jump.style.display = "";
            }
        } catch (error) {
            if (/404/.test(error.message)) {
                legacy = true;
                await pullHeaderOnly();
                return;
            }
            header.textContent = `Could not read the server log: ${error.message}`;
        }
    };

    refresh.addEventListener("click", () => {
        seen = -1;
        legacy = false;
        generation = "";
        lines = [];
        pinned = true;
        jump.style.display = "none";
        pull();
    });
    jump.addEventListener("click", () => {
        pinned = true;
        jump.style.display = "none";
        body.scrollTop = body.scrollHeight;
    });
    whichSelect.addEventListener("change", () => {
        which = whichSelect.value;
        seen = -1;
        generation = "";
        lines = [];
        pinned = true;
        jump.style.display = "none";
        body.textContent = "";
        pull();
    });
    followBox.addEventListener("change", () => {
        if (followBox.checked) {
            pinned = true;
            body.scrollTop = body.scrollHeight;
        }
    });
    reveal.addEventListener("click", async () => {
        const target = path || (await logPath());
        toast("info", "Server log file", target);
    });

    const timer = setInterval(pull, LOG_POLL_MS);
    pull();
    logView = {
        stop() {
            clearInterval(timer);
            restyle.disconnect();
            logView = null;
        },
    };
}

app.registerExtension({
    name: "comfyui-llm-session.node-ui",

    settings: [
        {
            id: TAB_SETTING,
            name: "Show the LLM Server log tab",
            tooltip:
                "Adds the 'LLM Server log' tab to the bottom panel, where the llama.cpp server's own output is read on demand. Off removes the tab; the node's 'Server log' button then only reports the log file path.",
            category: ["LLM Session", "Server log tab"],
            type: "boolean",
            defaultValue: true,
            onChange: (value) => {
                if (!applyTabSetting(value !== false)) {
                    toast("warning", "Could not change the tab", "Reload the page to apply the setting.");
                }
            },
        },
    ],

    bottomPanelTabs: [LOG_TAB],

    async nodeCreated(node) {
        if (node.constructor?.comfyClass !== NODE_NAME && node.comfyClass !== NODE_NAME) return;
        sessionNodes.add(node);

        const repaint = () => node.setDirtyCanvas?.(true, true);

        const replayStats = () => {
            // After a reload, or when the run came from ComfyUI's node cache,
            // the stored output is the only source for the pill.
            try {
                const saved = app.nodeOutputs?.[String(node.id)]?.[STATS_KEY];
                const entry = Array.isArray(saved) ? saved[0] : saved;
                if (entry) setStatus(node, entry.state === "warn" ? "warn" : "ok", { stats: entry });
            } catch {
                /* nodeOutputs layout differs between client versions; ignore. */
            }
        };

        // ---- optional image sockets: show one more slot once one is used ----
        let pending = false;
        const syncImageSlots = () => {
            pending = false;
            const connected = IMAGE_SLOTS.map((name) => node.inputs?.some((input) => input.name === name && input.link != null));
            const lastConnected = connected.lastIndexOf(true);
            const visible = Math.min(IMAGE_SLOTS.length, Math.max(1, lastConnected + 2));
            for (let i = (node.inputs?.length ?? 0) - 1; i >= 0; i--) {
                const index = IMAGE_SLOTS.indexOf(node.inputs[i].name);
                if (index >= visible && node.inputs[i].link == null) node.removeInput(i);
            }
            for (let i = 0; i < visible; i++) {
                if (!node.inputs?.some((input) => input.name === IMAGE_SLOTS[i])) node.addInput(IMAGE_SLOTS[i], "IMAGE");
            }
            node.graph?.setDirtyCanvas?.(true, true);
        };
        const scheduleImageSlots = () => {
            if (pending) return;
            pending = true;
            setTimeout(syncImageSlots, 0);
        };
        const onConnectionsChange = node.onConnectionsChange;
        node.onConnectionsChange = function (...args) {
            const result = onConnectionsChange?.apply(this, args);
            scheduleImageSlots();
            return result;
        };
        const onConfigure = node.onConfigure;
        node.onConfigure = function (...args) {
            const result = onConfigure?.apply(this, args);
            scheduleImageSlots();
            trimHeight();
            replayStats();
            return result;
        };

        // ---- projector selection ----
        // The rule itself lives in model_images.select_projector(); this only asks
        // for the answer and writes it into the row, so there is one matcher, not
        // two. Effect: the row always shows a real mmproj file picked from the
        // model name, and a projector you chose yourself survives re-runs.
        let projectorAsk = 0;
        const askProjector = async () => {
            const widget = node.widgets?.find((w) => w.name === "vision_projector");
            const model = node.widgets?.find((w) => w.name === "model")?.value;
            if (!widget || !model) return;
            const ask = ++projectorAsk;
            const pinned = node.properties?.[TEXT_ONLY_PIN_PROPERTY] === true;
            let decision;
            try {
                const response = await api.fetchApi(
                    `${PROJECTOR_ROUTE}?model=${encodeURIComponent(String(model))}` +
                        `&current=${encodeURIComponent(String(widget.value ?? ""))}` +
                        `&text_only_pinned=${pinned ? "1" : "0"}`
                );
                decision = await response.json();
            } catch {
                return; // route missing or server busy: leave the row alone
            }
            if (ask !== projectorAsk || !decision?.selected) return;
            const choices = decision.choices ?? widget.options?.values ?? [];
            if (choices.length && !choices.includes(decision.selected)) return;
            if (widget.value === decision.selected) return;
            if (widget.options && choices.length) widget.options.values = choices;
            widget.value = decision.selected;
            node.setDirtyCanvas?.(true, true);
        };
        for (const [name, onChange] of [
            ["vision_projector", (value) => {
                // A projector picked by hand is a deliberate choice. '(none)' needs
                // a flag of its own because it is also the matcher's fallback.
                node.properties = node.properties || {};
                node.properties[TEXT_ONLY_PIN_PROPERTY] = String(value) === NO_PROJECTOR;
            }],
            ["model", () => askProjector()],
        ]) {
            const widget = node.widgets?.find((w) => w.name === name);
            if (!widget) continue;
            const previous = widget.callback;
            widget.callback = function (...args) {
                const result = previous?.apply(this, args);
                onChange(args[0] ?? this.value);
                node.setDirtyCanvas?.(true, true);
                return result;
            };
        }
        // ComfyUI applies the saved node colour at the end of configure(), after
        // onConfigure has run, so the stale colour from an older version has to
        // be dropped here instead.
        // Extra (non-serialized) widgets are appended after the backend ones:
        // ComfyUI writes saved widget values by position, so anything inserted
        // between backend rows would shift every value after it.
        let afterConfigure = () => {};
        const configure = node.configure;
        node.configure = function (...args) {
            const result = configure.apply(this, args);
            clearLegacyColors(node);
            afterConfigure();
            return result;
        };
        scheduleImageSlots();
        setTimeout(trimHeight, 400);

        // Workflows saved from earlier builds of this pack carry a node height that
        // no longer matches the node's own content (the old status panel, the old
        // status rows). Bring the height back to what the node's layout maths asks
        // for, which includes the band the two status lines need. Per load rather
        // than written into the workflow: the node's rows no longer stretch to fill
        // extra height, so a taller box would only be empty space.
        function trimHeight() {
            const computed = node.computeSize?.();
            if (computed && node.size[1] > computed[1] + 12) {
                node.setSize?.([node.size[0], computed[1]]);
                node.graph?.setDirtyCanvas?.(true, true);
            }
        }


        // An earlier build put a projector toggle row here. Take that row of height
        // back once per node - and only when the saved height is out by about one
        // row, so a node the user sized by hand keeps its size.
        afterConfigure = () => {
            node.properties = node.properties || {};
            if (!node.properties[AUTO_TOGGLE_ROW_PROPERTY]) {
                node.properties[AUTO_TOGGLE_ROW_PROPERTY] = 1;
                const needed = node.computeSize?.()?.[1];
                if (needed) {
                    const lg = window.LiteGraph;
                    const row = (lg?.NODE_WIDGET_HEIGHT ?? 20) + (lg?.NODE_ROW_HEIGHT ?? 6);
                    const gap = node.size[1] - needed;
                    if (gap > row * 0.6 && gap < row * 2) {
                        node.size[1] -= row;
                        node.setDirtyCanvas?.(true, true);
                    }
                }
            }
            // Keep a missing saved GPU visible and queueable. Do not choose a
            // different card: the backend will report an unavailable selection.
            const gpuWidget = node.widgets?.find((w) => w.name === "gpu");
            const gpuLabels = gpuWidget?.options?.values ?? [];
            if (gpuWidget && Array.isArray(gpuLabels) && !gpuLabels.includes(String(gpuWidget.value))) {
                const saved = String(gpuWidget.value ?? "").trim();
                const oldName = saved.split(" - ")[0].trim();
                if (gpuLabels.includes(oldName)) {
                    gpuWidget.value = oldName; // old free-memory label, same card
                } else if (saved) {
                    gpuWidget.options.values = [...gpuLabels, saved];
                    gpuWidget.value = saved;
                }
                node.setDirtyCanvas?.(true, true);
            }
            // Older workflows saved non-backend button slots at these positions.
            // If those slots were null, use the new inputs' defaults.
            for (const [name, fallback, valid] of [
                ["server_lifetime", "keep_loaded", (v) => v === "keep_loaded" || v === "unload_after_generation"],
                ["gpu_layers", 99, (v) => Number.isInteger(v) && v >= -2 && v <= 999],
                ["cpu_moe_layers", 0, (v) => Number.isInteger(v) && v >= 0 && v <= 999],
                ["free_vram", false, (v) => typeof v === "boolean"],
            ]) {
                const widget = node.widgets?.find((w) => w.name === name);
                if (widget && !valid(widget.value)) widget.value = fallback;
            }
            // Runs after ComfyUI has written the saved widget values by position,
            // so the projector row is checked against the model that is really set.
            // An older build of this pack put its status lines in widget rows (a DOM
            // strip, then plain text widgets). configure() re-creates those from an
            // older draft, and they would show as empty editable rows: drop them,
            // but keep the ones this version installed.
            const ours = new Set((node.__llm?.elements ?? []).map((row) => row.widget));
            for (const stale of (node.widgets ?? []).filter(
                (w) => (w.name === "run" || w.name === "server") && !ours.has(w)
            )) {
                node.removeWidget(stale);
            }
            // Canvas renderer backup: if (statusRenderer() === "canvas") capFlexibleRows(node);
            askProjector();
        };

        // ---- two buttons, no status panel ----
        const clearButton = node.addWidget(
            "button",
            "Clear conversation",
            null,
            async () => {
                try {
                    const response = await api.fetchApi(CLEAR_ROUTE, {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ node_id: String(node.id) }),
                    });
                    const result = await response.json().catch(() => ({}));
                    if (!response.ok) throw new Error(result.error || `HTTP ${response.status}`);
                    setStatus(node, "idle");
                    toast("info", "Conversation cleared", "The model stays loaded; the next run starts a new conversation.");
                } catch (error) {
                    toast("error", "Could not clear the conversation", error.message);
                }
                repaint();
            },
            { serialize: false }
        );
        clearButton.options ??= {};
        clearButton.options.title = "Forget this node's conversation. The llama.cpp server and its loaded model stay running.";

        const logButton = node.addWidget(
            "button",
            "Server log",
            null,
            () => {
                openServerLogTab();
            },
            { serialize: false }
        );
        logButton.options ??= {};
        logButton.options.title = "Show the llama.cpp server output in the bottom panel. It never goes into ComfyUI's own log.";

        // Append the status after both buttons, including in Nodes 2.0.
        installStatusRows(node);
        refreshAllStrips();

        // ---- execution state drives the pill and the colour ----
        const onExecuted = node.onExecuted;
        node.onExecuted = function (detail) {
            const result = onExecuted?.apply(this, arguments);
            const entry = detail?.[STATS_KEY] ?? detail?.output?.[STATS_KEY];
            const stats = Array.isArray(entry) ? entry[0] : entry;
            if (stats) setStatus(this, stats.state === "warn" ? "warn" : "ok", { stats });
            return result;
        };

        // In this frontend the "executing" payload is the node id itself; older
        // builds wrap it in an object, so accept both shapes.
        const executingNodeId = (detail) =>
            detail && typeof detail === "object"
                ? detail.node ?? detail.display_node ?? detail.displayNode
                : detail;

        api.addEventListener("executing", ({ detail }) => {
            const id = executingNodeId(detail);
            if (String(id ?? "") === String(node.id)) {
                setStatus(node, "busy");
            } else if (id === null || id === undefined || id === "") {
                // Execution finished without this node having run (cached, or it
                // was skipped): do not leave it stuck on "generating...".
                if (node.__llm?.state === "busy") setStatus(node, node.__llm.stats ? "ok" : "idle");
            }
        });
        api.addEventListener("execution_error", ({ detail }) => {
            if (String(detail?.node_id ?? "") === String(node.id)) {
                setStatus(node, "error", { errorText: detail?.exception_message ?? "run failed" });
            }
        });

        // A node that was just created never went through configure(), so ask here
        // too; askProjector() does nothing when the row is already right.
        setTimeout(askProjector, 0);

        const onRemoved = node.onRemoved;
        node.onRemoved = function (...args) {
            sessionNodes.delete(this);
            return onRemoved?.apply(this, args);
        };

        // Stored node outputs can arrive slightly after configure.
        setTimeout(replayStats, 300);
    },
});

// Whether the tab exists is decided once per page load. The stored setting is the
// authority here: at extension time the settings store may still hold the default.
api.fetchApi(`/settings/${TAB_SETTING}`)
    .then((response) => response.json())
    .then((value) => applyTabSetting(value !== false))
    .catch(() => {
        /* no stored value: the tab stays, which is the default */
    });
