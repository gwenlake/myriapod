"""Self-contained HTML page for the live swarm graph (served at ``/``).

Kept as a Python string so packaging never has to worry about data files.
Cytoscape.js (plus dagre for the layered tree layout) is loaded from a CDN —
the page runs in the user's browser, which has normal internet access; the
myriapod process itself never fetches anything. If dagre fails to load the
tree layout falls back to cytoscape's built-in breadthfirst, so a blocked
CDN degrades the layout rather than breaking the page.
"""

PAGE_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>myriapod — live</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Ccircle cx='16' cy='16' r='6' fill='%236366f1'/%3E%3Ccircle cx='6' cy='7' r='3' fill='%2310b981'/%3E%3Ccircle cx='26' cy='7' r='3' fill='%23f59e0b'/%3E%3Ccircle cx='6' cy='25' r='3' fill='%230ea5e9'/%3E%3Ccircle cx='26' cy='25' r='3' fill='%2310b981'/%3E%3C/svg%3E"/>
<script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.31.0/cytoscape.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/dagre/0.8.5/dagre.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/cytoscape-dagre@2.5.0/cytoscape-dagre.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/cytoscape-node-html-label@1.2.2/dist/cytoscape-node-html-label.min.js"></script>
<style>
  :root {
    color-scheme: dark;
    --bg:#080d18; --bg-alt:#0e1526; --bg-card:#131c2e;
    --line:#1e293b; --line-2:#334155;
    --fg:#e6edf7; --fg-dim:#9aa8bd; --fg-faint:#64748b;
    --pending:#64748b; --running:#f59e0b; --done:#10b981;
    --failed:#f43f5e; --skipped:#475569; --root:#818cf8; --dep:#38bdf8;
  }
  * { box-sizing:border-box; }
  html, body { height:100%; }
  body {
    margin:0; display:flex; flex-direction:column; overflow:hidden;
    background:var(--bg); color:var(--fg);
    font:13px/1.5 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
    -webkit-font-smoothing:antialiased;
  }

  /* ---- header ---------------------------------------------------- */
  #bar {
    flex:none; display:flex; align-items:center; gap:12px;
    padding:9px 14px; background:linear-gradient(180deg,#101a2e,#0e1526);
    border-bottom:1px solid var(--line);
    box-shadow:0 1px 0 #38bdf81f;
  }
  /* the header is instrumentation: monospace, uppercase, tabular */
  .mono-ui {
    font-family:ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    letter-spacing:.04em;
  }
  #badge {
    flex:none; font-size:11px; font-weight:700; letter-spacing:.06em;
    padding:3px 10px; border-radius:999px; white-space:nowrap;
    background:#f59e0b1f; color:var(--running); border:1px solid #f59e0b55;
  }
  #badge.live::before {
    content:""; display:inline-block; width:6px; height:6px; margin-right:6px;
    border-radius:50%; background:currentColor; vertical-align:middle;
    animation:pulse 1.4s ease-in-out infinite;
  }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.25} }
  @media (prefers-reduced-motion: reduce) { #badge.live::before { animation:none } }
  #badge.ended   { background:#10b9811f; color:var(--done); border-color:#10b98155; }
  #badge.waiting { background:#64748b1f; color:var(--fg-dim); border-color:#64748b55; }
  #badge.error   { background:#f43f5e1f; color:var(--failed); border-color:#f43f5e55; }

  /* the goal is the root node; the bar only carries instruments */
  #spacer { flex:1 1 auto; min-width:0; }
  .legend {
    flex:none; display:flex; gap:11px; color:var(--fg-faint); font-size:10px;
    text-transform:uppercase;
  }
  .legend span { white-space:nowrap; }
  .legend span::before { content:"\25CF"; margin-right:4px; font-size:9px; vertical-align:1px; }
  .lp::before{color:var(--pending)}  .lr::before{color:var(--running)}
  .ld::before{color:var(--done)}     .lf::before{color:var(--failed)}
  .ls::before{color:var(--skipped)}
  #stats {
    flex:none; font-variant-numeric:tabular-nums; color:var(--fg);
    font-size:11px; white-space:nowrap; text-transform:uppercase;
  }
  .btn {
    flex:none; background:#1a2438; color:var(--fg);
    border:1px solid var(--line-2); border-radius:7px;
    padding:4px 10px; font:inherit; font-size:11px; cursor:pointer;
    text-transform:uppercase;
    transition:background .15s, border-color .15s;
  }
  .btn:hover { background:#243350; border-color:#47597a; }
  .btn:focus-visible { outline:2px solid var(--root); outline-offset:1px; }
  @media (max-width: 980px) { .legend { display:none } }

  /* thin completion strip under the header */
  #prog { flex:none; height:2px; background:#111a2b; display:flex; }
  #prog i { display:block; height:100%; width:0; transition:width .4s ease; }
  #prog .pdone { background:var(--done); }
  #prog .prun  { background:var(--running); }
  #prog .pfail { background:var(--failed); }

  /* ---- body ------------------------------------------------------- */
  #wrap { flex:1 1 auto; min-height:0; display:flex; position:relative; }
  #cy {
    flex:1 1 auto; min-width:0; min-height:0;
    background-color:var(--bg);
    background-image:
      radial-gradient(ellipse 90% 70% at 50% 34%, #14213a 0%, rgba(8,13,24,0) 72%),
      radial-gradient(circle at 1px 1px, #1b2740 1px, transparent 0);
    background-size:100% 100%, 26px 26px;
  }
  #panel {
    flex:none; width:352px; border-left:1px solid var(--line);
    background:var(--bg-alt); overflow-y:auto; display:none;
  }
  #panel.open { display:block; }
  #panel header {
    position:sticky; top:0; z-index:1; display:flex; align-items:center; gap:8px;
    padding:12px 14px; background:var(--bg-alt);
    border-bottom:1px solid var(--line);
  }
  #panel header .id {
    font:600 12px/1 ui-monospace, SFMono-Regular, Menlo, monospace;
    padding:4px 8px; border-radius:6px; background:#1a2438; color:var(--fg);
  }
  #panel header .st {
    font-size:10px; letter-spacing:.08em; text-transform:uppercase;
    font-family:ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    color:var(--fg-dim); flex:1;
  }
  #panel header .st::before {
    content:"\25CF"; margin-right:5px; font-size:9px; vertical-align:1px;
  }
  .st.s-pending::before{color:var(--pending)}
  .st.s-in_progress::before{color:var(--running)}
  .st.s-done::before{color:var(--done)}
  .st.s-failed::before{color:var(--failed)}
  .st.s-skipped::before{color:var(--skipped)}
  #panelBody { padding:4px 14px 20px; }
  #panelBody .k {
    color:var(--fg-faint); margin-top:14px; margin-bottom:3px;
    font-size:10px; text-transform:uppercase; letter-spacing:.07em;
  }
  #panelBody .v { white-space:pre-wrap; overflow-wrap:anywhere; color:var(--fg); }
  #panelBody .v.mono { font-family:ui-monospace, SFMono-Regular, Menlo, monospace; font-size:12px; }
  #panelBody .v.err { color:#fda4af; }
  /* three little stat tiles: tokens, cost, duration */
  .tiles { display:grid; grid-template-columns:repeat(3,1fr); gap:8px; margin-top:14px; }
  .tile {
    background:var(--bg-card); border:1px solid var(--line);
    border-radius:8px; padding:8px 9px;
  }
  .tile b {
    display:block; font:600 14px/1.3 ui-monospace, SFMono-Regular, Menlo, monospace;
    color:var(--fg); font-variant-numeric:tabular-nums;
  }
  .tile span {
    display:block; margin-top:2px; font-size:10px; letter-spacing:.06em;
    text-transform:uppercase; color:var(--fg-faint);
  }
  /* ---- HTML labels inside the cards ------------------------------- */
  /* A canvas label is one colour for the whole string. These panels carry
     four different things — task, agent, tokens, brief — so each gets its
     own colour, which is what makes a card scannable rather than a wall of
     mono text. Cards only: above CARD_LIMIT tasks the nodes are dots and
     the canvas label is enough. */
  .nl {
    width:152px; max-height:52px; overflow:hidden;
    pointer-events:none; text-align:center;
    font-family:ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    transition:opacity .2s;
  }
  .nl-root { width:212px; max-height:80px; }
  .nl.dim { opacity:.12; }
  .nl-meta {
    font-size:8.5px; letter-spacing:.02em; white-space:nowrap;
    overflow:hidden; text-overflow:ellipsis; margin-bottom:3px;
  }
  .nl-meta .g { margin-right:3px; }                 /* status glyph */
  .nl-meta .id { color:#67e8f9; }                   /* task id */
  .nl-meta .md { color:#c084fc; }                   /* the agent's model */
  .nl-meta .tk { color:#94a3b8; }                   /* tokens: secondary */
  .nl-meta .sep { color:#3f5170; margin:0 3px; }
  .nl-desc {
    font-size:9px; line-height:1.25; color:#dbe4f3;
    display:-webkit-box; -webkit-line-clamp:3; -webkit-box-orient:vertical;
    overflow:hidden;
  }
  .nl-goal {
    font-size:8.5px; line-height:1.35; color:#e6edf7;
    display:-webkit-box; -webkit-line-clamp:5; -webkit-box-orient:vertical;
    overflow:hidden;
  }
  .s-pending .g { color:#94a3b8 }
  .s-in_progress .g { color:#f59e0b }
  .s-done .g { color:#10b981 }
  .s-failed .g { color:#f43f5e }
  .s-skipped .g { color:#64748b }

  .plan { list-style:none; margin:4px 0 0; padding:0; }
  .plan .pl {
    padding:5px 0; border-bottom:1px solid #16203a; font-size:12px;
    color:var(--fg-dim); line-height:1.4;
  }
  .plan .pl:last-child { border-bottom:0; }
  .plan .g { margin-right:5px; }
  .plan .id {
    font-family:ui-monospace, SFMono-Regular, Menlo, monospace;
    color:#67e8f9; margin-right:5px;
  }

  #hint {
    position:absolute; left:14px; bottom:12px; pointer-events:none;
    color:var(--fg-faint); font-size:10px; text-transform:uppercase;
    letter-spacing:.05em;
    font-family:ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  }
  #credit {
    position:absolute; right:14px; bottom:12px; pointer-events:none;
    color:#4a5a75; font-size:10px; letter-spacing:.06em;
    font-family:ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  }
  #fallback {
    display:none; margin:auto; padding:24px; max-width:460px; text-align:center;
    color:var(--fg-dim);
  }
  #fallback code { color:var(--fg); }
</style>
</head>
<body>
<div id="bar">
  <span id="badge" class="waiting mono-ui">WAITING</span>
  <span id="spacer"></span>
  <span class="legend mono-ui"><span class="lp">pending</span><span class="lr">running</span>
    <span class="ld">done</span><span class="lf">failed</span><span class="ls">skipped</span></span>
  <span id="stats" class="mono-ui"></span>
  <button id="labelBtn" class="btn mono-ui" title="show or hide task labels">labels</button>
  <button id="layoutBtn" class="btn mono-ui" title="cycle layout: tree, radial, organic">tree</button>
  <button id="fitBtn" class="btn mono-ui" title="fit the graph to the viewport">fit</button>
  <button id="pngBtn" class="btn mono-ui"
    title="download the whole graph as a high-resolution PNG">png</button>
</div>
<div id="prog"><i class="pdone"></i><i class="prun"></i><i class="pfail"></i></div>
<div id="wrap">
  <div id="cy"></div>
  <div id="fallback">
    <p><strong>cytoscape.js could not be loaded.</strong></p>
    <p>The page fetches it from <code>cdnjs.cloudflare.com</code>. Check this
       browser's internet access, then reload.</p>
  </div>
  <aside id="panel">
    <header><span class="id"></span><span class="st"></span>
      <button class="btn" id="panelClose" title="close">✕</button></header>
    <div id="panelBody"></div>
  </aside>
</div>
<div id="credit">© GWENLAKE</div>
<div id="hint">scroll to zoom · drag to pan · click a task for details · arrows: parent ▸ subtask, output ▸ consumer</div>

<script>
"use strict";
const $ = id => document.getElementById(id);

if (typeof cytoscape === "undefined") {
  $("cy").style.display = "none";
  $("fallback").style.display = "block";
  $("hint").style.display = "none";
  $("badge").className = "error";
  $("badge").textContent = "NO CYTOSCAPE";
} else {
  main();
}

function main() {

/* dagre gives a proper layered tree (straight ranks, few crossings). It is
   optional: without it the tree layout falls back to breadthfirst. */
const HAS_DAGRE = typeof cytoscapeDagre !== "undefined" && typeof dagre !== "undefined";
if (HAS_DAGRE) cytoscape.use(cytoscapeDagre);

/* Per-field colour inside a card needs real markup: a cytoscape canvas label
   is a single colour. Optional — without the extension the cards keep their
   monochrome canvas labels. Resolved after `cy` exists: the extension hangs
   off the instance, so reading it here would be a temporal-dead-zone crash
   that takes the whole page down. */
let HAS_HTML_LABEL = false;

const LABEL_LIMIT = 70;    // above this many tasks, labels are noise
const CARD_LIMIT = 60;     // above this many tasks, cards become dots

const C = { pending:"#64748b", in_progress:"#f59e0b", done:"#10b981",
            failed:"#f43f5e", skipped:"#475569", root:"#818cf8" };

const cy = cytoscape({
  container: $("cy"),
  minZoom: 0.04, maxZoom: 4,
  wheelSensitivity: 0.25,
  textureOnViewport: true,     // keeps panning smooth on a 1000-node graph
  style: [
    /* --- dots: the default, and the only mode above CARD_LIMIT tasks --- */
    { selector: "node", style: {
        "label": "data(short)", "color": "#cfe0f5", "font-size": 9,
        /* monospace + a dark chip behind the text: telemetry, not prose —
           and it stays legible over edges at any zoom */
        "font-family": "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace",
        "text-wrap": "ellipsis", "text-max-width": 132, "text-valign": "bottom",
        "text-margin-y": 7, "text-outline-color": "#060b14",
        "text-outline-width": 2, "min-zoomed-font-size": 7,
        "text-background-color": "#060b14", "text-background-opacity": 0.5,
        "text-background-shape": "roundrectangle", "text-background-padding": 3,
        "background-color": "#33415c",
        /* the border is the status channel: thick enough to read at fit-zoom */
        "border-width": 2.5, "border-color": "#7c8ca6",
        "width": "mapData(tok, 0, 20000, 22, 58)",
        "height": "mapData(tok, 0, 20000, 22, 58)",
        "underlay-color": "#000", "underlay-opacity": 0, "underlay-padding": 0,
        "transition-property":
          "background-color, border-color, underlay-opacity, underlay-padding, opacity",
        "transition-duration": "260ms" } },
    { selector: "node.nolabel", style: { "label": "" } },

    { selector: "node.in_progress", style: {
        "background-color": C.in_progress, "border-color": "#fde68a",
        /* the halo is what makes "who is working right now" readable at a glance */
        "underlay-color": C.in_progress, "underlay-opacity": 0.3,
        "underlay-padding": 12 } },
    { selector: "node.done", style: {
        "background-color": C.done, "border-color": "#34d399" } },
    { selector: "node.failed", style: {
        "background-color": C.failed, "border-color": "#fda4af",
        "underlay-color": C.failed, "underlay-opacity": 0.28,
        "underlay-padding": 9 } },
    { selector: "node.skipped", style: {
        "background-color": "#16203a", "border-style": "dashed",
        "border-color": C.skipped, "color": "#8494ab" } },
    { selector: "node.root", style: {
        "shape": "round-diamond", "background-color": C.root,
        "border-color": "#c7d2fe", "border-width": 2.5,
        "width": 54, "height": 54, "font-size": 11, "color": "#e6edf7",
        "label": "data(short)", "text-max-width": 230,
        "underlay-color": C.root, "underlay-opacity": 0.22,
        "underlay-padding": 10 } },

    /* --- cards: small graphs get readable briefs instead of dots ------- */
    { selector: "node.card", style: {
        "shape": "round-rectangle", "width": 178, "height": 64,
        "label": "data(card)", "text-valign": "center", "text-halign": "center",
        "text-wrap": "wrap", "text-max-width": 156, "font-size": 9,
        "text-background-opacity": 0,
        /* cards are read at fit-zoom, where the dot-label floor would hide
           the text entirely */
        "min-zoomed-font-size": 3, "line-height": 1.25,
        "text-outline-width": 0, "text-margin-y": 0, "color": "#dbe4f3",
        "background-color": "#131c2e", "background-opacity": 0.96,
        "border-width": 2.5 } },
    { selector: "node.card.root", style: {
        /* a whole question, not a one-line brief: smaller type, more room */
        "shape": "round-rectangle", "width": 236, "height": 94,
        "text-max-width": 212, "font-size": 8.5, "line-height": 1.3,
        "background-color": "#1b2148" } },

    /* Two channels, and status owns the loud one. The border is the status:
       it is on every node, in the saturated colour, whether the node belongs
       to a dependency cluster or not. The fill is the cluster — a flat colour
       on dots, a dark wash under the text on cards so the brief stays
       readable. Nodes outside any cluster keep the neutral fill. */
    { selector: "node.grouped", style: { "background-color": "data(gcolor)" } },
    { selector: "node.card.grouped", style: {
        "background-color": "data(gcolor)", "background-opacity": 0.2 } },
    { selector: "edge.dep.grouped", style: {
        "line-color": "data(gcolor)", "target-arrow-color": "data(gcolor)",
        "opacity": 0.85, "width": 1.7 } },

    /* Status, declared last so it always owns the border. */
    { selector: "node.pending", style: { "border-color": "#7c8ca6" } },
    { selector: "node.in_progress", style: { "border-color": C.in_progress } },
    { selector: "node.done", style: { "border-color": C.done } },
    { selector: "node.failed", style: { "border-color": C.failed } },
    { selector: "node.skipped", style: {
        "border-color": C.skipped, "border-style": "dashed" } },
    /* The root is indigo at rest (its own rule, further up) and amber while
       the planner is thinking — the one status it can be in. */
    { selector: "node.root.in_progress", style: {
        "border-color": C.in_progress, "border-width": 3,
        "underlay-color": C.in_progress, "underlay-opacity": 0.3,
        "underlay-padding": 12 } },

    /* --- edges --------------------------------------------------------- */
    /* Every edge points somewhere, and the two kinds point at different
       things: a hierarchy edge runs parent → child ("I split this into
       these"), a dependency edge runs producer → consumer ("my output is
       your input"). Without arrowheads the graph shows that tasks are
       related but not who is waiting on whom. */
    { selector: "edge", style: {
        "width": 1.3, "line-color": "#38496a", "curve-style": "bezier",
        "opacity": 0.85,
        "target-arrow-shape": "triangle", "target-arrow-color": "#4a5f85",
        "target-arrow-fill": "filled", "arrow-scale": 0.65 } },
    /* orthogonal elbows in tree mode: reads as a diagram, not a hairball */
    { selector: "edge.flow", style: {
        "curve-style": "taxi", "taxi-direction": "downward",
        "taxi-turn": "44%", "taxi-turn-min-distance": 10, "taxi-radius": 10,
        "width": 1.6, "line-color": "#3d5273",
        "target-arrow-color": "#546d99" } },
    { selector: "edge.flow.lr", style: { "taxi-direction": "rightward" } },
    { selector: "edge.dep", style: {
        "line-color": C.dep, "line-style": "dashed", "width": 1.4,
        "opacity": 0.72, "curve-style": "unbundled-bezier",
        "control-point-distances": [-28], "control-point-weights": [0.5],
        "target-arrow-shape": "triangle", "target-arrow-color": C.dep,
        /* the hand-off is the interesting direction: bigger head than the
           hierarchy's */
        "arrow-scale": 1.05 } },

    { selector: ".faded", style: { "opacity": 0.08, "text-opacity": 0 } },
    { selector: "node.hi", style: {
        "border-color": "#f8fafc", "border-width": 3.5,
        "underlay-color": "#f8fafc", "underlay-opacity": 0.18,
        "underlay-padding": 10 } },
    { selector: "edge.hi", style: { "opacity": 1, "width": 2.2 } },

    /* Last word: an HTML label is drawn over the node, so the canvas label
       has to go or the two print on top of each other. */
    { selector: "node.htmllabel", style: { "label": "" } },
  ],
});

HAS_HTML_LABEL = typeof cy.nodeHtmlLabel === "function";

const LAYOUTS = ["tree", "radial", "organic"];
/* Tree by default and it stays there: the view changing under you mid-run
   is worse than a layout that suits the shape less well. The button cycles
   tree → radial → organic; radial is the one for a wide fan-out, where a
   layered layout degenerates into a single column of dots. */
let layoutName = "tree";
let showLabels = true;
let cardMode = true;
let dirty = false, fitted = false, runId = null;
let pendingEdges = [];             // edges whose endpoints have not arrived yet

/* ---------- element construction -------------------------------------- */

const clean = s => (s || "").replace(/\s+/g, " ").trim();
const cut = (s, n) => (s.length > n ? s.slice(0, n - 1) + "…" : s);

/* One glyph per status: colour alone is lost on a projector, and a
   status-coded prefix survives greyscale, colour-blindness and a screenshot. */
const GLYPH = { pending:"·", in_progress:"◈", done:"✔", failed:"✖", skipped:"⊘" };

/* "worker[anthropic:claude-haiku-4-5]" -> "HAIKU-4.5": the agent's identity is
   the single most useful thing on a node, but only the model part of it. */
function agentTag(worker) {
  let w = clean(worker);
  if (!w) return "";
  w = w.replace(/^[^[]*\[|\]$/g, "");            // drop the "worker[...]" wrapper
  w = w.slice(w.lastIndexOf(":") + 1);           // drop the provider
  w = w.replace(/^claude-/, "").replace(/-(\d+)-(\d+)$/, "-$1.$2");
  return w.toUpperCase();
}

function tokTag(n) {
  const t = (n.itok || 0) + (n.otok || 0);
  if (!t) return "";
  return t >= 1000 ? (t / 1000).toFixed(t >= 10000 ? 0 : 1) + "K" : String(t);
}

/* T-07 · HAIKU-4.5 · 2.4K — a telemetry line, not a sentence. */
function metaLine(n) {
  const bits = [GLYPH[n.status] + " T-" + String(n.id).padStart(2, "0")];
  const a = agentTag(n.worker);
  if (a) bits.push(a);
  const t = tokTag(n);
  if (t) bits.push(t);
  return bits.join(" · ");
}

/* The root carries the question itself — its shape and colour already say
   "this is the goal", so a label saying so too is a wasted line. */
function shortLabel(n) {
  const d = clean(n.desc);
  if (n.id === "root") return cut(d, 80);
  return metaLine(n) + "  " + cut(d, 52);
}

/* Cards wrap, so they get the telemetry line above the brief. */
function cardLabel(n) {
  const d = clean(n.desc);
  if (n.id === "root") {
    const bits = ["◆ PLANNER"];
    const a = agentTag(n.worker);
    if (a) bits.push(a);
    const t = tokTag(n);
    if (t) bits.push(t);
    return bits.join(" · ") + "\n" + cut(d, 132);
  }
  return metaLine(n) + "\n" + cut(d, 76);
}

/* The card, as markup: a telemetry row (status glyph, task, agent, tokens)
   over the brief. Everything here is escaped — descriptions are model
   output. */
function registerHtmlLabels() {
  if (!HAS_HTML_LABEL) return;
  cy.nodeHtmlLabel([{
    query: "node.card.htmllabel",
    halign: "center", valign: "center",
    halignBox: "center", valignBox: "center",
    tpl(d) {
      const dim = d.dim ? " dim" : "";
      const sep = '<span class="sep">·</span>';
      if (d.id === "root") {
        const rmd = d.agent ? sep + `<span class="md">${esc(d.agent)}</span>` : "";
        const rtok = d.toktag ? sep + `<span class="tk">${esc(d.toktag)}</span>` : "";
        return `<div class="nl nl-root${dim}">` +
          `<div class="nl-meta"><span class="g">◆</span>` +
          `<span class="id">PLANNER</span>${rmd}${rtok}</div>` +
          `<div class="nl-goal">${esc(d.goal)}</div></div>`;
      }
      const tok = d.toktag ? sep + `<span class="tk">${esc(d.toktag)}</span>` : "";
      const md = d.agent ? sep + `<span class="md">${esc(d.agent)}</span>` : "";
      return `<div class="nl${dim} s-${d.status}">` +
        `<div class="nl-meta"><span class="g">${esc(d.glyph)}</span>` +
        `<span class="id">${esc(d.tag)}</span>${md}${tok}</div>` +
        `<div class="nl-desc">${esc(d.brief)}</div></div>`;
    },
  }]);
}

function nodeData(n) {
  return { id: n.id, short: shortLabel(n), card: cardLabel(n),
           agent: agentTag(n.worker), glyph: GLYPH[n.status] || "·",
           tag: "T-" + String(n.id).padStart(2, "0"), toktag: tokTag(n),
           turn: n.turn || 0,
           brief: cut(clean(n.desc), 72), goal: cut(clean(n.desc), 190),
           pid: n.parent || "", desc: n.desc || "", status: n.status,
           worker: n.worker || "", attempts: n.attempts || 0, cost: n.cost || 0,
           tok: (n.itok || 0) + (n.otok || 0), itok: n.itok || 0, otok: n.otok || 0,
           dur: n.dur || 0, summary: n.summary || "", error: n.error || "" };
}

function edgesFor(n) {
  const out = [];
  const from = n.parent;
  // The graph holds agents only, so a wave is a *rank*, not a node: the edge
  // that carries a top-level task asks dagre for as many ranks as the turn
  // that planned it (`minLen` below), which lines wave 2 up to the right of
  // wave 1 instead of mixing both into one fan off the goal.
  if (from) out.push({ group: "edges", classes: "hier",
    data: { id: "h_" + from + "_" + n.id, source: from, target: n.id,
            minlen: (from === "root" && n.turn) ? n.turn : 1 } });
  for (const d of n.deps || []) out.push({ group: "edges", classes: "dep",
    data: { id: "d_" + d + "_" + n.id, source: d, target: n.id } });
  return out;
}

/* Cytoscape throws if an edge references a node it does not know yet, which
   happens whenever a task declares a dependency on a sibling that arrives
   later in the same delta. Nodes go in first, then only the edges whose two
   endpoints exist; the rest are retried on the next update. */
function addElements(nodes, edges) {
  cy.add(nodes);
  const ready = [], waiting = [];
  for (const e of edges.concat(pendingEdges)) {
    if (cy.$id(e.data.id).length) continue;
    (cy.$id(e.data.source).length && cy.$id(e.data.target).length ? ready : waiting).push(e);
  }
  if (ready.length) cy.add(ready);
  pendingEdges = waiting;
}

function applyNodes(nodes, fresh) {
  if (fresh) { cy.elements().remove(); pendingEdges = []; fitted = false; }
  const newNodes = [], newEdges = [];
  cy.batch(() => {
    for (const n of nodes) {
      const existing = cy.$id(n.id);
      if (existing.length) {
        existing.data(nodeData(n));
        existing.classes(nodeClasses(n.id, n.status));
      } else {
        newNodes.push({ group: "nodes", data: nodeData(n),
                        classes: nodeClasses(n.id, n.status) });
        newEdges.push(...edgesFor(n));
        dirty = true;
      }
    }
    if (newNodes.length || pendingEdges.length) addElements(newNodes, newEdges);
  });
  applyViewPolicy();
  paintFlowGroups();
  if (selectedId) renderPanel();
}

/* ---------- dependency groups ------------------------------------------ */

/* Tasks wired by dependency edges are one piece of work: someone prepares
   material, someone else consumes it. Giving each such cluster its own
   colour — on the borders and on the arrows between them — makes those
   hand-offs visible without tracing a single edge by eye. Status keeps its
   own channel (fill, glyph, halo), so nothing is lost. */
/* Cool hues only: green, amber and rose belong to status, and a group fill
   that looks like a status colour is worse than no grouping at all. */
const GROUP_COLORS = ["#22d3ee", "#a78bfa", "#38bdf8", "#c084fc", "#818cf8",
                      "#60a5fa", "#e879f9", "#7dd3fc"];

/* Stable across updates: the colour follows the cluster's lowest id, so a
   group keeps its colour as tasks are added around it. */
function colorFor(key) {
  let h = 0;
  for (let i = 0; i < key.length; i++) h = (h * 31 + key.charCodeAt(i)) >>> 0;
  return GROUP_COLORS[h % GROUP_COLORS.length];
}

function paintFlowGroups() {
  const deps = cy.edges(".dep");
  if (!deps.length) return;
  const parent = {};                                   // union-find
  const find = x => { while (parent[x] !== x) x = parent[x] = parent[parent[x]]; return x; };
  const add = x => { if (!(x in parent)) parent[x] = x; };
  deps.forEach(e => {
    const a = e.data("source"), b = e.data("target");
    add(a); add(b);
    const ra = find(a), rb = find(b);
    if (ra !== rb) parent[ra] = rb;
  });
  const groupKey = {};                                 // root id -> lowest member
  for (const id in parent) {
    const r = find(id);
    if (!(r in groupKey) || id.length < groupKey[r].length ||
        (id.length === groupKey[r].length && id < groupKey[r])) groupKey[r] = id;
  }
  cy.batch(() => {
    for (const id in parent) {
      const el = cy.$id(id);
      if (!el.length) continue;
      el.data("gcolor", colorFor(groupKey[find(id)]));
      el.addClass("grouped");
    }
    deps.forEach(e => {
      const src = e.data("source");
      if (!(src in parent)) return;
      e.data("gcolor", colorFor(groupKey[find(src)]));
      e.addClass("grouped");
    });
  });
}

function nodeClasses(id, status) {
  const out = [status];
  if (id === "root") out.push("root");
  if (cardMode) out.push("card");
  else if (!showLabels) out.push("nolabel");
  return out.join(" ");
}

/* ---------- layout ----------------------------------------------------- */

/* Depth over the hierarchy only — dependency edges would short-circuit it. */
function depthOf(el) {
  let d = 0, cur = el, guard = 0;
  while (cur && cur.length && cur.data("pid") && guard++ < 64) {
    cur = cy.$id(cur.data("pid"));
    if (!cur.length) break;
    d++;
  }
  return d;
}

/* Highest wave number on the board: two waves make the run a sequence. */
function maxTurn() {
  let m = 0;
  cy.nodes().forEach(el => { const t = el.data("turn") || 0; if (t > m) m = t; });
  return m;
}

/* Widest sibling row: a flat, wide tree is what decides LR over TB. */
function widestRank() {
  const kids = {};
  cy.nodes().forEach(el => {
    const pid = el.data("pid");
    if (pid) kids[pid] = (kids[pid] || 0) + 1;
  });
  let w = 0;
  for (const k in kids) if (kids[k] > w) w = kids[k];
  return w;
}

function layoutOptions() {
  const n = cy.nodes().length;
  const animate = n <= 250;               // animation on a big graph is soup
  const common = { animate, animationDuration: 320, fit: !fitted, padding: 46 };
  if (layoutName === "tree") {
    if (HAS_DAGRE) {
      // Siblings laid out left-to-right make a strip nothing can read on a
      // 16:9 screen; ranks flowing rightward use the height instead. Cards
      // are ~170px wide and dots ~40, so they tip over at different widths.
      // Past the first wave the run *is* a left-to-right sequence, so LR
      // stops being a width heuristic — it is the reading order.
      const lr = maxTurn() > 1 || widestRank() > (cardMode ? 5 : 12);
      // One rank per wave: a task planned by turn N sits N ranks off the
      // goal. Everything else keeps dagre's default of one.
      return Object.assign({ name: "dagre", rankDir: lr ? "LR" : "TB",
                             ranker: "tight-tree",
                             minLen: e => e.data("minlen") || 1,
                             nodeSep: cardMode ? (lr ? 16 : 24) : 12,
                             edgeSep: 10,
                             rankSep: cardMode ? (lr ? 90 : 78) : 62 }, common);
    }
    return Object.assign({ name: "breadthfirst", roots: "#root", directed: true,
                           spacingFactor: 1.1, grid: false, avoidOverlap: true },
                         common);
  }
  if (layoutName === "radial") {
    // Wide fan-outs (one task per item) read far better as rings than as one
    // 100-node row.
    return Object.assign({ name: "concentric", concentric: el => -depthOf(el),
                           levelWidth: () => 1, minNodeSpacing: cardMode ? 26 : 14,
                           avoidOverlap: true, spacingFactor: 1 }, common);
  }
  return Object.assign({ name: "cose", randomize: false, nodeRepulsion: 9000,
                         idealEdgeLength: 62, nestingFactor: 0.9, gravity: 0.6,
                         numIter: n > 400 ? 350 : 1000 }, common);
}

function relayout() {
  const opts = layoutOptions();
  // Elbow edges only make sense in the layered tree, and they have to elbow
  // the way the ranks flow.
  cy.batch(() => {
    const flow = layoutName === "tree";
    cy.edges(".hier").toggleClass("flow", flow);
    cy.edges(".hier").toggleClass("lr", flow && opts.rankDir === "LR");
  });
  const run = cy.layout(opts);
  run.one("layoutstop", () => {
    if (cy.zoom() > 1.25) { cy.zoom(1.25); cy.center(); }
  });
  run.run();
  fitted = true;
}

function syncLayoutBtn() { $("layoutBtn").textContent = layoutName; }

setInterval(() => {
  if (!dirty) return;
  dirty = false;
  relayout();
}, 900);

/* Cards carry their own label, so the label toggle only applies to dots. */
function applyViewPolicy() {
  const n = cy.nodes().length;
  cardMode = showLabels && n <= CARD_LIMIT;
  const hideDotLabels = !cardMode && (!showLabels || n > LABEL_LIMIT);
  cy.batch(() => cy.nodes().forEach(el => {
    el.toggleClass("card", cardMode);
    el.toggleClass("htmllabel", cardMode && HAS_HTML_LABEL);
    el.toggleClass("nolabel", hideDotLabels && el.id() !== "root");
  }));
  $("labelBtn").textContent = showLabels ? "labels on" : "labels off";
  $("labelBtn").title = showLabels && n > CARD_LIMIT
    ? "cards collapse to dots above " + CARD_LIMIT + " tasks (click one for details)"
    : "show or hide task labels";
}

$("layoutBtn").onclick = () => {
  layoutName = LAYOUTS[(LAYOUTS.indexOf(layoutName) + 1) % LAYOUTS.length];
  registerHtmlLabels();
syncLayoutBtn();
  relayout();
};
$("labelBtn").onclick = () => {
  showLabels = !showLabels;
  applyViewPolicy();
  relayout();                         // cards and dots need different spacing
};
$("fitBtn").onclick = () => cy.animate({ fit: { padding: 46 }, duration: 250 });

/* Export the whole graph, not just the viewport.

   The card labels are HTML drawn *over* the canvas, so cy.png() cannot see
   them: exporting as-is would produce cards with no text at all. The canvas
   labels are switched back on for the duration of the capture — the PNG
   loses the per-field colours, but it keeps every word, which is the point
   of an export. */
$("pngBtn").onclick = () => {
  const htmlWasOn = cy.nodes(".htmllabel").nonempty();
  if (htmlWasOn) cy.batch(() => cy.nodes().removeClass("htmllabel"));
  try {
    const bb = cy.elements().boundingBox();
    // 3x for a crisp export, backed off so a thousand-node graph cannot ask
    // the browser for a 30000px canvas and fail silently.
    const scale = Math.max(1, Math.min(3, 8000 / Math.max(bb.w, bb.h, 1)));
    const uri = cy.png({ full: true, scale, bg: "#080d18" });
    const a = document.createElement("a");
    a.href = uri;
    stampCredit(uri, a);
    a.download = "myriapod-" + (runId ? runId.slice(0, 8) : "graph") + ".png";
    document.body.appendChild(a);
    a.click();
    a.remove();
  } finally {
    if (htmlWasOn) applyViewPolicy();
  }
};

/* The page credit is DOM, so it is absent from a canvas export. Redraw the
   PNG with it burned in; if anything goes wrong the plain export is already
   on the anchor and still downloads. */
function stampCredit(uri, anchor) {
  const img = new Image();
  img.onload = () => {
    try {
      const c = document.createElement("canvas");
      c.width = img.width; c.height = img.height;
      const g = c.getContext("2d");
      g.drawImage(img, 0, 0);
      const size = Math.max(11, Math.round(img.width / 110));
      g.font = "600 " + size + "px ui-monospace, SFMono-Regular, Menlo, monospace";
      g.fillStyle = "#64748b";
      g.textAlign = "right";
      g.fillText("© GWENLAKE · myriapod", img.width - size, img.height - size);
      anchor.href = c.toDataURL("image/png");
    } catch (e) { /* keep the un-stamped export */ }
  };
  img.src = uri;
}

/* ---------- details panel ---------------------------------------------- */

const panel = $("panel"), panelBody = $("panelBody");
let selectedId = null;

const esc = s => String(s).replace(/[&<>"']/g,
  c => ({ "&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;" }[c]));

function renderPanel() {
  const el = cy.$id(selectedId);
  if (!el.length) { closePanel(); return; }
  const d = el.data();
  panel.querySelector(".id").textContent = d.id;
  const st = panel.querySelector(".st");
  st.className = "st s-" + d.status;
  st.textContent = d.status.replace("_", " ");
  const row = (k, v, cls) => v || v === 0
    ? `<div class="k">${esc(k)}</div><div class="v ${cls || ""}">${esc(v)}</div>` : "";
  const tile = (v, k) => `<div class="tile"><b>${esc(v)}</b><span>${esc(k)}</span></div>`;
  panelBody.innerHTML =
    `<div class="tiles">` +
      tile((d.itok + d.otok).toLocaleString(), "tokens") +
      tile("$" + (+d.cost).toFixed(4), "cost") +
      tile(d.dur ? (+d.dur).toFixed(1) + "s" : "—", "duration") +
    `</div>` +
    row("description", d.desc) +
    row(d.id === "root" ? "planner" : "worker", d.worker, "mono") +
    (d.attempts > 1 ? row("attempts", d.attempts) : "") +
    row("tokens", d.itok + " in / " + d.otok + " out", "mono") +
    row("summary", d.summary) +
    row("error", d.error, "err") +
    planRows(d.id);
}

/* What this node planned: its direct subtasks. On the root that is the plan
   itself, which is the thing you most want to read after clicking it. */
function planRows(id) {
  const kids = cy.nodes().filter(n => n.data("pid") === id);
  if (!kids.length) return "";
  const items = kids.sort((a, b) => (+a.id() || 0) - (+b.id() || 0)).map(n => {
    const k = n.data();
    return `<li class="pl s-${k.status}"><span class="g">${esc(k.glyph)}</span>` +
      `<span class="id">${esc(k.tag)}</span> ${esc(cut(clean(k.desc), 90))}</li>`;
  }).join("");
  return `<div class="k">plan — ${kids.length} subtask` +
    (kids.length === 1 ? "" : "s") + `</div><ul class="plan">${items}</ul>`;
}

function openPanel(id) {
  selectedId = id;
  panel.classList.add("open");
  renderPanel();
  cy.resize();                      // the canvas just lost 352px — tell cytoscape
}
function closePanel() {
  selectedId = null;
  panel.classList.remove("open");
  cy.elements().removeClass("faded hi");
  setDim(() => false);
  cy.resize();
}

cy.on("tap", "node", evt => {
  const el = evt.target;
  openPanel(el.id());
  if (el.hasClass("turn")) return;
  const keep = el.closedNeighborhood();
  cy.batch(() => {
    cy.elements().addClass("faded").removeClass("hi");
    keep.removeClass("faded");
    el.addClass("hi");
    el.connectedEdges().addClass("hi");
  });
  setDim(n => !keep.contains(n));
});

/* The HTML labels are plain divs on top of the canvas: the .faded class does
   not reach them, so the dim state rides in the node data the extension
   re-renders on. */
function setDim(pred) {
  if (!HAS_HTML_LABEL || !cardMode) return;
  cy.batch(() => cy.nodes().forEach(el => {
    const v = pred(el) ? 1 : 0;
    if (el.data("dim") !== v) el.data("dim", v);
  }));
}
cy.on("tap", evt => { if (evt.target === cy) closePanel(); });
$("panelClose").onclick = () => { cy.$(":selected").unselect(); closePanel(); };
document.addEventListener("keydown", e => { if (e.key === "Escape") closePanel(); });
window.addEventListener("resize", () => cy.resize());

/* ---------- live stream ------------------------------------------------- */

const badge = $("badge");
function setBadge(cls, text) { badge.className = cls; badge.textContent = text; }

function setProgress(stats) {
  const b = stats.by_status || {}, total = stats.tasks || 0;
  const pct = k => (total ? (100 * (b[k] || 0) / total) : 0) + "%";
  $("prog").querySelector(".pdone").style.width = pct("done");
  $("prog").querySelector(".prun").style.width = pct("in_progress");
  $("prog").querySelector(".pfail").style.width = pct("failed");
}

const es = new EventSource("/events");
es.onmessage = e => {
  const msg = JSON.parse(e.data);
  if (msg.type === "waiting") { setBadge("waiting", "WAITING"); return; }
  if (msg.run_id && msg.run_id !== runId) {
    runId = msg.run_id; cy.elements().remove(); pendingEdges = []; fitted = false;
  }
  if (msg.nodes) applyNodes(msg.nodes, msg.type === "snapshot");
  if (msg.stats) {
    const b = msg.stats.by_status || {};
    // A counter reading zero is noise: it says nothing the progress strip
    // below the bar does not already show. Only non-zero states earn a slot.
    const bits = [`${msg.stats.tasks} tasks`];
    if (b.done) bits.push(`✔ ${b.done}`);
    if (b.in_progress) bits.push(`▶ ${b.in_progress}`);
    if (b.pending) bits.push(`· ${b.pending}`);
    if (b.failed) bits.push(`✖ ${b.failed}`);
    bits.push(`$${(+msg.stats.cost).toFixed(4)}`);
    $("stats").textContent = bits.join(" · ");
    setProgress(msg.stats);
  }
  if (msg.type === "end") setBadge("ended", "ENDED");
  else setBadge("live", "LIVE");
};
es.onerror = () => setBadge("waiting", "DISCONNECTED");

registerHtmlLabels();
syncLayoutBtn();

}
</script>
</body>
</html>
"""
