/* Clip player: a canvas overlay driven by the exported landmark series.

   The skeleton topology and palette come from /static/topology.json, which
   tools/export_topology.py writes out of pose_estimation.drawing — the overlay
   draws what the pipeline draws.

   Two clocks.  The <video> element is the clock whenever the browser can decode
   the clip; when it cannot — HEVC on a build without a platform decoder, or the
   video layer hidden entirely — a rAF clock takes over, so the overlay reviews
   on its own. */

import { cssColour, el, json, num, panel, t, token } from "/static/app.js";

/** Confidence bands for the overlay's landmark dots.  This palette is the UI's
    own — pose_estimation.drawing colours a dot by body group, not by confidence,
    so these three are a review affordance and not pipeline output.  They stay
    fixed in both themes anyway, because they are drawn over video pixels rather
    than over a page surface.  The strip below the transport encodes the same
    quantity on a page surface, so it takes the themed tokens instead. */
const OVERLAY_VIS = { high: "#5ec9a4", mid: "#f2a65a", low: "#e0705c" };
let topology = null;

const view = {
  clips: [],
  filtered: [],
  search: "",
  clip: null,
  series: null,
  frame: 0,
  playing: false,
  layer: "show",
  showBody: true,
  showHands: true,
  showPoints: true,
  showLabels: false,
  threshold: 0.5,
  speed: 1,
  clock: 0,
  raf: null,
};

/** Positional event ordinal — see clips._number_families for why it is not an id. */
function familyTag(clip) {
  return `#${String(clip.family_no).padStart(3, "0")}`;
}

function label(clip) {
  const task = clip.task ? t(`task.${clip.task}`) : "—";
  const side = clip.side ? t(`side.${clip.side}`) : "";
  const viewName = clip.view ? t(`view.${clip.view}`) : "";
  return `${familyTag(clip)} ${task} ${side} · ${viewName}`.trim();
}

/* ------------------------------------------------------------------ drawing */

function colourFor(value, colours) {
  if (value === null || value === undefined) return null;
  if (value >= 0.75) return colours.high;
  if (value >= view.threshold) return colours.mid;
  return colours.low;
}

function drawSegments(ctx, points, segments, colours, scale) {
  for (const segment of segments) {
    const [a, b] = segment.points;
    const pa = points[a];
    const pb = points[b];
    if (!pa || !pb) continue;
    ctx.strokeStyle = colours[segment.group] || "#ffffff";
    ctx.beginPath();
    ctx.moveTo(pa[0] * scale, pa[1] * scale);
    ctx.lineTo(pb[0] * scale, pb[1] * scale);
    ctx.stroke();
  }
}

function drawChains(ctx, points, chains, colours, scale) {
  for (const chain of chains) {
    const path = chain.points.map((index) => points[index]).filter(Boolean);
    if (path.length < 2) continue;
    ctx.strokeStyle = colours[chain.group] || "#ffffff";
    ctx.beginPath();
    ctx.moveTo(path[0][0] * scale, path[0][1] * scale);
    for (let i = 1; i < path.length; i += 1) {
      ctx.lineTo(path[i][0] * scale, path[i][1] * scale);
    }
    ctx.stroke();
  }
}

/** Unpack one frame's flat [x, y, confidence, ...] row into visible points. */
function unpack(row, count, threshold) {
  const points = new Array(count).fill(null);
  const confidences = new Array(count).fill(null);
  if (!row) return { points, confidences };
  for (let i = 0; i < count; i += 1) {
    const x = row[i * 3];
    const y = row[i * 3 + 1];
    const confidence = row[i * 3 + 2];
    confidences[i] = confidence;
    if (x === null || y === null || confidence === null) continue;
    if (confidence < threshold) continue;
    points[i] = [x, y];
  }
  return { points, confidences };
}

function paint() {
  const canvas = document.getElementById("overlay");
  if (!canvas || !view.series || !topology) return;
  const stage = canvas.parentElement;
  const width = stage.clientWidth;
  const height = stage.clientHeight;
  const ratio = window.devicePixelRatio || 1;
  if (canvas.width !== Math.round(width * ratio)) {
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const series = view.series;
  const clip = view.clip;
  // Coordinates were normalised by max(frame_w, frame_h); the stage box has the
  // clip's own aspect ratio, so one factor maps both axes.
  const displayWidth = clip.width || 1;
  const scale = (series.scale || Math.max(clip.width, clip.height)) * (width / displayWidth);
  const frame = Math.min(view.frame, series.frames - 1);

  ctx.lineCap = "round";
  ctx.lineJoin = "round";

  if (view.showBody) {
    const body = unpack(series.body[frame], series.body_names.length, view.threshold);
    ctx.lineWidth = Math.max(width / 300, 1.6);
    drawSegments(ctx, body.points, topology.body.segments, topology.body.colors, scale);
    drawChains(ctx, body.points, topology.body.chains, topology.body.colors, scale);
    if (view.showPoints) {
      body.points.forEach((point, index) => {
        if (!point) return;
        ctx.fillStyle = colourFor(body.confidences[index], OVERLAY_VIS) || "#ffffff";
        ctx.beginPath();
        ctx.arc(point[0] * scale, point[1] * scale, Math.max(width / 320, 2), 0, Math.PI * 2);
        ctx.fill();
      });
    }
    if (view.showLabels) {
      ctx.fillStyle = "rgba(231,234,242,0.82)";
      ctx.font = `${Math.max(width / 90, 9)}px "Plex Mono", monospace`;
      body.points.forEach((point, index) => {
        if (!point) return;
        ctx.fillText(series.body_names[index], point[0] * scale + 5, point[1] * scale - 5);
      });
    }
  }

  if (view.showHands) {
    ctx.lineWidth = Math.max(width / 420, 1.2);
    for (const side of ["hand_left", "hand_right"]) {
      const hand = unpack(series[side][frame], series.hand_points, view.threshold);
      drawSegments(ctx, hand.points, topology.hand.segments, topology.hand.colors, scale);
      drawChains(ctx, hand.points, topology.hand.chains, topology.hand.colors, scale);
      if (!view.showPoints) continue;
      hand.points.forEach((point, index) => {
        if (!point) return;
        ctx.fillStyle = topology.hand.point_colors[index] || "#ffffff";
        ctx.beginPath();
        ctx.arc(point[0] * scale, point[1] * scale, Math.max(width / 480, 1.4), 0, Math.PI * 2);
        ctx.fill();
      });
    }
  }
  paintStrip();
}

/** Mean body visibility per frame — where tracking drops out is the first thing
    a reviewer looks for, and a single strip shows it for a whole clip. */
function paintStrip() {
  const canvas = document.getElementById("strip");
  if (!canvas || !view.series) return;
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const values = view.series.visibility;
  const bands = { high: cssColour("--good"), mid: cssColour("--amber"), low: cssColour("--warn") };
  const absent = cssColour("--line");
  const step = width / Math.max(values.length, 1);
  for (let i = 0; i < values.length; i += 1) {
    const value = values[i];
    ctx.fillStyle = colourFor(value, bands) || absent;
    const barHeight = Math.max((value || 0) * height, 1);
    ctx.fillRect(i * step, height - barHeight, Math.max(step, 1), barHeight);
  }
  ctx.strokeStyle = cssColour("--text");
  ctx.lineWidth = 1;
  const x = (view.frame / Math.max(values.length - 1, 1)) * width;
  ctx.beginPath();
  ctx.moveTo(x, 0);
  ctx.lineTo(x, height);
  ctx.stroke();
}

/* ------------------------------------------------------------------- clocks */

function fps() {
  return view.clip.fps || 30;
}

function setFrame(frame) {
  if (!view.series) return;
  view.frame = Math.max(0, Math.min(Math.round(frame), view.series.frames - 1));
  const seek = document.getElementById("seek");
  if (seek && Number(seek.value) !== view.frame) seek.value = String(view.frame);
  const readout = document.getElementById("readout");
  if (readout) {
    const seconds = view.series.timestamp_sec[view.frame];
    readout.textContent =
      `${t("player.frame")} ${view.frame + 1}/${view.series.frames} · ` +
      `${t("player.time")} ${(seconds ?? view.frame / fps()).toFixed(2)}s · ` +
      `${t("player.visibility")} ${(view.series.visibility[view.frame] ?? 0).toFixed(2)}`;
  }
  paint();
}

function videoUsable() {
  const video = document.getElementById("clip-video");
  return video && !video.error && video.readyState >= 2 && view.layer !== "hide";
}

function tick(timestamp) {
  if (!view.playing) return;
  const video = document.getElementById("clip-video");
  if (videoUsable() && !video.paused) {
    setFrame(video.currentTime * fps());
  } else {
    const elapsed = view.clock ? (timestamp - view.clock) / 1000 : 0;
    view.clock = timestamp;
    const next = view.frame + elapsed * fps() * view.speed;
    setFrame(next >= view.series.frames ? 0 : next);
  }
  view.raf = requestAnimationFrame(tick);
}

function play() {
  if (!view.series) return;
  view.playing = true;
  view.clock = 0;
  const video = document.getElementById("clip-video");
  if (video && !video.error) {
    video.playbackRate = view.speed;
    video.currentTime = view.frame / fps();
    video.play().catch(() => {});
  }
  document.getElementById("play").textContent = "⏸";
  view.raf = requestAnimationFrame(tick);
}

function pause() {
  view.playing = false;
  const video = document.getElementById("clip-video");
  if (video) video.pause();
  const button = document.getElementById("play");
  if (button) button.textContent = "▶";
  if (view.raf) cancelAnimationFrame(view.raf);
}

/* ---------------------------------------------------------------- fitting */

/** The player is the one view sized to the viewport instead of to its content:
    the transport has to stay on screen while a clip plays.  The chrome above the
    layout moves with the language and with the top bar's wrap, so it is measured
    rather than assumed.  The stylesheet owns the breakpoint and reports it back
    through `--fit-viewport`, so the stacked layout keeps its content height. */
function fitLayout() {
  const layout = document.querySelector(".player-layout");
  if (!layout) return;
  if (getComputedStyle(layout).getPropertyValue("--fit-viewport").trim() === "0") {
    layout.style.height = "";
  } else {
    const main = layout.closest("main");
    const below = main ? parseFloat(getComputedStyle(main).paddingBottom) || 0 : 0;
    const top = layout.getBoundingClientRect().top + window.scrollY;
    layout.style.height = `${Math.max(window.innerHeight - top - below, 360)}px`;
  }
  fitStage();
  // The strip spans the panel, so its backing store moves with a width the stage
  // can keep — a height-limited stage does not change when the window widens.
  // The strip spans the panel, so its backing store moves with a width the stage
  // can keep — a height-limited stage does not change when the window widens.
  paintStrip();
}

/** The largest box of the clip's aspect ratio that fits the space left over.
    The overlay maps landmarks onto the stage's *content* box, so the ratio is
    fitted there and the border added back — `box-sizing` is border-box. */
function fitStage() {
  const stage = document.getElementById("stage");
  if (!stage || !view.clip) return;
  const room = stage.parentElement.getBoundingClientRect();
  const style = getComputedStyle(stage);
  const borderX = parseFloat(style.borderLeftWidth) + parseFloat(style.borderRightWidth);
  const borderY = parseFloat(style.borderTopWidth) + parseFloat(style.borderBottomWidth);
  const aspect = (view.clip.width || 16) / (view.clip.height || 9);
  const width = Math.floor(Math.min(room.width - borderX, (room.height - borderY) * aspect));
  if (width < 1) return;
  const next = [`${width + borderX}px`, `${Math.floor(width / aspect) + borderY}px`];
  // The observer below watches a box this call resizes; re-entering on an
  // unchanged size is the loop that guard prevents.
  if (stage.style.width === next[0] && stage.style.height === next[1]) return;
  [stage.style.width, stage.style.height] = next;
  paint();
}

/** Every cause of a size change that is not a window resize — a font landing and
    retaking the top bar's height, a control row wrapping, a scrollbar — reaches
    the stage through its own box. */
let stageObserver = null;
let resizeBound = false;

function observeStage() {
  const stage = document.getElementById("stage");
  if (!stage || !window.ResizeObserver) return;
  if (!stageObserver) stageObserver = new ResizeObserver(() => fitStage());
  stageObserver.disconnect();
  stageObserver.observe(stage.parentElement);
}

/* -------------------------------------------------------------------- panes */

function clipList() {
  const list = el("div", { class: "clip-list" });
  const rows = view.filtered.map((clip) =>
    el(
      "button",
      {
        class: `clip-row${view.clip && clip.event_id === view.clip.event_id && clip.camera_name === view.clip.camera_name ? " on" : ""}`,
        type: "button",
        onclick: () => selectClip(clip),
      },
      el("span", {}, label(clip)),
      el(
        "span",
        { class: "meta" },
        `${clip.n_cameras ? `${clip.n_cameras}v · ` : ""}${clip.frames ? `${num(clip.frames)}f` : "—"}`,
      ),
    ),
  );
  list.append(...(rows.length ? rows : [el("p", { class: "empty" }, t("common.none"))]));
  return list;
}

function applyFilters() {
  const value = (id) => document.getElementById(id)?.value || "";
  const task = value("filter-task");
  const side = value("filter-side");
  const viewName = value("filter-view");
  const landmarksOnly = document.getElementById("filter-landmarks")?.checked;
  const term = view.search.trim().toLowerCase();
  view.filtered = view.clips.filter(
    (clip) =>
      (!task || clip.task === task) &&
      (!side || clip.side === side) &&
      (!viewName || clip.view === viewName) &&
      (!landmarksOnly || clip.has_landmarks) &&
      (!term || label(clip).toLowerCase().includes(term)),
  );
  const host = document.getElementById("clip-list-host");
  host.replaceChildren(
    el("p", { class: "note" }, `${view.filtered.length} / ${view.clips.length}`),
    clipList(),
  );
}

function filterSelect(id, key, values) {
  return el(
    "label",
    { class: "field" },
    t(key),
    el(
      "select",
      { id, onchange: applyFilters },
      el("option", { value: "" }, t("player.all")),
      values.map((value) =>
        el("option", { value }, t(`${key.split(".")[1]}.${value}`) || value),
      ),
    ),
  );
}

function toggle(id, key, field) {
  return el(
    "button",
    {
      class: `ctl${view[field] ? " on" : ""}`,
      id,
      type: "button",
      onclick: (event) => {
        view[field] = !view[field];
        event.currentTarget.classList.toggle("on", view[field]);
        paint();
      },
    },
    t(key),
  );
}

/** One labelled clip fact on the meta line — the same pairs the key/value list
    carried, laid out along the row so they cost one line instead of four. */
function fact(key, value) {
  return el(
    "span",
    { class: "fact" },
    el("span", { class: "fact-k" }, key),
    el("span", { class: "fact-v" }, value),
  );
}

function stagePane() {
  const clip = view.clip;
  const aspect = `${clip.width || 16} / ${clip.height || 9}`;
  const source = `/api/clip/${encodeURIComponent(clip.event_id)}/${encodeURIComponent(clip.camera_name)}/video`;
  const video = el("video", {
    id: "clip-video",
    src: clip.has_video ? source : null,
    preload: "metadata",
    playsinline: true,
    class: view.layer === "dim" ? "dim" : view.layer === "hide" ? "hidden" : "",
  });
  video.addEventListener("loadedmetadata", () => paint());
  video.addEventListener("seeked", () => setFrame(video.currentTime * fps()));
  video.addEventListener("ended", () => pause());
  // A third of the corpus is HEVC, which a Chromium build without a platform
  // decoder refuses silently: the stage goes black and a reviewer cannot tell a
  // decode failure from a bad overlay.  Say which one it is, then hide the layer
  // so the rAF clock keeps the overlay running.
  video.addEventListener("error", () => {
    const banner = document.getElementById("decode-note");
    if (banner) {
      banner.hidden = false;
      banner.textContent = `${t("player.decode_failed")} — ${clip.codec || "—"}`;
    }
    video.className = "hidden";
    fitStage();
    paint();
  });

  // The ratio is the shape the box holds before the first fit; `fitStage` then
  // sizes it in pixels, which makes the ratio inert until JS fails to run.
  const stage = el(
    "div",
    { class: "stage", id: "stage", style: { aspectRatio: aspect } },
    video,
    el("canvas", { class: "overlay", id: "overlay" }),
  );

  const layerButtons = el(
    "div",
    { class: "segmented" },
    ["show", "dim", "hide"].map((mode) =>
      el(
        "button",
        {
          class: view.layer === mode ? "on" : "",
          type: "button",
          onclick: () => {
            view.layer = mode;
            for (const button of layerButtons.children) button.classList.remove("on");
            layerButtons.children[["show", "dim", "hide"].indexOf(mode)].classList.add("on");
            video.className = mode === "dim" ? "dim" : mode === "hide" ? "hidden" : "";
            paint();
          },
        },
        t(`player.layer_${mode}`),
      ),
    ),
  );

  const transport = el(
    "div",
    { class: "transport" },
    el("button", { class: "ctl", id: "play", type: "button", onclick: () => (view.playing ? pause() : play()) }, "▶"),
    el("button", { class: "ctl", type: "button", onclick: () => { pause(); setFrame(view.frame - 1); } }, "◀|"),
    el("button", { class: "ctl", type: "button", onclick: () => { pause(); setFrame(view.frame + 1); } }, "|▶"),
    el("input", {
      class: "seek",
      id: "seek",
      type: "range",
      min: 0,
      max: Math.max((view.series?.frames || 1) - 1, 0),
      value: view.frame,
      oninput: (event) => {
        pause();
        const frame = Number(event.target.value);
        const element = document.getElementById("clip-video");
        if (element && !element.error) element.currentTime = frame / fps();
        setFrame(frame);
      },
    }),
    el("span", { class: "readout", id: "readout" }),
  );

  const thresholdValue = el("span", { class: "fact-v" }, view.threshold.toFixed(2));
  const options = el(
    "div",
    { class: "player-opts" },
    el("div", { class: "field inline" }, el("span", {}, t("player.layer")), layerButtons),
    el(
      "div",
      { class: "field inline" },
      el("span", {}, t("player.overlay")),
      toggle("toggle-body", "player.body", "showBody"),
      toggle("toggle-hands", "player.hands", "showHands"),
      toggle("toggle-points", "player.points", "showPoints"),
      toggle("toggle-labels", "player.labels", "showLabels"),
    ),
    el(
      "label",
      { class: "field inline" },
      el("span", {}, t("player.threshold")),
      el("input", {
        type: "range",
        min: 0,
        max: 1,
        step: 0.05,
        value: view.threshold,
        oninput: (event) => {
          view.threshold = Number(event.target.value);
          thresholdValue.textContent = view.threshold.toFixed(2);
          paint();
        },
      }),
      thresholdValue,
    ),
    el(
      "label",
      { class: "field inline" },
      el("span", {}, t("player.speed")),
      el(
        "select",
        {
          onchange: (event) => {
            view.speed = Number(event.target.value);
            const element = document.getElementById("clip-video");
            if (element) element.playbackRate = view.speed;
          },
        },
        [0.25, 0.5, 1, 2].map((rate) =>
          el("option", { value: rate, selected: rate === view.speed }, `${rate}×`),
        ),
      ),
    ),
  );

  const facts = el(
    "div",
    { class: "factline" },
    fact(t("player.geometry"), `${clip.width}×${clip.height} · ${(clip.fps || 0).toFixed(3)} fps`),
    // `f` rather than the frame word: the clip list already counts frames that
    // way, and it reads as one unit in both languages.
    fact(t("player.duration"), `${num(clip.duration_s, 2)} s · ${num(clip.frames)} f`),
    fact(t("player.rotation"), `${clip.rotation_deg ?? 0}° · ${clip.codec || "—"}`),
    fact(t("player.disposition"), token(clip.disposition || "—")),
  );

  // The banner carries the decode failure alone, so it takes no room until one
  // happens; the clip's identity is the selected row in the list beside it.
  return el(
    "section",
    { class: "panel stage-panel" },
    el("div", { class: "banner", id: "decode-note", hidden: true }),
    el("div", { class: "stage-wrap" }, stage),
    el(
      "div",
      { class: "player-bar" },
      transport,
      el(
        "div",
        { class: "strip-row" },
        el("span", { class: "strip-label" }, t("player.strip")),
        el("canvas", { class: "strip", id: "strip" }),
      ),
      options,
      facts,
    ),
  );
}

async function selectClip(clip) {
  pause();
  view.clip = clip;
  view.frame = 0;
  const host = document.getElementById("stage-host");
  host.replaceChildren(el("p", { class: "empty" }, t("common.loading")));
  const series = await json(
    `/api/clip/${encodeURIComponent(clip.event_id)}/${encodeURIComponent(clip.camera_name)}/landmarks`,
  );
  if (!series.frames) {
    view.series = null;
    host.replaceChildren(
      el("section", { class: "panel stage-panel" }, el("p", { class: "empty" }, t("player.no_landmarks"))),
    );
    applyFilters();
    return;
  }
  // One pass over the body confidences: the strip needs a per-frame mean and the
  // readout needs the same number, so it is computed once with the series.
  const count = series.body_names.length;
  series.visibility = series.body.map((row) => {
    let total = 0;
    let seen = 0;
    for (let i = 0; i < count; i += 1) {
      const value = row[i * 3 + 2];
      if (value === null || value === undefined) continue;
      total += value;
      seen += 1;
    }
    return seen ? total / seen : 0;
  });
  view.series = series;
  host.replaceChildren(stagePane());
  applyFilters();
  fitLayout();
  observeStage();
  setFrame(0);
}

export async function renderPlayer(root) {
  if (!topology) topology = await json("/static/topology.json");
  const data = await json("/api/clips");
  view.clips = data.clips;
  const distinct = (key) => [...new Set(view.clips.map((clip) => clip[key]).filter(Boolean))].sort();

  const clipsPanel = panel(
    t("player.clips"),
    el(
      "div",
      { class: "controls", style: { marginBottom: "10px" } },
      filterSelect("filter-task", "player.task", distinct("task")),
      filterSelect("filter-side", "player.side", distinct("side")),
      filterSelect("filter-view", "player.view", distinct("view")),
      el(
        "label",
        { class: "field", style: { flex: "1 1 120px" } },
        t("player.search"),
        el("input", {
          type: "search",
          placeholder: "#042",
          oninput: (event) => {
            view.search = event.target.value;
            applyFilters();
          },
        }),
      ),
    ),
    el(
      "label",
      { style: { display: "flex", gap: "6px", alignItems: "center", fontSize: "12px" } },
      el("input", { type: "checkbox", id: "filter-landmarks", checked: true, onchange: applyFilters }),
      t("player.with_landmarks"),
    ),
    el("p", { class: "note" }, t("player.family_note")),
    el("div", { id: "clip-list-host" }),
  );
  clipsPanel.classList.add("clips-panel");

  root.replaceChildren(
    el(
      "div",
      { class: "player-layout" },
      clipsPanel,
      el("div", { id: "stage-host" }, el("p", { class: "empty" }, t("player.select"))),
    ),
  );
  applyFilters();
  fitLayout();
  // A language or theme change re-renders this view, so the listener is bound
  // once for the session rather than once per render.
  if (!resizeBound) {
    window.addEventListener("resize", () => fitLayout(), { passive: true });
    resizeBound = true;
  }
  if (view.filtered.length) await selectClip(view.filtered[0]);
}
