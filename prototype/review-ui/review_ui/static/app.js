/* Shell: language, theme, routing, shared formatting and the Plotly theme.
   Views live in census.js / player.js / cohort.js and each exports render(el). */

import { renderCensus } from "/static/census.js";
import { renderPlayer } from "/static/player.js";
import { renderCohort } from "/static/cohort.js";

export const state = { lang: "ja", theme: "auto", strings: {} };

export const THEMES = ["auto", "light", "dark"];

export function t(key) {
  const entry = state.strings[key];
  if (!entry) return key;
  return entry[state.lang] || entry.en || key;
}

/** Published tokens (reason codes, statuses, dispositions) get a gloss where the
    UI has one and stay verbatim otherwise — an unglossed token is data, not a
    label, and inventing a translation for it would hide a schema change. */
export function token(value) {
  const key = `token.${value}`;
  return state.strings[key] ? t(key) : String(value);
}

/** A published token longer than its column has no space to break on, so CSS
    splits it mid-word.  Offering `<wbr>` after each separator wraps it where the
    schema already puts a boundary. */
export function wrappable(value) {
  const node = el("span", {});
  String(value)
    .split(/(?<=[_\-/])/)
    .forEach((part, index) => {
      if (index) node.append(document.createElement("wbr"));
      node.append(document.createTextNode(part));
    });
  return node;
}

export function num(value, digits = 0) {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value !== "number") return String(value);
  return value.toLocaleString(state.lang === "ja" ? "ja-JP" : "en-US", {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits > 0 && Math.abs(value) < 1000 ? Math.min(digits, 2) : 0,
  });
}

export function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "html") node.innerHTML = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else if (key === "style" && typeof value === "object") Object.assign(node.style, value);
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child.nodeType ? child : document.createTextNode(String(child)));
  }
  return node;
}

export function panel(title, ...children) {
  return el("section", { class: "panel" }, el("h2", {}, title), ...children);
}

export function table(headers, rows, aligns = []) {
  const head = el(
    "tr",
    {},
    headers.map((label, i) => el("th", { class: aligns[i] === "num" ? "num" : null }, label)),
  );
  const body = rows.map((row) =>
    el(
      "tr",
      {},
      row.map((cell, i) =>
        el(
          "td",
          { class: aligns[i] === "num" ? "num" : null },
          cell && cell.nodeType ? cell : cell === null || cell === undefined ? "—" : String(cell),
        ),
      ),
    ),
  );
  return el("table", {}, el("thead", {}, head), el("tbody", {}, body));
}

/* Every colour token is a `light-dark()` pair, and a custom property reads back
   as the text it was authored with — so `getPropertyValue("--accent")` returns
   the call, not a colour.  A hidden probe carrying the token on a real colour
   property resolves it for the scheme in force, which is what canvas and Plotly
   need.  Each read is one style recalculation, so the views call `palette()`
   once per render rather than per data point. */
const probe = el("span", { style: { display: "none" } });

export function cssColour(name) {
  if (!probe.isConnected) document.documentElement.append(probe);
  probe.style.color = `var(${name})`;
  return getComputedStyle(probe).color;
}

/** `rgb(r, g, b)` from `cssColour` plus an alpha, for a chart's fill under its
    own line colour.  Keeping the hue in one token beats a second token per
    translucent variant. */
export function withAlpha(colour, alpha) {
  const parts = colour.match(/[\d.]+/g) || [];
  return `rgba(${parts.slice(0, 3).join(", ")}, ${alpha})`;
}

/** The chart palette, read from the stylesheet so the figures move with the
    theme.  `series` extends the four semantic tokens for categorical panels. */
export function palette() {
  const accent = cssColour("--accent");
  const amber = cssColour("--amber");
  const good = cssColour("--good");
  const warn = cssColour("--warn");
  return {
    accent,
    amber,
    good,
    warn,
    text: cssColour("--text"),
    series: [good, accent, amber, warn, cssColour("--series-5"), cssColour("--series-6")],
  };
}

/** One Plotly layout for every chart, so the figures read as one document. */
export function chartLayout(overrides = {}) {
  const line = cssColour("--line");
  const muted = cssColour("--muted");
  return Object.assign(
    {
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { family: "Plex Sans, Plex Sans JP, sans-serif", size: 11, color: muted },
      margin: { l: 56, r: 12, t: 8, b: 40 },
      xaxis: { gridcolor: line, zerolinecolor: line, linecolor: line, automargin: true },
      yaxis: { gridcolor: line, zerolinecolor: line, linecolor: line, automargin: true },
      legend: { orientation: "h", y: -0.22, font: { size: 11 } },
      hoverlabel: { font: { family: "Plex Sans, Plex Sans JP, sans-serif", size: 11 } },
      barmode: "group",
      bargap: 0.28,
    },
    overrides,
  );
}

export const PLOT_CONFIG = { displayModeBar: false, responsive: true };

/** Resolves once the two chart faces are usable, so Plotly measures real metrics.
 *
 * `newPlot` measures legend and tick text at call time and lays out from that.
 * Loaded Plex metrics wrap the rotation legend to two rows; the narrower fallback
 * metrics fit it on one, which moves every chart below it.  Three of four captures
 * of one server agreed and the fourth differed on 20771 pixels, purely from which
 * side won the race.
 * `fonts.ready` alone is not enough — it resolves against whatever has been
 * requested so far, so each family is asked for by name first.
 */
const FONTS_READY = document.fonts
  ? Promise.all([
      document.fonts.load("11px 'Plex Sans'"),
      document.fonts.load("11px 'Plex Sans JP'"),
    ])
      .then(() => document.fonts.ready)
      .catch(() => undefined)
  : Promise.resolve();

export function chart(node, traces, layout) {
  FONTS_READY.then(() => Plotly.newPlot(node, traces, chartLayout(layout), PLOT_CONFIG));
  return node;
}

export async function json(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`${path} -> ${response.status}`);
  return response.json();
}

const VIEWS = {
  player: { render: renderPlayer, node: () => document.getElementById("view-player") },
  census: { render: renderCensus, node: () => document.getElementById("view-census") },
  cohort: { render: renderCohort, node: () => document.getElementById("view-cohort") },
};

/** The player is the landing view: the clips are what a reviewer comes to look
    at, and the census and the cohort explorer answer questions raised there. */
const DEFAULT_VIEW = "player";

let current = DEFAULT_VIEW;
const rendered = new Set();

async function show(name, force = false) {
  current = name;
  // Before the render await, not after it: a trailing write belongs to whichever
  // render finishes last, so a switch made while a slow view was still rendering
  // was dragged back by the older call — the back button off a first cohort visit
  // returned to cohort.  Written synchronously, a superseded call writes nothing,
  // and its render lands in its own hidden section.  The re-entrant `hashchange`
  // this fires reads `current` as already `name` and stops there.
  location.hash = name;
  for (const tab of document.querySelectorAll(".tab")) {
    tab.classList.toggle("active", tab.dataset.view === name);
  }
  for (const [key, view] of Object.entries(VIEWS)) {
    view.node().classList.toggle("active", key === name);
  }
  if (force || !rendered.has(name)) {
    rendered.add(name);
    const node = VIEWS[name].node();
    node.replaceChildren(el("p", { class: "empty" }, t("common.loading")));
    await VIEWS[name].render(node);
  }
}

function applyStaticText() {
  document.documentElement.lang = state.lang;
  for (const node of document.querySelectorAll("[data-i18n]")) {
    node.textContent = t(node.dataset.i18n);
  }
  document.getElementById("lang-toggle").textContent = state.lang === "ja" ? "EN" : "日本語";
  const theme = document.getElementById("theme-toggle");
  theme.textContent = t(`theme.${state.theme}`);
  theme.title = `${t("theme.label")} — ${t(`theme.${state.theme}`)}`;
}

/** `auto` is the absent attribute, which leaves the scheme to `light-dark()`. */
function applyTheme() {
  if (state.theme === "auto") delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = state.theme;
}

async function boot() {
  // `?lang=en&theme=light#cohort` makes a view deep-linkable in either language
  // and either theme, which is what a shared review link needs; a toggle then
  // persists the choice.  index.html already resolved the theme before the first
  // paint, so this reads the answer back rather than deriving it twice.
  const requested = new URLSearchParams(location.search).get("lang");
  state.lang = ["ja", "en"].includes(requested)
    ? requested
    : localStorage.getItem("review-ui-lang") || "ja";
  state.theme = document.documentElement.dataset.theme || "auto";
  state.strings = await json("/static/strings.json");
  applyStaticText();

  document.getElementById("tabs").addEventListener("click", (event) => {
    const tab = event.target.closest(".tab");
    if (tab) show(tab.dataset.view);
  });
  document.getElementById("lang-toggle").addEventListener("click", () => {
    state.lang = state.lang === "ja" ? "en" : "ja";
    localStorage.setItem("review-ui-lang", state.lang);
    applyStaticText();
    rendered.clear();
    show(current, true);
  });
  // Plotly bakes the palette into a figure at plot time, so the current view is
  // redrawn rather than restyled; the others redraw when they are next shown.
  document.getElementById("theme-toggle").addEventListener("click", () => {
    state.theme = THEMES[(THEMES.indexOf(state.theme) + 1) % THEMES.length];
    localStorage.setItem("review-ui-theme", state.theme);
    applyTheme();
    applyStaticText();
    rendered.clear();
    show(current, true);
  });
  // Under `auto` the stylesheet follows the system on its own, but a baked
  // figure and a painted canvas do not.
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    if (state.theme !== "auto") return;
    rendered.clear();
    show(current, true);
  });

  // The fragment is the view selector, and `show` writes one history entry per
  // switch, so the fragment moves without a load in two ordinary cases: the back
  // button, and a shared `#census` link opened in a tab that already holds the UI.
  // Reading it at boot alone leaves the address bar and the stage disagreeing —
  // and with the player as the landing view, that stale stage is patient video.
  // `show` assigns the same hash back, which is a no-op and re-enters nothing.
  addEventListener("hashchange", () => {
    const name = location.hash.replace("#", "");
    const target = VIEWS[name] ? name : DEFAULT_VIEW;
    if (target !== current) show(target);
  });

  const initial = location.hash.replace("#", "");
  await show(VIEWS[initial] ? initial : DEFAULT_VIEW);
}

boot();
