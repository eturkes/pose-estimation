/* Cohort explorer: one feature at a time across the 12 (task, side) cells.

   The publisher ships median / q25 / q75 / mean / sd and no extremes, so the
   chart draws exactly that — an interquartile bar with a median rule and a mean
   dot.  Drawing a whisker here would invent a number the export refuses to
   publish, for the reason the boundary panel states. */

import { chart, el, json, num, palette, panel, t, table, withAlpha, wrappable } from "/static/app.js";

const CELL_ORDER = ["cap", "coin", "glass", "key", "nut", "peg"];

const view = { data: null, level: "frame", feature: null, search: "" };

function cellLabel(row) {
  return `${t(`task.${row.task}`)}·${t(`side.${row.side}`)}`;
}

function sortCells(rows) {
  return [...rows].sort(
    (a, b) =>
      CELL_ORDER.indexOf(a.task) - CELL_ORDER.indexOf(b.task) || a.side.localeCompare(b.side),
  );
}

function rowsFor(feature) {
  return sortCells(
    view.data.rows.filter((row) => row.level === feature.level && row.feature === feature.feature),
  );
}

function featureLabel(feature, lang) {
  const text = lang === "ja" ? feature.ja : feature.en;
  return text || feature.feature;
}

function distributionChart(rows, feature) {
  const node = el("div", { class: "chart", style: { height: "300px" } });
  const { accent, amber, text } = palette();
  const labels = rows.map(cellLabel);
  const traces = [
    {
      type: "bar",
      name: "q25–q75",
      x: labels,
      base: rows.map((row) => row.q25),
      y: rows.map((row) => (row.q75 ?? 0) - (row.q25 ?? 0)),
      marker: { color: withAlpha(accent, 0.32), line: { color: accent, width: 1 } },
      hovertemplate: "%{x}<br>q25 %{base:.3f} · q75 %{customdata:.3f}<extra></extra>",
      customdata: rows.map((row) => row.q75),
    },
    {
      type: "scatter",
      mode: "markers",
      name: t("col.median"),
      x: labels,
      y: rows.map((row) => row.median),
      marker: { color: text, symbol: "line-ew-open", size: 26, line: { width: 3 } },
      hovertemplate: "%{x}<br>%{y:.4f}<extra></extra>",
    },
    {
      type: "scatter",
      mode: "markers",
      name: t("col.mean"),
      x: labels,
      y: rows.map((row) => row.mean),
      marker: { color: amber, size: 7 },
      hovertemplate: "%{x}<br>%{y:.4f}<extra></extra>",
    },
  ];
  // Plotly pads a bar axis at the zero side only, so a `base`+height bar ends
  // flush with the plot frame and reads as clipped.  The range is set from the
  // drawn statistics instead.
  const drawn = rows
    .flatMap((row) => [row.q25, row.q75, row.median, row.mean])
    .filter((value) => typeof value === "number");
  const low = Math.min(...drawn);
  const high = Math.max(...drawn);
  const pad = (high - low || Math.abs(high) || 1) * 0.08;
  requestAnimationFrame(() =>
    chart(node, traces, {
      yaxis: { title: { text: feature.unit || "" }, range: [low - pad, high + pad] },
      margin: { l: 64, r: 12, t: 8, b: 34 },
    }),
  );
  return node;
}

function dispersionChart(rows) {
  const node = el("div", { class: "chart", style: { height: "220px" } });
  requestAnimationFrame(() =>
    chart(
      node,
      [
        {
          type: "bar",
          x: rows.map(cellLabel),
          y: rows.map((row) => row.view_dispersion),
          marker: { color: palette().amber },
          hovertemplate: "%{x}<br>%{y:.3f}<extra></extra>",
        },
      ],
      { showlegend: false },
    ),
  );
  return node;
}

function statisticsTable(rows) {
  return table(
    [
      t("col.task"),
      t("col.side"),
      t("col.n_subjects"),
      t("col.n_events"),
      t("col.n_assets"),
      t("col.n_values"),
      t("col.median"),
      t("col.q25"),
      t("col.q75"),
      t("col.mean"),
      t("col.sd"),
      t("col.view_dispersion"),
      t("col.n_events_multiview"),
    ],
    rows.map((row) => [
      t(`task.${row.task}`),
      t(`side.${row.side}`),
      num(row.n_subjects),
      num(row.n_events),
      num(row.n_assets),
      num(row.n_values),
      num(row.median, 3),
      num(row.q25, 3),
      num(row.q75, 3),
      num(row.mean, 3),
      num(row.sd, 3),
      num(row.view_dispersion, 3),
      num(row.n_events_multiview),
    ]),
    ["", "", "num", "num", "num", "num", "num", "num", "num", "num", "num", "num", "num"],
  );
}

function cellsPanel() {
  const cells = sortCells(view.data.cells);
  return panel(
    t("cohort.cells"),
    el(
      "div",
      { class: "scroll", style: { maxHeight: "none" } },
      table(
        [
          t("col.task"),
          t("col.side"),
          t("col.n_subjects"),
          t("col.n_events"),
          t("col.n_assets"),
          t("col.n_frame_rows"),
          t("col.n_window_rows"),
        ],
        cells.map((cell) => [
          t(`task.${cell.task}`),
          t(`side.${cell.side}`),
          num(cell.n_subjects),
          num(cell.n_events),
          num(cell.n_assets),
          num(cell.n_frame_rows),
          num(cell.n_window_rows),
        ]),
        ["", "", "num", "num", "num", "num", "num"],
      ),
    ),
  );
}

function boundaryPanel() {
  const population = view.data.population || {};
  return panel(
    t("cohort.boundary"),
    el(
      "ul",
      { style: { margin: "0 0 12px", paddingLeft: "18px", color: "var(--muted)", fontSize: "12.5px" } },
      ["angle", "projection", "scale", "sample"].map((key) =>
        el("li", {}, t(`cohort.boundary.${key}`)),
      ),
    ),
    el(
      "dl",
      { class: "kv" },
      el("dt", {}, t("cohort.estimand")),
      el("dd", {}, wrappable(view.data.estimand || "—")),
      el("dt", {}, t("cohort.excluded")),
      el(
        "dd",
        {},
        wrappable(
          (view.data.columns_excluded || [])
            .map((column) => `${column.column} (${column.reason})`)
            .join(", ") || t("common.none"),
        ),
      ),
      // The census names its own population keys; a key with no gloss shows
      // verbatim rather than under an invented translation.
      ...Object.entries(population).flatMap(([key, value]) => [
        el("dt", {}, t(`cohort.pop.${key}`) === `cohort.pop.${key}` ? key : t(`cohort.pop.${key}`)),
        el("dd", {}, num(value)),
      ]),
    ),
  );
}

function featureOptions(lang) {
  const term = view.search.toLowerCase();
  return view.data.features
    .filter((feature) => feature.level === view.level)
    .filter(
      (feature) =>
        !term ||
        feature.feature.toLowerCase().includes(term) ||
        (feature.en || "").toLowerCase().includes(term) ||
        (feature.ja || "").includes(view.search),
    )
    .map((feature) =>
      el(
        "option",
        { value: feature.key, selected: view.feature && feature.key === view.feature.key },
        featureLabel(feature, lang),
      ),
    );
}

function draw(lang) {
  const feature = view.feature;
  const rows = rowsFor(feature);
  document.getElementById("cohort-body").replaceChildren(
    panel(
      `${featureLabel(feature, lang)} — ${t("cohort.chart")}`,
      el(
        "div",
        {},
        el("span", { class: "chip" }, feature.feature),
        el("span", { class: "chip" }, `${t("cohort.unit")}: ${feature.unit || "—"}`),
        el(
          "span",
          { class: "chip" },
          `${t("cohort.range")}: ${
            feature.range ? feature.range.map((bound) => (bound === null ? "∞" : bound)).join(" … ") : "—"
          }`,
        ),
      ),
      distributionChart(rows, feature),
      el("p", { class: "note" }, t("cohort.chart_note")),
      // 12 cells is the whole cohort — a vertical cap would hide the last two.
      el(
        "div",
        { class: "scroll", style: { marginTop: "10px", maxHeight: "none" } },
        statisticsTable(rows),
      ),
    ),
    el(
      "div",
      { class: "grid", style: { gridTemplateColumns: "repeat(auto-fit, minmax(340px, 1fr))", marginTop: "14px" } },
      panel(t("cohort.dispersion"), dispersionChart(rows), el("p", { class: "note" }, t("cohort.dispersion_note"))),
      cellsPanel(),
      boundaryPanel(),
    ),
  );
}

export async function renderCohort(root) {
  const data = await json("/api/cohort");
  view.data = data;
  const lang = document.documentElement.lang;
  if (!data.available || !data.features.length) {
    root.replaceChildren(el("div", { class: "banner" }, `${t("common.absent")}: cohort`));
    return;
  }
  view.feature = data.features.find((feature) => feature.level === view.level) || data.features[0];

  const select = el("select", {
    id: "feature-select",
    size: 1,
    onchange: (event) => {
      view.feature = data.features.find((feature) => feature.key === event.target.value);
      draw(lang);
    },
  });
  const refresh = () => {
    select.replaceChildren(...featureOptions(lang));
    const options = view.data.features.filter((feature) => feature.level === view.level);
    if (!options.some((feature) => view.feature && feature.key === view.feature.key)) {
      view.feature = options[0];
      select.value = view.feature.key;
    }
    draw(lang);
  };

  const levels = el(
    "div",
    { class: "segmented" },
    ["frame", "window"].map((level) =>
      el(
        "button",
        {
          class: view.level === level ? "on" : "",
          type: "button",
          onclick: (event) => {
            view.level = level;
            for (const button of event.currentTarget.parentElement.children) {
              button.classList.remove("on");
            }
            event.currentTarget.classList.add("on");
            refresh();
          },
        },
        t(`cohort.level_${level}`),
      ),
    ),
  );

  root.replaceChildren(
    el(
      "div",
      { class: "controls", style: { marginBottom: "14px" } },
      el("label", { class: "field" }, t("cohort.level"), levels),
      el(
        "label",
        { class: "field", style: { flex: "1 1 360px" } },
        t("cohort.feature"),
        select,
      ),
      el(
        "label",
        { class: "field" },
        t("cohort.search"),
        el("input", {
          type: "search",
          oninput: (event) => {
            view.search = event.target.value;
            select.replaceChildren(...featureOptions(lang));
          },
        }),
      ),
    ),
    el("div", { id: "cohort-body" }),
  );
  select.replaceChildren(...featureOptions(lang));
  draw(lang);
}
