/* Corpus census: what the corpus is, how uncontrolled it is, and which axes the
   pipeline measured.  Everything on this page comes from a publisher summary
   that is aggregates-only by its own contract. */

import {
  chart,
  el,
  json,
  num,
  palette,
  panel,
  t,
  table,
  token,
  withAlpha,
  wrappable,
} from "/static/app.js";

function tiles(data) {
  return el(
    "div",
    { class: "tiles" },
    data.headline.map((tile) =>
      el(
        "div",
        { class: "tile" },
        el("div", { class: "value" }, num(tile.value, Number.isInteger(tile.value) ? 0 : 2)),
        el("div", { class: "label" }, t(`census.${tile.key}`)),
        el("div", { class: "src" }, tile.source),
      ),
    ),
  );
}

function barChart(entries, colour, horizontal = false, height = 220) {
  const node = el("div", { class: "chart", style: { height: `${height}px` } });
  const labels = entries.map(([label]) => label);
  const values = entries.map(([, value]) => value);
  const trace = horizontal
    ? { type: "bar", orientation: "h", x: values, y: labels, marker: { color: colour } }
    : { type: "bar", x: labels, y: values, marker: { color: colour } };
  requestAnimationFrame(() =>
    chart(node, [trace], {
      margin: horizontal ? { l: 8, r: 40, t: 8, b: 32 } : { l: 48, r: 12, t: 8, b: 40 },
      yaxis: horizontal ? { automargin: true, type: "category" } : {},
      showlegend: false,
    }),
  );
  return node;
}

function shapesPanel(data) {
  const rows = data.shapes.map((shape) => [
    shape.resolution,
    shape.fps,
    shape.codec,
    `${shape.rotation}°`,
    num(shape.count),
  ]);
  // Pooled over frame rate: 25 distinct labels differ mostly by a 29.98x fps
  // reading, which crowds the axis without separating anything a reviewer acts on.
  const pooled = new Map();
  for (const shape of data.shapes) {
    const key = `${shape.resolution} ${shape.codec} rot${shape.rotation}`;
    pooled.set(key, (pooled.get(key) || 0) + shape.count);
  }
  return panel(
    t("census.shapes"),
    barChart(
      [...pooled.entries()].sort((a, b) => b[1] - a[1]),
      palette().accent,
      true,
      40 + 26 * pooled.size,
    ),
    el("p", { class: "note" }, t("census.shapes.chart")),
    el("p", { class: "note" }, t("census.shapes.note")),
    el(
      "div",
      { class: "scroll" },
      table(
        [t("col.resolution"), t("col.fps"), t("col.codec"), t("col.rotation"), t("common.count")],
        rows,
        ["", "num", "", "num", "num"],
      ),
    ),
  );
}

function rotationPanel(data) {
  const views = Object.keys(data.rotation_by_view);
  const angles = [...new Set(views.flatMap((view) => Object.keys(data.rotation_by_view[view])))];
  angles.sort((a, b) => Number(a) - Number(b));
  const node = el("div", { class: "chart", style: { height: "220px" } });
  const { accent, amber, good, warn } = palette();
  const traces = angles.map((angle, index) => ({
    type: "bar",
    name: `${angle}°`,
    x: views.map((view) => t(`view.${view}`)),
    y: views.map((view) => data.rotation_by_view[view][angle] || 0),
    marker: { color: [accent, amber, good, warn][index % 4] },
  }));
  requestAnimationFrame(() => chart(node, traces, { barmode: "stack" }));
  return panel(t("census.rotation"), node);
}

function countPanel(title, counts, colour, horizontal = false, glossed = false) {
  const entries = Object.entries(counts || {}).sort((a, b) => b[1] - a[1]);
  if (!entries.length) return null;
  const labelled = entries.map(([key, value]) => [glossed ? token(key) : key, value]);
  return panel(title, barChart(labelled, colour, horizontal, horizontal ? 40 + 26 * entries.length : 200));
}

function durationPanel(data) {
  const stats = data.duration_s || {};
  if (!stats.median) return null;
  const node = el("div", { class: "chart", style: { height: "150px" } });
  const { accent } = palette();
  requestAnimationFrame(() =>
    chart(
      node,
      [
        {
          type: "box",
          orientation: "h",
          y: [""],
          q1: [stats.p25],
          median: [stats.median],
          q3: [stats.p75],
          lowerfence: [stats.min],
          upperfence: [stats.p95],
          marker: { color: accent },
          line: { color: accent },
          fillcolor: withAlpha(accent, 0.18),
          name: "",
        },
      ],
      { margin: { l: 12, r: 16, t: 8, b: 34 }, showlegend: false, xaxis: { title: { text: "s" } } },
    ),
  );
  return panel(
    t("census.duration"),
    node,
    el("p", { class: "note" }, t("census.duration_note")),
    el(
      "dl",
      { class: "kv" },
      ["min", "p25", "median", "p75", "p95", "max"].flatMap((key) => [
        el("dt", {}, key),
        el("dd", {}, num(stats[key], 2)),
      ]),
    ),
  );
}

function syncPanel(data) {
  const groups = [
    [t("census.sync.events"), data.sync_status],
    [t("census.sync.cameras"), data.offset_status],
    [t("census.sync.pairs"), data.pair_status],
  ].filter(([, counts]) => counts && Object.keys(counts).length);
  const statuses = [...new Set(groups.flatMap(([, counts]) => Object.keys(counts)))];
  const { series } = palette();
  const node = el("div", { class: "chart", style: { height: "200px" } });
  const traces = statuses.map((status, index) => ({
    type: "bar",
    orientation: "h",
    name: token(status),
    y: groups.map(([label]) => label),
    x: groups.map(([, counts]) => counts[status] || 0),
    marker: { color: series[index % series.length] },
  }));
  requestAnimationFrame(() =>
    chart(node, traces, { barmode: "stack", margin: { l: 8, r: 16, t: 8, b: 30 } }),
  );
  return panel(t("census.sync"), node);
}

function flagsPanel(data) {
  const rows = Object.entries(data.qc_flags || {})
    .sort((a, b) => b[1] - a[1])
    .map(([flag, count]) => [
      el("span", { class: "mono" }, flag),
      token(flag),
      num(count),
    ]);
  return panel(
    t("census.qc"),
    table(["", "", t("common.count")], rows, ["", "", "num"]),
    el(
      "div",
      { style: { marginTop: "10px" } },
      el("span", { class: "chip good" }, `${t("census.axes")}: ${(data.measured_axes || []).join(" · ")}`),
      el(
        "span",
        { class: "chip warn" },
        `${t("census.axes_unmeasured")}: ${(data.unmeasured_axes || []).join(" · ") || t("common.none")}`,
      ),
    ),
  );
}

function rulingPanel(data) {
  const ruling = data.ruling || {};
  const order = [
    "ruling_grain",
    "recovery_status",
    "reason",
    "transfer_status",
    "keypoint_source",
    "intrinsics_basis",
    "image_height_px",
    "unrun_arm",
    "unrun_arm_status",
  ];
  const items = order
    .filter((key) => ruling[key] !== undefined)
    .flatMap((key) => [
      el("dt", {}, key),
      el(
        "dd",
        {},
        wrappable(`${ruling[key]}${token(ruling[key]) !== ruling[key] ? ` — ${token(ruling[key])}` : ""}`),
      ),
    ]);
  return panel(
    t("census.ruling"),
    el("dl", { class: "kv" }, items),
    el(
      "details",
      { style: { marginTop: "10px" } },
      el("summary", { class: "note" }, `${t("census.claims")} (${(data.claims || []).length})`),
      el(
        "ul",
        { style: { fontSize: "12px", color: "var(--muted)", paddingLeft: "18px" } },
        (data.claims || []).map((claim) => el("li", {}, claim)),
      ),
    ),
  );
}

function verdictPanel(data) {
  const verdicts = Object.entries(data.run_verdicts || {});
  const manifest = Object.entries(data.run_manifest_census || {}).filter(([, count]) => count);
  return panel(
    t("census.verdicts"),
    el(
      "div",
      {},
      verdicts.map(([name, ok]) =>
        el("span", { class: `chip ${ok ? "good" : "warn"}` }, `${ok ? "✓" : "✗"} ${name}`),
      ),
    ),
    el("h2", { style: { marginTop: "14px" } }, t("census.manifest")),
    table(
      ["", t("common.count")],
      manifest.map(([code, count]) => [token(code), num(count)]),
      ["", "num"],
    ),
    el(
      "dl",
      { class: "kv", style: { marginTop: "12px" } },
      Object.entries(data.run_configuration || {}).flatMap(([key, value]) => [
        el("dt", {}, key),
        el("dd", {}, String(value)),
      ]),
    ),
  );
}

function reasonPanel(data) {
  const entries = Object.entries(data.reason_codes || {});
  const nonzero = entries.filter(([, count]) => count > 0).sort((a, b) => b[1] - a[1]);
  return panel(
    t("census.reasons"),
    table(
      ["", t("common.count")],
      nonzero.map(([code, count]) => [el("span", { class: "mono" }, code), num(count)]),
      ["", "num"],
    ),
    el(
      "p",
      { class: "note" },
      `${t("census.zero_hidden")} — ${entries.length - nonzero.length} / ${entries.length}`,
    ),
  );
}

function provenancePanel(data) {
  return panel(
    t("census.provenance"),
    el(
      "dl",
      { class: "kv" },
      Object.entries(data.provenance || {})
        .filter(([, value]) => value)
        .flatMap(([key, value]) => [el("dt", {}, key), el("dd", {}, String(value))]),
    ),
  );
}

export async function renderCensus(root) {
  const data = await json("/api/census");
  const { accent, amber, good } = palette();
  const missing = Object.entries(data.available)
    .filter(([, present]) => !present)
    .map(([name]) => name);

  root.replaceChildren(
    missing.length
      ? el("div", { class: "banner" }, `${t("common.absent")}: ${missing.join(", ")}`)
      : el(
          "div",
          { class: "banner info" },
          `${t("census.sources")}: ${Object.keys(data.available).join(" · ")}`,
        ),
    tiles(data),
    el(
      "div",
      { class: "grid", style: { gridTemplateColumns: "repeat(auto-fit, minmax(340px, 1fr))" } },
      shapesPanel(data),
      rotationPanel(data),
      countPanel(t("census.codec"), data.codec, amber),
      countPanel(t("census.device"), data.device_config, good, true),
      countPanel(t("census.views"), data.view_coverage, accent),
      countPanel(t("census.cameras"), data.cameras_per_event, accent),
      durationPanel(data),
      syncPanel(data),
      flagsPanel(data),
      rulingPanel(data),
      verdictPanel(data),
      reasonPanel(data),
      provenancePanel(data),
    ),
  );
}
