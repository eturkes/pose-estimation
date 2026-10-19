// Overlay landmark->pixel map: every drawn landmark lands where the clip's own pixel sits on
// the stage. Fully synthetic — /api/clips + /api/clip/*/landmarks are routed to fabricated
// clips, so no recording is loaded, decoded or drawn and the run needs no published tree.
//
//   python -m review_ui --port <p> --repo <empty dir>   # any repo works; the API is routed
//   node tools/review-ui/tools/overlay_map_qa.mjs http://127.0.0.1:<p>
//
// Expected position = clip pixel (u, v) where the video element draws it: `object-fit: contain`
// (read from the computed style) inside the stage content box, k = min(w/W, h/H), centred —
// independent of the overlay's single max(W, H) · w/W factor. Measured = alpha-weighted centroid
// of each drawn dot in the backing store, mapped back through the canvas rect. Max deviation
// must stay < 1 CSS px over 2 aspects × 2 device ratios. Each seed serves a mutated player.js
// through the route (no file is edited) and must red at least one row: `axis` = fit from
// clip.height, `offby1` = a 1 px translate, `legacy` = the
// pre-fix w/W factor (reds the sub-pixel rows alone).

import { existsSync, readdirSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

const BASE = process.argv[2] || "http://127.0.0.1:8791";
const TOLERANCE = 1;
// The letterbox residual the pre-fix w/W mapping left (0.87 px); `legacy` replays it.
const SUBPIXEL = 0.25;

function resolveGlobal(pkg) {
  const globals = join(homedir(), ".local/share/pnpm/global");
  for (const dir of readdirSync(globals).sort((a, b) => Number(b) - Number(a))) {
    const entry = join(globals, dir, "node_modules", pkg);
    if (existsSync(entry)) return pathToFileURL(entry).href;
  }
  throw new Error(`${pkg} missing from ${globals}`);
}

const { BASE_ARGS, binaryPath } = await import(`${resolveGlobal("chromiumfish")}/dist/index.js`);
const pw = await import(`${resolveGlobal("playwright-core")}/index.mjs`);
const chromium = pw.chromium ?? pw.default.chromium;

const topology = await (await fetch(`${BASE}/static/topology.json`)).json();
const playerSource = await (await fetch(`${BASE}/static/player.js`)).text();
const NAMES = topology.body.names;

// Body points no segment or chain joins, so every visible point draws as a lone dot.
const joined = new Set();
for (const s of topology.body.segments) joined.add(`${s.points[0]}-${s.points[1]}`);
for (const c of topology.body.chains) {
  for (let i = 1; i < c.points.length; i += 1) joined.add(`${c.points[i - 1]}-${c.points[i]}`);
}
const adjacent = (a, b) => joined.has(`${a}-${b}`) || joined.has(`${b}-${a}`);
const lone = [];
for (let i = 0; i < NAMES.length && lone.length < 5; i += 1) {
  if (lone.every((j) => !adjacent(i, j))) lone.push(i);
}
// Corners, centre, interior: fractions of the clip frame.
const SPOTS = [[0.04, 0.05], [0.96, 0.94], [0.5, 0.5], [0.3, 0.72], [0.81, 0.18]];

const CLIPS = [
  { width: 1280, height: 720, view: "landscape" },
  { width: 720, height: 1280, view: "portrait" },
].map((c, i) => ({
  event_id: `synthetic-${i}`, camera_name: "cam", subject_ordinal: 1, task: "qa", side: "L",
  view: c.view, run_index: 1, n_cameras: 1, view_conflict: "", width: c.width, height: c.height,
  fps: 30, frames: 1, duration_s: 0.033, rotation_deg: 0, codec: "none", disposition: "ok",
  has_video: false, has_landmarks: true, family_no: i + 1,
}));

function series(clip) {
  const scale = Math.max(clip.width, clip.height);
  const body = new Array(NAMES.length * 3).fill(null);
  lone.forEach((index, k) => {
    body[index * 3] = (SPOTS[k][0] * clip.width) / scale;
    body[index * 3 + 1] = (SPOTS[k][1] * clip.height) / scale;
    body[index * 3 + 2] = 0.9;
  });
  return {
    clip, frames: 1, frame_idx: [0], timestamp_sec: [0], persons_max: 1, scale,
    body_names: NAMES, hand_points: 21, body: [body],
    hand_left: [new Array(63).fill(null)], hand_right: [new Array(63).fill(null)],
  };
}

const FIT = "const fit = Math.min(width / frameWidth, height / frameHeight);";
const SHIFT = "ratio * offsetX, ratio * offsetY);";
const SEEDS = {
  axis: [[FIT, "const fit = width / frameHeight;"]],
  offby1: [[SHIFT, "ratio * (offsetX + 1), ratio * (offsetY + 1));"]],
  legacy: [[FIT, "const fit = width / frameWidth;"], [SHIFT, "0, 0);"]],
};

async function measure(browser, ratio, seed) {
  const context = await browser.newContext({ viewport: { width: 1400, height: 950 } });
  // This build pins devicePixelRatio to 1 whatever the context says (→ review-ui.md drift).
  await context.addInitScript((r) => {
    Object.defineProperty(window, "devicePixelRatio", { get: () => r });
  }, ratio);
  await context.route("**/api/clips", (route) => route.fulfill({ json: { clips: CLIPS } }));
  await context.route("**/api/clip/*/*/landmarks", (route) => {
    const id = decodeURIComponent(new URL(route.request().url()).pathname.split("/")[3]);
    route.fulfill({ json: series(CLIPS.find((c) => c.event_id === id)) });
  });
  if (seed) {
    let body = playerSource;
    for (const [from, to] of SEEDS[seed]) {
      if (!body.includes(from)) throw new Error(`seed ${seed}: anchor absent from player.js`);
      body = body.replace(from, to);
    }
    await context.route("**/static/player.js", (route) =>
      route.fulfill({ body, contentType: "text/javascript" }));
  }
  const page = await context.newPage();
  await page.goto(`${BASE}/?lang=en#player`, { waitUntil: "load" });
  await page.waitForSelector(".clip-row", { timeout: 15000 });
  const rows = [];
  for (let index = 0; index < CLIPS.length; index += 1) {
    const clip = CLIPS[index];
    await page.locator(".clip-row").nth(index).click();
    await page.waitForFunction((n) => {
      const c = document.getElementById("overlay");
      return c && c.width > 0 && document.querySelectorAll(".clip-row")[n]?.classList.contains("on");
    }, index);
    await page.waitForTimeout(300);
    const got = await page.evaluate(([spots, W, H]) => {
      const canvas = document.getElementById("overlay");
      const stage = document.getElementById("stage");
      const style = getComputedStyle(stage);
      const s = stage.getBoundingClientRect();
      const box = {
        left: s.left + parseFloat(style.borderLeftWidth),
        top: s.top + parseFloat(style.borderTopWidth),
        width: stage.clientWidth,
        height: stage.clientHeight,
      };
      const r = canvas.getBoundingClientRect();
      const { data } = canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height);
      const sums = spots.map(() => [0, 0, 0]);
      const kx = r.width / canvas.width;
      const ky = r.height / canvas.height;
      const fit = getComputedStyle(document.getElementById("clip-video")).objectFit;
      const k = Math.min(box.width / W, box.height / H);
      const ox = box.left + (box.width - W * k) / 2;
      const oy = box.top + (box.height - H * k) / 2;
      const expected = spots.map(([fx, fy]) => [ox + fx * W * k, oy + fy * H * k]);
      for (let y = 0; y < canvas.height; y += 1) {
        for (let x = 0; x < canvas.width; x += 1) {
          const a = data[(y * canvas.width + x) * 4 + 3];
          if (!a) continue;
          const px = r.left + (x + 0.5) * kx;
          const py = r.top + (y + 0.5) * ky;
          let best = -1;
          let dist = 25;
          expected.forEach(([ex, ey], k) => {
            const d = Math.hypot(px - ex, py - ey);
            if (d < dist) [best, dist] = [k, d];
          });
          if (best < 0) continue;
          sums[best][0] += a * px;
          sums[best][1] += a * py;
          sums[best][2] += a;
        }
      }
      const deviations = sums.map(([sx, sy, w], k) =>
        w ? Math.hypot(sx / w - expected[k][0], sy / w - expected[k][1]) : Infinity);
      return { deviations, fit, aspect: box.width / box.height, backing: canvas.width / r.width };
    }, [SPOTS, clip.width, clip.height]);
    rows.push({ clip, ratio, ...got });
  }
  await context.close();
  return rows;
}

function grade(rows, print) {
  let failed = 0;
  const line = (name, ok, detail) => {
    if (!ok) failed += 1;
    if (print) console.log(`${ok ? "PASS" : "FAIL"}  ${name}  ${detail}`);
  };
  for (const { clip, ratio, deviations, fit, aspect, backing } of rows) {
    const tag = `${clip.view} ${clip.width}x${clip.height} dpr ${ratio}`;
    const want = clip.width / clip.height;
    line(`${tag} video object-fit contain`, fit === "contain", fit);
    line(`${tag} stage aspect`, Math.abs(aspect / want - 1) < 0.01, `${aspect.toFixed(4)} vs ${want.toFixed(4)}`);
    line(`${tag} backing ratio`, Math.abs(backing - ratio) < 0.01, `${backing.toFixed(3)}`);
    const worst = Math.max(...deviations);
    line(`${tag} max landmark deviation < ${TOLERANCE} px`, worst < TOLERANCE,
      `${deviations.length} dots, max ${worst.toFixed(3)} px`);
    line(`${tag} max landmark deviation < ${SUBPIXEL} px`, worst < SUBPIXEL, `max ${worst.toFixed(3)} px`);
  }
  return failed;
}

const browser = await chromium.launch({ executablePath: await binaryPath(), args: BASE_ARGS });
let failed = 0;
const baseline = [];
for (const ratio of [1, 2]) baseline.push(...(await measure(browser, ratio, null)));
failed += grade(baseline, true);
for (const seed of Object.keys(SEEDS)) {
  const rows = [];
  for (const ratio of [1, 2]) rows.push(...(await measure(browser, ratio, seed)));
  const red = grade(rows, process.argv.includes("--verbose"));
  const worst = Math.max(...rows.flatMap((r) => r.deviations));
  const fires = red > 0;
  if (!fires) failed += 1;
  console.log(`${fires ? "PASS" : "FAIL"}  seed ${seed} fires  ${red} rows red, max ${worst.toFixed(3)} px`);
}
await browser.close();
console.log(failed ? `FAIL ${failed}` : "PASS");
process.exit(failed ? 1 : 0);
