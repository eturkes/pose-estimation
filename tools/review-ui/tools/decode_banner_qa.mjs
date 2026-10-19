// HEVC decode failure: a clip the browser cannot decode shows the `player.decode_failed`
// banner, hides the video and keeps the overlay on the rAF clock; a decodable clip shows none.
// Fully synthetic — /api/clips, /landmarks and /video are routed; both clips (HEVC `hvc1` +
// H.264 control, flat grey, 1 s) are encoded at run time by PyAV in the root `.venv`, so no
// media is committed and no recording is loaded.
//
//   python -m review_ui --port <p> --repo <empty dir>   # any repo works; the API is routed
//   node tools/review-ui/tools/decode_banner_qa.mjs http://127.0.0.1:<p>
//
// Grades the build it runs on: an HEVC row reports `decoder present` when this Chromium
// decodes HEVC, since the banner half then has no failure to show.

import { spawnSync } from "node:child_process";
import { existsSync, mkdtempSync, readdirSync, readFileSync, rmSync } from "node:fs";
import { homedir, tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const BASE = process.argv[2] || "http://127.0.0.1:8791";
const ROOT = join(dirname(fileURLToPath(import.meta.url)), "../../..");

function resolveGlobal(pkg) {
  const globals = join(homedir(), ".local/share/pnpm/global");
  for (const dir of readdirSync(globals).sort((a, b) => Number(b) - Number(a))) {
    const entry = join(globals, dir, "node_modules", pkg);
    if (existsSync(entry)) return pathToFileURL(entry).href;
  }
  throw new Error(`${pkg} missing from ${globals}`);
}

const ENCODE = `
import sys, av, numpy as np
for path, codec, tag in (("hevc.mp4", "libx265", "hvc1"), ("h264.mp4", "libx264", None)):
    with av.open(sys.argv[1] + "/" + path, "w", format="mp4") as out:
        stream = out.add_stream(codec, rate=30)
        stream.width, stream.height, stream.pix_fmt = 320, 180, "yuv420p"
        if tag:
            stream.codec_tag = tag
        if codec == "libx265":
            stream.options = {"x265-params": "log-level=error"}
        for i in range(30):
            frame = av.VideoFrame.from_ndarray(np.full((180, 320, 3), 96 + i, np.uint8), format="rgb24")
            out.mux(stream.encode(frame))
        out.mux(stream.encode())
`;
const work = mkdtempSync(join(tmpdir(), "decode-qa-"));
const encoded = spawnSync(join(ROOT, ".venv/bin/python"), ["-c", ENCODE, work], { encoding: "utf8" });
if (encoded.status !== 0) throw new Error(`encode failed: ${encoded.stderr}`);
const MEDIA = { hevc: readFileSync(join(work, "hevc.mp4")), h264: readFileSync(join(work, "h264.mp4")) };
rmSync(work, { recursive: true });

const strings = await (await fetch(`${BASE}/static/strings.json`)).json();
const topology = await (await fetch(`${BASE}/static/topology.json`)).json();
const NAMES = topology.body.names;
const CLIPS = [["hevc", "hvc1"], ["h264", "avc1"]].map(([id, codec], i) => ({
  event_id: `synthetic-${id}`, camera_name: "cam", subject_ordinal: 1, task: "qa", side: "L",
  view: id, run_index: 1, n_cameras: 1, view_conflict: "", width: 320, height: 180, fps: 30,
  frames: 30, duration_s: 1, rotation_deg: 0, codec, disposition: "ok", has_video: true,
  has_landmarks: true, family_no: i + 1,
}));
const series = (clip) => {
  const body = Array.from({ length: 30 }, (_, f) => {
    const row = new Array(NAMES.length * 3).fill(null);
    [row[0], row[1], row[2]] = [0.2 + f / 60, 0.3, 0.9];
    return row;
  });
  return {
    clip, frames: 30, frame_idx: [...Array(30).keys()],
    timestamp_sec: [...Array(30).keys()].map((f) => f / 30), persons_max: 1, scale: 320,
    body_names: NAMES, hand_points: 21, body,
    hand_left: body.map(() => new Array(63).fill(null)),
    hand_right: body.map(() => new Array(63).fill(null)),
  };
};

const { BASE_ARGS, binaryPath } = await import(`${resolveGlobal("chromiumfish")}/dist/index.js`);
const pw = await import(`${resolveGlobal("playwright-core")}/index.mjs`);
const chromium = pw.chromium ?? pw.default.chromium;
const playerSource = await (await fetch(`${BASE}/static/player.js`)).text();
// Each seed serves a mutated player.js through the route (no file is edited) and must red rows.
const SEEDS = {
  nobanner: ["banner.hidden = false;", ""],
  nohide: ['video.className = "hidden";\n', ""],
};
const message = strings["player.decode_failed"].en;
const browser = await chromium.launch({ executablePath: await binaryPath(), args: BASE_ARGS });

async function run(seed) {
  const context = await browser.newContext({ viewport: { width: 1400, height: 950 } });
  await context.route("**/api/clips", (route) => route.fulfill({ json: { clips: CLIPS } }));
  await context.route("**/api/clip/*/*/*", (route) => {
    const parts = new URL(route.request().url()).pathname.split("/");
    const clip = CLIPS.find((c) => c.event_id === decodeURIComponent(parts[3]));
    if (parts[5] === "landmarks") return route.fulfill({ json: series(clip) });
    return route.fulfill({ body: MEDIA[clip.view], contentType: "video/mp4" });
  });
  if (seed) {
    const [from, to] = SEEDS[seed];
    if (!playerSource.includes(from)) throw new Error(`seed ${seed}: anchor absent from player.js`);
    await context.route("**/static/player.js", (route) =>
      route.fulfill({ body: playerSource.replace(from, to), contentType: "text/javascript" }));
  }
  const page = await context.newPage();
  await page.goto(`${BASE}/?lang=en#player`, { waitUntil: "load" });
  await page.waitForSelector(".clip-row", { timeout: 15000 });
  const rows = [];
  const line = (name, ok, detail) => rows.push({ name, ok, detail });
  const probe = await page.evaluate(() => {
    const v = document.createElement("video");
    return { hevc: v.canPlayType('video/mp4; codecs="hvc1.1.6.L60.B0"'), h264: v.canPlayType('video/mp4; codecs="avc1.42E01E"') };
  });
  for (const [index, clip] of CLIPS.entries()) {
    await page.locator(".clip-row").nth(index).click();
    // Settled = the element either decoded a frame or raised its error.
    await page.waitForFunction((n) => {
      const v = document.getElementById("clip-video");
      const on = document.querySelectorAll(".clip-row")[n]?.classList.contains("on");
      return on && v && (v.error || v.readyState >= 2);
    }, index, { timeout: 15000 }).catch(() => {});
    const state = await page.evaluate(() => {
      const v = document.getElementById("clip-video");
      const b = document.getElementById("decode-note");
      return { error: v?.error?.code ?? null, ready: v?.readyState ?? -1, hidden: v?.className === "hidden",
               banner: b && !b.hidden ? b.textContent : null };
    });
    const tag = `${clip.view} (${clip.codec}, canPlayType "${probe[clip.view]}")`;
    if (clip.view === "h264") {
      line(`${tag} decodes`, state.error === null && state.ready >= 2, `error ${state.error}, readyState ${state.ready}`);
      line(`${tag} no banner`, state.banner === null, `${state.banner === null ? "hidden" : "shown"}`);
      continue;
    }
    if (state.error === null) {
      line(`${tag} decoder present — banner unexercised on this build`, false, `readyState ${state.ready}`);
      continue;
    }
    line(`${tag} decode fails`, true, `MediaError code ${state.error}`);
    line(`${tag} banner shown`, state.banner === `${message} — ${clip.codec}`, `${JSON.stringify(state.banner)}`);
    line(`${tag} video hidden`, state.hidden, `${state.hidden}`);
    await page.locator("#play").click();
    const before = await page.evaluate(() => document.querySelector(".readout")?.textContent ?? "");
    await page.waitForTimeout(600);
    const after = await page.evaluate(() => document.querySelector(".readout")?.textContent ?? "");
    await page.locator("#play").click();
    line(`${tag} overlay advances on the rAF clock`, before !== after, `readout moved ${before !== after}`);
  }
  await context.close();
  return rows;
}

let failed = 0;
for (const row of await run(null)) {
  if (!row.ok) failed += 1;
  console.log(`${row.ok ? "PASS" : "FAIL"}  ${row.name}  ${row.detail}`);
}
for (const seed of Object.keys(SEEDS)) {
  const red = (await run(seed)).filter((row) => !row.ok).map((row) => row.name.split(") ").pop());
  if (!red.length) failed += 1;
  console.log(`${red.length ? "PASS" : "FAIL"}  seed ${seed} fires  red: ${red.join(", ") || "none"}`);
}
await browser.close();
console.log(failed ? `FAIL ${failed}` : "PASS");
process.exit(failed ? 1 : 0);
