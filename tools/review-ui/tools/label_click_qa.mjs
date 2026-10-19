// Labelling page click map: places all 14 points on a synthetic 1920x1080 frame through the
// two-step zoom and checks the stored source-pixel coordinates. Capture-free: serve a repo whose
// `.scratch/bench/user/` holds synthetic jobs + frames (`python -m review_ui --repo <dir>`).
// Usage: node tools/label_click_qa.mjs http://127.0.0.1:<port>   (job 0 must be the 1920x1080 frame)
import { existsSync, readdirSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

const BASE = process.argv[2] || "http://127.0.0.1:8791";
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

const browser = await chromium.launch({ executablePath: await binaryPath(), args: BASE_ARGS });
const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });
await page.goto(`${BASE}/label`);
await page.waitForFunction(() => document.getElementById("frame").width > 0);
const targets = Array.from({ length: 14 }, (_, i) => [137.3 + 113 * i, 101.7 + 61 * i]);
// Content box (inside the border): the canvas bitmap fills exactly this rectangle.
const box = await page.evaluate(() => {
  const el = document.getElementById("frame");
  const r = el.getBoundingClientRect();
  return { x: r.left + el.clientLeft, y: r.top + el.clientTop, width: el.clientWidth };
});
const dims = await page.evaluate(() => ({ w: document.getElementById("frame").width }));
const scale = box.width / 1920;
let worst = 0;
for (const [x, y] of targets) {
  // First click 9 source px off target; the zoom opens around it.
  await page.mouse.click(box.x + (x + 9) * scale, box.y + (y - 7) * scale);
  // In the zoom (content box, 360 bitmap px), the target sits at centre + 3 x (target - focus).
  const z = await page.evaluate(() => {
    const el = document.getElementById("loupe");
    const r = el.getBoundingClientRect();
    return { x: r.left + el.clientLeft, y: r.top + el.clientTop, w: el.clientWidth,
      fx: Number(el.dataset.fx), fy: Number(el.dataset.fy) };
  });
  const k = z.w / 360;
  await page.mouse.click(z.x + k * (180 + 3 * (x - z.fx)), z.y + k * (180 + 3 * (y - z.fy)));
}
await page.keyboard.press("Enter");
await page.waitForTimeout(500);
const answer = await (await fetch(`${BASE}/api/label/0/answer`)).json();
const keys = ["L.hand.wrist", "L.hand.thumb_tip", "L.hand.index_mcp", "L.hand.index_tip",
  "L.hand.little_tip", "L.body.shoulder", "L.body.elbow", "R.hand.wrist", "R.hand.thumb_tip",
  "R.hand.index_mcp", "R.hand.index_tip", "R.hand.little_tip", "R.body.shoulder", "R.body.elbow"];
keys.forEach((k, i) => {
  const p = answer.points[k];
  const err = Math.hypot(p.x - targets[i][0], p.y - targets[i][1]);
  worst = Math.max(worst, err);
  if (process.env.QA_VERBOSE) console.log(k, (p.x - targets[i][0]).toFixed(2), (p.y - targets[i][1]).toFixed(2));
});
const wrists = ["L", "R"].every((s) => answer.points[`${s}.body.wrist`].x === answer.points[`${s}.hand.wrist`].x);
console.log(`label-click: frame px ${dims.w}/1920 shown, 14 points, worst error ${worst.toFixed(2)} source px, wrist shared ${wrists}`);
await browser.close();
process.exit(worst <= 0.5 && wrists ? 0 : 1);
