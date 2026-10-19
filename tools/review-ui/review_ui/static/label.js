// Labelling page: one frame at a time, 14 points (per side: 5 hand joints, shoulder, elbow; the
// one wrist click serves the hand and the arm).
// A point is a click in a 3x zoom of the region around a first click; o / f / u record a hidden,
// outside-frame or unsure joint instead.
"use strict";

const SIDES = [["L", "LEFT"], ["R", "RIGHT"]];
const HAND = [["wrist", "wrist"], ["thumb_tip", "thumb tip"], ["index_mcp", "index knuckle"],
  ["index_tip", "index tip"], ["little_tip", "little finger tip"]];
const BODY = [["shoulder", "shoulder"], ["elbow", "elbow"]];
const STEPS = SIDES.flatMap(([side, word]) => [
  ...HAND.map(([name, text]) => ({ key: `${side}.hand.${name}`, side: word, text, short: `${side}${name[0]}` })),
  ...BODY.map(([name, text]) => ({ key: `${side}.body.${name}`, side: word, text, short: `${side}-${name[0]}` })),
]);
const ZOOM = 3;
const STATE_KEYS = { o: "occ", f: "oof", u: "?", "?": "?" };

const frame = document.getElementById("frame");
const loupe = document.getElementById("loupe");
const ctx = frame.getContext("2d");
const lctx = loupe.getContext("2d");
const image = new Image();
let summary = { n: 0, done: [] };
let job = 0;
let step = 0;
let points = {};
let scale = 1;
let focus = null; // frame px of the first click while the zoom is open

function nextUndone(from) {
  for (let i = 0; i < summary.n; i++) {
    const j = (from + i) % summary.n;
    if (!summary.done.includes(j)) return j;
  }
  return from;
}

function draw() {
  ctx.drawImage(image, 0, 0, frame.width, frame.height);
  ctx.lineWidth = 2;
  ctx.font = "13px system-ui";
  for (const s of STEPS) {
    const p = points[s.key];
    if (!p || p.state !== "vis") continue;
    const x = p.x * scale, y = p.y * scale;
    ctx.strokeStyle = "#ffffff";
    ctx.beginPath(); ctx.arc(x, y, 5, 0, 2 * Math.PI); ctx.stroke();
    ctx.strokeStyle = "#d1235f";
    ctx.beginPath(); ctx.arc(x, y, 3.5, 0, 2 * Math.PI); ctx.stroke();
    ctx.fillStyle = "#ffffff";
    ctx.fillText(s.short, x + 7, y - 6);
  }
  renderSteps();
}

function renderSteps() {
  const list = document.getElementById("steps");
  list.replaceChildren(...STEPS.map((s, i) => {
    const li = document.createElement("li");
    const p = points[s.key];
    const state = !p ? "" : p.state === "vis" ? " ✓" : ` (${{ occ: "hidden", oof: "outside", "?": "unsure" }[p.state]})`;
    li.textContent = `${s.side.toLowerCase()} ${s.text}${state}`;
    if (i === step) li.className = "now";
    return li;
  }));
  const prompt = document.getElementById("prompt");
  if (step < STEPS.length) {
    const s = STEPS[step];
    prompt.innerHTML = "";
    prompt.append("Patient's ");
    const side = document.createElement("span");
    side.className = "side";
    side.textContent = s.side;
    prompt.append(side, ` ${s.text}`);
  } else {
    prompt.textContent = "All points set — press Enter to save.";
  }
  document.getElementById("status").textContent =
    `Frame ${job + 1} of ${summary.n} · ${summary.done.length} saved`;
}

function closeLoupe() { loupe.style.display = "none"; focus = null; }

function openLoupe(fx, fy) {
  focus = { x: fx, y: fy };
  loupe.dataset.fx = String(fx); // read by tools/label_click_qa.mjs
  loupe.dataset.fy = String(fy);
  const half = loupe.width / (2 * ZOOM);
  lctx.imageSmoothingEnabled = false;
  lctx.fillStyle = "#808080";
  lctx.fillRect(0, 0, loupe.width, loupe.height);
  lctx.drawImage(image, fx - half, fy - half, 2 * half, 2 * half, 0, 0, loupe.width, loupe.height);
  lctx.strokeStyle = "rgba(255,255,255,0.6)";
  lctx.beginPath();
  lctx.moveTo(loupe.width / 2, 0); lctx.lineTo(loupe.width / 2, loupe.height);
  lctx.moveTo(0, loupe.height / 2); lctx.lineTo(loupe.width, loupe.height / 2);
  lctx.stroke();
  const left = Math.min(Math.max(fx * scale - loupe.width / 2, 0), frame.width - loupe.width);
  const top = Math.min(Math.max(fy * scale - loupe.height / 2, 0), frame.height - loupe.height);
  loupe.style.left = `${Math.max(left, 0)}px`;
  loupe.style.top = `${Math.max(top, 0)}px`;
  loupe.style.display = "block";
}

function setPoint(entry) {
  if (step >= STEPS.length) return;
  points[STEPS[step].key] = entry;
  step += 1;
  closeLoupe();
  draw();
}

// offsetX/Y + clientWidth measure the content box: the canvas bitmap fills it, the border does not.
frame.addEventListener("click", (event) => {
  const fx = (event.offsetX * frame.width) / frame.clientWidth / scale;
  const fy = (event.offsetY * frame.height) / frame.clientHeight / scale;
  openLoupe(fx, fy);
});

loupe.addEventListener("click", (event) => {
  if (!focus) return;
  const dx = (event.offsetX * loupe.width) / loupe.clientWidth - loupe.width / 2;
  const dy = (event.offsetY * loupe.height) / loupe.clientHeight - loupe.height / 2;
  const x = Math.min(Math.max(focus.x + dx / ZOOM, 0), image.naturalWidth - 0.5);
  const y = Math.min(Math.max(focus.y + dy / ZOOM, 0), image.naturalHeight - 0.5);
  setPoint({ state: "vis", x: Math.round(x * 10) / 10, y: Math.round(y * 10) / 10 });
});

document.addEventListener("keydown", (event) => {
  if (event.target instanceof HTMLInputElement) return;
  if (event.key === "Escape") { closeLoupe(); return; }
  if (event.key === "Backspace") {
    event.preventDefault();
    step = Math.max(0, step - 1);
    delete points[STEPS[step].key];
    closeLoupe();
    draw();
    return;
  }
  if (event.key === "Enter") { event.preventDefault(); save(); return; }
  if (event.key === "s") { const box = document.getElementById("setup"); box.checked = !box.checked; return; }
  const state = STATE_KEYS[event.key];
  if (state) setPoint({ state });
});

async function load(target) {
  job = target;
  const answer = await (await fetch(`/api/label/${job}/answer`)).json();
  points = answer ? answer.points : {};
  step = answer ? STEPS.length : 0;
  document.getElementById("setup").checked = Boolean(answer && answer.task === "setup");
  image.onload = () => {
    scale = Math.min(1, (window.innerWidth - 40) / image.naturalWidth, (window.innerHeight - 160) / image.naturalHeight);
    frame.width = Math.round(image.naturalWidth * scale);
    frame.height = Math.round(image.naturalHeight * scale);
    scale = frame.width / image.naturalWidth; // the bitmap's own ratio, not the pre-rounding one
    closeLoupe();
    draw();
    frame.focus();
  };
  image.src = `/api/label/${job}/frame`;
}

async function save() {
  if (step < STEPS.length) {
    document.getElementById("prompt").textContent = `Finish all ${STEPS.length} points first (use u for unsure).`;
    return;
  }
  const body = {
    points,
    task: document.getElementById("setup").checked ? "setup" : "task",
    width: image.naturalWidth,
    height: image.naturalHeight,
  };
  const response = await fetch(`/api/label/${job}/answer`, {
    method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body),
  });
  if (!response.ok) {
    document.getElementById("prompt").textContent = "Saving failed — reload the page and try again.";
    return;
  }
  summary = await response.json();
  load(nextUndone(job + 1));
}

document.getElementById("save").addEventListener("click", save);
document.getElementById("prev").addEventListener("click", () => load((job - 1 + summary.n) % summary.n));

(async () => {
  summary = await (await fetch("/api/label")).json();
  if (!summary.n) {
    document.getElementById("status").textContent =
      "No frames to label: run scripts/bench.py user-sample first.";
    return;
  }
  load(nextUndone(0));
})();
