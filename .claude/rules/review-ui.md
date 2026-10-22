---
paths:
  - "tools/review-ui/**"
---

# Review UI — check firing + scratch validators

- **QA browser = this machine's `chromiumfish` + `playwright-core`, resolved from the pnpm global
  store, by user ruling** (on `CLAUDE.md` `Authoring` portability): `tools/*.mjs`, the scratch
  validators below and `data-boundary.md`'s `webcap` capture name these tools as written; a machine
  without them cannot run those checks.

## Check firing evidence

- **`tools/review-ui/tools/label_click_qa.mjs` — labelling page click map, measured.** Serve a
  repo whose `.scratch/bench/user/` holds synthetic jobs + a 1920x1080 synthetic frame as job 0
  (`python -m review_ui --repo <dir>`), then `node tools/label_click_qa.mjs http://127.0.0.1:<port>`:
  14 points through the two-step zoom, worst 0.22 source px, wrist shared by hand + arm → rc 0
  (bar 0.5 px). Seed = loupe mapping dividing by `clientWidth + 4` (border-inclusive mapping) → worst 0.71 px, rc 1. Capture-free: never point it at real frames.
- **`tools/review-ui/tools/security_qa.py` — request boundary, measured.** `uv run --directory
  tools/review-ui python tools/security_qa.py` serves a synthetic repo (one job) on a free loopback
  port: foreign `Host` → 400, cross-site `text/plain` POST → 403 + no `labels.jsonl`, foreign
  `Origin` JSON → 403, malformed JSON / non-object `points` → 400, same-origin save stored, access
  log free of clip routes → 9/9, rc 0. Firing = the pre-hardening server (`eccbe39`): 2/9, rc 1
  (rebound Host served, the cross-site POST overwrote the labels, 500s, 9 logged paths). Any
  loopback server on this host is reachable from every page the browser opens; that is why each
  row exists.
- **`tools/review-ui/tools/check_glyphs.py` firing, measured** (prototype, outside the 16):
  live tree = 436 code points, 0 missing per weight, rc=0; `--extra 鬱` (the shipped firing input)
  = 1 missing per weight, `FAIL 2 missing`, rc=1. Absent `cohort/descriptors.yaml` → `charset()`
  refuses with a named cause, rc=1, in builder + check alike.
- **`tools/review-ui/tools/{overlay_map_qa,decode_banner_qa}.mjs` — self-seeding, measured.**
  `node <tool> http://127.0.0.1:<port>` against any `python -m review_ui` (`--repo <empty dir>`
  works: `/api/clip*` routed to synthetic clips, no recording loaded → capture-free + inside
  `data-boundary.md`). Seeds = `player.js` served mutated through `page.route`, no file edited, so
  nothing to restore; a seed that reds nothing fails the run.
  - overlay map: 5 lone dots × landscape/portrait × dpr 1/2, expected = the video's own
    `object-fit: contain` mapping. **20/20 green + 3/3 seeds firing, max 0.089 px**; `axis` + `offby1` red the
    < 1 px rows (Infinity / 1.510 px), `legacy` = the pre-fix `w/W` factor reds exactly the 4
    < 0.25 px rows at 0.779-0.871 px — the letterbox residual `fitStage`'s per-axis floor left.
  - decode banner: PyAV (root `.venv`, `libx265`/`libx264`) encodes 1 s 320×180 clips at run time,
    both decoding 30/30 frames in PyAV. This chromiumfish answers `canPlayType` `"probably"` for
    `hvc1` and fails it with MediaError 4 → **6/6 green + 2/2 seeds firing**; `nobanner` reds the banner row alone,
    `nohide` the hidden-video row alone. A build that decodes HEVC reds its own `decoder present` row.

## Scratch validators pending port

- **Ruled for the ports in flight (`wt/general-purpose-3`)**: the theme rehydration row reads a routed sentinel placed right after the inline resolver (`pre-paint parser checkpoint`; Playwright's `waitUntil: 'commit'` = response received, not script run) plus a seed deferring the resolver that must red it. The clock check grades on a deterministic rAF cadence (virtual `requestAnimationFrame` + `performance.now`) over the shipped `player.js`; the live arm reports a spread and sets no rc. The recorded 6-row clock-seed set is stale: `397cc8a` (round → floor) landed after `c61a18e` recorded it. Open at merge: whether the graded seed set is the current semantic failure set under a calibrated cadence rather than a cadence picked to reproduce 6 rows.

Scratch-local validators = temporary encodings (`CLAUDE.md` `Engineering`); each port = a `.agent/deferred.md` row.

- `.scratch/theme_qa.mjs` — review-ui theme control, the paths a still capture cannot show.
  `node .scratch/theme_qa.mjs http://127.0.0.1:<port>` against a running `python -m review_ui`;
  resolves `chromiumfish` + `playwright-core` out of the pnpm global store exactly as `webcap` does,
  and drives a **persistent** context, which is what makes `localStorage` outlive the reload the
  rehydration check needs. 13 checks: auto is the default and resolves light under CDP (this build
  reports `prefers-color-scheme` light), the cycle auto→light→dark→auto with its label and its
  stored value per step, `data-theme` present at document *commit* after a reload, `palette()`
  returning a used `rgb()` rather than the `light-dark()` call text, and the `:has()` stage backdrop
  across show/dim/hide. **13/13 green; 2 of 2 seeds fire** — neutering the `.stage:has(...)`
  background reds exactly the 2 backdrop rows, misspelling the pre-paint `review-ui-theme` key reds
  the 3 rehydration rows and nothing else, both files restored byte-identical by digest.
- `.scratch/player_layout_qa.mjs` — review-ui clip player layout: the stage fits the room the UI
  leaves and the transport stays on screen. `node .scratch/player_layout_qa.mjs http://127.0.0.1:<after>
  [<before>]` against a running `python -m review_ui`; same chromiumfish + `playwright-core`
  resolution as `theme_qa.mjs`, plain context. 54 checks (48 after-only): pane shape (no clip title, banner hidden,
  no codec sentence, one selected row, overlay + strip backed at device resolution), the fit over
  nine viewports, two of them stacked (< 761 px) (fits its room, fills an axis, ratio error < 1 %, video
  fills the stage, every control row on screen, page fits the viewport), a width-only resize
  rebacking the strip, and the census + cohort panel rectangles against the before state — the
  change touched shared CSS. **48/48 after-only green, including the 4 seed rows; each seed fires its own rows alone.** The second URL is a `git worktree` of the previous
  commit served with `--repo <primary>` and `UV_PROJECT_ENVIRONMENT=<primary>/tools/review-ui/.venv
  uv run --no-sync`, which measured controls below the stage at 322 px → 156 px. Seed requirements: `.stage`'s own `max-width: 100%` absorbs a `width` seed, so a seed that must escape the room
  needs `min-width`; and a `goto` differing only in its fragment is a same-document navigation, so an
  injected style tag survives into the next seed — the script carries a per-load query parameter for
  that. `.scratch/player_shot.mjs` takes the layout-only capture beside it (→ `data-boundary.md`).
- `.scratch/default_view_qa.mjs` — review-ui landing view: the player is the first tab and the view a
  bare URL opens. `node .scratch/default_view_qa.mjs http://127.0.0.1:<port>` against a running
  `python -m review_ui`; same chromiumfish + `playwright-core` resolution as `theme_qa.mjs`, plain
  context, no capture of any kind — it prints tab order, the active view, the hash and booleans, which
  is what keeps a landing-view check inside `data-boundary.md`. 16 checks: tab order, section order
  and the first tab's label, the bare URL settling on `#player` with its stage drawn, a `#cohort` deep
  link, a tab click, an unknown fragment falling back to the player, and the same-document group — a
  `location.hash` write selecting its view, the Back button, an unknown same-document hash.
  **16/16 green; 4 of 4 seeds fire their own rows and nothing else** — `DEFAULT_VIEW` player→census
  reds 6 landing rows, the `index.html` nav order player→census reds the 2 tab rows, deleting the
  `hashchange` listener reds the 3 same-document rows, and moving `location.hash = name` back after
  the render `await` reds exactly the Back row; both files restored byte-identical by sha256.
  Check requirements:
  - `show()` toggles the tab and view classes *before* it awaits the render, so a class-only wait
    reads a half-switched page and reported the previous hash under a seed. Every wait is hash +
    active view + drawn node, through `waitForFunction`.
  - A `waitForSelector` a seed can never satisfy must be caught into a `fail` row, or the seed
    reports as a crashed script rather than as a failing check.
  - **A row timed against a warm render grades the race, not the code.** The Back row passed under
    its own ordering seed once the cohort render went warm — the stale trailing write needs the
    superseded render still in flight. The row holds the window open itself (`page.route` delaying
    `/api/cohort` 1500 ms) and reads the state *after* the delay elapses, since a trailing write
    lands after Back settles. It is also two-sample — it requires the view to have BEEN cohort —
    because a Back row asserting only the destination passes vacuously wherever the hash never
    moved the view at all, which is exactly the state its own precondition row reds.
  The same-document fragment trap above applies here too and for a second reason: a fragment-only
  `goto` is not a load, and view selection now runs through `hashchange` rather than boot alone.
- `.scratch/player_clock_qa.mjs` — review-ui playback rate on the rAF clock.
  `node .scratch/player_clock_qa.mjs http://127.0.0.1:<port>` against a running `python -m review_ui`;
  same chromiumfish + `playwright-core` resolution as `theme_qa.mjs`, plain context, no capture —
  it prints frame advances and observed/expected rate ratios, which is what keeps a player-view
  check inside `data-boundary.md`. 10 checks: `hide` at 0.25×/0.5×/1× each advancing at all and
  within 20 % of the requested rate, the 0.5×:0.25× and 1×:0.25× advance ratios at 2 and 4, and
  `layer=show` still driven by the video clock. **10/10 green; the seed fires 6 rows and leaves both
  `show` rows green**, restored byte-identical by sha256 (`83bff032…`).
  `videoUsable()` returns false whenever `layer === "hide"`, so `hide` mode **and every clip without
  a platform decoder** — 123/379 are HEVC — play on the rAF branch of `tick()`. That branch must carry
  unrounded `view.framePos`: feeding `setFrame`'s rounded value back into the accumulator loses
  advances below half a frame. Measured rounded-accumulator reference: **0 frames in 2.5 s at
  0.5×, 1 frame at 0.25×, 1.20× the requested rate at 1×**. Check requirements:
  a `> 0` liveness row is too weak to catch a near-stall — 0.25× advanced 1 frame and passed it,
  and only the rate row reds; and the seed must leave the video-clock rows green, or the row set
  cannot show which clock it graded.
- `.scratch/player_time_qa.mjs` — review-ui overlay time map: the overlay draws the row exported for
  the decoded frame on screen. `node .scratch/player_time_qa.mjs http://127.0.0.1:<port>` against a
  running `python -m review_ui`; same resolution as `theme_qa.mjs`, plain context, no capture — it
  prints frame numbers, pixel counts and times. It picks the first listed clip whose first row sits
  past frame 0 and whose rows have an interior gap. 7 checks: nothing drawn before the first row or
  inside the gap, the first row and the row after the gap each at their own timestamp, and `|▶`
  moving the video to the overlay's frame. **7/7 green; 2 of 2 seeds fire their own rows** — series
  indexed by the playhead reds the 5 time-map rows, `|▶` without a video seek reds the step row —
  restored byte-identical by sha256 (`89806610…`). The series holds only frames that carry a row,
  so indexing it by the playhead drew every frame after the first missing one early: measured over
  379 clips, ≥ 2 frames on all 379 (352 start at frame 2 — `min_track_age = 3`), ≥ 10 on 69,
  max 1112 after a long gap.
- `.scratch/drift_qa.mjs` + `.scratch/ratchet_qa.mjs` — review-ui player layout stability; both
  layouts fit the viewport, side by side down to 761 px and a 2:5 stack below it.
  `node .scratch/<name>.mjs http://127.0.0.1:<port> ['#NNN Task Side · View']` against a running
  `python -m review_ui`; same resolution as `theme_qa.mjs`, no capture — layout numbers alone.
  Integral ratio-1 layout can mask both mechanisms; keep ratio-1 rows as controls alongside
  fractional-layout rows:
  - drift — `paintStrip` sets the strip's backing width to its laid-out width × devicePixelRatio, a
    canvas's natural width is its backing width, and a stacked bare `1fr` track grew to it every
    paint (live, ratio 1.1: strip 27.9M px). 22 rows; the structural row injects a 40× backing
    width and requires layout unmoved. **22/22 green; a stacked `1fr` column reds exactly the 4
    stacked injection rows.** `contain: size`/`strict` on the canvas measured ineffective. This
    build pins `devicePixelRatio` to 1 and ignores the context's `deviceScaleFactor`, so the script
    overrides the getter; that override never reproduced the per-paint growth itself, which is why
    the injection row, not the playback row, carries the check.
  - ratchet — a content-height stage pane made the stage-wrap `fitStage` reads as its room the
    stage's own height, and each floored fit shrank it, fastest at fractional layout (scale 1.1:
    530 → 375 px in 3 s, on to the 140 px floor; one ratio-1 seed run fired too). Real fractional layout needs a browser per ratio launched with
    `--force-device-scale-factor` (JS still reads 1). 60 rows (5 viewports × ratios
    1/1.1/1.25/1.5 × holds/room/onscreen). **60/60 green; stacked rows `auto auto` red 12 stacked
    room/onscreen rows and nothing else.** Both seeds restored byte-identical by sha256.
