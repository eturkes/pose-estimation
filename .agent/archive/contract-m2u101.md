# Contract — M2.10.1 hand presence gate: hysteresis on the exported hand score

Tier `kernel`. Base = the commit carrying this file. Frozen at dispatch; amendments append to §8.
User ruling (this session): adopt the gate, hysteresis on 0.65 / off 0.5 frozen, validated on
held-out labels with the visible-hand loss reported per subject.

## 1. Artifact

- `src/pose_estimation/keypoint_hygiene.py` — `HandPresenceGate`, `HAND_GATE_ON = 0.65`,
  `HAND_GATE_OFF = 0.5`, `HAND_GATE_MIN_POINTS = 10`.
- `src/pose_estimation/run.py` — `--hand-gate` (store_true, default off); `process_source(...,
  hand_gate=None)` takes a gate instance; `main` builds one when the flag is set and passes it
  through `_dispatch_sessions`.
- `src/pose_estimation/corpus_run.py` — `POSE_CONFIG_FIELDS` gains `hand_gate`.
- `scripts/corpus_run_2d.py`, `scripts/pilot_corpus_run.py` — `--hand-gate/--no-hand-gate`
  (BooleanOptionalAction, default on), forwarded as `--hand-gate`, in `configuration`,
  `REPORT_FIELDS` and `pose_config.json`; `GENERATOR_VERSION` corpus v5, pilot v4.
- `docs/technical/tracking-modes.md`, `docs/technical/entrypoints.md` — reference updates.

## 2. Design decisions

**D01 — the quantity.** Per person row and per hand (COCO-WholeBody 91-111 left, 112-132 right),
the gate reads the hand's 21 scores as the exporter will write them. `present` = at least
`HAND_GATE_MIN_POINTS` of the 21 are finite and > 0; `level` = mean of those positive scores.
Evidence: on 148 labelled frames (20 clips) of the published M2.9 tree, hallucinated hands
(drawn where no hand is) score `level` median 0.43 against 0.83 for real hands (AUC 0.03); the
thresholds were chosen there, on exactly this quantity read from the exported CSV.

**D02 — hysteresis.** State per (track key, hand), initially off. Off → on when `present` and
`level >= HAND_GATE_ON`. On → off when not `present` or `level < HAND_GATE_OFF`. While off, the
hand's 21 scores are set to 0 in the returned array; coordinates stay. Measured on the 20 clips
(offline replay over the published series): blink-outs 2.82 → 0.32 per 100 frames;
hallucinated hands dropped 93/103, real hands 8/204.

**D03 — where.** After the smoother and the bone-length smoother, before `filter_single_subject`
and the CSV export. Hygiene's "before the smoother" rule (M2.9.2 D04) is about observations a
filter must hold; this gate decides what is exported, and its thresholds are calibrated on the
smoother's output score (`min(EMA, raw)`), so it reads that score. Track key = the smoother's
`output_track_keys()` when a smoother runs; otherwise the tracker's `last_track_ids`; otherwise
the row index. Keys absent from `live_track_keys()` (or from the current keys when no smoother
runs) are pruned, so a re-born track starts off.

**D04 — interface.** `HandPresenceGate(on=HAND_GATE_ON, off=HAND_GATE_OFF,
min_points=HAND_GATE_MIN_POINTS)`; `__call__(scores, keys) -> scores` returns a new array, never
mutates its input, accepts `(0, 133)`, and passes rows with a keypoint count other than 133
through unchanged; `reset()` clears all state (called by `process_source` per source, like the
smoother); `prune(live_keys)`; `on <= off` raises `ValueError`. Counters on the instance:
`hand_frames_present` (hand-frames entering with `present`), `hand_frames_gated` (of those, zeroed),
both reset by `reset()`.

**D05 — generation identity.** `hand_gate` joins `POSE_CONFIG_FIELDS`, both drivers'
`REPORT_FIELDS` and `configuration`; drivers default it on; `run.py` defaults it off. A complete
event recorded without the gate is due on resume and refused by `--analyse-only` / `--reuse-run`.

**D06 — diagnostics.** `run.SOURCE_DIAGNOSTIC_FIELDS` gains `hand_frames_present` and
`hand_frames_gated`, written from the gate's counters (0 and 0 when no gate runs). This is the
firing counter the M2.9.2 P10 lesson asks for: the gate's effect is countable per asset.

## 3. Predicates

Tester-owned (synthetic, no media):

- **P01** Thresholds: a hand at `level` 0.64 never turns on; at 0.65 it turns on; once on it stays
  on at 0.50 and turns off at 0.49; off it needs 0.65 again. `present` needs >= 10 positive finite
  points (9 → absent → off). Each hand of each key is independent.
- **P02** Zeroing: an off hand's 21 scores are 0, every other score and every coordinate equal the
  input; the input array is not mutated; `(0, 133)` and non-133 rows pass through.
- **P03** State: per key; `prune` drops absent keys, so a returning key starts off; `reset` clears
  state and counters; `on <= off` raises.
- **P04** Counters: `hand_frames_present` / `hand_frames_gated` count exactly per D04.
- **P05** `process_source` with a gate: the exported CSV carries no coordinates for a gated hand
  (the exporter's presence rule then fails) and unchanged values for an admitted one; the gate
  runs after the smoother (a negative control placing it before the smoother changes the admitted
  hand's exported score); diagnostics carry both counters; without a gate the CSV equals the
  pre-change output byte for byte.
- **P06** CLI + drivers: `run.main` builds a gate only under `--hand-gate` and forwards it;
  drivers default on, forward `--hand-gate`, publish `hand_gate` in `configuration`,
  `REPORT_FIELDS`, `pose_config.json`; an event recorded without the gate is due on resume and
  refused by `--analyse-only` / `--reuse-run`.

MAIN-owned, real frames:

- **P07** Held-out: on frames labelled by `general-purpose-2/-3` (clips disjoint from the 20 tuning
  clips), replaying the gate over the published M2.9 series drops >= 0.80 of hands labelled
  hallucinated (`f`) and <= 0.06 of hands labelled real (`k`); the per-subject real-hand drop share
  is reported for every subject with >= 5 labelled real hands. Threshold values do not change on
  this result (frozen by ruling); a miss is reported to the user.
- **P08** In-pipeline equals replay: a watch-set pilot with the gate reproduces the offline
  replay's per-clip presence share within 0.01 on every clip.

## 4. Invariant surfaces

1. Every existing suite passes; `process_source` callers that pass no gate keep M2.9 behaviour.
2. CSV schema (304 columns), R stage, trackers, smoother: unchanged.

## 5. Gate identity

```sh
env -u LD_LIBRARY_PATH PYTHONPATH="$PWD/src" uv run --no-sync <ruff check|ruff format --check|ty check|pytest>
```

## 6. Negative controls

1. `on` compared with `>` instead of `>=` → P01 fires.
2. Gate before the smoother → P05 fires.
3. State keyed by row index while a smoother runs → P03/P05 fire on a key swap.
4. Drivers default the gate off → P06 fires.

## 7. Verdict table

Appended at close.

## 8. Amendments

Appended as ruled.
