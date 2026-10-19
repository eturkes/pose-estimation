---
paths:
  - "src/pose_estimation/bench/**"
  - "src/pose_estimation/score_calibration.py"
  - "scripts/bench.py"
  - "tools/cocowb-eval/**"
  - "docs/technical/benchmark.md"
  - "tests/test_m4u*.py"
---

# Benchmark instruments — what each can resolve, how to run them

Contracts `.agent/archive/contract-m4u{1,2,3a,3b,4,6}.md`; measured resolutions + regeneration →
`docs/technical/benchmark.md`.

- **Agents cannot grade fine hand accuracy.** Agent-placed hand points err ~0.10 of the hand box
  (COCO-WB, 2 labellers) — as coarse as RTMW-m; the blinded pairwise judge sat near chance at every
  error margin while answering consistently across colour swaps (stable, mostly wrong). Agent labels
  = gross-error ground truth (hand states, patient box, joints at 0.3 of the hand extent / 0.1 of
  the patient box); fine accuracy = public GT (`public-eval`, `hand-eval`) + the user's `/label`
  frames. Never brief a judge or labeller for a finer resolution than the measured one.
- **The user's 96 frames are one-use confirmation** (`.scratch/bench/user/labels.jsonl`): no agent
  reads, scores or summarises them before `contract-m4u4.md` names its candidates (A01), and no
  parameter is tuned on them after. `confirm.frozen_candidates` refuses to score without A01.
- **One OpenVINO GPU job at a time.** The Arc iGPU is shared with other projects' jobs on this
  host; two concurrent pose/detector processes died with `CL_OUT_OF_RESOURCES` (rc 134) — the
  M4.2 RTMW-X eval and the M4.1 pre-roll replay. Queue GPU work in one sequential script
  (`.scratch/impl/gpu_queue*.sh` pattern) and give a teammate the GPU by message only.
- **Every cache is resumable, every scorer re-runs from caches**: `public-eval`, `hand-eval`,
  `corpus-gross` save predictions before scoring (`*.npz`), and `preroll` reuses finished windows,
  so a crash costs the unfinished arm alone.
- **Raw scores never cross arms.** Each checkpoint has its own raw scale (RTMW-L 384 median 4.2 vs
  RTMW-m 0.74 on the same crops); compare by rank statistics (AP, AUROC) or after
  `score_calibration`.
- Patient-adjacent outputs (samples, sheets, labels, replays, corpus predictions) stay under
  `.scratch/bench/`; public-GT tables carry no corpus identifier and may be committed.

## Firing records

- `tools/cocowb-eval/check.py` (gate step `cocowb-eval`): perfect result AP 1.0 on every part,
  shifted right hand loses right-hand AP alone, swapped hands lose hand AP → rc 0; mutant
  `score.SIGMAS * 100` (OKS too loose to see a 25 px shift) → 2 FAIL rows, rc 1.
- `scripts/bench.py progress`: half-answered public set → `50/100`, rc 1; complete → rc 0.
- `scripts/check_review_report.py` on a seeded all-`unknown` reviewer table → `FAIL P03`.
