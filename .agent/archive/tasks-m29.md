# M2.9 request — closed `Tasks` rows

Cleared from `.agent/spec.md` `Tasks` at the agents@8bc2c15 template refresh (template: ticked
rows cleared at each MAINTAIN request's close; the M2.9 request closed at `0814560`). Rows
verbatim; their pointers cite their own time (→ `.claude/rules/retention.md`).

- [x] 481ba22 **M2.9.1 subject tracker (C02 C03 C08 C11)** — contract `.agent/archive/contract-m2u91.md`;
  diff-blind `tester` → own red witness → implementation → `reviewer` → gate → seven-arm re-sweep
  (P14-P17). Mechanism law → `.claude/rules/rtmlib-runtime.md`.
- [x] d4dafdb **M2.9.2 output hygiene (C05 subset, C07, C10)** — contract
  `.agent/archive/contract-m2u92.md`: out-of-frame points score 0, duplicate hand suppressed,
  face landmark sides corrected. General occlusion hallucination (legs under the table, hips
  from above) is not separable by score — hallucinated knee median 0.48 vs real wrist q25 0.49
  (right views) — so it stays a reported model limit.
- [x] 83b5b40 **M2.9.3 hands (C09) — evaluated twice, not adopted.** RTMW-X scores saturate; ungated
  hand-crop refinement moves the confidence scale + collapses occluded hands; the gated version
  (user-requested) removes the collapse but misses jitter <= 0.006 (0.0079) at +41 % wall →
  catalog § M2.9.3, queued.
- [x] a395e28 **M2.9.5 seated body-part drop (user ruling)** — contract
  `.agent/archive/contract-m2u95.md`: knees, ankles, feet score 0 everywhere; hips score 0 under
  the overhead camera.
- [x] 29a24e4 **M2.9.4 SPARC** — contract `.agent/archive/contract-m2u94.md`: the R stage's
  fixed-cutoff SAL became SPARC (adaptive cutoff, Balasubramanian 2015), method version v3; it
  runs inside the corpus rerun's own R stage.
- [x] 7a1201d **Corpus rerun** — one pass, 193/193 events, 379/379 `ok`, verdicts all true
  (→ `Artifacts`).
- [x] 0814560 **Corpus rerun downstream** — cohort republished from M2.9 (`cohort.md` trunk
  bullet); catalog after-table + P08 pass / P09 fail / P10 vacuous (`pose-weak-points.md`
  § M2.9 corpus, 3 `deferred.md` rows); `corpus-run.md` measured-whole = 5.559 h; spot-check
  sheets watched; review ledger `.agent/review.md`.
