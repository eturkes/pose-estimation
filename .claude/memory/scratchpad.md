# Scratchpad

Transient working notes — anything from "current investigation" to "half-finished idea". Safe to prune at session boundaries. Never treat as a source of truth; promote stable findings to `tech/`, decisions to `decisions.md`, lessons to `lessons.md`.

## How to use

- Start each session by skimming the most-recent entries (top of file).
- Append `## YYYY-MM-DD HH:MM — <session topic>` and write freely.
- When closing a session, either prune the entry or summarise it into a longer-lived file.

---

## 2026-06-08 — Proactive refactor pass

Done; durable record in `decisions.md` (refactor-pass entry). Net −~250 lines, 295/295 + 19 R tests green, ruff/ty clean, repomap regenerated. Survey method that worked: AST def-scan filtered to `samefile refs == 1` (defs with ≥2 in-file refs are normal module-internal use — first filter over-reported). Rejected-with-reason list lives in the decision entry; nothing pending. Next-roadmap seed unchanged: real 3-cam footage validation.

---

## 2026-06-08 — Maintenance cycle (Session 4)

Full pass; everything green. Durable records: `decisions.md` (cv2 single-wheel override), `lessons.md` (deny-list blocks Bash too), `tech/environment.md` (env-model rewrite).

- **Python deps:** coverage 7.14.1, idna 3.18, openvino 2026.2.0, ruff 0.15.16, tqdm 4.68.1, ty 0.0.44. 295/295 tests, ruff+format+ty clean on upgraded toolchain.
- **R deps:** callr 3.8.0, clipr 0.8.1, xfun 0.58; snapshotted; 19/19 R-pipeline tests.
- **cv2 hygiene:** rtmlib was coinstalling opencv-python + contrib over our headless wheel (file-stomping, mixed runtime). Excluded via `[tool.uv] override-dependencies`; headless reinstalled; suite green.
- **Security:** pip-audit on locked export — zero known vulns; `_safe_resolve` coverage unchanged and sound; new 3B/3C surface adds no data-driven path joins; subprocess use is list-form argv throughout. Web sweep surfaced nothing newer.
- **Env-model change discovered:** container home now separate (`/var/home/eturkes/debian`); project via `/run/host/...`; uv 0.11.x in-container; venv container-native — `uv sync` in-container is canonical now (environment.md rewritten). Host-side venv use would need a host-side sync (would then break container use).
- **Open question for user:** if the pipeline is ever launched from the host (NPU runs?), the container-native venv won't resolve there — flagged in session summary.

**Project status:** Phases 1–3 ✓ (3D pipeline synthetic-validated); Phase 4 maintenance current as of today. Next roadmap seed: real 3-cam footage validation; deferred items: 3D-aware downstream aggregation, multi-person cross-camera identity matching.
