"""Benchmark instrument CLI (contract M4.1). Patient-adjacent outputs → `.scratch/bench/` only.

  sample         frozen corpus frame sample (JSON)
  frame-sheets   pass A sheets for a sample
  crop-sheets    pass B body + hand sheets from pass A answers
  ingest         answers → codebook-v2 records (JSONL), validated
  public-sheets  COCO-WholeBody hand + body sheets for labeller qualification (GT sealed apart)
  public-score   labeller answers vs sealed GT → error quantiles + state confusion
  judge-public   blinded pairwise sheets from two arms' real errors on COCO-WholeBody hands
  judge-score    decoded judge answers vs GT → accuracy per error-margin bin (Wilson 95 %)
  user-sample    the confirmation frame set the user labels in the review UI (excludes the pilot)
  progress       answered / total sheets of one kind set + malformed answers (no GT read)
  public-eval    one arm over every COCO-WholeBody val person (GT boxes): official AP via
                 tools/cocowb-eval + hand / body PCK + score-visibility AUROC
  calibrate      fit an arm's score calibration on its cached COCO-WholeBody predictions →
                 src/pose_estimation/score_calibrations/<arm>.json (contract M4.3b)
  agree          two labellers' codebook-v2 records on the same frames: per-field agreement +
                 coordinate disagreement / labelled extent (hands) or patient-box diagonal (arms)
  corpus-gross   arms on the labelled pilot frames inside the consensus patient box: hand +
                 arm-joint gross localisation, side swaps, hand-presence AUROC, subject bootstrap
  screen         M4.2 D03: paired image bootstraps of COCO-WB hand AP + DexYCB hand PCK@0.1, arm
                 PCK@0.05 slack → the finalist set
  presence       M4.3b: whole-clip production replay per pilot clip → exported hand presence vs the
                 labellers' agreed hand states, F1 with a subject bootstrap
  confirm        M4.4: the user's frames vs candidate + baseline corpus-run rows → E1-E4,
                 subject bootstrap, decision table, adoption (refuses without contract A01)
  pilot-eval     M4.3: arm + baseline replayed over the pilot → presence F1, gross hand + arm,
                 side swaps; paired subject intervals → pass / refit / fail
  hand-eval      one arm over a hand set (DexYCB, HInt) with the shipped detector's person box:
                 hand PCK by visibility, coverage, score-occlusion AUROC
  preroll        replay the shipped pipeline over windows ending at sampled frames; compare each
                 window's last row with the published whole-clip row (accelerator recipe)

Run under the gate prefix: `scripts/gate.sh python scripts/bench.py <command> …`.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from pose_estimation.bench import (
    arms,
    handsets,
    judge,
    labels,
    public,
    qualify,
    sample,
    screen,
    sheets,
)
from pose_estimation.bench.render import frame_sheet

SIDE_WORDS = sheets.SIDE_WORDS


def _answers(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        rows.extend(labels.load(path))
    return rows


def cmd_sample(a: argparse.Namespace) -> int:
    clips = sample.corpus_clips(a.sessions, a.inventory, a.run)
    names = sample.family_labels(clips)
    watch_labels = set(json.loads(a.watch.read_text(encoding="utf-8")))
    watch = {k for k, label in names.items() if label in watch_labels}
    if len(watch) != len(watch_labels):
        raise SystemExit(f"watch set: {len(watch)} of {len(watch_labels)} labels resolve")
    frames = sample.corpus_sample(clips, a.seed, watch, a.subjects, a.per_view, a.stress)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps([asdict(f) for f in frames], indent=1), encoding="utf-8")
    strata = defaultdict(int)
    for f in frames:
        strata[(f.stratum, f.view)] += 1
    print(
        f"sample: {len(frames)} frames, {len({f.subject for f in frames})} subjects;", dict(strata)
    )
    return 0


def _media(sessions: Path, event_id: str, camera: str) -> Path:
    for path in sorted((sessions / event_id).glob(f"{camera}.*")):
        return path
    raise FileNotFoundError("a sampled clip has no media under --sessions")


def cmd_frame_sheets(a: argparse.Namespace) -> int:
    frames = json.loads(a.sample.read_text(encoding="utf-8"))
    a.out.mkdir(parents=True, exist_ok=True)
    entries: list[sheets.SheetEntry] = []
    by_clip: dict[tuple[str, str], list[int]] = defaultdict(list)
    for f in frames:
        by_clip[(f["event_id"], f["camera_name"])].append(f["frame"])
    for (event_id, camera), indices in by_clip.items():
        for index, image in sample.read_frames(_media(a.sessions, event_id, camera), indices):
            canvas, sm = frame_sheet(image, max_side=a.max_side)
            sheets.write_sheet(
                a.out, entries, canvas, sm, seed=a.seed, kind="frame",
                key=f"{event_id}/{camera}#{index}", index=index, shape=image.shape[:2],
            )  # fmt: skip
    sheets.save(a.out, entries, a.seed)
    print(f"frame-sheets: {len(entries)} sheets")
    return 0


def cmd_crop_sheets(a: argparse.Namespace) -> int:
    manifest = sheets.load_manifest(a.sheets)
    entries = list(manifest.values())
    pass_a = {Path(r["sheet"]).stem: r for r in _answers(a.answers)}
    by_clip: dict[tuple[str, str], dict[int, sheets.SheetEntry]] = defaultdict(dict)
    for t, entry in manifest.items():
        if entry.kind == "frame" and t in pass_a:
            clip, _, _ = entry.key.partition("#")
            event_id, _, camera = clip.partition("/")
            by_clip[(event_id, camera)][entry.index] = entry
    made = 0
    for (event_id, camera), wanted in by_clip.items():
        for index, image in sample.read_frames(_media(a.sessions, event_id, camera), wanted):
            entry = wanted[index]
            sm = sheets.SheetMap(**entry.map)
            answer = pass_a[entry.token]
            shape = image.shape[:2]
            patient = sheets.frame_box(sm, answer.get("patient"))
            if patient is not None:
                canvas, cm = sheets.body_crop(image, patient)
                sheets.write_sheet(
                    a.sheets, entries, canvas, cm, seed=a.seed, kind="body", key=entry.key,
                    index=index, shape=shape,
                    caption="Patient: shoulders, elbows, wrists. L/R = the patient's own sides.",
                )  # fmt: skip
                made += 1
            for side in labels.SIDES:
                hand = answer.get(side, {})
                extent = sheets.frame_box(sm, hand.get("extent"))
                if hand.get("hand") in ("vis", "part") and extent is not None:
                    canvas, cm = sheets.hand_crop(image, extent)
                    sheets.write_sheet(
                        a.sheets, entries, canvas, cm, seed=a.seed, kind="hand", key=entry.key,
                        index=index, shape=shape, side=side,
                        caption=f"Patient's {SIDE_WORDS[side]} hand: label its joints.",
                    )  # fmt: skip
                    made += 1
    sheets.save(a.sheets, entries, a.seed)
    print(f"crop-sheets: {made} sheets")
    return 0


def cmd_ingest(a: argparse.Namespace) -> int:
    records = sheets.ingest(a.sheets, _answers(a.answers), a.labeller)
    defects = {r["frame"]["key"]: labels.validate(r) for r in records}
    bad = {k: v for k, v in defects.items() if v}
    labels.write(a.out, records)
    print(f"ingest: {len(records)} records, {len(bad)} with defects")
    for n, issues in enumerate(bad.values(), 1):
        print(f"  record {n}: {'; '.join(issues)}")
    return 1 if bad else 0


def cmd_public_sheets(a: argparse.Namespace) -> int:
    files, persons = public.load_coco_wholebody(a.coco_wb)
    rng = np.random.default_rng(a.seed)
    idx = list(labels.HAND_JOINTS.values())
    hands = [
        h
        for p in persons
        for h in p.hands
        if max(h.box[2], h.box[3]) >= 24 and (h.kpts[idx, 2] == 2).all()
    ]
    diag = np.array([h.diagonal for h in hands])
    tert = np.quantile(diag, [1 / 3, 2 / 3])
    picked_hands = []
    for t in range(3):
        pool = [
            h
            for h, d in zip(hands, diag, strict=True)
            if (t == 0 or d >= tert[t - 1]) and (t == 2 or d < tert[t])
        ]
        picked_hands += [pool[i] for i in rng.choice(len(pool), a.hands // 3, replace=False)]
    bodies = [p for p in persons if p.area >= 64 * 64 and (p.body[5:11, 2] > 0).sum() >= 4]
    picked_bodies = [bodies[i] for i in rng.choice(len(bodies), a.persons, replace=False)]
    a.out.mkdir(parents=True, exist_ok=True)
    entries: list[sheets.SheetEntry] = []
    gt: dict[str, dict] = {}
    for h in picked_hands:
        image = cv2.imread(str(a.images / files[h.image_id]))
        x, y, w, hh = h.box
        canvas, sm = sheets.hand_crop(image, (x, y, x + w, y + hh))
        e = sheets.write_sheet(
            a.out, entries, canvas, sm, seed=a.seed, kind="hand", key=f"coco/{h.image_id}/{h.ann_id}",
            index=0, shape=image.shape[:2], side=h.side,
            caption=f"Person's {SIDE_WORDS[h.side]} hand (centre): label its joints.",
        )  # fmt: skip
        gt[e.token] = {"kind": "hand", "diag": h.diagonal, "kpts": h.kpts[idx].tolist()}
    for p in picked_bodies:
        image = cv2.imread(str(a.images / files[p.image_id]))
        x, y, w, hh = p.box
        canvas, sm = sheets.body_crop(image, (x, y, x + w, y + hh))
        e = sheets.write_sheet(
            a.out, entries, canvas, sm, seed=a.seed, kind="body", key=f"coco/{p.image_id}/{p.ann_id}",
            index=0, shape=image.shape[:2],
            caption="Centre person: shoulders, elbows, wrists. L/R = the person's own sides.",
        )  # fmt: skip
        gt[e.token] = {
            "kind": "body",
            "diag": float(np.hypot(w, hh)),
            "kpts": p.body[5:11].tolist(),
        }
    sheets.save(a.out, entries, a.seed)
    a.gt.parent.mkdir(parents=True, exist_ok=True)
    a.gt.write_text(json.dumps(gt), encoding="utf-8")
    print(f"public-sheets: {len(picked_hands)} hands, {len(picked_bodies)} persons")
    return 0


#: COCO body-17 indices of the labelled body joints, per anatomical side.
BODY17 = {
    "L": {"shoulder": 5, "elbow": 7, "wrist": 9},
    "R": {"shoulder": 6, "elbow": 8, "wrist": 10},
}


def cmd_public_score(a: argparse.Namespace) -> int:
    manifest = sheets.load_manifest(a.sheets)
    gt = json.loads(a.gt.read_text(encoding="utf-8"))
    errors: dict[str, list[float]] = defaultdict(list)
    states: list[tuple[str, str]] = []
    for answer in _answers(a.answers):
        t = Path(answer["sheet"]).stem
        entry, truth = manifest[t], gt[t]
        sm = sheets.SheetMap(**entry.map)
        k = np.asarray(truth["kpts"])
        if truth["kind"] == "hand":
            for j, name in enumerate(labels.HAND_JOINTS):
                got = answer.get("joints", {}).get(name, {})
                states.append(("vis", got.get("state", "?")))
                if got.get("state") == "vis" and got.get("uv") is not None:
                    xy = np.asarray(sm.to_frame(*got["uv"]))
                    errors[f"hand/{name}"].append(float(np.hypot(*(xy - k[j, :2])) / truth["diag"]))
        else:
            for side, joints in BODY17.items():
                for name, i in joints.items():
                    row = k[i - 5]
                    got = answer.get(side, {}).get(name, {})
                    want = {0: "?", 1: "occ", 2: "vis"}[int(row[2])]
                    states.append((want, got.get("state", "?")))
                    if row[2] == 2 and got.get("state") == "vis" and got.get("uv") is not None:
                        xy = np.asarray(sm.to_frame(*got["uv"]))
                        errors[f"body/{name}"].append(
                            float(np.hypot(*(xy - row[:2])) / truth["diag"])
                        )
    for name, vals in sorted(errors.items()):
        p50, p90 = qualify.quantiles(vals)
        print(f"public-score: {name:16s} n={len(vals):3d} err/diag p50={p50:.3f} p90={p90:.3f}")
    confusion: dict[tuple[str, str], int] = defaultdict(int)
    for pair in states:
        confusion[pair] += 1
    print("public-score: state (gt, label) counts:", dict(sorted(confusion.items())))
    return 0


def cmd_judge_public(a: argparse.Namespace) -> int:
    files, persons = public.load_coco_wholebody(a.coco_wb)
    used = {e.key for e in sheets.load_manifest(a.exclude).values()} if a.exclude else set()
    idx = list(labels.HAND_JOINTS.values())
    # Joints with any annotation (c > 0): COCO-WB marks visible joints 2, others 1 or a fraction.
    cands = [
        (p, h)
        for p in persons
        for h in p.hands
        if h.diagonal >= a.min_diag and f"coco/{h.image_id}/{h.ann_id}" not in used
    ]
    rng = np.random.default_rng(a.seed)
    cands = [cands[i] for i in rng.permutation(len(cands))[: a.pool]]
    models = {name: arms.load(arms.POSE_ARMS[name]) for name in (a.arm_a, a.arm_b)}
    by_bin: dict[int, list[tuple[judge.Pair, dict]]] = defaultdict(list)
    side_word = {"L": "LEFT", "R": "RIGHT"}
    for p, h in cands:
        image = cv2.imread(str(a.images / files[p.image_id]))
        x, y, w, hh = p.box
        pred = {n: arms.predict(m, image, [(x, y, x + w, y + hh)])[0][0] for n, m in models.items()}
        base = 91 if h.side == "L" else 112
        for j, name in zip(idx, labels.HAND_JOINTS, strict=True):
            if h.kpts[j, 2] <= 0:
                continue
            gt = h.kpts[j, :2]
            pa, pb = pred[a.arm_a][base + j], pred[a.arm_b][base + j]
            ea, eb = (float(np.hypot(*(q - gt)) / h.diagonal) for q in (pa, pb))
            if np.hypot(*(pa - pb)) < 1.0:
                continue  # indistinguishable on the sheet
            pair = judge.Pair(
                sheets.token(a.seed, "pair", h.ann_id, h.side, name),
                files[p.image_id],
                f"person's {side_word[h.side]} {name.replace('_', ' ')}",
                (float(pa[0]), float(pa[1])),
                (float(pb[0]), float(pb[1])),
                a.arm_a,
                a.arm_b,
                side=h.side,
                crop=2.0 * max(h.box[2], h.box[3]),
            )
            truth = {
                "truth": a.arm_a if ea < eb else a.arm_b,
                "margin": abs(ea - eb),
                "ea": ea,
                "eb": eb,
            }
            by_bin[judge.bin_of(abs(ea - eb))].append((pair, truth))
    chosen = []
    for b in range(len(judge.BINS) - 1):
        pool = by_bin[b]
        take = rng.permutation(len(pool))[: a.per_bin]
        chosen += [pool[i] for i in take]
        print(f"judge-public: bin {b} {len(pool)} candidate pairs, {len(take)} taken")
    jobs = judge.make_sheets(
        [c[0] for c in chosen],
        lambda name: cv2.imread(str(a.images / name)),
        a.out,
        a.key,
        a.seed,
    )
    (a.out / "jobs.json").write_text(json.dumps(jobs, indent=1), encoding="utf-8")
    truths = {c[0].pair_id: c[1] for c in chosen}
    a.truth.write_text(json.dumps(truths), encoding="utf-8")
    return 0


def cmd_judge_score(a: argparse.Namespace) -> int:
    key = list(labels.load(a.key))
    truths = json.loads(a.truth.read_text(encoding="utf-8"))
    answers = {Path(r["sheet"]).name: r["answer"] for r in _answers(a.answers)}
    rows = judge.decode(key, answers)
    margins = [truths[r["pair_id"]]["margin"] for r in rows]
    outcomes = [r["outcome"] for r in rows]
    truth = [truths[r["pair_id"]]["truth"] for r in rows]
    for row in qualify.judge_bins(margins, truth, outcomes, judge.BINS):
        print(
            "judge-score:", {k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}
        )
    by_pair: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        by_pair[r["pair_id"]].add(r["outcome"])
    consistent = sum(len(v) == 1 for v in by_pair.values())
    print(f"judge-score: colour-swap consistency {consistent}/{len(by_pair)} pairs")
    return 0


def cmd_user_sample(a: argparse.Namespace) -> int:
    clips = sample.corpus_clips(a.sessions, a.inventory, a.run)
    names = sample.family_labels(clips)
    watch_labels = set(json.loads(a.watch.read_text(encoding="utf-8")))
    watch = {k for k, label in names.items() if label in watch_labels}
    pilot = {
        (f["event_id"], f["camera_name"], f["frame"])
        for f in json.loads(a.exclude.read_text(encoding="utf-8"))
    }
    frames = [
        f
        for f in sample.corpus_sample(clips, a.seed, watch, a.subjects, a.per_view, stress=0)
        if (f.event_id, f.camera_name, f.frame) not in pilot
    ]
    order = sorted(frames, key=lambda f: sheets.token(a.seed, "user-order", f.event_id, f.frame))
    jobs = [asdict(f) | {"job": i} for i, f in enumerate(order)]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(jobs, indent=1), encoding="utf-8")
    frames_dir = a.out.parent / "frames"
    frames_dir.mkdir(exist_ok=True)
    by_clip: dict[tuple[str, str], dict[int, int]] = defaultdict(dict)
    for job in jobs:
        by_clip[(job["event_id"], job["camera_name"])][job["frame"]] = job["job"]
    for (event_id, camera), wanted in by_clip.items():
        for index, image in sample.read_frames(_media(a.sessions, event_id, camera), wanted):
            cv2.imwrite(
                str(frames_dir / f"{wanted[index]:03d}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 95]
            )
    print(f"user-sample: {len(jobs)} frames, {len({f.subject for f in order})} subjects")
    return 0


def cmd_progress(a: argparse.Namespace) -> int:
    jobs = {j["sheet"]: j for j in json.loads((a.sheets / "jobs.json").read_text(encoding="utf-8"))}
    wanted = {k for k, j in jobs.items() if not a.kind or j.get("kind") in a.kind}
    answered, malformed, unknown = set(), [], 0
    for line_no, row in enumerate(_answers(a.answers), 1):
        sheet = row.get("sheet")
        if sheet not in jobs:
            unknown += 1
            continue
        kind = jobs[sheet].get("kind", "judge")
        need = {
            "frame": ("task", "L", "R"),
            "body": ("L", "R"),
            "hand": ("joints",),
            "judge": ("answer",),
        }
        if any(f not in row for f in need[kind]) or (
            kind == "judge" and row["answer"] not in judge.RESPONSES
        ):
            malformed.append(line_no)
        answered.add(sheet)
    done = len(wanted & answered)
    print(
        f"progress: {done}/{len(wanted)} answered, {len(malformed)} malformed lines {malformed[:10]}, {unknown} unknown sheets"
    )
    return 0 if done == len(wanted) and not malformed and not unknown else 1


def _diag_row(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as stream:
        return next(csv.DictReader(stream))


def cmd_preroll(a: argparse.Namespace) -> int:
    from pose_estimation.bench import replay

    frames = [f for f in json.loads(a.sample.read_text(encoding="utf-8")) if f["stratum"] == "main"]
    chosen = []
    for f in frames:
        diag = a.run / f["event_id"] / f"{f['camera_name']}_diag.csv"
        row = _diag_row(diag)
        fps, n = float(row["fps_nominal"]), int(row["n_frames_decoded"])
        untrimmed = int(row["task_start_frame"]) == 0 and int(row["task_end_frame"]) == n
        if untrimmed and f["frame"] >= round(max(a.seconds) * fps):
            chosen.append((f, fps))
    chosen = chosen[: a.frames]
    if not chosen:
        raise SystemExit("preroll: no eligible sampled frame")
    table: dict[float, list[tuple[float, int]]] = defaultdict(list)
    for n_done, (f, fps) in enumerate(chosen, 1):
        media = _media(a.sessions, f["event_id"], f["camera_name"])
        # Reference = the whole clip up to the frame under THIS code: the published rows came from
        # an earlier pipeline (M4.6 changed gate keying + export hygiene), so equality against them
        # would measure the code change, not the pre-roll.
        ref_out = a.scratch / f"w{n_done}_whole"
        ref_csv = ref_out / f"{media.stem}.csv"
        if not (ref_csv.is_file() and replay.row_at(ref_csv, f["frame"]) is not None):
            ref_csv = replay.replay_window(media, 0, f["frame"], ref_out)
        want = replay.landmark_cells(replay.row_at(ref_csv, f["frame"]))
        for seconds in a.seconds:
            start = f["frame"] - round(seconds * fps)
            out = a.scratch / f"w{n_done}_{seconds}"
            done = out / f"{media.stem}.csv"
            # A finished window (its CSV holds the target row) is reused, so a crashed run resumes.
            if done.is_file() and replay.row_at(done, f["frame"] - start) is not None:
                got_csv = done
            else:
                got_csv = replay.replay_window(media, start, f["frame"], out)
            got = replay.landmark_cells(replay.row_at(got_csv, f["frame"] - start))
            table[seconds].append(replay.read_point_delta(want, got))
        print(f"preroll: {n_done}/{len(chosen)} frames replayed", flush=True)
    for seconds in a.seconds:
        rows = table[seconds]
        near = sum(d <= replay.PREROLL_TOLERANCE and not flips for d, flips in rows)
        print(
            f"preroll: {seconds:>4} s  equivalent {near}/{len(rows)} (every read point within "
            f"{replay.PREROLL_TOLERANCE}, no visibility flip)  max |dxy| {max(d for d, _ in rows):.5f}  "
            f"frames with a visibility flip {sum(bool(f) for _, f in rows)}"
        )
    return 0


def cmd_public_eval(a: argparse.Namespace) -> int:
    import subprocess

    files, persons = public.load_coco_wholebody(a.coco_wb)
    persons = [p for p in persons if p.box[2] > 1 and p.box[3] > 1]
    arm = arms.POSE_ARMS[a.arm]
    out = a.out / arm.name
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "coco_wb_pred.npz"
    if cache.is_file():
        z = np.load(cache)
        kps, scores = z["kps"], z["scores"]
    else:
        model = arms.load(arm, a.device)
        kps = np.zeros((len(persons), 133, 2))
        scores = np.zeros((len(persons), 133))
        by_image: dict[int, list[int]] = defaultdict(list)
        for i, p in enumerate(persons):
            by_image[p.image_id].append(i)
        for n, (image_id, idxs) in enumerate(by_image.items(), 1):
            image = cv2.imread(str(a.images / files[image_id]))
            boxes = [(persons[i].box[0], persons[i].box[1], persons[i].box[0] + persons[i].box[2],
                      persons[i].box[1] + persons[i].box[3]) for i in idxs]  # fmt: skip
            k, sc = arms.predict(model, image, boxes)
            kps[idxs], scores[idxs] = k, sc
            if n % 500 == 0:
                print(f"public-eval: {arm.name} {n}/{len(by_image)} images", flush=True)
        np.savez_compressed(cache, kps=kps, scores=scores)
    results = screen.result_rows(persons, kps, scores)
    res_path = out / "coco_wb_results.json"
    res_path.write_text(json.dumps(results), encoding="utf-8")
    evaluator = Path(__file__).resolve().parents[1] / "tools" / "cocowb-eval"
    stats = json.loads(
        subprocess.run(
            ["uv", "run", "--quiet", "--directory", str(evaluator), "python", "score.py",
             str(a.coco_wb.resolve()), str(res_path.resolve())],
            check=True, capture_output=True, text=True,
        ).stdout
    )  # fmt: skip
    hand_err = screen.hand_errors(persons, kps)
    body_err, vis_scores, occ_scores = screen.arm_errors(persons, kps, scores)
    summary = {
        "arm": arm.name,
        "persons": len(persons),
        "official": stats,
        "hand_pck": {sel: screen.pck(v, screen.PCK_HAND) for sel, v in hand_err.items()},
        "arm_pck": screen.pck(body_err, screen.PCK_ARM),
        "arm_visibility_auroc": screen.auroc(np.asarray(vis_scores), np.asarray(occ_scores)),
    }
    (out / "coco_wb_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary))
    return 0


def _person_boxes(hands: list, cache: Path) -> dict[str, list[list[float]]]:
    """Shipped YOLOX boxes per image, computed once per set and reused by every arm."""
    if cache.is_file():
        return json.loads(cache.read_text(encoding="utf-8"))
    import rtmlib

    from pose_estimation.rtmlib_openvino import _patch_rtmlib_openvino
    from pose_estimation.run import _DET_INPUT_SIZE, _DET_URL

    _patch_rtmlib_openvino()
    det = rtmlib.YOLOX(_DET_URL, model_input_size=_DET_INPUT_SIZE, backend="openvino", device="GPU")
    out: dict[str, list[list[float]]] = {}
    for n, image in enumerate(sorted({str(h.image) for h in hands}), 1):
        boxes = np.asarray(det(cv2.imread(image)), dtype=float).reshape(-1, 4)
        out[image] = boxes.round(2).tolist()
        if n % 500 == 0:
            print(f"hand-eval: detector {n} images", flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out), encoding="utf-8")
    return out


def cmd_hand_eval(a: argparse.Namespace) -> int:
    hands = list(handsets.dexycb(a.root) if a.set == "dexycb" else handsets.hint(a.root))
    if a.limit:
        rng = np.random.default_rng(a.seed)
        hands = [
            hands[i]
            for i in sorted(rng.choice(len(hands), min(a.limit, len(hands)), replace=False))
        ]
    base = a.out / a.set
    boxes = _person_boxes(hands, base / f"det_{len(hands)}_{a.seed}.json")
    arm = arms.POSE_ARMS[a.arm]
    cache = base / arm.name / f"pred_{len(hands)}_{a.seed}.npz"
    if cache.is_file():
        z = np.load(cache)
        kps, scores, covered = z["kps"], z["scores"], z["covered"]
    else:
        model = arms.load(arm, a.device)
        kps, scores = np.zeros((len(hands), 21, 2)), np.zeros((len(hands), 21))
        covered = np.zeros(len(hands), bool)
        for i, h in enumerate(hands):
            box = handsets.person_box(
                np.asarray(boxes[str(h.image)], dtype=float).reshape(-1, 4), h
            )
            if box is None:
                continue
            k, sc = arms.predict(model, cv2.imread(str(h.image)), [box])
            first = 91 if h.side == "L" else 112
            kps[i], scores[i], covered[i] = (
                k[0, first : first + 21],
                sc[0, first : first + 21],
                True,
            )
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, kps=kps, scores=scores, covered=covered)
    errors: dict[str, list[float]] = defaultdict(list)
    vis_scores, occ_scores = [], []
    for i, h in enumerate(hands):
        if not covered[i]:
            continue
        err = np.hypot(*(kps[i] - h.kpts).T) / h.diagonal
        for j in np.flatnonzero(h.valid):
            errors["occluded" if h.occluded[j] else "visible"].append(float(err[j]))
            (occ_scores if h.occluded[j] else vis_scores).append(float(scores[i, j]))
    summary = {
        "set": a.set,
        "arm": arm.name,
        "hands": len(hands),
        "covered": int(covered.sum()),
        "pck": {
            part: {f"@{t}": float(np.mean(np.asarray(v) < t)) for t in (0.05, 0.1, 0.2)}
            | {"n": len(v)}
            for part, v in errors.items()
        },
        "median_error": {part: float(np.median(v)) for part, v in errors.items()},
        "occlusion_auroc": screen.auroc(np.asarray(vis_scores), np.asarray(occ_scores))
        if occ_scores
        else None,
    }
    (cache.parent / f"summary_{len(hands)}_{a.seed}.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))
    return 0


def cmd_calibrate(a: argparse.Namespace) -> int:
    """M4.3b: the arm's raw score → the reference arm's raw score of equal P(correct)."""
    from pose_estimation import score_calibration

    _, persons = public.load_coco_wholebody(a.coco_wb)
    persons = [p for p in persons if p.box[2] > 1 and p.box[3] > 1]
    arm_z = np.load(a.out / a.arm / "coco_wb_pred.npz")
    ref_z = np.load(a.out / a.reference / "coco_wb_pred.npz")
    raw, correct = screen.calibration_targets(persons, arm_z["kps"], arm_z["scores"])
    ref_raw, ref_correct = screen.calibration_targets(persons, ref_z["kps"], ref_z["scores"])
    curves = {
        g: score_calibration.fit_reference(raw[g], correct[g], ref_raw[g], ref_correct[g])
        for g in raw
    }
    out = score_calibration.CALIBRATION_DIR / f"{a.arm}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(score_calibration.ScoreCalibration(curves).to_json() + "\n", encoding="utf-8")
    for g, c in curves.items():
        print(
            f"calibrate: {a.arm} {g:5s} n={len(raw[g])} correct={correct[g].mean():.3f} "
            f"knots={len(c.x)} raw {c.x[0]:.3f}..{c.x[-1]:.3f} -> {c.y[0]:.3f}..{c.y[-1]:.3f}"
        )
    return 0


#: COCO-WholeBody indices of the labelled arm joints per anatomical side.
_ARM_IDX = {
    "L": {"shoulder": 5, "elbow": 7, "wrist": 9},
    "R": {"shoulder": 6, "elbow": 8, "wrist": 10},
}


def cmd_corpus_gross(a: argparse.Namespace) -> int:
    first = {r["frame"]["key"]: r for r in labels.load(a.labels[0])}
    second = {r["frame"]["key"]: r for r in labels.load(a.labels[1])}
    consensus = {k: labels.consensus(first[k], second[k]) for k in sorted(set(first) & set(second))}
    consensus = {k: c for k, c in consensus.items() if c["patient"]}
    meta = {
        f"{f['event_id']}/{f['camera_name']}#{f['frame']}": f
        for f in json.loads(a.sample.read_text(encoding="utf-8"))
    }
    keys = sorted(consensus)
    arm = arms.POSE_ARMS[a.arm]
    cache = a.out / f"{arm.name}.npz"
    if cache.is_file():
        z = np.load(cache)
        kps, scores = z["kps"], z["scores"]
    else:
        model = arms.load(arm, a.device)
        kps, scores = np.zeros((len(keys), 133, 2)), np.zeros((len(keys), 133))
        by_clip: dict[tuple[str, str], dict[int, int]] = defaultdict(dict)
        for i, k in enumerate(keys):
            m = meta[k]
            by_clip[(m["event_id"], m["camera_name"])][m["frame"]] = i
        for (event_id, camera), wanted in by_clip.items():
            for index, image in sample.read_frames(_media(a.sessions, event_id, camera), wanted):
                i = wanted[index]
                k_, s_ = arms.predict(model, image, [consensus[keys[i]]["patient"]])
                kps[i], scores[i] = k_[0], s_[0]
        a.out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, kps=kps, scores=scores)
    rows = []  # one per (frame, side): subject + metrics
    for i, k in enumerate(keys):
        c, subject = consensus[k], meta[k]["subject"]
        diag = float(np.hypot(c["patient"][2] - c["patient"][0], c["patient"][3] - c["patient"][1]))
        centres = {}
        for side in labels.SIDES:
            sd = c["sides"][side]
            pts = [
                j["xy"] for n, j in sd["joints"].items() if n.startswith("joints.") and "xy" in j
            ]
            if sd["hand"] in ("vis", "part") and pts:
                centres[side] = np.mean(pts, axis=0)
        for side in labels.SIDES:
            sd = c["sides"][side]
            first_idx = 91 if side == "L" else 112
            hand_pred = kps[i, first_idx : first_idx + 21]
            row = {
                "subject": subject,
                "hand": sd["hand"],
                "hand_score": float(scores[i, first_idx : first_idx + 21].mean()),
            }
            if sd["extent"] and sd["hand"] in ("vis", "part"):
                ed = float(
                    np.hypot(sd["extent"][2] - sd["extent"][0], sd["extent"][3] - sd["extent"][1])
                )
                errs = [
                    float(np.hypot(*(hand_pred[labels.HAND_JOINTS[n.split(".", 1)[1]]] - j["xy"])))
                    / ed
                    for n, j in sd["joints"].items()
                    if n.startswith("joints.") and "xy" in j
                ]
                row["hand_err"] = errs
            if len(centres) == 2 and side in centres:
                mine = np.hypot(*(hand_pred.mean(axis=0) - centres[side]))
                other = np.hypot(*(hand_pred.mean(axis=0) - centres["R" if side == "L" else "L"]))
                row["swap"] = bool(other < mine)
            row["arm_err"] = [
                float(np.hypot(*(kps[i, _ARM_IDX[side][n.split(".", 1)[1]]] - j["xy"]))) / diag
                for n, j in sd["joints"].items()
                if n.startswith("body.") and "xy" in j
            ]
            rows.append(row)

    def frac(values, t):
        return float(np.mean(np.asarray(values) < t)) if values else float("nan")

    hand_errs = [e for r in rows for e in r.get("hand_err", [])]
    arm_errs = [e for r in rows for e in r["arm_err"]]
    pos = [r["hand_score"] for r in rows if r["hand"] in ("vis", "part")]
    neg = [r["hand_score"] for r in rows if r["hand"] in ("occ", "oof")]
    swaps = [r["swap"] for r in rows if "swap" in r]
    summary = {
        "arm": arm.name,
        "frames": len(keys),
        "subjects": len({r["subject"] for r in rows}),
        "hand_joints": len(hand_errs),
        "hand_within_0.15": frac(hand_errs, 0.15),
        "hand_within_0.3": frac(hand_errs, 0.3),
        "arm_joints": len(arm_errs),
        "arm_within_0.05": frac(arm_errs, 0.05),
        "arm_within_0.1": frac(arm_errs, 0.1),
        "side_swaps": f"{sum(swaps)}/{len(swaps)}",
        "presence_auroc": screen.auroc(np.asarray(pos), np.asarray(neg)) if pos and neg else None,
        "per_subject_hand_within_0.3": {
            str(s): frac([e for r in rows if r["subject"] == s for e in r.get("hand_err", [])], 0.3)
            for s in sorted({r["subject"] for r in rows})
        },
    }
    (a.out / f"{arm.name}_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_subject_hand_within_0.3"}))
    return 0


def _run_trees(specs: list[str] | None) -> dict[str, Path]:
    """`ARM=DIR` pairs → arm → corpus run tree (`corpus_run_2d.py --model ARM --out DIR`)."""
    out = {}
    for spec in specs or []:
        arm, _, tree = spec.partition("=")
        if not tree:
            raise SystemExit(f"--from-run takes ARM=DIR, got {spec!r}")
        out[arm] = Path(tree)
    return out


def _require_clips(trees: dict[str, Path], clips: set[tuple[str, str]]) -> None:
    """Refuse unless every run tree holds each requested `(event_id, camera)` landmark CSV.

    A missing CSV is a failed or unfinished event, not an empty export: scored, it reads as the
    arm missing every point. The message names arm + count alone (paths and event ids stay out).
    """
    for arm, tree in sorted(trees.items()):
        missing = sum(not (tree / event / f"{camera}.csv").is_file() for event, camera in clips)
        if missing:
            raise SystemExit(
                f"--from-run {arm}: {missing} of {len(clips)} requested clips have no landmark CSV"
                " (failed or unfinished events)"
            )


def _tree_row(tree: Path, event_id: str, camera: str, frame: int) -> dict[str, str] | None:
    """The exported subject row at `frame` of a whole-clip corpus run (frame_idx = source)."""
    from pose_estimation.bench import replay

    return replay.row_at(tree / event_id / f"{camera}.csv", frame)


def _pilot_replays(a: argparse.Namespace, arm: str):
    """Per agreed pilot frame: `(key, sample meta, consensus, exported row, max dim, wall s)`.

    With `--from-run ARM=DIR` the rows come from that full corpus run (the same whole-clip
    production state, wall None); otherwise each clip is replayed here.

    Each pilot clip is replayed whole, frame 0 to its last decoded frame — the corpus run of
    that clip under `arm`. M4.1 D05(e) measured no finite pre-roll reaching that state: hands
    converge by 4 s, but the sticky subject choice still differs on 4/20 frames at 8 s
    (contract-m4u1 A05). Clips cache under `<out>/<arm>/whole/<sha256(clip)[:16]>/`; wall =
    the clip's replay seconds over its sampled frames.
    """
    from pose_estimation.bench import pilot, replay

    try:
        pilot.check_request(arm)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    first = {r["frame"]["key"]: r for r in labels.load(a.labels[0])}
    second = {r["frame"]["key"]: r for r in labels.load(a.labels[1])}
    meta = {
        f"{f['event_id']}/{f['camera_name']}#{f['frame']}": f
        for f in json.loads(a.sample.read_text(encoding="utf-8"))
    }
    keys = sorted(set(first) & set(second))
    tree = _run_trees(a.from_run).get(arm)
    if tree is not None:
        _require_clips({arm: tree}, {(meta[k]["event_id"], meta[k]["camera_name"]) for k in keys})
        for key in keys:
            m = meta[key]
            row = _tree_row(tree, m["event_id"], m["camera_name"], m["frame"])
            size = max(first[key]["frame"]["width"], first[key]["frame"]["height"])
            yield key, m, labels.consensus(first[key], second[key]), row, size, None
        return
    by_clip: dict[str, list[str]] = defaultdict(list)
    for key in keys:
        by_clip[key.split("#", 1)[0]].append(key)
    for clip, clip_keys in sorted(by_clip.items()):
        m0 = meta[clip_keys[0]]
        diag = a.run / m0["event_id"] / f"{m0['camera_name']}_diag.csv"
        n = int(_diag_row(diag)["n_frames_decoded"])
        window = a.out / arm / "whole" / hashlib.sha256(clip.encode()).hexdigest()[:16]
        media = _media(a.sessions, m0["event_id"], m0["camera_name"])
        done = window / f"{media.stem}.csv"
        wall_file = window / "wall.json"
        if not wall_file.is_file():
            t0 = time.perf_counter()
            replay.replay_window(media, 0, n - 1, window, extra=("--model", arm))
            wall_file.write_text(json.dumps({"seconds": time.perf_counter() - t0}))
        if not done.is_file():
            raise SystemExit(
                f"{arm}: a pilot clip replay wrote no landmark CSV (see its replay.log)"
            )
        wall = json.loads(wall_file.read_text())["seconds"] / len(clip_keys)
        for key in clip_keys:
            m = meta[key]
            row = replay.row_at(done, m["frame"])
            size = max(first[key]["frame"]["width"], first[key]["frame"]["height"])
            yield key, m, labels.consensus(first[key], second[key]), row, size, wall


def cmd_presence(a: argparse.Namespace) -> int:
    rows: list[dict] = []
    for _, m, c, row, _, _ in _pilot_replays(a, a.arm):
        states = {side: c["sides"][side]["hand"] for side in labels.SIDES}
        rows.extend(qualify.presence_rows(m["subject"], states, qualify.exported_hands(row)))
    summary = {"arm": a.arm, "rows_from": "run" if a.arm in _run_trees(a.from_run) else "replay"}
    summary |= qualify.presence_summary(rows, BOOT_SEED_PRESENCE)
    (a.out / a.arm).mkdir(parents=True, exist_ok=True)
    (a.out / a.arm / "presence.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=1)
    )
    print(json.dumps(summary))
    return 0


def cmd_pilot_eval(a: argparse.Namespace) -> int:
    """M4.3: the arm and the baseline replayed over the pilot → G1-G4, paired intervals, state."""
    from pose_estimation.bench import pilot

    rows, walls = {}, {}
    for arm in (a.arm, a.baseline):
        rows[arm], walls[arm] = [], []
        for _, m, c, row, size, wall in _pilot_replays(a, arm):
            rows[arm] += pilot.side_rows(m["subject"], c, row, size)
            if wall is not None:
                walls[arm].append(wall)
    intervals = pilot.paired_intervals(rows[a.arm], rows[a.baseline], a.n, BOOT_SEED_PILOT)
    # M4.1 D05(d): per-subject presence F1 difference → its SD over subjects (M4.4 power input).
    f1 = {
        arm: {
            s: qualify.presence_f1([r for r in rs if r["subject"] == s and "present" in r])
            for s in {r["subject"] for r in rs}
        }
        for arm, rs in rows.items()
    }
    diffs = {s: [f1[a.arm][s] - f1[a.baseline][s]] for s in f1[a.arm] if s in f1[a.baseline]}
    sd, n_sd = qualify.paired_subject_sd(diffs) if len(diffs) >= 2 else (None, len(diffs))
    report = {
        "arm": a.arm,
        "baseline": a.baseline,
        "endpoints": {arm: pilot.endpoints(r) for arm, r in rows.items()},
        "intervals": {
            g: dict(zip(("point", "lo", "hi"), v, strict=True)) for g, v in intervals.items()
        },
        "state": pilot.decide(intervals),
        "wall_s_per_replay": {arm: float(np.median(w)) if w else None for arm, w in walls.items()},
        "subjects": len({r["subject"] for r in rows[a.baseline]}),
        "sides": {arm: len(r) for arm, r in rows.items()},
        "presence_f1_paired_subject_sd": {"sd": sd, "subjects": n_sd},
        # Arm names alone: a run-tree path stays out of stdout + reports (data boundary).
        "rows_from": sorted(_run_trees(a.from_run)) or "replay",
    }
    out = a.out / a.arm / "pilot_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report | {"rows": rows}, indent=1), encoding="utf-8")
    print(json.dumps(report))
    return 0


BOOT_SEED_PILOT = 20261023


def cmd_confirm(a: argparse.Namespace) -> int:
    """M4.4: the user's frames vs each candidate's and the baseline's corpus-run rows."""
    from pose_estimation.bench import confirm

    try:
        candidates = confirm.frozen_candidates(a.contract)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    trees = _run_trees(a.from_run)
    missing = sorted((set(candidates) | {a.baseline}) - set(trees))
    if missing:
        raise SystemExit(f"no --from-run tree for {missing}")
    jobs = {
        f"{j['event_id']}/{j['camera_name']}#{j['frame']}": j
        for j in json.loads(a.jobs.read_text(encoding="utf-8"))
    }
    records = list(labels.load(a.labels))  # every arm scores the same records
    _require_clips(
        {arm: trees[arm] for arm in (*candidates, a.baseline)},
        {
            (jobs[r["frame"]["key"]]["event_id"], jobs[r["frame"]["key"]]["camera_name"])
            for r in records
        },
    )
    scores = {}
    for arm in (*candidates, a.baseline):
        frames = []
        for record in records:
            job = jobs[record["frame"]["key"]]
            row = _tree_row(trees[arm], job["event_id"], job["camera_name"], job["frame"])
            frames.append(confirm.score_frame(record, row, str(job["subject"])))
        scores[arm] = confirm.subject_means(frames)
    states, intervals = {}, {}
    for arm in candidates:
        intervals[arm] = {
            e: confirm.bootstrap(scores[arm], scores[a.baseline], e)
            for e in ("E1", "E2", "E3", "E4")
        }
        states[arm] = confirm.decide(intervals[arm])
    adopted = confirm.adopt(states, a.public_rank, a.baseline)
    report = {
        "candidates": candidates,
        "baseline": a.baseline,
        "frames": len(records),
        "subjects": len(scores[a.baseline]),
        "intervals": intervals,
        "states": states,
        "adopted": adopted,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report))
    return 0


BOOT_SEED_PRESENCE = 20261022


def cmd_screen(a: argparse.Namespace) -> int:
    """M4.2 D03: paired bootstraps of COCO-WB hand AP + DexYCB hand PCK@0.1 → the finalist set."""
    import subprocess

    evaluator = Path(__file__).resolve().parents[1] / "tools" / "cocowb-eval"
    specs = [f"{arm}={(a.public / arm / 'coco_wb_results.json').resolve()}" for arm in a.arms]
    hand_ap = json.loads(
        subprocess.run(
            ["uv", "run", "--quiet", "--directory", str(evaluator), "python", "bootstrap.py",
             str(a.coco_wb.resolve()), *specs, "--n", str(a.n), "--seed", str(a.seed)],
            check=True, capture_output=True, text=True,
        ).stdout
    )  # fmt: skip
    hands = list(handsets.dexycb(a.dexycb))
    valid = np.stack([h.valid for h in hands])
    correct = {}
    for arm in a.arms:
        z = np.load(a.handsets / "dexycb" / arm / f"pred_{len(hands)}_{a.hand_seed}.npz")
        diag = np.array([h.diagonal for h in hands])
        err = np.hypot(*(z["kps"] - np.stack([h.kpts for h in hands])).transpose(2, 0, 1))
        correct[arm] = (err / diag[:, None] < 0.1) & z["covered"][:, None]
        covered = z["covered"]
    dexycb = screen.paired_pck_bootstrap(correct, valid & covered[:, None], a.n, a.seed)
    arm_pck = {
        arm: json.loads((a.public / arm / "coco_wb_summary.json").read_text(encoding="utf-8"))[
            "arm_pck"
        ]["@0.05"]
        for arm in a.arms
    }
    verdicts = screen.finalists(arm_pck, [hand_ap["pairs"], dexycb["pairs"]], a.baseline)
    report = {
        "hand_ap": hand_ap,
        "dexycb_pck_0.1": dexycb
        | {"hands": int(covered.sum()), "joints": int(valid[covered].sum())},
        "arm_pck_0.05": arm_pck,
        "verdicts": verdicts,
        "finalists": sorted(arm for arm, reasons in verdicts.items() if not reasons),
    }
    a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    for arm in a.arms:
        ap_, dx = hand_ap["arms"][arm], dexycb["arms"][arm]
        print(
            f"screen: {arm:16s} handAP {ap_['mean']:.4f} [{ap_['interval'][0]:.4f}, "
            f"{ap_['interval'][1]:.4f}]  DexYCB {dx['pck']:.4f} [{dx['interval'][0]:.4f}, "
            f"{dx['interval'][1]:.4f}]  armPCK {arm_pck[arm]:.4f}  "
            f"{'FINALIST' if not verdicts[arm] else '; '.join(verdicts[arm])}"
        )
    return 0


def cmd_agree(a: argparse.Namespace) -> int:
    first = {r["frame"]["key"]: r for r in labels.load(a.labels[0])}
    second = {r["frame"]["key"]: r for r in labels.load(a.labels[1])}
    keys = sorted(set(first) & set(second))
    fields: dict[str, tuple[list[str], list[str]]] = defaultdict(lambda: ([], []))
    hand_err: dict[str, list[float]] = defaultdict(list)
    body_err: dict[str, list[float]] = defaultdict(list)
    patient_iou = []
    for key in keys:
        r1, r2 = first[key], second[key]
        fields["task"][0].append(r1["task"])
        fields["task"][1].append(r2["task"])
        if r1["patient"] and r2["patient"]:
            patient_iou.append(qualify.box_iou(r1["patient"], r2["patient"]))
        diag = None
        if r1["patient"] and r2["patient"]:
            b = r1["patient"]
            diag = float(np.hypot(b[2] - b[0], b[3] - b[1]))
        for side in labels.SIDES:
            s1, s2 = r1["sides"][side], r2["sides"][side]
            fields["hand"][0].append(s1["hand"])
            fields["hand"][1].append(s2["hand"])
            ext = s1.get("extent") or s2.get("extent")
            for name in labels.HAND_JOINTS:
                j1, j2 = s1["joints"].get(name, {}), s2["joints"].get(name, {})
                fields["hand_joint"][0].append(j1.get("state", "?"))
                fields["hand_joint"][1].append(j2.get("state", "?"))
                if "xy" in j1 and "xy" in j2 and ext:
                    d = float(np.hypot(ext[2] - ext[0], ext[3] - ext[1]))
                    hand_err[name].append(float(np.hypot(*np.subtract(j1["xy"], j2["xy"]))) / d)
            for name in labels.BODY_JOINTS:
                j1, j2 = s1["body"].get(name, {}), s2["body"].get(name, {})
                fields["body_joint"][0].append(j1.get("state", "?"))
                fields["body_joint"][1].append(j2.get("state", "?"))
                if "xy" in j1 and "xy" in j2 and diag:
                    body_err[name].append(float(np.hypot(*np.subtract(j1["xy"], j2["xy"]))) / diag)
    print(f"agree: {len(keys)} frames labelled by both")
    for name, (x, y) in fields.items():
        k, n = qualify.agreement(x, y)
        lo, hi = qualify.wilson(k, n)
        confusion: dict[tuple[str, str], int] = defaultdict(int)
        for pair in zip(x, y, strict=True):
            confusion[pair] += 1
        print(
            f"agree: {name:11s} {k}/{n} = {k / n:.3f} (Wilson {lo:.3f}-{hi:.3f}) {dict(sorted(confusion.items()))}"
        )
    if patient_iou:
        print(
            f"agree: patient box IoU >= 0.5 on {sum(i >= 0.5 for i in patient_iou)}/{len(patient_iou)}"
        )
    for label, table in (("hand / extent diag", hand_err), ("arm / patient diag", body_err)):
        for name, v in sorted(table.items()):
            p50, p90 = qualify.quantiles(v)
            print(f"agree: {label} {name:11s} n={len(v):3d} p50={p50:.3f} p90={p90:.3f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--sessions", type=Path, default=Path("sessions"))
    s.add_argument("--inventory", type=Path, default=Path("inventory"))
    s.add_argument("--run", type=Path, default=Path("output/corpus-2d"))
    s.add_argument("--watch", type=Path, required=True, help="JSON list of watch-set clip labels")
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--subjects", type=int, default=8)
    s.add_argument("--per-view", type=int, default=3)
    s.add_argument("--stress", type=int, default=30)
    s.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("frame-sheets")
    s.add_argument("--sample", type=Path, required=True)
    s.add_argument("--sessions", type=Path, default=Path("sessions"))
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--max-side", type=int, default=1280)
    s.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("crop-sheets")
    s.add_argument("--sheets", type=Path, required=True)
    s.add_argument("--answers", type=Path, nargs="+", required=True)
    s.add_argument("--sessions", type=Path, default=Path("sessions"))
    s.add_argument("--seed", type=int, required=True)
    s = sub.add_parser("ingest")
    s.add_argument("--sheets", type=Path, required=True)
    s.add_argument("--answers", type=Path, nargs="+", required=True)
    s.add_argument("--labeller", required=True)
    s.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("public-sheets")
    s.add_argument("--coco-wb", type=Path, required=True)
    s.add_argument("--images", type=Path, required=True)
    s.add_argument("--hands", type=int, default=60)
    s.add_argument("--persons", type=int, default=40)
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--gt", type=Path, required=True, help="sealed GT file, outside --out")
    s = sub.add_parser("public-score")
    s.add_argument("--sheets", type=Path, required=True)
    s.add_argument("--gt", type=Path, required=True)
    s.add_argument("--answers", type=Path, nargs="+", required=True)
    s = sub.add_parser("judge-public")
    s.add_argument("--coco-wb", type=Path, required=True)
    s.add_argument("--images", type=Path, required=True)
    s.add_argument("--exclude", type=Path, help="sheet dir whose hands stay out (labeller set)")
    s.add_argument("--arm-a", default="rtmw-m-256", choices=sorted(arms.POSE_ARMS))
    s.add_argument("--arm-b", default="rtmw-x-384", choices=sorted(arms.POSE_ARMS))
    s.add_argument("--min-diag", type=float, default=40.0)
    s.add_argument("--pool", type=int, default=400)
    s.add_argument("--per-bin", type=int, default=40)
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--key", type=Path, required=True, help="sealed key, outside --out")
    s.add_argument("--truth", type=Path, required=True, help="sealed truth, outside --out")
    s = sub.add_parser("judge-score")
    s.add_argument("--key", type=Path, required=True)
    s.add_argument("--truth", type=Path, required=True)
    s.add_argument("--answers", type=Path, nargs="+", required=True)
    s = sub.add_parser("user-sample")
    s.add_argument("--sessions", type=Path, default=Path("sessions"))
    s.add_argument("--inventory", type=Path, default=Path("inventory"))
    s.add_argument("--run", type=Path, default=Path("output/corpus-2d"))
    s.add_argument("--watch", type=Path, required=True)
    s.add_argument("--exclude", type=Path, required=True, help="pilot sample JSON")
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--subjects", type=int, default=16)
    s.add_argument("--per-view", type=int, default=2)
    s.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("progress")
    s.add_argument("--sheets", type=Path, required=True)
    s.add_argument("--answers", type=Path, nargs="+", required=True)
    s.add_argument("--kind", nargs="*", choices=["frame", "body", "hand"])
    s = sub.add_parser("public-eval")
    s.add_argument("--coco-wb", type=Path, required=True)
    s.add_argument("--images", type=Path, required=True)
    s.add_argument("--arm", required=True, choices=sorted(arms.POSE_ARMS))
    s.add_argument("--device", default="GPU")
    s.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("calibrate")
    s.add_argument("--coco-wb", type=Path, required=True)
    s.add_argument("--arm", required=True, choices=sorted(arms.POSE_ARMS))
    s.add_argument("--reference", default="rtmw-m-256", choices=sorted(arms.POSE_ARMS))
    s.add_argument("--out", type=Path, default=Path(".scratch/bench/public"))
    s = sub.add_parser("corpus-gross")
    s.add_argument("--labels", type=Path, nargs=2, required=True)
    s.add_argument("--sample", type=Path, required=True)
    s.add_argument("--sessions", type=Path, default=Path("sessions"))
    s.add_argument("--arm", required=True, choices=sorted(arms.POSE_ARMS))
    s.add_argument("--device", default="GPU")
    s.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("presence")
    s.add_argument("--labels", type=Path, nargs=2, required=True)
    s.add_argument("--from-run", action="append", metavar="ARM=DIR", help="read a corpus run")
    s.add_argument("--sample", type=Path, required=True)
    s.add_argument("--run", type=Path, default=Path("output/corpus-2d"))
    s.add_argument("--sessions", type=Path, default=Path("sessions"))
    s.add_argument("--arm", required=True, help="run.MODEL_REGISTRY key")
    s.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("confirm")
    s.add_argument("--labels", type=Path, required=True, help="the user's /label records")
    s.add_argument("--jobs", type=Path, required=True, help="user-sample jobs.json")
    s.add_argument("--contract", type=Path, default=Path(".agent/archive/contract-m4u4.md"))
    s.add_argument("--from-run", action="append", required=True, metavar="ARM=DIR")
    s.add_argument("--baseline", default="rtmw-m-256")
    s.add_argument("--public-rank", nargs="+", required=True, help="candidates, M4.2 order")
    s.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("pilot-eval")
    s.add_argument("--labels", type=Path, nargs=2, required=True)
    s.add_argument("--from-run", action="append", metavar="ARM=DIR", help="read a corpus run")
    s.add_argument("--sample", type=Path, required=True)
    s.add_argument("--run", type=Path, default=Path("output/corpus-2d"))
    s.add_argument("--sessions", type=Path, default=Path("sessions"))
    s.add_argument("--arm", required=True, help="run.MODEL_REGISTRY key")
    s.add_argument("--baseline", default="rtmw-m-256")
    s.add_argument("--n", type=int, default=2000)
    s.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("agree")
    s.add_argument("--labels", type=Path, nargs=2, required=True)
    s = sub.add_parser("hand-eval")
    s.add_argument("--set", required=True, choices=["dexycb", "hint"])
    s.add_argument("--root", type=Path, required=True)
    s.add_argument("--arm", required=True, choices=sorted(arms.POSE_ARMS))
    s.add_argument("--device", default="GPU")
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--seed", type=int, default=20261019)
    s.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("screen")
    s.add_argument("--coco-wb", type=Path, required=True)
    s.add_argument("--public", type=Path, required=True, help="public-eval --out")
    s.add_argument("--dexycb", type=Path, required=True, help="hand-eval --root of the dexycb set")
    s.add_argument("--handsets", type=Path, required=True, help="hand-eval --out")
    s.add_argument("--hand-seed", type=int, default=20261019)
    s.add_argument("--arms", nargs="+", required=True)
    s.add_argument("--baseline", default="rtmw-m-256")
    s.add_argument("--n", type=int, default=1000)
    s.add_argument("--seed", type=int, default=20261019)
    s.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("preroll")
    s.add_argument("--sample", type=Path, required=True)
    s.add_argument("--run", type=Path, default=Path("output/corpus-2d"))
    s.add_argument("--sessions", type=Path, default=Path("sessions"))
    s.add_argument("--seconds", type=float, nargs="+", default=[0, 1, 2, 4, 8])
    s.add_argument("--frames", type=int, default=20)
    s.add_argument("--scratch", type=Path, required=True)
    a = p.parse_args(argv)
    return {
        "sample": cmd_sample,
        "frame-sheets": cmd_frame_sheets,
        "crop-sheets": cmd_crop_sheets,
        "ingest": cmd_ingest,
        "public-sheets": cmd_public_sheets,
        "public-score": cmd_public_score,
        "judge-public": cmd_judge_public,
        "judge-score": cmd_judge_score,
        "progress": cmd_progress,
        "user-sample": cmd_user_sample,
        "preroll": cmd_preroll,
        "hand-eval": cmd_hand_eval,
        "agree": cmd_agree,
        "presence": cmd_presence,
        "screen": cmd_screen,
        "pilot-eval": cmd_pilot_eval,
        "confirm": cmd_confirm,
        "corpus-gross": cmd_corpus_gross,
        "calibrate": cmd_calibrate,
        "public-eval": cmd_public_eval,
    }[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
