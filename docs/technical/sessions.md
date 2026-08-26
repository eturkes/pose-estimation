# Session tree

`pose-estimation-sessions` turns the corpus registry into a tree of recording events. Each event becomes one directory that `multicam.discover_session` reads. The tool never walks the corpus. It reads the registry that `pose-estimation-inventory` published, and it opens only the files those rows name.

## Run the generator

```bash
pose-estimation-sessions --inventory inventory --corpus videos/3-cam --out sessions
pose-estimation-sessions --strict
python -m pose_estimation.sessions --out sessions
```

Use `--inventory` to select the published registry. Use `--corpus` to select the root that the registry's relative paths resolve against. Use `--out` to select the output directory. Use `--strict` to return status 1 when the tool holds any asset out. Status 2 reports a usage or registry error.

The registry stores corpus-relative paths and omits the root, so `--corpus` is required information. The defaults are `inventory`, `videos/3-cam`, and `sessions`.

## Identity

An event is one performance of one task and one side by one subject. One to three hand-started cameras record it.

```text
event_id = f"{capture_id}_run-{run_index:02d}"        s02-cap-l_run-01
```

**No `event_id` can equal a `capture_id`.** A `capture_id` names a task-side family, and a family can hold more than one physical take. Calibration must bind to an event, never to a family. The `_run-` component makes that mistake unrepresentable instead of merely forbidden.

`run-<index>` follows the BIDS entity of the same name. BIDS defines a run as one uninterrupted acquisition, and it requires an index when two acquisitions share every other entity. A retake is exactly that case.

`run_index` orders the events of one family. It is not a time and it asserts no provenance. The order comes from the registry's own row order, which makes it deterministic.

The index field is exactly two digits. A family that needs more than 99 runs raises `SessionsError` instead of emitting an identifier that fails the pattern above.

## Take resolution

Each event carries `take_resolution`.

| Value | Meaning |
| --- | --- |
| `family` | The registry shows one take, so every view of the family joins one event. |
| `unresolved` | The registry shows more than one take, and no evidence assigns a view to a take. |

The generator asserts no multi-camera event for a family with `view_conflict=1`. Each asset of that family becomes its own single-camera event. **Do not read the run count of an unresolved family as a performance count.** The true grouping stays unknown, and the per-asset events understate it.

No published multi-camera pipeline infers same-take membership from a filename, a file order, a duration, a frame count, or a creation time. Take membership comes from acquisition metadata or from an operator. Temporal alignment is a separate later step, and it needs a decoded shared signal.

Unequal frame counts across views stay compatible with one event after offset estimation. They are evidence of neither sameness nor difference.

## Placement

Every discovered asset reaches exactly one outcome. A hold-out is a qualification verdict on an asset that the registry described correctly. The generator holds an asset out for these reasons only.

| Reason | Meaning |
| --- | --- |
| `quarantined_stem` | The registry could not parse a family from the name. |
| `excluded_asset` | The registry excluded the entry before parsing. |
| `extension_not_discoverable` | The suffix is outside `multicam.VIDEO_EXTENSIONS`. |

`inventory` admits `.flv` and `multicam` does not, so `extension_not_discoverable` prevents an event that discovery could never read.

A canonical row that this tool cannot decode or resolve is a different case. The registry disagrees with the corpus, so the run fails with status 2 and publishes nothing. These conditions raise:

- The `source_path` cell is absolute, empty, traversing, or carries a NUL.
- The `source_path` cell carries an escape that `inventory` never writes.
- The listed path resolves outside `--corpus`.
- The listed path is not a regular file.

A hold-out here would drop one camera from an event and then publish the smaller event as if it were whole. Publish the registry again to correct any of these.

Active corpus: **193 events from 382 assets** — 58 one-camera, 84 two-camera, and 51 three-camera. 186 events resolve as `family` and 7 as `unresolved`. 379 assets reach an event and 3 stay quarantined.

## Session directory

Each event directory holds one `session.json` and one symbolic link per camera.

```json
{
  "format_version": 1,
  "session_id": "s02-cap-l_run-01",
  "capture_id": "s02-cap-l",
  "run_index": 1,
  "subject_ordinal": "2",
  "task": "cap",
  "side": "l",
  "take_resolution": "family",
  "n_cameras": 1,
  "grammar_version": "v1",
  "generator_version": "v1",
  "cameras": [
    {"name": "cam-above", "sync_offset": 0, "view": "above",
     "asset_id": "a-0123456789abcdef", "content_sha256": "…"}
  ]
}
```

`session_id` carries the `event_id`, because that is the field name the consumer already uses. The project vocabulary says `event_id` everywhere else. `session_id` stays reserved for a later visit-level grouping.

Rules that the manifest must keep:

- **No camera declares `file`.** `_safe_resolve` follows the symbolic link and then rejects the target for leaving the session directory. A manifest without `file` finds the same link and accepts it.
- **The camera name starts with `cam-`.** The glob path matches `cam*`, so the tree stays discoverable when a manifest is absent, and both paths report the same camera names.
- **The link name ends with a lowercase suffix.** Discovery matches lowercase suffixes only, and 380 of 382 sources end `.MOV`.
- **The link target is relative.** The container and the host reach this checkout through different absolute paths.
- **`sync_offset` is 0 and unmeasured.** It asserts nothing about starts or rates. M2.5 owns alignment.
- **No manifest declares `calibration`.** A default reference would bind calibration before evidence exists.

## Artifacts

| Artifact | Grain | Handling rule |
| --- | --- | --- |
| `<event_id>/` | One recording event | Keep it local because link targets contain source paths. |
| `events.csv` | One event | Keep it local because it supports linkage across a family. |
| `placements.csv` | One discovered asset | Keep it local for the same reason. |
| `generation.json` | The published tree | Keep it local. It also marks the tree as owned. |

`events.csv` sorts by `event_id` and carries `event_id`, `capture_id`, `subject_ordinal`, `task`, `side`, `run_index`, `take_resolution`, `n_cameras`, `views`, `view_conflict`, `grammar_version`, and `generator_version`.

`placements.csv` sorts by `asset_id` and carries `asset_id`, `capture_id`, `disposition`, `placement`, `placement_reason`, `event_id`, and `camera_name`. It holds one row for every registry row, so no asset leaves the tree silently.

`generation.json` records the digest of each table, a digest of the tree, and the upstream `inventory` generation block.

The tree digest covers every entry under the output directory except `generation.json`, which cannot digest itself. For each entry it records the relative name, the kind, and either the exact symbolic link target text or the SHA-256 of the file bytes. It never reads a link target's contents, so corpus bytes stay outside the digest. It excludes inode, mtime, and permissions, because those are not a function of the registry.

The kind of each entry is load-bearing. A directory test follows a symbolic link, so an event directory that is replaced by a link to an outside directory would otherwise digest as the directory it points at. Recording the kind also catches an injected directory and an unexplained root file.

## Publication and validation

The generator builds the complete tree in a sibling staging directory. It then renames the old tree aside and renames the new tree into place. Staging and retiring directories are always siblings of `--out`. `discover_sessions` reads the children of the root, so a half-built directory under the root would become a discoverable session.

A kill between the two renames leaves both siblings on disk. Each carries the process identifier that made it, so the next run removes the siblings whose process is gone and leaves a concurrent generator's siblings alone.

The generator refuses a non-empty `--out` unless it holds a `generation.json` that this tool wrote. The marker must read as UTF-8, parse as a JSON object, and carry `generator_version`. A missing, unreadable, malformed, or foreign marker stops the run before any write, so the tool never deletes a directory that it does not own. Ownership never depends on the digests: a tree whose digests went stale must stay regenerable, and `validate_generation` is the function that rejects staleness.

```python
from pose_estimation.sessions import validate_generation

generation = validate_generation("sessions", inventory_dir="inventory")
```

Every consumer must call `validate_generation` before it reads a row or opens a camera. The function returns the generation block, or it raises `SessionsError`.

`generation.json` **is** the block. The registry's equivalent is nested inside `census.json` because that file also carries aggregates; this file carries the block alone.

Its schema is closed. The key set must be exactly `events.csv`, `placements.csv`, `tree`, `inventory`, and `generator_version`, and `generator_version` must match this generator. An added, renamed, or missing key means another writer or an edit, which no digest inside the document can catch.

Pass `inventory_dir` to also prove that the registry on disk is the generation that produced the tree. That check is the only one that catches a registry rebuilt under a tree which still looks internally consistent. A consumer that omits the argument gets no protection against that case. The argument is optional because the tree must stay consumable after it moves away from the registry.

Unchanged inputs produce a byte-identical tree. The generator verifies this against a shuffled directory order, four locales, a changed hash seed, a changed time zone, and a different `--out` name.

## Data boundary

The tree is patient-adjacent. Link targets, manifests, and both tables contain source paths. `.gitignore` covers `sessions` and `sessions.*`, and the read-exclusion list covers `sessions/`. Console output reports counts only.

See `entrypoints.md` for exit codes, `inventory.md` for the registry schema, and `multicam.md` for the discovery contract.
