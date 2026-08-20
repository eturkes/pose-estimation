# Corpus inventory

`pose-estimation-inventory` inventories entries, parses admitted names, probes media headers, and optionally computes full-file SHA-256 values. It never calls `VideoCapture.read` or `VideoCapture.grab`, so it never decodes a frame. With checksums enabled, the active corpus requires approximately 18.3 GB of source-byte reads.

## Run the inventory

```bash
pose-estimation-inventory --corpus videos/3-cam --out inventory
pose-estimation-inventory --corpus videos/3-cam --out inventory --no-checksums
pose-estimation-inventory --corpus videos/3-cam --out inventory --strict
python -m pose_estimation.inventory --corpus videos/3-cam --out inventory
```

Use `--corpus` to select the recursively searched directory. Use `--out` to select the output, which defaults to `inventory`. Use `--no-checksums` to skip full-byte SHA-256 scans. Use `--strict` to return status 1 when any asset is not canonical. The output must sit outside the corpus. The discovery layer also skips the output for library callers.
The module defaults `OPENCV_FFMPEG_LOGLEVEL` to `-8` before any probe opens. FFmpeg writes native diagnostics directly to stderr, and some diagnostics quote the source URL. If you need native FFmpeg diagnostics, set `OPENCV_FFMPEG_LOGLEVEL` before the command.
`--out` follows the operating umask; the tool sets no file mode. `assets.csv` carries corpus-relative paths, so `--out` is as sensitive as the corpus.

## Data boundary

| Artifact | Grain | Handling rule |
| --- | --- | --- |
| `assets.csv` | One discovered entry | Keep it local because it contains source paths and stable pseudonyms. |
| `captures.csv` | One canonical task-side family | Keep it local because it supports linkage across family members. |
| `census.json` | Aggregate corpus | Quote its values when you need redaction-safe corpus numbers. |

Success and handled-error text do not print filesystem paths. Every consumer must call `validate_generation(out_dir)` before reading any row.

## Identity and claim boundary

`asset_id` identifies one entry through its true corpus-relative path bytes. It remains stable for that path, while a rename or move changes it.

```text
asset_id = "a-" + blake2b(relative_path.encode("utf-8", "surrogateescape"),
                          digest_size=8, person=b"pose3cam-asset").hexdigest()
```

The identifier contains a 64-bit digest and is unique with overwhelming probability. It is not unique by construction. The explicit collision check refuses publication when two rows share one `asset_id`.
`capture_id` is a task-side family key for one subject, one task, and one side. It is not a physical-take key. The stable identity is the pair `(grammar_version, capture_id)`. A grammar migration can change family membership while the readable key survives.

```text
capture_id = f"s{subject_ordinal:02d}-{task}-{side}"
```

`capture_id` is a low-entropy stable pseudonym. It supports linkage and remains enumerable. The subject tokens `3` and `03` intentionally map to the same ordinal and family key. The grammar sets no subject-ordinal range.
A family with `view_conflict=1` contains more than one physical take. If a consumer needs one recording event, reject or resolve every conflicting family. The active corpus has 188 families, including two conflicts. Exactly one asset has a repeat marker, and it belongs to one conflicting family.
`view` is a lexical filename label only. Header agreement establishes no geometry, synchronization, fixed-rig provenance, calibration, 3D recoverability, or metric scale.

## Filename grammar

The parser applies these steps to the full filename in order. The synthetic filename `4_above_peg_l.MOV` is canonical.

1. N1 removes each trailing supported suffix without case sensitivity.
2. N2 converts the remaining text to lowercase.
3. N3 removes leading and trailing whitespace.
4. N4 replaces each whitespace run with one underscore.
5. N5 replaces each underscore run with one underscore.
6. N6 removes leading underscores but retains a trailing underscore.
7. N7 consumes a final `(<digits>)` marker when its integer exceeds zero.
8. N8 splits the remaining stem on underscores.
9. N9 repairs the task token through the closed spelling table.

The suffixes are `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`, and `.flv`. A consumed marker sets `repeat` to its integer. A canonical name without one sets `repeat` to 0. The marker `(0)` is invalid. A remaining final `)` produces `repeat_marker_unrecognized`.
The parser requires `<subject_ordinal>_<view>_<task>_<side>`. The subject token must contain ASCII digits only. The view must be `above`, `left`, or `right`. The side must be `l` or `r`. The task must be `cap`, `coin`, `glass`, `key`, `nut`, or `peg`.

| Input task | Canonical task |
| --- | --- |
| `coini` | `coin` |
| `gcap` | `cap` |
| `gpeg` | `peg` |
| `grass` | `glass` |

These four repairs cover 15 assets in the 382-asset active corpus. No unknown task token remains in that corpus.
The normalization vocabulary contains `case_folded`, `leading_separator_stripped`, `media_suffix_doubled`, and `outer_trimmed`. It also contains `repeat_marker`, `task_repaired`, `underscore_collapsed`, and `whitespace_collapsed`. Each label records a transform that changed the text. Every multi-valued CSV cell joins values with `|`.
The subject ordinal must stay constant within each immediate parent. One ordinal must also belong to only one parent. The tool quarantines all affected canonical candidates when either rule fails. A root-level file uses `.` as its parent grain.

## Dispositions and reason codes

Each asset has one disposition and one reason code. Invalid container facts do not change the disposition.

| Disposition | Reason code | Meaning |
| --- | --- | --- |
| `canonical` | `ok` | The entry was admitted, the container opened, and the grammar succeeded. |
| `quarantined` | `repeat_marker_unrecognized` | The stem ends with an invalid parenthesized repeat marker. |
| `quarantined` | `side_missing` | The fourth token exists but is empty. |
| `quarantined` | `side_unknown` | The side is nonempty and outside the closed vocabulary. |
| `quarantined` | `subject_token_conflict` | A parent and subject ordinal fail the corpus cross-check. |
| `quarantined` | `subject_token_nonnumeric` | The subject token contains a non-ASCII digit or another character. |
| `quarantined` | `task_unknown` | The repaired task is outside the closed vocabulary. |
| `quarantined` | `token_count` | The normalized stem does not contain exactly four tokens. |
| `quarantined` | `view_unknown` | The view is outside the closed vocabulary. |
| `excluded` | `broken_symlink` | The symbolic link has no target. |
| `excluded` | `control_character_in_path` | The relative path contains a C0, DEL, or C1 control character. |
| `excluded` | `not_a_regular_file` | The entry does not resolve to a regular file. |
| `excluded` | `path_escapes_root` | The entry resolves outside the corpus root. |
| `excluded` | `path_not_utf8` | The path bytes are not valid UTF-8. |
| `excluded` | `probe_unreadable` | OpenCV cannot open the admitted media file. |
| `excluded` | `read_error` | A size read or enabled SHA-256 scan failed. |
| `excluded` | `symlink_within_corpus` | The symbolic link is a second reference to an in-corpus file. |
| `excluded` | `unsupported_extension` | The final suffix is not a supported media suffix. |

The first matching exclusion wins in this order:

1. `control_character_in_path`
2. `path_not_utf8`
3. `broken_symlink`
4. `path_escapes_root`
5. `symlink_within_corpus`
6. `not_a_regular_file`
7. `unsupported_extension`
8. `read_error`
9. `probe_unreadable`

The first seven reasons exclude before parsing and probing. Their `normalizations` cells are empty because the parser did not run. `read_error` and `probe_unreadable` occur after parsing and retain the normalization trace. An excluded row never participates in subject or family cross-checks.
`source_path` escapes a backslash first, then renders each unrenderable byte as `\xNN`. A control code point renders the `\xNN` of each of its own UTF-8 bytes. A byte that is not valid UTF-8 renders as that single byte. U+0080 therefore renders `\xc2\x80`, and the raw byte `0x80` renders `\x80`. The encoding is injective, so the cell reverses to the original path bytes. `asset_id` always hashes the true path bytes.
`fact_flags` uses a closed, lexically sorted vocabulary.

| Flag | Meaning |
| --- | --- |
| `dimensions_invalid` | The reported width or height is not greater than zero. |
| `fps_invalid` | The reported average FPS is non-finite or not greater than zero. |
| `frame_count_invalid` | The reported frame count is not greater than zero. |
| `rotation_unexpected` | The reported rotation is outside 0, 90, 180, and 270 degrees. |

An unopened or skipped probe has no flags.

## `assets.csv` schema

Rows sort by `source_path` in code-point order. Every non-canonical row leaves all parsed identity columns empty, including `repeat`.

| Column | Meaning |
| --- | --- |
| `asset_id` | Identifies the entry through its true relative path bytes. |
| `capture_id` | Stores the canonical task-side family key; empty otherwise. |
| `disposition` | Contains `canonical`, `quarantined`, or `excluded`. |
| `reason_code` | Explains the disposition through the closed vocabulary. |
| `source_path` | Stores the escaped corpus-relative POSIX path. |
| `subject_ordinal` | Stores the parsed integer for canonical assets. |
| `view` | Stores the canonical lexical view label. |
| `task` | Stores the repaired canonical task. |
| `side` | Stores the canonical side. |
| `repeat` | Stores 0 or the consumed positive repeat integer. |
| `normalizations` | Lists fired normalization steps in lexical order. |
| `size_bytes` | Stores the size, or stays empty when the size read fails. |
| `content_sha256` | Stores the optional full-file SHA-256. |
| `reported_width` | Stores the width that OpenCV reports. |
| `reported_height` | Stores the height that OpenCV reports. |
| `reported_avg_fps` | Stores OpenCV's average-rate claim, rounded to six decimals. |
| `reported_frame_count` | Stores OpenCV's frame-count claim. |
| `reported_rotation_deg` | Stores OpenCV's orientation metadata in degrees. |
| `reported_fourcc` | Stores the four FOURCC characters verbatim, or an empty value. |
| `nominal_duration_s` | Stores a header-derived nominal duration, rounded to four decimals. |
| `fact_flags` | Lists invalid or unexpected header facts. |
| `probe_status` | Contains `opened`, `open_failed`, or `skipped`. |
| `grammar_version` | Forms the stable family identity with `capture_id`. |
| `tool_version` | Identifies the writer as `v1`. |

A non-finite reported float produces an empty CSV cell. `nominal_duration_s` requires a finite positive rate and a positive reported frame count. It equals `reported_frame_count / reported_avg_fps` and is not a measured duration.

OpenCV requests orientation auto before header reads. Every opened asset must return the same readback, or publication fails. A bounded full-decode check covered ten clips and 7,061 decoded frames. The header counts matched that bounded sample, while presentation timestamps varied in every clip.

## `captures.csv` schema

Rows sort by `capture_id`, and only canonical assets contribute.

| Column | Meaning |
| --- | --- |
| `capture_id` | Identifies the task-side family. |
| `subject_ordinal` | Stores the family's normalized subject ordinal. |
| `task` | Stores the family's canonical task. |
| `side` | Stores the family's side. |
| `n_assets` | Counts canonical assets in the family. |
| `views` | Lists distinct lexical views in lexical order. |
| `n_views` | Counts distinct lexical views. |
| `view_conflict` | Equals 1 when several assets claim one view. |
| `reported_frame_count_min` | Stores the smallest reported frame count. |
| `reported_frame_count_max` | Stores the largest reported frame count. |
| `reported_fps_min` | Stores the smallest reported average FPS. |
| `reported_fps_max` | Stores the largest reported average FPS. |
| `reported_fps_spread_hz` | Stores the reported-rate spread. |
| `nominal_duration_min_s` | Stores the smallest available nominal duration. |
| `nominal_duration_max_s` | Stores the largest available nominal duration. |
| `nominal_duration_spread_s` | Stores the nominal-duration spread. |
| `reported_resolution_agree` | Equals 1 when every asset reports one width-height pair. |
| `reported_rotation_agree` | Equals 1 when every asset reports one rotation value. |
| `grammar_version` | Forms the stable family identity with `capture_id`. |
| `tool_version` | Identifies the writer as `v1`. |

Reported FPS values and spreads use six-decimal rounding. Nominal duration values and spreads use four-decimal rounding. Header agreement does not qualify the family as one recording event.

## `census.json` schema

This literal key tree lists every fixed key. Angle-bracket entries represent histogram keys.

```text
census.json
├── tool_version; grammar_version; opencv_version; backend_name
├── orientation_auto; checksums
├── generation
│   ├── assets.csv
│   ├── captures.csv
│   └── census.json
├── assets
│   ├── discovered; canonical; quarantined; excluded
│   ├── total_bytes; distinct_sha256; reported_frames_total
│   ├── nominal_minutes_total
│   └── nominal_duration_s.{count,min,p25,median,p75,p95,max}
├── reason_codes.<every_closed_reason>
├── normalization
│   ├── applied.<normalization_step>
│   └── task_repairs.<input_task>
├── extension_case.<literal_extension_or_none>
├── shapes.<width>x<height>@<fps>/<fourcc>/rot<rotation>
├── rotation_by_view.<view>.<rotation_deg>
├── directories_mixing_codecs; subject_directories
├── captures
│   ├── total; with_view_conflict; multi_view
│   ├── same_resolution; same_fps_3dp
│   ├── frame_parity_within_5pct; frame_parity_within_20pct
│   ├── view_coverage.<n_views>
│   └── duration_spread_s.{count,min,p25,median,p75,p95,max}
└── duration_spread_all_captures_s.{count,min,p25,median,p75,p95,max}
```

`tool_version`, `grammar_version`, and `opencv_version` record the writer, grammar, and OpenCV versions. `backend_name` joins distinct observed backends in lexical order with `|`. It is empty when no asset opens. `orientation_auto` stores the common opened-asset readback, or false when nothing opens. `checksums` records the checksum mode.
`assets.discovered` counts rows, and the three disposition fields partition it. `assets.total_bytes` sums available sizes. `assets.distinct_sha256` counts nonempty digests. `assets.reported_frames_total` sums nonnegative header counts. `assets.nominal_minutes_total` sums available nominal durations with four-decimal rounding. `assets.nominal_duration_s` summarizes valid nominal durations for opened assets.
`reason_codes` contains every closed reason, including zero counts and `ok`. The console prints only non-zero reasons. `normalization.applied` counts observed row-level labels. `normalization.task_repairs` counts each repaired input spelling.

`extension_case` counts each supported media suffix exactly as stored, which is how it surveys `.MOV` against `.mov`. Any other suffix counts as `<unsupported>`, and an absent suffix counts as `<none>`. A filename suffix is free text, and `census.json` must stay redaction-safe. `shapes` groups opened assets by dimensions, three-decimal FPS, FOURCC, and rotation. An empty FOURCC uses `?`. `rotation_by_view` counts reported rotations for canonical lexical views.

`directories_mixing_codecs` counts immediate parents with several opened FOURCC values. `subject_directories` counts immediate parents of admitted canonical and quarantined assets. Neither field stores a directory name.

`captures.total` counts task-side families. `captures.view_coverage` maps each distinct-view count to its family count. `captures.with_view_conflict` counts families that contain several physical takes. `captures.multi_view` counts families with several lexical views. `captures.same_resolution` and `captures.same_fps_3dp` evaluate multi-view families. Frame parity is `(maximum - minimum) / maximum`; a nonpositive maximum produces 1. The parity fields count multi-view families within their named thresholds.

`captures.duration_spread_s` summarizes multi-view family spreads. `duration_spread_all_captures_s` summarizes all available family spreads. Every quantile block contains `count`. A block with no values is exactly `{"count": 0}`. Values round to four decimals. Each percentile selects `sorted[min(len-1, int(fraction * len))]`.

## Publication and validation

Each artifact atomically replaces a process-specific sibling temporary file. The three-file publication is not set-atomic. The tool writes `census.json` last and records all three digests under `generation`.

The two table digests cover the exact published bytes, so a change of line endings alone fails validation. The `census.json` digest covers every remaining census field, because a document cannot carry a digest of itself. Census detection is therefore content-level: it catches an edited value, an added key, and a removed key, and it accepts insignificant JSON whitespace.

```python
from pose_estimation.inventory import validate_generation

census = validate_generation(out_dir)
```

Every consumer must call `validate_generation(out_dir)` before reading a row. The function returns the census or raises `InventoryError` for an inconsistent set. Do not reimplement its checksum rules in each consumer.

CSV uses UTF-8, commas, minimal quoting, and `\n` line endings. JSON uses sorted keys, two-space indentation, UTF-8, and one trailing newline. Unchanged inputs produce byte-identical artifacts.

Quote aggregate numbers only from a validated `census.json`. Keep the complete output directory on the controlled machine. See `entrypoints.md` for exit codes and console behavior.
