"""Publish the corpus-level calibration ruling as an evidence set.

Fifth artifact-publishing tool, after ``inventory``, ``sessions``, ``qualify``
and the ``measure`` sidecar, and it inherits their publication contract whole:
whole-tree swap, per-file digests, a self-describing marker, and a
``validate_generation`` every consumer calls before reading a row.

This tool computes no statistic.  Extrinsic recovery from RTMW-L keypoints was
measured unachievable on this corpus at 1080p under per-model intrinsic priors,
and the measurement lives in two committed probes; the ruling those probes
support is what this tool publishes.  Those three qualifiers travel with every
statement of the negative, here included: a lower-bias keypoint source, a
detector trained for multi-view consistency and any prospectively calibrated
capture all stay outside the measured bound.  Evidence therefore arrives as
captured probe stdout, is validated, and is republished as rows -- a tool that
recomputed a probe number would own that number's correctness, and the probes
already own it.

The ruling itself is a module constant rather than an argument.  The tree
publishes one ruling, so an operator cannot spell a different verdict through
the CLI, and a different verdict is a different generator version.

What the arguments decide is narrower than the ruling is wide, and the gap is
worth stating.  Publication checks the ``bias_transfer`` capture -- its arms,
its per-arm population and every statistic the rows carry -- and checks that
both cited scripts are present and carry the bytes their recorded digests name.
It does not check any ``calibration_bias`` output, which is cited and digested
but never ingested, so the claims that rest on that probe publish without their
numbers being seen here.  A digest match binds a capture to one version of a
script; it authenticates nothing, so hand-written stdout carrying the live
digest reads exactly like a real run.  What the check rules out is a capture
gone missing, a capture whose probe has since been edited, and a probe schema
that moved out from under the rows.

``qualification/`` is read-only here.  Its per-event geometry cells stay
``geom_unmeasured`` and its per-asset scale cells stay ``scale_unmeasured``;
this set sits beside that tree at a different grain and never patches it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import shutil
import stat
import sys
from typing import Any

from . import inventory, qualify, sessions

# v1 is the first publication of the ruling.  A published set is
# self-describing only if this moves with the schema: validate_generation
# refuses a document whose generator_version is not this one, and that refusal
# is the whole mechanism by which one generation cannot be read as another.
# Ownership includes the version, so the first run of a new version needs the
# previous tree removed by hand.
GENERATOR_VERSION = "v1"

CALIBRATION_QC_FILENAME = "calibration_qc.json"
CORPUS_QC_FILENAME = "corpus_qc.csv"
EVIDENCE_QC_FILENAME = "evidence_qc.csv"
CSV_FILENAMES: tuple[str, ...] = (CORPUS_QC_FILENAME, EVIDENCE_QC_FILENAME)
GENERATION_KEYS: tuple[str, ...] = (
    *CSV_FILENAMES,
    "tree",
    "qualification",
    "probes",
    "generator_version",
    "census",
)


class CalibrationQcError(Exception):
    """A ruling that cannot be published, or a published set that cannot be trusted."""

    def __init__(self, message: str, *, reason: str = "") -> None:
        super().__init__(message)
        self.reason = reason


# --- the ruling ----------------------------------------------------------------------------------

# Closed token sets, one per ruling cell.  Each alphabet is built from its set
# rather than from a shape that resembles it, so a verdict this tool never means
# to write is unspellable instead of merely unwritten.
RULING_GRAINS = frozenset({"corpus"})
RECOVERY_STATUSES = frozenset({"unachievable"})
RECOVERY_REASONS = frozenset({"cross_view_keypoint_bias"})
TRANSFER_STATUSES = frozenset({"absent"})
KEYPOINT_SOURCES = frozenset({"rtmw_l"})
INTRINSICS_BASES = frozenset({"per_model_prior"})
# `unrun` is the only spelling this cell accepts, and the exclusion is the
# point: the per-event joint bias-and-pose control was never built, so
# `failed`, `refused` and `unachievable` would each report a measurement that
# does not exist.
ARM_RUN_STATUSES = frozenset({"unrun"})

RULING: dict[str, str] = {
    "ruling_grain": "corpus",
    "recovery_status": "unachievable",
    "reason": "cross_view_keypoint_bias",
    "transfer_status": "absent",
    "keypoint_source": "rtmw_l",
    "image_height_px": "1080",
    "intrinsics_basis": "per_model_prior",
    "unrun_arm": "per_event_double_centered_bias_and_pose",
    "unrun_arm_status": "unrun",
    "cited_probes": "bias_transfer|calibration_bias",
}
CORPUS_COLUMNS: tuple[str, ...] = tuple(RULING)

EVIDENCE_COLUMNS: tuple[str, ...] = (
    "probe",
    "probe_sha256",
    "arm",
    "statistic",
    "n",
    "median",
    "min",
    "max",
    "above_0p5",
)

# --- the evidence --------------------------------------------------------------------------------

# Script names, never paths: the directory arrives as an argument, so a run
# cannot silently digest a script outside the tree the operator named.
PROBE_SCRIPTS: dict[str, str] = {
    "bias_transfer": "probe_bias_transfer.py",
    "calibration_bias": "probe_calibration_bias.py",
}

# Only `bias_transfer` is ingested as rows.  Its stdout is one uniform record
# per arm -- a label plus the same four statistic dicts -- which is the shape
# D06 presumes.  `calibration_bias` emits four differently-shaped record
# families, and flattening them needs a per-family adapter, which is this tool
# taking a position on what each family means.  That position is estimator
# knowledge, so the script is cited and digested while its numbers reach humans
# through the claim-bounded report instead.
INGESTED_PROBES: tuple[str, ...] = ("bias_transfer",)
ARM_KEY = "label"
# Every nested dict a `bias_transfer` arm carries.  A record missing one is a
# probe whose schema moved, which must refuse rather than publish a short row.
STATISTIC_KEYS: tuple[str, ...] = (
    "between_event_r",
    "between_event_r_abs",
    "within_event_r",
    "median_abs_px",
)
STATISTIC_FIELDS: tuple[str, ...] = ("n", "median", "min", "max", "above_0p5")
# `above_0p5` is absent on `median_abs_px` by design, so it stays nullable; the
# rest are numbers the ruling cites, and a dropped one has to refuse rather than
# publish a short row that reads as an unmeasured statistic.
REQUIRED_STATISTIC_FIELDS: tuple[str, ...] = ("n", "median", "min", "max")

# The record shape `bias_transfer` emits, closed.  Closing it is what makes the
# redaction boundary a refusal instead of a silent discard: an identifier-
# bearing key would otherwise be dropped on the way to a closed output schema,
# and the operator would keep a capture nothing ever told them was unsafe.  The
# token check `_assert_schema_is_redaction_safe` runs cannot be reused here,
# because the probe's own `between_event_r` carries a forbidden token itself.
RECORD_KEYS = frozenset(
    {ARM_KEY, "pairs", "events", "realizations", "shared_fraction", *STATISTIC_KEYS}
)

# The arms the ruling quotes by value.  Losing any one of them costs the ruling
# a load-bearing number: the four groupings the transfer claim names, and the
# permutation null they are read against.  The two reference bands follow as
# prefixes because their parameter sweeps are open.
REQUIRED_ARMS = frozenset(
    {
        "REAL same view pair",
        "REAL same view pair + same model pair",
        "REAL same view pair + same task",
        "REAL same view pair + same subject",
        "REAL same view pair, keypoints permuted (null)",
    }
)
# Every required arm is measured over the whole eligible population, and the
# transfer claim quotes that coverage.  A structurally complete capture whose
# arms ran on one pair would otherwise certify the negative, so the ruled
# population is pinned like the ruling is: a corpus that moves is a new
# generator version, not a new number in this cell.
RULED_POPULATION: dict[str, int] = {"pairs": 178, "events": 103}
REQUIRED_ARM_PREFIXES: tuple[str, ...] = (
    "SYNTH shared image bias ",
    "SYNTH per-event bias ",
    "SYNTH noise sigma=",
)

# --- the claim boundary --------------------------------------------------------------------------

# Every statement the published set must carry.  They are published, so a
# consumer reads the bound from the artifact rather than from a document that
# can drift away from it.
CLAIMS: tuple[str, ...] = (
    "Extrinsic recovery from RTMW-L keypoints on this corpus at 1080p under per-model "
    "intrinsic priors is measured unachievable.",
    "Within-event cross-view RTMW-L correspondence carries a measured 15-20 px systematic "
    "component at 1080p.",
    "The shipped estimator is exact on exact synthetic correspondence, and independent bundle "
    "adjustment worsens corpus closure.",
    "No disjointly selected RTMW-L subset beats all 65 keypoints on the measured corpus folds.",
    "Signed bias transfer is absent at the tested view-pair, device-model, task and subject "
    "groupings over the full eligible population.",
    "The same keypoints share difficulty across events while the signed offset direction is "
    "redrawn every event, so that magnitude is not a correctable coordinate offset.",
    "Held-out reprojection on the solve's own keypoint family is self-consistency.",
    "This evidence is internal geometric and QC evidence only.",
    "Every pixel and degree statistic here stays separate from absolute metric accuracy.",
    "No marker-based comparison was run.",
    "A lower-bias keypoint source and a detector trained for multi-view consistency stay "
    "outside the measured bound.",
    "Prospective calibrated capture stays outside the measured bound and is the route that can "
    "reopen 3D.",
    "The per-event double-centered bias-and-pose synthetic-control arm is unrun.",
    "One corpus-level ruling holds while every per-event geometry cell stays unmeasured.",
    "Each synthetic arm is instrument calibration whose meaning arises only in contrast with "
    "the corpus row.",
)

# An arm that never ran has no outcome, so every measured outcome spelled
# against it is prohibited text.  Free-form arm labels are what make this
# reachable: an added arm can spell a verdict beside a ruling cell whose
# alphabet admits `unrun` alone.
UNRUN_ARM_OUTCOMES: tuple[str, ...] = (
    "failed",
    "refused",
    "impossible",
    "unachievable",
    "succeeded",
    "ran",
)

# The overreach each claim refuses, checked as an absence over every published
# byte.  Kept here and never published: a set that carried this list would
# contain the text the scan exists to keep out of it.
PROHIBITED_PARAPHRASES: tuple[str, ...] = (
    "extrinsic calibration from human keypoints is impossible",
    "the detector is inaccurate by 15-20 px in absolute image coordinates",
    "no estimator could recover extrinsics",
    "no keypoint subset from any detector",
    "no bias model could ever work",
    "the detector has no repeatable keypoint-dependent behavior",
    "measures calibration accuracy",
    "clinical validity",
    "clinically invalid",
    "absolute metric accuracy or absolute metric error",
    "marker-based equivalence",
    "equivalent to marker-based",
    "no detector could do this",
    "no prospective capture could do this",
    "cannot recover known extrinsics",
    "independently failed calibration",
    "proves this corpus unrecoverable",
    # Derived from the ruling cell rather than spelled a second time.  `_fold`
    # flattens `_` and `-` alike, so one entry catches the published snake_case
    # cell and a hyphenated arm label emitted by a probe.
    *(f"{RULING['unrun_arm']} {outcome}" for outcome in UNRUN_ARM_OUTCOMES),
)

# --- alphabets -----------------------------------------------------------------------------------

INTEGER_CELL = re.compile(r"[0-9]+")
POSITIVE_INTEGER_CELL = re.compile(r"[1-9][0-9]*")
DECIMAL_CELL = re.compile(r"-?[0-9]+\.[0-9]+")
SHA256_CELL = re.compile(r"[0-9a-f]{64}")
# Probe stdout spells its own arm labels, so the alphabet is the observed
# spelling rather than a slug this tool invents: normalising a label would make
# the published cell a value no probe ever emitted.  Edge spaces are excluded
# because no label carries one, and an alphabet laxer than its producer forfeits
# exactly the detection it exists for.
_ARM_BODY = r"[A-Za-z0-9 ()+.,=|_-]"
ARM_CELL = re.compile(rf"[A-Za-z0-9]{_ARM_BODY}*[A-Za-z0-9)]|[A-Za-z0-9]")


def _token_alphabet(tokens: frozenset[str]) -> re.Pattern[str]:
    """Build a cell alphabet that is the token set, not a shape resembling it.

    A shape pattern such as ``[a-z_]+`` accepts every token the set excludes, so
    a verdict this tool never means to write would publish cleanly.
    """
    return re.compile("|".join(re.escape(token) for token in sorted(tokens)))


def _flag_alphabet(tokens: frozenset[str]) -> re.Pattern[str]:
    """A pipe-joined list over one closed token set, in any order."""
    one = "|".join(re.escape(token) for token in sorted(tokens))
    return re.compile(rf"(?:{one})(?:\|(?:{one}))*")


PROBE_NAMES = frozenset(PROBE_SCRIPTS)
CORPUS_CELL_ALPHABETS: dict[str, re.Pattern[str]] = {
    "ruling_grain": _token_alphabet(RULING_GRAINS),
    "recovery_status": _token_alphabet(RECOVERY_STATUSES),
    "reason": _token_alphabet(RECOVERY_REASONS),
    "transfer_status": _token_alphabet(TRANSFER_STATUSES),
    "keypoint_source": _token_alphabet(KEYPOINT_SOURCES),
    "image_height_px": POSITIVE_INTEGER_CELL,
    "intrinsics_basis": _token_alphabet(INTRINSICS_BASES),
    "unrun_arm": re.compile(r"[a-z][a-z0-9_]*"),
    "unrun_arm_status": _token_alphabet(ARM_RUN_STATUSES),
    "cited_probes": _flag_alphabet(PROBE_NAMES),
}
EVIDENCE_CELL_ALPHABETS: dict[str, re.Pattern[str]] = {
    "probe": _token_alphabet(PROBE_NAMES),
    "probe_sha256": SHA256_CELL,
    "arm": ARM_CELL,
    "statistic": _token_alphabet(frozenset(STATISTIC_KEYS)),
    "n": INTEGER_CELL,
    "median": DECIMAL_CELL,
    "min": DECIMAL_CELL,
    "max": DECIMAL_CELL,
    "above_0p5": INTEGER_CELL,
}

# --- redaction -----------------------------------------------------------------------------------

# A column name carrying one of these could only key a row to a recording, a
# person or a file.  The check runs over the schema at import, so the shape that
# would carry an identifier does not exist rather than merely going unwritten.
FORBIDDEN_KEY_TOKENS = frozenset(
    {
        "asset",
        "camera",
        "capture",
        "event",
        "family",
        "filename",
        "id",
        "path",
        "session",
        "stem",
        "subject",
        "video",
        "view",
    }
)
# The two identifier spellings this project publishes elsewhere: a capture
# pseudonym and a blake2b-64 asset id.  A cell matching either is an identifier
# that reached a redaction-safe table through its free-text column.
IDENTIFIER_SHAPES: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bs[0-9]{2}-[a-z]+-[lr]\b"),
    re.compile(r"\b[0-9a-f]{16}\b"),
)


def _assert_schema_is_redaction_safe() -> None:
    for table, columns in (
        (CORPUS_QC_FILENAME, CORPUS_COLUMNS),
        (EVIDENCE_QC_FILENAME, EVIDENCE_COLUMNS),
    ):
        for column in columns:
            parts = set(column.split("_"))
            forbidden = sorted(parts & FORBIDDEN_KEY_TOKENS)
            if forbidden:
                raise CalibrationQcError(
                    f"{table}: column {column!r} carries the identifier token "
                    f"{forbidden[0]!r}; this set is redaction-safe by contract.",
                    reason="forbidden_key",
                )


_assert_schema_is_redaction_safe()


def _assert_cells_carry_no_identifier(rows: list[dict[str, str]], filename: str) -> None:
    for row in rows:
        for column, cell in row.items():
            for shape in IDENTIFIER_SHAPES:
                if shape.search(cell):
                    raise CalibrationQcError(
                        f"{filename}: {column} cell carries an identifier-shaped value.",
                        reason="forbidden_value",
                    )


# --- evidence ingestion --------------------------------------------------------------------------


def _digest_bytes(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def probe_digests(probes_dir: str | os.PathLike[str]) -> dict[str, str]:
    """Digest every cited probe script, refusing a script that is not there.

    The ruling cites these scripts by name, so a citation whose script is
    missing or unreadable makes the ruling untraceable and must refuse.
    """
    root = pathlib.Path(probes_dir)
    digests: dict[str, str] = {}
    for probe in sorted(PROBE_SCRIPTS):
        path = root / PROBE_SCRIPTS[probe]
        try:
            digests[probe] = _digest_bytes(path)
        except OSError as error:
            raise CalibrationQcError(
                f"The cited probe {probe} is missing from the probe directory.",
                reason="probe_missing",
            ) from error
    return digests


def _read_capture(path: pathlib.Path) -> list[dict[str, Any]]:
    """Parse one captured probe stdout as line-delimited JSON.

    The probes stream one flushed object per arm and close with a
    pretty-printed key list, so the trailing document spans several lines and is
    not an arm.  Parsing line by line keeps the arms and refuses a capture cut
    mid-line, which is what a killed or redirected-and-truncated run leaves.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise CalibrationQcError(
            f"The evidence capture {path.name} is missing or cannot be read.",
            reason="evidence_missing",
        ) from error
    records: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line, object_pairs_hook=qualify._reject_duplicate_keys)
        except ValueError:
            # The closing summary is pretty-printed across several lines, and
            # its opening line is a bare "{".  Anything else that opens an
            # object and does not parse is a record cut mid-line, which is what
            # a killed or truncated redirect leaves -- and skipping it would
            # drop an arm the ruling may cite.
            if line.startswith("{") and line.strip() != "{":
                raise CalibrationQcError(
                    f"The evidence capture {path.name} carries a line that is not one "
                    "unambiguous JSON document.",
                    reason="evidence_malformed",
                ) from None
            continue
        if isinstance(record, dict) and ARM_KEY in record:
            records.append(record)
    if not records:
        raise CalibrationQcError(
            f"The evidence capture {path.name} carries no arm record.",
            reason="evidence_empty",
        )
    return records


def _cell(value: Any) -> str:
    """Render one statistic field, keeping an absent field an empty cell.

    A missing field publishes ``""`` for the same reason an unmeasured axis does
    in ``qualify``: an empty cell cannot be mistaken for a measurement, and no
    alphabet has to admit a sentinel that could be.
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        raise CalibrationQcError(
            "A probe statistic is a boolean, which no statistic field spells.",
            reason="evidence_malformed",
        )
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    raise CalibrationQcError("A probe statistic is not a number.", reason="evidence_malformed")


def evidence_rows(probe: str, digest: str, records: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Flatten one probe's arm records into one row per (arm, statistic).

    Long form rather than one wide row per arm: the arms do not share a
    statistic set, so a wide table would spend most of its cells on emptiness
    and would need a schema change every time a probe gains a statistic.
    """
    rows: list[dict[str, str]] = []
    for record in records:
        unexpected = sorted(set(record) - RECORD_KEYS)
        if unexpected:
            raise CalibrationQcError(
                f"{probe}: an arm record carries the key {unexpected[0]!r}, which this "
                "redaction-safe set does not accept; a key it cannot publish is refused "
                "rather than dropped.",
                reason="forbidden_key",
            )
        arm = record[ARM_KEY]
        if not isinstance(arm, str) or not arm:
            raise CalibrationQcError(
                f"{probe}: an arm record carries no label.", reason="evidence_malformed"
            )
        for statistic in STATISTIC_KEYS:
            block = record.get(statistic)
            if not isinstance(block, dict):
                raise CalibrationQcError(
                    f"{probe}: arm {arm!r} omits the statistic {statistic!r}; the probe's "
                    "schema moved and the ruling can no longer cite it.",
                    reason="evidence_schema",
                )
            absent = [field for field in REQUIRED_STATISTIC_FIELDS if block.get(field) is None]
            if absent:
                raise CalibrationQcError(
                    f"{probe}: arm {arm!r} statistic {statistic!r} omits {absent[0]!r}; the "
                    "probe's schema moved and the ruling can no longer cite it.",
                    reason="evidence_schema",
                )
            rows.append(
                {
                    "probe": probe,
                    "probe_sha256": digest,
                    "arm": arm,
                    "statistic": statistic,
                    **{field: _cell(block.get(field)) for field in STATISTIC_FIELDS},
                }
            )
    return rows


def _assert_cited_arms(arms: frozenset[str]) -> None:
    """Refuse a capture that has lost an arm the ruling quotes by value."""
    missing = sorted(REQUIRED_ARMS - arms)
    if missing:
        raise CalibrationQcError(
            f"The evidence no longer carries the cited arm {missing[0]!r}, so the ruling "
            "cannot be published from it.",
            reason="arm_missing",
        )
    for prefix in REQUIRED_ARM_PREFIXES:
        if not any(arm.startswith(prefix) for arm in arms):
            raise CalibrationQcError(
                f"The evidence no longer carries a reference arm spelled {prefix!r}, so the "
                "corpus reading has nothing to be contrasted with.",
                reason="arm_missing",
            )


def _assert_arm_population(probe: str, records: list[dict[str, Any]]) -> None:
    """Refuse a duplicate arm, and a required arm short of the ruled population.

    The arm set stays open -- the evidence table is a transcript of whatever the
    probe emitted, and closing it would make every probe revision a publisher
    change.  Two constraints hold anyway: one label is one row key, and the arms
    the transfer claim quotes have to carry the population it quotes.
    """
    seen: set[str] = set()
    for record in records:
        arm = record[ARM_KEY]
        if arm in seen:
            raise CalibrationQcError(
                f"{probe}: the arm {arm!r} appears twice, so one label would key two rows.",
                reason="arm_duplicate",
            )
        seen.add(arm)
        if arm not in REQUIRED_ARMS:
            continue
        for field, ruled in RULED_POPULATION.items():
            if record.get(field) != ruled:
                raise CalibrationQcError(
                    f"{probe}: arm {arm!r} reports {field}={record.get(field)!r} rather than "
                    f"the ruled {ruled}; the claim quotes the full eligible population.",
                    reason="population_mismatch",
                )


# --- publication ---------------------------------------------------------------------------------


def _canonical(rows: list[dict[str, str]], key: tuple[str, ...]) -> list[dict[str, str]]:
    """Fix published row order as a function of the rows, never of a loader."""
    return sorted(rows, key=lambda row: tuple(row[name] for name in key))


def _assert_cell_alphabets(
    rows: list[dict[str, str]], alphabets: dict[str, re.Pattern[str]], filename: str
) -> None:
    """Refuse to publish a cell this tool cannot spell.

    ``fullmatch``, never ``match``: ``^...$`` would accept a trailing newline,
    which is exactly how a smuggled cell survives a pattern that looks strict.
    """
    for row in rows:
        for column, pattern in alphabets.items():
            cell = row[column]
            if cell and not pattern.fullmatch(cell):
                raise CalibrationQcError(
                    f"{filename}: {column} cell {cell!r} does not match {pattern.pattern}",
                    reason="cell_alphabet",
                )


def build_census(
    corpus_rows: list[dict[str, str]], evidence_rows_: list[dict[str, str]]
) -> dict[str, Any]:
    """Return the redaction-safe census: counts, the ruling, and the claim bound."""
    arms = sorted({row["arm"] for row in evidence_rows_})
    return {
        "claims": list(CLAIMS),
        "corpus": {"rows": len(corpus_rows), "ruling": dict(RULING)},
        "evidence": {
            "rows": len(evidence_rows_),
            "arms": len(arms),
            "probes": sorted({row["probe"] for row in evidence_rows_}),
            "statistics": sorted({row["statistic"] for row in evidence_rows_}),
        },
        "schema_version": GENERATOR_VERSION,
    }


def tree_digest(out_dir: str | os.PathLike[str]) -> str:
    """Digest every entry of the set except the marker that carries this value."""
    lines: list[str] = []

    def visit(entry: pathlib.Path, label: str) -> None:
        if entry.is_symlink():
            # os.readlink, not Path.readlink: the latter drops a leading "./",
            # so the digested text would not be the link's own text.
            lines.append(f"{label}\tlink\t{os.readlink(entry)}\n")  # noqa: PTH115
        elif entry.is_dir():
            lines.append(f"{label}\tdir\n")
            for child in sorted(entry.iterdir()):
                visit(child, f"{label}/{child.name}")
        else:
            lines.append(f"{label}\tfile\t{hashlib.sha256(entry.read_bytes()).hexdigest()}\n")

    for entry in sorted(pathlib.Path(out_dir).iterdir()):
        if entry.name != CALIBRATION_QC_FILENAME:
            visit(entry, entry.name)
    return hashlib.sha256("".join(lines).encode("utf-8", "surrogateescape")).hexdigest()


def census_digest(census: dict[str, Any]) -> str:
    """Digest the census exactly as published, minus its own marker.

    The digest lives inside the document it certifies, so it is taken over the
    document without that one self-referential key -- and over everything else,
    the generation block included.  Excluding the whole block instead would
    leave the upstream provenance consumers trust most as the only claim in the
    set nothing covers.

    Detection, not authentication: the set carries no key, so an edit that also
    recomputes this digest is indistinguishable from a publication.  What it
    rules out is corruption and every edit that stops at the claim.
    """
    body = dict(census)
    if isinstance(body.get("generation"), dict):
        body["generation"] = {
            key: value for key, value in body["generation"].items() if key != "census"
        }
    return hashlib.sha256(inventory.render_json(body).encode("utf-8")).hexdigest()


def _is_own_generation(generation: Any) -> bool:
    """Whether this block is the key set this generator publishes.

    Shape and version only, never a digest: a set whose upstream has since moved
    is stale but still this tool's to replace, and requiring freshness here
    would strand it behind a manual delete.
    """
    if not isinstance(generation, dict) or generation.get("generator_version") != GENERATOR_VERSION:
        return False
    return set(generation) == set(GENERATION_KEYS)


def _read_marker(out: pathlib.Path) -> dict[str, Any]:
    """Read the marker as the kind of file this tool would itself have written.

    The marker is the set's trust root and the one entry ``tree_digest`` cannot
    cover, so its own identity is all that stands behind it.  A symlink puts
    that root outside the set it certifies, and through ``_assert_owned`` lets a
    foreign directory license its own deletion.  A duplicate key puts two claims
    in one document, of which ``json.loads`` silently keeps the last.
    """
    path = out / CALIBRATION_QC_FILENAME
    if not stat.S_ISREG(path.lstat().st_mode):
        raise OSError(f"{CALIBRATION_QC_FILENAME} is not a regular file.")
    return json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=qualify._reject_duplicate_keys
    )


def _fold(text: str) -> str:
    """Case-fold, and flatten ``_`` and ``-`` to spaces.

    A snake_case cell and a hyphenated arm label are the two places an overreach
    enters the artifact reading as a token rather than as prose, so both sides
    of the scan fold and one prohibited entry covers both spellings.
    """
    return text.casefold().replace("_", " ").replace("-", " ")


def _assert_claim_conformance(staging: pathlib.Path) -> None:
    """Refuse to publish a set that overclaims, or that has dropped a claim.

    Run over the staged bytes rather than over the in-memory rows, because the
    published text is what a consumer quotes.  Presence catches a claim silently
    lost; absence catches an overreach reaching the artifact through any cell.
    """
    published = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(staging.rglob("*"))
        if path.is_file() and not path.is_symlink()
    )
    folded = _fold(published)
    for claim in CLAIMS:
        if claim not in published:
            raise CalibrationQcError(
                f"The published set does not carry the required claim: {claim[:60]}...",
                reason="claim_missing",
            )
    for paraphrase in PROHIBITED_PARAPHRASES:
        if _fold(paraphrase) in folded:
            raise CalibrationQcError(
                f"The published set carries the prohibited claim {paraphrase!r}, which the "
                "measurement does not support.",
                reason="claim_prohibited",
            )


def _build(
    staging: pathlib.Path,
    corpus_rows: list[dict[str, str]],
    evidence: list[dict[str, str]],
    *,
    upstream_qualification: dict[str, Any],
    probes: dict[str, str],
) -> None:
    # One corpus row is the headline shape of the set, so it is checked where
    # the bytes are written rather than only where the row is built.
    if len(corpus_rows) != 1:
        raise CalibrationQcError(
            f"The corpus table would carry {len(corpus_rows)} rows; the ruling is one row.",
            reason="corpus_cardinality",
        )
    staging.mkdir(parents=True)
    rows_by_table = {CORPUS_QC_FILENAME: corpus_rows, EVIDENCE_QC_FILENAME: evidence}
    for name, columns in (
        (CORPUS_QC_FILENAME, CORPUS_COLUMNS),
        (EVIDENCE_QC_FILENAME, EVIDENCE_COLUMNS),
    ):
        (staging / name).write_text(
            inventory.render_csv(columns, rows_by_table[name]), encoding="utf-8", newline=""
        )
    census = build_census(corpus_rows, evidence)
    census["generation"] = {
        # Keyed off the published tuple, so a table added to the set without a
        # digest is impossible rather than merely caught.
        **{name: _digest_bytes(staging / name) for name in CSV_FILENAMES},
        # Catches what the per-file digests cannot: a file added to the set.
        "tree": tree_digest(staging),
        "qualification": dict(upstream_qualification),
        "probes": dict(probes),
        "generator_version": GENERATOR_VERSION,
    }
    # Last, because it digests every other key including this block's own.
    census["generation"]["census"] = census_digest(census)
    (staging / CALIBRATION_QC_FILENAME).write_text(
        inventory.render_json(census), encoding="utf-8", newline=""
    )
    _assert_claim_conformance(staging)


def _remove(path: pathlib.Path) -> None:
    # `rmtree(ignore_errors=True)` swallows NotADirectoryError and leaves a
    # regular file in place, which then blocks the staging mkdir and the swap.
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    else:
        shutil.rmtree(path, ignore_errors=True)


def _is_within(child: str, parent: str) -> bool:
    """True when *child* is *parent* or sits under it.

    Compared on separator-terminated text rather than on prefixes, so a sibling
    named ``calibration_qc-old`` is not read as living inside ``calibration_qc``.
    """
    child = child.rstrip(os.sep)
    parent = parent.rstrip(os.sep)
    return child == parent or child.startswith(parent + os.sep)


def _assert_disjoint(out: pathlib.Path, other: str | os.PathLike[str], label: str) -> None:
    """Refuse an output that overlaps an input, in either direction.

    Publication replaces the whole output tree, so an output containing the
    evidence deletes the captures, and one inside the qualification tree deletes
    the upstream it just validated.
    """
    here = os.path.realpath(out)
    there = os.path.realpath(other)
    if _is_within(here, there) or _is_within(there, here):
        raise CalibrationQcError(
            f"The output directory must sit outside the {label}.", reason="output_overlap"
        )


def _sweep_orphans(out: pathlib.Path) -> None:
    """Remove staging and retiring siblings that no live process owns."""
    for sibling in out.parent.glob(f"{out.name}.*"):
        stage, _, pid = sibling.name[len(out.name) + 1 :].rpartition(".")
        if stage not in ("staging", "retiring") or pid == str(os.getpid()):
            continue
        try:
            os.kill(int(pid), 0)
        except (ValueError, OverflowError, ProcessLookupError):
            _remove(sibling)
        except PermissionError:
            continue


def _assert_owned(out_dir: pathlib.Path) -> None:
    """Refuse a non-empty destination this tool did not publish."""
    if not out_dir.exists():
        return
    if not out_dir.is_dir():
        raise CalibrationQcError(
            "The output path exists and is not a directory.", reason="output_not_directory"
        )
    if not any(out_dir.iterdir()):
        return
    refusal = CalibrationQcError(
        "The output directory is not empty and carries no generation marker this tool wrote. "
        "Publishing would delete a directory this tool does not own.",
        reason="not_owned",
    )
    try:
        marker = _read_marker(out_dir)
    except (OSError, ValueError) as error:
        raise refusal from error
    # This generator's shape and version, not merely some tool's marker: the
    # next statement after this one deletes the whole tree.
    if not isinstance(marker, dict) or not _is_own_generation(marker.get("generation")):
        raise refusal


def run(
    qualification_dir: str | os.PathLike[str],
    evidence_dir: str | os.PathLike[str],
    probes_dir: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    *,
    sessions_dir: str | os.PathLike[str] | None = None,
    inventory_dir: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Publish the calibration ruling, replacing any generation this tool owns."""
    qualification_path = pathlib.Path(qualification_dir)
    evidence_path = pathlib.Path(evidence_dir)
    out = pathlib.Path(os.path.realpath(out_dir))
    # Binding the qualification tree binds the whole chain: the observability
    # probe that produced these caches validates inventory, sessions and
    # qualification together and stores all three in its cache fingerprint.
    qualification_census = qualify.validate_generation(
        qualification_path, sessions_dir=sessions_dir, inventory_dir=inventory_dir
    )
    for other, label in (
        (qualification_path, "qualification tree"),
        (evidence_path, "evidence directory"),
        (probes_dir, "probe directory"),
    ):
        _assert_disjoint(out, other, label)
    # Every input is validated before the output is touched at all.  A refusal
    # that arrives after ownership has been judged has already renamed a
    # retiring sibling and swept orphans on behalf of a run that was never
    # going to publish.
    digests = probe_digests(probes_dir)
    evidence: list[dict[str, str]] = []
    for probe in INGESTED_PROBES:
        capture = evidence_path / f"{probe}.jsonl"
        sidecar = evidence_path / f"{probe}.sha256"
        # The directory check above resolves the directory, not its entries: a
        # capture that is a symlink into the output is an input the swap
        # deletes, and its link text lives outside the tree that was compared.
        for entry in (capture, sidecar):
            _assert_disjoint(out, entry, f"evidence capture {entry.name}")
        records = _read_capture(capture)
        recorded = _read_recorded_digest(sidecar, probe)
        if recorded != digests[probe]:
            raise CalibrationQcError(
                f"The capture for {probe} was produced by a different script than the one "
                "committed, so the ruling it backs is not re-derivable.",
                reason="probe_digest",
            )
        _assert_arm_population(probe, records)
        evidence.extend(evidence_rows(probe, digests[probe], records))
    _assert_cited_arms(frozenset(row["arm"] for row in evidence))

    corpus_rows = [dict(RULING)]
    evidence = _canonical(evidence, ("probe", "arm", "statistic"))
    _assert_cell_alphabets(corpus_rows, CORPUS_CELL_ALPHABETS, CORPUS_QC_FILENAME)
    _assert_cell_alphabets(evidence, EVIDENCE_CELL_ALPHABETS, EVIDENCE_QC_FILENAME)
    _assert_cells_carry_no_identifier(corpus_rows, CORPUS_QC_FILENAME)
    _assert_cells_carry_no_identifier(evidence, EVIDENCE_QC_FILENAME)

    # A same-pid retirement is not always debris.  Pids are reused, so this can
    # be the sole complete generation a kill left between the two renames --
    # the exact state the sweep below is ordered to protect.  Restore it before
    # ownership is judged, so the tree that survived the crash is the tree this
    # run replaces rather than the tree this run deletes.
    retiring = out.with_name(f"{out.name}.retiring.{os.getpid()}")
    if retiring.is_dir() and not out.exists():
        retiring.rename(out)
    _assert_owned(out)

    staging = out.with_name(f"{out.name}.staging.{os.getpid()}")
    _remove(staging)
    _remove(retiring)
    try:
        _build(
            staging,
            corpus_rows,
            evidence,
            upstream_qualification=qualification_census,
            probes=digests,
        )
        if out.exists():
            out.rename(retiring)
        try:
            staging.rename(out)
        except OSError:
            if retiring.exists() and not out.exists():
                retiring.rename(out)
            raise
        # Swept only once the swap has landed: after a kill between the two
        # renames the sole complete generation sits under a dead pid.
        _sweep_orphans(out)
        _remove(retiring)
    finally:
        _remove(staging)
    return _read_marker(out)


def _read_recorded_digest(path: pathlib.Path, probe: str) -> str:
    """Read the script digest the capture was taken under.

    Written by the operator beside the capture, so the pair binds stdout to one
    version of the script.  Without it an edited probe would keep certifying a
    capture it can no longer produce.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise CalibrationQcError(
            f"The capture for {probe} carries no recorded script digest.",
            reason="digest_missing",
        ) from error
    # `sha256sum` writes "<digest>  <path>"; the digest is the first field.
    recorded = text.strip().split()[0] if text.strip() else ""
    if not SHA256_CELL.fullmatch(recorded):
        raise CalibrationQcError(
            f"The recorded script digest for {probe} is not a SHA-256 digest.",
            reason="digest_malformed",
        )
    return recorded


def validate_generation(
    out_dir: str | os.PathLike[str],
    qualification_dir: str | os.PathLike[str] | None = None,
    sessions_dir: str | os.PathLike[str] | None = None,
    inventory_dir: str | os.PathLike[str] | None = None,
    probes_dir: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Return the generation block of *out_dir*, or raise when it is stale.

    Every consumer calls this before reading a row.  With *qualification_dir* it
    also proves the set was published from the upstream generation still on
    disk; with *probes_dir* it proves the cited scripts still carry the bytes
    the ruling was published against, which is the check that catches a probe
    edited after the claim it supports.
    """
    out = pathlib.Path(out_dir)
    try:
        census = _read_marker(out)
    except (OSError, ValueError) as error:
        raise CalibrationQcError(
            "The published set is unusable: calibration_qc.json is missing, is not a regular "
            "file, or is not one unambiguous JSON document.",
            reason="marker_unreadable",
        ) from error
    if not isinstance(census, dict) or not isinstance(census.get("generation"), dict):
        raise CalibrationQcError(
            "The published set is unusable: calibration_qc.json has no generation.",
            reason="marker_shape",
        )
    generation = census["generation"]
    if set(generation) != set(GENERATION_KEYS) or generation["generator_version"] != (
        GENERATOR_VERSION
    ):
        raise CalibrationQcError(
            "The published set is unusable: calibration_qc.json is not this generator's document.",
            reason="generation_foreign",
        )
    for name in CSV_FILENAMES:
        try:
            digest = _digest_bytes(out / name)
        except OSError as error:
            raise CalibrationQcError(
                f"The published set is unusable: {name} is missing or cannot be read.",
                reason="table_missing",
            ) from error
        if digest != generation.get(name):
            raise CalibrationQcError(
                f"The published set is inconsistent: {name} is a different generation.",
                reason="table_digest",
            )
    if census_digest(census) != generation.get("census"):
        raise CalibrationQcError(
            "The published set is inconsistent: calibration_qc.json was edited after publication.",
            reason="census_digest",
        )
    try:
        current = tree_digest(out)
    except OSError as error:
        raise CalibrationQcError(
            "The published set is unusable: it cannot be walked.", reason="tree_unreadable"
        ) from error
    if current != generation.get("tree"):
        raise CalibrationQcError(
            "The published set is inconsistent: a file was added, removed or changed "
            "after publication.",
            reason="tree_digest",
        )
    if qualification_dir is not None:
        upstream = qualify.validate_generation(
            pathlib.Path(qualification_dir),
            sessions_dir=sessions_dir,
            inventory_dir=inventory_dir,
        )
        if upstream != generation.get("qualification"):
            raise CalibrationQcError(
                "The published set is stale: the qualification tree is a different generation.",
                reason="qualification_stale",
            )
    if probes_dir is not None and probe_digests(probes_dir) != generation.get("probes"):
        raise CalibrationQcError(
            "The published set is stale: a cited probe script has changed since publication.",
            reason="probe_stale",
        )
    return generation


def render_summary(census: dict[str, Any]) -> str:
    """Return the console summary.

    Counts and the ruling only.  The set is redaction-safe by contract, and the
    console stays inside that contract rather than beside it.
    """
    ruling = census["corpus"]["ruling"]
    evidence = census["evidence"]
    return "\n".join(
        [
            f"Ruling grain: {ruling['ruling_grain']}",
            f"  recovery: {ruling['recovery_status']} ({ruling['reason']})",
            f"  transfer: {ruling['transfer_status']}",
            f"  bound to: {ruling['keypoint_source']} at {ruling['image_height_px']}p, "
            f"{ruling['intrinsics_basis']}",
            f"  unrun arm: {ruling['unrun_arm']} ({ruling['unrun_arm_status']})",
            f"Evidence rows: {evidence['rows']} over {evidence['arms']} arms",
            f"  probes: {', '.join(evidence['probes'])}",
            f"Claims published: {len(census['claims'])}",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pose-estimation-calibration-qc",
        description="Publish the corpus-level calibration ruling and the evidence behind it.",
    )
    parser.add_argument(
        "--qualification", required=True, help="Directory that holds qualification.json."
    )
    parser.add_argument(
        "--evidence",
        required=True,
        help="Directory that holds one <probe>.jsonl capture and one <probe>.sha256 beside it.",
    )
    parser.add_argument(
        "--probes", required=True, help="Directory that holds the cited probe scripts."
    )
    parser.add_argument("--out", required=True, help="Directory to publish the ruling into.")
    parser.add_argument("--sessions", help="Session tree to check the upstream against.")
    parser.add_argument("--inventory", help="Registry directory to check the upstream against.")
    arguments = parser.parse_args(argv)
    try:
        census = run(
            arguments.qualification,
            arguments.evidence,
            arguments.probes,
            arguments.out,
            sessions_dir=arguments.sessions,
            inventory_dir=arguments.inventory,
        )
    # The upstream chain is validated inside `run`, so its refusals reach the
    # operator through this command and must arrive as a message and exit 2
    # rather than as a traceback.
    except (
        CalibrationQcError,
        qualify.QualifyError,
        sessions.SessionsError,
        inventory.InventoryError,
    ) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    print(render_summary(census))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
