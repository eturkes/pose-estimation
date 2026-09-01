# `calibration_qc` regression fixtures

This directory holds de-identified test data for the calibration ruling publisher
(`src/pose_estimation/calibration_qc.py`). No byte here comes from patient data.

## What is here

| path | content |
| ---- | ------- |
| `inputs/upstream/` | A synthetic published qualification generation. |
| `inputs/probes/` | Two synthetic probe scripts. The publisher digests these files. |
| `inputs/evidence/` | One synthetic probe capture and the digest it was taken under. |
| `expected/published/` | The generation the publisher must emit from `inputs/`. |
| `negatives/<reason>/` | One corrupted input set for each refusal the publisher can raise. |
| `manifest.json` | The synthetic namespace, the input digests and the refusal matrix. |

Directory names avoid the four tree names that `.gitignore` matches at any depth. A
fixture below one of those names is not committed. See the contract for the list.

## How to regenerate

```sh
env -u LD_LIBRARY_PATH PYTHONPATH="$PWD/src" uv run --no-sync \
  python scripts/make_calibration_qc_fixtures.py --force
```

The generator is idempotent. Two runs from a clean base give identical bytes.

## How to check

```sh
env -u LD_LIBRARY_PATH PYTHONPATH="$PWD/src" uv run --no-sync \
  python scripts/check_calibration_qc_fixtures.py
```

The command exits 0 when all 12 predicates pass. It exits 1 and prints each failure.
The test `tests/test_calibration_qc_fixtures.py` calls the same code.

## What the privacy scan forbids

The scan reads every file in this directory, this file included. It fails on any of
these:

- A capture identifier outside the synthetic subject namespace.
- Path text that names one of the patient-adjacent trees.
- An absolute path to this repository.
- A real corpus statistic. Only numbers that are already constants in
  `calibration_qc.py` are permitted.

The scan holds the needles it searches for. This file must not spell them, because a
document inside the scanned set carries every string it quotes. `check_calibration_qc_fixtures.py`
keeps that list, and the contract explains the rule.
