# Review UI

A local web UI over the 3-camera corpus. It has three views: the corpus census, a
clip player with a pose overlay, and the cohort statistics explorer. It reads the
published trees and writes nothing.

The UI is bilingual. Japanese is the default. The button at the top right changes
the language, and `?lang=en` sets it in a link.

The UI has three themes. The default is auto, which follows the operating system.
The button beside the language button steps through auto, light and dark. The
browser keeps your choice. `?theme=light` or `?theme=dark` sets the theme in a
link.

## Run

```sh
uv run --directory prototype/review-ui python -m review_ui
```

Open `http://127.0.0.1:8791/`. The server prints each published tree as `ok` or
`absent` before it starts.

Options:

- `--port N` selects a different port.
- `--repo PATH` points at a different repository root. The environment variable
  `REVIEW_UI_REPO` does the same.
- `--host H` binds a different address. The default binds the loopback address
  only.

## The three views

**Corpus census** shows what the recordings are. It reads `inventory/census.json`,
`qualification/qualification.json`, `calibration_qc/calibration_qc.json`,
`cohort/cohort.json` and `output/corpus-2d/run_report.json`. A row of tiles gives
the headline counts, and each tile names the tree it came from. One group of
panels covers the recordings: the capture formats, the rotation per view, the
codecs and the devices. A second group covers the clips: the views per capture,
the cameras per event, the clip durations and the synchronisation status. A third
group covers the rulings: the QC flags, the 3D recovery ruling, the run verdicts,
the registry reason codes and the generator versions.

**Clip player** draws the exported landmarks on a canvas over the video. The
skeleton and the palette come from `pose_estimation.drawing`, so the overlay draws
what the pipeline draws. Each list row names one camera artifact. The `#nnn`
prefix is a recording event. Rows with the same number are the other views of that
event. The stage fills the space that the window leaves. It keeps the shape of
the clip, so the whole frame stays on screen with the controls under it. The UI
fits the stage again after a window resize. The controls set the video layer to
show, dim or hide, switch the body, hands, points and labels, and move the
visibility threshold. The strip below the transport shows the mean body
visibility per frame, so a tracking dropout is visible for the whole clip at
once. The overlay palette does not change with the
theme, because the overlay draws over video and not over a page surface. The stage
turns dark when you dim or hide the video, which keeps the overlay readable. The
dot colours show the confidence band. The pipeline colours a dot by body group, so
those three colours belong to this UI alone.

**Cohort statistics** shows one feature at a time across the 12 `(task, side)`
cells. The bar is the interquartile range, the rule is the median and the dot is
the mean. The export publishes no extremes, so the chart draws no whisker. The
boundary panel states what the numbers mean and what they do not mean.

## Data

`.gitignore` excludes every published tree. A clone therefore carries none of
them. Each view degrades on its own: an absent tree becomes a stated gap and the
rest of the UI still runs.

This repository commits no video at all. The player therefore needs the published
trees: a clone without them lists no clips and says so. There is no demo clip.

The clip list carries no `event_id`, no filename and no subject identifier. The
`#nnn` ordinal is positional.

## Regenerate

Run each command from the directory it names.

```sh
# Skeleton topology and palette, from pose_estimation.drawing. Repository root.
env -u LD_LIBRARY_PATH PYTHONPATH="$PWD/src" uv run --no-sync \
    python prototype/review-ui/tools/export_topology.py

# IBM Plex subsets and the Plotly cartesian bundle. Needs network.
uv run --directory prototype/review-ui python tools/build_assets.py

# proof/ — four view captures and the API transcript. Needs webcap.
uv run --directory prototype/review-ui python tools/capture_proof.py
```

`build_assets.py` subsets the Japanese faces to the characters this UI renders.
The charset is the UI strings plus every `ja` label in `cohort/descriptors.yaml`.
Rerun it after you add a Japanese string, or the new characters render as tofu.

## Proof

`proof/` holds `census-ja.png`, `cohort-ja.png`, `census-en.png`,
`census-ja-dark.png` and `run.txt`. Each capture pins its theme, because the
default follows the machine that takes the capture. The three light captures are
the baseline. The dark capture shows the theme control. The transcript records the
run command, the published trees, and one probe per endpoint the views consume. It carries no timestamp, host path or
process id, so a rerun over the same trees rewrites it byte for byte.

The player view takes no capture. Its stage plays patient video, and the clip it
selects is a real recording, so a committed PNG of that view would carry a frame
of one subject. The clip and landmark endpoints appear in the transcript instead,
under a placeholder path. Those rows report the keypoint counts, which are model
schema. They also report the response status and the byte count that the range
request asks for. They report no frame count, no person count, no scale and no
file size, because each of those measures the one clip that answered.

The captures are stable in layout. They are not stable byte for byte. Every chart
waits for the Plex faces before it measures its text, which pins the legend rows:
six captures of the census view agreed exactly. A rerun can still differ by a few
pixels, because the software rasterizer rounds some edges differently. Two full
runs differed on 31 pixels of one capture, each pixel by one intensity level.
Compare a capture visually. Do not compare digests.

## Limits

- The browser must decode the clip. Chromium plays the H.264 clips. It does not
  play the HEVC clips on a build without a platform decoder. The player names the
  failure in a banner and hides the video layer. The overlay then plays on its own
  clock.
- The theme needs the CSS `light-dark()` function and the `:has()` selector.
  Chrome 123, Firefox 121 and Safari 17.5 support both.
- The prototype carries no tests and no gate. It is a behavioural reference for
  the IMPLEMENT phase, not its code base.
- The server trusts its caller. It binds the loopback address and serves
  read-only. Every request resolves through the built clip index, so a request
  supplies a key and never a path.
