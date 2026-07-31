# Project memory

Context retained only when source, tests, technical docs, roadmap, and git do not expose it cheaply.

- An unrelated-history collaborator fork previously contained a patient-video blob. Reconcile any future fork by content onto `main` with source-only attribution; importing its Git object history would retain patient data even after a file deletion.
- The container + host share the checkout through different absolute paths, so uv environments are layer-specific. Container work uses `.venv`; host work uses `.venv-host`. Recreate the matching environment after a move; repair text shebang/activation/editable-path metadata only when offline, and regenerate binary/cache artifacts.
- GPU/NPU access depends on the inherited Intel driver/runtime environment (`LD_LIBRARY_PATH`, ICD, Level Zero, and any accelerator `PYTHONPATH`). Confirm with `openvino.Core().available_devices` in the correct uv environment; the pip dependency remains required for generic CPU-only checkouts.
