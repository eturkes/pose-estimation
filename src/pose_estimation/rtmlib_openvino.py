"""Monkey-patch for rtmlib's OpenVINO backend to support NPU / GPU devices."""

import numpy as np

# ---------------------------------------------------------------------------
# Monkey-patch rtmlib's OpenVINO backend to support NPU / GPU devices
# ---------------------------------------------------------------------------
# rtmlib hardcodes device_name='CPU' in its OpenVINO backend.  The patch
# below overrides that so we can pass --device NPU (or GPU) and have it
# forwarded to OpenVINO's compile_model().  For NPU, models are also
# reshaped to static shapes (batch=1) before compilation.
_ORIG_BASE_INIT = None  # set lazily after import


def _patch_rtmlib_openvino():
    """Allow rtmlib's OpenVINO backend to use non-CPU devices.

    Also generalises the output-layer handling so that models with any
    number of outputs (e.g. 3 for RTMW3D) work correctly.
    """
    from rtmlib.tools import base as rtmlib_base

    global _ORIG_BASE_INIT
    if _ORIG_BASE_INIT is not None:
        return  # already patched

    _ORIG_BASE_INIT = rtmlib_base.BaseTool.__init__

    def _patched_init(
        self,
        onnx_model=None,
        model_input_size=None,
        mean=None,
        std=None,
        backend="opencv",
        device="cpu",
    ):
        if backend == "openvino":
            from pathlib import Path

            from openvino import Core
            from rtmlib.tools.file import download_checkpoint

            if onnx_model is None:
                raise ValueError("onnx_model is required for the openvino backend")
            if not Path(onnx_model).exists():
                onnx_model = download_checkpoint(onnx_model)

            core = Core()
            model_onnx = core.read_model(model=onnx_model)

            ov_device = device.upper() if device else "CPU"

            # NPU requires static shapes — freeze any dynamic dimensions
            if ov_device == "NPU":
                input_shape = model_onnx.input(0).partial_shape
                if input_shape.is_dynamic:
                    static = []
                    for dim in input_shape:
                        if dim.is_dynamic:
                            static.append(1)
                        else:
                            static.append(dim.get_length())
                    print(f"  Reshaping to static {static} for NPU")
                    model_onnx.reshape(static)

            try:
                self.compiled_model = core.compile_model(
                    model=model_onnx, device_name=ov_device, config={"PERFORMANCE_HINT": "LATENCY"}
                )
            except RuntimeError as exc:
                if ov_device != "CPU":
                    print(
                        f"WARNING: Failed to compile on {ov_device} ({exc}), falling back to CPU."
                    )
                    model_onnx = core.read_model(model=onnx_model)
                    self.compiled_model = core.compile_model(
                        model=model_onnx, device_name="CPU", config={"PERFORMANCE_HINT": "LATENCY"}
                    )
                    ov_device = "CPU"
                else:
                    raise

            n_outputs = len(self.compiled_model.outputs)
            self.input_layer = self.compiled_model.input(0)
            self._ov_output_layers = [self.compiled_model.output(i) for i in range(n_outputs)]
            # Backward compat for rtmlib code that uses these directly
            self.output_layer0 = self._ov_output_layers[0]
            self.output_layer1 = self._ov_output_layers[1]

            print(f"load {onnx_model} with openvino/{ov_device} backend ({n_outputs} outputs)")

            self.onnx_model = onnx_model
            self.model_input_size = model_input_size
            self.mean = mean
            self.std = std
            self.backend = backend
            self.device = device
        else:
            # rtmlib typed these `str = None` / `tuple = None`; in practice
            # non-None is always passed when the non-openvino branch runs.
            assert onnx_model is not None
            assert model_input_size is not None
            assert mean is not None
            assert std is not None
            _ORIG_BASE_INIT(
                self,
                onnx_model=onnx_model,
                model_input_size=model_input_size,
                mean=mean,
                std=std,
                backend=backend,
                device=device,
            )

    rtmlib_base.BaseTool.__init__ = _patched_init  # ty: ignore[invalid-assignment]

    # Patch inference() so models with >2 outputs work.
    _orig_inference = rtmlib_base.BaseTool.inference

    def _patched_inference(self, img):
        if self.backend != "openvino" or not hasattr(self, "_ov_output_layers"):
            return _orig_inference(self, img)

        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        input_tensor = img[None, :, :, :]

        results = self.compiled_model(input_tensor)
        return [results[layer] for layer in self._ov_output_layers]

    rtmlib_base.BaseTool.inference = _patched_inference  # ty: ignore[invalid-assignment]
