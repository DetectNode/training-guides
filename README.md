# DetectNode — Training Guides

This folder explains how to train or obtain models for
[DetectNode](https://github.com/DetectNode) or other applications and how to get them into the camera
app — either by packaging them as a `.detectnode` file, or by uploading the
individual files (model, architecture JSON, labels) manually. 
The `.detectnode` package is a convenience wrapper that also enables signing and the camera binding; everything you need to build one by hand is described below.

A `.detectnode` package is a plain ZIP archive that bundles a model, an
architecture description (the `model.json` parser schema), optional class labels,
and optional metadata (camera binding, licenses, signature).

## Contents

- [Supported model paths](#supported-model-paths)
- [The `.detectnode` package](#the-detectnode-package)
- [The architecture JSON (`model.json`)](#the-architecture-json-modeljson)
- [Model guides](#model-guides)
  - [PaDiM (anomaly detection)](#padim-anomaly-detection)
  - [YOLOv5 (TFLite / DLPU)](#yolov5-tflite--dlpu)
- [Signing models](#signing-models)
- [Contributing](#contributing)

## Supported model paths

DetectNode supports three task types — **detection**, **anomaly** and
**segmentation** — and imports either `.onnx` (ONNX Runtime) or `.tflite`
(Larod / DLPU) model files. The guides in this folder currently cover:

| Model | Task | Format | Guide |
| --- | --- | --- | --- |
| PaDiM | anomaly | `.onnx` | included in the app (see below) |
| YOLOv5 | detection | `.tflite` | Axis model zoo guide (linked below) |

The app accepts **both** formats regardless of task: PaDiM runs on the ONNX
backend, YOLOv5 runs on the TFLite/DLPU backend, and any other ONNX or TFLite
model can be imported through the manual flow with a matching `model.json`.

> **Note on backends:** the ONNX backend runs on the **CPU**, while the TFLite
> backend can run on the **DLPU or the CPU**. ONNX is therefore a good fit for
> anomaly (PaDiM) and segmentation workloads, but detection models are generally
> less useful on ONNX — prefer the TFLite path for detection to benefit from
> DLPU hardware acceleration.

These guides are **open to contributions** — see [Contributing](#contributing)
if you want to add a model (YOLOv8, RT-DETR, a segmentation model, …).

## The `.detectnode` package

A `.detectnode` file is a ZIP archive. The uploader on the camera recognizes the
following entries (filenames are case-insensitive):

| Entry | Required | Purpose |
| --- | --- | --- |
| `model.onnx` / `model.tflite` | yes (exactly one) | the inference model |
| `model.json` (any `.json`) | yes (exactly one) | the architecture / parser schema |
| `labels.txt` (any `labels`/`classes` `.txt`) | detection only | one class name per line, in class-index order |
| `camera.json` | PaDiM only | binds the model to the camera view it was trained on |
| `LICENSE`, `THIRD_PARTY_NOTICES`, `README`, … | no | license/notice files, shown in the app |
| `signature.json` | no | optional Ed25519 signature (see [Signing](#signing-models)) |

Rules enforced on import:

- exactly **one** model file and exactly **one** architecture `.json`;
- a **detection** model must carry a labels file;
- a **PaDiM** dataset model must carry `camera.json`.

Use [`build_package.py`](build_package.py) to assemble a package instead of
zipping by hand:

```bash
python3 build_package.py \
  --model  best.onnx \
  --json   model.json \
  --labels labels.txt
```

### Uploading files manually

You do **not** have to build a `.detectnode` package. DetectNode also accepts the
files individually: upload the model (`.onnx` or `.tflite`), then set or import
the architecture JSON in the model editor, and (for detection models) upload the
labels file separately.

The `.detectnode` package is purely a **convenience** wrapper that bundles those
pieces into one file. It is also the only format that can carry:

- a **signature** (`signature.json`) — see [Signing models](#signing-models);
- a **camera binding** (`camera.json`) — required for PaDiM models.

For a plain detection model, uploading the model, JSON and labels manually works
exactly the same way.

## The architecture JSON (`model.json`)

The `model.json` describes how DetectNode should preprocess the input and decode
the model's output tensors. It is the **parser schema** defined in the main
repository at `app/schema/parser_definition.json` (generated into
`ParserSchema_generated.hpp`). The canonical form is:

```json
{
  "version": 1,
  "input":  { "...": "..." },
  "task":   {
    "task_type": "detection | anomaly | segmentation",
    "output":     { "...": "..." },
    "postprocess": { "...": "..." }   // detection only
  },
  "dlpu": "artpec8 | artpec9"          // optional hardware-target hint
}
```

A flat form (with `output`, `postprocess` and `task_type` at the top level) is
also accepted for backwards compatibility and normalized automatically.

### `input` fields

| Field | Type | Required | Values / notes |
| --- | --- | --- | --- |
| `input.layout` | enum | yes | `NCHW` or `NHWC` |
| `input.width` | int | yes | 1–8192 |
| `input.height` | int | yes | 1–8192 |
| `input.preprocess` | enum | yes | `letterbox`, `resize` or `crop` |
| `input.channel_order` | enum | yes | `bgr` or `rgb` |
| `input.debug_stats` | bool | yes | opt-in per-tensor debug logging |
| `input.normalize.mean` | array[3] | no | per-channel mean (empty = off) |
| `input.normalize.std` | array[3] | no | per-channel std (must be > 0) |
| `dlpu` | string | no | `""`, `artpec8` or `artpec9` |

### `task.output` — detection

| Field | Required | Values / notes |
| --- | --- | --- |
| `output.layout` | yes | `row-major`, `column-major`, `two-output`, `four-output` |
| `output.box_format` | yes | `center-wh` (`cx, cy, w, h`) or `corners` (`x1, y1, x2, y2`) |
| `output.box_coordinate_order` | yes | `xy` or `yx` (TensorFlow historical) |
| `output.class_mode` | row/col | `argmax`, `argmax-x-obj` or `explicit-id` |
| `output.boxes_layout` | two/four | `row-major` or `column-major` |
| `output.logits_layout` | two | `row-major` or `column-major` |
| `output.box_offset` | row/col | index of the first box coordinate |
| `output.confidence_index` | explicit-id | index of the confidence value |
| `output.class_offset` | row/col | first class-score index (or class-id index for `explicit-id`) |
| `output.objectness_index` | argmax-x-obj | index of the objectness value |
| `output.background_class_index` | row/col/two | class to ignore during argmax (`-1` = off) |
| `output.num_classes` | row/col/two | number of classes (`0` = derive from tensor) |
| `output.boxes_tensor_index` | two/four | index of the boxes tensor |
| `output.logits_tensor_index` | two/four | index of the logits / class-id tensor |
| `output.scores_tensor_index` | four | index of the confidences tensor |
| `output.num_det_tensor_index` | four | index of the valid-detection-count tensor |
| `output.normalized_coords` | yes | boxes in `[0,1]` (scaled to model size) vs pixels |
| `output.sigmoid_logits` | row/col/two | apply `1/(1+e^-x)` to the final score |

### `task.postprocess` — detection

| Field | Required | Notes |
| --- | --- | --- |
| `postprocess.top_k_before_nms` | yes | sort + cap before NMS (`0` = unlimited) |
| `postprocess.has_builtin_nms` | yes | the model already applies NMS (e.g. TFLite postprocess) |

### `task.output` — anomaly (PaDiM)

| Field | Required | Notes |
| --- | --- | --- |
| `output.heatmap_tensor_index` | yes | index of the anomaly-map tensor |
| `output.compute_global_score` | yes | `true` = max of the heatmap, `false` = dedicated tensor |
| `output.global_score_tensor_index` | yes | index of the global-score tensor (`-1` when computed) |
| `output.recommended_threshold_pct` | no | suggested operating point shown in the UI |
| `output.min_score` / `max_score` | yes | heatmap pixel normalization range |
| `output.global_min_score` / `global_max_score` | yes | image-level score range |

### `task.output` — segmentation

| Field | Required | Notes |
| --- | --- | --- |
| `output.mask_tensor_index` | yes | index of the mask tensor |
| `output.mask_mode` | yes | `binary` or `multiclass` |
| `output.mask_threshold` | yes | per-class decode threshold (0–1) |
| `output.mask_apply_sigmoid` | binary | apply sigmoid to the channel probability |
| `output.mask_channel_index` | yes | channel of the class of interest |
| `output.mask_num_classes` | yes | number of classes (1–255) |

## Model guides

### PaDiM (anomaly detection)

PaDiM training is **included in the app**: when you download a PaDiM dataset from
DetectNode, the downloaded archive contains everything needed to train — the
dataset (`data/dataset.json`, `data/camera.json`, `data/images/`) **and** a
training script.

- **Windows:** double-click `start_windows.bat`
- **macOS / Linux:** `bash start_linux.sh`

The script creates a shared Python environment, trains PaDiM with
[Anomalib](https://github.com/openvinotoolkit/anomalib), exports an ONNX model,
and produces a ready-to-import `.detectnode` package (with `model.json` and
`camera.json`) — no manual packaging required.

The kit is mirrored in [`padim/`](padim/) for reference:

- [`padim/start.py`](padim/start.py) — the training + export script
- [`padim/requirements.txt`](padim/requirements.txt) — Python dependencies
- [`padim/start_linux.sh`](padim/start_linux.sh) / [`padim/start_windows.bat`](padim/start_windows.bat) — launchers
- [`padim/README.txt`](padim/README.txt) — quick start

See [`examples/padim.model.json`](examples/padim.model.json) for the architecture
JSON that the script generates (values are auto-filled from calibration).

### YOLOv5 (TFLite / DLPU)

For ARTPEC-8 / ARTPEC-9 cameras, export YOLOv5 to TFLite and run it on the DLPU.
Follow the Axis model zoo guide:

> <https://github.com/AxisCommunications/axis-model-zoo/blob/main/docs/yolov5.md>

It covers cloning `ultralytics/yolov5`, applying the Axis patch, training, and
exporting with `--include tflite --int8`.

The exported TFLite model runs the detection postprocess (NMS) on-device and
produces **four** output tensors. Use
[`examples/yolov5-tflite.model.json`](examples/yolov5-tflite.model.json) as the
architecture JSON and adjust `num_classes` / the labels file.

## Signing models

DetectNode supports **Ed25519 signing** of `.detectnode` packages: an optional
`signature.json` entry carrying a detached signature over a canonical manifest of
every other entry. A valid signature shows a "Signed Model" badge in the app UI.
The signing/verification tool lives in the main repository at
`tools/sign_detectnode.py`.

> **Status:** signing is **not available yet** — the app is currently in beta and
> the trusted public key is not baked into published builds. The feature exists to
> demonstrate the capability; in the meantime, **unsigned models run without any
> restriction** on the camera.

## Contributing

The training guides are **open to contributions** for adding models beyond the
ones documented here (YOLOv8, RT-DETR/DEIM, segmentation models, …).

To add a model:

1. Add a guide (a `README.md` under its own subfolder, or extend an existing one)
   that explains how to train and export the model.
2. Add an example architecture JSON under [`examples/`](examples/).
3. Mention it in the [Supported model paths](#supported-model-paths) table.
4. Open a pull request.

Ready-made example models and configs are also welcome at
<https://github.com/DetectNode/example-models>.
