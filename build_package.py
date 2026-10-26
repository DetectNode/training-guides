#!/usr/bin/env python3
"""
Build a DetectNode ``.detectnode`` model package.

A ``.detectnode`` package is a plain ZIP archive containing:

    model.onnx | model.tflite    the inference model (required)
    model.json                    the architecture / parser schema (required)
    labels.txt                    class labels (required for detection models)
    camera.json                   camera binding (PaDiM models, optional)
    LICENSE / THIRD_PARTY_NOTICES / README ...
                                  license & notice files (optional)
    signature.json                Ed25519 signature (added separately, see below)

Run ``python3 build_package.py --help`` for the full option list. The script only
validates that the pieces are consistent; it does not train or convert anything.

Example:

    python3 build_package.py \
        --model  runs/export/best.onnx \
        --json   model.json \
        --labels data/labels.txt

Signing
-------
DetectNode supports Ed25519 signing of packages (an optional ``signature.json``
entry). The signing tool lives in the main repository at ``tools/sign_detectnode.py``.
Signing is currently a demonstration-only feature while the app is in beta: the
camera runs unsigned models without any restriction, and the signing key is not
shipped. See the training guide README for details.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path


def infer_task_type(schema: dict) -> str:
    """Return 'detection', 'anomaly' or 'segmentation' from a schema object."""
    task = schema.get("task")
    if isinstance(task, dict) and task.get("task_type") in (
        "detection",
        "anomaly",
        "segmentation",
    ):
        return task["task_type"]

    # Flat (non-nested) form: the output keys identify the task.
    output = schema.get("output")
    if isinstance(output, dict):
        if "mask_tensor_index" in output:
            return "segmentation"
        if "heatmap_tensor_index" in output:
            return "anomaly"
    return "detection"


def load_schema(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"error: cannot read architecture JSON: {path} ({exc})")
    if not isinstance(data, dict):
        raise SystemExit(f"error: architecture JSON must be an object: {path}")
    if "version" not in data:
        raise SystemExit(f"error: architecture JSON has no 'version' field: {path}")
    return data


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Build a DetectNode .detectnode model package."
    )
    parser.add_argument("--model", required=True, help="model file (.onnx or .tflite)")
    parser.add_argument("--json", required=True, help="architecture / parser schema JSON")
    parser.add_argument("--labels", help="labels.txt (required for detection models)")
    parser.add_argument("--camera", help="camera.json (PaDiM camera binding)")
    parser.add_argument(
        "--notice",
        action="append",
        default=[],
        metavar="FILE",
        help="license/notice file (repeatable): LICENSE, THIRD_PARTY_NOTICES, README, ...",
    )
    parser.add_argument(
        "--out",
        help="output package path (default: <model-stem>.detectnode in the current directory)",
    )
    args = parser.parse_args(argv)

    model = Path(args.model)
    if not model.is_file():
        raise SystemExit(f"error: model file not found: {model}")
    ext = model.suffix.lower()
    if ext not in (".onnx", ".tflite"):
        raise SystemExit("error: --model must be a .onnx or .tflite file")

    json_path = Path(args.json)
    schema = load_schema(json_path)
    task_type = infer_task_type(schema)

    labels_path = Path(args.labels) if args.labels else None
    if labels_path is not None and not labels_path.is_file():
        raise SystemExit(f"error: labels file not found: {labels_path}")

    if task_type == "detection" and labels_path is None:
        raise SystemExit(
            "error: detection models require --labels (a labels.txt file).\n"
            "       One class name per line, in tensor class order."
        )

    camera_path = Path(args.camera) if args.camera else None
    if camera_path is not None and not camera_path.is_file():
        raise SystemExit(f"error: camera.json not found: {camera_path}")

    notice_paths = [Path(p) for p in args.notice]
    for notice in notice_paths:
        if not notice.is_file():
            raise SystemExit(f"error: notice file not found: {notice}")

    out_path = Path(args.out) if args.out else Path(model.stem + ".detectnode")

    entries: list[tuple[str, Path]] = [
        (f"model{ext}", model),
        ("model.json", json_path),
    ]
    if labels_path is not None:
        entries.append(("labels.txt", labels_path))
    if camera_path is not None:
        entries.append(("camera.json", camera_path))
    for notice in notice_paths:
        # Keep the basename so the uploader recognizes it as a notice file.
        entries.append((notice.name, notice))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for arcname, path in entries:
            zf.write(path, arcname=arcname)

    print(f"Built {out_path} ({task_type})")
    print(f"  model:  {model.name}")
    print(f"  schema: {json_path.name}")
    if labels_path is not None:
        print(f"  labels: {labels_path.name}")
    if camera_path is not None:
        print(f"  camera: {camera_path.name}")
    for notice in notice_paths:
        print(f"  notice: {notice.name}")
    print()
    print("Drop this file into DetectNode (or sign it with tools/sign_detectnode.py).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
