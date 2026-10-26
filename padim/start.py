#!/usr/bin/env python3
"""DetectNode PaDiM training kit.

The downloaded zip contains:

    data/dataset.json
    data/camera.json
    data/images/...
    start.py
    requirements.txt

Run ``start_windows.bat`` (Windows) or ``start_linux.sh`` (macOS/Linux).
The script creates a shared environment in the user's data folder and installs
the requirements there. It never uses Anomalib or other packages from the
user's global Python environment; the same environment is reused for every kit
and is kept when the dataset folder is deleted.

All generated folders stay under ``data/`` (augmented images, checkpoints,
model export); only the final ``.detectnode`` package is written at the kit
root.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import warnings
import zipfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

# ============================================================
# Paths and parameters
# ============================================================
KIT_DIR = Path(__file__).resolve().parent
DATA_DIR = KIT_DIR / "data"
AUGMENTED_DIR = DATA_DIR / "augmented"
OUTPUT_DIR = DATA_DIR / "output_model"
RESULTS_DIR = DATA_DIR / "results"
TRAINING_LOG = DATA_DIR / "training.log"


def shared_venv_dir() -> Path:
    """Store the reusable environment outside the dataset folder.

    Users often delete the downloaded dataset after training. Keeping the venv
    in a standard per-user data directory also avoids the dataset/Downloads
    location, which antivirus scanners tend to treat more aggressively.
    """
    if os.name == "nt":
        base = Path(
            os.environ.get(
                "LOCALAPPDATA", Path.home() / "AppData" / "Local"
            )
        )
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(
            os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")
        )
    return base / "DetectNode" / "padim-venv"


VENV_DIR = shared_venv_dir()
REQUIREMENTS = KIT_DIR / "requirements.txt"
VENV_MARKER = VENV_DIR / ".detectnode-ready"
# Bump this whenever the dependency bootstrap changes (for example when moving
# from CUDA torch to CPU-only torch). It forces a clean venv on the next run.
BOOTSTRAP_VERSION = 2

IMAGE_SIZE = (256, 256)
BATCH_SIZE = 4
AUGMENTATIONS_PER_IMAGE = 10
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]
SOURCE_NORMAL_DIR = "images"


def progress(percent: int, text: str) -> None:
    """Small, cross-platform terminal progress line."""
    width = 28
    filled = max(0, min(width, round(width * percent / 100)))
    bar = "#" * filled + "." * (width - filled)
    print(f"\r[{bar}] {percent:3d}% {text}", end="", flush=True)
    if percent >= 100:
        print()


# ============================================================
# Isolated Python environment
# ============================================================
def venv_python() -> Path:
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def requirements_hash() -> str:
    return hashlib.sha256(REQUIREMENTS.read_bytes()).hexdigest()


def marker_value() -> str:
    return f"{BOOTSTRAP_VERSION}:{requirements_hash()}"


def marker_is_current() -> bool:
    if not VENV_MARKER.exists():
        return False
    try:
        return VENV_MARKER.read_text(encoding="utf-8").strip() == marker_value()
    except OSError:
        return False


def marker_has_current_bootstrap() -> bool:
    if not VENV_MARKER.exists():
        return False
    try:
        value = VENV_MARKER.read_text(encoding="utf-8").strip()
    except OSError:
        return False
    return value.startswith(f"{BOOTSTRAP_VERSION}:")


def inside_our_venv() -> bool:
    # Do not compare sys.executable paths: a venv's python is a symlink to the
    # system interpreter, so resolve() would make a global Python look like
    # ours. sys.prefix is the venv root when the venv is active.
    try:
        return Path(sys.prefix).resolve() == VENV_DIR.resolve()
    except OSError:
        return False


def install_requirements() -> None:
    """Install requirements into the shared venv, reusing it when unchanged."""
    if marker_is_current():
        return

    python_exe = venv_python()
    print(f"[SETUP] Shared environment: {VENV_DIR}")
    print(f"[SETUP] Full log: {TRAINING_LOG}")

    def run_logged(command: list[str], label: str) -> None:
        TRAINING_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(TRAINING_LOG, "a", encoding="utf-8") as log:
            log.write(f"\n=== {label} ===\n")
            log.flush()
            try:
                subprocess.check_call(command, stdout=log, stderr=log)
            except (subprocess.CalledProcessError, OSError) as exc:
                raise SystemExit(
                    f"{label} failed ({exc}). See {TRAINING_LOG}"
                ) from exc

    print("[SETUP] Installing CPU-only PyTorch (no CUDA download)...")
    run_logged(
        [
            str(python_exe),
            "-m",
            "pip",
            "install",
            "--index-url",
            "https://download.pytorch.org/whl/cpu",
            "torch",
            "torchvision",
        ],
        "CPU PyTorch install",
    )

    print("[SETUP] Installing the remaining training requirements...")
    print("[SETUP] This needs internet and can take several minutes.")
    run_logged(
        [str(python_exe), "-m", "pip", "install", "-r", str(REQUIREMENTS)],
        "training requirements install",
    )
    VENV_MARKER.write_text(marker_value() + "\n", encoding="utf-8")


def ensure_environment() -> None:
    """Always run training from the DetectNode-owned virtual environment."""
    if sys.version_info < (3, 10):
        raise SystemExit("Python 3.10 or newer is required.")

    if not REQUIREMENTS.exists():
        raise SystemExit("requirements.txt is missing from the training kit.")

    # An older shared env (for example one that pulled CUDA torch) is recreated
    # once so the dependency set matches this kit.
    if VENV_DIR.exists() and VENV_MARKER.exists() and not marker_has_current_bootstrap():
        print("[SETUP] Recreating the shared environment with the updated dependency setup...")
        shutil.rmtree(VENV_DIR, ignore_errors=True)

    python_exe = venv_python()
    if not python_exe.exists():
        print("[SETUP] Creating the shared DetectNode environment...")
        try:
            VENV_DIR.parent.mkdir(parents=True, exist_ok=True)
            subprocess.check_call(
                [sys.executable, "-m", "venv", str(VENV_DIR)]
            )
        except (subprocess.CalledProcessError, OSError) as exc:
            raise SystemExit(
                "Could not create the shared environment: "
                f"{exc}\n"
                "Install Python 3.10+ (with venv support) and run the script again."
            )

    try:
        install_requirements()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            "Could not install the training requirements: "
            f"{exc}\n"
            "Check your internet connection, then run the script again."
        )

    if inside_our_venv():
        return

    # A global Anomalib (or a different Python environment) must never be used.
    print("[SETUP] Restarting inside the DetectNode environment...")
    raise SystemExit(
        subprocess.call(
            [str(python_exe), str(Path(__file__).resolve()), *sys.argv[1:]]
        )
    )


# ============================================================
# Dataset helpers
# ============================================================
def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_dataset() -> dict[str, Any]:
    """Validate the dataset and keep the camera binding mandatory."""
    dataset_path = DATA_DIR / "dataset.json"
    camera_path = DATA_DIR / "camera.json"

    if not dataset_path.exists():
        raise SystemExit("data/dataset.json is missing from the training kit.")
    if not camera_path.exists():
        raise SystemExit(
            "data/camera.json is missing. It is required so the trained model "
            "stays bound to the camera view that produced this dataset."
        )

    dataset = read_json(dataset_path)
    camera = read_json(camera_path)

    if dataset.get("format") != "detectnode-padim-dataset":
        raise SystemExit("data/dataset.json is not a DetectNode PaDiM dataset.")
    if camera.get("channel") in (None, "") and camera.get("source_id") in (None, ""):
        raise SystemExit("data/camera.json has no camera view information.")

    expected = dataset.get("camera") or {}
    for key in ("source_id", "channel", "name"):
        if str(camera.get(key, "")) != str(expected.get(key, "")):
            raise SystemExit(
                "data/camera.json does not match data/dataset.json. Keep both "
                "files as they were downloaded; the model must stay bound to "
                "this camera."
            )

    return dataset


def check_images() -> list[Path]:
    images = sorted(
        p for p in (DATA_DIR / SOURCE_NORMAL_DIR).iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not images:
        raise SystemExit(f"No images found in data/{SOURCE_NORMAL_DIR}/.")
    return images


def create_augmented_dataset(
    source_images: Path, target_dir: Path, augmentations_per_image: int
) -> int:
    """Create the letterboxed/augmented training set used by the reference config."""
    import torchvision.transforms.v2 as T
    from PIL import Image

    normal_dst = target_dir / "normal"
    if target_dir.exists():
        shutil.rmtree(target_dir)
    normal_dst.mkdir(parents=True, exist_ok=True)

    def letterbox(image: Image.Image) -> Image.Image:
        """Fit the full camera view into a square with black bars.

        DetectNode's live VDO stream uses image.fit=scale, which preserves the
        aspect ratio and pads the square model input. Training must use the
        same geometry or PaDiM flags the bars as anomalies.
        """
        image = image.convert("RGB")
        width, height = image.size
        scale = min(IMAGE_SIZE[0] / width, IMAGE_SIZE[1] / height)
        new_w = max(1, round(width * scale))
        new_h = max(1, round(height * scale))
        # A dataset captured at the model content size (256x144) is already
        # aligned; avoid a needless resample.
        resized = (
            image
            if (new_w == width and new_h == height)
            else image.resize((new_w, new_h), Image.Resampling.BILINEAR)
        )
        canvas = Image.new("RGB", IMAGE_SIZE, (0, 0, 0))
        canvas.paste(
            resized,
            ((IMAGE_SIZE[0] - new_w) // 2, (IMAGE_SIZE[1] - new_h) // 2),
        )
        return canvas

    augmenter = T.Compose([
        T.ColorJitter(brightness=0.06, contrast=0.06, saturation=0.03),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
        T.RandomAffine(degrees=0, translate=(0.02, 0.02)),
    ])

    image_paths = sorted(
        p for p in source_images.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    count = 0
    for image_path in image_paths:
        with Image.open(image_path) as raw_image:
            image = raw_image.convert("RGB")
            aligned = letterbox(image)
            aligned.save(normal_dst / f"orig_{count}.png")
            count += 1

            for k in range(augmentations_per_image):
                augmented = augmenter(aligned)
                augmented.save(normal_dst / f"aug_{count}_{k}.png")

    print(f"\n[DATA] Prepared {count} source images ({len(image_paths)} + augmentations).")
    return count


# ============================================================
# Training, calibration and export
# ============================================================
def train_and_export() -> tuple[Path, dict[str, float]]:
    """Train PaDiM, calibrate the score range and export the core model."""
    import torch
    import torchvision.transforms.v2 as T
    from anomalib.data import Folder
    from anomalib.engine import Engine
    from anomalib.models import Padim

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    progress(25, "Preparing the training dataset...")
    create_augmented_dataset(
        DATA_DIR / SOURCE_NORMAL_DIR, AUGMENTED_DIR, AUGMENTATIONS_PER_IMAGE
    )

    datamodule = Folder(
        name="custom_dataset",
        root=AUGMENTED_DIR,
        normal_dir="normal",
        train_batch_size=BATCH_SIZE,
        num_workers=0,
    )
    datamodule.setup()

    model = Padim()
    engine = Engine(
        accelerator="cpu",
        default_root_dir=RESULTS_DIR,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )

    progress(40, "Training PaDiM (this can take several minutes)...")
    try:
        # Third-party libraries are noisy. Keep their full output in
        # data/training.log and show only the clean progress line.
        with open(TRAINING_LOG, "a", encoding="utf-8") as log:
            with redirect_stdout(log), redirect_stderr(log):
                engine.fit(model=model, datamodule=datamodule)
    except Exception:
        print("\n[ERROR] Training failed. Full log: data/training.log")
        raise

    device = torch.device("cpu")
    core_model = model.model.eval().to(device)

    calibration_transform = T.Compose([
        T.Resize(IMAGE_SIZE, antialias=True),
        T.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])

    progress(70, "Calibrating thresholds...")
    all_scores: list[float] = []
    with torch.no_grad():
        for batch in datamodule.train_dataloader():
            images = batch["image"].to(device)
            if images.dtype == torch.uint8 or images.max() > 1.0:
                images = images.float() / 255.0

            normalized = calibration_transform(images)
            output = core_model(normalized)

            if hasattr(output, "anomaly_map"):
                heatmap = output.anomaly_map
            elif isinstance(output, dict) and "anomaly_map" in output:
                heatmap = output["anomaly_map"]
            elif isinstance(output, (tuple, list)):
                heatmap = output[0]
            else:
                heatmap = output

            if not isinstance(heatmap, torch.Tensor) or heatmap.dim() < 2:
                raise SystemExit("The model did not return a usable anomaly map.")

            batch_max = heatmap.view(heatmap.size(0), -1).max(dim=1).values
            all_scores.extend(batch_max.cpu().tolist())

    if not all_scores:
        raise SystemExit("No calibration scores were produced.")

    min_score = float(min(all_scores))
    max_normal = float(max(all_scores))
    max_score = max_normal * 1.60
    if max_score <= min_score:
        max_score = min_score + max(1e-3, abs(min_score) * 0.1 + 1e-3)

    # Suggested operating point, stored as metadata for the frontend. It is
    # NOT used at runtime: the per-scenario analysis threshold decides.
    recommended_raw = max_score * 0.75
    denom = max_score - min_score
    recommended_pct = (
        (recommended_raw - min_score) / denom * 100.0 if denom > 0 else 50.0
    )
    recommended_pct = max(0.0, min(100.0, recommended_pct))

    print(
        f"\n[CALIB] min={min_score:.4f} max_normal={max_normal:.4f} "
        f"recommended={recommended_pct:.1f}%"
    )

    class AnomalyMapExport(torch.nn.Module):
        """Expose only the anomaly map as the single ONNX output."""

        def __init__(self, core: torch.nn.Module) -> None:
            super().__init__()
            self.core = core

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.core(x)
            if hasattr(out, "anomaly_map"):
                return out.anomaly_map
            if isinstance(out, dict):
                return out["anomaly_map"]
            if isinstance(out, (tuple, list)):
                return out[0]
            return out

    wrapper = AnomalyMapExport(core_model).eval().to(device)
    dummy_input = torch.randn(
        1, 3, IMAGE_SIZE[1], IMAGE_SIZE[0], dtype=torch.float32, device=device
    )
    onnx_path = OUTPUT_DIR / "model.onnx"

    progress(85, "Exporting the model to ONNX...")
    torch.onnx.export(
        wrapper,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamo=False,
    )

    return onnx_path, {
        "min_score": min_score,
        "max_score": max_score,
        "global_min_score": min_score,
        "global_max_score": max_score,
        "recommended_threshold_pct": recommended_pct,
    }


def write_model_json(calibration: dict[str, float]) -> Path:
    """Write the anomaly parser schema expected by DetectNode."""
    config = {
        "version": 1,
        "input": {
            "layout": "NCHW",
            "width": IMAGE_SIZE[0],
            "height": IMAGE_SIZE[1],
            "preprocess": "letterbox",
            "channel_order": "rgb",
            "normalize": {
                "mean": NORM_MEAN,
                "std": NORM_STD,
            },
        },
        "task": {
            "task_type": "anomaly",
            "output": {
                "heatmap_tensor_index": 0,
                "compute_global_score": True,
                "global_score_tensor_index": -1,
                "recommended_threshold_pct": calibration[
                    "recommended_threshold_pct"
                ],
                "min_score": calibration["min_score"],
                "max_score": calibration["max_score"],
                "global_min_score": calibration["global_min_score"],
                "global_max_score": calibration["global_max_score"],
            },
        },
    }

    out_path = OUTPUT_DIR / "model.json"
    out_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return out_path


def build_package(
    dataset: dict[str, Any], onnx_path: Path, model_json: Path
) -> Path:
    """Build the .detectnode package expected by DetectNode."""
    name = dataset.get("model", {}).get("name") or "padim_model"
    package_path = KIT_DIR / f"{name}.detectnode"

    with zipfile.ZipFile(package_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(onnx_path, arcname="model.onnx")
        zf.write(model_json, arcname="model.json")
        # Mandatory: keeps the model tied to this exact camera view.
        zf.write(DATA_DIR / "camera.json", arcname="camera.json")

        readme = KIT_DIR / "README.txt"
        if readme.exists():
            zf.write(readme, arcname="README.txt")

    return package_path


def main() -> int:
    ensure_environment()
    progress(10, "Environment ready")

    dataset = load_dataset()
    images = check_images()
    progress(20, f"Dataset ready ({len(images)} images)")

    print()
    print("=" * 70)
    print("DetectNode PaDiM training")
    print(f"  Dataset: {DATA_DIR}")
    print(f"  Camera:  {dataset.get('camera', {}).get('name', '?')}")
    print(f"  Images:  {len(images)}")
    print(f"  Python:  {sys.executable}")
    print("=" * 70)

    try:
        onnx_path, calibration = train_and_export()
        model_json = write_model_json(calibration)
        progress(95, "Building the model package...")
        package_path = build_package(dataset, onnx_path, model_json)
    finally:
        if AUGMENTED_DIR.exists():
            shutil.rmtree(AUGMENTED_DIR, ignore_errors=True)

    progress(100, "Done")
    print()
    print("=" * 70)
    print("  DONE — model package ready")
    print()
    print(f"  {package_path}")
    print()
    print("  Drop this file into DetectNode.")
    print("=" * 70)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
