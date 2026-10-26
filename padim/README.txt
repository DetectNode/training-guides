DetectNode PaDiM training kit

1. Install Python 3.11 or 3.12 (3.10+ works) from python.org/downloads.
2. Train:
   Windows:      double-click start_windows.bat
   macOS/Linux:  bash start_linux.sh
   First run creates a shared environment and installs CPU PyTorch +
   requirements.
3. When [DONE] appears, drop the generated .detectnode into DetectNode.

Manual alternative (the script creates/reuses the shared environment):
   Windows:      py -3 start.py
   macOS/Linux:  python3 start.py

Shared environment (kept outside the dataset folder):
   Windows: %LOCALAPPDATA%\DetectNode\padim-venv
   macOS:   ~/Library/Application Support/DetectNode/padim-venv
   Linux:   ~/.local/share/DetectNode/padim-venv

It is reused for every kit and is not deleted when the dataset folder is
deleted. Use bash start_linux.sh; no chmod is required.

data/ contains dataset.json, camera.json, images/ and all training output
(augmented/, output_model/, results/) plus training.log. Only the generated
.detectnode package is written at the kit root.
Keep camera.json inside the generated .detectnode.
