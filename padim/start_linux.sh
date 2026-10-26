#!/usr/bin/env bash
cd "$(dirname "$0")" || exit 1

export PYTHONWARNINGS=ignore

if command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
elif command -v python >/dev/null 2>&1; then
  PYTHON=python
else
  printf '\nPython 3 is not installed.\n'
  printf 'Install Python 3.10 or newer from https://www.python.org/downloads/\n\n'
  exit 1
fi

"$PYTHON" start.py
status=$?

if [ "$status" -ne 0 ]; then
  printf '\nTraining failed (exit %s). See the error above.\n\n' "$status"
fi

# Keep the window open when launched from a desktop, but do not block scripts
# or CI runs that have no terminal attached.
if [ -t 0 ]; then
  read -r -p "Press Enter to close this window..." _
fi

exit "$status"
