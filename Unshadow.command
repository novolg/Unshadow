#!/bin/zsh
# Double-click to start the Unshadow editor and open it in the browser.
cd "$(dirname "$0")" || exit 1
PORT="${UNSHADOW_PORT:-8765}"
URL="http://127.0.0.1:$PORT"

# Already running? Just open the page.
if curl -s -o /dev/null "$URL/api/status"; then
  open "$URL"
  exit 0
fi

# Find a Python that has OpenCV + NumPy.
PY=""
for p in /opt/homebrew/bin/python3.11 /usr/local/bin/python3.11 python3.11 python3; do
  if command -v "$p" >/dev/null 2>&1 && "$p" -c "import cv2, numpy" >/dev/null 2>&1; then PY="$p"; break; fi
done
if [[ -z "$PY" ]]; then
  echo "No Python with OpenCV found. Install: python3.11 -m pip install numpy opencv-contrib-python"
  read -k 1 "?Press any key to close."
  exit 1
fi
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg not found. Install: brew install ffmpeg"
  read -k 1 "?Press any key to close."
  exit 1
fi

echo "Starting Unshadow at $URL  (close this window to stop it)"
( for i in {1..40}; do curl -s -o /dev/null "$URL/api/status" && open "$URL" && break; sleep 0.25; done ) &
exec "$PY" src/editor.py --port "$PORT"
