#!/bin/bash
# Regenerate paper.pdf from paper.html via headless Chrome.
set -euo pipefail
cd "$(dirname "$0")"
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --headless --disable-gpu \
  --print-to-pdf="paper.pdf" \
  --no-pdf-header-footer \
  "file://$(pwd)/paper.html"
echo "Wrote $(pwd)/paper.pdf"
