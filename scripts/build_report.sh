#!/usr/bin/env bash
# Build a PDF from the Markdown report via markdown-it (+KaTeX) and headless Chrome.
# No LaTeX needed. Usage: scripts/build_report.sh [path/to/report.md]
#   CHROME_BIN          override the Chrome/Chromium binary (default: google-chrome)
#   REPORT_BUILD_CACHE  where to keep the npm deps (default: ~/.cache/report-build)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPORT="${1:-$ROOT/reports/Kristof_Nemeth_report.md}"
PDF="${REPORT%.md}.pdf"
HTML="$(dirname "$REPORT")/.report_build.html"
DEPS="${REPORT_BUILD_CACHE:-$HOME/.cache/report-build}"
CHROME="${CHROME_BIN:-google-chrome}"

# Install the two render deps once into a cache dir (keeps the repo clean).
mkdir -p "$DEPS"
if [ ! -d "$DEPS/node_modules/katex" ]; then
  echo "Installing render deps into $DEPS ..."
  ( cd "$DEPS" && npm init -y >/dev/null 2>&1 && \
    npm install --no-audit --no-fund markdown-it markdown-it-katex katex >/dev/null 2>&1 )
fi

REPORT_BUILD_DEPS="$DEPS" node "$ROOT/scripts/render_report.mjs" "$REPORT" "$HTML"

# --virtual-time-budget lets the CDN KaTeX stylesheet/fonts load before printing,
# so display math is styled correctly in the PDF.
"$CHROME" --headless=new --no-sandbox --disable-gpu --no-pdf-header-footer \
  --virtual-time-budget=20000 --run-all-compositor-stages-before-draw \
  --print-to-pdf="$PDF" "file://$HTML" >/dev/null 2>&1 || true

rm -f "$HTML"
echo "Wrote $PDF"
