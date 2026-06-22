"""Render docs/report_benedek.md to docs/report_benedek.pdf via headless Chrome.

Steps:
  1. markdown -> HTML (with tables, fenced code, attr_list)
  2. wrap in a paper-styled HTML shell
  3. headless Chrome --print-to-pdf

Run from repo root:
    poetry run python scripts/build_report_pdf.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import markdown

REPO = Path(__file__).resolve().parents[1]
SRC_MD = REPO / "docs" / "report_benedek.md"
OUT_PDF = REPO / "docs" / "report_benedek.pdf"
TMP_HTML = REPO / "docs" / "_report_benedek.tmp.html"

CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    shutil.which("google-chrome") or "",
    shutil.which("chromium") or "",
]

CSS = """
@page { size: A4; margin: 18mm 18mm 20mm 18mm; }
html { font-family: "Charter", "Georgia", "Times New Roman", serif; font-size: 10.5pt; color: #111; }
body { line-height: 1.42; max-width: 100%; }
h1 { font-size: 18pt; margin: 0 0 4pt; }
h2 { font-size: 13pt; margin: 18pt 0 6pt; border-bottom: 1px solid #888; padding-bottom: 2pt; }
h3 { font-size: 11.5pt; margin: 12pt 0 4pt; }
h4 { font-size: 10.5pt; margin: 10pt 0 4pt; }
p { margin: 6pt 0; text-align: justify; }
ul, ol { margin: 6pt 0 6pt 18pt; }
li { margin: 2pt 0; }
code, pre { font-family: "Menlo", "Consolas", monospace; font-size: 9pt; }
pre { background: #f5f5f5; padding: 6pt 8pt; border-radius: 3pt; overflow-x: auto; }
code { background: #f5f5f5; padding: 1px 3px; border-radius: 2pt; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; margin: 8pt 0; font-size: 9.5pt; width: 100%; }
th, td { border: 1px solid #888; padding: 3pt 5pt; text-align: left; vertical-align: top; }
th { background: #eaeaea; font-weight: 600; }
img { max-width: 100%; height: auto; display: block; margin: 6pt auto; }
hr { border: 0; border-top: 1px solid #aaa; margin: 12pt 0; }
strong { font-weight: 600; }
em { font-style: italic; }
a { color: #1565c0; text-decoration: none; }
blockquote { border-left: 3px solid #ccc; margin: 6pt 0; padding-left: 10pt; color: #444; }
h2, h3 { page-break-after: avoid; }
table, img, pre { page-break-inside: avoid; }
"""


def find_chrome() -> str:
    for path in CHROME_CANDIDATES:
        if path and Path(path).exists():
            return path
    sys.exit("ERROR: no Chrome/Chromium/Edge binary found. Install Chrome or "
             "install pandoc + LaTeX as an alternative.")


def build_html(md_text: str) -> str:
    body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "attr_list", "sane_lists", "toc"],
        output_format="html5",
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>report_benedek</title>
<style>{CSS}</style>
</head>
<body>
{body}
</body>
</html>"""


def main() -> None:
    md_text = SRC_MD.read_text(encoding="utf-8")
    TMP_HTML.write_text(build_html(md_text), encoding="utf-8")
    chrome = find_chrome()
    url = f"file://{TMP_HTML.resolve()}"
    cmd = [
        chrome,
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={OUT_PDF.resolve()}",
        "--virtual-time-budget=4000",
        url,
    ]
    print("running:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stdout + result.stderr)
        sys.exit(result.returncode)
    TMP_HTML.unlink(missing_ok=True)
    size_kb = OUT_PDF.stat().st_size / 1024
    print(f"wrote {OUT_PDF} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
