#!/usr/bin/env python3
"""
Export docs/SDD.md → docs/SDD.pdf and docs/SDD.docx (Windows, no extra installs)
==============================================================================
  1. Build an HTML page that renders the Markdown (marked) and Mermaid diagrams
     client-side, then converts each diagram to a PNG image.
  2. Headless Chrome prints it to PDF and dumps the rendered page.
  3. Microsoft Word opens the rendered page and saves it as DOCX, embedding images.

Usage (from the project root, in Windows):
  python scripts/export_sdd.py
"""

import base64
import html
import json
import os
import re
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "docs", "SDD.md")
BUILD = os.path.join(ROOT, "docs", "build")
PDF = os.path.join(ROOT, "docs", "SDD.pdf")
DOCX = os.path.join(ROOT, "docs", "SDD.docx")

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
]

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>{title}</title>
<style>
  @page {{ size: A4; margin: 18mm 16mm; }}
  body {{ font-family: "Segoe UI", Calibri, Arial, sans-serif; font-size: 10.5pt; line-height: 1.45;
         color: #111; background: #fff; max-width: 180mm; margin: 0 auto; }}
  h1 {{ font-size: 18pt; margin: 20pt 0 6pt; page-break-before: always; border-bottom: 1px solid #ccc; padding-bottom: 3pt; }}
  h1:first-of-type {{ page-break-before: avoid; font-size: 22pt; border: 0; }}
  h2 {{ font-size: 14pt; margin: 16pt 0 4pt; }}
  h3 {{ font-size: 11.5pt; margin: 12pt 0 4pt; }}
  table {{ border-collapse: collapse; width: 100%; margin: 6pt 0 10pt; font-size: 9.5pt; page-break-inside: avoid; }}
  th, td {{ border: 1px solid #bbb; padding: 3pt 5pt; text-align: left; vertical-align: top; }}
  th {{ background: #eef1f5; }}
  code {{ font-family: Consolas, "Courier New", monospace; font-size: 9pt; background: #f4f4f4; padding: 0 2pt; }}
  pre {{ background: #f6f6f6; border: 1px solid #ddd; padding: 6pt; font-size: 8.5pt; overflow-wrap: anywhere;
        white-space: pre-wrap; page-break-inside: avoid; }}
  pre code {{ background: none; padding: 0; }}
  .diagram {{ text-align: center; margin: 8pt 0 12pt; page-break-inside: avoid; }}
  .diagram img {{ max-width: 100%; max-height: 210mm; }}
  hr {{ border: 0; }}
</style>
<script src="https://cdn.jsdelivr.net/npm/marked@12.0.2/marked.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/mermaid@11.4.1/dist/mermaid.min.js"></script>
</head><body>
<main id="doc"></main>
<script id="source" type="application/json">{source}</script>
<script>
(async () => {{
  const md = JSON.parse(document.getElementById("source").textContent);
  document.getElementById("doc").innerHTML = marked.parse(md);
  // htmlLabels:false keeps diagrams pure SVG so they can be rasterised for Word
  mermaid.initialize({{ startOnLoad: false, theme: "neutral", securityLevel: "strict",
                        flowchart: {{ htmlLabels: false }}, class: {{ htmlLabels: false }} }});
  const blocks = [...document.querySelectorAll("pre > code.language-mermaid")];
  const errors = [];
  for (let i = 0; i < blocks.length; i++) {{
   try {{
    const pre = blocks[i].parentElement;
    const {{ svg }} = await mermaid.render("d" + i, blocks[i].textContent);
    const holder = document.createElement("div");
    holder.className = "diagram";
    holder.innerHTML = svg;
    pre.replaceWith(holder);
    const el = holder.querySelector("svg");
    const box = el.getBoundingClientRect();
    const scale = 2;
    const w = Math.ceil(box.width), h = Math.ceil(box.height);
    el.setAttribute("width", w); el.setAttribute("height", h);
    const img = new Image();
    img.src = "data:image/svg+xml;base64," + btoa(unescape(encodeURIComponent(new XMLSerializer().serializeToString(el))));
    await img.decode();
    const canvas = document.createElement("canvas");
    canvas.width = w * scale; canvas.height = h * scale;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    const png = document.createElement("img");
    png.src = canvas.toDataURL("image/png");
    png.width = w; png.height = h;
    png.alt = "Diagram " + (i + 1);
    holder.replaceChildren(png);
   }} catch (e) {{
    errors.push("diagram " + (i + 1) + ": " + (e && e.message ? e.message : e));
   }}
  }}
  document.querySelectorAll("script").forEach(s => s.remove());
  document.body.setAttribute("data-errors", errors.join(" | "));
  document.body.setAttribute("data-ready", "1");
}})();
</script>
</body></html>
"""

WORD_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
$word = New-Object -ComObject Word.Application
$word.Visible = $false
try {
  $doc = $word.Documents.Open('{html}', $false, $true)
  foreach ($shape in @($doc.InlineShapes)) {
    if ($shape.LinkFormat) { $shape.LinkFormat.SavePictureWithDocument = $true; $shape.LinkFormat.BreakLink() }
  }
  $doc.SaveAs2('{docx}', 16)
  $doc.Close($false)
} finally { $word.Quit() }
"""


def find_chrome() -> str:
    for path in CHROME_CANDIDATES:
        if os.path.exists(path):
            return path
    sys.exit("Chrome or Edge not found")


def build_page() -> str:
    with open(SRC, encoding="utf-8") as f:
        md = f.read()
    md = re.sub(r"\A---\n.*?\n---\n", "", md, flags=re.S)  # drop YAML front matter
    title = "OS-Level Guardrails — Software Design Description"
    source = json.dumps(md).replace("</", "<\\/")
    os.makedirs(BUILD, exist_ok=True)
    page = os.path.join(BUILD, "SDD.html")
    with open(page, "w", encoding="utf-8") as f:
        f.write(PAGE.format(title=html.escape(title), source=source))
    return page


def chrome(chrome_exe: str, *args) -> subprocess.CompletedProcess:
    return subprocess.run([chrome_exe, "--headless=new", "--disable-gpu", "--no-first-run",
                           "--virtual-time-budget=30000", "--run-all-compositor-stages-before-draw", *args],
                          capture_output=True, text=True, encoding="utf-8", timeout=180)


def extract_images(static_html: str) -> str:
    """Write data-URI images to files so Word can import and embed them."""
    img_dir = os.path.join(BUILD, "img")
    os.makedirs(img_dir, exist_ok=True)
    counter = iter(range(1, 10_000))

    def save(match):
        n = next(counter)
        name = f"diagram-{n:02d}.png"
        with open(os.path.join(img_dir, name), "wb") as f:
            f.write(base64.b64decode(match.group(1)))
        return f'src="img/{name}"'

    return re.sub(r'src="data:image/png;base64,([^"]+)"', save, static_html)


def main():
    chrome_exe = find_chrome()
    page = build_page()
    url = "file:///" + page.replace("\\", "/")

    result = chrome(chrome_exe, "--no-pdf-header-footer", f"--print-to-pdf={PDF}", url)
    if not os.path.exists(PDF):
        sys.exit(f"PDF export failed:\n{result.stderr[-2000:]}")
    print(f"PDF:  {PDF}")

    dom = chrome(chrome_exe, "--dump-dom", url).stdout
    if 'data-ready="1"' not in dom:
        sys.exit("Rendering did not finish (network access to cdn.jsdelivr.net is needed for marked/mermaid)")
    static = os.path.join(BUILD, "SDD.static.html")
    with open(static, "w", encoding="utf-8") as f:
        f.write("<!doctype html>\n" + extract_images(dom))

    script = WORD_SCRIPT.replace("{html}", static).replace("{docx}", DOCX)
    word = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                          capture_output=True, text=True, timeout=300)
    if word.returncode != 0 or not os.path.exists(DOCX):
        sys.exit(f"DOCX export failed:\n{word.stderr[-2000:]}")
    print(f"DOCX: {DOCX}")


if __name__ == "__main__":
    main()
