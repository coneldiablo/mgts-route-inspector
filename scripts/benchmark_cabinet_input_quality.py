"""Measure the actual browser gate on originals and deterministic local corruptions."""
from __future__ import annotations

import argparse
import io
import json
import mimetypes
import shutil
import subprocess
import tempfile
from base64 import b64encode
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from scripts.cabinet_final import read_dss

ROOT = Path(__file__).resolve().parents[1]
NODE = r"""
const path = require('node:path');
const fs = require('node:fs');
const http = require('node:http');
const { chromium } = require(process.argv[2]);
const base = process.argv[3];
const gate = process.argv[4];
const report = process.argv[5];
const payload = JSON.parse(fs.readFileSync(path.join(base, 'images.json'), 'utf8'));

(async () => {
  const server = http.createServer((req, res) => {
    if (req.url === '/input-quality.js') {
      res.setHeader('Content-Type', 'application/javascript; charset=utf-8');
      res.end(fs.readFileSync(gate));
      return;
    }
    res.setHeader('Content-Type', 'text/html; charset=utf-8');
    res.end('<!doctype html><title>Local input quality benchmark</title>');
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  let browser;
  try {
    browser = await chromium.launch({ headless: true });
    const page = await browser.newPage({ viewport: { width: 390, height: 844 } });
    const url = `http://127.0.0.1:${server.address().port}`;
    await page.route('**/*', route => route.request().url().startsWith(`${url}/`) ? route.continue() : route.abort());
    await page.goto(url);
    const totals = await page.evaluate(async () => ({
      policy: (await import('/input-quality.js')).INPUT_QUALITY_POLICY,
      cases: {},
    }));
    for (const record of payload) {
      const response = await page.evaluate(async ({ data, mime }) => {
        const { assessInputQuality } = await import('/input-quality.js');
        const bytes = Uint8Array.from(atob(data), c => c.charCodeAt(0));
        const bitmap = await createImageBitmap(new Blob([bytes], { type: mime }));
        try {
          return assessInputQuality(bitmap);
        } finally {
          bitmap.close();
        }
      }, { data: record.png || fs.readFileSync(record.source).toString('base64'), mime: record.mime });
      const tally = totals.cases[record.variant] ||= { total: 0, passed: 0, rejected: 0, rejection_reasons: {} };
      tally.total++;
      if (response.ok) {
        tally.passed++;
      } else {
        if (!response.reason || !response.request) throw Error('Rejection requires a reason and reshoot request');
        tally.rejected++;
        tally.rejection_reasons[response.reason] = (tally.rejection_reasons[response.reason] || 0) + 1;
      }
    }
    for (const tally of Object.values(totals.cases)) tally.rejected_fraction = tally.rejected / tally.total;
    fs.writeFileSync(report, `${JSON.stringify({ browser: `Chromium ${browser.version()}`, ...totals }, null, 2)}\n`);
  } finally {
    if (browser) await browser.close();
    server.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
"""

VARIANTS = {
    "original": "Raw supplied photo, decoded by the browser with EXIF orientation.",
    "blur_moderate": "Gaussian blur with a 3 px radius at the gate's 224 px analysis scale.",
    "blur_severe": "Gaussian blur with a 14 px radius at the gate's 224 px analysis scale.",
    "dark_severe": "RGB intensity multiplied by 0.08.",
    "clipped_severe": "RGB remapped to [250, 255], clipping highlights and removing contrast.",
    "tiny": "Longest side 112 px.",
    "flat": "Uniform fill with the mean color of the original.",
}


def reduced(image: Image.Image, pixels: int) -> Image.Image:
    preview = image.copy()
    preview.thumbnail((pixels, pixels), Image.Resampling.LANCZOS)
    return preview


def variants(image: Image.Image):
    working = reduced(image, 512)
    mean_color = tuple(np.asarray(working).mean(axis=(0, 1)).round().astype(int))
    return {
        "blur_moderate": working.filter(ImageFilter.GaussianBlur(3 * max(working.size) / 224)),
        "blur_severe": working.filter(ImageFilter.GaussianBlur(14 * max(working.size) / 224)),
        "dark_severe": ImageEnhance.Brightness(working).enhance(.08),
        "clipped_severe": working.point(lambda channel: 250 + round(channel * 5 / 255)),
        "tiny": reduced(image, 112),
        "flat": Image.new("RGB", working.size, mean_color),
    }


def payload(sources: list[Path]) -> list[dict]:
    cases = []
    for frame in read_dss(sources):
        if frame.truth is None:
            continue
        cases.append({"variant": "original", "source": str(frame.path.resolve()),
                      "mime": mimetypes.guess_type(frame.path)[0]})
        with Image.open(frame.path) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
        for name, transformed in variants(image).items():
            stream = io.BytesIO()
            transformed.save(stream, format="PNG", compress_level=1)
            cases.append({"variant": name, "png": b64encode(stream.getvalue()).decode("ascii"), "mime": "image/png"})
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, nargs="+", required=True)
    parser.add_argument("--playwright", required=True, help="Installed Playwright Node module path or package name")
    parser.add_argument("--output", type=Path, default=ROOT / "browser-ml" / "input-quality-report.json")
    args = parser.parse_args()
    if not shutil.which("node"):
        parser.error("Node.js is required to run Playwright")

    with tempfile.TemporaryDirectory(prefix="cabinet-quality-") as folder:
        temp = Path(folder)
        (temp / "images.json").write_text(json.dumps(payload(args.sources)), encoding="utf-8")
        (temp / "browser.cjs").write_text(NODE, encoding="utf-8")
        (temp / "report.json").touch()
        subprocess.run(["node", str(temp / "browser.cjs"), args.playwright, str(temp),
                        str(ROOT / "browser-ml" / "input-quality.js"), str(temp / "report.json")], check=True)
        report = json.loads((temp / "report.json").read_text(encoding="utf-8"))
    report.update({
        "version": 1,
        "method": "Actual JavaScript gate in headless Chromium, with a 390 px mobile viewport and external requests blocked.",
        "sample": "All labeled DSS frames as supplied, plus one deterministic transform of each per distortion.",
        "variants": VARIANTS,
        "limits": [
            "Corrupted cases are paired with the original frames. They are controlled diagnostic cases, not measured field rejection rates.",
            "The gate tests readability and has no semantics or object-location signal.",
        ],
    })
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report["cases"], indent=2))


if __name__ == "__main__":
    main()
