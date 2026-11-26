"""Compare Gemini via Vertex AI ADC on public, boxed manhole test images."""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from inspector.classifier import defect_type_metrics
from inspector.detector import box_iou
from inspector.manifest import load_manifest
from inspector.quality import crop_bbox, open_image
from inspector.vertex_vision import VertexVision, decode_response as decode_response, request_body as request_body


def classify_box(vision: VertexVision, item, *, full_frame=False):
    cohort, record = item
    start = time.monotonic()
    orig = open_image(record.image_path)
    image = orig if full_frame else crop_bbox(orig, record.bbox, margin=0.08)
    result = vision.inspect(image, full_frame=full_frame)
    bbox = result.pop("bbox", None)
    if bbox is not None:
        result["box_iou"] = round(box_iou(bbox, record.bbox), 4)
    if result["status"] == 0:
        result["error"] = "network_error"
    row = {"cohort": cohort, "image_id": record.image_id, "label": record.label,
           "full_frame": full_frame}
    return {**row, **result, "seconds": round(time.monotonic() - start, 3)}


def summarize(rows):
    accepted = [int(row["prediction"] != "uncertain") for row in rows]
    return {
        **defect_type_metrics([r["label"] for r in rows], [r["prediction"] for r in rows], accepted, 0.5),
        "false_alerts": sum(r["prediction"] == "open_hatch" and r["label"] == "normal" for r in rows),
        "api_errors": sum("error" in r for r in rows),
        "correct_with_iou50": (sum(r.get("box_iou", 0) >= 0.5 and r["prediction"] == r["label"] for r in rows)
                               if rows and rows[0].get("full_frame") else None),
        "total_reported_tokens": sum(r.get("usage", {}).get("totalTokenCount", 0) for r in rows),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Vertex AI через ADC на кропах публичных фото")
    parser.add_argument("--data", action="append", required=True, metavar="NAME=DIR")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--model", default="gemini-3.5-flash")
    parser.add_argument("--project", help="Иначе используется проект ADC")
    parser.add_argument("--limit", type=int, default=12, help="Максимум платных вызовов по всем наборам")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--full-frame", action="store_true", help="запрашивать рамку и класс по всему публичному снимку")
    parser.add_argument("--retry-from", type=Path, help="повторить только ошибки сети, 429 и 5xx из указанного JSONL")
    parser.add_argument("--select", action="append", default=[], metavar="COHORT=IMAGE_ID")
    args = parser.parse_args(argv)
    if args.limit < 1 or not 1 <= args.workers <= 8:
        parser.error("Нужны положительный лимит и от одного до восьми параллельных запросов.")
    items = []
    used_names = set()
    for data in args.data:
        cohort, sep, folder = data.partition("=")
        if not sep or not cohort or not folder or cohort in used_names:
            parser.error("Каждая выборка задаётся один раз в формате NAME=DIR.")
        used_names.add(cohort)
        for record in load_manifest(folder):
            if record.label not in {"normal", "open_hatch"} or not record.bbox:
                parser.error(f"Для {record.image_id} нужны рамка и метка normal/open_hatch.")
            items.append((cohort, record))
    known_keys = {(name, r.image_id) for name, r in items}
    chosen = set()
    for selection in args.select:
        name, sep, image_id = selection.partition("=")
        if not sep or (name, image_id) not in known_keys:
            parser.error(f"Кадр {selection} отсутствует в указанных выборках.")
        chosen.add((name, image_id))
    if args.retry_from:
        with args.retry_from.open(encoding="utf-8") as receipt:
            failed = [json.loads(line) for line in receipt if line.strip()]
        chosen.update((r["cohort"], r["image_id"]) for r in failed if r["status"] in {0, 429} or r["status"] >= 500)
    if args.select or args.retry_from:
        items = [(name, r) for name, r in items if (name, r.image_id) in chosen]
    items = items[:args.limit]
    if not items:
        parser.error("Отсутствуют выбранные кадры для запроса.")
    try:
        vision = VertexVision(model=args.model, project=args.project)
    except ValueError as exc:
        parser.error(str(exc))
    args.out.mkdir(parents=True, exist_ok=True)

    def run(item):
        return classify_box(vision, item, full_frame=args.full_frame)

    with (args.out / "predictions.jsonl").open("x", encoding="utf-8") as output:
        first = run(items[0])
        rows = [first]
        print(json.dumps(first, ensure_ascii=False), file=output, flush=True)
        if first["status"] in {400, 401, 403, 404, 429}:
            print(f"Vertex HTTP {first['status']} {first.get('error', '')}")
            return 2
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for row in executor.map(run, items[1:]):
                rows.append(row)
                print(json.dumps(row, ensure_ascii=False), file=output, flush=True)
    cohorts = list(dict.fromkeys(r["cohort"] for r in rows))
    summary = {"model": args.model, "image_input": "full_frame" if args.full_frame else "provided_bbox", "all": summarize(rows),
               "cohorts": {name: summarize([r for r in rows if r["cohort"] == name]) for name in cohorts}}
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
