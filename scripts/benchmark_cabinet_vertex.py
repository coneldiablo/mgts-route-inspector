"""Compare a frozen cloud second opinion to the five existing grouped v0.4 evaluations."""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from inspector.cabinet_vertex import (
    CLOUD_WEIGHT, FINGERPRINT, GENERATION, HEADS, MAX_SIDE, MODEL, OPINION_THRESHOLD, PROMPT, PROMPT_VERSION,
    CabinetArgon, CabinetArgonUnavailable,
)
from scripts.cabinet_compare import report_scores
from scripts.cabinet_final import read_dss

SEEDS = ("42", "17", "23", "71", "113")
WORKERS = 4
MAX_FRAMES = 62
STOP_STATUSES = {"adc_unavailable", "access_denied", "model_unavailable", "quota_exhausted", "vertex_error"}
PROTOCOL = {
    "model": MODEL,
    "prompt_version": PROMPT_VERSION,
    "prompt_sha256": FINGERPRINT,
    "prompt": PROMPT,
    "generation_config": GENERATION,
    "input": {"format": "metadata-free JPEG q82", "maximum_side_pixels": MAX_SIDE, "filenames_sent": False},
    "design": {
        "local_reference": "v0.4 grouped out-of-fold",
        "seeds": [int(seed) for seed in SEEDS],
        "threshold": OPINION_THRESHOLD,
        "cloud_weight_per_score": CLOUD_WEIGHT,
        "unreadable_or_failed": "standalone abstains; blend falls back to local",
        "maximum_frames": MAX_FRAMES,
        "calls_per_frame": 1,
        "concurrency": WORKERS,
        "timeouts_seconds": {"connect": 8, "read": 65},
    },
}


def write_json(path: Path, value: dict) -> None:
    staged = path.with_suffix(".tmp")
    staged.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    staged.replace(path)


def measure(y: np.ndarray, scores: np.ndarray) -> dict | None:
    if not len(y):
        return None
    metrics = report_scores(y, scores)
    confidence = {}
    for i, head in enumerate(HEADS):
        p = scores[:, i]
        correct = (p >= OPINION_THRESHOLD) == y[:, i]
        certainty = np.maximum(p, 1 - p)
        confidence[head] = {
            "average_correct": float(certainty[correct].mean()) if correct.any() else None,
            "average_incorrect": float(certainty[~correct].mean()) if (~correct).any() else None,
            "brier": float(np.mean((p - y[:, i]) ** 2)),
            "uncertain_40_60": int(np.count_nonzero((p >= .40) & (p <= .60))),
        }
    return metrics | {"confidence": confidence}


def comparisons(data: dict, responses: dict) -> dict:
    runs = {}
    for seed in SEEDS:
        rows = data[seed]
        truth = np.asarray([[row["truth"][head] for head in HEADS] for row in rows])
        local = np.asarray([
            [row["object_type_score"] if row["object_type"] == "telecom_cabinet" else 1 - row["object_type_score"],
             row["scores"]["mechanical"], row["scores"]["coating"]]
            for row in rows
        ])
        usable = np.asarray([responses.get(row["image_id"], {}).get("status") == "ok" for row in rows])
        vertex = np.asarray([
            [responses[row["image_id"]]["scores"][head] for head in HEADS]
            for row, ok in zip(rows, usable, strict=True) if ok
        ]).reshape(-1, len(HEADS))
        blended = local.copy()
        blended[usable, 1:] = (1 - CLOUD_WEIGHT) * local[usable, 1:] + CLOUD_WEIGHT * vertex[:, 1:]
        runs[seed] = {
            "labeled": len(rows),
            "vertex_usable": int(usable.sum()),
            "local_full": measure(truth, local),
            "local_on_usable_subset": measure(truth[usable], local[usable]),
            "vertex_on_usable_subset": measure(truth[usable], vertex),
            "blend_full_with_fallback": measure(truth, blended),
        }
    gains = [
        row["blend_full_with_fallback"]["defect_macro_f1"] - row["local_full"]["defect_macro_f1"]
        for row in runs.values()
    ]
    return {"runs": runs, "mean_blend_macro_f1_gain": float(np.mean(gains)),
            "positive_gain_seeds": sum(delta > 0 for delta in gains)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--local-oof", type=Path, default=Path("runs/final-cabinet/oof_by_seed.json"))
    parser.add_argument("--out", type=Path, default=Path("runs/cabinet-vertex"))
    args = parser.parse_args()
    frames = read_dss(args.source)
    if len(frames) != MAX_FRAMES:
        parser.error(f"Expected the fixed set of {MAX_FRAMES} input frames.")
    sources = {f"DSS-{i:03}": frame for i, frame in enumerate(frames, 1)}
    oof = json.loads(args.local_oof.read_text())
    expected = {key: dict(zip(HEADS, value.truth, strict=True)) for key, value in sources.items() if value.truth is not None}
    if any(
        len(oof[seed]) != len(expected) or {r["image_id"]: r["truth"] for r in oof[seed]} != expected
        for seed in SEEDS
    ):
        raise ValueError("Input frames do not match the existing out-of-fold ground truth.")

    args.out.mkdir(parents=True, exist_ok=True)
    protocol = args.out / "public-protocol.json"
    if protocol.exists() and json.loads(protocol.read_text()) != PROTOCOL:
        raise ValueError("Frozen protocol differs from the existing file.")
    write_json(protocol, PROTOCOL)
    cache = args.out / "private-responses.json"
    responses = json.loads(cache.read_text()) if cache.exists() else {}
    if any(key not in sources or row.get("prompt_sha256") != FINGERPRINT for key, row in responses.items()):
        raise ValueError("Cached requests do not match the fixed dataset and prompt.")
    cached_before = len(responses)
    pending = [key for key in sources if key not in responses]
    abort_reason = None
    started = time.monotonic()

    def request(key: str) -> dict:
        try:
            return vertex.inspect_path(sources[key].path)
        except OSError:
            return {"status": "invalid_local_image", "prompt_sha256": FINGERPRINT}

    if pending and not any(item["status"] in STOP_STATUSES for item in responses.values()):
        try:
            vertex = CabinetArgon()
        except CabinetArgonUnavailable:
            abort_reason = "adc_unavailable"
        else:
            with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                while pending and abort_reason is None:
                    batch_size = WORKERS if responses else 1
                    batch = pending[:batch_size]
                    del pending[:batch_size]
                    for key in batch:
                        responses[key] = {"status": "attempt_reserved", "prompt_sha256": FINGERPRINT}
                    write_json(cache, responses)
                    futures = {pool.submit(request, key): key for key in batch}
                    for future in as_completed(futures):
                        key = futures[future]
                        result = future.result()
                        responses[key] = result
                        write_json(cache, responses)
                        state = result["status"]
                        print(f"{len(responses) - sum(r['status'] == 'attempt_reserved' for r in responses.values())}/{MAX_FRAMES}: {state}", flush=True)
                        if state in STOP_STATUSES:
                            abort_reason = state

    counts = Counter(row["status"] for row in responses.values())
    abort_reason = abort_reason or next((status for status in sorted(STOP_STATUSES) if counts[status]), None)
    daily_usage = Counter()
    for row in responses.values():
        daily_usage.update(row.get("usage", {}))
    report_path = args.out / "public-benchmark.json"
    wall_seconds = round(time.monotonic() - started, 3)
    if len(responses) == cached_before and report_path.exists():
        previous = json.loads(report_path.read_text())
        wall_seconds = previous["wall_seconds_latest_run"]
    successful_latency = np.asarray([row["seconds"] for row in responses.values() if row["status"] == "ok"])
    output = {
        "model": MODEL, "prompt_version": PROMPT_VERSION, "prompt_sha256": FINGERPRINT,
        "planned_frames": MAX_FRAMES, "labeled_frames": len(expected),
        "attempt_statuses": dict(counts),
        "unattempted": MAX_FRAMES - len(responses),
        "abort_reason": abort_reason,
        "usage": dict(daily_usage),
        "wall_seconds_latest_run": wall_seconds,
        "latency_seconds_vertex_success": {
            "median": float(np.median(successful_latency)),
            "p95": float(np.quantile(successful_latency, .95)),
        } if len(successful_latency) else None,
        "field_serving": {
            "combined_scores": ["mechanical", "coating"],
            "object_type": "local ROI classifier",
            "combined_scores_are_advisory": True,
            "unchanged": ["local operational action", "findings", "retake requests"],
        },
        **comparisons(oof, responses),
    }
    write_json(report_path, output)
    print(json.dumps({key: value for key, value in output.items() if key != "runs"}, ensure_ascii=False))
    return 1 if abort_reason else 0


if __name__ == "__main__":
    sys.exit(main())
