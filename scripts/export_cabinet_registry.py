"""Export an allowlisted cabinet registry without source identifiers or media."""
from __future__ import annotations

import argparse
import csv
import hmac
import json
import math
import secrets
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OBJECTS = {"distribution_board": "РЩ", "telecom_cabinet": "ТКШ"}
ACTIONS = {"repair", "inspect", "no_action"}
SIGNALS = ("mechanical", "coating")


def score(value):
    checked = float(value)
    if not math.isfinite(checked) or not 0 <= checked <= 1:
        raise ValueError("Invalid model score")
    return checked


def index(rows):
    keyed = {}
    for row in rows:
        key = row["image_id"]
        if key in keyed:
            raise ValueError("Repeated source key")
        keyed[key] = row
    return keyed


def project(row):
    action = row["action"]
    if action not in ACTIONS:
        raise ValueError("Unknown action")
    if type(row["quality"]["ok"]) is not bool:
        raise ValueError("Invalid quality flag")
    quality_ok = row["quality"]["ok"]
    scores = {code: score(row["scores"][code]) for code in SIGNALS}
    findings = [code for code in SIGNALS if scores[code] >= .50]
    if set(findings) != {finding["code"] for finding in row["findings"]}:
        raise ValueError("Model score and findings disagree")
    condition = "insufficient" if not quality_ok else "damage" if findings else "normal"
    basis = "quality" if not quality_ok else action
    return {
        "object": OBJECTS[row["object_type"]], "objectScore": score(row["object_type_score"]),
        "condition": condition, "action": action, "basis": basis, "qualityOk": quality_ok,
        "scores": scores, "findings": findings if quality_ok else [],
    }


def project_argon(row, cue_labels):
    if row.get("status") == "timeout":
        return {"status": "timeout"}
    if row.get("status") != "ok" or type(row.get("readable")) is not bool:
        raise ValueError("Unexpected response status or readability")
    cues = row["cues"]
    if not isinstance(cues, list) or any(not isinstance(code, str) or code not in cue_labels for code in cues):
        raise ValueError("Unknown saved visual cue")
    scores = row["scores"]
    if any(type(scores[code]) not in (float, int) for code in SIGNALS):
        raise ValueError("Unexpected saved score")
    return {
        "status": "ok", "readable": row["readable"],
        "scores": {code: score(scores[code]) for code in SIGNALS},
        "cues": [cue_labels[code] for code in dict.fromkeys(cues)],
    }


def export(run, identity_key, argon_path=None, argon_benchmark=None):
    fitted = index(json.loads((run / "predictions.json").read_text(encoding="utf-8")))
    oof = index(json.loads((run / "oof_predictions.json").read_text(encoding="utf-8")))
    with (run / "inspection_queue.csv").open(encoding="utf-8-sig", newline="") as file:
        queue = index(csv.DictReader(file))
    report = json.loads((run / "public_metrics.json").read_text(encoding="utf-8"))
    if queue.keys() != fitted.keys() or not oof.keys() <= fitted.keys():
        raise ValueError("Run exports contain different row sets")
    if (len(fitted), len(oof)) != (report["total"], report["labeled"]):
        raise ValueError("Run exports disagree with aggregate counts")

    argon = None
    argon_config = None
    if argon_benchmark is not None and argon_path is None:
        raise ValueError("Saved responses are required with the Argon benchmark")
    if argon_path is not None:
        from inspector.cabinet_vertex import CLOUD_WEIGHT, CUE_LABELS, OPINION_THRESHOLD

        opinions = json.loads(argon_path.read_text(encoding="utf-8"))
        if opinions.keys() != fitted.keys():
            raise ValueError("Saved Argon responses cover a different set of frames")
        argon = {key: project_argon(value, CUE_LABELS) for key, value in opinions.items()}
        argon_config = {"weight": CLOUD_WEIGHT, "threshold": OPINION_THRESHOLD}
        if argon_benchmark is not None:
            benchmark = json.loads(argon_benchmark.read_text(encoding="utf-8"))
            if benchmark["labeled_frames"] != len(oof) or benchmark["attempt_statuses"] != Counter(r["status"] for r in argon.values()):
                raise ValueError("Argon benchmark disagrees with response coverage")
            blend = benchmark["runs"]["42"]["blend_full_with_fallback"]
            detected = blend["any_damage"]
            argon_config["validation"] = {
                "macroF1": score(blend["defect_macro_f1"]), "tp": int(detected["tp"]), "fp": int(detected["fp"]),
                "positives": int(detected["tp"] + detected["fn"]), "negatives": int(detected["fp"] + detected["tn"]),
            }

    # Reuse a local secret for stable aliases without publishing a reversible ordering.
    public_order = sorted(fitted, key=lambda key: hmac.digest(identity_key, key.encode(), "sha256"))
    rows = []
    for i, key in enumerate(public_order, start=1):
        prediction, queued = fitted[key], queue[key]
        if any(prediction[field] != queued[field] for field in ("action", "object_type")):
            raise ValueError("Queue diverges from the full-fit predictions")
        if any(score(prediction["scores"][code]) != score(queued[code]) for code in SIGNALS):
            raise ValueError("Queue scores diverge from the full-fit predictions")
        public_row = {
            "id": f"К-{i:03d}", "coordinates": "unconfirmed", "labeled": key in oof,
            "fitted": project(prediction), "oof": project(oof[key]) if key in oof else None,
        }
        if argon is not None:
            public_row["argon"] = argon[key]
        rows.append(public_row)
    if Counter(row["fitted"]["action"] for row in rows) != report["prediction_counts"]:
        raise ValueError("Queue does not match the published policy counts")

    metrics = report["metrics"]
    policy = report["policy_oof"]
    return {
        "version": 1, "rows": rows,
        **({"argonConfig": argon_config} if argon_config is not None else {}),
        "validation": {
            "folds": int(report["folds"]), "groups": int(report["groups"]),
            "defectMacroF1": score(metrics["defect_macro_f1"]),
            "objectAccuracy": score(metrics["telecom"]["accuracy"]),
            "repairConfirmed": int(policy["repair"]["damage"]),
            "repairTotal": sum(int(policy["repair"][key]) for key in ("damage", "normal")),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "runs" / "final-cabinet")
    parser.add_argument("--output", type=Path, default=ROOT / "registry" / "results.json")
    parser.add_argument("--identity-key", type=Path, help="Local private key file (32+ bytes) to preserve public aliases")
    parser.add_argument("--argon-responses", type=Path, help="Optional private saved response file to allowlist")
    parser.add_argument("--argon-benchmark", type=Path, help="Optional aggregate validation report")
    args = parser.parse_args()
    identity_key = args.identity_key.read_bytes() if args.identity_key else secrets.token_bytes(32)
    if len(identity_key) < 32:
        parser.error("--identity-key must contain at least 32 bytes")
    data = export(args.run, identity_key, args.argon_responses, args.argon_benchmark)
    args.output.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f'{len(data["rows"])} public rows written to {args.output}')


if __name__ == "__main__":
    main()
