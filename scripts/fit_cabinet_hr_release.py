"""Join the evaluated HR defect blend and ROI type head into one local release."""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from scipy.special import expit

from inspector.cabinet import (
    CABINET_RELEASE, DEFAULT_CABINET_MODEL, HEADS, ROI_RELEASE, blend_defect_scores, cabinet_result, head_probabilities,
)
from inspector.cabinet_hr import DIM, REVISION, SHA256
from inspector.quality import Quality
from scripts.cabinet_compare import SEEDS, report_scores
from scripts.cabinet_release import CSV_FIELDS


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def recompute_row(roi_row, control_score, hr_score):
    mixed = blend_defect_scores(control_score, hr_score)
    decision = cabinet_result(mixed, Quality(**roi_row["quality"]))
    return roi_row | {key: decision[key] for key in ("scores", "findings", "action", "reason")} | {
        "model_release": CABINET_RELEASE, "defect_score_source": "siglip_hr_blend",
    }


def score_vector(score_map):
    return [score_map[head] for head in HEADS]


def scores_for_type(rows):
    return np.array([
        row["object_type_score"] if row["object_type"] == "telecom_cabinet" else 1 - row["object_type_score"]
        for row in rows
    ])


def type_spec(path: Path, expected_rows: list[dict]):
    bundle = joblib.load(path)
    if bundle.get("type_roi") is None or bundle.get("defect_hr") is not None:
        raise ValueError("Для объединённого релиза нужна базовая модель v0.3-ROI.")
    if any(row.get("model_release") != ROI_RELEASE for row in expected_rows):
        raise ValueError("Входной отчёт должен относиться к v0.3-ROI.")
    return bundle


def same_images(a, b):
    return [row["image_id"] for row in a] == [row["image_id"] for row in b]


def publish(out: Path, report, rows, oof, comparison, by_seed) -> None:
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "predictions.json", rows)
    write_json(out / "oof_predictions.json", oof)
    write_json(out / "oof_by_seed.json", by_seed)
    write_json(out / "comparison.json", comparison)
    fields = (*CSV_FIELDS, "defect_score_source", "model_release")
    temporary = out / "inspection_queue.csv.tmp"
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row["scores"].get(field, row.get(field)) for field in fields})
    os.replace(temporary, out / "inspection_queue.csv")
    write_json(out / "public_metrics.json", report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-features", required=True, type=Path)
    parser.add_argument("--roi-release", default=Path("runs/final-cabinet-v0.3-release"), type=Path)
    parser.add_argument("--roi-paired", default=Path("runs/final-cabinet-v0.3-roi"), type=Path)
    parser.add_argument("--roi-model", default=Path("artifacts/final-cabinet-v0.3-roi/model.joblib"), type=Path)
    parser.add_argument("--control-oof", default=Path("runs/final-cabinet-experiment/oof_comparison.json"), type=Path)
    parser.add_argument("--hr", default=Path("runs/cabinet-hr-siglip"), type=Path)
    parser.add_argument("--out", default=Path("runs/final-cabinet-v0.4-hr-roi"), type=Path)
    parser.add_argument("--artifact", default=Path("artifacts/final-cabinet-v0.4-hr-roi"), type=Path)
    parser.add_argument("--activate", action="store_true")
    args = parser.parse_args()

    release_rows = read(args.roi_release / "predictions.json")
    report = read(args.roi_release / "public_metrics.json")
    if report["model"]["release"] != ROI_RELEASE:
        raise ValueError("Входные агрегаты должны относиться к v0.3-ROI.")
    bundle = type_spec(args.roi_model, release_rows)
    with np.load(args.hr / "heads.npz", allow_pickle=False) as heads:
        weights, bias = heads["weights"], heads["bias"]
    if weights.shape != (len(HEADS), DIM) or bias.shape != (len(HEADS),):
        raise ValueError("Некорректные размеры весов SigLIP 2 HR.")
    if not (np.isfinite(weights).all() and np.isfinite(bias).all()):
        raise ValueError("Весовые коэффициенты HR должны быть конечными.")
    protocol = read(args.hr / "protocol.json")
    if protocol["revision"] != REVISION:
        raise ValueError("Протокол HR использует другую версию энкодера.")
    bundle["defect_hr"] = {
        "weights": weights[1:].copy(), "bias": bias[1:].copy(),
        "encoder_revision": REVISION, "encoder_sha256": SHA256, "validation_device": protocol["device"],
    }
    bundle["release"] = CABINET_RELEASE

    hr_rows = read(args.hr / "full_fit_scores.json")
    if not same_images(release_rows, hr_rows):
        raise ValueError("Снимки полного прогона HR не совпадают с v0.3-ROI.")
    with np.load(args.hr / "private-features.npz", allow_pickle=False) as cached_hr:
        compiled = expit(cached_hr["x"] @ weights.T + bias)
    np.testing.assert_allclose(compiled, [score_vector(row["scores"]) for row in hr_rows], rtol=0, atol=1e-10)
    with np.load(args.reference_features, allow_pickle=False) as cached:
        features = {name: cached[name] for specs in bundle["heads"].values() for name in
                    (spec["variant"] for spec in specs)}
        control_full = {head: head_probabilities(specs, features) for head, specs in bundle["heads"].items()}
    for i, row in enumerate(release_rows):
        for head in HEADS[1:]:
            if round(float(control_full[head][i]), 5) != row["scores"][head]:
                raise ValueError("Кэш полного прогона не воспроизводит дефекты v0.3.")
    rows = [
        recompute_row(row, {head: float(control_full[head][i]) for head in HEADS}, hr_rows[i]["scores"])
        for i, row in enumerate(release_rows)
    ]

    controls = read(args.control_oof)
    roi_oof = read(args.roi_paired / "oof_by_seed.json")
    hr_oof_primary = read(args.hr / "oof_primary.json")
    benchmark = read(args.hr / "benchmark.json")
    by_seed, audits = {}, {}
    with np.load(args.hr / "oof.npz", allow_pickle=False) as hr_oof:
        for seed in SEEDS:
            key = str(seed)
            raw_control = controls[key]
            named_roi = roi_oof[key]
            if not same_images(raw_control, named_roi) or any(
                rr[key] != cr[key] for rr, cr in zip(named_roi, raw_control, strict=True)
                for key in ("fold", "group", "truth")
            ):
                raise ValueError("Нарушено парное сравнение контрольных фолдов.")
            truth = np.asarray([score_vector(row["truth"]) for row in raw_control])
            scores = np.asarray([score_vector(row["scores"]["control"]) for row in raw_control])
            raw_hr = hr_oof[f"{seed}_hr"]
            combined = (scores + raw_hr) / 2
            np.testing.assert_allclose(combined, hr_oof[f"{seed}_blend"], rtol=0, atol=1e-12)
            if seed == 42:
                if not same_images(raw_control, hr_oof_primary):
                    raise ValueError("Идентификаторы основного фолда HR не совпадают.")
                for a, b in zip(raw_control, hr_oof_primary, strict=True):
                    if any(a[key] != b[key] for key in ("fold", "group", "truth")):
                        raise ValueError("Разметка или группы HR расходятся с контролем.")
            refined = combined.copy()
            refined[:, 0] = scores_for_type(named_roi)
            base = scores.copy()
            base[:, 0] = refined[:, 0]
            metrics, control_metrics = report_scores(truth, refined), report_scores(truth, base)
            if not np.isclose(metrics["defect_macro_f1"], benchmark["seeds"][key]["v02_hr"]["defect_macro_f1"]):
                raise RuntimeError("Проверка не воспроизвела benchmark HR.")
            candidates = [
                recompute_row(row, raw_control[i]["scores"]["control"],
                              dict(zip(HEADS, raw_hr[i].tolist(), strict=True)))
                for i, row in enumerate(named_roi)
            ]
            actions = Counter((row["action"], bool(truth[i, 1:].any())) for i, row in enumerate(candidates))
            audits[key] = {
                "control": control_metrics, "candidate": metrics,
                "policy_oof": {action: {"damage": actions[action, True], "normal": actions[action, False]}
                               for action in ("repair", "inspect", "no_action")},
            }
            by_seed[key] = candidates
    primary = audits["42"]
    if any(audit["candidate"]["defect_macro_f1"] <= audit["control"]["defect_macro_f1"] for audit in audits.values()):
        raise ValueError("Ансамбль HR не улучшил дефекты на каждом предзаданном разбиении.")
    score = primary["candidate"]
    report["model"].update(
        name="SigLIP 2 HR × DINOv2 · тип по области шкафа",
        release=CABINET_RELEASE,
        validation="5 внешних групповых OOF-фолдов; фиксированное смешивание дефектов 50/50; ROI только для типа",
        macroF1=score["defect_macro_f1"], fallback_release=ROI_RELEASE,
        defect_classifier={"method": "siglip_hr_blend", "blend": [0.5, 0.5], "input_side": 384,
                           "validation_device": protocol["device"], "encoder_revision": REVISION,
                           "encoder_sha256": SHA256},
    )
    report["model"]["signalMetrics"] = [row | score[row["code"]] for row in report["model"]["signalMetrics"]]
    report["metrics"] = score
    report["policy_oof"] = primary["policy_oof"]
    if "seconds" in report:
        report["control_training_seconds"] = report.pop("seconds")
    report["prediction_counts"] = dict(Counter(row["action"] for row in rows))
    report["defect_source_counts"] = dict(Counter(row["defect_score_source"] for row in rows))
    report["sensitivity"] = {
        "seeds": list(SEEDS), "wins": len(SEEDS), "mean_control_macroF1": float(np.mean([
            audits[str(i)]["control"]["defect_macro_f1"] for i in SEEDS
        ])), "mean_macroF1": float(np.mean([
            audits[str(i)]["candidate"]["defect_macro_f1"] for i in SEEDS
        ])),
    }
    args.artifact.mkdir(parents=True, exist_ok=True)
    candidate = args.artifact / "model.joblib"
    temporary = candidate.with_name(candidate.name + ".tmp")
    joblib.dump(bundle, temporary)
    os.replace(temporary, candidate)
    publish(args.out, report, rows, by_seed["42"], audits, by_seed)
    if args.activate:
        publish(Path("runs/final-cabinet"), report, rows, by_seed["42"], audits, by_seed)
        target = DEFAULT_CABINET_MODEL
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        shutil.copyfile(candidate, temporary)
        os.replace(temporary, target)
    print(f"{CABINET_RELEASE} · macro-F1 дефектов {score['defect_macro_f1']:.2%}; "
          f"активирована: {'да' if args.activate else 'нет'}", flush=True)


if __name__ == "__main__":
    main()
