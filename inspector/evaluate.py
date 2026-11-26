"""Evaluation at the level that matters: a unique object, not a frame."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .classifier import asset_threshold_table, classification_image, defect_type_metrics, load_model, select_threshold, validate_artifact_task
from .config import QualityConfig, quality_config
from .manifest import data_fingerprint, image_sha256, load_manifest
from .quality import assess_quality, open_image


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def _asset_metrics(oof: pd.DataFrame, threshold: float, k: int = 10) -> dict[str, Any]:
    oof = oof.copy()
    oof["pred"] = oof["p_finding_oof"] >= threshold
    oof["positive"] = oof["y_binary"] == 1
    agg = oof.groupby("asset_id", dropna=False).agg(
        positive=("positive", "max"),
        predicted=("pred", "max"),
        max_p=("p_finding_oof", "max"),
    )
    tp = int(((agg.positive) & (agg.predicted)).sum())
    fp = int((~agg.positive & agg.predicted).sum())
    fn = int((agg.positive & ~agg.predicted).sum())
    tn = int((~agg.positive & ~agg.predicted).sum())
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    positive_assets = int(agg.positive.sum())
    top = agg.sort_values(["max_p", "asset_id"], ascending=[False, True]).head(k)
    top_precision = _safe_div(int(top.positive.sum()), len(top)) if len(top) else 0.0
    prevalence = _safe_div(positive_assets, len(agg))
    return {
        "tp_assets": tp,
        "fp_assets": fp,
        "fn_assets": fn,
        "tn_assets": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_cards_per_100_assets": round(_safe_div(fp, len(agg)) * 100, 2),
        "positive_assets": positive_assets,
        "n_assets": int(len(agg)),
        "prevalence": round(prevalence, 4),
        f"top{k}_precision": round(top_precision, 4),
        f"top{k}_lift": round(_safe_div(top_precision, prevalence), 2) if prevalence else 0.0,
        "top_k": int(k),
    }


def _condition_breakdown(oof: pd.DataFrame, threshold: float) -> list[dict[str, Any]]:
    rows = []
    for condition, part in oof.groupby(oof["condition"].fillna("unknown")):
        y = part["y_binary"].to_numpy()
        pred = part["p_finding_oof"].to_numpy() >= threshold
        pos = y == 1
        neg = ~pos
        rows.append(
            {
                "condition": condition,
                "n_images": int(len(part)),
                "n_positive": int(pos.sum()),
                "recall": round(_safe_div(float((pred & pos).sum()), float(pos.sum())), 4),
                "false_positive_rate": round(_safe_div(float((pred & neg).sum()), float(neg.sum())), 4),
            }
        )
    return sorted(rows, key=lambda r: r["condition"])


def _type_report(meta, threshold, labels, predictions, scores, refused_labels, evaluation):
    refused_labels = list(refused_labels)
    return {
        "mode": "defect_type", "backbone": meta.get("backbone"), "task": meta.get("task"),
        "threshold": threshold, "evaluation": evaluation, "n_quality_refused": len(refused_labels),
        **defect_type_metrics([*labels, *refused_labels], [*predictions, *([""] * len(refused_labels))],
                              [*scores, *([float("-inf")] * len(refused_labels))], threshold),
    }


def _write_type_report(artifact_dir: Path, stem: str, result: dict[str, Any]) -> None:
    (artifact_dir / f"{stem}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Классификация видов подтверждённых повреждений", "",
        f"Оценка — {result['evaluation']}. Порог — {result['threshold']:.2f}.", "",
        f"Macro-F1 с учётом отказов — {_fmt_share(result['type_macro_f1'], 1)}.",
        f"Вывод по {result['n_accepted']} из {result['n_images']} кадров. "
        f"Доля правильных выводов — {_fmt_share(result['accuracy_accepted'], 1)}.", "",
        "| Вид | Кадров | Верных выводов | Precision | Recall |", "|---|---:|---:|---:|---:|",
    ]
    for label, row in result["per_class"].items():
        safe_label = label.replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {safe_label} | {row['support']} | {row['correct']} | {row['precision']:.1%} | {row['recall']:.1%} |")
    (artifact_dir / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_artifact(
    artifact_dir: str | Path,
    data_dir: str | Path | None = None,
    threshold: float | None = None,
    write: bool = True,
) -> dict[str, Any]:
    artifact_dir = Path(artifact_dir)
    meta = json.loads((artifact_dir / "meta.json").read_text(encoding="utf-8"))
    validate_artifact_task(artifact_dir, meta)
    oof_path = artifact_dir / "oof.csv"
    if meta.get("mode") == "anomaly" or not oof_path.exists():
        result = {
            "mode": meta.get("mode", "unknown"),
            "backbone": meta.get("backbone"),
            "n_images": meta.get("n_images"),
            "n_assets": meta.get("n_assets"),
            "note": "Полнота не оценена: нет положительных примеров или out-of-fold прогноза.",
        }
        if write:
            (artifact_dir / "eval.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    oof = pd.read_csv(oof_path, dtype={"asset_id": str, "image_id": str, "label": str, "pred_label_oof": str}, keep_default_na=False)
    threshold = float(threshold if threshold is not None else meta["threshold"])
    if meta.get("mode") == "defect_type":
        refused = []
        if data_dir is not None:
            records = load_manifest(data_dir)
            if data_fingerprint(records) != meta.get("data_fingerprint"):
                raise ValueError("Для пересчёта OOF укажите исходную папку обучения.")
            usable_ids = set(oof["image_id"])
            refused = [r.label for r in records if r.label and r.label != meta.get("unusable_label") and r.image_id not in usable_ids]
        result = _type_report(meta, threshold, oof["label"], oof["pred_label_oof"], oof["p_finding_oof"],
                              refused, "OOF по объектам")
        if write:
            _write_type_report(artifact_dir, "eval", result)
        return result
    y = oof["y_binary"].to_numpy(dtype=int)
    p = oof["p_finding_oof"].to_numpy(dtype=float)
    pred = p >= threshold

    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum())
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    auc = float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan")
    ap = float(average_precision_score(y, p)) if len(set(y)) > 1 else float("nan")

    assets = _asset_metrics(oof, threshold)
    coverage = _safe_div(len(oof), meta.get("n_manifest_labeled", len(oof)))
    result: dict[str, Any] = {
        "mode": meta.get("mode"),
        "backbone": meta.get("backbone"),
        "model_version": meta.get("version"),
        "task": meta.get("task"),
        "threshold": threshold,
        "threshold_reason": meta.get("threshold_reason"),
        "n_images": int(len(oof)),
        "n_assets": int(oof["asset_id"].nunique()),
        "n_positive_images": int(y.sum()),
        "prevalence_images": round(_safe_div(int(y.sum()), len(y)), 4),
        "roc_auc": round(auc, 4) if np.isfinite(auc) else None,
        "average_precision": round(ap, 4) if np.isfinite(ap) else None,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_images_per_100": round(_safe_div(fp, len(oof)) * 100, 2),
        "coverage_of_labelled_usable": round(coverage, 4),
        "refusal_share": round(1 - coverage, 4),
        "conditions": _condition_breakdown(oof, threshold),
        "synthetic": bool((oof["source"] == "synthetic").all()),
        "source_counts": oof["source"].value_counts().to_dict(),
        "note": (
            "OOF-оценка по объектам. Если source=synthetic, это проверка механики пайплайна, "
            "а не качество на реальных снимках МГТС."
        ),
    }
    result["precision_images"], result["recall_images"], result["f1_images"] = result["precision"], result["recall"], result["f1"]
    result.update(assets)

    # End-to-end recall must account for objects whose every frame was unusable.
    if data_dir is not None:
        try:
            records = load_manifest(data_dir)
            expected = meta.get("data_fingerprint")
            if expected and data_fingerprint(records, legacy=not expected.startswith("v2-")) != expected:
                raise ValueError("Манифест отличается от обучающего. Для других данных укажите --holdout.")
            labeled = [r for r in records if r.label and r.label != meta.get("unusable_label", "unusable")]
            positives = {r.asset_id for r in labeled if r.label != meta.get("normal_label", "normal")}
            detected = set(oof.loc[oof["p_finding_oof"] >= threshold, "asset_id"])
            result["labelled_manifest_images"] = len(labeled)
            result["positive_manifest_assets"] = len(positives)
            result["end_to_end_recall_assets"] = round(_safe_div(len(positives & detected), len(positives)), 4)
            result["end_to_end_refusal_assets"] = int(len(positives - set(oof["asset_id"])))
            result["coverage_of_labelled_usable"] = round(_safe_div(len(oof), len(labeled)), 4)
            result["refusal_share"] = round(1 - result["coverage_of_labelled_usable"], 4)
        except FileNotFoundError as exc:
            result["manifest_note"] = f"Не удалось пересчитать end-to-end: {exc}"

    # Threshold sweep for the report and the UI explanation.
    table = asset_threshold_table(y, p, oof["asset_id"].to_numpy())
    result["threshold_options"] = {
        "max_f1": select_threshold(table)[0],
        "max_recall_fpr_5": _threshold_for_fpr(table, 0.05),
        "max_recall_fpr_10": _threshold_for_fpr(table, 0.10),
    }

    if write:
        (artifact_dir / "eval.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        (artifact_dir / "eval.md").write_text(render_markdown(result), encoding="utf-8")
        rows = oof.copy()
        rows["pred"] = pred
        rows.sort_values("p_finding_oof", ascending=False).head(40).to_csv(
            artifact_dir / "top_predictions.csv", index=False
        )
    return result


def _threshold_for_fpr(table: pd.DataFrame, max_fpr: float) -> float | None:
    allowed = table[table["fpr"] <= max_fpr]
    if allowed.empty:
        return None
    best_recall = allowed["recall"].max()
    return float(allowed[allowed["recall"] == best_recall].sort_values("threshold").iloc[0]["threshold"])


def _wilson(successes: int, total: int, z: float = 1.96) -> list[float] | None:
    """95% Wilson interval; small holdouts need an honest uncertainty band."""
    if total <= 0:
        return None
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * ((p * (1 - p) / total + z * z / (4 * total * total)) ** 0.5) / denom
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def evaluate_holdout(
    artifact_dir: str | Path,
    data_dir: str | Path,
    threshold: float | None = None,
    quality: QualityConfig | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Inference-only evaluation on a split that was not used for training.

    The public dataset has one photo per asset and weak image-level labels, so
    this is a photo-level check. It is still useful: the threshold was selected
    on out-of-fold predictions of the training part, never on this split.
    """
    artifact_dir = Path(artifact_dir)
    meta = json.loads((artifact_dir / "meta.json").read_text(encoding="utf-8"))
    model = load_model(artifact_dir)
    threshold = float(threshold if threshold is not None else meta["threshold"])
    quality = quality or (QualityConfig(**meta["quality_config"]) if meta.get("quality_config")
                          else quality_config(meta.get("quality_profile", "default")))
    normal_label = meta.get("normal_label", "normal")
    unusable_label = meta.get("unusable_label", "unusable")

    records = load_manifest(data_dir)
    calibration = artifact_dir / ("anomaly_oof.csv" if model.mode == "anomaly" else "oof.csv")
    if calibration.is_file():
        trained = pd.read_csv(calibration, usecols=lambda c: c in {"asset_id", "image_sha256"},
                              dtype=str, keep_default_na=False)
        seen = set(trained["asset_id"])
        overlap = seen.intersection(record.asset_id for record in records)
        if overlap:
            raise ValueError(f"Отложенная выборка пересекается с обучением по {len(overlap)} ID объектов.")
        if "image_sha256" in trained:
            hashes = set(trained["image_sha256"])
            if any(image_sha256(r.image_path) in hashes for r in records):
                raise ValueError("В отложенной выборке есть копия обучающего изображения под другим ID.")
    labelled = [r for r in records if r.label and r.label != unusable_label]
    if not labelled:
        raise ValueError("В отложенной выборке нет размеченных кадров.")
    if model.mode == "defect_type" and any(r.label == normal_label for r in labelled):
        raise ValueError("Для оценки модели видов используйте только подтверждённые повреждения.")
    usable_records = []
    images = []
    refused: list[dict[str, Any]] = []
    for record in labelled:
        image = open_image(record.image_path)
        frame_quality = assess_quality(image, record.bbox, quality)
        if frame_quality.ok:
            usable_records.append(record)
            images.append(classification_image(image, record, meta.get("crop_mode", "auto")))
        else:
            refused.append(
                {
                    "image_id": record.image_id,
                    "asset_id": record.asset_id,
                    "label": record.label,
                    "reasons": "|".join(frame_quality.reasons),
                }
            )

    predictions = model.predict_images(images)
    rows = []
    for record, prediction in zip(usable_records, predictions, strict=True):
        rows.append(
            {
                "image_id": record.image_id,
                "asset_id": record.asset_id,
                "label": record.label,
                "condition": record.condition or "unknown",
                "source": record.source,
                "p_finding": float(prediction.p_finding),
                "pred_label": prediction.pred_label,
                "sign": prediction.sign,
            }
        )
    frame = pd.DataFrame(rows, columns=["image_id", "asset_id", "label", "condition", "source",
                                        "p_finding", "pred_label", "sign"])

    if model.mode == "defect_type":
        result = _type_report(meta, threshold, frame["label"], frame["pred_label"], frame["p_finding"],
                              (r["label"] for r in refused), "отложенные фото")
        result["data_fingerprint"] = data_fingerprint(records)
        result["evaluation_sources"] = sorted({r.source for r in labelled})
        if write:
            _write_type_report(artifact_dir, "eval_holdout", result)
            frame.assign(accepted=frame["p_finding"] >= threshold).to_csv(
                artifact_dir / "holdout_predictions.csv", index=False,
            )
            pd.DataFrame(refused, columns=["image_id", "asset_id", "label", "reasons"]).to_csv(
                artifact_dir / "holdout_refused.csv", index=False,
            )
        return result

    frame["y_binary"] = (frame["label"] != normal_label).astype(int)
    frame["p_finding_oof"] = frame["p_finding"]
    frame["pred"] = frame["p_finding"] >= threshold
    y = frame["y_binary"].to_numpy(dtype=int)
    p = frame["p_finding"].to_numpy(dtype=float)
    pred = p >= threshold
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum())
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    auc = float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan")
    ap = float(average_precision_score(y, p)) if len(set(y)) > 1 else float("nan")
    assets = _asset_metrics(frame, threshold)

    per_label = []
    for label, part in frame.groupby("label"):
        n = len(part)
        positive = bool((part["y_binary"] == 1).any())
        hits = int(part["pred"].sum())
        per_label.append(
            {
                "label": label,
                "n_images": n,
                "recall": round(_safe_div(hits, n), 4) if positive else None,
                "false_positive_rate": round(_safe_div(hits, n), 4) if not positive else None,
            }
        )

    refused_positive = sum(1 for r in refused if r["label"] != normal_label)
    result: dict[str, Any] = {
        "mode": meta.get("mode"),
        "backbone": meta.get("backbone"),
        "model_version": meta.get("version"),
        "task": meta.get("task"),
        "threshold": threshold,
        "threshold_reason": meta.get("threshold_reason") if threshold == meta["threshold"] else "порог задан для оценки вручную",
        "quality_profile": meta.get("quality_profile"),
        "n_images": len(labelled),
        "n_usable": int(len(frame)),
        "n_refused": len(refused),
        "coverage": round(_safe_div(len(frame), len(labelled)), 4),
        "refused_positive": int(refused_positive),
        "refused_negative": int(len(refused) - refused_positive),
        "n_positive_images": int(y.sum()),
        "n_negative_images": int(len(y) - y.sum()),
        "prevalence": round(_safe_div(int(y.sum()), len(y)), 4),
        "roc_auc": round(auc, 4) if np.isfinite(auc) else None,
        "average_precision": round(ap, 4) if np.isfinite(ap) else None,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "precision_ci95": _wilson(tp, tp + fp),
        "recall_ci95": _wilson(tp, tp + fn),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "false_images_per_100": round(_safe_div(fp, len(frame)) * 100, 2),
        "per_label": sorted(per_label, key=lambda row: row["label"]),
        "source_counts": frame["source"].value_counts().to_dict(),
        "note": "Разметка задана на уровне фото. Порог и источник его выбора указаны отдельно.",
    }
    if not tp + fp:
        result["precision"] = None
    if not tp + fn:
        result["recall"] = None
    if not len(frame):
        result["f1"] = None
    result.update({key if key not in result else f"{key}_assets": value for key, value in assets.items()})
    result["end_to_end_recall_images"] = round(_safe_div(tp, tp + fn + refused_positive), 4) if tp + fn + refused_positive else None
    result["evaluation_sources"] = sorted({r.source for r in labelled})
    result["data_fingerprint"] = data_fingerprint(records)
    result["evaluation_dir"] = str(Path(data_dir))
    result["positive_manifest_assets"] = len({r.asset_id for r in labelled if r.label != normal_label})
    found_assets = set(frame.loc[frame["pred"], "asset_id"])
    positive_assets = {r.asset_id for r in labelled if r.label != normal_label}
    result["end_to_end_recall_assets"] = (round(len(found_assets & positive_assets) / len(positive_assets), 4)
                                          if positive_assets else None)

    if write:
        (artifact_dir / "eval_holdout.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (artifact_dir / "eval_holdout.md").write_text(render_holdout_markdown(result), encoding="utf-8")
        output = frame.copy()
        output["pred"] = pred
        output.sort_values("p_finding", ascending=False).to_csv(
            artifact_dir / "holdout_predictions.csv", index=False
        )
        pd.DataFrame(refused, columns=["image_id", "asset_id", "label", "reasons"]).to_csv(
            artifact_dir / "holdout_refused.csv", index=False
        )
    return result


def _fmt_share(value: float | None, digits: int = 0) -> str:
    return "—" if value is None else f"{value:.{digits}%}"


def _fmt_interval(interval: list[float] | None) -> str:
    return "—" if interval is None else "–".join(_fmt_share(v, 1) for v in interval)


def render_holdout_markdown(r: dict[str, Any]) -> str:
    labels = "\n".join(
        f"| {row['label']} | {row['n_images']} | "
        f"{_fmt_share(row['recall'])} | {_fmt_share(row['false_positive_rate'], 1)} |"
        for row in r.get("per_label", [])
    )
    return f"""# Оценка на отложенной части

Проверяемая задача — {r.get("task") or "классы манифеста"}.

Источники по манифесту: {", ".join(r["evaluation_sources"])}. Метрики по фото ниже рассчитаны на пригодных размеченных изображениях. Сквозная полнота с учётом отказов: {_fmt_share(r["end_to_end_recall_images"], 1)}.

- Backbone: `{r.get('backbone')}`
- Профиль качества: `{r.get('quality_profile')}`
- Порог: **{r.get('threshold'):.2f}** — {r.get('threshold_reason')}
- Фото: **{r.get('n_images')}**, пригодно после quality gate: **{r.get('n_usable')}**
- ROC-AUC: **{r.get('roc_auc')}**, Average Precision: **{r.get('average_precision')}**

## Решение на уровне фото

| Метрика | Значение |
|---|---:|
| Precision | {_fmt_share(r.get('precision'), 1)} (95% ДИ {_fmt_interval(r.get('precision_ci95'))}) |
| Recall | {_fmt_share(r.get('recall'), 1)} (95% ДИ {_fmt_interval(r.get('recall_ci95'))}) |
| F1 | {_fmt_share(r.get('f1'), 1)} |
| TP / FP / FN / TN | {r.get('tp')} / {r.get('fp')} / {r.get('fn')} / {r.get('tn')} |
| Лишних карточек на 100 фото | {r.get('false_images_per_100')} |
| Точность первых 10 карточек | {r.get('top10_precision'):.0%} (lift ×{r.get('top10_lift')}) |

## По классам

| Класс | Фото | Recall | FPR |
|---|---:|---:|---:|
{labels}

## Отказы quality gate

- Отказано: **{r.get('n_refused')}** из {r.get('n_images')} (покрытие {r.get('coverage'):.0%})
- Среди отказов положительных: {r.get('refused_positive')}, отрицательных: {r.get('refused_negative')}

## Как читать

Источник порога указан в начале отчёта.
Единица решения — фото; в реальном маршруте несколько кадров одного объекта объединяются
в одну карточку. Отказы не прячут пропуски: они показаны отдельно.
"""


def render_markdown(r: dict[str, Any]) -> str:
    cond_rows = "\n".join(
        f"| {c['condition']} | {c['n_images']} | {c['n_positive']} | {c['recall']:.0%} | {c['false_positive_rate']:.1%} |"
        for c in r.get("conditions", [])
    )
    synthetic = (
        "\n> **Важно:** данные синтетические. Цифры ниже проверяют, что пайплайн обучается и считает метрики, "
        "но не являются оценкой качества на реальных снимках МГТС.\n"
        if r.get("synthetic")
        else ""
    )
    return f"""# Оценка модели

{synthetic}
Задача — {r.get("task") or "классы манифеста"}.

- Backbone: `{r.get('backbone')}`
- Порог: **{r.get('threshold'):.2f}** — {r.get('threshold_reason')}
- Кадров в OOF: **{r.get('n_images')}**, объектов: **{r.get('n_assets')}**
- ROC-AUC: **{r.get('roc_auc')}**, Average Precision: **{r.get('average_precision')}**

## Решение на уровне объекта

| Метрика | Значение |
|---|---:|
| Precision | {r.get('precision'):.3f} |
| Recall | {r.get('recall'):.3f} |
| F1 | {r.get('f1'):.3f} |
| TP / FP / FN / TN (объекты) | {r.get('tp_assets')} / {r.get('fp_assets')} / {r.get('fn_assets')} / {r.get('tn_assets')} |
| Лишних карточек на 100 объектов | {r.get('false_cards_per_100_assets')} |
| Точность первых 10 карточек | {r.get('top10_precision'):.0%} (lift ×{r.get('top10_lift')}) |
| Покрытие пригодных кадров | {r.get('coverage_of_labelled_usable'):.0%} |
| End-to-end recall (с отказами) | {r.get('end_to_end_recall_assets', '—')} |

## Ошибки по условиям

| Условие | Кадров | Положительных | Recall | FPR |
|---|---:|---:|---:|---:|
{cond_rows}

## Как читать

Единица решения — объект в осмотре, а не кадр. Несколько кадров одного объекта
объединяются в одну карточку, поэтому FP считается по уникальным объектам.
Отказ («недостаточно данных») не прячет пропуск: он отдельно учтён в end-to-end
recall. Порог выбран на out-of-fold прогнозах; официальные тестовые изображения
в оценке не участвовали.
"""
