"""Supervised state model, threshold calibration and anomaly fallback."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .config import ModelConfig, QualityConfig, quality_config, PUBLIC_SOURCE, PUBLIC_TASK
from .features import get_embedder
from .manifest import Record, data_fingerprint, image_sha256, load_manifest
from .quality import assess_quality, crop_bbox, open_image

POSITIVE_SIGN_FALLBACK = "visible_sign"


@dataclass
class Prediction:
    p_finding: float
    p_normal: float
    probs: dict[str, float]
    pred_label: str
    sign: str
    distance: float | None = None


def classification_image(img: Image.Image, record: Record, crop_mode: str = "auto") -> Image.Image:
    if record.bbox is not None and crop_mode in {"auto", "bbox"}:
        return crop_bbox(img, record.bbox)
    return img


def _fit_head(X: np.ndarray, y: np.ndarray, cfg: ModelConfig):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = LogisticRegression(
        max_iter=3000,
        C=1.0,
        class_weight="balanced",
        random_state=cfg.random_state,
    )
    clf.fit(Xs, y)
    return scaler, clf


def _p_finding(clf, probs: np.ndarray, classes: np.ndarray, normal_label: str) -> np.ndarray:
    if normal_label in list(classes):
        idx = list(classes).index(normal_label)
        return np.clip(1.0 - probs[:, idx], 0.0, 1.0)
    return np.ones(len(probs), dtype=np.float32)


def _cv_predict(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    cfg: ModelConfig,
    normal_label: str | None,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Out-of-fold p_finding and predicted labels, grouped by physical asset."""
    n = len(y)
    p_oof = np.full(n, np.nan, dtype=np.float64)
    label_oof = np.array([""] * n, dtype=object)
    folds_used: list[int] = []
    unique_groups = len(set(groups))
    if unique_groups >= 2:
        n_splits = min(cfg.cv_folds, unique_groups)
        if normal_label is None:
            class_groups = min(len(set(groups[y == label])) for label in set(y))
            for n_groups in range(min(n_splits, class_groups), 1, -1):
                splitter = StratifiedGroupKFold(n_splits=n_groups, shuffle=True, random_state=cfg.random_state)
                splits = list(splitter.split(X, y, groups))
                if all(set(y[tr]) == set(y) for tr, _ in splits):
                    break
            else:
                raise ValueError("Для независимой оценки видов нужны несколько объектов каждого вида.")
        else:
            splits = list(GroupKFold(n_splits=n_splits).split(X, y, groups))
    else:
        raise ValueError("Для независимой оценки нужны как минимум два разных объекта.")

    for fold, (tr, te) in enumerate(splits):
        if len(set(y[tr])) < 2 or (normal_label is not None and normal_label not in y[tr]):
            continue
        scaler, clf = _fit_head(X[tr], y[tr], cfg)
        probs = clf.predict_proba(scaler.transform(X[te]))
        p_oof[te] = probs.max(axis=1) if normal_label is None else _p_finding(clf, probs, clf.classes_, normal_label)
        label_oof[te] = clf.classes_[probs.argmax(axis=1)]
        folds_used.append(fold)

    if np.isnan(p_oof).any():
        raise ValueError(
            "Невозможно оценить модель по независимым объектам: в обучающей части "
            "одного из разбиений недостаточно примеров нормы или отклонения. Добавьте объекты каждого класса."
        )
    return p_oof, label_oof, folds_used


def defect_type_metrics(y_true, y_pred, scores, threshold: float) -> dict[str, Any]:
    """Low-confidence and unusable frames count as missed types."""
    truth = np.asarray(y_true, dtype=object)
    pred = np.asarray(y_pred, dtype=object)
    accepted = np.asarray(scores, dtype=float) >= threshold
    correct = accepted & (truth == pred)
    per_class = {}
    for label in sorted(set(truth)):
        relevant = truth == label
        proposed = accepted & (pred == label)
        tp = int((relevant & proposed).sum())
        support, predicted = int(relevant.sum()), int(proposed.sum())
        per_class[label] = {
            "support": support, "correct": tp,
            "precision": tp / predicted if predicted else 0.0,
            "recall": tp / support,
            "f1": 2 * tp / (support + predicted),
        }
    count = int(accepted.sum())
    return {
        "n_images": len(truth), "n_accepted": count, "n_correct": int(correct.sum()),
        "coverage": count / len(truth) if len(truth) else None,
        "accuracy_accepted": int(correct.sum()) / count if count else None,
        "type_macro_f1": float(np.mean([row["f1"] for row in per_class.values()])) if per_class else None,
        "per_class": per_class,
    }


def threshold_table(y_binary: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    rows = []
    for t in np.arange(101) / 100:
        pred = p >= t
        tp = int(np.sum(pred & (y_binary == 1)))
        fp = int(np.sum(pred & (y_binary == 0)))
        fn = int(np.sum(~pred & (y_binary == 1)))
        tn = int(np.sum(~pred & (y_binary == 0)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        fpr = fp / (fp + tn) if fp + tn else 0.0
        rows.append(
            {
                "threshold": round(float(t), 3),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "fpr": fpr,
            }
        )
    return pd.DataFrame(rows)


def asset_threshold_table(y_binary: np.ndarray, p: np.ndarray, groups: np.ndarray) -> pd.DataFrame:
    """Threshold curve at the real decision unit: a unique object, not a frame.

    An object is predicted positive if at least one of its usable frames is above
    the threshold. This is exactly what the queue does, so the threshold is chosen
    on the same unit the master sees.
    """
    frame = pd.DataFrame({"asset_id": groups, "y": y_binary, "p": p})
    grouped = frame.groupby("asset_id").agg({"y": "max", "p": "max"})
    return threshold_table(grouped["y"].to_numpy(), grouped["p"].to_numpy())


def select_threshold(table: pd.DataFrame, mode: str = "f1", max_fpr: float = 0.10) -> tuple[float, str]:
    if mode == "budget":
        allowed = table[table["fpr"] <= max_fpr]
        if not allowed.empty:
            best_recall = allowed["recall"].max()
            candidates = allowed[allowed["recall"] == best_recall]
            row = candidates.sort_values("threshold").iloc[0]
            return float(row["threshold"]), "максимальная полнота при FPR <= {:.0%}".format(max_fpr)
        raise ValueError("Не найден порог, удовлетворяющий заданному ограничению FPR.")
    best_f1 = table["f1"].max()
    candidates = table[table["f1"] == best_f1].sort_values("threshold")
    row = candidates.iloc[-1]
    return float(row["threshold"]), "максимум F1 на out-of-fold объектах (самый высокий порог на плато)"


@dataclass
class InspectorModel:
    embedder_name: str
    scaler: Any
    clf: Any | None
    classes: list[str]
    normal_label: str
    threshold: float
    mode: str
    meta: dict[str, Any]
    anomaly_bank: np.ndarray | None = None
    anomaly_nn: Any | None = None
    anomaly_threshold: float = 0.5

    @property
    def positive_classes(self) -> list[str]:
        return [c for c in self.classes if c != self.normal_label]

    def predict_embeddings(self, X: np.ndarray) -> list[Prediction]:
        if self.mode == "anomaly":
            return self._predict_anomaly(X)
        Xs = self.scaler.transform(X)
        probs = self.clf.predict_proba(Xs)  # type: ignore[union-attr]
        classes = list(self.clf.classes_)  # type: ignore[union-attr]
        normal_idx = classes.index(self.normal_label) if self.normal_label in classes else None
        p_finding = (
            probs.max(axis=1) if self.mode == "defect_type" else
            1.0 - probs[:, normal_idx] if normal_idx is not None else np.ones(len(probs))
        )
        pred_idx = probs.argmax(axis=1)
        pos = [i for i, c in enumerate(classes) if c != self.normal_label]
        out: list[Prediction] = []
        for i in range(len(probs)):
            prob_map = {c: float(probs[i, j]) for j, c in enumerate(classes)}
            sign = classes[max(pos, key=lambda j: probs[i, j])] if pos else POSITIVE_SIGN_FALLBACK
            out.append(
                Prediction(
                    p_finding=float(np.clip(p_finding[i], 0, 1)),
                    p_normal=float(prob_map.get(self.normal_label, 0.0)),
                    probs=prob_map,
                    pred_label=classes[pred_idx[i]],
                    sign=sign,
                )
            )
        return out

    def _predict_anomaly(self, X: np.ndarray) -> list[Prediction]:
        xn = _l2(X)
        dist, _ = self.anomaly_nn.kneighbors(xn, n_neighbors=min(3, len(self.anomaly_bank)))  # type: ignore[arg-type]
        d = dist.mean(axis=1)
        scale = max(0.05 * self.anomaly_threshold, 1e-6)
        p = 1.0 / (1.0 + np.exp(-(d - self.anomaly_threshold) / scale))
        return [
            Prediction(
                p_finding=float(pi),
                p_normal=float(1 - pi),
                probs={"anomaly": float(pi), self.normal_label: float(1 - pi)},
                pred_label="anomaly" if pi >= 0.5 else self.normal_label,
                sign="anomaly",
                distance=float(di),
            )
            for pi, di in zip(p, d)
        ]

    def predict_images(self, images: Sequence[Image.Image]) -> list[Prediction]:
        if not images:
            return []
        embedder = getattr(self, "_embedder", None)
        if embedder is None:
            embedder, _, _ = get_embedder(self.embedder_name, self.meta.get("image_size", 224))
            self._embedder = embedder
        X = embedder.embed(images)
        return self.predict_embeddings(X)

    def save(self, out_dir: str | Path, config: ModelConfig) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "embedder_name": self.embedder_name,
            "scaler": self.scaler,
            "clf": self.clf,
            "classes": self.classes,
            "normal_label": self.normal_label,
            "threshold": self.threshold,
            "mode": self.mode,
            "meta": self.meta,
            "anomaly_bank": self.anomaly_bank,
            "anomaly_threshold": self.anomaly_threshold,
        }
        path = out_dir / "model.joblib"
        joblib.dump(payload, path)
        (out_dir / "meta.json").write_text(
            json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (out_dir / "config.json").write_text(
            json.dumps(config.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        for name in ("eval.json", "eval.md", "eval_holdout.json", "eval_holdout.md",
                     "holdout_predictions.csv", "holdout_refused.csv", "eval_official_normal.json", "comparison.json",
                     "oof.csv", "anomaly_oof.csv", "top_predictions.csv", "thresholds.csv", "thresholds_images.csv"):
            (out_dir / name).unlink(missing_ok=True)
        return path


def validate_artifact_task(artifact_dir: Path, meta: dict) -> None:
    if not meta.get("task") and "open_hatch" in meta.get("classes", []):
        calibration = artifact_dir / "oof.csv"
        if calibration.is_file():
            sources = pd.read_csv(calibration, usecols=["source"])["source"]
            if sources.eq(PUBLIC_SOURCE).any():
                raise ValueError("Архивная модель использовала void как открытый люк. Пересоберите публичное демо.")


def load_model(artifact_dir: str | Path) -> InspectorModel:
    payload = joblib.load(Path(artifact_dir) / "model.joblib")
    validate_artifact_task(Path(artifact_dir), payload["meta"])
    model = InspectorModel(
        embedder_name=payload["embedder_name"],
        scaler=payload["scaler"],
        clf=payload["clf"],
        classes=payload["classes"],
        normal_label=payload["normal_label"],
        threshold=payload["threshold"],
        mode=payload["mode"],
        meta=payload["meta"],
        anomaly_bank=payload.get("anomaly_bank"),
        anomaly_threshold=payload.get("anomaly_threshold", 0.5),
    )
    if model.mode == "anomaly" and model.anomaly_bank is not None:
        nn = NearestNeighbors(n_neighbors=min(3, len(model.anomaly_bank)), metric="cosine")
        nn.fit(model.anomaly_bank)
        model.anomaly_nn = nn
    return model


def _l2(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, 1e-8)


def _load_training_rows(
    records: Iterable[Record],
    cfg: ModelConfig,
    crop_mode: str = "auto",
    quality: QualityConfig | None = None,
):
    quality = quality or quality_config(cfg.quality_profile)
    rows = []
    for r in records:
        if r.label is None or r.label == cfg.unusable_label:
            continue
        img = open_image(r.image_path)
        frame_quality = assess_quality(img, r.bbox, quality)
        if not frame_quality.ok:
            continue
        rows.append((r, classification_image(img, r, crop_mode)))
    return rows


def train_model(
    data_dir: str | Path,
    out_dir: str | Path,
    cfg: ModelConfig | None = None,
    backbone: str | None = None,
    crop_mode: str = "auto",
    quality: QualityConfig | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    cfg = cfg or ModelConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    quality = quality or quality_config(cfg.quality_profile)
    records = load_manifest(data_dir)
    rows = _load_training_rows(records, cfg, crop_mode, quality)
    if verbose:
        labeled = [r for r in records if r.label and r.label != cfg.unusable_label]
        label_counts: dict[str, int] = {}
        for r in labeled:
            label_counts[r.label or "?"] = label_counts.get(r.label or "?", 0) + 1
        print(
            f"Кадров: {len(records)}; с меткой: {len(labeled)}; "
            f"пригодно после quality gate: {len(rows)}; метки: {label_counts}"
        )
    if len(rows) < 4:
        raise ValueError(
            "Слишком мало пригодных размеченных кадров. Нужны хотя бы два класса и несколько объектов."
        )
    labels = np.array([r.label for r, _ in rows], dtype=object)
    groups = np.array([r.asset_id for r, _ in rows], dtype=object)
    if cfg.defect_only and (cfg.normal_label in labels or len(set(labels)) < 2):
        raise ValueError("Режиму видов повреждений нужны как минимум два вида и пакет повреждённых объектов.")
    if len(set(labels)) < 2:
        if labels[0] != cfg.normal_label:
            raise ValueError("Поиск аномалий требует примеров нормального состояния, а не только дефектного класса.")
        if len(set(groups)) < 2:
            raise ValueError("Для поиска аномалий нужны нормальные снимки нескольких независимых объектов.")
        if verbose:
            print("Размечен только один класс — включаю режим поиска аномалий (№ 10).")
        return _train_anomaly(rows, out_dir, cfg, backbone or cfg.backbone, crop_mode, quality)

    if cfg.normal_label not in labels and not cfg.defect_only:
        raise ValueError("Добавьте класс нормы или используйте --defect-only для пакета подтверждённых дефектов.")

    backbone = backbone or cfg.backbone
    t0 = time.time()
    embedder, actual_name, warning = get_embedder(backbone, cfg.image_size)
    images = [im for _, im in rows]
    X = embedder.embed(images)
    if verbose:
        print(f"Признаки: {actual_name}, {X.shape[0]} кадров × {X.shape[1]} признаков, {time.time() - t0:.1f} с")

    classes = sorted(set(labels.tolist()))
    y_binary = (labels != cfg.normal_label).astype(int)
    p_oof, label_oof, folds_used = _cv_predict(X, labels, groups, cfg, None if cfg.defect_only else cfg.normal_label)
    if cfg.defect_only:
        mode = "defect_type"
        table = pd.DataFrame([
            {"threshold": float(t), **{k: v for k, v in defect_type_metrics(labels, label_oof, p_oof, t).items()
                                       if k != "per_class"}}
            for t in np.arange(101) / 100
        ])
        threshold = float(table.sort_values(["type_macro_f1", "coverage", "threshold"], ascending=False).iloc[0]["threshold"])
        reason = "максимум macro-F1 по видам повреждений на OOF с учётом отказов"
        image_table = table
        auc = ap = None
    else:
        mode = "supervised"
        image_table = threshold_table(y_binary, p_oof)
        table = asset_threshold_table(y_binary, p_oof, groups)
        threshold, reason = select_threshold(table, cfg.threshold_mode, cfg.max_fpr)
        auc = float(roc_auc_score(y_binary, p_oof))
        ap = float(average_precision_score(y_binary, p_oof))

    scaler, clf = _fit_head(X, labels, cfg)
    meta = {
        "version": "0.1.0",
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "mode": mode,
        "backbone": actual_name,
        "requested_backbone": backbone,
        "warning": warning,
        "crop_mode": crop_mode,
        "image_size": cfg.image_size,
        "quality_profile": cfg.quality_profile,
        "quality_config": quality.to_dict(),
        "classes": classes,
        "normal_label": cfg.normal_label,
        "unusable_label": cfg.unusable_label,
        "n_images": int(len(labels)),
        "n_assets": int(len(set(groups))),
        "n_manifest_images": int(len(records)),
        "task": ("Вид подтверждённого повреждения" if cfg.defect_only else
                 PUBLIC_TASK if all(r.source == PUBLIC_SOURCE for r in records) else None),
        "n_manifest_labeled": int(len([r for r in records if r.label and r.label != cfg.unusable_label])),
        "n_positive_images": int(y_binary.sum()),
        "n_negative_images": int((1 - y_binary).sum()),
        "threshold": threshold,
        "threshold_reason": reason,
        "threshold_mode": "type_macro_f1" if cfg.defect_only else cfg.threshold_mode,
        "threshold_level": "frame" if cfg.defect_only else "asset",
        "max_fpr": None if cfg.defect_only else cfg.max_fpr,
        "cv_folds": len(set(folds_used)),
        "roc_auc_oof": auc,
        "average_precision_oof": ap,
        "data_fingerprint": data_fingerprint(records),
        "training_seconds": round(time.time() - t0, 1),
        "notes": "OOF по объектам обучающей части; отложенная часть оценивается отдельно.",
    }
    model = InspectorModel(
        embedder_name=actual_name,
        scaler=scaler,
        clf=clf,
        classes=classes,
        normal_label=cfg.normal_label,
        threshold=threshold,
        mode=mode,
        meta=meta,
    )
    model.save(out_dir, cfg)
    table.to_csv(out_dir / "thresholds.csv", index=False)
    image_table.to_csv(out_dir / "thresholds_images.csv", index=False)
    oof = pd.DataFrame(
        {
            "image_id": [r.image_id for r, _ in rows],
            "image_sha256": [image_sha256(r.image_path) for r, _ in rows],
            "asset_id": [r.asset_id for r, _ in rows],
            "label": labels,
            "y_binary": y_binary,
            "p_finding_oof": p_oof,
            "pred_label_oof": label_oof,
            "condition": [r.condition or "unknown" for r, _ in rows],
            "source": [r.source for r, _ in rows],
            "captured_at": [r.captured_at for r, _ in rows],
        }
    )
    oof.to_csv(out_dir / "oof.csv", index=False)
    (out_dir / "summary.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if verbose:
        fit_report = (f"OOF macro-F1 {defect_type_metrics(labels, label_oof, p_oof, threshold)['type_macro_f1']:.3f}"
                      if cfg.defect_only else f"OOF ROC-AUC {auc:.3f}, AP {ap:.3f}")
        print(f"Порог {threshold:.2f} ({reason}). {fit_report}. Классы — {', '.join(classes)}.")
        print(f"Модель сохранена: {out_dir / 'model.joblib'}")
    return meta


def _train_anomaly(
    rows,
    out_dir: Path,
    cfg: ModelConfig,
    backbone: str,
    crop_mode: str,
    quality: QualityConfig,
) -> dict[str, Any]:
    """Fallback when only normal examples exist: nearest-normal distance."""
    embedder, actual_name, warning = get_embedder(backbone, cfg.image_size)
    X = _l2(embedder.embed([im for _, im in rows]))
    groups = np.array([r.asset_id for r, _ in rows])
    # Leave-one-asset-out distances to a bank built from the other assets.
    distances = np.zeros(len(X), dtype=np.float64)
    for g in set(groups):
        train = groups != g
        nn = NearestNeighbors(n_neighbors=min(3, int(train.sum())), metric="cosine").fit(X[train])
        d, _ = nn.kneighbors(X[~train])
        distances[~train] = d.mean(axis=1)
    threshold_d = max(float(np.quantile(distances, 0.95)), 1e-6)
    meta = {
        "version": "0.1.0",
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "mode": "anomaly",
        "backbone": actual_name,
        "requested_backbone": backbone,
        "warning": warning,
        "crop_mode": crop_mode,
        "image_size": cfg.image_size,
        "quality_profile": cfg.quality_profile,
        "quality_config": quality.to_dict(),
        "classes": [cfg.normal_label, "anomaly"],
        "normal_label": cfg.normal_label,
        "unusable_label": cfg.unusable_label,
        "n_images": int(len(X)),
        "n_assets": int(len(set(groups))),
        "threshold": 0.5,
        "threshold_reason": "аномальность выше 95-го перцентиля нормальных объектов",
        "anomaly_distance_threshold": threshold_d,
        "data_fingerprint": data_fingerprint([r for r, _ in rows]),
        "notes": "Обучение только на норме; полнота обнаружения не оценена без положительных примеров.",
    }
    model = InspectorModel(
        embedder_name=actual_name,
        scaler=None,
        clf=None,
        classes=[cfg.normal_label, "anomaly"],
        normal_label=cfg.normal_label,
        threshold=0.5,
        mode="anomaly",
        meta=meta,
        anomaly_bank=X,
        anomaly_threshold=threshold_d,
    )
    nn = NearestNeighbors(n_neighbors=min(3, len(X)), metric="cosine").fit(X)
    model.anomaly_nn = nn
    model.save(out_dir, cfg)
    pd.DataFrame(
        {
            "asset_id": [r.asset_id for r, _ in rows],
            "image_id": [r.image_id for r, _ in rows],
            "image_sha256": [image_sha256(r.image_path) for r, _ in rows],
            "anomaly_distance": distances,
        }
    ).to_csv(out_dir / "anomaly_oof.csv", index=False)
    return meta


def compare_backbones(
    data_dir: str | Path,
    main_artifact: str | Path,
    baseline_artifact: str | Path,
    cfg: ModelConfig | None = None,
) -> dict[str, Any]:
    """Train the classical baseline on the same rows and compare OOF metrics."""
    cfg = cfg or ModelConfig()
    primary = json.loads(Path(main_artifact, "meta.json").read_text())
    train_model(
        data_dir,
        baseline_artifact,
        cfg=cfg,
        backbone="classical",
        crop_mode=primary["crop_mode"],
        quality=QualityConfig(**primary["quality_config"]),
        verbose=False,
    )
    from .evaluate import evaluate_artifact

    main = evaluate_artifact(main_artifact, write=False)
    base = evaluate_artifact(baseline_artifact, write=False)
    comparison = {
        "main": {k: main.get(k) for k in ("backbone", "roc_auc", "average_precision", "precision", "recall", "f1", "false_cards_per_100_assets", "top10_precision")},
        "baseline": {k: base.get(k) for k in ("backbone", "roc_auc", "average_precision", "precision", "recall", "f1", "false_cards_per_100_assets", "top10_precision")},
        "n_images": main.get("n_images"),
        "n_assets": main.get("n_assets"),
        "note": "Обе модели обучены и оценены на одних и тех же объектах; baseline — классические признаки.",
    }
    Path(main_artifact, "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return comparison
