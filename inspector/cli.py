"""Command line entry point.

Examples:
    python -m inspector intake --data data/final --table data/final/observations.xlsx
    python -m inspector demo
    python -m inspector synth --out data/demo --assets 48
    python -m inspector train --data data/demo --out artifacts/demo
    python -m inspector run --data data/demo --artifact artifacts/demo --out runs/demo
    python -m inspector serve --run runs/demo --artifact artifacts/demo
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import AppConfig, ModelConfig

BACKBONES = ("auto", "mobilenet_v3_small", "hybrid", "classical", "dinov2_small", "dinov2_hybrid")


def _print_holdout(result: dict) -> None:
    if result.get("mode") == "defect_type":
        _print_types(result)
        return
    print("\nОценка на отложенной части (фото):")
    print(
        f"  precision {result.get('precision')}  recall {result.get('recall')}  "
        f"F1 {result.get('f1')}"
    )
    print(
        f"  фото: {result.get('n_usable')} пригодно из {result.get('n_images')}; "
        f"TP/FP/FN/TN: {result.get('tp')}/{result.get('fp')}/{result.get('fn')}/{result.get('tn')}"
    )
    print(
        f"  ROC-AUC {result.get('roc_auc')}; AP {result.get('average_precision')}; "
        f"лишних карточек на 100 фото: {result.get('false_images_per_100')}"
    )
    print(f"  отчёты: eval_holdout.md, holdout_predictions.csv")


def _print_eval(result: dict) -> None:
    if result.get("mode") == "defect_type":
        _print_types(result)
        return
    if result.get("mode") == "anomaly":
        print(f"Режим: {result.get('mode')}; полнота не оценена.")
        return
    print("\nОценка (out-of-fold, по объектам):")
    print(f"  precision {result.get('precision')}  recall {result.get('recall')}  F1 {result.get('f1')}")
    print(
        f"  объектов: {result.get('n_assets')}; TP/FP/FN: "
        f"{result.get('tp_assets')}/{result.get('fp_assets')}/{result.get('fn_assets')}"
    )
    print(
        f"  лишних карточек на 100 объектов: {result.get('false_cards_per_100_assets')}; "
        f"top-10 precision {result.get('top10_precision')}"
    )
    print(f"  покрытие пригодных кадров: {result.get('coverage_of_labelled_usable')}")
    if result.get("synthetic"):
        print("  ВНИМАНИЕ: синтетические данные — только проверка механики.")


def _print_types(result: dict) -> None:
    print(f"Виды повреждений · {result['evaluation']}")
    accuracy = result["accuracy_accepted"]
    print(f"  macro-F1 {result['type_macro_f1']:.3f}, вывод по {result['n_accepted']} из {result['n_images']} фото")
    if accuracy is not None:
        print(f"  доля верных среди принятых {accuracy:.1%}")


def cmd_synth(args):
    from .synth import generate

    result = generate(args.out, n_assets=args.assets, seed=args.seed)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_intake(args):
    from .intake import IntakeError, prepare_manifest

    mapping = {}
    for binding in args.column:
        canonical, sep, heading = binding.partition("=")
        canonical, heading = canonical.strip(), heading.strip()
        if not sep or not canonical or not heading or canonical in mapping:
            raise ValueError(f"Некорректная привязка столбца {binding!r}. Используйте поле=название.")
        mapping[canonical] = heading
    try:
        report = prepare_manifest(
            args.data, args.table, columns=mapping, sheet=args.sheet,
            overwrite=args.overwrite, source=args.source,
        )
    except IntakeError as exc:
        print(json.dumps(exc.report, ensure_ascii=False, indent=2))
        raise
    print(json.dumps(report, ensure_ascii=False, indent=2))


def cmd_train(args):
    from .classifier import compare_backbones, train_model
    from .evaluate import evaluate_artifact

    cfg = ModelConfig(
        backbone=args.backbone,
        threshold_mode=args.threshold_mode,
        max_fpr=args.max_fpr,
        cv_folds=args.folds,
        normal_label=args.normal_label,
        unusable_label=args.unusable_label,
        quality_profile=args.quality_profile,
        defect_only=args.defect_only,
    )
    meta = train_model(args.data, args.out, cfg=cfg, backbone=args.backbone, crop_mode=args.crop_mode)
    result = evaluate_artifact(args.out, data_dir=args.data, write=True)
    _print_eval(result)
    if args.compare and meta.get("mode") == "supervised" and meta.get("backbone") != "classical_cv":
        print("\nОбучение простого baseline на тех же объектах…")
        comparison = compare_backbones(args.data, args.out, Path(args.out) / "baseline", cfg)
        print("  main:    ", {k: comparison["main"].get(k) for k in ("precision", "recall", "f1", "roc_auc")})
        print("  baseline:", {k: comparison["baseline"].get(k) for k in ("precision", "recall", "f1", "roc_auc")})
        print(f"  comparison.json: {Path(args.out) / 'comparison.json'}")
    print(f"Отчёт: {Path(args.out) / 'eval.md'}")


def cmd_evaluate(args):
    from .evaluate import evaluate_artifact, evaluate_holdout

    if args.holdout:
        result = evaluate_holdout(args.artifact, args.holdout, threshold=args.threshold)
        _print_holdout(result)
        print(f"Отчёт: {Path(args.artifact) / 'eval_holdout.md'}")
        return
    result = evaluate_artifact(args.artifact, data_dir=args.data, threshold=args.threshold, write=True)
    _print_eval(result)
    print(f"Отчёт: {Path(args.artifact) / 'eval.md'}")


def cmd_run(args):
    from .pipeline import analyze

    summary = analyze(
        args.data,
        args.artifact,
        args.out,
        quality_profile=args.quality_profile,
        threshold=args.threshold,
        crop_mode=args.crop_mode,
        include_normal=not args.no_normal,
        verbose=not args.json,
        registry_path=args.registry,
        location_radius_m=args.location_radius,
        locator_path=args.locator,
        single_target=args.single_target,
        full_frame_artifact=args.full_frame_artifact,
        cloud_review_model=args.cloud_review,
        cloud_limit=args.cloud_limit,
    )
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))


def cmd_serve(args):
    import uvicorn

    from .web import create_app

    app = create_app(args.run, args.artifact, AppConfig(), quality_profile=args.quality_profile,
                     locator_path=args.locator, single_target=args.single_target,
                     full_frame_artifact=args.full_frame_artifact, cloud_review_model=args.cloud_review)
    print(f"Откройте http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


def cmd_demo(args):
    from .evaluate import evaluate_holdout

    data = Path(args.data)
    artifact = Path(args.artifact)
    run = Path(args.run)
    holdout = Path(args.holdout)
    if not (data / "manifest.csv").exists():
        raise FileNotFoundError(
            f"Нет {data / 'manifest.csv'}. Сначала загрузите публичный набор: "
            "python scripts/fetch_public_manholes.py"
        )
    if not (holdout / "manifest.csv").is_file():
        raise FileNotFoundError(f"Для публичного демо нужна отложенная выборка {holdout / 'manifest.csv'}")
    ns = argparse.Namespace(
        data=str(data),
        out=str(artifact),
        backbone=args.backbone,
        threshold_mode="f1",
        max_fpr=0.10,
        folds=5,
        crop_mode="auto",
        normal_label="normal",
        unusable_label="unusable",
        quality_profile=args.quality_profile,
        compare=not args.no_compare,
        defect_only=False,
    )
    cmd_train(ns)
    print("\nОцениваю на отложенной части…")
    _print_holdout(evaluate_holdout(artifact, holdout))
    if ns.compare and (artifact / "comparison.json").is_file():
        print("\nКлассическая модель на той же отложенной части…")
        _print_holdout(evaluate_holdout(artifact / "baseline", holdout))
    run_ns = argparse.Namespace(
        data=str(holdout),
        artifact=str(artifact),
        out=str(run),
        threshold=None,
        crop_mode=None,
        no_normal=False,
        json=False,
        quality_profile=args.quality_profile,
        registry=None,
        location_radius=30.0,
        locator=None,
        single_target=False,
        full_frame_artifact=None,
        cloud_review=None,
        cloud_limit=8,
    )
    cmd_run(run_ns)
    print(f"\nЗапуск интерфейса: python -m inspector serve --run {run} --artifact {artifact}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="inspector", description="Маршрутный осмотр: очередь объектов по видимому признаку")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("intake", help="Подготовить manifest.csv из таблицы и папки с фото")
    p.add_argument("--data", required=True, help="каталог с исходными фото")
    p.add_argument("--table", required=True, help="выданная таблица CSV или XLSX")
    p.add_argument("--sheet", help="имя листа XLSX (по умолчанию первый)")
    p.add_argument("--column", action="append", default=[], metavar="ПОЛЕ=СТОЛБЕЦ", help="явная привязка столбца")
    p.add_argument("--source", default="mgts", help="название источника фото")
    p.add_argument("--overwrite", action="store_true", help="заменить существующий manifest.csv после проверки")
    p.set_defaults(func=cmd_intake)

    p = sub.add_parser("synth", help="Сгенерировать синтетический набор (только для автотестов)")
    p.add_argument("--out", default="data/demo")
    p.add_argument("--assets", type=int, default=48)
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(func=cmd_synth)

    p = sub.add_parser("train", help="Обучить модель на manifest.csv")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--backbone", default="auto", choices=BACKBONES)
    p.add_argument("--threshold-mode", default="f1", choices=["f1", "budget"])
    p.add_argument("--max-fpr", type=float, default=0.10)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--crop-mode", default="auto", choices=["auto", "bbox", "full"])
    p.add_argument("--normal-label", default="normal", help="название штатного класса в manifest")
    p.add_argument("--defect-only", action="store_true", help="определять виды в пакете подтверждённых повреждений")
    p.add_argument("--unusable-label", default="unusable", help="метка непригодного кадра")
    p.add_argument(
        "--quality-profile",
        default="default",
        choices=["default", "lowres"],
        help="default — полноразмерные кадры; lowres — публичный набор 224×224",
    )
    p.add_argument("--compare", action="store_true", default=True)
    p.add_argument("--no-compare", dest="compare", action="store_false")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="Пересчитать метрики по OOF или отложенной части")
    p.add_argument("--artifact", required=True)
    p.add_argument("--data", default=None, help="данные для пересчёта end-to-end по OOF")
    p.add_argument("--holdout", default=None, help="отложенная папка с manifest.csv для честного замера")
    p.add_argument("--threshold", type=float, default=None)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("run", help="Построить очередь карточек по папке с manifest.csv")
    p.add_argument("--data", required=True)
    p.add_argument("--artifact", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--registry", help="CSV реестра объектов с asset_id, lat, lon")
    p.add_argument("--locator", help="артефакт локализатора люков для фото без рамки")
    p.add_argument("--single-target", action="store_true", help="брать лучшую рамку, если по ТЗ на фото один целевой люк")
    p.add_argument("--full-frame-artifact", help="запасной классификатор полного кадра при одном целевом люке")
    p.add_argument("--cloud-review", metavar="GEMINI", help="перепроверять сложные кадры через Vertex по ADC")
    p.add_argument("--cloud-limit", type=int, default=8, help="максимум отправляемых в Vertex кадров за прогон")
    p.add_argument("--location-radius", type=float, default=30.0, metavar="МЕТРЫ", help="радиус поиска кандидатов")
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--crop-mode", default=None, choices=["auto", "bbox", "full"])
    p.add_argument(
        "--quality-profile",
        default=None,
        choices=["default", "lowres"],
        help="по умолчанию берётся профиль, сохранённый вместе с моделью",
    )
    p.add_argument("--no-normal", action="store_true", help="Не сохранять карточки без замечания")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("serve", help="Открыть интерфейс мастера")
    p.add_argument("--run", required=True)
    p.add_argument("--artifact", default=None)
    p.add_argument("--locator", help="артефакт локализатора для новых фото")
    p.add_argument("--single-target", action="store_true", help="ожидать один целевой люк на новом фото")
    p.add_argument("--full-frame-artifact", help="запасной классификатор полного кадра при одном целевом люке")
    p.add_argument("--cloud-review", metavar="GEMINI", help="перепроверять сложные загруженные фото через Vertex по ADC")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument(
        "--quality-profile",
        default=None,
        choices=["default", "lowres"],
        help="по умолчанию берётся профиль, сохранённый вместе с моделью",
    )
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("demo", help="Публичные фото → обучение → отложенная оценка → очередь")
    p.add_argument("--data", default="data/real", help="обучающая папка с manifest.csv")
    p.add_argument("--holdout", default="data/real_holdout", help="отложенная папка с manifest.csv")
    p.add_argument("--artifact", default="artifacts/public")
    p.add_argument("--run", default="runs/public")
    p.add_argument(
        "--backbone",
        default="auto",
        choices=BACKBONES,
        help="auto — CNN+CV при наличии torch, иначе classical; classical — принудительный baseline",
    )
    p.add_argument(
        "--quality-profile",
        default="lowres",
        choices=["default", "lowres"],
        help="для публичного набора 224×224 нужен lowres",
    )
    p.add_argument("--no-compare", action="store_true")
    p.set_defaults(func=cmd_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
        return 0
    except (FileNotFoundError, ValueError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
