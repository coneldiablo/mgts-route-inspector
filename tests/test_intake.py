from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import pytest

from inspector.intake import IntakeError, prepare_manifest


def write_table(path: Path, rows, delimiter: str = ",", encoding: str = "utf-8") -> None:
    with path.open("w", encoding=encoding, newline="") as handle:
        csv.writer(handle, delimiter=delimiter).writerows(rows)


def photo(root: Path, relative: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fixture")
    return path


def manifest(root: Path) -> list[dict[str, str]]:
    with (root / "manifest.csv").open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_csv_maps_russian_fields_and_reports_coverage(tmp_path):
    root = tmp_path / "dataset"
    photo(root, "Маршрут 1/Фото 001.jpg")
    photo(root, "Маршрут 2/кадр__два.png")
    photo(root, "Маршрут 2/дополнительный.webp")
    table = tmp_path / "организаторы.csv"
    write_table(
        table,
        [
            ["Фото", "Номер фото", "Номер объекта", "Вид повреждения", "Статус повреждённого объекта",
             "Широта объекта", "Долгота объекта", "Широта камеры", "Долгота камеры", "Маршрут"],
            ["Маршрут 1\\Фото 001.jpg", "0004", "0012", "смещение", "ремонт", "55,753", "37,62", "", "", "001"],
            ["кадр__два", "", "", "скол", "проверка", "", "", "55,755", "37,60", "002"],
        ],
        delimiter=";",
        encoding="utf-8-sig",
    )
    before = table.read_bytes()

    report = prepare_manifest(root, table, source="final")
    first, second = manifest(root)

    assert (first["image"], first["image_id"], first["asset_id"], first["id_origin"]) == (
        "Маршрут 1/Фото 001.jpg", "0004", "0012", "provided",
    )
    assert (first["label"], first["source_status"], first["source"], first["route_id"]) == (
        "смещение", "ремонт", "final", "001",
    )
    assert first["lat"] == "55.753" and not first["camera_lat"]
    assert (second["image_id"], second["asset_id"], second["id_origin"]) == (
        "Маршрут 2/кадр__два.png", "", "image",
    )
    assert second["camera_lat"] == "55.755" and not second["lat"]
    assert report["rows_total"] == report["rows_valid"] == 2
    assert report["with_asset_id"] == report["without_asset_id"] == report["assets_known"] == 1
    assert report["with_object_coordinates"] == report["with_camera_coordinates"] == 1
    assert report["source_statuses"] == {"ремонт": 1, "проверка": 1}
    assert report["labels"] == {"смещение": 1, "скол": 1}
    assert report["unlisted_image_count"] == 1
    assert report["unlisted_image_examples"] == ["Маршрут 2/дополнительный.webp"]
    assert table.read_bytes() == before


def test_explicit_mapping_disambiguates_and_keeps_gps_provenance(tmp_path):
    root = tmp_path / "data"
    photo(root, "set/0006.jpg")
    table = tmp_path / "unknown_schema.csv"
    write_table(
        table,
        [
            ["Фото", "Имя файла", "Latitude", "Longitude", "Наименование проблемы", "Решение"],
            ["not-used.jpg", "0006.jpg", "55.7", "37.6", "край сломан", "проверка"],
        ],
        delimiter="\t",
    )
    with pytest.raises(IntakeError) as error:
        prepare_manifest(root, table)
    assert error.value.report["error_counts"] == {"ambiguous_column": 1}

    report = prepare_manifest(
        root, table,
        columns={
            "image": "Имя файла", "camera_lat": "Latitude", "camera_lon": "Longitude",
            "label": "Наименование проблемы", "source_status": "Решение",
        },
    )
    row, = manifest(root)
    assert row["lat"] == row["lon"] == ""
    assert (row["camera_lat"], row["camera_lon"]) == ("55.7", "37.6")
    assert (row["label"], row["source_status"]) == ("край сломан", "проверка")
    assert report["unmapped_columns"] == ["Фото"]


def test_bad_rows_report_errors_and_keep_old_manifest(tmp_path):
    root = tmp_path / "data"
    photo(root, "a/duplicate.jpg")
    photo(root, "b/duplicate.jpg")
    outside = photo(tmp_path, "outside.jpg")
    (root / "escape.jpg").symlink_to(outside)
    (root / "manifest.csv").write_bytes(b"previous")
    table = tmp_path / "new.csv"
    write_table(
        table,
        [
            ["image", "image_id", "asset_id", "object_type", "lat", "lon"],
            ["a/duplicate.jpg", "0001", "A", "well", "", ""],
            ["b/duplicate.jpg", "0001", "A", "well", "", ""],
            ["duplicate.jpg", "0002", "", "", "", ""],
            ["absent.jpg", "0003", "", "", "", ""],
            ["../outside.jpg", "0004", "", "", "", ""],
            ["escape.jpg", "0005", "", "", "", ""],
            ["a/duplicate.jpg", "0006", "B", "well", "54", ""],
            ["a/duplicate.jpg", "0007", "A", "pole", "", ""],
        ],
    )

    with pytest.raises(IntakeError) as error:
        prepare_manifest(root, table, overwrite=True)
    report = error.value.report
    assert report["rows_total"] == 8
    assert report["rows_valid"] == 1
    assert report["missing_image_rows"] == 1
    assert report["error_counts"] == {
        "duplicate_image_id": 1, "ambiguous_image": 1, "missing_image": 1, "outside_root": 2,
        "incomplete_coordinates": 1, "conflicting_asset_type": 1,
    }
    assert report["errors"][0]["row"] == 3
    assert (root / "manifest.csv").read_bytes() == b"previous"
    assert not list(root.glob(".manifest-*"))


def test_can_reuse_photo_for_two_explicit_annotations(tmp_path):
    root = tmp_path / "data"
    photo(root, "wide.jpg")
    table = tmp_path / "inventory.csv"
    write_table(
        table,
        [
            ["image", "image_id", "asset_id", "x1", "y1", "x2", "y2"],
            ["wide.jpg", "frame-left", "KK-1", "0", "20", "60", "90"],
            ["wide.jpg", "frame-right", "KK-2", "70", "20", "120", "90"],
        ],
    )
    report = prepare_manifest(root, table)
    assert report["distinct_image_files"] == report["repeated_image_rows"] == 1
    assert report["with_bbox"] == report["rows_valid"] == 2
    assert manifest(root)[1]["bbox_x1"] == "70.0"


def test_existing_manifest_requires_explicit_replacement(tmp_path):
    root = tmp_path / "data"
    photo(root, "p.jpg")
    table = tmp_path / "inventory.csv"
    write_table(table, [["image", "label"], ["p.jpg", "normal"]])
    prepare_manifest(root, table)

    write_table(table, [["image", "label"], ["p.jpg", "смещение"]])
    with pytest.raises(IntakeError) as error:
        prepare_manifest(root, table)
    assert error.value.report["error_counts"] == {"manifest_exists": 1}
    assert manifest(root)[0]["label"] == "normal"

    prepare_manifest(root, table, overwrite=True)
    assert manifest(root)[0]["label"] == "смещение"
    with pytest.raises(IntakeError) as error:
        prepare_manifest(root, root / "manifest.csv", overwrite=True)
    assert error.value.report["error_counts"] == {"same_input_output": 1}


def test_xlsx_sheet_selection_preserves_text_ids_and_native_dates(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    root = tmp_path / "data"
    photo(root, "a.jpg")
    table = tmp_path / "выдача.xlsx"
    workbook = openpyxl.Workbook()
    workbook.active.title = "Инструкция"
    sheet = workbook.create_sheet("Данные")
    sheet.append(["photo", "object_id", "captured_at", "label", "status"])
    sheet.append(["a.jpg", "0042", datetime(2026, 11, 26, 10, 15), "норма", "без работ"])
    workbook.save(table)
    workbook.close()

    report = prepare_manifest(root, table, sheet="Данные")

    assert report["sheet"] == "Данные"
    row, = manifest(root)
    assert (row["asset_id"], row["captured_at"], row["source_status"]) == ("0042", "2026-11-26T10:15:00", "без работ")
