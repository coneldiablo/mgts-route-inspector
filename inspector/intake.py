"""Prepare a canonical manifest from an organizer's photo inventory."""
from __future__ import annotations

import csv
import math
import os
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import date, datetime, time
from io import StringIO
from pathlib import Path, PurePosixPath, PureWindowsPath
from tempfile import NamedTemporaryFile
from typing import Any
from xml.etree.ElementTree import ParseError
from zipfile import BadZipFile


_ALIASES = {
    "image": ("file", "filename", "file_name", "image_path", "image_filename", "photo", "photo_path", "photo_name",
              "фото", "файл", "изображение", "имя файла", "имя фото", "название файла", "имя фотографии",
              "фото объекта", "путь к фото", "путь к файлу", "файл фотографии", "путь к изображению"),
    "image_id": ("photo_id", "frame_id", "ид фото", "id фото", "номер фото", "идентификатор кадра"),
    "asset_id": ("object_id", "registry_id", "инвентарный номер", "инвентарный id",
                 "ид объекта", "id объекта", "номер объекта", "идентификатор объекта"),
    "object_type": ("asset_type", "тип объекта", "вид объекта", "категория объекта"),
    "label": ("defect", "class", "defect_type", "defect_label", "damage_type",
              "метка", "класс", "вид дефекта", "тип дефекта", "вид повреждения", "тип повреждения"),
    "source_status": ("status", "статус", "статус дефекта", "статус объекта", "статус повреждения",
                      "статус поврежденного объекта", "статус повреждённого объекта"),
    "source": ("источник",),
    "captured_at": ("timestamp", "дата съёмки", "время съёмки", "дата и время"),
    "lat": ("object_lat", "asset_lat", "object_latitude", "широта объекта"),
    "lon": ("object_lon", "asset_lon", "object_longitude", "долгота объекта"),
    "camera_lat": ("camera_latitude", "vehicle_lat", "vehicle_latitude", "широта камеры", "широта машины"),
    "camera_lon": ("camera_longitude", "vehicle_lon", "vehicle_longitude", "долгота камеры", "долгота машины"),
    "condition": ("условия", "условия съёмки"),
    "criticality": ("критичность",),
    "route_id": ("route", "id маршрута", "маршрут", "номер маршрута"),
    "visit_id": ("visit", "id осмотра", "номер осмотра", "id визита"),
    "bbox_x1": ("x1", "xmin", "x_min"),
    "bbox_y1": ("y1", "ymin", "y_min"),
    "bbox_x2": ("x2", "xmax", "x_max"),
    "bbox_y2": ("y2", "ymax", "y_max"),
}
_FIELDS = list(_ALIASES) + ["id_origin"]
_COORDINATES = (("lat", "lon"), ("camera_lat", "camera_lon"))
_BBOX = ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif"}
_MAX_DISPLAYED_ERRORS = 40


class IntakeError(ValueError):
    """The source was rejected before publishing a replacement manifest."""

    def __init__(self, report: dict[str, Any]):
        self.report = report
        first = report["errors"][0]["message"]
        super().__init__(f"Импорт не выполнен. Ошибок: {report['error_count']}. {first}")


class _Issue(ValueError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


def _text(value: Any) -> str:
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    return "" if value is None else str(value).strip()


def _heading(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold().replace("ё", "е")
    return re.sub(r"[\s_-]+", " ", value).strip()


def _index_name(value: str) -> str:
    return unicodedata.normalize("NFC", value).casefold()


def _source_columns(header: Sequence[Any], explicit: Mapping[str, str]) -> dict[str, tuple[int, str]]:
    headings: dict[str, tuple[int, str]] = {}
    for index, cell in enumerate(header):
        title = _text(cell)
        key = _heading(title)
        if not key:
            continue
        if key in headings:
            raise _Issue("duplicate_header", f"Неоднозначный заголовок столбца: «{title}».")
        headings[key] = index, title

    mapped: dict[str, tuple[int, str]] = {}
    for field, column in explicit.items():
        if field not in _ALIASES:
            raise _Issue("unknown_field", f"Неизвестное поле манифеста: «{field}».")
        if not isinstance(column, str) or _heading(column) not in headings:
            raise _Issue("unknown_column", f"Для {field} не найден столбец «{column}».")
        mapped[field] = headings[_heading(column)]
    claimed = {index for index, _ in mapped.values()}
    if len(claimed) != len(mapped):
        raise _Issue("mapped_twice", "Один столбец указан для нескольких полей манифеста.")

    for field, aliases in _ALIASES.items():
        if field in mapped:
            continue
        keys = {_heading(name) for name in (field, *aliases)}
        matches = [item for key, item in headings.items() if key in keys and item[0] not in claimed]
        if len(matches) > 1:
            choices = ", ".join(title for _, title in matches)
            raise _Issue("ambiguous_column", f"Для {field} подходят столбцы «{choices}». Задайте соответствие явно.")
        if matches:
            mapped[field] = matches[0]
            claimed.add(matches[0][0])
    if "image" not in mapped:
        raise _Issue("missing_image_column", "Столбец с путём или именем фото не найден. Укажите поле image явно.")
    return mapped


def _csv_delimiter(sample: str, explicit: Mapping[str, str]) -> str:
    known = {_heading(name) for field, aliases in _ALIASES.items() for name in (field, *aliases)}
    known.update(_heading(name) for name in explicit.values() if isinstance(name, str))

    def score(delimiter: str) -> tuple[int, int]:
        rows = csv.reader(StringIO(sample), delimiter=delimiter)
        header = next((row for row in rows if any(_text(cell) for cell in row)), [])
        return sum(_heading(cell) in known for cell in header), len(header)

    return max((",", ";", "\t"), key=score)


@contextmanager
def _table_rows(
    path: Path, sheet: str | None, explicit: Mapping[str, str],
) -> Iterator[tuple[str | None, Iterable[Sequence[Any]]]]:
    if path.suffix.casefold() == ".csv":
        if sheet is not None:
            raise _Issue("unexpected_sheet", "Выбор листа доступен только для XLSX.")
        with path.open(encoding="utf-8-sig", newline="") as handle:
            delimiter = _csv_delimiter(handle.read(65_536), explicit)
            handle.seek(0)
            yield None, csv.reader(handle, delimiter=delimiter, strict=True)
        return
    if path.suffix.casefold() != ".xlsx":
        raise _Issue("table_format", "Поддерживаются таблицы CSV и XLSX.")

    try:
        from openpyxl import load_workbook
        from openpyxl.utils.exceptions import InvalidFileException
    except ImportError as exc:
        raise _Issue("xlsx_dependency", "Для чтения XLSX установите openpyxl или сохраните таблицу в CSV UTF-8.") from exc
    try:
        workbook = load_workbook(path, read_only=True, data_only=True)
    except (BadZipFile, InvalidFileException, KeyError, ParseError) as exc:
        raise _Issue("invalid_xlsx", f"Не удалось прочитать XLSX: {path.name}.") from exc
    try:
        sheets = {worksheet.title: worksheet for worksheet in workbook.worksheets}
        if not sheets:
            raise _Issue("missing_sheet", "В книге XLSX нет листов с ячейками.")
        if sheet is not None and sheet not in sheets:
            raise _Issue("missing_sheet", f"Лист «{sheet}» не найден. Доступны: {', '.join(sheets)}.")
        title = sheet if sheet is not None else next(iter(sheets))
        yield title, sheets[title].values
    finally:
        workbook.close()


class _ImageIndex:
    def __init__(self, root: Path):
        self.root = root
        self.paths: list[Path] | None = None
        self.files: dict[tuple[str, str], list[Path]] | None = None

    def photos(self) -> list[Path]:
        if self.paths is None:
            self.paths = sorted(
                path for path in self.root.rglob("*")
                if path.suffix.casefold() in _IMAGE_SUFFIXES and path.is_file()
            )
        return self.paths

    def unlisted(self, referenced: set[str]) -> list[str]:
        included = set()
        for photo in self.photos():
            path = photo.resolve()
            if path.is_relative_to(self.root):
                included.add(path.relative_to(self.root).as_posix())
        return sorted(included - referenced)

    def _inside_root(self, candidate: Path) -> Path:
        resolved = candidate.resolve()
        if not resolved.is_relative_to(self.root):
            raise _Issue("outside_root", f"Фото находится вне папки данных: «{candidate}».")
        return resolved

    def relative_path(self, value: str) -> str:
        if not value:
            raise _Issue("empty_image", "В строке отсутствует путь к фото.")
        if "\0" in value:
            raise _Issue("invalid_path", "В пути к фото присутствует нулевой символ.")
        path = PurePosixPath(value.replace("\\", "/"))
        if path.is_absolute() or PureWindowsPath(value).drive or ".." in path.parts:
            raise _Issue("outside_root", f"Укажите относительный путь внутри папки данных: «{value}».")
        if not path.name:
            raise _Issue("invalid_path", f"Укажите непустое имя фото: «{value}».")
        direct = self._inside_root(self.root / path)
        if direct.is_file():
            if direct.suffix.casefold() not in _IMAGE_SUFFIXES:
                raise _Issue("image_format", f"Расширение файла фото не поддерживается: «{value}».")
            return direct.relative_to(self.root).as_posix()

        if self.files is None:
            self.files = defaultdict(list)
            for image in self.photos():
                self.files["name", _index_name(image.name)].append(image)
                self.files["stem", _index_name(image.stem)].append(image)
        key = _index_name(path.name)
        matches = self.files.get(("name", key)) or self.files.get(("stem", key), [])
        if not matches:
            raise _Issue("missing_image", f"Фото «{value}» не найдено.")
        if len(matches) > 1:
            choices = ", ".join(str(image.relative_to(self.root)) for image in matches[:3])
            raise _Issue("ambiguous_image", f"Для «{value}» найдено файлов: {len(matches)}. Уточните путь: {choices}.")
        return self._inside_root(matches[0]).relative_to(self.root).as_posix()


def _number(value: str, field: str) -> float:
    try:
        number = float(value.replace(",", "."))
    except ValueError as exc:
        raise _Issue("invalid_number", f"{field} должно быть числом. Получено «{value}».") from exc
    if not math.isfinite(number):
        raise _Issue("invalid_number", f"{field} должно быть конечным числом.")
    return number


def _validate_numbers(row: dict[str, str]) -> None:
    for lat, lon in _COORDINATES:
        if not (row[lat] or row[lon]):
            continue
        if not (row[lat] and row[lon]):
            raise _Issue("incomplete_coordinates", f"Заполните оба поля {lat} и {lon}.")
        latitude, longitude = _number(row[lat], lat), _number(row[lon], lon)
        if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
            raise _Issue("invalid_coordinates", f"Координаты {lat}/{lon} выходят за диапазон WGS84.")
        row[lat], row[lon] = str(latitude), str(longitude)

    if any(row[key] for key in _BBOX):
        if not all(row[key] for key in _BBOX):
            raise _Issue("incomplete_bbox", "Укажите все четыре координаты рамки.")
        x1, y1, x2, y2 = (_number(row[key], key) for key in _BBOX)
        if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
            raise _Issue("invalid_bbox", "Координаты рамки должны быть неотрицательными, ширина и высота — положительными.")
        row.update(zip(_BBOX, map(str, (x1, y1, x2, y2))))

    if row["criticality"]:
        criticality = _number(row["criticality"], "criticality")
        if not 0 <= criticality <= 1:
            raise _Issue("invalid_criticality", "Критичность должна быть в диапазоне от 0 до 1.")
        row["criticality"] = str(criticality)


def _publish(path: Path, rows: list[dict[str, str]], overwrite: bool) -> None:
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            dir=path.parent, prefix=".manifest-", suffix=".csv", delete=False,
            encoding="utf-8", newline="", mode="w",
        ) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        if overwrite:
            os.replace(temporary, path)
        else:
            # Publish the complete file without overwriting a concurrent import.
            os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def prepare_manifest(
    data_dir: str | Path,
    table: str | Path,
    *,
    columns: Mapping[str, str] | None = None,
    sheet: str | None = None,
    overwrite: bool = False,
    source: str = "mgts",
) -> dict[str, Any]:
    """Validate every row and publish data_dir/manifest.csv atomically."""
    root = Path(data_dir).resolve()
    table_path = Path(table).resolve()
    output = root / "manifest.csv"
    report: dict[str, Any] = {
        "manifest": str(output), "table": str(table_path), "sheet": None,
        "rows_total": 0, "rows_valid": 0, "distinct_image_files": 0, "repeated_image_rows": 0,
        "assets_known": 0, "with_asset_id": 0, "without_asset_id": 0,
        "with_object_coordinates": 0, "with_camera_coordinates": 0, "with_bbox": 0,
        "labels": {}, "source_statuses": {}, "mapped_columns": {}, "unmapped_columns": [],
        "missing_image_rows": 0, "unlisted_image_count": 0, "unlisted_image_examples": [],
        "error_count": 0, "error_counts": {}, "errors": [],
    }

    def add_error(issue: _Issue, line: int | None = None) -> None:
        report["error_count"] += 1
        counts = report["error_counts"]
        counts[issue.code] = counts.get(issue.code, 0) + 1
        if len(report["errors"]) < _MAX_DISPLAYED_ERRORS:
            report["errors"].append({"row": line, "code": issue.code, "message": str(issue)})

    explicit = columns if columns is not None else {}
    images = _ImageIndex(root)
    referenced: set[str] = set()
    seen_ids: set[str] = set()
    asset_types: dict[str, str] = {}
    prepared: list[dict[str, str]] = []
    try:
        if not root.is_dir():
            raise _Issue("invalid_data_dir", f"Папка данных не найдена: {root}.")
        if not table_path.is_file():
            raise _Issue("missing_table", f"Таблица не найдена: {table_path}.")
        if table_path == output.resolve():
            raise _Issue("same_input_output", "Исходная таблица совпадает с выходным manifest.csv.")
        if output.exists() and not overwrite:
            raise _Issue("manifest_exists", "manifest.csv уже существует. Подтвердите замену явно.")
        with _table_rows(table_path, sheet, explicit) as (title, values):
            report["sheet"] = title
            numbered = enumerate(values, start=1)
            heading = next(((line, row) for line, row in numbered if any(_text(cell) for cell in row)), None)
            if heading is None:
                raise _Issue("empty_table", "Таблица не содержит заголовков.")
            heading_line, header = heading
            try:
                mapped = _source_columns(header, explicit)
            except _Issue as issue:
                add_error(issue, heading_line)
                raise IntakeError(report) from issue
            report["mapped_columns"] = {key: name for key, (_, name) in mapped.items()}
            report["unmapped_columns"] = [_text(cell) for cell in header
                                          if _text(cell) and _text(cell) not in report["mapped_columns"].values()]
            for line, values_row in numbered:
                cells = [_text(cell) for cell in values_row]
                if not any(cells):
                    continue
                report["rows_total"] += 1
                try:
                    if any(cells[len(header):]):
                        raise _Issue("extra_values", "В строке больше значений, чем столбцов в заголовке.")
                    row = dict.fromkeys(_FIELDS, "")
                    for field, (index, _) in mapped.items():
                        if index < len(cells):
                            row[field] = cells[index]
                    row["image"] = images.relative_path(row["image"])
                    referenced.add(row["image"])
                    row["image_id"] = row["image_id"] or row["image"]
                    if row["image_id"] in seen_ids:
                        raise _Issue("duplicate_image_id", f"Повторяющийся image_id: «{row['image_id']}».")
                    _validate_numbers(row)
                    asset_id, object_type = row["asset_id"], row["object_type"]
                    if asset_id and object_type:
                        normalized_type = _heading(object_type)
                        if asset_types.setdefault(asset_id, normalized_type) != normalized_type:
                            raise _Issue("conflicting_asset_type", f"Для объекта «{asset_id}» указаны разные типы.")
                    row["id_origin"] = "provided" if asset_id else "image"
                    row["source"] = row["source"] or source
                    seen_ids.add(row["image_id"])
                    prepared.append(row)
                except _Issue as issue:
                    add_error(issue, line)
            if not report["rows_total"]:
                raise _Issue("empty_rows", "После заголовка в таблице нет строк с данными.")
            unlisted = images.unlisted(referenced)
            report["unlisted_image_count"] = len(unlisted)
            report["unlisted_image_examples"] = unlisted[:10]
    except _Issue as issue:
        add_error(issue)
    except (OSError, UnicodeError, csv.Error, BadZipFile, ParseError) as exc:
        add_error(_Issue("read_error", f"Таблица или исходные файлы не прочитаны: {exc}"))

    object_ids = [row["asset_id"] for row in prepared if row["asset_id"]]
    distinct_files = len({row["image"] for row in prepared})
    report.update(
        rows_valid=len(prepared), distinct_image_files=distinct_files,
        repeated_image_rows=len(prepared) - distinct_files, assets_known=len(set(object_ids)),
        with_asset_id=len(object_ids), without_asset_id=len(prepared) - len(object_ids),
        with_object_coordinates=sum(bool(row["lat"]) for row in prepared),
        with_camera_coordinates=sum(bool(row["camera_lat"]) for row in prepared),
        with_bbox=sum(bool(row["bbox_x1"]) for row in prepared),
        labels=dict(Counter(row["label"] for row in prepared if row["label"])),
        source_statuses=dict(Counter(row["source_status"] for row in prepared if row["source_status"])),
        missing_image_rows=sum(report["error_counts"].get(code, 0) for code in ("missing_image", "empty_image")),
    )
    if report["error_count"]:
        raise IntakeError(report)
    try:
        _publish(output, prepared, overwrite)
    except FileExistsError as exc:
        add_error(_Issue("manifest_exists", "manifest.csv уже существует. Подтвердите замену явно."))
        raise IntakeError(report) from exc
    except OSError as exc:
        add_error(_Issue("write_error", f"Не удалось сохранить manifest.csv: {exc}"))
        raise IntakeError(report) from exc
    return report
