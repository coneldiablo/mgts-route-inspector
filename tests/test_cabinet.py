from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from inspector.cabinet import cabinet_result
from inspector.config import quality_config
from inspector.quality import assess_quality
from scripts.cabinet_final import Frame, assign_groups, read_dss


def test_explicit_dual_label_and_unlabelled_frame_are_retained(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    source = tmp_path / "DSS"
    labeled = source / "МП. ТКШ"
    labeled.mkdir(parents=True)
    for filename in ("a.JPG", "b.png"):
        (labeled / filename).touch()
    (source / "extra.jpg").touch()
    sheet = openpyxl.Workbook()
    sheet.active.append(["Имя файла", "Статус", "Описание"])
    sheet.active.append(["a", "Деформация с повреждением ЛКП.", ""])
    sheet.active.append(["b.png", "Деформация без повреждения ЛКП.", ""])
    sheet.save(labeled / "assets.xlsx")

    frames = read_dss([source])
    assert [frame.truth for frame in frames] == [(1, 1, 1), (1, 1, 0), None]


def test_related_frames_share_group_without_merging_unrelated():
    frames = [Frame(Path(f"{name}.jpg"), None) for name in ("a", "b", "c")]
    metadata = np.array([["one", None], ["two", None], ["three", None]], dtype=object)
    assign_groups(frames, [0, (1 << 32) - 1, (1 << 64) - 1], metadata,
                  [["DSS-001", "DSS-003"]])
    assert frames[0].group == frames[2].group
    assert frames[0].group != frames[1].group


def test_findings_are_independent_and_poor_quality_requires_inspection():
    rng = np.random.default_rng(42)
    rich = Image.fromarray(rng.integers(0, 255, size=(350, 350, 3), dtype=np.uint8))
    tiny = Image.new("RGB", (8, 8), "black")
    cfg = quality_config("lowres")
    scores = {"telecom": .9, "mechanical": .87, "coating": .63}

    good = cabinet_result(scores, assess_quality(rich, bbox=None, cfg=cfg))
    bad = cabinet_result(scores, assess_quality(tiny, bbox=None, cfg=cfg))
    assert good["action"] == "repair"
    assert [item["code"] for item in good["findings"]] == ["mechanical", "coating"]
    assert bad["action"] == "inspect"
