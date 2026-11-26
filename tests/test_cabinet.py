from pathlib import Path
from hashlib import sha256

import numpy as np
import pytest
from PIL import Image

from inspector.cabinet import CabinetInspector, MODEL_FORMAT, cabinet_result, head_probabilities
from inspector.cabinet_roi import CabinetRegion, REVISION
from inspector.cabinet_hr import DIM, REVISION as HR_REVISION, SHA256 as HR_SHA256
from inspector.config import quality_config
from inspector.quality import assess_quality
from scripts.cabinet_final import Frame, assign_groups, read_dss
from scripts.cabinet_compare import C_MULTIPLIERS, FAMILIES, SEEDS, promoted_recipe, select_learners
from scripts.cabinet_release import fetch_verified


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


def test_family_selection_retains_at_most_one_variant_per_family():
    scores = {(name, c): float(i == 1) + c / 100 for i, family in enumerate(FAMILIES)
              for name in family for c in C_MULTIPLIERS}
    selected = select_learners(scores, "family-balanced")
    matches = [next(i for i, family in enumerate(FAMILIES) if name in family)
               for name, c in selected]
    assert len(matches) == len(set(matches)) == 3
    assert selected[0] == ("dino", 24.0)


def test_head_probabilities_support_unweighted_legacy_artifacts():
    class Constant:
        def __init__(self, positive):
            self.positive = positive

        def predict_proba(self, features):
            return np.tile([1 - self.positive, self.positive], (len(features), 1))

    specs = [{"variant": "x", "estimator": Constant(.2)}, {"variant": "x", "estimator": Constant(.8)}]
    assert np.allclose(head_probabilities(specs, {"x": np.zeros((3, 2))}), [.5, .5, .5])


def test_promotion_rejects_more_false_alarms_even_if_macro_f1_rises():
    def measurements(macro, false_alarms):
        return {seed: {"defect_macro_f1": macro, "any_damage": {"fp": false_alarms, "f1": .80}}
                for seed in SEEDS}

    summary = {"control": measurements(.67, 10), "base-expanded": measurements(.77, 13),
               "family-balanced": measurements(.70, 10)}
    assert promoted_recipe(summary) == "family-balanced"
    summary["family-balanced"] = measurements(.674, 10)
    assert promoted_recipe(summary) == "control"


def test_grounded_type_cannot_clear_existing_inspection():
    rng = np.random.default_rng(48)
    rich = Image.fromarray(rng.integers(0, 255, size=(350, 350, 3), dtype=np.uint8))
    quality = assess_quality(rich, bbox=None, cfg=quality_config("lowres"))
    scores = {"telecom": .49, "mechanical": .20, "coating": .29}
    anchored = cabinet_result(scores, quality)
    refined = cabinet_result(scores, quality, reported_type_score=.92)
    assert anchored["object_type"] == "distribution_board"
    assert refined["object_type"] == "telecom_cabinet"
    assert anchored["action"] == refined["action"] == "inspect"
    assert anchored["scores"] == refined["scores"]


@pytest.mark.parametrize(("located", "source"), [
    (False, "full_frame"),
    (True, "grounded_roi"),
])
def test_grounded_adapter_and_full_frame_fallback(tmp_path, monkeypatch, located, source):
    class Constant:
        def __init__(self, value):
            self.value = value

        def predict_proba(self, vectors):
            return np.tile([1 - self.value, self.value], (len(vectors), 1))

    class Vectors:
        def extract(self, images, variants):
            return {name: np.zeros((len(images), 1)) for name in variants}

    artifact = {
        "format": MODEL_FORMAT,
        "heads": {name: [{"variant": "full", "estimator": Constant(value)}]
                  for name, value in {"telecom": .49, "mechanical": .20, "coating": .29}.items()},
        "type_roi": {"estimator": Constant(.92), "grounding_revision": REVISION},
    }
    monkeypatch.setenv("INSPECTOR_CABINET_ROI_DIR", str(tmp_path / "missing"))
    predictor = CabinetInspector(artifact)
    predictor.features = Vectors()
    if located:
        predictor._cabinet_region = lambda image: CabinetRegion((.2, .2, .8, .8), .6)
    result = predictor.predict_image(Image.new("RGB", (300, 300), "gray"))
    assert result["object_type_source"] == source
    assert ("object_region" in result) == located
    assert result["object_type"] == ("telecom_cabinet" if located else "distribution_board")
    assert result["scores"] == {"mechanical": .20, "coating": .29}
    assert result["model_release"] == "v0.3-roi"


def test_prepare_verifies_before_replacing_cached_file(tmp_path):
    upstream = tmp_path / "upstream"
    upstream.write_bytes(b"verified public weights")
    target = tmp_path / "cache" / "weights"
    expected = sha256(upstream.read_bytes()).hexdigest()

    fetch_verified(upstream.as_uri(), target, expected)
    assert target.read_bytes() == upstream.read_bytes()
    upstream.write_bytes(b"bad substitute")
    fetch_verified(upstream.as_uri(), target, expected, verify_only=True)
    with pytest.raises(ValueError, match="SHA-256"):
        fetch_verified(upstream.as_uri(), target, "0" * 64)
    assert target.read_bytes() == b"verified public weights"
    assert list(target.parent.iterdir()) == [target]


@pytest.mark.parametrize("installed", [False, True])
def test_hr_blend_reports_actual_release_and_preserves_type_fallback(tmp_path, monkeypatch, installed):
    class Constant:
        def __init__(self, value):
            self.value = value

        def predict_proba(self, vectors):
            return np.tile([1 - self.value, self.value], (len(vectors), 1))

    class Vectors:
        def extract(self, images, variants):
            return {name: np.zeros((len(images), 1)) for name in variants}

    class HighRes:
        def embed_image(self, image):
            return np.zeros(DIM)

    artifact = {
        "format": MODEL_FORMAT,
        "heads": {name: [{"variant": "full", "estimator": Constant(value)}]
                  for name, value in {"telecom": .49, "mechanical": .20, "coating": .29}.items()},
        "type_roi": {"estimator": Constant(.92), "grounding_revision": REVISION},
        "defect_hr": {"weights": np.zeros((2, DIM)), "bias": np.log([4, 1]),
                      "encoder_revision": HR_REVISION, "encoder_sha256": HR_SHA256},
    }
    monkeypatch.setenv("INSPECTOR_CABINET_ROI_DIR", str(tmp_path / "missing-roi"))
    monkeypatch.setenv("INSPECTOR_SIGLIP_HR_DIR", str(tmp_path / "missing-hr"))
    predictor = CabinetInspector(artifact)
    predictor.features = Vectors()
    if installed:
        predictor.hr_encoder = HighRes()
    result = predictor.predict_image(Image.new("RGB", (300, 300), "gray"))
    assert result["model_release"] == ("v0.4-hr-roi" if installed else "v0.3-roi")
    assert result["defect_score_source"] == ("siglip_hr_blend" if installed else "v02_ensemble")
    assert result["object_type_source"] == "full_frame"
    assert result["scores"] == ({"mechanical": .5, "coating": .395} if installed else
                                {"mechanical": .2, "coating": .29})
