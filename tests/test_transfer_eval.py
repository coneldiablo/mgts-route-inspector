import pytest
from PIL import Image, PngImagePlugin

from inspector.manifest import Record
from scripts.benchmark_transfer import _check_disjoint, _exclude_known_pairs, summarize


def test_decoded_rgb_overlap_catches_reencoded_copy(tmp_path):
    image = Image.new("RGB", (20, 20), "brown")
    first = tmp_path / "train.png"
    second = tmp_path / "holdout.png"
    image.save(first)
    info = PngImagePlugin.PngInfo()
    info.add_text("sample", "duplicate")
    image.save(second, pnginfo=info)

    assert first.read_bytes() != second.read_bytes()
    with pytest.raises(ValueError, match="'decoded_rgb': 1"):
        _check_disjoint([Record("first", first, "a")], {"test": [Record("second", second, "b")]})


def test_only_pairs_with_current_training_id_are_excluded(tmp_path):
    pair_path = tmp_path / "known.csv"
    pair_path.write_text("query,train\nfirst,training\nsecond,another_run\nfirst,training\n")
    train = [Record("training", tmp_path / "train.jpg", "a")]
    folds = {"new_name": [Record("first", tmp_path / "a.jpg", "b"), Record("second", tmp_path / "b.jpg", "c")]}

    assert _exclude_known_pairs(pair_path, train, folds) == {"new_name": {"first": ["training"]}}


def test_fpr_on_normal_only_challenge_without_claiming_defect_recall():
    rows = [
        {"label": "normal", "status": "finding", "decision": "unconfirmed_cover"},
        {"label": "normal", "status": "uncertain", "decision": None},
        {"label": "normal", "status": "rephoto", "decision": None},
        {"label": "normal", "status": "normal", "decision": "normal"},
    ]
    report = summarize(rows, "normal")

    assert report["fpr_e2e"] == 0.25
    assert report["fpr_usable"] == 0.333333
    assert report["review_rate"] == 0.75
    assert report["recall_e2e"] is None and report["f1_e2e"] is None


def test_refused_positive_counts_in_end_to_end_and_class_accuracies():
    rows = [
        {"label": "shifted", "status": "finding", "decision": "shifted"},
        {"label": "shifted", "status": "finding", "decision": "broken"},
        {"label": "shifted", "status": "rephoto", "decision": None},
        {"label": "broken", "status": "uncertain", "decision": None},
    ]
    report = summarize(rows, "normal")

    assert report["recall_usable"] == 0.666667
    assert report["recall_e2e"] == 0.5
    assert report["per_label"]["shifted"]["recall"] == 0.333333
    assert report["fpr_e2e"] is None
