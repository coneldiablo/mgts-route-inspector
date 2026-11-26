import json
import sys
from types import ModuleType, SimpleNamespace

from PIL import Image
import pytest

from inspector.yolo_locator import YoloLocator, load_yolo_locator


class FakeTensor(list):
    def cpu(self):
        return self

    def tolist(self):
        return list(self)


def test_yolo_batch_filters_classes_and_invalid_boxes_in_original_pixels():
    boxes = SimpleNamespace(
        xyxy=FakeTensor([
            (10, 12, 60, 90),
            (-2, -5, 12, 14),
            (0, 0, 9, 9),
            (0, 0, 10, 10),
            (0, 0, 10, 10),
            (0, 0, float("inf"), 14),
            (70, 2, 80, 14),
        ]),
        conf=FakeTensor([0.6, 0.8, 0.99, 0.1, float("nan"), 0.9, 0.8]),
        cls=FakeTensor([0, 0, 1, 0, 0, 0, 0]),
    )
    calls = []

    def predict(*, source, **options):
        assert [image.mode for image in source] == ["RGB"] * len(source)
        calls.append(([(image.width, image.height) for image in source], options))
        return [SimpleNamespace(boxes=boxes), SimpleNamespace(boxes=None)][:len(source)]

    locator = YoloLocator(SimpleNamespace(predict=predict), 512, 0.45, "YOLO26n")
    first = Image.new("RGBA", (40, 30))
    second = Image.new("L", (10, 20))
    expected = [((0.0, 0.0, 12, 14), 0.8), ((10, 12, 40.0, 30.0), 0.6)]

    assert locator.locate_batch([]) == []
    assert locator.locate_batch([first, second]) == [expected, []]
    assert locator.locate(first) == expected
    assert len(calls) == 2
    assert calls[0] == ([(40, 30), (10, 20)], {
        "device": "cpu", "imgsz": 512, "conf": 0.45, "classes": [0], "max_det": 20, "verbose": False,
    })


@pytest.fixture
def descriptor(tmp_path):
    weights = tmp_path / "best.pt"
    weights.touch()
    path = tmp_path / "locator.json"
    metadata = {
        "format": "mgts-yolo-locator-v1",
        "weights": weights.name,
        "image_size": 512,
        "score_threshold": 0.45,
        "model": "YOLO26n",
    }
    path.write_text(json.dumps(metadata), encoding="utf-8")
    return path, metadata


def test_yolo_descriptor_constructs_local_detector(descriptor, monkeypatch):
    path, metadata = descriptor
    predictor = object()

    def fake_yolo(weights, *, task, verbose):
        assert weights == str(path.with_name(metadata["weights"]).resolve())
        assert task == "detect"
        assert verbose is False
        return predictor

    ultralytics = ModuleType("ultralytics")
    ultralytics.YOLO = fake_yolo
    monkeypatch.setitem(sys.modules, "ultralytics", ultralytics)

    locator = load_yolo_locator(path)

    assert locator.backend == "YOLO26n"
    assert locator.score_threshold == 0.45


@pytest.mark.parametrize("metadata_update", [
    {"format": "unknown"},
    {"weights": "../best.pt"},
    {"weights": "/best.pt"},
    {"weights": "other.json"},
    {"image_size": True},
    {"score_threshold": float("nan")},
    {"score_threshold": 1.1},
    {"model": " "},
])
def test_yolo_descriptor_rejects_invalid_metadata(descriptor, metadata_update):
    path, metadata = descriptor
    path.write_text(json.dumps(metadata | metadata_update), encoding="utf-8")

    with pytest.raises(ValueError):
        load_yolo_locator(path)


def test_yolo_descriptor_rejects_weight_symlink_outside_local_directory(descriptor):
    path, metadata = descriptor
    outside_weights = path.parent.parent / f"{path.parent.name}-outside.pt"
    outside_weights.touch()
    path.with_name("outside.pt").symlink_to(outside_weights)
    path.write_text(json.dumps(metadata | {"weights": "outside.pt"}), encoding="utf-8")

    with pytest.raises(ValueError, match="каталоге дескриптора"):
        load_yolo_locator(path)
