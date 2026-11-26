from PIL import Image
import pytest

from inspector.detector import ManholeLocator, box_iou


def test_box_iou_counts_shared_pixels():
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert box_iou((0, 0, 10, 10), (10, 10, 20, 20)) == 0.0
    assert box_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)


def test_locator_sorts_and_clips_detections_in_original_pixel_space():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")

    class FakeModel:
        def eval(self):
            return self

        def __call__(self, frames):
            assert [tuple(frame.shape) for frame in frames] == [(3, 80, 120)]
            return [{
                "boxes": torch.tensor([[0, 0, 20, 20], [35, 40, 150, 100], [-8, -4, 10, 12], [30, 30, 12, 12]]),
                "scores": torch.tensor([0.1, 0.8, 0.9, 0.8]),
                "labels": torch.tensor([1, 1, 1, 1]),
            }]

    locator = ManholeLocator(FakeModel(), 0.5)
    detections = locator.locate(Image.new("RGB", (120, 80)))

    assert [box for box, score in detections] == [(0.0, 0.0, 10.0, 12.0), (35.0, 40.0, 120.0, 80.0)]
    assert [score for box, score in detections] == pytest.approx([0.9, 0.8])
