import io
from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from inspector import jury_api


def jpeg():
    buffer = io.BytesIO()
    image = Image.new("RGB", (32, 48), "grey")
    exif = image.getexif()
    exif[270] = "private-original-description"
    image.save(buffer, format="JPEG", exif=exif)
    return buffer.getvalue()


class Local:
    def predict_image(self, image):
        assert image.mode == "RGB" and not image.getexif()
        return {"model_release": "v0.4-hr-roi", "object_type": "distribution_board", "object_type_score": .87,
                "scores": {"mechanical": .8, "coating": .2}}


class Cloud:
    def __init__(self, readable=True):
        self.readable = readable
        self.calls = 0

    def inspect(self, image):
        assert not image.getexif()
        self.calls += 1
        return {"model_version": "gemini-4-argon", "status": "ok", "readable": self.readable,
                "scores": {"telecom": 1, "mechanical": .4, "coating": 1}}


def post(client, photo=None):
    return client.post("/analyze", content=jpeg() if photo is None else photo,
                       headers={"Content-Type": "image/jpeg"})


def test_hybrid_keeps_local_type_and_has_a_fixed_budget(monkeypatch):
    monkeypatch.setattr(jury_api, "MIN_CLOUD_INTERVAL", 0)
    day = date(2026, 11, 26)
    monkeypatch.setattr(jury_api, "_utc_day", lambda: day)
    cloud = Cloud()
    client = TestClient(jury_api.create_app(predictor=Local(), cloud_client=cloud, cloud_budget=1))
    assert client.get("/healthz").json()["status"] == "ok"
    assert client.get("/field").status_code == 404
    response = post(client).json()
    assert response["mode"] == "hybrid"
    assert response["model_version"] == "v0.4-hr-roi"
    assert response["object_type"] == "distribution_board"
    assert response["object_type_score"] == .87
    assert response["scores"] == pytest.approx({"mechanical": .7, "coating": .4})
    assert response["analysis"]["weights"] == {"local": .75, "cloud": .25}
    assert response["analysis"]["cloud"]["model"] == "gemini-4-argon"
    assert response["analysis"]["local"]["scores"] == {"mechanical": .8, "coating": .2}
    fallback = post(client).json()
    assert fallback["mode"] == "server_local"
    assert fallback["cloud_status"] == "budget_exhausted"
    assert fallback["scores"] == {"mechanical": .8, "coating": .2}
    assert fallback["analysis"]["cloud"]["scores"] is None
    assert fallback["analysis"]["weights"] == {"local": 1, "cloud": 0}
    assert cloud.calls == 1
    day += timedelta(days=1)
    assert post(client).json()["mode"] == "hybrid"
    assert cloud.calls == 2


def test_unreadable_vertex_result_falls_back():
    client = TestClient(jury_api.create_app(predictor=Local(), cloud_client=Cloud(readable=False)))
    response = post(client).json()
    assert response["mode"] == "server_local"
    assert response["cloud_status"] == "unreadable"
    assert response["analysis"]["cloud"]["scores"] is None


def test_invalid_images_never_reach_inference(monkeypatch):
    cloud = Cloud()
    client = TestClient(jury_api.create_app(predictor=Local(), cloud_client=cloud))
    assert client.post("/analyze", content=jpeg()).status_code == 415
    assert post(client, b"not a photograph").status_code == 422
    monkeypatch.setattr(jury_api, "MAX_PIXELS", 32)
    assert post(client).status_code == 413
    monkeypatch.setattr(jury_api, "MAX_BYTES", 32)
    assert post(client).status_code == 413
    assert cloud.calls == 0


@pytest.mark.parametrize(("origin", "status"), [
    ("https://coneldiablo.github.io", 200), ("http://127.0.0.1:8081", 200), ("https://localhost:8081", 200), ("https://other.example", 400),
])
def test_cors_preflight(origin, status):
    client = TestClient(jury_api.create_app(predictor=Local()))
    response = client.options("/analyze", headers={"Origin": origin, "Access-Control-Request-Method": "POST",
                                                   "Access-Control-Request-Headers": "Content-Type, ngrok-skip-browser-warning"})
    assert response.status_code == status
    assert response.headers.get("Access-Control-Allow-Origin") == (origin if status == 200 else None)


def test_two_admitted_requests_wait_for_cloud_slot_and_third_gets_429(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from time import monotonic

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(jury_api, "MIN_CLOUD_INTERVAL", .08)
    monkeypatch.setattr(jury_api, "CLOUD_WAIT_TIMEOUT", 2)
    first_active = Event()
    second_local = Event()
    release_first = Event()

    class CountingLocal(Local):
        calls = 0

        def predict_image(self, image):
            result = super().predict_image(image)
            self.calls += 1
            if self.calls == 2:
                second_local.set()
            return result

    class WaitingCloud(Cloud):
        starts = []

        def inspect(self, image):
            self.starts.append(monotonic())
            if len(self.starts) == 1:
                first_active.set()
                assert release_first.wait(2)
            return super().inspect(image)

    cloud = WaitingCloud()
    app = jury_api.create_app(predictor=CountingLocal(), cloud_client=cloud)
    with TestClient(app) as client, ThreadPoolExecutor(max_workers=3) as workers:
        first = workers.submit(post, client)
        try:
            assert first_active.wait(2)
            second = workers.submit(post, client)
            assert second_local.wait(2)
            assert not second.done()
            busy = workers.submit(post, client).result(timeout=2)
            assert busy.status_code == 429
            assert busy.headers["Retry-After"] == "3"
        finally:
            release_first.set()
        assert first.result(timeout=2).json()["mode"] == "hybrid"
        assert second.result(timeout=2).json()["mode"] == "hybrid"
    assert cloud.calls == 2
    assert cloud.starts[1] - cloud.starts[0] >= .075
    assert not list(tmp_path.iterdir())
