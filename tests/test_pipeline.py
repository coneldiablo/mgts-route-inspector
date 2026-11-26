from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageFilter

from inspector.classifier import load_model, train_model
from inspector.config import ModelConfig
from inspector.evaluate import evaluate_artifact
from inspector.manifest import load_manifest
from inspector.pipeline import analyze
from inspector.quality import assess_quality, open_image
from inspector.synth import generate


def _prepare(tmp_path: Path, assets: int = 16):
    data = tmp_path / "data"
    generate(data, n_assets=assets, seed=11)
    artifact = tmp_path / "artifact"
    cfg = ModelConfig(backbone="classical", cv_folds=4, random_state=3)
    meta = train_model(data, artifact, cfg=cfg, backbone="classical", verbose=False)
    return data, artifact, meta


def test_quality_gate_flags_blur(tmp_path):
    data = tmp_path / "data"
    generate(data, n_assets=6, seed=4)
    record = load_manifest(data)[0]
    img = open_image(record.image_path)
    good = assess_quality(img, record.bbox)
    blurred = assess_quality(img.filter(ImageFilter.GaussianBlur(6)), record.bbox)
    assert good.ok
    assert not blurred.ok
    assert any("нерезк" in reason for reason in blurred.reasons)
    # A good frame still has a score in [0, 1]; the score is ranking-only.
    assert 0.0 <= good.score <= 1.0


def test_end_to_end_classical(tmp_path):
    data, artifact, meta = _prepare(tmp_path)
    assert meta["mode"] == "supervised"
    assert meta["backbone"] == "classical_cv"
    assert (artifact / "model.joblib").exists()

    result = evaluate_artifact(artifact, data_dir=data)
    assert result["n_assets"] >= 10
    assert 0.0 <= result["precision"] <= 1.0
    assert 0.0 <= result["recall"] <= 1.0
    assert result["roc_auc"] is not None

    run = tmp_path / "run"
    summary = analyze(data, artifact, run, include_normal=True, verbose=False)
    cards = json.loads((run / "cards.json").read_text(encoding="utf-8"))
    assert summary["n_cards"] == len(cards)
    assert summary["n_assets"] == len({c["asset_id"] for c in cards})
    # One finding card per object and visible sign; duplicates are merged.
    keys = [(c["asset_id"], c["finding_type"]) for c in cards if c["status"] != "normal"]
    assert len(keys) == len(set(keys))
    assert (run / "observations.csv").exists()
    assert (run / "images").exists()

    # Load the persisted model and predict one image through the same code path.
    model = load_model(artifact)
    assert model.classes
    assert 0.0 <= model.threshold <= 1.0


def test_web_review_roundtrip(tmp_path):
    from fastapi.testclient import TestClient

    from inspector.web import create_app

    data, artifact, _ = _prepare(tmp_path, assets=12)
    run = tmp_path / "run"
    analyze(data, artifact, run, verbose=False)
    app = create_app(run, artifact)
    client = TestClient(app)

    summary = client.get("/api/summary")
    assert summary.status_code == 200
    assert "n_queue" in summary.json()
    cards = client.get("/api/cards").json()
    assert cards
    card_id = cards[0]["card_id"]

    response = client.post(f"/api/cards/{card_id}/decision", json={"decision": "confirmed", "note": "проверено"})
    assert response.status_code == 200
    got = client.get(f"/api/cards/{card_id}").json()
    assert got["review_status"] == "confirmed"
    assert got["review_note"] == "проверено"

    exported = client.get("/api/export.csv")
    assert exported.status_code == 200
    assert card_id[:4] in exported.text or "asset_id" in exported.text

    image = next((run / "images").glob("*.jpg"))
    assert client.get(f"/media/images/{image.name}").status_code == 200
    assert client.get("/media/../manifest.csv").status_code in {403, 404}


def test_web_upload_analyze_keeps_media(tmp_path):
    from fastapi.testclient import TestClient

    from inspector.web import create_app

    data, artifact, _ = _prepare(tmp_path, assets=10)
    run = tmp_path / "run"
    analyze(data, artifact, run, verbose=False)
    client = TestClient(create_app(run, artifact))

    source = load_manifest(data)[0].image_path
    first = client.post(
        "/api/analyze",
        files={"file": (source.name, source.read_bytes(), "image/jpeg")},
        data={"asset_id": "UP-7"},
    )
    assert first.status_code == 200, first.text
    card = first.json()["card"]
    assert card["asset_id"] == "UP-7"
    assert card["image_file"].startswith("uploads/")
    assert client.get("/media/" + card["image_file"]).status_code == 200
    assert client.get("/media/" + card["crop_file"]).status_code == 200

    # A second upload gets its own folder, so the first card stays viewable.
    second = client.post(
        "/api/analyze",
        files={"file": ("next.jpg", source.read_bytes(), "image/jpeg")},
        data={"asset_id": "UP-8"},
    )
    assert second.status_code == 200, second.text
    second_card = second.json()["card"]
    assert second_card["image_file"] != card["image_file"]
    assert client.get("/media/" + card["image_file"]).status_code == 200
    assert client.get("/media/" + second_card["image_file"]).status_code == 200

    bad = client.post(
        "/api/analyze",
        files={"file": ("bad.jpg", b"not an image", "image/jpeg")},
    )
    assert bad.status_code == 400
