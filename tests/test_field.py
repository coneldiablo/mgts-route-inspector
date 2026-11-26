import io
import json
import sqlite3

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from inspector.field import create_app


class Both:
    def predict_image(self, image):
        return dict(object_type="telecom_cabinet", scores=dict(mechanical=.85, coating=.75),
                    action="repair", findings=[{"code":"mechanical"},{"code":"coating"}])


class Scores:
    def __init__(self, mechanical, coating, quality_ok):
        self.result = {
            "object_type": "distribution_board",
            "scores": {"mechanical": mechanical, "coating": coating},
            "quality": {"ok": quality_ok},
            "action": "inspect",
            "findings": [],
        }

    def predict_image(self, image):
        return self.result


def upload(color):
    data = io.BytesIO()
    Image.new("RGB", (48, 64), color).save(data, format="JPEG")
    return {"image": ("photo.jpg", data.getvalue(), "image/jpeg")}


def test_two_work_orders_and_scoped_evidence(tmp_path):
    database = tmp_path / "db"
    c = TestClient(create_app(database, Both()))
    out = io.BytesIO()
    Image.new("RGB", (48, 64)).save(out, format="JPEG")
    photo = {"image": ("evidence.jpg", out.getvalue(), "image/jpeg")}
    resp = c.post("/api/inspections", files=photo)
    assert resp.status_code == 201
    url = "/api/inspections/" + resp.json()["id"]
    alt = "/api/inspections/" + c.post("/api/inspections",files=photo).json()["id"]
    for code in ["mechanical","coating"]:
        assert c.post(url+"/findings/"+code,json={"action":"confirm"}).status_code == 200
    def attach(purpose, **data):
        out = io.BytesIO()
        color = {"mechanical": "#2887a1", "coating": "#8040aa"}.get(data.get("scope"), "#f42254")
        Image.new("RGB", (48, 64), color).save(out, format="JPEG")
        return c.post(url+"/photos",files={"image": ("new.jpg",out.getvalue(),"image/jpeg")},
                      data=dict(purpose=purpose, **data)).json()["photos"][-1]["id"]
    assert c.post(url+"/photos", files=photo, data=dict(purpose="after",scope="mechanical")).status_code == 409
    pending = c.post(url+"/retakes",json={"scope":"mechanical"}).json()["requests"][0]
    assert "петли" in pending["instruction"]
    assert pending["origin"] == "manual"
    assert pending["source_photo_id"] is None
    retake = attach("retake",request_id=pending["id"])
    assert c.post(url+"/findings/mechanical",json={"action":"accept","photo_id":retake}).status_code == 409
    after1 = attach("after",scope="mechanical")
    assert c.post(url+"/findings/coating",json={"action":"accept","photo_id":after1}).status_code == 409
    partial = c.post(url+"/findings/mechanical",json={"action":"accept","photo_id":after1}).json()
    assert not partial["closed"]
    assert c.get(alt+"/photos/"+after1).status_code == 404
    after2 = attach("after",scope="coating")
    assert c.post(url+"/findings/coating",json={"action":"accept","photo_id":after2}).json()["closed"]
    restarted = TestClient(create_app(database,Both()))
    assert restarted.get(url).json()["closed"]
    assert after2 in restarted.get("/api/export.csv").text
    assert c.post("/api/inspections",files=photo,data={"lat":55}).status_code == 422


def test_bad_photo_does_not_create_card(tmp_path):
    c=TestClient(create_app(tmp_path/"db",Both()))
    assert c.post("/api/inspections",files={"image":("x.jpg",b"wrong","image/jpeg")}).status_code == 422
    assert c.get("/api/inspections").json() == []


@pytest.mark.parametrize(("mechanical", "coating", "quality_ok", "scope"), [
    (.40, .12, True, "mechanical"),
    (.90, .60, True, "coating"),
    (.55, .49, True, "coating"),
    (.50, .44, True, "mechanical"),
    (.55, .45, True, "mechanical"),
    (.39999, .60001, True, None),
    (.81, .93, True, None),
    (.49, .51, False, "object"),
    (.85, .95, False, "object"),
])
def test_automatic_retake_selection(tmp_path, mechanical, coating, quality_ok, scope):
    client = TestClient(create_app(tmp_path / "card.sqlite3", Scores(mechanical, coating, quality_ok)))
    response = client.post("/api/inspections", files=upload("navy"))
    assert response.status_code == 201
    card = response.json()
    if scope is None:
        assert card["requests"] == []
        return

    request, = card["requests"]
    assert request["scope"] == scope
    assert request["origin"] == "automatic"
    assert request["source_photo_id"] == card["photos"][0]["id"]
    assert request["evidence"] is None
    if scope == "mechanical":
        assert "петли" in request["instruction"] and "крепления" in request["instruction"]
    elif scope == "coating":
        assert "участок покрытия" in request["instruction"] and "рассеянном свете" in request["instruction"]
    else:
        assert "шкаф целиком" in request["instruction"]


def test_automatic_retake_and_both_photos_survive_reload(tmp_path):
    db_path = tmp_path / "card.sqlite3"
    predictor = Scores(.22, .51, True)
    with TestClient(create_app(db_path, predictor)) as client:
        card = client.post("/api/inspections", files=upload("navy")).json()
        url = f"/api/inspections/{card['id']}"
        request, = card["requests"]
        initial = request["source_photo_id"]
        before = client.get(url + "/photos/" + initial)
        assert before.status_code == 200
        assert before.headers["content-type"] == "image/jpeg"
        retake = client.post(url + "/photos", files=upload("beige"),
                             data={"purpose": "retake", "request_id": request["id"]})
        assert retake.status_code == 201
        new_photo = retake.json()["photos"][-1]
        assert new_photo["purpose"] == "retake"
        assert new_photo["scope"] == "coating"

    with TestClient(create_app(db_path, predictor)) as restarted:
        loaded = restarted.get(url).json()
        saved_request, = loaded["requests"]
        assert saved_request == request | {"evidence": new_photo["id"]}
        assert restarted.get(url + "/photos/" + initial).content == before.content
        assert restarted.get(url + "/photos/" + new_photo["id"]).status_code == 200


def test_legacy_retake_keeps_manual_origin_and_instruction(tmp_path):
    db_path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(db_path) as db:
        db.executescript("""
            CREATE TABLE cards(id TEXT PRIMARY KEY, created TEXT, prediction TEXT, lat REAL, lon REAL);
            CREATE TABLE requests(id TEXT PRIMARY KEY, card TEXT, scope TEXT, instruction TEXT, evidence TEXT);
        """)
        db.execute("INSERT INTO cards VALUES(?,?,?,?,?)",
                   ("legacy", "2026-11-26", json.dumps(Both().predict_image(None)), None, None))
        db.execute("INSERT INTO requests VALUES(?,?,?,?,?)",
                   ("legacy-request", "legacy", "object", "Сохранённая инструкция", None))

    with TestClient(create_app(db_path, Both())) as client:
        request, = client.get("/api/inspections/legacy").json()["requests"]
        assert request == {
            "id": "legacy-request",
            "scope": "object",
            "instruction": "Сохранённая инструкция",
            "origin": "manual",
            "evidence": None,
            "source_photo_id": None,
        }
