import io
from fastapi.testclient import TestClient
from PIL import Image
from inspector.field import create_app


class Both:
    def predict_image(self, image):
        return dict(object_type="telecom_cabinet", scores=dict(mechanical=.85, coating=.75),
                    action="repair", findings=[{"code":"mechanical"},{"code":"coating"}])


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
