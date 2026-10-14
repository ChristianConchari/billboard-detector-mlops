"""Validación de entrada de la API con un detector falso (sin torch ni MLflow)."""

import io
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "dockerfiles" / "fastapi")]


class FakeDetector:
    default_threshold = 0.15
    labels = ["billboard"]

    def detect(self, image, score_threshold=None):
        w, h = image.size
        t = self.default_threshold if score_threshold is None else score_threshold
        dets = [{"label": "billboard", "score": s, "box": [1.0, 2.0, 3.0, 4.0]}
                for s in (0.9, 0.3, 0.1) if s >= t]
        return {"width": w, "height": h}, dets


@pytest.fixture
def client(monkeypatch):
    from fastapi.testclient import TestClient

    import app as api

    monkeypatch.setattr(api, "_load_until_ready", lambda: _noop())
    api.service.detector = FakeDetector()
    api.service.info = {"name": "billboard-detector", "version": "1", "alias": "champion",
                        "labels": ["billboard"], "score_threshold": 0.15, "metrics": {}}
    monkeypatch.setattr(api.service, "persist", lambda r: None)
    with TestClient(api.app) as c:
        yield c
    api.service.detector = None
    api.service.info = None


async def _noop():
    return None


def png(size=(64, 32)):
    buf = io.BytesIO()
    Image.new("RGB", size).save(buf, "PNG")
    return buf.getvalue()


def post(client, data, ctype="image/png", **params):
    return client.post("/v1/detect", files={"image": ("x", data, ctype)}, params=params)


def test_ok_usa_umbral_calibrado(client):
    r = post(client, png())
    assert r.status_code == 200
    body = r.json()
    assert body["image"] == {"width": 64, "height": 32}
    assert [d["score"] for d in body["detections"]] == [0.9, 0.3]  # 0.1 < 0.15
    assert body["model"]["alias"] == "champion"


def test_umbral_explicito(client):
    assert len(post(client, png(), score_threshold=0.5).json()["detections"]) == 1


@pytest.mark.parametrize("value", ["1.5", "-0.1", "abc"])
def test_umbral_invalido_422(client, value):
    assert post(client, png(), score_threshold=value).status_code == 422


def test_tipo_no_soportado_415(client):
    assert post(client, b"GIF89a", ctype="image/gif").status_code == 415


def test_demasiado_grande_413(client):
    assert post(client, b"0" * (5 * 1024 * 1024 + 2)).status_code == 413


def test_imagen_ilegible_422(client):
    assert post(client, b"no soy una imagen").status_code == 422


def test_dimensiones_fuera_de_rango_422(client):
    assert post(client, png((4097, 8))).status_code == 422


def test_health_y_model(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}
    assert client.get("/v1/model").json()["version"] == "1"


def test_sin_modelo_503():
    from fastapi.testclient import TestClient

    import app as api

    api.service.detector = None
    api.service.info = None
    with TestClient(api.app, raise_server_exceptions=False) as c:
        assert c.get("/health").status_code == 503
        assert c.post("/v1/detect", files={"image": ("x", png(), "image/png")}).status_code == 503
