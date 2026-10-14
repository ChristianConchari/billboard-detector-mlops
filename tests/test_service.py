"""Validación y umbral de DetectionService con un detector falso (sin torch ni MLflow)."""

import io

import pytest
from PIL import Image

from vallavision.service import DetectionError, DetectionService, ModelNotLoaded


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
def service():
    s = DetectionService()
    s.detector = FakeDetector()
    s.info = {"name": "billboard-detector", "version": "1", "alias": "champion",
              "labels": ["billboard"], "score_threshold": 0.15, "metrics": {}}
    return s


def png(size=(64, 32)):
    buf = io.BytesIO()
    Image.new("RGB", size).save(buf, "PNG")
    return buf.getvalue()


def status_of(service, data, **kw):
    with pytest.raises(DetectionError) as e:
        service.detect(data, **kw)
    return e.value.status


def test_ok_usa_umbral_calibrado(service):
    r = service.detect(png(), content_type="image/png")
    assert r["image"] == {"width": 64, "height": 32}
    assert [d["score"] for d in r["detections"]] == [0.9, 0.3]  # 0.1 < 0.15
    assert r["model"]["alias"] == "champion"
    assert set(r["image_stats"]) == {"brightness", "contrast"}


def test_umbral_explicito(service):
    assert len(service.detect(png(), score_threshold=0.5)["detections"]) == 1


@pytest.mark.parametrize("value", [1.5, -0.1])
def test_umbral_invalido_422(service, value):
    assert status_of(service, png(), score_threshold=value) == 422


def test_tipo_no_soportado_415(service):
    assert status_of(service, b"GIF89a", content_type="image/gif") == 415


def test_demasiado_grande_413(service):
    assert status_of(service, b"0" * (5 * 1024 * 1024 + 2)) == 413


def test_imagen_ilegible_422(service):
    assert status_of(service, b"no soy una imagen") == 422


def test_dimensiones_fuera_de_rango_422(service):
    assert status_of(service, png((4097, 8))) == 422


def test_sin_modelo():
    with pytest.raises(ModelNotLoaded):
        DetectionService().detect(png())
