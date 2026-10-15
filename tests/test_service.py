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
    s.set_model(FakeDetector(), {"name": "billboard-detector", "version": "1", "alias": "champion",
                                 "labels": ["billboard"], "score_threshold": 0.15, "metrics": {}})
    return s


def png(size=(64, 32)):
    return encoded(size, "PNG")


def encoded(size, fmt, **save_kw):
    buf = io.BytesIO()
    Image.new("RGB", size, "red").save(buf, fmt, **save_kw)
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
    # Campos de monitoreo: el umbral efectivo queda registrado para filtrar el drift
    assert r["score_threshold"] == 0.15 and r["threshold_is_default"] is True
    assert r["latency_ms"] >= r["inference_ms"] >= 0


def test_umbral_explicito_queda_registrado(service):
    r = service.detect(png(), score_threshold=0.5)
    assert r["score_threshold"] == 0.5 and r["threshold_is_default"] is False


def test_latencia_incluye_la_espera(service):
    import time

    r = service.detect(png(), received_at=time.perf_counter() - 0.2)
    assert r["latency_ms"] >= 200


def test_umbral_explicito(service):
    assert len(service.detect(png(), score_threshold=0.5)["detections"]) == 1


@pytest.mark.parametrize("value", [1.5, -0.1])
def test_umbral_invalido_422(service, value):
    assert status_of(service, png(), score_threshold=value) == 422


def test_tipo_no_soportado_415(service):
    assert status_of(service, b"GIF89a", content_type="image/gif") == 415


@pytest.mark.parametrize("fmt", ["GIF", "BMP", "TIFF"])
def test_formato_real_no_soportado_415_aunque_declare_jpeg(service, fmt):
    # El tipo declarado por el cliente no alcanza: se mira el formato real
    assert status_of(service, encoded((32, 32), fmt), content_type="image/jpeg") == 415
    assert status_of(service, encoded((32, 32), fmt)) == 415  # gRPC no manda tipo


def test_bomba_de_descompresion_422_sin_decodificar(service):
    # 20000x20000 en un PNG de pocos KB: PIL la marca como bomba al abrirla
    from PIL import Image

    buf = io.BytesIO()
    Image.new("1", (20000, 20000)).save(buf, "PNG")
    assert status_of(service, buf.getvalue()) == 422


def test_dimensiones_se_validan_antes_de_decodificar(service, monkeypatch):
    from PIL import ImageFile

    def no_decodificar(self):
        raise AssertionError("decodificó una imagen fuera de rango")

    monkeypatch.setattr(ImageFile.ImageFile, "load", no_decodificar)
    assert status_of(service, png((5000, 8))) == 422


def test_orientacion_exif(service):
    from PIL import Image

    exif = Image.Exif()
    exif[0x0112] = 6  # rotada 90°: se ve como 32x64
    data = encoded((64, 32), "JPEG", exif=exif.tobytes())
    assert service.detect(data)["image"] == {"width": 32, "height": 64}


def test_persist_guarda_solo_la_prediccion():
    class S3:
        def __init__(self):
            self.keys = []

        def put_object(self, Bucket, Key, Body, **kw):
            self.keys.append(Key)

    s = DetectionService()
    s._s3 = S3()
    s.persist({"request_id": "r1"})
    assert len(s._s3.keys) == 1 and s._s3.keys[0].startswith("predictions/date=") and s._s3.keys[0].endswith("/r1.json")
