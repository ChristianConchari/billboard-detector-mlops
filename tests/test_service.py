"""DetectionService validation and threshold handling with a fake detector (no torch, no MLflow)."""

import io

import pytest
from PIL import Image

from billboard_detector.service import DetectionError, DetectionService, ModelNotLoaded


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


def test_ok_uses_the_calibrated_threshold(service):
    r = service.detect(png(), content_type="image/png")
    assert r["image"] == {"width": 64, "height": 32}
    assert [d["score"] for d in r["detections"]] == [0.9, 0.3]  # 0.1 < 0.15
    assert r["model"]["alias"] == "champion"
    assert set(r["image_stats"]) == {"brightness", "contrast"}
    # Monitoring fields: the effective threshold is recorded so drift can filter on it
    assert r["score_threshold"] == 0.15 and r["threshold_is_default"] is True
    assert r["latency_ms"] >= r["inference_ms"] >= 0


def test_explicit_threshold_is_recorded(service):
    r = service.detect(png(), score_threshold=0.5)
    assert r["score_threshold"] == 0.5 and r["threshold_is_default"] is False


def test_latency_includes_the_wait(service):
    import time

    r = service.detect(png(), received_at=time.perf_counter() - 0.2)
    assert r["latency_ms"] >= 200


def test_explicit_threshold(service):
    assert len(service.detect(png(), score_threshold=0.5)["detections"]) == 1


@pytest.mark.parametrize("value", [1.5, -0.1])
def test_invalid_threshold_422(service, value):
    assert status_of(service, png(), score_threshold=value) == 422


def test_unsupported_content_type_415(service):
    assert status_of(service, b"GIF89a", content_type="image/gif") == 415


@pytest.mark.parametrize("fmt", ["GIF", "BMP", "TIFF"])
def test_actual_format_415_even_when_declared_jpeg(service, fmt):
    # The type declared by the client is not enough: the actual file format is checked
    assert status_of(service, encoded((32, 32), fmt), content_type="image/jpeg") == 415
    assert status_of(service, encoded((32, 32), fmt)) == 415  # gRPC sends no content type


def test_decompression_bomb_422_without_decoding(service):
    # 20000x20000 in a PNG of a few KB: PIL flags it as a bomb when opening it
    buf = io.BytesIO()
    Image.new("1", (20000, 20000)).save(buf, "PNG")
    assert status_of(service, buf.getvalue()) == 422


def test_dimensions_are_checked_before_decoding(service, monkeypatch):
    from PIL import ImageFile

    def must_not_decode(self):
        raise AssertionError("decoded an out-of-range image")

    monkeypatch.setattr(ImageFile.ImageFile, "load", must_not_decode)
    assert status_of(service, png((5000, 8))) == 422


def test_exif_orientation(service):
    exif = Image.Exif()
    exif[0x0112] = 6  # rotated 90°: displayed as 32x64
    data = encoded((64, 32), "JPEG", exif=exif.tobytes())
    assert service.detect(data)["image"] == {"width": 32, "height": 64}


def test_persist_stores_only_the_prediction():
    class S3:
        def __init__(self):
            self.keys = []

        def put_object(self, Bucket, Key, Body, **kw):
            self.keys.append(Key)

    s = DetectionService()
    s._s3 = S3()
    s.persist({"request_id": "r1"})
    assert len(s._s3.keys) == 1 and s._s3.keys[0].startswith("predictions/date=") and s._s3.keys[0].endswith("/r1.json")


def test_no_model():
    with pytest.raises(ModelNotLoaded):
        DetectionService().detect(png())
