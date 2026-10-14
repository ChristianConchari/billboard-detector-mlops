"""Gateway REST con un stub gRPC falso (sin inference, torch ni MLflow)."""

import grpc
import pytest
from fastapi.testclient import TestClient

from vallavision.grpc_gen import detector_pb2 as pb

PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 32  # el gateway no decodifica: solo reenvía


class FakeRpcError(grpc.RpcError):
    def __init__(self, code, details):
        self._code, self._details = code, details

    def code(self):
        return self._code

    def details(self):
        return self._details


class FakeStub:
    def __init__(self):
        self.loaded = True
        self.error = None
        self.last = None

    def Detect(self, request, timeout=None):
        self.last = request
        if self.error:
            raise self.error
        return pb.DetectResponse(
            request_id="abc", model=pb.ModelRef(name="billboard-detector", version="1", alias="champion"),
            image=pb.ImageSize(width=1920, height=1080),
            detections=[pb.Detection(label="billboard", score=0.98729997, box=[1480.7001, 443.5999, 1818.2, 772.0])],
            latency_ms=1234.5678)

    def GetModel(self, request, timeout=None):
        if self.error:
            raise self.error
        if not self.loaded:
            return pb.ModelInfo(loaded=False)
        return pb.ModelInfo(loaded=True, model=pb.ModelRef(name="billboard-detector", version="1", alias="champion"),
                            labels=["billboard"], score_threshold=0.1502, metrics={"map": 0.5})

    def Reload(self, request, timeout=None):
        return self.GetModel(request)


@pytest.fixture
def stub(monkeypatch):
    import app

    s = FakeStub()
    monkeypatch.setattr(app, "stub", s)
    return s


@pytest.fixture
def client(stub):
    import app

    return TestClient(app.app)


def post(client, data=PNG, ctype="image/png", **params):
    return client.post("/v1/detect", files={"image": ("x", data, ctype)}, params=params)


def test_ok_redondea_float32(client):
    r = post(client)
    assert r.status_code == 200
    assert r.json() == {
        "request_id": "abc",
        "model": {"name": "billboard-detector", "version": "1", "alias": "champion"},
        "image": {"width": 1920, "height": 1080},
        "detections": [{"label": "billboard", "score": 0.9873, "box": [1480.7, 443.6, 1818.2, 772.0]}],
        "latency_ms": 1234.6,
    }


def test_umbral_se_reenvia_solo_si_viene(client, stub):
    post(client)
    assert not stub.last.HasField("score_threshold")
    post(client, score_threshold=0.5)
    assert stub.last.score_threshold == pytest.approx(0.5)


@pytest.mark.parametrize("value", ["1.5", "-0.1", "abc"])
def test_umbral_invalido_422_sin_llamar_a_inference(client, stub, value):
    assert post(client, score_threshold=value).status_code == 422
    assert stub.last is None


def test_tipo_no_soportado_415(client, stub):
    assert post(client, ctype="image/gif").status_code == 415
    assert stub.last is None


def test_demasiado_grande_413(client, stub):
    assert post(client, b"0" * (5 * 1024 * 1024 + 2)).status_code == 413
    assert stub.last is None


@pytest.mark.parametrize("code,http", [
    (grpc.StatusCode.INVALID_ARGUMENT, 422),
    (grpc.StatusCode.RESOURCE_EXHAUSTED, 413),
    (grpc.StatusCode.UNAVAILABLE, 503),
    (grpc.StatusCode.DEADLINE_EXCEEDED, 504),
    (grpc.StatusCode.INTERNAL, 502),
])
def test_errores_grpc_a_http(client, stub, code, http):
    stub.error = FakeRpcError(code, "detalle")
    r = post(client)
    assert r.status_code == http
    assert r.json()["detail"] == "detalle"


def test_health_y_model(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}
    m = client.get("/v1/model").json()
    assert m["version"] == "1" and m["labels"] == ["billboard"] and m["score_threshold"] == 0.1502


def test_reload(client):
    assert client.post("/v1/reload").json() == {"version": "1"}


def test_sin_modelo_503(client, stub):
    stub.loaded = False
    assert client.get("/health").status_code == 503
    assert client.get("/v1/model").status_code == 503
    assert client.post("/v1/reload").status_code == 503


def test_inference_caido_503(client, stub):
    stub.error = FakeRpcError(grpc.StatusCode.UNAVAILABLE, "no hay conexión")
    assert client.get("/health").status_code == 503
