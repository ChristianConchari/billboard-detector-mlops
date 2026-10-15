"""REST gateway with a fake gRPC stub (no inference service, torch or MLflow)."""

import grpc
import pytest
from fastapi.testclient import TestClient

from billboard_detector.grpc_gen import detector_pb2 as pb

PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 32  # the gateway does not decode: it only forwards


class FakeRpcError(grpc.RpcError):
    def __init__(self, code, details, trailer=()):
        self._code, self._details, self._trailer = code, details, trailer

    def trailing_metadata(self):
        return self._trailer

    def code(self):
        return self._code

    def details(self):
        return self._details


class FakeStub:
    def __init__(self):
        self.loaded = True
        self.error = None
        self.last = None

    def Detect(self, request, timeout=None, metadata=None):
        self.last = request
        if self.error:
            raise self.error
        return pb.DetectResponse(
            request_id="abc", model=pb.ModelRef(name="billboard-detector", version="1", alias="champion"),
            image=pb.ImageSize(width=1920, height=1080),
            detections=[pb.Detection(label="billboard", score=0.98729997, box=[1480.7001, 443.5999, 1818.2, 772.0])],
            latency_ms=1234.5678)

    def GetModel(self, request, timeout=None, metadata=None):
        if self.error:
            raise self.error
        if not self.loaded:
            return pb.ModelInfo(loaded=False)
        return pb.ModelInfo(loaded=True, model=pb.ModelRef(name="billboard-detector", version="1", alias="champion"),
                            labels=["billboard"], score_threshold=0.1502, metrics={"gate_map": 0.5},
                            eval_set="reviewed/test.json@abc")

    def Reload(self, request, timeout=None, metadata=None):
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

    return TestClient(app.app, headers={"X-API-Key": "user-key"})


@pytest.fixture
def admin(stub):
    import app

    return TestClient(app.app, headers={"X-API-Key": "admin-key"})


def post(client, data=PNG, ctype="image/png", **params):
    return client.post("/v1/detect", files={"image": ("x", data, ctype)}, params=params)


def test_ok_rounds_float32(client):
    r = post(client)
    assert r.status_code == 200
    assert r.json() == {
        "request_id": "abc",
        "model": {"name": "billboard-detector", "version": "1", "alias": "champion"},
        "image": {"width": 1920, "height": 1080},
        "detections": [{"label": "billboard", "score": 0.9873, "box": [1480.7, 443.6, 1818.2, 772.0]}],
        "latency_ms": 1234.6,
    }


def test_threshold_is_forwarded_only_when_given(client, stub):
    post(client)
    assert not stub.last.HasField("score_threshold")
    post(client, score_threshold=0.5)
    assert stub.last.score_threshold == pytest.approx(0.5)


@pytest.mark.parametrize("value", ["1.5", "-0.1", "abc"])
def test_invalid_threshold_422_without_calling_inference(client, stub, value):
    assert post(client, score_threshold=value).status_code == 422
    assert stub.last is None


def test_unsupported_type_415(client, stub):
    assert post(client, ctype="image/gif").status_code == 415
    assert stub.last is None


def test_too_large_413(client, stub):
    assert post(client, b"0" * (5 * 1024 * 1024 + 2)).status_code == 413
    assert stub.last is None


@pytest.mark.parametrize("code,http", [
    (grpc.StatusCode.INVALID_ARGUMENT, 422),
    (grpc.StatusCode.RESOURCE_EXHAUSTED, 413),
    (grpc.StatusCode.UNAVAILABLE, 503),
    (grpc.StatusCode.DEADLINE_EXCEEDED, 504),
    (grpc.StatusCode.INTERNAL, 502),
])
def test_grpc_errors_to_http(client, stub, code, http):
    stub.error = FakeRpcError(code, "detail")
    r = post(client)
    assert r.status_code == http
    # Expected codes keep their message; unexpected ones never leak internal details
    assert r.json()["detail"] == ("detail" if http != 502 else "inference service error")


def test_health_and_model(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}
    m = client.get("/v1/model").json()
    assert m["version"] == "1" and m["labels"] == ["billboard"] and m["score_threshold"] == 0.1502
    assert m["metrics"] == {"gate_map": 0.5} and m["eval_set"] == "reviewed/test.json@abc"


def test_reload_admin_only(client, admin):
    assert client.post("/v1/reload").status_code == 403
    assert admin.post("/v1/reload").json() == {"version": "1"}


def test_no_model_503(client, admin, stub):
    stub.loaded = False
    assert client.get("/health").status_code == 503
    assert client.get("/v1/model").status_code == 503
    assert admin.post("/v1/reload").status_code == 503
    assert client.get("/livez").json() == {"status": "ok", "model_loaded": False}


def test_inference_down_503(client, stub):
    stub.error = FakeRpcError(grpc.StatusCode.UNAVAILABLE, "no connection")
    assert client.get("/health").status_code == 503


def test_inference_down_livez_503(client, stub):
    stub.error = FakeRpcError(grpc.StatusCode.UNAVAILABLE, "no connection")
    assert client.get("/livez").status_code == 503


# --- authentication, rate limit and size -------------------------------------------------

def test_missing_key_401_invalid_key_403(stub):
    import app

    anon = TestClient(app.app)
    assert post(anon).status_code == 401
    assert anon.get("/v1/model").status_code == 401
    assert post(TestClient(app.app, headers={"X-API-Key": "other"})).status_code == 403
    assert stub.last is None
    # /health and /livez stay open for orchestrators and load balancers
    assert anon.get("/health").status_code == 200


def test_rate_limit_429(client, monkeypatch):
    import app

    monkeypatch.setattr(app, "limiter", app.RateLimiter(2))
    assert [post(client).status_code for _ in range(3)] == [200, 200, 429]
    r = post(client)
    assert r.status_code == 429 and int(r.headers["retry-after"]) >= 1


def test_rate_limiter_sliding_window():
    import app

    rl = app.RateLimiter(2)
    assert rl.check("k", now=0) is None and rl.check("k", now=1) is None
    assert rl.check("k", now=2) is not None
    assert rl.check("other", now=2) is None
    assert rl.check("k", now=61) is None


def test_large_content_length_413_without_reading_the_body(client, stub):
    r = client.post("/v1/detect", content=b"x", headers={"Content-Length": str(50 * 1024 * 1024)})
    assert r.status_code == 413 and stub.last is None


def test_missing_content_length_411(client, stub):
    def chunks():
        yield b"--b\r\n"

    r = client.post("/v1/detect", content=chunks(), headers={"Content-Type": "multipart/form-data; boundary=b"})
    assert r.status_code == 411


def test_bodiless_post_needs_no_content_length(admin):
    r = admin.post("/v1/reload")
    assert r.status_code == 200


def test_actual_format_415_arrives_as_415(client, stub):
    stub.error = FakeRpcError(grpc.StatusCode.INVALID_ARGUMENT, "unsupported image format: GIF",
                              (("http-status", "415"),))
    assert post(client).status_code == 415


def test_openapi_declares_the_api_key(client):
    # /docs shows "Authorize" and the protected routes ask for the key
    spec = client.get("/openapi.json").json()
    assert spec["components"]["securitySchemes"]["APIKeyHeader"] == {
        "type": "apiKey", "in": "header", "name": "X-API-Key", "description": "A key from GATEWAY_API_KEYS"}
    assert spec["paths"]["/v1/detect"]["post"]["security"] == [{"APIKeyHeader": []}]
    assert "security" not in spec["paths"]["/health"]["get"]
