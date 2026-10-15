"""Gateway REST con un stub gRPC falso (sin inference, torch ni MLflow)."""

import grpc
import pytest
from fastapi.testclient import TestClient

from vallavision.grpc_gen import detector_pb2 as pb

PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 32  # el gateway no decodifica: solo reenvía


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

    return TestClient(app.app, headers={"X-API-Key": "clave-usuario"})


@pytest.fixture
def admin(stub):
    import app

    return TestClient(app.app, headers={"X-API-Key": "clave-admin"})


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
    # Los códigos previstos conservan el mensaje; los no previstos no filtran detalles internos
    assert r.json()["detail"] == ("detalle" if http != 502 else "error del servicio de inferencia")


def test_health_y_model(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}
    m = client.get("/v1/model").json()
    assert m["version"] == "1" and m["labels"] == ["billboard"] and m["score_threshold"] == 0.1502
    assert m["metrics"] == {"gate_map": 0.5} and m["eval_set"] == "reviewed/test.json@abc"


def test_reload_solo_admin(client, admin):
    assert client.post("/v1/reload").status_code == 403
    assert admin.post("/v1/reload").json() == {"version": "1"}


def test_sin_modelo_503(client, admin, stub):
    stub.loaded = False
    assert client.get("/health").status_code == 503
    assert client.get("/v1/model").status_code == 503
    assert admin.post("/v1/reload").status_code == 503
    assert client.get("/livez").json() == {"status": "ok", "model_loaded": False}


def test_inference_caido_503(client, stub):
    stub.error = FakeRpcError(grpc.StatusCode.UNAVAILABLE, "no hay conexión")
    assert client.get("/health").status_code == 503


def test_inference_caido_livez_503(client, stub):
    stub.error = FakeRpcError(grpc.StatusCode.UNAVAILABLE, "no hay conexión")
    assert client.get("/livez").status_code == 503


# --- autenticación, rate limit y tamaño ---------------------------------------------

def test_sin_clave_401_clave_invalida_403(stub):
    import app

    anon = TestClient(app.app)
    assert post(anon).status_code == 401
    assert anon.get("/v1/model").status_code == 401
    assert post(TestClient(app.app, headers={"X-API-Key": "otra"})).status_code == 403
    assert stub.last is None
    # /health y /livez quedan abiertos para orquestadores y balanceadores
    assert anon.get("/health").status_code == 200


def test_rate_limit_429(client, monkeypatch):
    import app

    monkeypatch.setattr(app, "limiter", app.RateLimiter(2))
    assert [post(client).status_code for _ in range(3)] == [200, 200, 429]
    r = post(client)
    assert r.status_code == 429 and int(r.headers["retry-after"]) >= 1


def test_rate_limiter_ventana_deslizante():
    import app

    rl = app.RateLimiter(2)
    assert rl.check("k", now=0) is None and rl.check("k", now=1) is None
    assert rl.check("k", now=2) is not None
    assert rl.check("otra", now=2) is None
    assert rl.check("k", now=61) is None


def test_content_length_grande_413_sin_leer_cuerpo(client, stub):
    r = client.post("/v1/detect", content=b"x", headers={"Content-Length": str(50 * 1024 * 1024)})
    assert r.status_code == 413 and stub.last is None


def test_sin_content_length_411(client, stub):
    def chunks():
        yield b"--b\r\n"

    r = client.post("/v1/detect", content=chunks(), headers={"Content-Type": "multipart/form-data; boundary=b"})
    assert r.status_code == 411


def test_post_sin_cuerpo_no_exige_content_length(admin):
    r = admin.post("/v1/reload")
    assert r.status_code == 200


# --- GraphQL ----------------------------------------------------------------------------

def gql(client, query, **variables):
    return client.post("/graphql", json={"query": query, "variables": variables})


def test_graphql_modelo(client):
    r = gql(client, "{ modelo { version alias umbral conjuntoEvaluacion metricas { nombre valor } } }")
    assert r.status_code == 200, r.text
    assert r.json()["data"]["modelo"] == {"version": "1", "alias": "champion", "umbral": 0.1502,
                                          "conjuntoEvaluacion": "reviewed/test.json@abc",
                                          "metricas": [{"nombre": "gate_map", "valor": 0.5}]}


def test_graphql_requiere_clave(stub):
    import app

    assert gql(TestClient(app.app), "{ modelo { version } }").status_code == 401


def test_graphql_sin_introspeccion(client):
    r = gql(client, "{ __schema { types { name } } }")
    assert r.json().get("errors") and "introspection" in r.json()["errors"][0]["message"].lower()


def test_graphql_limita_profundidad(stub):
    import app

    q = "{ modelo { metricas { nombre } } }"  # lo más profundo que admite el esquema
    assert app.build_schema().execute_sync(q).errors is None
    errors = app.build_schema(max_depth=app.GRAPHQL_MAX_DEPTH - 1).execute_sync(q).errors
    assert errors and "exceeds maximum operation depth" in errors[0].message


def test_graphql_versiones(client, monkeypatch):
    import app

    def fake(path, **params):
        if path == "model-versions/search":
            return {"model_versions": [
                {"version": "1", "aliases": ["champion"], "creation_timestamp": "0",
                 "tags": [{"key": "weights_uri", "value": "s3://data/models/billboard-detector/v1/"},
                          {"key": "eval_set", "value": "reviewed/test.json@abc"},
                          {"key": "eval_run_id", "value": "r1"}]},
                {"version": "2", "aliases": ["challenger"], "creation_timestamp": "0", "tags": []}]}
        return {"run": {"data": {"metrics": [{"key": "gate_map", "value": 0.64}]}}}

    monkeypatch.setattr(app, "_mlflow_get", fake)
    data = gql(client, "{ versiones { version aliases gateMap conjuntoEvaluacion } }").json()["data"]
    assert data["versiones"] == [
        {"version": "2", "aliases": ["challenger"], "gateMap": None, "conjuntoEvaluacion": ""},
        {"version": "1", "aliases": ["champion"], "gateMap": 0.64, "conjuntoEvaluacion": "reviewed/test.json@abc"}]


def test_graphql_predicciones_limita_y_filtra(client, monkeypatch):
    import io
    import json

    import app

    record = {"request_id": "r1", "model": {"version": "1"}, "image": {"width": 10, "height": 5},
              "latency_ms": 12.0, "created_at": "2026-10-14T00:00:00+00:00",
              "detections": [{"label": "billboard", "score": 0.9, "box": [0, 0, 1, 1]},
                             {"label": "billboard", "score": 0.2, "box": [0, 0, 1, 1]}]}

    class S3:
        def list_objects_v2(self, Bucket, Prefix, MaxKeys):
            self.max_keys = MaxKeys
            return {"Contents": [{"Key": Prefix + "r1.json"}]}

        def get_object(self, Bucket, Key):
            return {"Body": io.BytesIO(json.dumps(record).encode())}

    s3 = S3()
    monkeypatch.setattr(app, "_s3", lambda: s3)
    q = "query($f: String!) { predicciones(fecha: $f, limite: 5000, scoreMinimo: 0.5) { requestId detecciones { score } } }"
    data = gql(client, q, f="2026-10-14").json()["data"]
    assert data["predicciones"] == [{"requestId": "r1", "detecciones": [{"score": 0.9}]}]
    assert s3.max_keys == app.GRAPHQL_MAX_ITEMS
    assert gql(client, q, f="../../etc").json()["errors"]


def test_415_del_formato_real_llega_como_415(client, stub):
    stub.error = FakeRpcError(grpc.StatusCode.INVALID_ARGUMENT, "formato no soportado: GIF", (("http-status", "415"),))
    assert post(client).status_code == 415
