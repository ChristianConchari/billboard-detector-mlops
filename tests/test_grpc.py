"""Servidor gRPC real (puerto efímero) con un servicio falso."""

import io
import grpc
import pytest
from PIL import Image

from vallavision.grpc_gen import detector_pb2 as pb
from vallavision.grpc_gen import detector_pb2_grpc as pb_grpc
from vallavision.grpc_server import build_server
from vallavision.service import DetectionService
from tests.test_service import FakeDetector


@pytest.fixture
def stub():
    service = DetectionService()
    service.set_model(FakeDetector(), {"name": "billboard-detector", "version": "1", "alias": "champion",
                                       "labels": ["billboard"], "score_threshold": 0.15,
                                       "metrics": {"gate_map": 0.5}, "eval_set": "reviewed/test.json@abc"})
    service.persisted = []
    service.persist = lambda r: service.persisted.append(r["request_id"])
    server = build_server(service, max_workers=4)
    port = server.add_insecure_port("localhost:0")
    server.start()
    channel = grpc.insecure_channel(f"localhost:{port}")
    yield pb_grpc.DetectorStub(channel), service
    channel.close()
    server.stop(0)


def png(size=(64, 32), fmt="PNG"):
    buf = io.BytesIO()
    Image.new("RGB", size).save(buf, fmt)
    return buf.getvalue()


def test_detect(stub):
    s, _ = stub
    r = s.Detect(pb.DetectRequest(image=png()))
    assert r.image.width == 64 and [d.score for d in r.detections] == pytest.approx([0.9, 0.3])
    assert r.model.alias == "champion"


def test_detect_con_umbral(stub):
    s, _ = stub
    assert len(s.Detect(pb.DetectRequest(image=png(), score_threshold=0.5)).detections) == 1


def test_frame_id_vuelve_en_la_respuesta(stub):
    s, _ = stub
    assert s.Detect(pb.DetectRequest(image=png(), frame_id="cam1-7")).frame_id == "cam1-7"


def test_bmp_por_grpc_415_invalid_argument(stub):
    s, _ = stub
    with pytest.raises(grpc.RpcError) as e:
        s.Detect(pb.DetectRequest(image=png(fmt="BMP")))
    assert e.value.code() == grpc.StatusCode.INVALID_ARGUMENT and "formato" in e.value.details()
    assert dict(e.value.trailing_metadata())["http-status"] == "415"


def test_deadline_vencido_no_persiste(stub):
    import threading
    import time

    s, service = stub
    gate = threading.Event()
    real = service.detect

    def lento(*a, **kw):
        gate.wait(2)
        return real(*a, **kw)

    service.detect = lento
    with pytest.raises(grpc.RpcError) as e:
        s.Detect(pb.DetectRequest(image=png()), timeout=0.2)
    assert e.value.code() == grpc.StatusCode.DEADLINE_EXCEEDED
    gate.set()
    time.sleep(0.3)
    assert service.persisted == []


def test_error_interno_no_filtra_detalles(stub):
    s, service = stub
    service.detect = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("secreto: /ruta/interna"))
    with pytest.raises(grpc.RpcError) as e:
        s.Detect(pb.DetectRequest(image=png()))
    assert e.value.code() == grpc.StatusCode.INTERNAL and "secreto" not in e.value.details()


def test_token_obligatorio():
    from tests.test_service import FakeDetector as Fake

    service = DetectionService()
    service.set_model(Fake(), {"name": "m", "version": "1", "alias": "champion", "labels": [],
                               "score_threshold": 0.1, "metrics": {}})
    service.persist = lambda r: None
    server = build_server(service, token="s3cr3t", max_workers=2)
    port = server.add_insecure_port("localhost:0")
    server.start()
    try:
        with grpc.insecure_channel(f"localhost:{port}") as ch:
            st = pb_grpc.DetectorStub(ch)
            for md in (None, [("authorization", "Bearer otro")]):
                with pytest.raises(grpc.RpcError) as e:
                    st.GetModel(pb.GetModelRequest(), metadata=md)
                assert e.value.code() == grpc.StatusCode.UNAUTHENTICATED
            with pytest.raises(grpc.RpcError) as e:
                st.Detect(pb.DetectRequest(image=png()))
            assert e.value.code() == grpc.StatusCode.UNAUTHENTICATED
            ok = st.GetModel(pb.GetModelRequest(), metadata=[("authorization", "Bearer s3cr3t")])
            assert ok.loaded
    finally:
        server.stop(0)


@pytest.mark.parametrize("request_,code", [
    (pb.DetectRequest(image=b"no soy una imagen"), grpc.StatusCode.INVALID_ARGUMENT),
    (pb.DetectRequest(image=b"0" * (5 * 1024 * 1024 + 2)), grpc.StatusCode.RESOURCE_EXHAUSTED),
    (pb.DetectRequest(image=b"", score_threshold=3.0), grpc.StatusCode.INVALID_ARGUMENT),
])
def test_errores(stub, request_, code):
    s, _ = stub
    with pytest.raises(grpc.RpcError) as e:
        s.Detect(request_)
    assert e.value.code() == code


def test_sin_modelo_unavailable(stub):
    s, service = stub
    service._model = None
    with pytest.raises(grpc.RpcError) as e:
        s.Detect(pb.DetectRequest(image=png()))
    assert e.value.code() == grpc.StatusCode.UNAVAILABLE
    assert s.GetModel(pb.GetModelRequest()).loaded is False


def test_get_model(stub):
    s, _ = stub
    m = s.GetModel(pb.GetModelRequest())
    assert m.loaded and m.model.version == "1" and list(m.labels) == ["billboard"]
    assert m.score_threshold == pytest.approx(0.15) and dict(m.metrics) == {"gate_map": 0.5}
    assert m.eval_set == "reviewed/test.json@abc"


def test_reload_sin_mlflow_unavailable(stub):
    s, service = stub
    service.load_champion = lambda: (_ for _ in ()).throw(RuntimeError("mlflow caído"))
    with pytest.raises(grpc.RpcError) as e:
        s.Reload(pb.ReloadRequest())
    assert e.value.code() == grpc.StatusCode.UNAVAILABLE
