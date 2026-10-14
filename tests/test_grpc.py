"""Servidor gRPC real (puerto efímero) con un servicio falso."""

import io
from concurrent import futures

import grpc
import pytest
from PIL import Image

from vallavision.grpc_gen import detector_pb2 as pb
from vallavision.grpc_gen import detector_pb2_grpc as pb_grpc
from vallavision.grpc_server import DetectorServicer
from vallavision.service import DetectionService
from tests.test_service import FakeDetector


@pytest.fixture
def stub():
    service = DetectionService()
    service.detector = FakeDetector()
    service.info = {"name": "billboard-detector", "version": "1", "alias": "champion",
                    "labels": ["billboard"], "score_threshold": 0.15, "metrics": {"map": 0.5}}
    service.persist = lambda r: None
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    pb_grpc.add_DetectorServicer_to_server(DetectorServicer(service), server)
    port = server.add_insecure_port("localhost:0")
    server.start()
    channel = grpc.insecure_channel(f"localhost:{port}")
    yield pb_grpc.DetectorStub(channel), service
    channel.close()
    server.stop(0)


def png(size=(64, 32)):
    buf = io.BytesIO()
    Image.new("RGB", size).save(buf, "PNG")
    return buf.getvalue()


def test_detect(stub):
    s, _ = stub
    r = s.Detect(pb.DetectRequest(image=png()))
    assert r.image.width == 64 and [d.score for d in r.detections] == pytest.approx([0.9, 0.3])
    assert r.model.alias == "champion"


def test_detect_con_umbral(stub):
    s, _ = stub
    assert len(s.Detect(pb.DetectRequest(image=png(), score_threshold=0.5)).detections) == 1


def test_detect_stream_en_orden(stub):
    s, _ = stub
    batch = pb.DetectBatchRequest(frames=[pb.DetectRequest(image=png((w, 10))) for w in (10, 20, 30)])
    assert [r.image.width for r in s.DetectStream(batch)] == [10, 20, 30]


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
    service.detector = None
    with pytest.raises(grpc.RpcError) as e:
        s.Detect(pb.DetectRequest(image=png()))
    assert e.value.code() == grpc.StatusCode.UNAVAILABLE
    assert s.GetModel(pb.GetModelRequest()).loaded is False


def test_get_model(stub):
    s, _ = stub
    m = s.GetModel(pb.GetModelRequest())
    assert m.loaded and m.model.version == "1" and list(m.labels) == ["billboard"]
    assert m.score_threshold == pytest.approx(0.15) and dict(m.metrics) == {"map": 0.5}


def test_reload_sin_mlflow_unavailable(stub):
    s, service = stub
    service.load_champion = lambda: (_ for _ in ()).throw(RuntimeError("mlflow caído"))
    with pytest.raises(grpc.RpcError) as e:
        s.Reload(pb.ReloadRequest())
    assert e.value.code() == grpc.StatusCode.UNAVAILABLE
