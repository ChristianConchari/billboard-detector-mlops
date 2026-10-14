"""Gateway REST de VallaVision (docs/contrato.md §2).

No carga el modelo: valida lo barato, reenvía la imagen por gRPC al servicio `inference` y
traduce los errores a HTTP. Sin torch, MLflow ni boto3.
"""

import json
import os
from typing import Optional

import grpc
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

from vallavision.grpc_gen import detector_pb2 as pb
from vallavision.grpc_gen import detector_pb2_grpc as pb_grpc
from vallavision.validation import MAX_BYTES, DetectionError, check_size, check_threshold, check_type

INFERENCE_ADDR = os.getenv("INFERENCE_ADDR", "inference:50051")
TIMEOUT_S = float(os.getenv("GATEWAY_TIMEOUT_S", "60"))
HEALTH_TIMEOUT_S = 5.0

# El DNS de Docker devuelve la IP de cada réplica de `inference`: round_robin las reparte
_CHANNEL_OPTIONS = [
    ("grpc.service_config", json.dumps({"loadBalancingConfig": [{"round_robin": {}}]})),
    ("grpc.max_send_message_length", 8 * 1024 * 1024),
]
stub = pb_grpc.DetectorStub(grpc.insecure_channel(f"dns:///{INFERENCE_ADDR}", options=_CHANNEL_OPTIONS))

_HTTP_STATUS = {
    grpc.StatusCode.INVALID_ARGUMENT: 422,
    grpc.StatusCode.RESOURCE_EXHAUSTED: 413,
    grpc.StatusCode.UNAVAILABLE: 503,
    grpc.StatusCode.DEADLINE_EXCEEDED: 504,
}


class ModelRef(BaseModel):
    name: str
    version: str
    alias: str


class ImageSize(BaseModel):
    width: int
    height: int


class Detection(BaseModel):
    label: str
    score: float
    box: list[float]  # [x_min, y_min, x_max, y_max] en píxeles de la imagen original


class DetectResponse(BaseModel):
    request_id: str
    model: ModelRef
    image: ImageSize
    detections: list[Detection]
    latency_ms: float


app = FastAPI(title="VallaVision", version="1")


def _http_error(e: grpc.RpcError) -> HTTPException:
    return HTTPException(_HTTP_STATUS.get(e.code(), 502), detail=e.details())


def _to_json(r):
    """DetectResponse (float32) → JSON del contrato, redondeado como antes."""
    return {
        "request_id": r.request_id,
        "model": {"name": r.model.name, "version": r.model.version, "alias": r.model.alias},
        "image": {"width": r.image.width, "height": r.image.height},
        "detections": [{"label": d.label, "score": round(d.score, 4),
                        "box": [round(v, 1) for v in d.box]} for d in r.detections],
        "latency_ms": round(r.latency_ms, 1),
    }


def _model_json(m):
    return {
        "name": m.model.name,
        "version": m.model.version,
        "alias": m.model.alias,
        "labels": list(m.labels),
        "score_threshold": round(m.score_threshold, 4),
        "metrics": dict(m.metrics),
    }


def _get_model():
    try:
        return stub.GetModel(pb.GetModelRequest(), timeout=HEALTH_TIMEOUT_S)
    except grpc.RpcError as e:
        raise _http_error(e) from None


@app.get("/health")
def health():
    if not _get_model().loaded:
        raise HTTPException(503, detail={"status": "unavailable", "model_loaded": False})
    return {"status": "ok", "model_loaded": True}


@app.get("/v1/model")
def model_info():
    m = _get_model()
    if not m.loaded:
        raise HTTPException(503, detail="no hay modelo champion cargado")
    return _model_json(m)


@app.post("/v1/reload")
def reload_model():
    """Fuerza la recarga del champion en el servicio de inferencia.

    Con varias réplicas la recibe una sola; las demás lo recogen solas (cada 30 s).
    """
    try:
        m = stub.Reload(pb.ReloadRequest(), timeout=TIMEOUT_S)
    except grpc.RpcError as e:
        raise _http_error(e) from None
    if not m.loaded:
        raise HTTPException(503, detail="no hay modelo champion cargado")
    return {"version": m.model.version}


@app.post("/v1/detect", response_model=DetectResponse)
def detect(
    image: UploadFile = File(...),
    score_threshold: Optional[float] = Query(
        None, description="0.0–1.0; por defecto el umbral calibrado del modelo"),
):
    data = image.file.read(MAX_BYTES + 1)
    try:
        check_type(image.content_type)
        check_size(len(data))
        check_threshold(score_threshold)
    except DetectionError as e:
        raise HTTPException(e.status, detail=e.message) from None

    request = pb.DetectRequest(image=data)
    if score_threshold is not None:
        request.score_threshold = score_threshold
    try:
        return _to_json(stub.Detect(request, timeout=TIMEOUT_S))
    except grpc.RpcError as e:
        raise _http_error(e) from None
