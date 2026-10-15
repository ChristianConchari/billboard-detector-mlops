"""Gateway REST de VallaVision (docs/contrato.md §2).

No carga el modelo: autentica (API key), limita la tasa y el tamaño de las requests,
valida lo barato, reenvía la imagen por gRPC al servicio `inference` y traduce los
errores a HTTP. Sin torch, MLflow ni acceso al Data Lake. Controles: docs/seguridad.md.
"""

import hmac
import json
import os
import threading
import time
from collections import defaultdict, deque
from typing import Optional

import grpc
from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from vallavision.grpc_gen import detector_pb2 as pb
from vallavision.grpc_gen import detector_pb2_grpc as pb_grpc
from vallavision.validation import MAX_BYTES, DetectionError, check_size, check_threshold, check_type

INFERENCE_ADDR = os.getenv("INFERENCE_ADDR", "inference:50051")
INFERENCE_TOKEN = os.getenv("INFERENCE_TOKEN", "")
TIMEOUT_S = float(os.getenv("GATEWAY_TIMEOUT_S", "60"))
HEALTH_TIMEOUT_S = 5.0
# Por clave: muy por encima de lo que atiende una réplica (~2,4 req/s); corta abusos, no tráfico legítimo
RATE_LIMIT_PER_MIN = int(os.getenv("GATEWAY_RATE_LIMIT_PER_MIN", "600"))
# Margen para los encabezados multipart sobre el tamaño máximo de la imagen
MAX_BODY_BYTES = MAX_BYTES + 64 * 1024


def _keys(name):
    return {k.strip() for k in os.getenv(name, "").split(",") if k.strip()}


API_KEYS = _keys("GATEWAY_API_KEYS")
ADMIN_KEYS = _keys("GATEWAY_ADMIN_KEYS")
AUTH_DISABLED = os.getenv("GATEWAY_AUTH_DISABLED", "").lower() == "true"
if not (API_KEYS or ADMIN_KEYS or AUTH_DISABLED):
    raise RuntimeError("definir GATEWAY_API_KEYS (scripts/generar_env.sh) o GATEWAY_AUTH_DISABLED=true")

# El DNS de Docker devuelve la IP de cada réplica de `inference`: round_robin las reparte
_CHANNEL_OPTIONS = [
    ("grpc.service_config", json.dumps({"loadBalancingConfig": [{"round_robin": {}}]})),
    ("grpc.max_send_message_length", 8 * 1024 * 1024),
]
stub = pb_grpc.DetectorStub(grpc.insecure_channel(f"dns:///{INFERENCE_ADDR}", options=_CHANNEL_OPTIONS))
_METADATA = [("authorization", f"Bearer {INFERENCE_TOKEN}")] if INFERENCE_TOKEN else None

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


# --- seguridad: API key + rate limit por clave ---------------------------------------

class RateLimiter:
    """Ventana deslizante de 60 s por clave, en memoria (una réplica de gateway)."""

    def __init__(self, per_minute):
        self.per_minute = per_minute
        self._hits = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key, now=None):
        """None si pasa; si no, segundos hasta que se libere un lugar."""
        now = time.monotonic() if now is None else now
        with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] >= 60:
                hits.popleft()
            if len(hits) >= self.per_minute:
                return max(1, int(60 - (now - hits[0])) + 1)
            hits.append(now)
            return None


limiter = RateLimiter(RATE_LIMIT_PER_MIN)


def _match(sent, keys):
    return any(hmac.compare_digest(sent.encode(), k.encode()) for k in keys)


def _authenticate(request: Request, admin: bool):
    if AUTH_DISABLED:
        return "anon"
    sent = request.headers.get("x-api-key", "")
    if not sent:
        raise HTTPException(401, detail="falta X-API-Key", headers={"WWW-Authenticate": "ApiKey"})
    if _match(sent, ADMIN_KEYS):
        key = sent
    elif not admin and _match(sent, API_KEYS):
        key = sent
    else:
        raise HTTPException(403, detail="API key inválida o sin permiso")
    retry = limiter.check(key)
    if retry is not None:
        raise HTTPException(429, detail="demasiadas requests", headers={"Retry-After": str(retry)})
    return key


def require_key(request: Request):
    return _authenticate(request, admin=False)


def require_admin(request: Request):
    return _authenticate(request, admin=True)


app = FastAPI(title="VallaVision", version="1")


@app.middleware("http")
async def limit_body(request: Request, call_next):
    """Corta por Content-Length antes de leer el cuerpo: un upload gigante no llega a disco."""
    if request.method in ("POST", "PUT", "PATCH"):
        length = request.headers.get("content-length")
        if length is None:
            # Sin Content-Length y sin cuerpo (p. ej. /v1/reload) está bien; un cuerpo chunked
            # sin largo declarado no se puede acotar antes de leerlo
            if "transfer-encoding" in request.headers:
                return JSONResponse({"detail": "falta Content-Length"}, status_code=411)
            return await call_next(request)
        if not length.isdigit() or int(length) > MAX_BODY_BYTES:
            return JSONResponse({"detail": f"cuerpo > {MAX_BODY_BYTES} bytes"}, status_code=413)
    return await call_next(request)


def _call(method, request, timeout):
    return getattr(stub, method)(request, timeout=timeout, metadata=_METADATA)


def _contract_status(e):
    """Código HTTP del contrato que manda inference en el trailer `http-status` (415 vs 422)."""
    try:
        trailer = dict(e.trailing_metadata() or ())
    except Exception:  # noqa: BLE001 — errores de canal no traen trailer
        return None
    value = trailer.get("http-status", "")
    return int(value) if value in ("413", "415", "422") else None


def _http_error(e: grpc.RpcError) -> HTTPException:
    status = _contract_status(e) or _HTTP_STATUS.get(e.code())
    if status is None:
        # Errores no previstos: no reenviar detalles internos al cliente
        print(f"inference respondió {e.code().name}: {e.details()}", flush=True)
        return HTTPException(502, detail="error del servicio de inferencia")
    return HTTPException(status, detail=e.details())


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
        "eval_set": m.eval_set,
    }


def _get_model():
    try:
        return _call("GetModel", pb.GetModelRequest(), HEALTH_TIMEOUT_S)
    except grpc.RpcError as e:
        raise _http_error(e) from None


@app.get("/livez")
def livez():
    """Vivo si el gateway alcanza a `inference`, tenga o no champion (healthcheck de Docker)."""
    return {"status": "ok", "model_loaded": _get_model().loaded}


@app.get("/health")
def health():
    if not _get_model().loaded:
        raise HTTPException(503, detail={"status": "unavailable", "model_loaded": False})
    return {"status": "ok", "model_loaded": True}


@app.get("/v1/model", dependencies=[Depends(require_key)])
def model_info():
    m = _get_model()
    if not m.loaded:
        raise HTTPException(503, detail="no hay modelo champion cargado")
    return _model_json(m)


@app.post("/v1/reload", dependencies=[Depends(require_admin)])
def reload_model():
    """Fuerza la recarga del champion en el servicio de inferencia (solo claves admin).

    Con varias réplicas la recibe una sola; las demás lo recogen solas (cada 30 s).
    """
    try:
        m = _call("Reload", pb.ReloadRequest(), TIMEOUT_S)
    except grpc.RpcError as e:
        raise _http_error(e) from None
    if not m.loaded:
        raise HTTPException(503, detail="no hay modelo champion cargado")
    return {"version": m.model.version}


@app.post("/v1/detect", response_model=DetectResponse, dependencies=[Depends(require_key)])
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
        return _to_json(_call("Detect", request, TIMEOUT_S))
    except grpc.RpcError as e:
        raise _http_error(e) from None
