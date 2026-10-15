"""REST gateway of the billboard detector (docs/contrato.md §2).

It does not load the model: it authenticates (API key), rate-limits and caps request size,
runs the cheap checks, forwards the image to the `inference` service over gRPC and maps
errors to HTTP. No torch, no MLflow and no data-lake access. Controls: docs/seguridad.md.
"""

import hmac
import json
import os
import threading
import time
from collections import defaultdict, deque
from typing import Optional

import grpc
from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, Security, UploadFile
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

from billboard_detector.grpc_gen import detector_pb2 as pb
from billboard_detector.grpc_gen import detector_pb2_grpc as pb_grpc
from billboard_detector.validation import MAX_BYTES, DetectionError, check_size, check_threshold, check_type

INFERENCE_ADDR = os.getenv("INFERENCE_ADDR", "inference:50051")
INFERENCE_TOKEN = os.getenv("INFERENCE_TOKEN", "")
TIMEOUT_S = float(os.getenv("GATEWAY_TIMEOUT_S", "60"))
HEALTH_TIMEOUT_S = 5.0
# Per key: far above what one replica serves (~2.4 req/s); it stops abuse, not legitimate traffic
RATE_LIMIT_PER_MIN = int(os.getenv("GATEWAY_RATE_LIMIT_PER_MIN", "600"))
# Allowance for the multipart headers on top of the maximum image size
MAX_BODY_BYTES = MAX_BYTES + 64 * 1024


def _keys(name):
    return {k.strip() for k in os.getenv(name, "").split(",") if k.strip()}


API_KEYS = _keys("GATEWAY_API_KEYS")
ADMIN_KEYS = _keys("GATEWAY_ADMIN_KEYS")
AUTH_DISABLED = os.getenv("GATEWAY_AUTH_DISABLED", "").lower() == "true"
if not (API_KEYS or ADMIN_KEYS or AUTH_DISABLED):
    raise RuntimeError("set GATEWAY_API_KEYS (scripts/generate_env.sh) or GATEWAY_AUTH_DISABLED=true")

# Docker's DNS returns the IP of every `inference` replica: round_robin spreads the calls
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
    box: list[float]  # [x_min, y_min, x_max, y_max] in pixels of the original image


class DetectResponse(BaseModel):
    request_id: str
    model: ModelRef
    image: ImageSize
    detections: list[Detection]
    latency_ms: float


# --- security: API key + per-key rate limit ---------------------------------------------

class RateLimiter:
    """60 s sliding window per key, in memory (a single gateway replica)."""

    def __init__(self, per_minute):
        self.per_minute = per_minute
        self._hits = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key, now=None):
        """None if allowed; otherwise, seconds until a slot frees up."""
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
        raise HTTPException(401, detail="missing X-API-Key", headers={"WWW-Authenticate": "ApiKey"})
    if _match(sent, ADMIN_KEYS):
        key = sent
    elif not admin and _match(sent, API_KEYS):
        key = sent
    else:
        raise HTTPException(403, detail="invalid API key or insufficient permissions")
    retry = limiter.check(key)
    if retry is not None:
        raise HTTPException(429, detail="too many requests", headers={"Retry-After": str(retry)})
    return key


# Declaring the scheme makes /docs show the "Authorize" button to enter the key once;
# _authenticate does the validation (auto_error=False: the 401/403 are the contract's)
api_key_scheme = APIKeyHeader(name="X-API-Key", auto_error=False, description="A key from GATEWAY_API_KEYS")


def require_key(request: Request, _: Optional[str] = Security(api_key_scheme)):
    return _authenticate(request, admin=False)


def require_admin(request: Request, _: Optional[str] = Security(api_key_scheme)):
    return _authenticate(request, admin=True)


app = FastAPI(title="Detector de Vallas Publicitarias", version="1")


@app.middleware("http")
async def limit_body(request: Request, call_next):
    """Rejects by Content-Length before reading the body: a huge upload never reaches disk."""
    if request.method in ("POST", "PUT", "PATCH"):
        length = request.headers.get("content-length")
        if length is None:
            # No Content-Length and no body (e.g. /v1/reload) is fine; a chunked body with no
            # declared length cannot be capped before reading it
            if "transfer-encoding" in request.headers:
                return JSONResponse({"detail": "missing Content-Length"}, status_code=411)
            return await call_next(request)
        if not length.isdigit() or int(length) > MAX_BODY_BYTES:
            return JSONResponse({"detail": f"body larger than {MAX_BODY_BYTES} bytes"}, status_code=413)
    return await call_next(request)


def _call(method, request, timeout):
    return getattr(stub, method)(request, timeout=timeout, metadata=_METADATA)


def _contract_status(e):
    """Contract HTTP status sent by inference in the `http-status` trailer (415 vs 422)."""
    try:
        trailer = dict(e.trailing_metadata() or ())
    except Exception:  # noqa: BLE001 — channel errors carry no trailer
        return None
    value = trailer.get("http-status", "")
    return int(value) if value in ("413", "415", "422") else None


def _http_error(e: grpc.RpcError) -> HTTPException:
    status = _contract_status(e) or _HTTP_STATUS.get(e.code())
    if status is None:
        # Unexpected errors: never forward internal details to the client
        print(f"inference answered {e.code().name}: {e.details()}", flush=True)
        return HTTPException(502, detail="inference service error")
    return HTTPException(status, detail=e.details())


def _to_json(r):
    """DetectResponse (float32) → the contract's JSON, rounded."""
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
    """Alive if the gateway reaches `inference`, with or without a champion (Docker healthcheck)."""
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
        raise HTTPException(503, detail="no champion model loaded")
    return _model_json(m)


@app.post("/v1/reload", dependencies=[Depends(require_admin)])
def reload_model():
    """Forces the inference service to reload the champion (admin keys only).

    With several replicas only one receives it; the others pick it up on their own (every 30 s).
    """
    try:
        m = _call("Reload", pb.ReloadRequest(), TIMEOUT_S)
    except grpc.RpcError as e:
        raise _http_error(e) from None
    if not m.loaded:
        raise HTTPException(503, detail="no champion model loaded")
    return {"version": m.model.version}


@app.post("/v1/detect", response_model=DetectResponse, dependencies=[Depends(require_key)])
def detect(
    image: UploadFile = File(...),
    score_threshold: Optional[float] = Query(
        None, description="0.0–1.0; defaults to the model's calibrated threshold"),
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
