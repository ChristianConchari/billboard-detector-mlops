"""Inference service of the billboard detector: the only owner of the model (docs/contrato.md §3).

Its clients are the REST gateway and the streaming consumer (it is an internal service). It
scales with replicas: each one loads its own copy of the model and runs `INFERENCE_CONCURRENCY`
forward passes at a time (1 by default: on CPU a forward pass already uses every thread it has).

When `INFERENCE_TOKEN` is set, every call must carry the metadata
`authorization: Bearer <token>` (interceptor; see docs/seguridad.md).
"""

import hmac
import os
import threading
import time
from concurrent import futures

import grpc

from billboard_detector.grpc_gen import detector_pb2 as pb
from billboard_detector.grpc_gen import detector_pb2_grpc as pb_grpc
from billboard_detector.service import DetectionError, DetectionService, ModelNotLoaded

PORT = int(os.getenv("GRPC_PORT", "50051"))
CONCURRENCY = int(os.getenv("INFERENCE_CONCURRENCY", "1"))
RELOAD_RETRY_SECONDS = 30
# Closing old connections forces clients (the gateway) to re-resolve DNS and discover new
# replicas: without this, a channel created with 1 replica never uses the 2nd one
CONNECTION_AGE_MS = int(os.getenv("GRPC_MAX_CONNECTION_AGE_S", "30")) * 1000
MAX_MESSAGE_BYTES = 8 * 1024 * 1024
_SLOT_POLL_S = 0.1

_STATUS = {
    413: grpc.StatusCode.RESOURCE_EXHAUSTED,
    415: grpc.StatusCode.INVALID_ARGUMENT,
    422: grpc.StatusCode.INVALID_ARGUMENT,
}


class Cancelled(Exception):
    """The client left (deadline or cancellation): nothing is inferred or persisted."""


def to_proto(r, frame_id=""):
    return pb.DetectResponse(
        request_id=r["request_id"],
        model=pb.ModelRef(**r["model"]),
        image=pb.ImageSize(**r["image"]),
        detections=[pb.Detection(label=d["label"], score=d["score"], box=d["box"])
                    for d in r["detections"]],
        latency_ms=r["latency_ms"],
        frame_id=frame_id,
    )


def model_info(service):
    info = service.info
    if info is None:
        return pb.ModelInfo(loaded=False)
    return pb.ModelInfo(
        loaded=True,
        model=pb.ModelRef(name=info["name"], version=info["version"], alias=info["alias"]),
        labels=info["labels"],
        score_threshold=info["score_threshold"],
        metrics=info["metrics"],
        eval_set=info.get("eval_set", ""),
    )


class TokenInterceptor(grpc.ServerInterceptor):
    """Rejects calls without `authorization: Bearer <token>` with UNAUTHENTICATED."""

    def __init__(self, token):
        self._expected = f"Bearer {token}".encode()

        def deny(request, context):
            context.abort(grpc.StatusCode.UNAUTHENTICATED, "missing or invalid token")

        self._deny = deny

    def intercept_service(self, continuation, handler_call_details):
        sent = dict(handler_call_details.invocation_metadata or ()).get("authorization", "")
        if hmac.compare_digest(sent.encode(), self._expected):
            return continuation(handler_call_details)
        if continuation(handler_call_details) is None:
            return None
        return grpc.unary_unary_rpc_method_handler(self._deny)  # every RPC is unary


class DetectorServicer(pb_grpc.DetectorServicer):
    def __init__(self, service, concurrency=CONCURRENCY):
        self.service = service
        self._slots = threading.BoundedSemaphore(concurrency)

    def _acquire(self, context):
        # Wait for a slot without blocking forever: if the client already left, skip inference
        while not self._slots.acquire(timeout=_SLOT_POLL_S):
            if not context.is_active():
                raise Cancelled()
        if not context.is_active():
            self._slots.release()
            raise Cancelled()

    def _infer(self, request, context):
        """One image → response dict. Raises DetectionError, ModelNotLoaded or Cancelled."""
        received = time.perf_counter()
        threshold = request.score_threshold if request.HasField("score_threshold") else None
        self._acquire(context)
        try:
            response = self.service.detect(request.image, score_threshold=threshold, received_at=received)
        finally:
            self._slots.release()
        if not context.is_active():
            raise Cancelled()
        self.service.persist(response)
        return response

    def Detect(self, request, context):
        try:
            return to_proto(self._infer(request, context), request.frame_id)
        except ModelNotLoaded:
            context.abort(grpc.StatusCode.UNAVAILABLE, "no champion model loaded")
        except DetectionError as e:
            # gRPC cannot tell 415 from 422: the contract's status travels in the trailer
            context.set_trailing_metadata([("http-status", str(e.status))])
            context.abort(_STATUS[e.status], e.message)
        except Cancelled:
            context.abort(grpc.StatusCode.CANCELLED, "the client cancelled the call")
        except Exception as e:  # noqa: BLE001 — never leak internal details to the client
            print(f"internal error in Detect: {type(e).__name__}: {e}", flush=True)
            context.abort(grpc.StatusCode.INTERNAL, "internal error in the inference service")

    def GetModel(self, request, context):
        return model_info(self.service)

    def Reload(self, request, context):
        try:
            self.service.load_champion()
        except Exception as e:  # noqa: BLE001 — MLflow/MinIO down, no champion or tampered weights
            print(f"Reload failed: {type(e).__name__}: {e}", flush=True)
            context.abort(grpc.StatusCode.UNAVAILABLE, "could not load the champion")
        return model_info(self.service)


def build_server(service, token=None, max_workers=16, concurrency=CONCURRENCY):
    interceptors = [TokenInterceptor(token)] if token else []
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        interceptors=interceptors,
        options=[
            ("grpc.max_receive_message_length", MAX_MESSAGE_BYTES),
            ("grpc.max_connection_age_ms", CONNECTION_AGE_MS),
            ("grpc.max_connection_age_grace_ms", 15_000),
        ],
    )
    pb_grpc.add_DetectorServicer_to_server(DetectorServicer(service, concurrency), server)
    return server


def main():
    threads = os.getenv("TORCH_NUM_THREADS")
    if threads:
        import torch

        torch.set_num_threads(int(threads))

    token = os.getenv("INFERENCE_TOKEN")
    if not token:
        print("WARNING: INFERENCE_TOKEN is empty, the service accepts unauthenticated calls", flush=True)
    service = DetectionService()
    # Serve right away (Detect → UNAVAILABLE) and look for the champion in the background:
    # the DAG may not have run yet
    server = build_server(service, token)
    server.add_insecure_port(f"[::]:{PORT}")
    server.start()
    print(f"gRPC listening on :{PORT} (concurrency {CONCURRENCY}"
          f"{', torch threads ' + threads if threads else ''})", flush=True)
    announced = None
    while True:
        try:
            version = service.load_champion()
            if version != announced:
                print(f"champion model v{version} loaded", flush=True)
                announced = version
        except Exception as e:  # noqa: BLE001
            if not service.loaded or "SHA-256" in str(e):
                print(f"no loadable champion ({type(e).__name__}): {e}", flush=True)
        time.sleep(RELOAD_RETRY_SECONDS)


if __name__ == "__main__":
    main()
