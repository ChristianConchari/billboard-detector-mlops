"""Servicio de inferencia del sistema de detección de vallas: único dueño del modelo (docs/contrato.md §3).

Lo consumen el gateway REST y el consumidor de streaming (servicio interno). Se puede
escalar con réplicas: cada una carga su copia del modelo y atiende `INFERENCE_CONCURRENCY`
forward a la vez (1 por defecto: en CPU un forward ya usa los hilos que tiene).

Si `INFERENCE_TOKEN` está definido, cada llamada tiene que traer el metadata
`authorization: Bearer <token>` (interceptor; ver docs/seguridad.md).
"""

import hmac
import os
import threading
import time
from concurrent import futures

import grpc

from vallavision.grpc_gen import detector_pb2 as pb
from vallavision.grpc_gen import detector_pb2_grpc as pb_grpc
from vallavision.service import DetectionError, DetectionService, ModelNotLoaded

PORT = int(os.getenv("GRPC_PORT", "50051"))
CONCURRENCY = int(os.getenv("INFERENCE_CONCURRENCY", "1"))
RELOAD_RETRY_SECONDS = 30
# Cerrar las conexiones viejas obliga a los clientes (el gateway) a re-resolver el DNS y
# descubrir réplicas nuevas: sin esto un canal creado con 1 réplica nunca usa la 2.ª
CONNECTION_AGE_MS = int(os.getenv("GRPC_MAX_CONNECTION_AGE_S", "30")) * 1000
MAX_MESSAGE_BYTES = 8 * 1024 * 1024
_SLOT_POLL_S = 0.1

_STATUS = {
    413: grpc.StatusCode.RESOURCE_EXHAUSTED,
    415: grpc.StatusCode.INVALID_ARGUMENT,
    422: grpc.StatusCode.INVALID_ARGUMENT,
}


class Cancelled(Exception):
    """El cliente se fue (deadline o cancelación): no se infiere ni se persiste."""


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
    """Rechaza con UNAUTHENTICATED las llamadas sin `authorization: Bearer <token>`."""

    def __init__(self, token):
        self._expected = f"Bearer {token}".encode()

        def deny(request, context):
            context.abort(grpc.StatusCode.UNAUTHENTICATED, "token inválido o ausente")

        self._deny = deny

    def intercept_service(self, continuation, handler_call_details):
        sent = dict(handler_call_details.invocation_metadata or ()).get("authorization", "")
        if hmac.compare_digest(sent.encode(), self._expected):
            return continuation(handler_call_details)
        if continuation(handler_call_details) is None:
            return None
        return grpc.unary_unary_rpc_method_handler(self._deny)  # todos los RPCs son unarios


class DetectorServicer(pb_grpc.DetectorServicer):
    def __init__(self, service, concurrency=CONCURRENCY):
        self.service = service
        self._slots = threading.BoundedSemaphore(concurrency)

    def _acquire(self, context):
        # Esperar turno sin bloquear para siempre: si el cliente ya se fue, no hay que inferir
        while not self._slots.acquire(timeout=_SLOT_POLL_S):
            if not context.is_active():
                raise Cancelled()
        if not context.is_active():
            self._slots.release()
            raise Cancelled()

    def _infer(self, request, context):
        """Una imagen → respuesta (dict). Lanza DetectionError, ModelNotLoaded o Cancelled."""
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
            context.abort(grpc.StatusCode.UNAVAILABLE, "no hay modelo champion cargado")
        except DetectionError as e:
            # gRPC no distingue 415 de 422: el código del contrato viaja en el trailer
            context.set_trailing_metadata([("http-status", str(e.status))])
            context.abort(_STATUS[e.status], e.message)
        except Cancelled:
            context.abort(grpc.StatusCode.CANCELLED, "el cliente canceló la llamada")
        except Exception as e:  # noqa: BLE001 — no filtrar detalles internos al cliente
            print(f"error interno en Detect: {type(e).__name__}: {e}", flush=True)
            context.abort(grpc.StatusCode.INTERNAL, "error interno del servicio de inferencia")

    def GetModel(self, request, context):
        return model_info(self.service)

    def Reload(self, request, context):
        try:
            self.service.load_champion()
        except Exception as e:  # noqa: BLE001 — MLflow/MinIO caídos, sin champion o pesos alterados
            print(f"Reload falló: {type(e).__name__}: {e}", flush=True)
            context.abort(grpc.StatusCode.UNAVAILABLE, "no se pudo cargar el champion")
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
        print("ADVERTENCIA: INFERENCE_TOKEN vacío, el servicio acepta llamadas sin autenticar", flush=True)
    service = DetectionService()
    # Sirve ya (Detect → UNAVAILABLE) y busca el champion en segundo plano: el DAG puede no haber corrido
    server = build_server(service, token)
    server.add_insecure_port(f"[::]:{PORT}")
    server.start()
    print(f"gRPC escuchando en :{PORT} (concurrencia {CONCURRENCY}"
          f"{', hilos torch ' + threads if threads else ''})", flush=True)
    announced = None
    while True:
        try:
            version = service.load_champion()
            if version != announced:
                print(f"modelo champion v{version} cargado", flush=True)
                announced = version
        except Exception as e:  # noqa: BLE001
            if not service.loaded or "SHA-256" in str(e):
                print(f"sin champion cargable ({type(e).__name__}): {e}", flush=True)
        time.sleep(RELOAD_RETRY_SECONDS)


if __name__ == "__main__":
    main()
