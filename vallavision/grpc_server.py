"""Servicio de inferencia de VallaVision: único dueño del modelo (docs/contrato.md §3).

Lo consumen directo los clientes gRPC y, por dentro, el gateway REST. Se puede escalar
con réplicas: cada una carga su copia del modelo y atiende `INFERENCE_CONCURRENCY` forward
a la vez (1 por defecto: en CPU un forward ya usa los hilos que tiene).
"""

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

_STATUS = {
    413: grpc.StatusCode.RESOURCE_EXHAUSTED,
    415: grpc.StatusCode.INVALID_ARGUMENT,
    422: grpc.StatusCode.INVALID_ARGUMENT,
}


def to_proto(r):
    return pb.DetectResponse(
        request_id=r["request_id"],
        model=pb.ModelRef(**r["model"]),
        image=pb.ImageSize(**r["image"]),
        detections=[pb.Detection(label=d["label"], score=d["score"], box=d["box"])
                    for d in r["detections"]],
        latency_ms=r["latency_ms"],
    )


def model_info(service):
    if not service.loaded:
        return pb.ModelInfo(loaded=False)
    i = service.info
    return pb.ModelInfo(
        loaded=True,
        model=pb.ModelRef(name=i["name"], version=i["version"], alias=i["alias"]),
        labels=i["labels"],
        score_threshold=i["score_threshold"],
        metrics=i["metrics"],
    )


class DetectorServicer(pb_grpc.DetectorServicer):
    def __init__(self, service, concurrency=CONCURRENCY):
        self.service = service
        self._slots = threading.BoundedSemaphore(concurrency)

    def _detect(self, request, context):
        threshold = request.score_threshold if request.HasField("score_threshold") else None
        try:
            with self._slots:
                response = self.service.detect(request.image, score_threshold=threshold)
        except ModelNotLoaded:
            context.abort(grpc.StatusCode.UNAVAILABLE, "no hay modelo champion cargado")
        except DetectionError as e:
            context.abort(_STATUS[e.status], e.message)
        self.service.persist(response)
        return to_proto(response)

    def Detect(self, request, context):
        return self._detect(request, context)

    def DetectStream(self, request, context):
        for frame in request.frames:
            yield self._detect(frame, context)

    def GetModel(self, request, context):
        return model_info(self.service)

    def Reload(self, request, context):
        try:
            self.service.load_champion()
        except Exception as e:  # noqa: BLE001 — MLflow/MinIO caídos o sin champion
            context.abort(grpc.StatusCode.UNAVAILABLE, f"no se pudo cargar el champion: {e}")
        return model_info(self.service)


def main():
    threads = os.getenv("TORCH_NUM_THREADS")
    if threads:
        import torch

        torch.set_num_threads(int(threads))

    service = DetectionService()
    # Sirve ya (Detect → UNAVAILABLE) y busca el champion en segundo plano: el DAG puede no haber corrido
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=16),
        options=[
            ("grpc.max_receive_message_length", 8 * 1024 * 1024),
            ("grpc.max_connection_age_ms", CONNECTION_AGE_MS),
            ("grpc.max_connection_age_grace_ms", 15_000),
        ],
    )
    pb_grpc.add_DetectorServicer_to_server(DetectorServicer(service), server)
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
            if not service.loaded:
                print(f"sin champion todavía ({type(e).__name__}): {e}", flush=True)
        time.sleep(RELOAD_RETRY_SECONDS)


if __name__ == "__main__":
    main()
