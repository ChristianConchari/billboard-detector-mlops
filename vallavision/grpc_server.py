"""Servidor gRPC de VallaVision: mismo servicio y contrato que REST (docs/contrato.md §3)."""

import os
import time
from concurrent import futures

import grpc

from vallavision.grpc_gen import detector_pb2 as pb
from vallavision.grpc_gen import detector_pb2_grpc as pb_grpc
from vallavision.service import DetectionError, DetectionService, ModelNotLoaded

PORT = int(os.getenv("GRPC_PORT", "50051"))
RELOAD_RETRY_SECONDS = 30

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


class DetectorServicer(pb_grpc.DetectorServicer):
    def __init__(self, service):
        self.service = service

    def _detect(self, request, context):
        threshold = request.score_threshold if request.HasField("score_threshold") else None
        try:
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


def main():
    service = DetectionService()
    # Espera al champion (el DAG puede no haber corrido todavía), pero sirve ya: Detect → UNAVAILABLE
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4),
        options=[("grpc.max_receive_message_length", 8 * 1024 * 1024)],
    )
    pb_grpc.add_DetectorServicer_to_server(DetectorServicer(service), server)
    server.add_insecure_port(f"[::]:{PORT}")
    server.start()
    print(f"gRPC escuchando en :{PORT}")
    while True:
        try:
            version = service.load_champion()
            if not getattr(main, "_announced", None) == version:
                print(f"modelo champion v{version} cargado")
                main._announced = version
        except Exception as e:  # noqa: BLE001
            if not service.loaded:
                print(f"sin champion todavía ({type(e).__name__}): {e}")
        time.sleep(RELOAD_RETRY_SECONDS)


if __name__ == "__main__":
    main()
