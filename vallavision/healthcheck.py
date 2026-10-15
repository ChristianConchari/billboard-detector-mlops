"""Healthcheck del servicio de inferencia: sano solo si responde y tiene el champion cargado.

    python -m vallavision.healthcheck   # exit 0 = sano
"""

import os
import sys

import grpc

from vallavision.grpc_gen import detector_pb2 as pb
from vallavision.grpc_gen import detector_pb2_grpc as pb_grpc


def main():
    port = os.getenv("GRPC_PORT", "50051")
    token = os.getenv("INFERENCE_TOKEN")
    metadata = [("authorization", f"Bearer {token}")] if token else None
    with grpc.insecure_channel(f"localhost:{port}") as channel:
        try:
            info = pb_grpc.DetectorStub(channel).GetModel(pb.GetModelRequest(), timeout=3, metadata=metadata)
        except grpc.RpcError as e:
            sys.exit(f"inference no responde: {e.code().name}")
    if not info.loaded:
        sys.exit("inference sin champion cargado")


if __name__ == "__main__":
    main()
