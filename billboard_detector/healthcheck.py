"""Inference service healthcheck: healthy only if it answers and has the champion loaded.

    python -m billboard_detector.healthcheck   # exit code 0 = healthy
"""

import os
import sys

import grpc

from billboard_detector.grpc_gen import detector_pb2 as pb
from billboard_detector.grpc_gen import detector_pb2_grpc as pb_grpc


def main():
    port = os.getenv("GRPC_PORT", "50051")
    token = os.getenv("INFERENCE_TOKEN")
    metadata = [("authorization", f"Bearer {token}")] if token else None
    with grpc.insecure_channel(f"localhost:{port}") as channel:
        try:
            info = pb_grpc.DetectorStub(channel).GetModel(pb.GetModelRequest(), timeout=3, metadata=metadata)
        except grpc.RpcError as e:
            sys.exit(f"inference is not responding: {e.code().name}")
    if not info.loaded:
        sys.exit("inference has no champion loaded")


if __name__ == "__main__":
    main()
