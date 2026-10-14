"""Compara latencia y tamaño de mensaje: REST (multipart / base64) vs gRPC (bytes).

Requiere el stack arriba y un champion registrado. Genera el código gRPC antes:
    scripts/gen_proto.sh
    uv run --with requests --with grpcio --with grpcio-tools scripts/bench_rest_vs_grpc.py IMG [-n 50]
"""

import argparse
import base64
import statistics
import sys
import time
from pathlib import Path

import grpc
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vallavision.grpc_gen import detector_pb2 as pb  # noqa: E402
from vallavision.grpc_gen import detector_pb2_grpc as pb_grpc  # noqa: E402


def stats(ms):
    ms = sorted(ms)
    return statistics.median(ms), ms[int(0.95 * (len(ms) - 1))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("-n", type=int, default=50)
    ap.add_argument("--rest", default="http://localhost:8800")
    ap.add_argument("--grpc", default="localhost:50051")
    args = ap.parse_args()
    data = Path(args.image).read_bytes()

    stub = pb_grpc.DetectorStub(grpc.insecure_channel(args.grpc))
    req = pb.DetectRequest(image=data)
    sess = requests.Session()
    rest = lambda: sess.post(f"{args.rest}/v1/detect", files={"image": ("i.jpg", data, "image/jpeg")})  # noqa: E731
    grpc_call = lambda: stub.Detect(req)  # noqa: E731

    rest(), grpc_call()  # calentamiento
    results = {}
    for name, fn in (("REST", rest), ("gRPC", grpc_call)):
        times = []
        for _ in range(args.n):
            t0 = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t0) * 1000)
        results[name] = stats(times)

    sizes = {
        "REST multipart": len(data),
        "REST JSON + base64": len(('{"image":"%s"}' % base64.b64encode(data).decode()).encode()),
        "gRPC bytes": req.ByteSize(),
    }
    print(f"{'':10}{'p50 ms':>10}{'p95 ms':>10}")
    for name, (p50, p95) in results.items():
        print(f"{name:10}{p50:10.1f}{p95:10.1f}")
    print("\nTamaño del request:")
    for name, size in sizes.items():
        print(f"  {name:20}{size:>10,} B")


if __name__ == "__main__":
    main()
