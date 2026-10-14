"""Compara latencia, throughput y tamaño de mensaje: REST (multipart / base64) vs gRPC (bytes).

Requiere el stack arriba y un champion registrado. Genera el código gRPC antes:
    scripts/gen_proto.sh
    uv run --with requests --with grpcio --with grpcio-tools scripts/bench_rest_vs_grpc.py IMG -n 20 -c 1 2 4

`-c` acepta varias concurrencias: con cada una se lanzan `-n` requests desde N hilos y se mide
el throughput (req/s) además de p50/p95. REST apunta al gateway y gRPC directo a inference.
"""

import argparse
import base64
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import grpc
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vallavision.grpc_gen import detector_pb2 as pb  # noqa: E402
from vallavision.grpc_gen import detector_pb2_grpc as pb_grpc  # noqa: E402


def percentile(ms, q):
    ms = sorted(ms)
    return ms[min(len(ms) - 1, int(q * (len(ms) - 1) + 0.5))]


def run(fn, n, concurrency):
    """n llamadas desde `concurrency` hilos → (throughput req/s, p50 ms, p95 ms)."""
    def timed(_):
        t0 = time.perf_counter()
        fn()
        return (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    with ThreadPoolExecutor(concurrency) as pool:
        times = list(pool.map(timed, range(n)))
    wall = time.perf_counter() - t0
    return n / wall, statistics.median(times), percentile(times, 0.95)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("-n", type=int, default=20, help="requests por protocolo y concurrencia")
    ap.add_argument("-c", "--concurrency", type=int, nargs="+", default=[1])
    ap.add_argument("--rest", default="http://localhost:8800")
    ap.add_argument("--grpc", default="localhost:50051")
    args = ap.parse_args()
    data = Path(args.image).read_bytes()

    stub = pb_grpc.DetectorStub(grpc.insecure_channel(args.grpc))
    req = pb.DetectRequest(image=data)
    local = threading.local()

    def rest():
        if not hasattr(local, "s"):
            local.s = requests.Session()
        r = local.s.post(f"{args.rest}/v1/detect", files={"image": ("i.jpg", data, "image/jpeg")})
        r.raise_for_status()

    def grpc_call():
        stub.Detect(req)

    rest(), grpc_call()  # calentamiento
    print(f"{'protocolo':10}{'c':>3}{'req/s':>9}{'p50 ms':>10}{'p95 ms':>10}")
    for c in args.concurrency:
        for name, fn in (("REST", rest), ("gRPC", grpc_call)):
            rps, p50, p95 = run(fn, args.n, c)
            print(f"{name:10}{c:>3}{rps:9.2f}{p50:10.0f}{p95:10.0f}", flush=True)

    sizes = {
        "REST multipart": len(data),
        "REST JSON + base64": len(('{"image":"%s"}' % base64.b64encode(data).decode()).encode()),
        "gRPC bytes": req.ByteSize(),
    }
    print("\nTamaño del request:")
    for name, size in sizes.items():
        print(f"  {name:20}{size:>10,} B")


if __name__ == "__main__":
    main()
