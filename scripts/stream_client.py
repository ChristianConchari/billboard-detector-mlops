"""Streaming simulado: reproduce una carpeta de imágenes como una cámara y la infiere por gRPC.

Inferencia evento a evento (un `Detect` por frame) y métricas por ventana de N frames:
throughput, latencia p50/p95, detecciones por frame y, si hay referencia, drift (KS).
Cada frame queda además en `predictions/` (lo guarda el servidor), así que el DAG
`monitoreo_drift` ve el mismo tráfico.

    uv run --with grpcio --with pillow --with numpy --with scipy --with boto3 \
        scripts/stream_client.py data/billboards-2025-09-14-sur-lp/images/ --fps 5 \
        --reference s3://data/curated/reference/billboard-detector/v1.json

`--degrade oscurecer|desenfocar|ruido` altera los frames para provocar drift en la demo.
"""

import argparse
import io
import json
import random
import sys
import time
from pathlib import Path

import grpc
from PIL import Image, ImageEnhance, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vallavision import drift  # noqa: E402
from vallavision.grpc_gen import detector_pb2 as pb  # noqa: E402
from vallavision.grpc_gen import detector_pb2_grpc as pb_grpc  # noqa: E402

DEGRADE = {
    "oscurecer": lambda im: ImageEnhance.Brightness(im).enhance(0.35),
    "desenfocar": lambda im: im.filter(ImageFilter.GaussianBlur(6)),
    "ruido": lambda im: Image.blend(im, Image.effect_noise(im.size, 80).convert("RGB"), 0.5),
}


def load_reference(path):
    if path.startswith("s3://"):
        import boto3

        bucket, key = path[5:].split("/", 1)
        body = boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()
        return json.loads(body)
    return json.loads(Path(path).read_text())


def frames(folder, loop):
    paths = sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not paths:
        sys.exit(f"sin imágenes en {folder}")
    while True:
        random.shuffle(paths)
        yield from paths
        if not loop:
            return


def encode(path, degrade):
    im = Image.open(path).convert("RGB")
    if degrade:
        im = DEGRADE[degrade](im)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    return buf.getvalue(), im


def report(window, wall, reference):
    lat = [w["latency_ms"] for w in window]
    out = {
        "frames": len(window),
        "throughput_fps": round(len(window) / wall, 2),
        "latency_ms_p50": drift.percentile(lat, 50),
        "latency_ms_p95": drift.percentile(lat, 95),
        "detections_per_frame": round(sum(w["features"]["n_detections"] for w in window) / len(window), 2),
    }
    if reference:
        res = drift.compare(reference, [w["features"] for w in window])
        out["drift"] = res["drift"]
        out["drift_features"] = [f for f, m in res["metrics"].items() if m.get("drift")]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--fps", type=float, default=5)
    ap.add_argument("--window", type=int, default=30, help="frames por ventana")
    ap.add_argument("--loop", action="store_true", help="repetir la carpeta indefinidamente")
    ap.add_argument("--degrade", choices=sorted(DEGRADE))
    ap.add_argument("--reference", help="ruta local o s3://bucket/key de la referencia")
    ap.add_argument("--grpc", default="localhost:50051")
    args = ap.parse_args()

    reference = load_reference(args.reference) if args.reference else None
    stub = pb_grpc.DetectorStub(grpc.insecure_channel(args.grpc))
    window, t_window = [], time.perf_counter()
    period = 1.0 / args.fps
    try:
        for path in frames(args.folder, args.loop):
            t0 = time.perf_counter()
            data, im = encode(path, args.degrade)
            r = stub.Detect(pb.DetectRequest(image=data))
            dets = [{"score": d.score} for d in r.detections]
            window.append({"latency_ms": r.latency_ms,
                           "features": drift.frame_features(dets, drift.image_stats(im))})
            if len(window) >= args.window:
                print(json.dumps(report(window, time.perf_counter() - t_window, reference)), flush=True)
                window, t_window = [], time.perf_counter()
            time.sleep(max(0.0, period - (time.perf_counter() - t0)))  # ritmo de "cámara"
    except KeyboardInterrupt:
        pass
    except grpc.RpcError as e:
        sys.exit(f"gRPC: {e.code().name}: {e.details()}")


if __name__ == "__main__":
    main()
