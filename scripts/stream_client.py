"""Streaming directo por gRPC: reproduce frames como una cámara con `DetectLive` (bidireccional).

El cliente manda frames a medida que los "captura" y recibe una respuesta por frame en el
mismo stream; reporta por ventana de N frames throughput, latencia p50/p95, detecciones por
frame y, si hay referencia, drift. Cada frame queda en `predictions/` (lo guarda el
servidor), así que `monitoreo_drift` ve el mismo tráfico. El camino con broker (Redpanda,
ventanas por event time) es `scripts/camara_productor.py` + el servicio `stream-consumer`.

Por defecto reproduce, en orden de frame, las imágenes del `test` revisado: videos que no
están en la referencia de drift (control negativo honesto). `--degrade oscurecer` es el
control positivo.

    uv run --with grpcio --with protobuf --with pillow --with numpy --with scipy --with boto3 \\
        scripts/stream_client.py --fps 2 --reference s3://data/curated/reference/billboard-detector/v1.json

El servicio de inferencia exige el token de `.env` (INFERENCE_TOKEN): se lee de ahí.
"""

import argparse
import io
import json
import os
import random
import sys
import time
from pathlib import Path

import grpc
from PIL import Image, ImageEnhance, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from vallavision import drift  # noqa: E402
from vallavision.grpc_gen import detector_pb2 as pb  # noqa: E402
from vallavision.grpc_gen import detector_pb2_grpc as pb_grpc  # noqa: E402

DATASET = ROOT / "data" / "billboards-2025-09-14-sur-lp" / "images"
DEGRADE = {
    "oscurecer": lambda im: ImageEnhance.Brightness(im).enhance(0.35),
    "desenfocar": lambda im: im.filter(ImageFilter.GaussianBlur(6)),
    "ruido": lambda im: Image.blend(im, Image.effect_noise(im.size, 80).convert("RGB"), 0.5),
}


def env(name):
    if os.getenv(name):
        return os.getenv(name)
    path = ROOT / ".env"
    if path.exists():
        for line in path.read_text().splitlines():
            if line.startswith(f"{name}="):
                return line.split("=", 1)[1].strip()
    return None


def load_reference(path):
    if path.startswith("s3://"):
        import boto3

        bucket, key = path[5:].split("/", 1)
        s3 = boto3.client("s3", endpoint_url=os.getenv("AWS_ENDPOINT_URL_S3", "http://localhost:9000"),
                          aws_access_key_id=env("MINIO_ACCESS_KEY"), aws_secret_access_key=env("MINIO_SECRET_ACCESS_KEY"))
        return json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
    return json.loads(Path(path).read_text())


def frame_paths(annotations, images_dir, shuffle):
    """Imágenes del COCO en orden de video y de frame (o mezcladas con --shuffle)."""
    coco = json.loads(Path(annotations).read_text())

    def order(name):
        video, _, frame = name.rpartition("_")
        return video, int(Path(frame).stem)

    paths = [Path(images_dir) / n for n in sorted((i["file_name"] for i in coco["images"]), key=order)]
    if shuffle:
        random.Random(0).shuffle(paths)
    return paths


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
    ap.add_argument("--annotations", default=str(ROOT / "annotations" / "reviewed" / "test.json"))
    ap.add_argument("--images", default=str(DATASET))
    ap.add_argument("--fps", type=float, default=2)
    ap.add_argument("--window", type=int, default=30, help="frames por ventana")
    ap.add_argument("--shuffle", action="store_true", help="mezclar frames (rompe la autocorrelación)")
    ap.add_argument("--degrade", choices=sorted(DEGRADE))
    ap.add_argument("--reference", help="ruta local o s3://bucket/key de la referencia")
    ap.add_argument("--grpc", default="localhost:50051")
    args = ap.parse_args()

    reference = load_reference(args.reference) if args.reference else None
    token = env("INFERENCE_TOKEN")
    metadata = [("authorization", f"Bearer {token}")] if token else None
    stub = pb_grpc.DetectorStub(grpc.insecure_channel(args.grpc, options=[
        ("grpc.max_send_message_length", 8 * 1024 * 1024)]))
    paths = frame_paths(args.annotations, args.images, args.shuffle)
    sent = {}
    period = 1.0 / args.fps

    def produce():
        for i, path in enumerate(paths):
            t0 = time.perf_counter()
            data, im = encode(path, args.degrade)
            sent[str(i)] = im
            yield pb.DetectRequest(image=data, frame_id=str(i))
            time.sleep(max(0.0, period - (time.perf_counter() - t0)))  # ritmo de "cámara"

    window, t_window = [], time.perf_counter()
    try:
        for r in stub.DetectLive(produce(), metadata=metadata):
            im = sent.pop(r.frame_id)
            if r.error.status:
                print(f"frame {r.frame_id}: {r.error.status} {r.error.message}", flush=True)
                continue
            dets = [{"score": d.score} for d in r.detections]
            window.append({"latency_ms": r.latency_ms,
                           "features": drift.frame_features(dets, drift.image_stats(im))})
            if len(window) >= args.window:
                print(json.dumps(report(window, time.perf_counter() - t_window, reference)), flush=True)
                window, t_window = [], time.perf_counter()
    except KeyboardInterrupt:
        pass
    except grpc.RpcError as e:
        sys.exit(f"gRPC: {e.code().name}: {e.details()}")
    if len(window) >= drift.MIN_FRAMES:
        print(json.dumps(report(window, time.perf_counter() - t_window, reference)), flush=True)


if __name__ == "__main__":
    main()
