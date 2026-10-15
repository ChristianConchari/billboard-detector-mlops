"""Productor de streaming: simula cámaras publicando frames en Redpanda (API de Kafka).

Cada video del dataset es una cámara: sus frames se publican en orden, con la cámara como
clave (todos los frames de una cámara van a la misma partición y conservan el orden) y el
instante de captura en el encabezado `event_time_ms`. El consumidor (servicio
`stream-consumer`) los infiere y arma ventanas por event time.

Por defecto usa las imágenes del `test` revisado, que vienen de videos que NO están en la
referencia de drift (`val`): es el control negativo honesto. `--degrade oscurecer` es el
control positivo; `--late 0.1` manda el 10 % de los frames con atraso para ver el watermark.

    uv run --with kafka-python --with pillow scripts/camara_productor.py --fps 2
"""

import argparse
import io
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "billboards-2025-09-14-sur-lp" / "images"

DEGRADE = {
    "oscurecer": lambda im: ImageEnhance.Brightness(im).enhance(0.35),
    "desenfocar": lambda im: im.filter(ImageFilter.GaussianBlur(6)),
    "ruido": lambda im: Image.blend(im, Image.effect_noise(im.size, 80).convert("RGB"), 0.5),
}


def cameras(annotations, images_dir, max_cameras):
    """{cámara: [rutas en orden de frame]} a partir de un COCO (nombre = <video>_<frame>.jpg)."""
    coco = json.loads(Path(annotations).read_text())
    by_cam = defaultdict(list)
    for img in coco["images"]:
        video, _, frame = img["file_name"].rpartition("_")
        by_cam[video].append((int(Path(frame).stem), Path(images_dir) / img["file_name"]))
    cams = sorted(by_cam, key=lambda c: -len(by_cam[c]))[:max_cameras]
    return {c[:8]: [p for _, p in sorted(by_cam[c])] for c in cams}


def encode(path, degrade):
    im = Image.open(path).convert("RGB")
    if degrade:
        im = DEGRADE[degrade](im)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def schedule(cams, fps, loops, late, seed=0):
    """Orden de envío: [(cámara, event_time_ms, frame_id, ruta)].

    El event time avanza a `fps` por cámara (instante de captura simulado); el orden de envío
    intercala las cámaras y, con `late` > 0, demora algunos frames varias posiciones.
    """
    rnd = random.Random(seed)
    t0 = int(time.time() * 1000)
    period_ms = 1000 / fps
    events = []
    for loop in range(loops):
        n = max(len(v) for v in cams.values())
        for i in range(n):
            for cam, paths in cams.items():
                if i < len(paths):
                    k = loop * len(paths) + i
                    events.append((cam, t0 + int(k * period_ms), f"{cam}-{k}", paths[i]))
    order = []
    for pos, ev in enumerate(events):
        delay = rnd.randint(6, 18) if rnd.random() < late else 0
        order.append((pos + delay, pos, ev))
    return [ev for _, _, ev in sorted(order)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", default=str(ROOT / "annotations" / "reviewed" / "test.json"),
                    help="COCO con las imágenes a reproducir (por defecto: videos fuera de la referencia)")
    ap.add_argument("--images", default=str(DATASET))
    ap.add_argument("--cameras", type=int, default=3)
    ap.add_argument("--fps", type=float, default=2, help="frames por segundo por cámara (event time)")
    ap.add_argument("--rate", type=float, default=2, help="mensajes por segundo publicados (tiempo real)")
    ap.add_argument("--loops", type=int, default=2, help="veces que se repite cada cámara")
    ap.add_argument("--degrade", choices=sorted(DEGRADE))
    ap.add_argument("--late", type=float, default=0.0, help="fracción de frames enviados con atraso")
    ap.add_argument("--bootstrap", default="localhost:19092")
    ap.add_argument("--topic", default="frames")
    args = ap.parse_args()

    from kafka import KafkaProducer

    cams = cameras(args.annotations, args.images, args.cameras)
    if not cams:
        sys.exit("sin imágenes")
    producer = KafkaProducer(bootstrap_servers=args.bootstrap, max_request_size=8 * 1024 * 1024)
    events = schedule(cams, args.fps, args.loops, args.late)
    print(f"{len(events)} frames de {len(cams)} cámaras ({', '.join(cams)}) a {args.rate} msg/s", flush=True)
    for cam, event_time, frame_id, path in events:
        tick = time.perf_counter()
        producer.send(args.topic, key=cam.encode(), value=encode(path, args.degrade),
                      headers=[("event_time_ms", str(event_time).encode()), ("frame_id", frame_id.encode())])
        time.sleep(max(0.0, 1 / args.rate - (time.perf_counter() - tick)))
    producer.flush()
    print(f"{len(events)} frames publicados en {args.topic}")


if __name__ == "__main__":
    main()
