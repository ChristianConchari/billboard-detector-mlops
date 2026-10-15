"""Streaming producer: simulates survey vehicles publishing frames to Redpanda (Kafka API).

Every video of the dataset is one camera: its frames are published in order, with the
camera as the key (all of a camera's frames go to the same partition and keep their order)
and the capture time in the `event_time_ms` header. The consumer (`stream-consumer`
service) runs inference on them and builds event-time windows.

By default it uses the reviewed `test` images, which come from videos that are NOT in the
drift reference (`val`): an honest negative control. `--degrade darken` is the positive
control; `--late 0.1` sends 10 % of the frames late to exercise the watermark.

    python scripts/camera_producer.py --rate 2
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
    "darken": lambda im: ImageEnhance.Brightness(im).enhance(0.35),
    "blur": lambda im: im.filter(ImageFilter.GaussianBlur(6)),
    "noise": lambda im: Image.blend(im, Image.effect_noise(im.size, 80).convert("RGB"), 0.5),
}


def cameras(annotations, images_dir, max_cameras):
    """{camera: [paths in frame order]} from a COCO file (file name = <video>_<frame>.jpg)."""
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
    """Publishing order: [(camera, event_time_ms, frame_id, path)].

    Event time advances at `fps` per camera (simulated capture time); the publishing order
    interleaves the cameras and, with `late` > 0, delays some frames by a few positions.
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
                    help="COCO file with the images to replay (default: videos outside the reference)")
    ap.add_argument("--images", default=str(DATASET))
    ap.add_argument("--cameras", type=int, default=3)
    ap.add_argument("--fps", type=float, default=2, help="frames per second per camera (event time)")
    ap.add_argument("--rate", type=float, default=2, help="messages published per second (wall clock)")
    ap.add_argument("--loops", type=int, default=2, help="times each camera is replayed")
    ap.add_argument("--degrade", choices=sorted(DEGRADE))
    ap.add_argument("--late", type=float, default=0.0, help="fraction of frames sent late")
    ap.add_argument("--bootstrap", default="localhost:19092")
    ap.add_argument("--topic", default="frames")
    args = ap.parse_args()

    from kafka import KafkaProducer

    cams = cameras(args.annotations, args.images, args.cameras)
    if not cams:
        sys.exit("no images")
    producer = KafkaProducer(bootstrap_servers=args.bootstrap, max_request_size=8 * 1024 * 1024)
    events = schedule(cams, args.fps, args.loops, args.late)
    print(f"{len(events)} frames from {len(cams)} cameras ({', '.join(cams)}) at {args.rate} msg/s", flush=True)
    for cam, event_time, frame_id, path in events:
        tick = time.perf_counter()
        producer.send(args.topic, key=cam.encode(), value=encode(path, args.degrade),
                      headers=[("event_time_ms", str(event_time).encode()), ("frame_id", frame_id.encode())])
        time.sleep(max(0.0, 1 / args.rate - (time.perf_counter() - tick)))
    producer.flush()
    print(f"{len(events)} frames published to {args.topic}")


if __name__ == "__main__":
    main()
