"""Streaming consumer (Session 4): Redpanda → inference → event-time window metrics.

Reads the `frames` topic (key = camera, value = JPEG, headers `event_time_ms` and
`frame_id`), runs each frame through `Detect` over gRPC and groups them per camera into
`WINDOW_MS` tumbling windows by event time, with a watermark that tolerates
`ALLOWED_LATENESS_MS` of disorder. If a camera sends nothing for `IDLE_FLUSH_S`, its open
windows are closed anyway. When a window closes:

- operational metrics: frames, throughput, p50/p95 latency, detections per frame, late frames;
- a report under `curated/stream_windows/`, plus an alert under `monitoring/alerts/` if the
  p95 latency exceeds `LATENCY_SLO_MS`.

Drift is NOT computed here: `drift_monitoring` computes it over every prediction (REST and
streaming), in a single place and with a single alerting policy.

At-least-once semantics: the offset is committed after inference. If the consumer dies
between inference and commit, that frame is processed twice (same `frame_id`). Several
instances in the same group split the partitions. The partition comes from the camera's
hash: a camera's frames always land in the same one (per-camera order), but several cameras
may share a partition.
"""

import json
import os
import time
from datetime import datetime, timezone

import grpc

from billboard_detector.grpc_gen import detector_pb2 as pb
from billboard_detector.grpc_gen import detector_pb2_grpc as pb_grpc
from billboard_detector.streaming import TumblingWindows

BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "redpanda:9092")
TOPIC = os.getenv("KAFKA_TOPIC", "frames")
GROUP = os.getenv("KAFKA_GROUP", "detector")
PARTITIONS = int(os.getenv("KAFKA_PARTITIONS", "3"))
INFERENCE_ADDR = os.getenv("INFERENCE_ADDR", "inference:50051")
INFERENCE_TOKEN = os.getenv("INFERENCE_TOKEN", "")
WINDOW_MS = int(os.getenv("WINDOW_MS", "15000"))
ALLOWED_LATENESS_MS = int(os.getenv("ALLOWED_LATENESS_MS", "2000"))
IDLE_FLUSH_S = float(os.getenv("IDLE_FLUSH_S", "10"))
LATENCY_SLO_MS = float(os.getenv("LATENCY_SLO_MS", "2000"))
BUCKET = os.getenv("DATA_REPO_BUCKET_NAME", "data")


def percentile(values, q):
    """q-th percentile (0–100) by nearest rank; None when there are no values."""
    if not values:
        return None
    s = sorted(values)
    return s[min(len(s) - 1, max(0, -(-q * len(s) // 100) - 1))]


def window_report(camera, start, end, events, late, slo_ms=LATENCY_SLO_MS):
    """Metrics of a closed window (pure function: used by the consumer and the tests)."""
    lat = [e["latency_ms"] for e in events]
    report = {
        "camera": camera,
        "window": [start, end],
        "frames": len(events),
        "late_frames": late,
        "throughput_fps": round(len(events) / ((end - start) / 1000), 3),
        "latency_ms_p50": percentile(lat, 50),
        "latency_ms_p95": percentile(lat, 95),
        "detections_per_frame": round(sum(e["detections"] for e in events) / len(events), 3) if events else 0,
    }
    p95 = report["latency_ms_p95"]
    report["alerts"] = [f"p95 latency {p95:.0f} ms > SLO {slo_ms:.0f} ms"] if p95 is not None and p95 > slo_ms else []
    return report


class Consumer:
    def __init__(self):
        import boto3

        self.s3 = boto3.client("s3", endpoint_url=os.getenv("AWS_ENDPOINT_URL_S3", "http://s3:9000"))
        self.stub = pb_grpc.DetectorStub(grpc.insecure_channel(INFERENCE_ADDR))
        self.metadata = [("authorization", f"Bearer {INFERENCE_TOKEN}")] if INFERENCE_TOKEN else None
        self.windows = TumblingWindows(WINDOW_MS, ALLOWED_LATENESS_MS)
        self.last_seen = {}  # camera -> last frame received (wall clock)

    def process(self, camera, event_time_ms, frame_id, image):
        r = self.stub.Detect(pb.DetectRequest(image=image, frame_id=frame_id), timeout=60, metadata=self.metadata)
        self.last_seen[camera] = time.monotonic()
        return self.windows.add(camera, event_time_ms, {"latency_ms": r.latency_ms, "detections": len(r.detections)})

    def flush_idle(self):
        """Closes the windows of cameras that have sent nothing for IDLE_FLUSH_S."""
        now = time.monotonic()
        closed = []
        for camera, seen in list(self.last_seen.items()):
            if now - seen >= IDLE_FLUSH_S:
                closed += self.windows.flush_key(camera)
                del self.last_seen[camera]
        return closed

    def emit(self, closed):
        for camera, start, end, events in closed:
            report = window_report(camera, start, end, events, self.windows.late[camera])
            self.windows.late[camera] = 0
            when = datetime.fromtimestamp(start / 1000, timezone.utc)
            key = f"curated/stream_windows/date={when:%Y-%m-%d}/{camera}-{start}.json"
            self.s3.put_object(Bucket=BUCKET, Key=key, Body=json.dumps(report).encode())
            print(json.dumps(report), flush=True)
            if report["alerts"]:
                now = datetime.now(timezone.utc)
                alert = {"kind": "stream", "camera": camera, "window": [start, end],
                         "message": "; ".join(report["alerts"]), "report": key, "created_at": now.isoformat()}
                self.s3.put_object(Bucket=BUCKET, Key=f"monitoring/alerts/{now:%Y-%m-%dT%H%M%S}-stream-{camera}.json",
                                   Body=json.dumps(alert).encode())
                print(f"ALERT [{camera}] {alert['message']}", flush=True)


def ensure_topic():
    from kafka.admin import KafkaAdminClient, NewTopic
    from kafka.errors import TopicAlreadyExistsError

    admin = KafkaAdminClient(bootstrap_servers=BOOTSTRAP, client_id="billboard-detector-admin")
    try:
        admin.create_topics([NewTopic(TOPIC, num_partitions=PARTITIONS, replication_factor=1)])
        print(f"topic {TOPIC} created with {PARTITIONS} partitions", flush=True)
    except TopicAlreadyExistsError:
        pass
    finally:
        admin.close()


def main():
    from kafka import KafkaConsumer
    from kafka.errors import KafkaError

    for _ in range(30):
        try:
            ensure_topic()
            break
        except KafkaError as e:  # the broker does not accept connections yet
            print(f"waiting for {BOOTSTRAP} ({type(e).__name__})", flush=True)
            time.sleep(2)
    consumer = KafkaConsumer(TOPIC, bootstrap_servers=BOOTSTRAP, group_id=GROUP,
                             enable_auto_commit=False, auto_offset_reset="earliest",
                             max_poll_records=10)
    c = Consumer()
    print(f"consuming {TOPIC} (group {GROUP}); {WINDOW_MS} ms windows, "
          f"{ALLOWED_LATENESS_MS} ms allowed lateness, idle flush after {IDLE_FLUSH_S:.0f} s", flush=True)
    while True:
        batch = consumer.poll(timeout_ms=1000)
        for records in batch.values():
            for msg in records:
                headers = dict(msg.headers or [])
                event_time = int(headers.get("event_time_ms", b"0").decode() or msg.timestamp)
                frame_id = headers.get("frame_id", b"").decode()
                camera = msg.key.decode() if msg.key else "no-camera"
                try:
                    closed = c.process(camera, event_time, frame_id, msg.value)
                except grpc.RpcError as e:
                    # Invalid frame: log it and move on; inference down: retried after a restart
                    if e.code() == grpc.StatusCode.INVALID_ARGUMENT:
                        print(f"frame {frame_id} dropped: {e.details()}", flush=True)
                        closed = []
                    else:
                        raise
                c.emit(closed)
        if batch:
            consumer.commit()
        c.emit(c.flush_idle())


if __name__ == "__main__":
    main()
