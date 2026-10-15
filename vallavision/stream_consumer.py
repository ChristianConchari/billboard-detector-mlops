"""Consumidor de streaming (Sesión 4): Redpanda → inferencia → ventanas por event time.

Lee el topic `frames` (clave = cámara, valor = JPEG, encabezados `event_time_ms` y
`frame_id`), infiere cada frame con `Detect` por gRPC y agrupa por cámara en ventanas
tumbling de `WINDOW_MS` según el event time, con un watermark que tolera
`ALLOWED_LATENESS_MS` de desorden. Al cerrar cada ventana:

- métricas: frames, throughput, latencia p50/p95, detecciones por frame, frames tardíos;
- drift contra la referencia del champion (mismos tests que `monitoreo_drift`);
- reporte en `curated/stream_windows/` y, si hay drift o el p95 supera `LATENCY_SLO_MS`,
  una alerta en `monitoring/alerts/`.

Semántica at-least-once: el offset se confirma después de inferir. Si el consumidor se cae
entre la inferencia y el commit, ese frame se procesa dos veces (mismo `frame_id`).
Varias instancias en el mismo grupo se reparten las particiones (una cámara = una partición).
"""

import io
import json
import os
import time
from datetime import datetime, timezone

import grpc
from PIL import Image

from vallavision import drift
from vallavision.grpc_gen import detector_pb2 as pb
from vallavision.grpc_gen import detector_pb2_grpc as pb_grpc
from vallavision.streaming import TumblingWindows

BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "redpanda:9092")
TOPIC = os.getenv("KAFKA_TOPIC", "frames")
GROUP = os.getenv("KAFKA_GROUP", "detector")
PARTITIONS = int(os.getenv("KAFKA_PARTITIONS", "3"))
INFERENCE_ADDR = os.getenv("INFERENCE_ADDR", "inference:50051")
INFERENCE_TOKEN = os.getenv("INFERENCE_TOKEN", "")
WINDOW_MS = int(os.getenv("WINDOW_MS", "15000"))
ALLOWED_LATENESS_MS = int(os.getenv("ALLOWED_LATENESS_MS", "2000"))
LATENCY_SLO_MS = float(os.getenv("LATENCY_SLO_MS", "2000"))
BUCKET = os.getenv("DATA_REPO_BUCKET_NAME", "data")
MODEL_NAME = "billboard-detector"


def window_report(camera, start, end, events, late, reference):
    """Métricas de una ventana cerrada (función pura: la usan el consumidor y los tests)."""
    lat = [e["latency_ms"] for e in events]
    feats = [e["features"] for e in events]
    report = {
        "camera": camera,
        "window": [start, end],
        "frames": len(events),
        "late_frames": late,
        "throughput_fps": round(len(events) / ((end - start) / 1000), 3),
        "latency_ms_p50": drift.percentile(lat, 50),
        "latency_ms_p95": drift.percentile(lat, 95),
        "detections_per_frame": round(sum(f["n_detections"] for f in feats) / len(feats), 3) if feats else 0,
    }
    if reference is not None:
        res = drift.compare(reference, feats)
        report["drift"] = res["drift"]
        report["drift_features"] = [f for f, m in res["metrics"].items() if m.get("drift")]
    alerts = []
    if report.get("drift"):
        alerts.append(f"drift en {', '.join(report['drift_features'])}")
    if report["latency_ms_p95"] is not None and report["latency_ms_p95"] > LATENCY_SLO_MS:
        alerts.append(f"latencia p95 {report['latency_ms_p95']:.0f} ms > SLO {LATENCY_SLO_MS:.0f} ms")
    report["alerts"] = alerts
    return report


class Consumer:
    def __init__(self):
        import boto3

        self.s3 = boto3.client("s3", endpoint_url=os.getenv("AWS_ENDPOINT_URL_S3", "http://s3:9000"))
        self.stub = pb_grpc.DetectorStub(grpc.insecure_channel(INFERENCE_ADDR))
        self.metadata = [("authorization", f"Bearer {INFERENCE_TOKEN}")] if INFERENCE_TOKEN else None
        self.windows = TumblingWindows(WINDOW_MS, ALLOWED_LATENESS_MS)
        self.reference = None
        self.reference_for = None

    def _reference(self):
        """Referencia de drift del champion actual (se recarga si cambia el modelo)."""
        info = self.stub.GetModel(pb.GetModelRequest(), timeout=5, metadata=self.metadata)
        if not info.loaded or info.weights_uri == self.reference_for:
            return self.reference
        weights_version = info.weights_uri.rstrip("/").rsplit("/v", 1)[1]
        key = f"curated/reference/{MODEL_NAME}/v{weights_version}.json"
        try:
            self.reference = json.loads(self.s3.get_object(Bucket=BUCKET, Key=key)["Body"].read())
            self.reference_for = info.weights_uri
            print(f"referencia de drift: {key}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"sin referencia de drift ({e}); las ventanas no tendrán veredicto", flush=True)
        return self.reference

    def process(self, camera, event_time_ms, frame_id, image):
        r = self.stub.Detect(pb.DetectRequest(image=image, frame_id=frame_id), timeout=60, metadata=self.metadata)
        stats = drift.image_stats(Image.open(io.BytesIO(image)))
        event = {"latency_ms": r.latency_ms,
                 "features": drift.frame_features([{"score": d.score} for d in r.detections], stats)}
        return self.windows.add(camera, event_time_ms, event)

    def emit(self, closed):
        reference = self._reference() if closed else None
        for camera, start, end, events in closed:
            report = window_report(camera, start, end, events, self.windows.late[camera], reference)
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
                print(f"ALERTA [{camera}] {alert['message']}", flush=True)


def ensure_topic():
    from kafka.admin import KafkaAdminClient, NewTopic
    from kafka.errors import TopicAlreadyExistsError

    admin = KafkaAdminClient(bootstrap_servers=BOOTSTRAP, client_id="vallavision-admin")
    try:
        admin.create_topics([NewTopic(TOPIC, num_partitions=PARTITIONS, replication_factor=1)])
        print(f"topic {TOPIC} creado con {PARTITIONS} particiones", flush=True)
    except TopicAlreadyExistsError:
        pass
    finally:
        admin.close()


def main():
    from kafka import KafkaConsumer
    from kafka.errors import KafkaError

    for attempt in range(30):
        try:
            ensure_topic()
            break
        except KafkaError as e:  # el broker todavía no acepta conexiones
            print(f"esperando a {BOOTSTRAP} ({type(e).__name__})", flush=True)
            time.sleep(2)
    consumer = KafkaConsumer(TOPIC, bootstrap_servers=BOOTSTRAP, group_id=GROUP,
                             enable_auto_commit=False, auto_offset_reset="earliest",
                             max_poll_records=10)
    c = Consumer()
    print(f"consumiendo {TOPIC} (grupo {GROUP}); ventanas de {WINDOW_MS} ms, "
          f"tolerancia {ALLOWED_LATENESS_MS} ms", flush=True)
    while True:
        batch = consumer.poll(timeout_ms=1000)
        for records in batch.values():
            for msg in records:
                headers = dict(msg.headers or [])
                event_time = int(headers.get("event_time_ms", b"0").decode() or msg.timestamp)
                frame_id = headers.get("frame_id", b"").decode()
                camera = msg.key.decode() if msg.key else "sin-camara"
                try:
                    closed = c.process(camera, event_time, frame_id, msg.value)
                except grpc.RpcError as e:
                    # Frame inválido: se registra y se sigue; inference caído: se reintenta al reiniciar
                    if e.code() == grpc.StatusCode.INVALID_ARGUMENT:
                        print(f"frame {frame_id} descartado: {e.details()}", flush=True)
                        closed = []
                    else:
                        raise
                c.emit(closed)
        if batch:
            consumer.commit()


if __name__ == "__main__":
    main()
