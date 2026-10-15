"""Consumidor de streaming (Sesión 4): Redpanda → inferencia → métricas por ventana de event time.

Lee el topic `frames` (clave = cámara, valor = JPEG, encabezados `event_time_ms` y
`frame_id`), infiere cada frame con `Detect` por gRPC y agrupa por cámara en ventanas
tumbling de `WINDOW_MS` según el event time, con un watermark que tolera
`ALLOWED_LATENESS_MS` de desorden. Si una cámara deja de mandar frames por `IDLE_FLUSH_S`,
sus ventanas abiertas se cierran igual. Al cerrar cada ventana:

- métricas operativas: frames, throughput, latencia p50/p95, detecciones por frame, tardíos;
- reporte en `curated/stream_windows/` y una alerta en `monitoring/alerts/` si el p95 supera
  `LATENCY_SLO_MS`.

El drift NO se calcula acá: lo calcula `monitoreo_drift` sobre todas las predicciones (REST y
streaming), en un solo lugar y con una sola política de alertas.

Semántica at-least-once: el offset se confirma después de inferir. Si el consumidor se cae
entre la inferencia y el commit, ese frame se procesa dos veces (mismo `frame_id`).
Varias instancias en el mismo grupo se reparten las particiones. La partición sale del hash de la
cámara: los frames de una cámara van siempre a la misma (orden por cámara), pero varias cámaras
pueden compartir una partición.
"""

import json
import os
import time
from datetime import datetime, timezone

import grpc

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
IDLE_FLUSH_S = float(os.getenv("IDLE_FLUSH_S", "10"))
LATENCY_SLO_MS = float(os.getenv("LATENCY_SLO_MS", "2000"))
BUCKET = os.getenv("DATA_REPO_BUCKET_NAME", "data")


def percentile(values, q):
    """Percentil q (0–100) por rango más cercano; None si no hay valores."""
    if not values:
        return None
    s = sorted(values)
    return s[min(len(s) - 1, max(0, -(-q * len(s) // 100) - 1))]


def window_report(camera, start, end, events, late, slo_ms=LATENCY_SLO_MS):
    """Métricas de una ventana cerrada (función pura: la usan el consumidor y los tests)."""
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
    report["alerts"] = [f"latencia p95 {p95:.0f} ms > SLO {slo_ms:.0f} ms"] if p95 is not None and p95 > slo_ms else []
    return report


class Consumer:
    def __init__(self):
        import boto3

        self.s3 = boto3.client("s3", endpoint_url=os.getenv("AWS_ENDPOINT_URL_S3", "http://s3:9000"))
        self.stub = pb_grpc.DetectorStub(grpc.insecure_channel(INFERENCE_ADDR))
        self.metadata = [("authorization", f"Bearer {INFERENCE_TOKEN}")] if INFERENCE_TOKEN else None
        self.windows = TumblingWindows(WINDOW_MS, ALLOWED_LATENESS_MS)
        self.last_seen = {}  # cámara -> último frame recibido (reloj de pared)

    def process(self, camera, event_time_ms, frame_id, image):
        r = self.stub.Detect(pb.DetectRequest(image=image, frame_id=frame_id), timeout=60, metadata=self.metadata)
        self.last_seen[camera] = time.monotonic()
        return self.windows.add(camera, event_time_ms, {"latency_ms": r.latency_ms, "detections": len(r.detections)})

    def flush_idle(self):
        """Cierra las ventanas de las cámaras que no mandan frames hace IDLE_FLUSH_S."""
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

    for _ in range(30):
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
          f"tolerancia {ALLOWED_LATENESS_MS} ms, cierre por inactividad {IDLE_FLUSH_S:.0f} s", flush=True)
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
        c.emit(c.flush_idle())


if __name__ == "__main__":
    main()
