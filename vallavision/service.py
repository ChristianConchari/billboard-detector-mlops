"""Servicio de detección compartido por REST y gRPC (docs/contrato.md).

Carga el modelo con alias `champion` desde MLflow/MinIO, valida la imagen, arma la
respuesta del contrato §2 y persiste cada predicción en `predictions/` (§4).
"""

import io
import json
import os
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone

from vallavision.drift import image_stats

MODEL_NAME = os.getenv("MODEL_NAME", "billboard-detector")
ALIAS = os.getenv("MODEL_ALIAS", "champion")
BUCKET = os.getenv("DATA_REPO_BUCKET_NAME", "data")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

MAX_BYTES = 5 * 1024 * 1024
MAX_SIDE = 4096
ALLOWED_TYPES = ("image/jpeg", "image/png")


class DetectionError(Exception):
    """Error de entrada con el código HTTP del contrato (413/415/422)."""

    def __init__(self, status, message):
        super().__init__(message)
        self.status = status
        self.message = message


class ModelNotLoaded(Exception):
    """No hay modelo `champion` cargado (503)."""


def _s3():
    import boto3

    return boto3.client("s3", endpoint_url=os.getenv("AWS_ENDPOINT_URL_S3", "http://s3:9000"))


class DetectionService:
    def __init__(self):
        self.detector = None
        self.info = None  # {name, version, alias, labels, metrics, score_threshold}
        self._lock = threading.Lock()
        self._s3 = None

    @property
    def loaded(self):
        return self.detector is not None

    def load_champion(self):
        """Carga (o recarga) el champion. Devuelve la versión; no hace nada si ya es la actual."""
        import mlflow
        from mlflow.tracking import MlflowClient

        from vallavision.detector import BillboardDetector

        mlflow.set_tracking_uri(MLFLOW_URI)
        client = MlflowClient()
        mv = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
        if self.info and self.info["version"] == mv.version:
            return mv.version

        uri = mv.tags["weights_uri"]
        prefix = uri[len(f"s3://{BUCKET}/"):]
        s3 = _s3()
        with tempfile.TemporaryDirectory() as tmp:
            for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=prefix):
                for obj in page.get("Contents", []):
                    rel = obj["Key"][len(prefix):].lstrip("/")
                    if rel and rel != "metadata.json":
                        path = os.path.join(tmp, rel)
                        os.makedirs(os.path.dirname(path), exist_ok=True)
                        s3.download_file(BUCKET, obj["Key"], path)
            detector = BillboardDetector(tmp)  # los pesos quedan en memoria; tmp se borra

        metrics = client.get_run(mv.run_id).data.metrics
        with self._lock:
            self.detector = detector
            self.info = {
                "name": MODEL_NAME,
                "version": mv.version,
                "alias": ALIAS,
                "labels": detector.labels,
                "score_threshold": detector.default_threshold,
                "metrics": {k: v for k, v in metrics.items() if k in
                            ("map", "map_50", "latency_ms_mean", "latency_ms_p95")},
            }
        return mv.version

    def model_ref(self):
        i = self.info
        return {"name": i["name"], "version": i["version"], "alias": i["alias"]}

    def detect(self, image_bytes, content_type=None, score_threshold=None):
        """Valida, infiere y persiste. Devuelve el JSON del contrato §2."""
        if not self.loaded:
            raise ModelNotLoaded()
        if content_type is not None and content_type not in ALLOWED_TYPES:
            raise DetectionError(415, f"tipo no soportado: {content_type}")
        if len(image_bytes) > MAX_BYTES:
            raise DetectionError(413, f"imagen > {MAX_BYTES // (1024 * 1024)} MB")
        if score_threshold is not None and not 0.0 <= score_threshold <= 1.0:
            raise DetectionError(422, "score_threshold debe estar entre 0 y 1")

        from PIL import Image, UnidentifiedImageError

        try:
            image = Image.open(io.BytesIO(image_bytes))
            image.load()
        except (UnidentifiedImageError, OSError) as e:
            raise DetectionError(422, f"imagen ilegible: {e}") from e
        if max(image.size) > MAX_SIDE:
            raise DetectionError(422, f"lado mayor > {MAX_SIDE} px")

        t0 = time.perf_counter()
        size, detections = self.detector.detect(image, score_threshold=score_threshold)
        response = {
            "request_id": uuid.uuid4().hex,
            "model": self.model_ref(),
            "image": size,
            "detections": detections,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            # Solo para monitoreo: REST/gRPC lo descartan, queda en predictions/
            "image_stats": image_stats(image),
        }
        return response

    def persist(self, response):
        """predictions/date=YYYY-MM-DD/<request_id>.json. No debe tumbar la request si falla."""
        try:
            if self._s3 is None:
                self._s3 = _s3()
            day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            self._s3.put_object(
                Bucket=BUCKET,
                Key=f"predictions/date={day}/{response['request_id']}.json",
                Body=json.dumps(response).encode(),
                ContentType="application/json",
            )
        except Exception as e:  # noqa: BLE001 — la persistencia es best-effort
            print(f"no se pudo guardar la predicción {response['request_id']}: {e}")
