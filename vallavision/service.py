"""Servicio de detección compartido por REST y gRPC (docs/contrato.md).

Carga el modelo con alias `champion` desde MLflow/MinIO (verificando el SHA-256 de los
pesos), valida la imagen, arma la respuesta del contrato §2 y persiste cada predicción
en `predictions/` y una muestra de las imágenes en `raw/stream/` (§4).
"""

import io
import json
import os
import random
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone

from vallavision.drift import image_stats
from vallavision.integrity import sha256_dir
from vallavision.validation import (  # noqa: F401 — reexportados para quien importe desde service
    MAX_BYTES,
    MAX_SIDE,
    DetectionError,
    ModelNotLoaded,
    check_dimensions,
    check_format,
    check_size,
    check_threshold,
    check_type,
)

MODEL_NAME = os.getenv("MODEL_NAME", "billboard-detector")
ALIAS = os.getenv("MODEL_ALIAS", "champion")
BUCKET = os.getenv("DATA_REPO_BUCKET_NAME", "data")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
# Fracción de imágenes que se guardan en raw/stream/ para reetiquetar si hay drift
PERSIST_IMAGES_RATIO = float(os.getenv("PERSIST_IMAGES_RATIO", "0.1"))
# Métricas del run de evaluación que se publican en /v1/model (las de gate)
PUBLIC_METRICS = ("gate_map", "gate_map_50", "gate_map_ci_low", "gate_map_ci_high")
_EXT = {"JPEG": "jpg", "PNG": "png"}


def _s3():
    import boto3

    return boto3.client("s3", endpoint_url=os.getenv("AWS_ENDPOINT_URL_S3", "http://s3:9000"))


class IntegrityError(Exception):
    """Los pesos descargados no coinciden con el SHA-256 registrado en MLflow."""


class DetectionService:
    def __init__(self, persist_images_ratio=PERSIST_IMAGES_RATIO):
        # (detector, info) se reemplaza entero: una request nunca ve el detector de una
        # versión con la info de otra
        self._model = None
        self._load_lock = threading.Lock()
        self._s3 = None
        self.persist_images_ratio = persist_images_ratio

    @property
    def loaded(self):
        return self._model is not None

    @property
    def info(self):
        return self._model[1] if self._model else None

    def set_model(self, detector, info):
        self._model = (detector, info)

    def load_champion(self):
        """Carga (o recarga) el champion. Devuelve la versión; no hace nada si ya es la actual."""
        import mlflow
        from mlflow.tracking import MlflowClient

        from vallavision.detector import BillboardDetector

        # Un solo load a la vez: el loop de 30 s y un Reload no cargan dos copias en paralelo
        with self._load_lock:
            mlflow.set_tracking_uri(MLFLOW_URI)
            client = MlflowClient()
            mv = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
            if self.info and self.info["version"] == mv.version:
                return mv.version

            uri = mv.tags["weights_uri"]
            expected = mv.tags.get("weights_dir_sha256") or mv.tags.get("weights_sha256")
            if not expected:
                raise IntegrityError(f"la versión {mv.version} no tiene weights_dir_sha256: no se carga")
            prefix = uri[len(f"s3://{BUCKET}/"):]
            s3 = _s3()
            with tempfile.TemporaryDirectory() as tmp:
                for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=prefix):
                    for obj in page.get("Contents", []):
                        rel = obj["Key"][len(prefix):].lstrip("/")
                        if rel and rel not in ("metadata.json", ".lock"):
                            path = os.path.join(tmp, rel)
                            os.makedirs(os.path.dirname(path), exist_ok=True)
                            s3.download_file(BUCKET, obj["Key"], path)
                actual = sha256_dir(tmp)
                if actual != expected:
                    raise IntegrityError(f"SHA-256 de {uri} = {actual[:12]}…, MLflow dice {expected[:12]}…")
                detector = BillboardDetector(tmp)  # los pesos quedan en memoria; tmp se borra

            eval_run = mv.tags.get("eval_run_id") or mv.run_id
            metrics = client.get_run(eval_run).data.metrics
            self.set_model(detector, {
                "name": MODEL_NAME,
                "version": mv.version,
                "alias": ALIAS,
                "labels": detector.labels,
                "score_threshold": detector.default_threshold,
                "metrics": {k: v for k, v in metrics.items() if k in PUBLIC_METRICS},
                "eval_set": mv.tags.get("eval_set", ""),
                "weights_uri": uri,
            })
            return mv.version

    def model_ref(self):
        i = self.info
        return {"name": i["name"], "version": i["version"], "alias": i["alias"]}

    def detect(self, image_bytes, content_type=None, score_threshold=None, received_at=None):
        """Valida, infiere y arma el JSON del contrato §2 más los campos de monitoreo.

        `received_at` (perf_counter) es cuándo llegó la request: `latency_ms` incluye la
        espera por un turno de inferencia y el decodificado, no solo el forward.
        """
        started = received_at if received_at is not None else time.perf_counter()
        model = self._model
        if model is None:
            raise ModelNotLoaded()
        detector, info = model
        check_type(content_type)
        check_size(len(image_bytes))
        check_threshold(score_threshold)

        from PIL import Image, ImageOps, UnidentifiedImageError

        try:
            # open() solo lee el encabezado: formato y tamaño se validan antes de decodificar
            image = Image.open(io.BytesIO(image_bytes))
        except Image.DecompressionBombError as e:
            raise DetectionError(422, f"lado mayor > {MAX_SIDE} px") from e
        except (UnidentifiedImageError, OSError) as e:
            raise DetectionError(422, "imagen ilegible") from e
        check_format(image.format)
        check_dimensions(*image.size)
        image_format = image.format
        try:
            image.load()
            # Las cajas se devuelven sobre la imagen como se ve, no como está guardada
            image = ImageOps.exif_transpose(image)
        except (OSError, SyntaxError, ValueError) as e:
            raise DetectionError(422, "imagen ilegible") from e

        threshold = score_threshold if score_threshold is not None else detector.default_threshold
        t0 = time.perf_counter()
        size, detections = detector.detect(image, score_threshold=threshold)
        inference_ms = (time.perf_counter() - t0) * 1000
        return {
            "request_id": uuid.uuid4().hex,
            "model": {"name": info["name"], "version": info["version"], "alias": info["alias"]},
            "image": size,
            "detections": detections,
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            # Solo para monitoreo: REST/gRPC los descartan, quedan en predictions/
            "inference_ms": round(inference_ms, 1),
            "score_threshold": round(float(threshold), 4),
            "threshold_is_default": score_threshold is None,
            "image_stats": image_stats(image),
            "image_format": image_format,
        }

    def persist(self, response, image_bytes=None):
        """predictions/date=…/<request_id>.json y, con probabilidad `persist_images_ratio`,
        la imagen en raw/stream/date=…/. No debe tumbar la request si falla."""
        try:
            if self._s3 is None:
                self._s3 = _s3()
            now = datetime.now(timezone.utc)
            day = now.strftime("%Y-%m-%d")
            record = {**response, "created_at": now.isoformat()}
            if image_bytes is not None and random.random() < self.persist_images_ratio:
                ext = _EXT.get(response.get("image_format"), "bin")
                key = f"raw/stream/date={day}/{response['request_id']}.{ext}"
                self._s3.put_object(Bucket=BUCKET, Key=key, Body=image_bytes)
                record["image_key"] = key
            self._s3.put_object(
                Bucket=BUCKET,
                Key=f"predictions/date={day}/{response['request_id']}.json",
                Body=json.dumps(record).encode(),
                ContentType="application/json",
            )
        except Exception as e:  # noqa: BLE001 — la persistencia es best-effort
            print(f"no se pudo guardar la predicción {response['request_id']}: {e}")
