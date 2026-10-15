"""Detection service behind the gRPC server (docs/contrato.md).

Loads the model aliased `champion` from MLflow/MinIO (verifying the weights' SHA-256),
validates the image, builds the response defined in contract §2 and persists every
prediction under `predictions/` (§4). Images are not stored (faces and license plates;
see docs/seguridad.md).
"""

import io
import json
import os
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone

from billboard_detector.drift import image_stats
from billboard_detector.integrity import sha256_dir
from billboard_detector.validation import (  # noqa: F401 — re-exported for importers of service
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
# Metrics of the evaluation run published through /v1/model (the gate metrics)
PUBLIC_METRICS = ("gate_map", "gate_map_50", "gate_map_ci_low", "gate_map_ci_high")


def _s3():
    import boto3

    return boto3.client("s3", endpoint_url=os.getenv("AWS_ENDPOINT_URL_S3", "http://s3:9000"))


class IntegrityError(Exception):
    """The downloaded weights do not match the SHA-256 registered in MLflow."""


class DetectionService:
    def __init__(self):
        # (detector, info) is replaced as a whole: a request never sees one version's
        # detector together with another version's info
        self._model = None
        self._load_lock = threading.Lock()
        self._s3 = None

    @property
    def loaded(self):
        return self._model is not None

    @property
    def info(self):
        return self._model[1] if self._model else None

    def set_model(self, detector, info):
        self._model = (detector, info)

    def load_champion(self):
        """Loads (or reloads) the champion. Returns its version; no-op if already loaded."""
        import mlflow
        from mlflow.tracking import MlflowClient

        from billboard_detector.detector import BillboardDetector

        # One load at a time: the 30 s loop and a Reload never load two copies in parallel
        with self._load_lock:
            mlflow.set_tracking_uri(MLFLOW_URI)
            client = MlflowClient()
            mv = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
            if self.info and self.info["version"] == mv.version:
                return mv.version

            uri = mv.tags["weights_uri"]
            expected = mv.tags.get("weights_dir_sha256") or mv.tags.get("weights_sha256")
            if not expected:
                raise IntegrityError(f"version {mv.version} has no weights_dir_sha256: refusing to load it")
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
                    raise IntegrityError(f"SHA-256 of {uri} is {actual[:12]}…, MLflow expects {expected[:12]}…")
                detector = BillboardDetector(tmp)  # the weights stay in memory; tmp is deleted

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
            })
            return mv.version

    def model_ref(self):
        i = self.info
        return {"name": i["name"], "version": i["version"], "alias": i["alias"]}

    def detect(self, image_bytes, content_type=None, score_threshold=None, received_at=None):
        """Validates, runs inference and builds the contract §2 JSON plus monitoring fields.

        `received_at` (perf_counter) is when the request arrived: `latency_ms` includes the
        wait for an inference slot and the decoding, not just the forward pass.
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
            # open() only reads the header: format and size are validated before decoding
            image = Image.open(io.BytesIO(image_bytes))
        except Image.DecompressionBombError as e:
            raise DetectionError(422, f"longest side larger than {MAX_SIDE} px") from e
        except (UnidentifiedImageError, OSError) as e:
            raise DetectionError(422, "unreadable image") from e
        check_format(image.format)
        check_dimensions(*image.size)
        try:
            image.load()
            # Boxes refer to the image as displayed, not as stored
            image = ImageOps.exif_transpose(image)
        except (OSError, SyntaxError, ValueError) as e:
            raise DetectionError(422, "unreadable image") from e

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
            # Monitoring only: REST/gRPC drop these, they are kept in predictions/
            "inference_ms": round(inference_ms, 1),
            "score_threshold": round(float(threshold), 4),
            "threshold_is_default": score_threshold is None,
            "image_stats": image_stats(image),
        }

    def persist(self, response):
        """predictions/date=…/<request_id>.json. A failure must never break the request."""
        try:
            if self._s3 is None:
                self._s3 = _s3()
            now = datetime.now(timezone.utc)
            day = now.strftime("%Y-%m-%d")
            record = {**response, "created_at": now.isoformat()}
            self._s3.put_object(
                Bucket=BUCKET,
                Key=f"predictions/date={day}/{response['request_id']}.json",
                Body=json.dumps(record).encode(),
                ContentType="application/json",
            )
        except Exception as e:  # noqa: BLE001 — persistence is best-effort
            print(f"could not store prediction {response['request_id']}: {e}")
