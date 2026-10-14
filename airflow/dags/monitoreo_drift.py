"""DAG monitoreo_drift: predictions/ recientes vs la referencia del champion.

Lee las predicciones de las últimas `window_hours` horas (las que guardan REST y gRPC),
compara sus features con la referencia calculada sobre `val` por `registrar_modelo` y
deja un reporte en monitoring/drift/. Si hay drift la tarea falla (alerta visible en
Airflow). Contrato en docs/contrato.md §5.
"""

import json
import os
from datetime import datetime, timedelta, timezone

from airflow.sdk import Param, dag, get_current_context, task

BUCKET = os.getenv("DATA_REPO_BUCKET_NAME", "data")
MODEL_NAME = "billboard-detector"
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")


def _s3():
    import boto3

    return boto3.client("s3", endpoint_url=os.getenv("AWS_ENDPOINT_URL_S3", "http://s3:9000"))


@dag(
    dag_id="monitoreo_drift",
    schedule="@hourly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["vallavision", "monitoreo"],
    params={
        "window_hours": Param(1, type="integer", minimum=1, description="Horas de predicciones a analizar"),
        "alpha": Param(0.01, type="number", minimum=0, maximum=1, description="Nivel del test KS"),
        "fail_on_drift": Param(True, type="boolean", description="Fallar la ejecución si hay drift"),
    },
    doc_md=__doc__,
)
def monitoreo_drift():

    @task
    def leer_predicciones():
        """Features por frame de predictions/ en la ventana, solo de la versión champion."""
        import mlflow
        from mlflow.tracking import MlflowClient
        from vallavision import drift

        params = get_current_context()["params"]
        mlflow.set_tracking_uri(MLFLOW_URI)
        mv = MlflowClient().get_model_version_by_alias(MODEL_NAME, "champion")
        version = mv.version
        # La referencia se guarda por versión de los pesos en MinIO (v<N>), que no es la
        # versión de MLflow: se llega a ella por el tag weights_uri
        weights_version = mv.tags["weights_uri"].rstrip("/").rsplit("/v", 1)[1]

        s3 = _s3()
        since = datetime.now(timezone.utc) - timedelta(hours=int(params["window_hours"]))
        frames = []
        for day in {since.date(), datetime.now(timezone.utc).date()}:
            prefix = f"predictions/date={day:%Y-%m-%d}/"
            for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=prefix):
                for obj in page.get("Contents", []):
                    if obj["LastModified"] < since:
                        continue
                    rec = json.loads(s3.get_object(Bucket=BUCKET, Key=obj["Key"])["Body"].read())
                    if rec["model"]["version"] != version or "image_stats" not in rec:
                        continue
                    frames.append(drift.frame_features(rec["detections"], rec["image_stats"]))
        print(f"{len(frames)} predicciones de la versión {version} desde {since:%F %T}")
        return {"version": version, "weights_version": weights_version, "frames": frames}

    @task
    def comparar(datos):
        from vallavision import drift

        params = get_current_context()["params"]
        s3 = _s3()
        key = f"curated/reference/{MODEL_NAME}/v{datos['weights_version']}.json"
        reference = json.loads(s3.get_object(Bucket=BUCKET, Key=key)["Body"].read())
        result = drift.compare(reference, datos["frames"], alpha=float(params["alpha"]))
        report = {"model_version": datos["version"], "reference": key,
                  "created_at": datetime.now(timezone.utc).isoformat(), **result}
        out = f"monitoring/drift/{datetime.now(timezone.utc):%Y-%m-%dT%H%M%S}.json"
        s3.put_object(Bucket=BUCKET, Key=out, Body=json.dumps(report, indent=2).encode())
        print(json.dumps(report, indent=2))
        print(f"reporte: s3://{BUCKET}/{out}")
        if result["drift"] and params["fail_on_drift"]:
            flagged = [f for f, m in result["metrics"].items() if m.get("drift")]
            raise RuntimeError(f"drift detectado en: {', '.join(flagged)}")
        if result["drift"] is None:
            print(f"muy pocas predicciones ({result['n_frames']} < {drift.MIN_FRAMES}): sin veredicto")

    comparar(leer_predicciones())


monitoreo_drift()
