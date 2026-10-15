"""DAG monitoreo_drift: predicciones del intervalo vs la referencia del champion.

1. `compactar`: lee las predicciones del intervalo de datos de la corrida
   (`data_interval_start`–`data_interval_end`, así re-ejecutar un run pasado analiza la misma
   ventana) y deja sus features en un solo Parquet en `curated/predictions/` (evita leer miles
   de JSON chicos dos veces y no pasa los datos por XCom). Solo entran las del champion y con el
   umbral calibrado: un cliente que pide otro umbral no puede disparar una alarma.
2. `comparar`: chi² (conteo de detecciones) y KS (continuas) con corrección de Holm contra la
   referencia (`val` revisado) → reporte en `monitoring/drift/`.
3. `alertar`: si hay drift, escribe una alerta `drift` en `monitoring/alerts/` y falla con
   `DriftDetected` (la corrida queda en rojo en Airflow). Un error técnico falla con otra
   excepción y el callback deja una alerta `error`. Qué hacer ante el drift lo decide una
   persona: el reentrenamiento vive en el repo del modelo, fuera de este sistema.

Contrato en docs/contrato.md §5.
"""

import io
import json
import os
from datetime import datetime, timedelta, timezone

from airflow.sdk import Param, dag, get_current_context, task

BUCKET = os.getenv("DATA_REPO_BUCKET_NAME", "data")
MODEL_NAME = "billboard-detector"
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")


class DriftDetected(Exception):
    """Hay drift: no es un error técnico (el callback lo distingue)."""


def _s3():
    import boto3

    return boto3.client("s3", endpoint_url=os.getenv("AWS_ENDPOINT_URL_S3", "http://s3:9000"))


def _write_alert(kind, message, **extra):
    now = datetime.now(timezone.utc)
    alert = {"kind": kind, "message": message, "created_at": now.isoformat(), **extra}
    key = f"monitoring/alerts/{now:%Y-%m-%dT%H%M%S}-{kind}.json"
    _s3().put_object(Bucket=BUCKET, Key=key, Body=json.dumps(alert, indent=2).encode())
    print(f"ALERTA [{kind}] {message} → s3://{BUCKET}/{key}")
    return key


def _drift_alert_exists(run_id):
    s3 = _s3()
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix="monitoring/alerts/"):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith("-drift.json"):
                alert = json.loads(s3.get_object(Bucket=BUCKET, Key=obj["Key"])["Body"].read())
                if alert.get("run_id") == run_id:
                    return True
    return False


def _alert_on_failure(context):
    """Alerta `error` en monitoring/alerts/ si la tarea falló por un problema técnico.

    El drift no pasa por acá como error: `alertar` ya dejó su alerta `drift` con el run_id.
    (En Airflow 3 el callback no siempre recibe la excepción, por eso se mira la alerta.)
    """
    ti = context["ti"]
    exc = context.get("exception")
    try:
        if type(exc).__name__ == "DriftDetected" or _drift_alert_exists(ti.run_id):
            return
        message = f"{type(exc).__name__}: {exc}" if exc else f"falló la tarea {ti.task_id} (ver su log)"
        _write_alert("error", message, dag_id=ti.dag_id, task_id=ti.task_id, run_id=ti.run_id)
    except Exception as e:  # noqa: BLE001 — la alerta no puede tapar el error original
        print(f"no se pudo guardar la alerta: {e}")


def _window(params):
    """[inicio, fin) de la corrida: el intervalo de datos, o `window_hours` si es manual."""
    ctx = get_current_context()
    start, end = ctx.get("data_interval_start"), ctx.get("data_interval_end")
    if start is None or end is None or start == end:
        end = ctx.get("run_after") or datetime.now(timezone.utc)
        start = end - timedelta(hours=int(params["window_hours"]))
    return start, end


@dag(
    dag_id="monitoreo_drift",
    schedule="@hourly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["vallavision", "monitoreo"],
    default_args={"on_failure_callback": _alert_on_failure},
    params={
        "window_hours": Param(1, type="integer", minimum=1,
                              description="Horas a analizar en corridas manuales (sin intervalo de datos)"),
        "alpha": Param(0.01, type="number", minimum=0, maximum=1,
                       description="Error por familia (Holm) de los 4 tests"),
        "thin": Param(1, type="integer", minimum=1,
                      description="Quedarse con 1 de cada k frames (atenúa la autocorrelación)"),
        "fail_on_drift": Param(True, type="boolean", description="Fallar la ejecución si hay drift"),
    },
    doc_md=__doc__,
)
def monitoreo_drift():

    @task
    def compactar():
        """predictions/ del intervalo → curated/predictions/…/<inicio>.parquet (features por frame)."""
        import mlflow
        import pandas as pd
        from mlflow.tracking import MlflowClient
        from vallavision import drift

        params = get_current_context()["params"]
        start, end = _window(params)
        mlflow.set_tracking_uri(MLFLOW_URI)
        mv = MlflowClient().get_model_version_by_alias(MODEL_NAME, "champion")
        # La referencia se guarda por versión de los pesos en MinIO (v<N>), que no es la
        # versión de MLflow: se llega a ella por el tag weights_uri
        weights_version = mv.tags["weights_uri"].rstrip("/").rsplit("/v", 1)[1]
        ref_key = f"curated/reference/{MODEL_NAME}/v{weights_version}.json"
        s3 = _s3()
        reference = json.loads(s3.get_object(Bucket=BUCKET, Key=ref_key)["Body"].read())

        rows, skipped = [], 0
        day = start.date()
        while day <= end.date():
            prefix = f"predictions/date={day:%Y-%m-%d}/"
            for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=prefix):
                for obj in page.get("Contents", []):
                    if not start <= obj["LastModified"] < end:
                        continue
                    rec = json.loads(s3.get_object(Bucket=BUCKET, Key=obj["Key"])["Body"].read())
                    if rec["model"]["version"] != mv.version or not drift.usable_for_drift(
                            rec, reference.get("score_threshold")):
                        skipped += 1
                        continue
                    rows.append({"request_id": rec["request_id"], "created_at": rec.get("created_at"),
                                 **drift.frame_features(rec["detections"], rec["image_stats"])})
            day += timedelta(days=1)

        key = (f"curated/predictions/model=v{mv.version}/date={start:%Y-%m-%d}/"
               f"{start:%H%M%S}-{end:%H%M%S}.parquet")
        buf = io.BytesIO()
        pd.DataFrame(rows, columns=["request_id", "created_at", *drift.FEATURES]) \
            .sort_values("created_at").to_parquet(buf, index=False)
        s3.put_object(Bucket=BUCKET, Key=key, Body=buf.getvalue())
        print(f"{len(rows)} predicciones de v{mv.version} en [{start:%F %T}, {end:%F %T}) "
              f"(descartadas {skipped}: otra versión o umbral no calibrado) → {key}")
        return {"version": mv.version, "parquet": key, "reference": ref_key, "n_frames": len(rows),
                "start": start.isoformat(), "end": end.isoformat()}

    @task
    def comparar(datos):
        import pandas as pd
        from vallavision import drift

        params = get_current_context()["params"]
        s3 = _s3()
        reference = json.loads(s3.get_object(Bucket=BUCKET, Key=datos["reference"])["Body"].read())
        df = pd.read_parquet(io.BytesIO(s3.get_object(Bucket=BUCKET, Key=datos["parquet"])["Body"].read()))
        frames = [{f: (None if pd.isna(r[f]) else r[f]) for f in drift.FEATURES} for _, r in df.iterrows()]
        frames = drift.thin(frames, int(params["thin"]))
        result = drift.compare(reference, frames, alpha=float(params["alpha"]))
        report = {"model_version": datos["version"], "reference": datos["reference"],
                  "window": [datos["start"], datos["end"]], "features": datos["parquet"],
                  "thin": int(params["thin"]), "created_at": datetime.now(timezone.utc).isoformat(), **result}
        out = f"monitoring/drift/{datetime.now(timezone.utc):%Y-%m-%dT%H%M%S}.json"
        s3.put_object(Bucket=BUCKET, Key=out, Body=json.dumps(report, indent=2).encode())
        print(json.dumps(report, indent=2))
        print(f"reporte: s3://{BUCKET}/{out}")
        if result["drift"] is None:
            print(f"muy pocas predicciones ({result['n_frames']} < {drift.MIN_FRAMES}): sin veredicto")
        return {"report": out, "drift": result["drift"],
                "flagged": [f for f, m in result["metrics"].items() if m.get("drift")]}

    @task
    def alertar(datos, resultado):
        """Con drift: alerta `drift` en monitoring/alerts/ y la corrida falla (visible en Airflow)."""
        if not resultado["drift"]:
            return None
        reason = f"drift en {', '.join(resultado['flagged'])}"
        key = _write_alert("drift", reason, report=resultado["report"], model_version=datos["version"],
                           window=[datos["start"], datos["end"]], run_id=get_current_context()["run_id"])
        if get_current_context()["params"]["fail_on_drift"]:
            raise DriftDetected(f"{reason}; reporte: {resultado['report']}")
        return key

    datos = compactar()
    alertar(datos, comparar(datos))


monitoreo_drift()
